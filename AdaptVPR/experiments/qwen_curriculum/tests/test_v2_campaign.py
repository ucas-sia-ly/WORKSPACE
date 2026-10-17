"""CPU integration tests for authenticated historical import and v2 resumption.

Historical authentication and metric conversion have independent focused tests.
Here their interfaces are stubbed; migration, immutable journals, saved-image
verification, request reservations and continuation auditing run for real.
"""
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.qwen_curriculum import common, run, fast_campaign, v2_campaign as v2, finetune_all_cities

RECOVER_ORPHAN = v2.recover_orphan


class Verifier:
    def evaluate(self, source, generated, condition):
        return {"status": "passed", "passed": True, "eligible_for_training": True,
                "weather_ok": True, "s_geo": 0.91, "weather_shift": 1.5}


class V2CampaignTests(unittest.TestCase):
    def test_training_uses_current_quality_loader_after_migration(self):
        with patch.object(v2, "load_collection", side_effect=RuntimeError("v2 loader selected")) as current, \
                patch.object(finetune_all_cities.all_city_campaign, "load_collection") as old:
            with self.assertRaisesRegex(RuntimeError, "v2 loader selected"):
                finetune_all_cities.prepare(self.root / "training", self.directory, 10)
        current.assert_called_once_with(self.directory)
        old.assert_not_called()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "legacy"
        self.source.mkdir()
        self.directory = self.root / "v2"
        self.service = self.root / "service"
        self.service.mkdir()
        self.jobs = []
        for number in range(5):
            image = self.root / f"source_{number}.png"
            Image.new("RGB", (20, 16), (number * 25, 80, 45)).save(image)
            body = {"sample_id": "qwen_" + common.fingerprint(number)[:24],
                "source_path": str(image), "source_sha256": common.file_sha256(image),
                "source_dimensions": [20, 16], "condition": "night", "city": "City",
                "place_id": number, "seed": 123 + number,
                "prompt": f"source anchored night prompt {number}", "negative_prompt": ""}
            self.jobs.append({**body, "record_sha256": common.fingerprint(body)})
        self.health = {"status": "ok", "model_loaded": True, "generator_ready": True,
            "error": None, "model_id": "Qwen/Qwen-Image-Edit-2511", "source_modified": False,
            "canvas_policy": "source_aspect_v1", "sampling": {"infer_steps": 4, "guidance_scale": 1.0},
            "canvas": {"target_shape_order": "height,width", "target_pixels": 720,
                "multiple": 1, "min_side": 1, "max_side": 128, "rounding": "nearest_multiple",
                "unsupported_aspect_ratio": "reject"}}
        master = self.source / "additional_plan"
        master.mkdir()
        common.write_jsonl(master / "plan.jsonl", self.jobs)
        self.plan = fast_campaign.frozen(master / "plan_config.json", {
            "output_dir": str(master), "num_images": 5, "seed": 42,
            "domain_quotas": {"night": 5}, "plan_sha256": common.file_sha256(master / "plan.jsonl")})
        self.old_execution = fast_campaign.frozen(self.source / "execution_config.json", {
            **run.execution_definition(SimpleNamespace(run_dir=self.source, device="cuda", cpu_threads=4)),
            "stage": "generate", "plan_fingerprint": self.plan["fingerprint"], "max_calls": 5,
            "qwen_url": "http://127.0.0.1:8001/generate", "request_timeout": 600.0,
            "service": run.service_identity(self.health),
            "call_policy": "one POST per job; durable reservation; no automatic retries",
            "source_encoding": "lossless RGB PNG decoded from original"})
        self.generated = [self.image_row(job) for job in self.jobs[:3]]
        self.results = [run.sealed({**self.generated[0], "status": "passed", "passed": True,
            "eligible_for_training": True}), run.sealed({**self.generated[1], "status": "rejected",
            "passed": False, "eligible_for_training": False, "rejection_reasons": ["geometry_too_few_matches"]})]
        self.attempt = run.sealed_attempt({"sample_id": self.jobs[3]["sample_id"],
            "record_sha256": self.jobs[3]["record_sha256"],
            "execution_fingerprint": self.old_execution["fingerprint"], "call_number": 1})
        self.error = run.sealed({**self.jobs[3], "execution_fingerprint": self.old_execution["fingerprint"],
            "status": "generation_error", "passed": False, "eligible_for_training": False,
            "error": "interrupted original request"})
        common.write_jsonl(self.source / "generated.jsonl", self.generated)
        common.write_jsonl(self.source / "results.jsonl", self.results + [self.error])
        common.write_jsonl(self.source / "attempts.jsonl", [self.attempt])
        self.legacy_files = {str(p): p.read_bytes() for p in self.source.rglob("*") if p.is_file()}
        old_campaign = fast_campaign.frozen(self.source / "all_city_campaign_config.json", {
            "parent_run_dir": str(self.source), "combined_plan": str(master / "plan.jsonl"),
            "total_images": 5, "additional_images": 5, "reference_execution": self.old_execution,
            "stages": [], "runtime_profile": "resident_bf16_48g", "domain_quotas": {"night": 5},
            "city_quotas": {"City": 5}})
        self.history = SimpleNamespace(campaign=old_campaign, jobs=self.jobs,
            generated=self.generated, results=self.results + [self.error],
            attempts={self.jobs[3]["sample_id"]: [{"run_dir": str(self.source),
                "execution": self.old_execution, "attempt": self.attempt}]},
            files_sha256={str(p): common.file_sha256(p) for p in self.source.rglob("*") if p.is_file()},
            code_proofs=[])
        self.addCleanup(patch.stopall)
        patch.object(v2, "ensure_stopped").start()
        # Keep tests independent from concurrent implementation edits. The
        # actual migration's own imported-input hash checks remain active.
        patch.object(v2, "migration_code", return_value=[]).start()
        patch.object(v2.history_v2, "audit_history", return_value=self.history).start()
        patch.object(v2.requalify_v2, "requalify", side_effect=self.requalify).start()
        patch.object(run, "service_output_directory", return_value=self.service).start()
        patch.object(v2, "recover_orphan", return_value=None).start()
        self.campaign = v2.migrate(self.source, self.directory)

    def image_row(self, job):
        directory = self.root / f"image_{job['place_id']}"
        directory.mkdir()
        raw, output = directory / "raw.png", directory / "image.png"
        Image.new("RGB", (30, 24), (job["place_id"] * 25, 90, 100)).save(raw)
        with Image.open(raw) as opened:
            opened.resize((20, 16), Image.Resampling.LANCZOS).save(output)
        return run.sealed({**job, "execution_fingerprint": self.old_execution["fingerprint"],
            "output_path": str(output), "output_sha256": common.file_sha256(output),
            "raw_output_path": str(raw), "raw_output_sha256": common.file_sha256(raw)})

    def requalify(self, row, execution):
        if row.get("status") not in {"passed", "rejected"}:
            return None
        return run.sealed({**row, "execution_fingerprint": execution["fingerprint"],
            "status": "passed", "passed": True, "eligible_for_training": True,
            "rejection_reasons": [], "quality_migration": {
                "original_execution_fingerprint": row["execution_fingerprint"],
                "original_result_sha256": row["result_sha256"], "model_calls": 0, "http_calls": 0}})

    def catalog(self):
        return v2.Catalog(self.directory, self.campaign)

    def assert_legacy_unchanged(self):
        for path, data in self.legacy_files.items():
            self.assertEqual(Path(path).read_bytes(), data, path)

    def verify_saved(self, command, catalog, state):
        self.assertEqual(command[3], "verify-one")
        destination = Path(command[command.index("--verification-dir") + 1])
        with patch.object(run, "QwenQualityVerifier", return_value=Verifier()):
            v2.base.verify_one(destination)
        catalog.refresh().publish(state)

    def generate_stage(self, stage, *, fail=False):
        session, health = Mock(), Mock()
        health.json.return_value = self.health
        session.get.return_value = health
        if fail:
            session.post.side_effect = requests.Timeout("simulated interrupted HTTP request")
        else:
            def post(url, json, timeout):
                jobs = run.load_plan(Path(stage["run_dir"]))[1]
                job = next(j for j in jobs if j["seed"] == json["seed"])
                self.assertEqual(json["prompt"], job["prompt"])
                self.assertEqual(json["infer_steps"], 4)
                self.assertEqual(json["guidance_scale"], 1.0)
                raw = self.service / f"{job['sample_id']}.png"
                Image.new("RGB", (30, 24), (90, 110, 120)).save(raw)
                result = Mock()
                result.json.return_value = {"result_path": str(raw), "canvas_policy": "source_aspect_v1",
                    "source_dimensions": [20, 16], "raw_dimensions": [30, 24], "target_shape": [24, 30]}
                return result
            session.post.side_effect = post
        args = run.parse_args(["generate", "--run-dir", stage["run_dir"], "--max-calls", str(stage["max_calls"]),
            "--request-timeout", "600", "--device", "cuda", "--cpu-threads", "4"])
        with patch.object(run.requests, "Session", return_value=session), \
                patch.object(run, "QwenQualityVerifier", return_value=Verifier()):
            if fail:
                with self.assertRaisesRegex(requests.Timeout, "interrupted HTTP"):
                    run.generate(args)
            else:
                run.generate(args)
        return session

    def finish(self):
        catalog = self.catalog()
        with patch.object(v2.base, "run_child", side_effect=self.verify_saved):
            v2.base.recover_saved(catalog)
        stage = v2.continuation(catalog, catalog.pending())
        session = self.generate_stage(stage)
        catalog.refresh().publish("complete")
        return catalog, session

    def make_orphan(self, name="orphan"):
        raw = self.service / f"{name}.png"
        Image.new("RGB", (30, 24), (85, 95, 105)).save(raw)
        common.write_json(raw.with_suffix(".json"), {
            "source_path": str(self.source / "source_inputs" / f"{self.jobs[3]['sample_id']}.png"),
            "seed": self.jobs[3]["seed"], "result_path": str(raw),
            "metadata_path": str(raw.with_suffix(".json")), "canvas_policy": "source_aspect_v1",
            "source_dimensions": [20, 16], "raw_dimensions": [30, 24], "target_shape": [24, 30]})
        return raw

    def test_migration_reuses_saved_images_and_promotes_rejected_without_generation(self):
        catalog = self.catalog()
        self.assertEqual(catalog.pending(), self.jobs[3:])
        self.assertEqual(len(catalog.generated), 3)
        self.assertEqual(len(catalog.results), 2)
        promoted = catalog.results[self.jobs[1]["sample_id"]]
        self.assertTrue(promoted["passed"])
        self.assertEqual(promoted["output_sha256"], self.generated[1]["output_sha256"])
        self.assertEqual(promoted["quality_migration"]["original_result_sha256"], self.results[1]["result_sha256"])
        migration = v2.config_file(self.directory / "migration_config.json")
        self.assertEqual(migration["report"]["promoted_by_v2"], 1)
        self.assertEqual(migration["report"]["new_generation_calls"], 0)
        self.assertEqual(v2.migrate(self.source, self.directory), self.campaign)
        self.assert_legacy_unchanged()

    def test_imported_saved_raw_is_verified_without_post_and_keeps_image_bytes(self):
        catalog = self.catalog()
        with patch.object(v2.base, "run_child", side_effect=self.verify_saved), \
                patch.object(run.requests, "Session") as http:
            v2.base.recover_saved(catalog)
        http.assert_not_called()
        self.assertEqual(len(catalog.results), 3)
        self.assertEqual(catalog.results[self.jobs[2]["sample_id"]]["raw_output_sha256"],
                         self.generated[2]["raw_output_sha256"])
        self.assertEqual(catalog.pending(), self.jobs[3:])
        self.assert_legacy_unchanged()

    def test_orphan_recovery_copies_audited_raw_without_mutating_historical_bytes(self):
        raw = self.make_orphan()
        original = {p: p.read_bytes() for p in (raw, raw.with_suffix(".json"))}
        imported = self.directory / "imported"
        execution = v2.config_file(imported / "execution_config.json")
        value = self.history.attempts[self.jobs[3]["sample_id"]][0]
        with patch.object(run.requests, "Session") as http:
            saved = RECOVER_ORPHAN(self.history, self.jobs[3], value, imported, execution)
        http.assert_not_called()
        self.assertEqual(saved["raw_output_sha256"], common.file_sha256(raw))
        self.assertEqual(saved["migration"]["original_attempt_sha256"], self.attempt["attempt_sha256"])
        self.assertEqual(saved["migration"]["new_generation_calls"], 0)
        self.assertEqual(saved["migration"]["sidecar_does_not_prove"], ["prompt", "service_identity"])
        run.validate_records([saved], execution, [self.jobs[3]])
        with Image.open(saved["output_path"]) as normalized:
            self.assertEqual(normalized.size, (20, 16))
        self.assertEqual(original, {p: p.read_bytes() for p in original})
        self.assert_legacy_unchanged()

    def test_orphan_with_wrong_canvas_or_ambiguous_raw_cannot_be_imported(self):
        raw = self.make_orphan()
        metadata_path = raw.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text())
        metadata["target_shape"] = [30, 24]
        common.write_json(metadata_path, metadata)
        imported = self.directory / "imported"
        execution = v2.config_file(imported / "execution_config.json")
        value = self.history.attempts[self.jobs[3]["sample_id"]][0]
        with self.assertRaisesRegex(ValueError, "canvas differ"):
            RECOVER_ORPHAN(self.history, self.jobs[3], value, imported, execution)
        self.make_orphan()
        self.make_orphan("second")
        with self.assertRaisesRegex(ValueError, "Multiple matching service outputs"):
            RECOVER_ORPHAN(self.history, self.jobs[3], value, imported, execution)
        self.assertFalse((imported / "raw" / f"{self.jobs[3]['sample_id']}.png").exists())

    def test_continuation_keeps_exact_jobs_and_links_interrupted_legacy_request(self):
        catalog = self.catalog()
        stage = v2.continuation(catalog, catalog.pending())
        self.assertEqual(run.load_plan(Path(stage["run_dir"]))[1], self.jobs[3:])
        self.assertEqual(stage["legacy_retry_of"], {self.jobs[3]["sample_id"]: {
            "run_dir": str(self.source), "attempt_sha256": self.attempt["attempt_sha256"]}})
        self.assertEqual(stage["retry_of"], {})
        self.assertEqual(stage["max_calls"], 2)
        self.assertEqual(len(catalog.refresh().stages), 1)
        self.assert_legacy_unchanged()

    def test_failed_request_is_retried_only_in_a_new_immutable_execution(self):
        catalog = self.catalog()
        first = v2.continuation(catalog, catalog.pending())
        session = self.generate_stage(first, fail=True)
        self.assertEqual(session.post.call_count, 1)
        first_dir = Path(first["run_dir"])
        before = {name: (first_dir / name).read_bytes() for name in
                  ("execution_config.json", "attempts.jsonl", "results.jsonl")}
        catalog.refresh()
        second = v2.continuation(catalog, catalog.pending())
        self.assertNotEqual(first["run_dir"], second["run_dir"])
        self.assertEqual(second["retry_of"][self.jobs[3]["sample_id"]]["run_dir"], first["run_dir"])
        self.assertNotIn(self.jobs[4]["sample_id"], second["retry_of"])
        session = self.generate_stage(second)
        self.assertEqual(session.post.call_count, 2)
        catalog.refresh()
        self.assertEqual(catalog.pending(), [])
        self.assertEqual(len(catalog.attempts[self.jobs[3]["sample_id"]]), 2)
        self.assertEqual(len(catalog.results), 4)  # Imported saved-only row still needs review.
        self.assertEqual(before, {name: (first_dir / name).read_bytes() for name in before})
        self.assert_legacy_unchanged()

    def test_complete_requires_every_saved_image_to_have_quality_results(self):
        catalog = self.catalog()
        stage = v2.continuation(catalog, catalog.pending())
        self.generate_stage(stage)
        catalog.refresh().publish("complete")
        self.assertEqual(catalog.pending(), [])
        self.assertEqual(len(catalog.results), 4)
        with self.assertRaisesRegex(ValueError, "Finish current-quality"):
            v2.load_collection(self.directory)
        with patch.object(v2.base, "run_child", side_effect=self.verify_saved):
            v2.base.recover_saved(catalog)
        catalog.publish("complete")
        rows, _ = v2.load_collection(self.directory)
        self.assertEqual(len(rows), 5)
        self.assertEqual(Counter(r["condition"] for r in rows), self.campaign["domain_quotas"])
        self.assert_legacy_unchanged()

    def test_completed_resume_preserves_summary_and_training_snapshots(self):
        _, session = self.finish()
        self.assertEqual(session.post.call_count, 2)
        names = ("summary.json", "results.jsonl", "generated.jsonl", "training_manifest.jsonl")
        before = {name: (self.directory / name).read_bytes() for name in names}
        with patch.object(v2.base, "stop_qwen") as stop, patch.object(v2.base, "start_qwen") as start, \
                patch.object(v2.base, "run_child") as child:
            v2.execute(self.directory, self.campaign)
        self.assertEqual(stop.call_count, 2)
        start.assert_not_called()
        child.assert_not_called()
        self.assertEqual(before, {name: (self.directory / name).read_bytes() for name in names})
        self.assert_legacy_unchanged()

    def test_input_or_imported_image_mutation_is_rejected(self):
        path = self.directory / "combined_plan.jsonl"
        original = path.read_bytes()
        path.write_text("[]\n")
        with self.assertRaisesRegex(ValueError, "Migration input/code changed"):
            v2.check_inputs(self.directory)
        path.write_bytes(original)
        Path(self.generated[0]["raw_output_path"]).write_bytes(b"corrupted image")
        with self.assertRaises(ValueError):
            self.catalog()

    def test_forged_legacy_retry_provenance_is_rejected_even_with_new_stage_seal(self):
        catalog = self.catalog()
        stage = v2.continuation(catalog, catalog.pending())
        stage["legacy_retry_of"][self.jobs[3]["sample_id"]]["attempt_sha256"] = "0" * 64
        body = {k: value for k, value in stage.items() if k != "fingerprint"}
        common.write_json(Path(stage["run_dir"]) / "stage_config.json",
                          {**body, "fingerprint": common.fingerprint(body)})
        with self.assertRaisesRegex(ValueError, "no original request reservation"):
            catalog.refresh()

    def test_execute_failure_stops_qwen_and_next_start_can_complete(self):
        def child(command, catalog, state):
            if command[3] == "verify-one":
                return self.verify_saved(command, catalog, state)
            stage = v2.config_file(Path(command[command.index("--run-dir") + 1]) / "stage_config.json")
            self.generate_stage(stage, fail=True)
            raise RuntimeError("simulated worker failed")
        with patch.object(v2.base, "stop_qwen") as stop, patch.object(v2.base, "start_qwen"), \
                patch.object(v2.base, "run_child", side_effect=child):
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                v2.execute(self.directory, self.campaign)
        self.assertEqual(stop.call_count, 2)
        self.assertEqual(json.loads((self.directory / "summary.json").read_text())["state"], "stopped")
        self.assertEqual(self.catalog().pending(), self.jobs[3:])

        def succeed(command, catalog, state):
            self.assertEqual(command[3], "generate")
            stage = v2.config_file(Path(command[command.index("--run-dir") + 1]) / "stage_config.json")
            self.generate_stage(stage)
            catalog.refresh().publish(state)
        with patch.object(v2.base, "stop_qwen"), patch.object(v2.base, "start_qwen"), \
                patch.object(v2.base, "run_child", side_effect=succeed):
            v2.execute(self.directory, self.campaign)
        self.assertEqual(len(v2.load_collection(self.directory)[0]), 5)
        self.assert_legacy_unchanged()

    def test_competing_worker_lock_fails_before_stopping_active_qwen(self):
        catalog = self.catalog()
        stage = v2.continuation(catalog, catalog.pending())
        with run.run_lock(Path(stage["run_dir"])), patch.object(v2.base, "stop_qwen") as stop:
            with self.assertRaisesRegex(RuntimeError, "Another worker owns"):
                v2.execute(self.directory, self.campaign)
        stop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
