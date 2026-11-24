"""CPU integration coverage for the only supported 23-city runtime."""
from collections import Counter
import json
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.qwen_curriculum import common, run, campaign, resume_recovery, image_store


class Verifier:
    def evaluate(self, source, generated, condition):
        return {"status": "passed", "passed": True, "eligible_for_training": True,
                "weather_ok": True, "s_geo": 0.91, "weather_shift": 1.5}


class CampaignTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "state"
        self.parent = self.directory / "imported"
        self.parent.mkdir(parents=True)
        self.service = self.root / "service"
        self.service.mkdir()
        self.jobs = []
        for number in range(4):
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
        master = self.directory / "additional_plan"
        master.mkdir()
        self.plan = campaign.frozen(master / "plan_config.json", {
            "output_dir": str(master), "num_images": 4, "seed": 42,
            "domain_quotas": {"night": 4}, "plan_sha256": "historical-unused"})
        common.write_jsonl(self.directory / "combined_plan.jsonl", self.jobs)
        reference = {**run.execution_definition(SimpleNamespace(run_dir=self.directory, device="cuda", cpu_threads=4)),
            "stage": "generate", "plan_fingerprint": self.plan["fingerprint"], "max_calls": 4,
            "qwen_url": "http://127.0.0.1:8001/generate", "request_timeout": 600.0,
            "service": run.service_identity(self.health),
            "call_policy": "one POST per job; durable reservation; no automatic retries",
            "source_encoding": "lossless RGB PNG decoded from original"}
        self.execution = campaign.frozen(self.parent / "execution_config.json", {
            **run.execution_definition(SimpleNamespace(run_dir=self.parent, device="cuda", cpu_threads=4)),
            "stage": "quality_v2_import", "max_calls": 0})
        self.generated = [self.image_row(job) for job in self.jobs[:2]]
        self.results = [run.sealed({**self.generated[0], "status": "passed", "passed": True,
                                   "eligible_for_training": True})]
        common.write_jsonl(self.parent / "generated.jsonl", self.generated)
        common.write_jsonl(self.parent / "results.jsonl", self.results)
        common.write_jsonl(self.parent / "historical_attempts.jsonl", [])
        self.config = campaign.frozen(self.directory / "all_city_campaign_config.json", {
            "stage": "current_quality_23city_migration", "parent_run_dir": str(self.parent),
            "combined_plan": str(self.directory / "combined_plan.jsonl"), "total_images": 4,
            "additional_images": 2, "reference_execution": reference, "stages": [],
            "runtime_profile": "resident_bf16_48g", "domain_quotas": {"night": 4}, "city_quotas": {"City": 4}})
        campaign.frozen(self.directory / "resume_config.json", {"campaign_fingerprint": self.config["fingerprint"],
            "files_sha256": {"removed_old_coordinator.py": "historical-evidence-only"}})
        files = [self.directory / "all_city_campaign_config.json", self.directory / "combined_plan.jsonl",
                 master / "plan_config.json"]
        files += list(self.parent.glob("*.json*"))
        files += [Path(module.__file__).resolve() for module in (campaign, common, run, resume_recovery, image_store)]
        campaign.frozen(self.directory / "runtime_config.json", {"schema_version": 1,
            "campaign_fingerprint": self.config["fingerprint"],
            "files_sha256": {str(p): common.file_sha256(p) for p in files}})
        self.imported_snapshot = {p: p.read_bytes() for p in self.parent.glob("*.json*")}
        self.service_patch = patch.object(run, "service_output_directory", return_value=self.service)
        self.service_patch.start()
        self.addCleanup(self.service_patch.stop)
        self.catalog().publish("paused")

    def image_row(self, job):
        directory = self.parent / "saved" / job["sample_id"]
        directory.mkdir(parents=True)
        raw, output = directory / "raw.png", directory / "image.png"
        Image.new("RGB", (30, 24), (job["place_id"] * 25, 90, 100)).save(raw)
        with Image.open(raw) as image:
            image.resize((20, 16), Image.Resampling.LANCZOS).save(output)
        return run.sealed({**job, "execution_fingerprint": self.execution["fingerprint"],
            "output_path": str(output), "output_sha256": common.file_sha256(output),
            "raw_output_path": str(raw), "raw_output_sha256": common.file_sha256(raw)})

    def catalog(self):
        return campaign.Catalog(self.directory, self.config)

    def verify_saved(self, command, catalog, state):
        self.assertEqual(command[3], "verify-one")
        directory = Path(command[command.index("--verification-dir") + 1])
        with patch.object(run, "QwenQualityVerifier", return_value=Verifier()):
            campaign.verify_one(directory)
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
                self.assertEqual((json["infer_steps"], json["guidance_scale"]), (4, 1.0))
                raw = self.service / f"{job['sample_id']}.png"
                Image.new("RGB", (30, 24), (90, 110, 120)).save(raw)
                response = Mock()
                response.json.return_value = {"result_path": str(raw), "canvas_policy": "source_aspect_v1",
                    "source_dimensions": [20, 16], "raw_dimensions": [30, 24], "target_shape": [24, 30]}
                return response
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

    def succeed(self, command, catalog, state):
        if command[3] == "verify-one":
            return self.verify_saved(command, catalog, state)
        stage = campaign.config_file(Path(command[command.index("--run-dir") + 1]) / "stage_config.json")
        session = self.generate_stage(stage)
        catalog.refresh().publish(state)
        return session

    def finish(self):
        with patch.object(campaign, "stop_qwen"), patch.object(campaign, "start_qwen"), \
                patch.object(campaign, "run_child", side_effect=self.succeed):
            campaign.execute(self.directory, self.config)

    def test_saved_images_verify_without_http_and_keep_imported_journals(self):
        catalog = self.catalog()
        with patch.object(campaign, "run_child", side_effect=self.verify_saved), \
                patch.object(run.requests, "Session") as http:
            campaign.recover_saved(catalog)
        http.assert_not_called()
        self.assertEqual(len(catalog.results), 2)
        self.assertEqual(catalog.pending(), self.jobs[2:])
        self.assertEqual(self.imported_snapshot, {p: p.read_bytes() for p in self.imported_snapshot})

    def test_pending_verification_resumes_after_repeated_process_interruptions_without_post(self):
        catalog = self.catalog()
        sid = self.jobs[1]["sample_id"]
        original, origin = catalog.generated[sid], catalog.origins[sid]
        directory = self.directory / "resume_state/verifications" / sid / origin["execution"]["fingerprint"][:24]
        # The independent execution and generated row are durable, but quality
        # verification has not committed a result when this process stops.
        campaign.make_verification(directory, original, origin["execution"])
        before = {p: p.read_bytes() for p in directory.glob("*.json*")}
        for _ in range(2):
            with patch.object(campaign, "stop_qwen"), patch.object(campaign, "start_qwen") as start, \
                    patch.object(campaign, "run_child", side_effect=KeyboardInterrupt), \
                    patch.object(run.requests, "Session") as http:
                with self.assertRaises(KeyboardInterrupt):
                    campaign.execute(self.directory, self.config)
            start.assert_not_called()
            http.assert_not_called()
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            self.assertFalse((directory / "results.jsonl").exists())
        catalog = self.catalog()
        self.assertEqual(catalog.origins[sid]["execution"]["fingerprint"], origin["execution"]["fingerprint"])
        with patch.object(campaign, "run_child", side_effect=self.verify_saved), \
                patch.object(run.requests, "Session") as http:
            campaign.recover_saved(catalog)
        http.assert_not_called()
        self.assertEqual(catalog.results[sid]["output_sha256"], original["output_sha256"])
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertEqual(len(list((self.directory / "resume_state/verifications" / sid).glob("*/execution_config.json"))), 1)

    def test_execution_finishes_exact_plan_and_collects_images(self):
        self.finish()
        rows, cfg = campaign.load_collection(self.directory)
        self.assertEqual(cfg, self.config)
        self.assertEqual(Counter(r["condition"] for r in rows), {"night": 4})
        for row in rows:
            for field in ("output_path", "raw_output_path"):
                path = Path(row[field])
                self.assertTrue(path.is_symlink())
                self.assertTrue(path.resolve().is_relative_to(self.root / "images"))
                self.assertEqual(common.file_sha256(path), row[field.replace("_path", "_sha256")])
        self.assertEqual(len(list((self.root / "images/accepted/night").glob("*.png"))), 4)

    def test_runtime_seal_rejects_changed_input_and_ignores_legacy_code_evidence(self):
        self.assertEqual(campaign.check_inputs(self.directory), self.config)
        path = self.directory / "combined_plan.jsonl"
        path.write_text("[]\n")
        with self.assertRaisesRegex(ValueError, "Runtime input/code changed"):
            campaign.check_inputs(self.directory)

    def test_runtime_must_bind_the_exact_campaign(self):
        path = self.directory / "runtime_config.json"
        value = campaign.config_file(path)
        value.update(campaign_fingerprint="another-campaign")
        value.pop("fingerprint")
        path.unlink()
        campaign.frozen(path, value)
        with self.assertRaisesRegex(ValueError, "different campaign"):
            campaign.check_inputs(self.directory)

    def test_runtime_schema_cannot_be_bool_text_or_a_different_version(self):
        path = self.directory / "runtime_config.json"
        original = campaign.config_file(path)
        for schema in (True, "1", 2):
            body = {k: value for k, value in original.items() if k != "fingerprint"}
            body["schema_version"] = schema
            path.unlink()
            campaign.frozen(path, body)
            with self.subTest(schema=schema), self.assertRaisesRegex(ValueError, "Unsupported runtime schema"):
                campaign.check_inputs(self.directory)

    def test_runtime_requires_static_inputs_and_current_code_even_with_a_valid_seal(self):
        path = self.directory / "runtime_config.json"
        original = campaign.config_file(path)
        for omitted in (self.directory / "combined_plan.jsonl", Path(run.__file__).resolve()):
            body = {k: value for k, value in original.items() if k != "fingerprint"}
            body["files_sha256"] = {k: value for k, value in body["files_sha256"].items() if k != str(omitted)}
            path.unlink()
            campaign.frozen(path, body)
            with self.subTest(omitted=omitted), self.assertRaisesRegex(ValueError, "every required static input"):
                campaign.check_inputs(self.directory)

    def test_failed_runtime_validation_does_not_rewrite_progress_or_start_models(self):
        before = {name: (self.directory / name).read_bytes() for name in
                  ("summary.json", "results.jsonl", "generated.jsonl", "training_manifest.jsonl")}
        (self.directory / "combined_plan.jsonl").write_text("changed plan\n")
        with patch.object(campaign, "stop_qwen"), patch.object(campaign, "start_qwen") as start, \
                patch.object(campaign, "run_child") as child:
            with self.assertRaisesRegex(ValueError, "Runtime input/code changed"):
                campaign.execute(self.directory, self.config)
        start.assert_not_called()
        child.assert_not_called()
        self.assertEqual(before, {name: (self.directory / name).read_bytes() for name in before})

    def test_incomplete_campaign_never_becomes_trainable(self):
        for require_complete in (True, False):
            with self.assertRaisesRegex(ValueError, "Finish all 23-city"):
                campaign.load_collection(self.directory, require_complete=require_complete)
        catalog = self.catalog()
        self.generate_stage(campaign.continuation(catalog, catalog.pending()))
        catalog.refresh().publish("complete")
        with self.assertRaisesRegex(ValueError, "Finish all 23-city"):
            campaign.load_collection(self.directory)

    def test_failure_reserves_one_call_then_explicit_resume_creates_new_stage(self):
        catalog = self.catalog()
        first = campaign.continuation(catalog, catalog.pending())
        self.assertEqual(self.generate_stage(first, fail=True).post.call_count, 1)
        first_dir = Path(first["run_dir"])
        before = {name: (first_dir / name).read_bytes() for name in
                  ("execution_config.json", "attempts.jsonl", "results.jsonl")}
        catalog.refresh()
        second = campaign.continuation(catalog, catalog.pending())
        self.assertEqual(second["retry_of"][self.jobs[2]["sample_id"]]["run_dir"], str(first_dir))
        self.assertNotEqual(first["run_dir"], second["run_dir"])
        self.assertEqual(self.generate_stage(second).post.call_count, 2)
        self.assertEqual(before, {name: (first_dir / name).read_bytes() for name in before})
        catalog.refresh()
        self.assertEqual(catalog.pending(), [])
        self.assertEqual(len(catalog.attempts[self.jobs[2]["sample_id"]]), 2)

    def test_forged_retry_link_is_rejected_even_if_stage_is_resealed(self):
        catalog = self.catalog()
        first = campaign.continuation(catalog, catalog.pending())
        self.generate_stage(first, fail=True)
        catalog.refresh()
        second = campaign.continuation(catalog, catalog.pending())
        path = Path(second["run_dir"]) / "stage_config.json"
        value = campaign.config_file(path)
        value["retry_of"][self.jobs[2]["sample_id"]]["attempt_sha256"] = "0" * 64
        value.pop("fingerprint")
        path.unlink()
        campaign.frozen(path, value)
        with self.assertRaisesRegex(ValueError, "no matching prior request"):
            catalog.refresh()

    def test_interrupted_saved_raw_recovers_without_an_extra_request(self):
        catalog = self.catalog()
        first = campaign.continuation(catalog, catalog.pending())
        self.generate_stage(first, fail=True)
        stage_dir = Path(first["run_dir"])
        raw = self.service / "interrupted.png"
        Image.new("RGB", (30, 24), (90, 110, 120)).save(raw)
        common.write_json(raw.with_suffix(".json"), {"source_path": str(stage_dir / "source_inputs" / f"{self.jobs[2]['sample_id']}.png"),
            "seed": self.jobs[2]["seed"], "result_path": str(raw), "metadata_path": str(raw.with_suffix(".json")),
            "canvas_policy": "source_aspect_v1", "source_dimensions": [20, 16], "raw_dimensions": [30, 24], "target_shape": [24, 30]})
        calls = []
        def child(command, managed, state):
            result = self.succeed(command, managed, state)
            if result is not None:
                calls.append(result.post.call_count)
        with patch.object(campaign, "stop_qwen"), patch.object(campaign, "start_qwen"), \
                patch.object(campaign, "run_child", side_effect=child):
            campaign.execute(self.directory, self.config)
        self.assertEqual(calls, [1])
        rows = campaign.load_collection(self.directory)[0]
        recovered = next(r for r in rows if r["sample_id"] == self.jobs[2]["sample_id"])
        self.assertEqual(recovered["recovery"]["http_calls"], 0)
        self.assertEqual(recovered["raw_output_sha256"], common.file_sha256(raw))

    def test_saved_continuation_needs_no_post_after_verification_interruption(self):
        catalog = self.catalog()
        stage = campaign.continuation(catalog, catalog.pending())
        args = run.parse_args(["generate", "--run-dir", stage["run_dir"], "--max-calls", "2", "--request-timeout", "600"])
        session, health, response = Mock(), Mock(), Mock()
        health.json.return_value = self.health
        session.get.return_value = health
        raw = self.service / "quality_interrupted.png"
        Image.new("RGB", (30, 24), (90, 110, 120)).save(raw)
        response.json.return_value = {"result_path": str(raw), "canvas_policy": "source_aspect_v1",
            "source_dimensions": [20, 16], "raw_dimensions": [30, 24], "target_shape": [24, 30]}
        session.post.return_value = response
        broken = Mock()
        broken.evaluate.side_effect = RuntimeError("verification interrupted")
        with patch.object(run.requests, "Session", return_value=session), \
                patch.object(run, "QwenQualityVerifier", return_value=broken):
            with self.assertRaisesRegex(RuntimeError, "verification interrupted"):
                run.generate(args)
        self.assertEqual(session.post.call_count, 1)
        catalog.refresh()
        with patch.object(campaign, "run_child", side_effect=self.verify_saved), \
                patch.object(run.requests, "Session") as http:
            campaign.recover_saved(catalog)
        http.assert_not_called()
        self.assertEqual(len(catalog.results), 3)
        self.assertEqual(catalog.pending(), self.jobs[3:])

    def test_completed_resume_preserves_frozen_training_snapshots(self):
        self.finish()
        names = ("summary.json", "results.jsonl", "generated.jsonl", "training_manifest.jsonl")
        before = {name: (self.directory / name).read_bytes() for name in names}
        with patch.object(campaign, "stop_qwen") as stop, patch.object(campaign, "start_qwen") as start, \
                patch.object(campaign, "run_child") as child:
            campaign.execute(self.directory, self.config)
        self.assertEqual(stop.call_count, 2)
        start.assert_not_called()
        child.assert_not_called()
        self.assertEqual(before, {name: (self.directory / name).read_bytes() for name in names})

    def test_storage_failure_interrupts_owned_worker_and_releases_qwen(self):
        synchronize = campaign.Catalog.synchronize
        def fail_during_generation(catalog):
            if json.loads((self.directory / "summary.json").read_text())["state"] == "generating":
                raise OSError("simulated image library disk failure")
            return synchronize(catalog)
        worker = Mock()
        worker.poll.return_value = None
        worker.wait.return_value = 0
        with patch.object(campaign.Catalog, "synchronize", autospec=True, side_effect=fail_during_generation), \
                patch.object(campaign, "stop_qwen") as stop, patch.object(campaign, "start_qwen"), \
                patch.object(campaign, "recover_saved"), patch.object(campaign.subprocess, "Popen", return_value=worker):
            with self.assertRaisesRegex(OSError, "image library disk failure"):
                campaign.execute(self.directory, self.config)
        worker.send_signal.assert_called_once_with(signal.SIGINT)
        worker.wait.assert_called_once_with(timeout=10)
        self.assertEqual(stop.call_count, 2)
        self.assertEqual(json.loads((self.directory / "summary.json").read_text())["state"], "stopped")

    def test_competing_worker_lock_is_checked_before_stopping_qwen(self):
        catalog = self.catalog()
        stage = campaign.continuation(catalog, catalog.pending())
        with run.run_lock(Path(stage["run_dir"])), patch.object(campaign, "stop_qwen") as stop:
            with self.assertRaisesRegex(RuntimeError, "Another worker owns"):
                campaign.execute(self.directory, self.config)
        stop.assert_not_called()

    def test_stage_budget_and_exact_jobs_cannot_change(self):
        catalog = self.catalog()
        with self.assertRaisesRegex(ValueError, "1-1000 exact original"):
            campaign.continuation(catalog, catalog.pending() * 501)
        with self.assertRaisesRegex(ValueError, "1-1000 exact original"):
            campaign.continuation(catalog, [{**self.jobs[2], "seed": 99}])
        with self.assertRaisesRegex(ValueError, "1-1000 exact original"):
            campaign.continuation(catalog, [self.jobs[0]])


if __name__ == "__main__":
    unittest.main()
