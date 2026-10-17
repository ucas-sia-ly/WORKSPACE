"""CPU continuation tests exercise unchanged generation and independent journals."""
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
from experiments.qwen_curriculum import common, run, fast_campaign, resume_campaign as resume, finetune_all_cities


class Verifier:
    def evaluate(self, source, generated, condition):
        return {"status": "passed", "passed": True, "eligible_for_training": True,
                "s_geo": 0.91, "weather_shift": 1.5}


class ResumeCampaignTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "campaign"
        self.directory.mkdir()
        self.original = self.directory / "additional_001_1000"
        self.original.mkdir()
        self.service = self.root / "service"
        self.service.mkdir()
        self.jobs = []
        for number in range(4):
            source = self.root / f"source_{number}.png"
            Image.new("RGB", (20, 16), (number * 25, 80, 45)).save(source)
            body = {"sample_id": "qwen_" + common.fingerprint(number)[:24], "source_path": str(source),
                "source_sha256": common.file_sha256(source), "source_dimensions": [20, 16],
                "condition": "night", "city": "City", "place_id": number, "seed": 123 + number,
                "prompt": f"source anchored night prompt {number}", "negative_prompt": ""}
            self.jobs.append({**body, "record_sha256": common.fingerprint(body)})
        common.write_jsonl(self.directory / "combined_plan.jsonl", self.jobs)
        common.write_jsonl(self.original / "plan.jsonl", self.jobs[1:])
        plan = fast_campaign.frozen(self.original / "plan_config.json", {
            "output_dir": str(self.original), "num_images": 3, "seed": 42,
            "domain_quotas": {"night": 3}, "plan_sha256": common.file_sha256(self.original / "plan.jsonl")})
        master = self.directory / "additional_plan"
        master.mkdir()
        common.write_json(master / "plan_config.json", plan)
        self.health = {"status": "ok", "model_loaded": True, "generator_ready": True, "error": None,
            "model_id": "Qwen/Qwen-Image-Edit-2511", "source_modified": False,
            "canvas_policy": "source_aspect_v1", "sampling": {"infer_steps": 4, "guidance_scale": 1.0},
            "canvas": {"target_shape_order": "height,width", "target_pixels": 720, "multiple": 1,
                "min_side": 1, "max_side": 128, "rounding": "nearest_multiple", "unsupported_aspect_ratio": "reject"}}
        self.execution = run.freeze(self.original, {
            **run.execution_definition(SimpleNamespace(run_dir=self.original, device="cuda", cpu_threads=4)),
            "stage": "generate", "plan_fingerprint": plan["fingerprint"], "max_calls": 3,
            "qwen_url": "http://127.0.0.1:8001/generate", "request_timeout": 600.0,
            "service": run.service_identity(self.health),
            "call_policy": "one POST per job; durable reservation; no automatic retries",
            "source_encoding": "lossless RGB PNG decoded from original"})
        self.attempts = [run.sealed_attempt({"sample_id": job["sample_id"], "record_sha256": job["record_sha256"],
            "execution_fingerprint": self.execution["fingerprint"], "call_number": number})
            for number, job in enumerate(self.jobs[1:], 1)]
        common.write_jsonl(self.original / "attempts.jsonl", self.attempts)
        self.error = run.sealed({**self.jobs[1], "execution_fingerprint": self.execution["fingerprint"],
            "status": "generation_error", "passed": False, "eligible_for_training": False, "error": "interrupted"})
        self.saved = self.image_row(self.jobs[2], self.execution)
        rejected_saved = self.image_row(self.jobs[3], self.execution)
        self.rejected = run.sealed({**rejected_saved, "status": "rejected", "passed": False,
                                   "eligible_for_training": False})
        common.write_jsonl(self.original / "generated.jsonl", [self.saved, rejected_saved])
        common.write_jsonl(self.original / "results.jsonl", [self.error, self.rejected])
        parent_saved = self.image_row(self.jobs[0], {"fingerprint": "original-parent-execution"})
        self.parent = run.sealed({**parent_saved, "status": "passed", "passed": True, "eligible_for_training": True})
        (self.root / "parent").mkdir()
        common.write_jsonl(self.root / "parent/generated.jsonl", [parent_saved])
        self.campaign = fast_campaign.frozen(self.directory / "all_city_campaign_config.json", {
            "parent_run_dir": str(self.root / "parent"), "combined_plan": str(self.directory / "combined_plan.jsonl"),
            "total_images": 4, "additional_images": 3, "reference_execution": self.execution,
            "stages": [{"run_dir": str(self.original), "num_images": 3, "max_calls": 3,
                "plan_fingerprint": plan["fingerprint"]}], "runtime_profile": "resident_bf16_48g",
            "domain_quotas": {"night": 4}, "city_quotas": {"City": 4}})
        self.addCleanup(patch.stopall)
        patch.object(resume.scale_campaign, "load_collection", return_value=([self.parent], {})).start()
        patch.object(resume.all_city_campaign, "check_inputs").start()
        patch.object(run, "service_output_directory", return_value=self.service).start()
        resume.prepare(self.directory, self.campaign)
        self.immutable = {str(self.original / name): (self.original / name).read_bytes() for name in
            (*resume.JOURNALS, "plan.jsonl", "plan_config.json")}

    def image_row(self, job, execution):
        base = self.root / f"saved_{job['place_id']}"
        base.mkdir()
        raw, image = base / "raw.png", base / "image.png"
        Image.new("RGB", (30, 24), (job["place_id"] * 30, 90, 100)).save(raw)
        with Image.open(raw) as opened:
            opened.resize((20, 16), Image.Resampling.LANCZOS).save(image)
        return run.sealed({**job, "execution_fingerprint": execution["fingerprint"],
            "raw_output_path": str(raw), "raw_output_sha256": common.file_sha256(raw),
            "output_path": str(image), "output_sha256": common.file_sha256(image)})

    def catalog(self):
        return resume.Catalog(self.directory, self.campaign)

    def assert_original_unchanged(self):
        for path, data in self.immutable.items():
            self.assertEqual(Path(path).read_bytes(), data, path)

    def verify_saved(self, command, catalog, state):
        self.assertEqual(command[3], "verify-one")
        destination = Path(command[command.index("--verification-dir") + 1])
        with patch.object(run, "QwenQualityVerifier", return_value=Verifier()):
            resume.verify_one(destination)
        catalog.refresh().publish(state)

    def make_orphan(self):
        raw = self.service / "orphan.png"
        Image.new("RGB", (30, 24), (85, 95, 105)).save(raw)
        common.write_json(raw.with_suffix(".json"), {
            "source_path": str(self.original / "source_inputs" / f"{self.jobs[1]['sample_id']}.png"),
            "seed": self.jobs[1]["seed"], "result_path": str(raw), "metadata_path": str(raw.with_suffix(".json")),
            "canvas_policy": "source_aspect_v1", "source_dimensions": [20, 16],
            "raw_dimensions": [30, 24], "target_shape": [24, 30]})
        return raw

    def generate_stage(self, stage, fail=False):
        response = Mock()
        response.json.return_value = self.health
        session = Mock()
        session.get.return_value = response
        if fail:
            session.post.side_effect = requests.Timeout("simulated interrupted HTTP call")
        else:
            def post(url, json, timeout):
                self.assertEqual(json["seed"], self.jobs[1]["seed"])
                self.assertEqual(json["prompt"], self.jobs[1]["prompt"])
                self.assertEqual(json["infer_steps"], 4)
                self.assertEqual(json["guidance_scale"], 1.0)
                raw = self.service / "new-success.png"
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
        self.assertEqual(session.post.call_count, 1)
        return session

    def test_completed_rejected_and_saved_jobs_are_not_pending_new_posts(self):
        catalog = self.catalog()
        self.assertEqual(catalog.pending(), [self.jobs[1]])
        self.assertEqual(catalog.results[self.jobs[3]["sample_id"]], self.rejected)
        self.assertIn(self.jobs[2]["sample_id"], catalog.generated)
        self.assertNotIn(self.jobs[2]["sample_id"], catalog.results)

    def test_saved_verification_uses_no_post_or_mutation_of_original_generated_row(self):
        catalog = self.catalog()
        with patch.object(resume, "run_child", side_effect=self.verify_saved), \
                patch.object(run.requests, "Session") as http:
            resume.recover_saved(catalog)
        http.assert_not_called()
        self.assertIn(self.jobs[2]["sample_id"], catalog.results)
        self.assertEqual(catalog.pending(), [self.jobs[1]])
        self.assertEqual(catalog.results[self.jobs[2]["sample_id"]]["output_sha256"], self.saved["output_sha256"])
        self.assert_original_unchanged()

    def test_error_raw_is_recovered_and_verified_before_any_new_generation(self):
        self.make_orphan()
        catalog = self.catalog()
        with patch.object(resume, "run_child", side_effect=self.verify_saved), \
                patch.object(run.requests, "Session") as http:
            resume.recover_saved(catalog)
        http.assert_not_called()
        self.assertEqual(catalog.pending(), [])
        self.assertEqual(len(catalog.results), 4)
        report = catalog.publish("complete")
        self.assertEqual((report["generation_errors"], report["retry_calls_reserved"]), (0, 0))
        self.assertEqual(report["recovered_without_generation"], 1)
        self.assertEqual(len(resume.load_collection(self.directory)[0]), 4)
        self.assert_original_unchanged()

    def test_independent_retry_plan_keeps_exact_original_job_and_failure_journals(self):
        catalog = self.catalog()
        stage = resume.continuation(catalog, catalog.pending())
        _, jobs = run.load_plan(Path(stage["run_dir"]))
        self.assertEqual(jobs, [self.jobs[1]])
        self.assertEqual(stage["retry_of"][self.jobs[1]["sample_id"]]["attempt_sha256"], self.attempts[0]["attempt_sha256"])
        self.assertEqual(stage["max_calls"], 1)
        self.assert_original_unchanged()

    def test_retry_failure_then_next_explicit_resume_has_one_new_attempt_and_one_final_result(self):
        catalog = self.catalog()
        with patch.object(resume, "run_child", side_effect=self.verify_saved):
            resume.recover_saved(catalog)
        first = resume.continuation(catalog, catalog.pending())
        self.generate_stage(first, fail=True)
        catalog.refresh()
        self.assertEqual(catalog.pending(), [self.jobs[1]])
        self.assertEqual(len(catalog.attempts[self.jobs[1]["sample_id"]]), 2)
        self.assertEqual(len(run.read_rows(Path(first["run_dir"]), "attempts.jsonl")), 1)
        first_error = (Path(first["run_dir"]) / "results.jsonl").read_bytes()
        # A new explicit resume recovers any saved outputs, then allocates a new
        # execution.  The previous one never receives a second POST.
        with patch.object(resume, "run_child", side_effect=self.verify_saved):
            resume.recover_saved(catalog)
        second = resume.continuation(catalog, catalog.pending())
        self.assertNotEqual(first["run_dir"], second["run_dir"])
        self.assertEqual(second["retry_of"][self.jobs[1]["sample_id"]]["run_dir"], first["run_dir"])
        self.generate_stage(second)
        catalog.refresh()
        self.assertEqual(catalog.pending(), [])
        self.assertEqual(len(catalog.results), 4)
        report = catalog.publish("complete")
        self.assertEqual(report["retry_calls_reserved"], 2)
        self.assertEqual(len(catalog.attempts[self.jobs[1]["sample_id"]]), 3)
        self.assertEqual(len(run.read_rows(Path(second["run_dir"]), "attempts.jsonl")), 1)
        self.assertEqual((Path(first["run_dir"]) / "results.jsonl").read_bytes(), first_error)
        final = resume.load_collection(self.directory)[0]
        self.assertEqual(len({row["sample_id"] for row in final}), 4)
        self.assertEqual(Counter(row["condition"] for row in final), self.campaign["domain_quotas"])
        self.assert_original_unchanged()

    def test_forged_complete_summary_cannot_skip_pending_quality_checks(self):
        catalog = self.catalog()
        catalog.publish("complete")
        with self.assertRaisesRegex(ValueError, "Finish the resumed campaign"):
            resume.load_collection(self.directory)
        # Even when all raw bytes exist, the saved image lacking an evaluation
        # cannot enter training by changing only summary.state.
        self.make_orphan()
        with patch.object(resume, "run_child"):
            resume.recover_saved(catalog)
        self.assertEqual(catalog.pending(), [])
        catalog.publish("complete")
        with self.assertRaisesRegex(ValueError, "Finish the resumed campaign"):
            resume.load_collection(self.directory)

    def test_original_journal_edit_after_migration_is_rejected(self):
        common.write_jsonl(self.original / "results.jsonl", [])
        with self.assertRaisesRegex(ValueError, "original journal changed"):
            resume.prepare(self.directory, self.campaign)

    def test_interrupt_between_verification_config_and_generated_commit_can_resume(self):
        catalog = self.catalog()
        dest = self.directory / "resume_state/verifications" / self.saved["sample_id"] / self.execution["fingerprint"][:24]
        resume.make_verification(dest, self.saved, self.execution)
        (dest / "generated.jsonl").unlink()
        catalog.refresh()  # Partial commit must not permanently block recovery.
        with patch.object(resume, "run_child", side_effect=self.verify_saved):
            resume.recover_saved(catalog)
        self.assertIn(self.saved["sample_id"], catalog.results)
        self.assert_original_unchanged()

    def test_completed_resume_preserves_snapshot_used_by_training(self):
        self.make_orphan()
        catalog = self.catalog()
        with patch.object(resume, "run_child", side_effect=self.verify_saved):
            resume.recover_saved(catalog)
        catalog.publish("complete")
        names = ("summary.json", "results.jsonl", "generated.jsonl", "training_manifest.jsonl")
        before = {n: (self.directory / n).read_bytes() for n in names}
        with patch.object(resume, "stop_qwen") as stop, patch.object(resume, "start_qwen") as start:
            resume.execute(self.directory, self.campaign)
        start.assert_not_called()
        self.assertEqual(stop.call_count, 2)
        self.assertEqual(before, {n: (self.directory / n).read_bytes() for n in names})

    def test_competing_stage_lock_is_rejected_before_stopping_qwen(self):
        with run.run_lock(self.original), patch.object(resume, "stop_qwen") as stop:
            with self.assertRaisesRegex(RuntimeError, "Another worker owns"):
                resume.execute(self.directory, self.campaign)
        stop.assert_not_called()

    def test_generation_failure_releases_qwen_and_retries_only_next_start(self):
        def fail(command, catalog, state):
            if command[3] == "verify-one":
                return self.verify_saved(command, catalog, state)
            stage = resume.config_file(Path(command[command.index("--run-dir") + 1]) / "stage_config.json")
            self.generate_stage(stage, fail=True)
            raise RuntimeError("simulated worker failure")
        with patch.object(resume, "stop_qwen") as stop, patch.object(resume, "start_qwen"), \
                patch.object(resume, "run_child", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "simulated worker failure"):
                resume.execute(self.directory, self.campaign)
        self.assertEqual(stop.call_count, 2)
        catalog = self.catalog()
        self.assertEqual(len(catalog.attempts[self.jobs[1]["sample_id"]]), 2)
        self.assertEqual(catalog.pending(), [self.jobs[1]])
        self.assertEqual(json.loads((self.directory / "summary.json").read_text())["state"], "stopped")
        self.assert_original_unchanged()

    def test_finetune_routes_resumed_campaign_to_validated_effective_results(self):
        with patch.object(resume, "load_collection", side_effect=RuntimeError("resumed loader reached")) as new, \
                patch.object(resume.all_city_campaign, "load_collection") as old:
            with self.assertRaisesRegex(RuntimeError, "resumed loader reached"):
                finetune_all_cities.prepare(self.root / "training", self.directory, 10)
        new.assert_called_once_with(self.directory)
        old.assert_not_called()


if __name__ == "__main__":
    unittest.main()
