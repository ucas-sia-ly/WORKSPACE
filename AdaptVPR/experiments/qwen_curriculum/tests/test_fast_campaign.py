import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.qwen_curriculum import common, run, fast_campaign as fast


class FastCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.parent, self.recovery, self.new = [self.root / name for name in ("old", "recovered", "new")]
        for path in (self.parent, self.recovery, self.new):
            path.mkdir()
        self.jobs = []
        for index in range(4):
            path = self.root / f"source_{index}.png"
            Image.new("RGB", (4, 3), (index, 20, 30)).save(path)
            body = {"sample_id": "qwen_" + f"{index:024x}", "condition": "night",
                    "source_path": str(path), "source_sha256": common.file_sha256(path),
                    "source_dimensions": [4, 3], "seed": index, "prompt": "edit", "city": "City", "place_id": index}
            self.jobs.append({**body, "record_sha256": common.fingerprint(body)})
        self.old_plan = self.plan(self.parent, self.jobs[:3])
        self.new_plan = self.plan(self.new, self.jobs[3:])
        self.old_execution = fast.frozen(self.parent / "execution_config.json", {
            "plan_fingerprint": self.old_plan["fingerprint"], "max_calls": 3})
        self.recovery_execution = fast.frozen(self.recovery / "execution_config.json", {"max_calls": 0})
        self.new_execution = fast.frozen(self.new / "execution_config.json", {
            "plan_fingerprint": self.new_plan["fingerprint"], "max_calls": 1})
        self.old_row = self.row(self.jobs[0], self.old_execution)
        errors = [run.sealed({**job, "execution_fingerprint": self.old_execution["fingerprint"],
                             "status": "generation_error", "error": "unknown", "passed": False}) for job in self.jobs[1:3]]
        common.write_jsonl(self.parent / "results.jsonl", [self.old_row, *errors])
        common.write_jsonl(self.parent / "generated.jsonl", [self.old_row])
        self.attempts(self.parent, self.jobs[:3], self.old_execution)
        recovered = [self.row(job, self.recovery_execution) for job in self.jobs[1:3]]
        for name in ("results.jsonl", "generated.jsonl"):
            common.write_jsonl(self.recovery / name, recovered)
            common.write_jsonl(self.new / name, [self.row(self.jobs[3], self.new_execution, "rejected")])
        self.attempts(self.new, self.jobs[3:], self.new_execution)
        common.write_jsonl(self.recovery / "recovery_inputs.jsonl", [])
        common.write_jsonl(self.root / "combined_plan.jsonl", self.jobs)
        self.config = fast.frozen(self.root / "fast_campaign_config.json", {
            "total_images": 4, "parent_run_dir": str(self.parent), "recovery_run_dir": str(self.recovery),
            "generation_run_dir": str(self.new), "combined_plan": str(self.root / "combined_plan.jsonl"),
            "combined_plan_sha256": common.file_sha256(self.root / "combined_plan.jsonl"),
            "recovery_ids": [row["sample_id"] for row in self.jobs[1:3]], "runtime_profile": "resident_bf16_48g",
            "parent_files_sha256": {name: common.file_sha256(self.parent / name) for name in
                ("plan_config.json", "execution_config.json", "results.jsonl", "generated.jsonl", "attempts.jsonl")},
            "recovery_inputs_sha256": common.file_sha256(self.recovery / "recovery_inputs.jsonl"),
            "new_plan_fingerprint": self.new_plan["fingerprint"],
            "recovery_execution_fingerprint": self.recovery_execution["fingerprint"], "implementation_sha256": {}})

    def plan(self, directory, jobs):
        common.write_jsonl(directory / "plan.jsonl", jobs)
        return fast.frozen(directory / "plan_config.json", {
            "num_images": len(jobs), "plan_sha256": common.file_sha256(directory / "plan.jsonl")})

    def row(self, job, execution, status="passed"):
        return run.sealed({**job, "execution_fingerprint": execution["fingerprint"],
            "output_path": job["source_path"], "output_sha256": job["source_sha256"],
            "raw_output_path": job["source_path"], "raw_output_sha256": job["source_sha256"],
            "status": status, "passed": status == "passed", "eligible_for_training": status == "passed"})

    def attempts(self, directory, jobs, execution):
        common.write_jsonl(directory / "attempts.jsonl", [run.sealed_attempt({
            "sample_id": job["sample_id"], "record_sha256": job["record_sha256"],
            "execution_fingerprint": execution["fingerprint"], "call_number": index})
            for index, job in enumerate(jobs, 1)])

    def test_collect_saved_recoveries_without_changing_old_error_journal(self):
        before = (self.parent / "results.jsonl").read_bytes()
        rows = fast.collected(self.config, "results.jsonl")
        self.assertEqual([r["sample_id"] for r in rows], [r["sample_id"] for r in self.jobs])
        self.assertEqual(rows[0], self.old_row)
        self.assertEqual((self.parent / "results.jsonl").read_bytes(), before)
        summary = fast.publish(self.root, self.config, "generating")
        self.assertEqual((summary["generated_images"], summary["quality_evaluated"], summary["generation_errors"]), (4, 4, 0))
        self.assertEqual(len(fast.read(self.root, "training_manifest.jsonl")), 3)

    def test_trainer_loads_all_stage_seals_and_quality_rejects(self):
        fast.publish(self.root, self.config, "complete")
        rows, execution, plan = fast.load_collection(self.root, 4)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[-1]["status"], "rejected")
        self.assertEqual(execution, self.config["fingerprint"])
        self.assertEqual(plan, self.config["combined_plan_sha256"])

    def test_incomplete_campaign_cannot_enter_training(self):
        fast.publish(self.root, self.config, "generating")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            fast.load_collection(self.root, 4)
        # The coordinator can verify all bytes before publishing complete.
        self.assertEqual(len(fast.load_collection(self.root, 4, require_complete=False)[0]), 4)

    def test_tampered_old_record_or_new_seal_is_rejected(self):
        fast.publish(self.root, self.config, "complete")
        row = fast.read(self.new, "results.jsonl")[0]
        common.write_jsonl(self.new / "results.jsonl", [{**row, "passed": True}])
        with self.assertRaisesRegex(ValueError, "checksum"):
            fast.load_collection(self.root, 4)
        with (self.parent / "results.jsonl").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "Original stage changed"):
            fast.check_inputs(self.config)

    def test_changed_image_or_combined_manifest_cannot_enter_training(self):
        fast.publish(self.root, self.config, "complete")
        Path(self.jobs[3]["source_path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            fast.load_collection(self.root, 4)

    def test_source_or_task_duplicates_are_refused(self):
        common.write_jsonl(self.new / "results.jsonl", [self.old_row])
        with self.assertRaisesRegex(ValueError, "Duplicate successful"):
            fast.collected(self.config, "results.jsonl")
        common.write_jsonl(self.new / "results.jsonl", [{**self.row(self.jobs[3], self.new_execution),
            "source_sha256": self.jobs[0]["source_sha256"]}])
        with self.assertRaisesRegex(ValueError, "Duplicate source content"):
            fast.collected(self.config, "results.jsonl")

    def test_training_checks_combined_records_against_stage_journals(self):
        fast.publish(self.root, self.config, "complete")
        common.write_jsonl(self.root / "results.jsonl", [])
        with self.assertRaisesRegex(ValueError, "Combined results differ"):
            fast.load_collection(self.root, 4)

    def test_missing_image_does_not_become_complete_and_target_cannot_change(self):
        common.write_jsonl(self.new / "results.jsonl", [])
        fast.publish(self.root, self.config, "complete")
        with self.assertRaisesRegex(ValueError, "lacks an evaluated image"):
            fast.load_collection(self.root, 4)

    def test_active_original_training_never_starts_generation(self):
        with patch.object(fast, "unit_state", return_value="active"), \
             patch.object(fast, "recover_saved") as recovery, patch.object(fast.subprocess, "Popen") as child:
            with self.assertRaisesRegex(RuntimeError, "still active"):
                fast.execute(self.root, self.config)
            recovery.assert_not_called()
            child.assert_not_called()

    def test_saved_unknown_call_recovery_is_zero_http_and_idempotent(self):
        raw = self.root / "saved_service.png"
        Image.new("RGB", (16, 12), (120, 40, 60)).save(raw)
        side = raw.with_suffix(".json")
        side.write_text(json.dumps({"source": "original frozen worker input"}))
        job = self.jobs[1]
        intent = {"source_raw_path": str(raw), "source_metadata_path": str(side),
                  "raw_sha256": common.file_sha256(raw), "metadata_sha256": common.file_sha256(side),
                  "execution_fingerprint": self.old_execution["fingerprint"], "intent_sha256": "audit-proof"}
        common.write_jsonl(self.recovery / "recovery_inputs.jsonl", [{
            "job": job, "intent": intent, "original_error_result_sha256": "original-error-proof"}])
        common.write_jsonl(self.recovery / "results.jsonl", [])
        common.write_jsonl(self.recovery / "generated.jsonl", [])
        original = (self.parent / "results.jsonl").read_bytes()
        class Verifier:
            def evaluate(self, source, generated, condition):
                return {"status": "passed", "passed": True, "eligible_for_training": True}
        with patch("experiments.qwen_curriculum.quality.QwenQualityVerifier", return_value=Verifier()), \
             patch.object(fast.requests, "Session") as http:
            rows = fast.recover_saved(self.config)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["recovery"]["http_calls"], 0)
            first = (self.recovery / "results.jsonl").read_bytes()
            self.assertEqual(fast.recover_saved(self.config), rows)
            self.assertEqual((self.recovery / "results.jsonl").read_bytes(), first)
            http.assert_not_called()
        self.assertEqual((self.parent / "results.jsonl").read_bytes(), original)
        with Image.open(rows[0]["output_path"]) as image:
            self.assertEqual(image.size, (4, 3))
        self.assertFalse(raw.exists())
        self.assertTrue(Path(rows[0]["raw_output_path"]).exists())


if __name__ == "__main__":
    unittest.main()
