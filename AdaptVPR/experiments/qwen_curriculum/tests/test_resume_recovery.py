"""CPU-only independent raw recovery with immutable original failure journals."""
import fcntl
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.qwen_curriculum import common, run, resume_recovery as recovery


class ResumeRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "original"
        self.directory.mkdir()
        self.service = self.root / "service"
        self.service.mkdir()
        self.output = self.root / "resume_state" / "recoveries" / "job"
        source = self.root / "source.png"
        Image.new("RGB", (20, 16), (15, 80, 45)).save(source)
        self.job = recovery._sealed({"sample_id": "qwen_" + common.fingerprint("original")[:24],
            "source_path": str(source), "source_sha256": common.file_sha256(source),
            "source_dimensions": [20, 16], "condition": "night", "city": "City", "place_id": 1,
            "seed": 123, "prompt": "unchanged source-anchored night prompt", "negative_prompt": ""}, "record_sha256")
        common.write_jsonl(self.directory / "plan.jsonl", [self.job])
        plan = recovery._sealed({"num_images": 1, "output_dir": str(self.directory),
            "plan_sha256": common.file_sha256(self.directory / "plan.jsonl"),
            "domain_quotas": {"night": 1}, "seed": 42})
        common.write_json(self.directory / "plan_config.json", plan)
        self.execution = recovery._sealed({"stage": "generate", "run_dir": str(self.directory),
            "plan_fingerprint": plan["fingerprint"], "max_calls": 1,
            "implementation_sha256": {str(Path(run.__file__).relative_to(common.ADAPTVPR_ROOT)):
                common.file_sha256(Path(run.__file__))},
            "quality": {"original_quality_definition": "unaltered"}, "matcher_device": "cuda", "cpu_threads": 4,
            "service": {"model_id": "Qwen/Qwen-Image-Edit-2511", "source_modified": False,
                "canvas_policy": "source_aspect_v1", "sampling": {"infer_steps": 4, "guidance_scale": 1.0},
                "canvas": {"target_shape_order": "height,width", "target_pixels": 720, "multiple": 1,
                    "min_side": 1, "max_side": 128, "rounding": "nearest_multiple",
                    "unsupported_aspect_ratio": "reject"}}})
        common.write_json(self.directory / "execution_config.json", self.execution)
        self.attempt = run.sealed_attempt({"call_number": 1, "sample_id": self.job["sample_id"],
            "record_sha256": self.job["record_sha256"], "execution_fingerprint": self.execution["fingerprint"]})
        common.write_jsonl(self.directory / "attempts.jsonl", [self.attempt])
        self.error = run.sealed({**self.job, "execution_fingerprint": self.execution["fingerprint"],
            "status": "generation_error", "passed": False, "eligible_for_training": False,
            "error": "Interrupted HTTP request; original failure must remain immutable"})
        common.write_jsonl(self.directory / "results.jsonl", [self.error])
        common.write_jsonl(self.directory / "generated.jsonl", [])
        self.raw, self.metadata = self.make_output("orphan")
        self.original_files = {str(path): path.read_bytes() for path in self.directory.rglob("*") if path.is_file()}
        self.service_files = {str(path): path.read_bytes() for path in self.service.iterdir() if path.is_file()}

    def make_output(self, name):
        raw = self.service / f"{name}.png"
        Image.new("RGB", (30, 24), (80, 90, 100)).save(raw)
        metadata = raw.with_suffix(".json")
        common.write_json(metadata, {"source_path": str(self.directory / "source_inputs" / f"{self.job['sample_id']}.png"),
            "seed": self.job["seed"], "result_path": str(raw), "metadata_path": str(metadata),
            "canvas_policy": "source_aspect_v1", "source_dimensions": [20, 16],
            "raw_dimensions": [30, 24], "target_shape": [24, 30]})
        return raw, metadata

    def recover(self):
        return recovery.recover(self.directory, self.job, self.execution, self.attempt, self.service, self.output)

    def assert_original_unchanged(self):
        for path, expected in self.original_files.items():
            self.assertEqual(Path(path).read_bytes(), expected, path)
        for path, expected in self.service_files.items():
            self.assertEqual(Path(path).read_bytes(), expected, path)

    def test_recorded_failure_is_recovered_without_http_or_old_journal_changes(self):
        with patch("requests.Session") as http:
            row = self.recover()
        http.assert_not_called()
        self.assert_original_unchanged()
        self.assertEqual(Path(row["raw_output_path"]).read_bytes(), self.raw.read_bytes())
        self.assertNotEqual(row["execution_fingerprint"], self.execution["fingerprint"])
        self.assertNotIn("passed", row)
        self.assertNotIn("sampling", row)
        self.assertEqual(row["recovery"]["http_calls"], 0)
        self.assertEqual(row["recovery"]["sidecar_does_not_prove"], ["prompt", "service_identity"])
        recovered, execution = recovery.validate_recovery(self.output, self.job, self.execution, self.attempt)
        self.assertEqual(row, recovered)
        self.assertEqual(execution["quality"], self.execution["quality"])
        self.assertEqual(execution["stage"], "saved_raw_recovery")
        self.assertEqual(execution["max_calls"], 1)
        self.assertFalse((self.output / "attempts.jsonl").exists())
        self.assertEqual(execution["recovery_provenance"]["original_error_result_sha256"], self.error["result_sha256"])
        with Image.open(self.raw) as raw, Image.open(row["output_path"]) as normalized:
            self.assertEqual(normalized.size, (20, 16))
            self.assertEqual(normalized.tobytes(), raw.convert("RGB").resize((20, 16), Image.Resampling.LANCZOS).tobytes())

    def test_unknown_reserved_attempt_is_recovered_and_later_failure_does_not_change_seals(self):
        common.write_jsonl(self.directory / "results.jsonl", [])
        row = self.recover()
        before = (self.output / "execution_config.json").read_bytes()
        common.write_jsonl(self.directory / "results.jsonl", [self.error])
        self.assertEqual(self.recover(), row)
        self.assertEqual((self.output / "execution_config.json").read_bytes(), before)

    def test_raw_moved_but_sidecar_still_in_service_is_recovered(self):
        local = self.directory / "raw" / f"{self.job['sample_id']}.png"
        local.parent.mkdir()
        self.raw.replace(local)
        row = self.recover()
        self.assertEqual(Path(row["raw_output_path"]).read_bytes(), local.read_bytes())
        intent = json.loads((self.output / "intent.json").read_text())
        self.assertEqual(intent["source_raw_path"], str(local))
        self.assertEqual(intent["source_metadata_path"], str(self.metadata))

    def test_raw_and_sidecar_already_moved_are_recovered(self):
        local = self.directory / "raw" / f"{self.job['sample_id']}.png"
        local.parent.mkdir()
        self.raw.replace(local)
        self.metadata.replace(local.with_suffix(".json"))
        row = self.recover()
        self.assertEqual(Path(row["raw_output_path"]).read_bytes(), local.read_bytes())

    def test_incomplete_json_is_skipped_with_audited_warning(self):
        broken = self.service / "unfinished.json"
        broken.write_text('{"source_path":')
        with self.assertWarnsRegex(RuntimeWarning, "incomplete/unreadable sidecar"):
            row = self.recover()
        self.assertTrue(row)
        warnings = json.loads((self.output / "audit.json").read_text())["warnings"]
        self.assertEqual(warnings[0]["path"], str(broken))
        self.assertEqual(row["recovery"]["warnings"], warnings)

    def test_two_matching_outputs_fail_closed_without_writing_intent(self):
        self.make_output("second")
        with self.assertRaisesRegex(ValueError, "Multiple matching"):
            self.recover()
        self.assertFalse((self.output / "intent.json").exists())
        self.assertFalse((self.output / "generated.jsonl").exists())
        self.assert_original_unchanged()

    def test_invalid_png_returns_none_and_allows_independent_retry(self):
        self.raw.write_bytes(b"interrupted PNG")
        with self.assertWarnsRegex(RuntimeWarning, "invalid raw PNG"):
            self.assertIsNone(self.recover())
        self.assertFalse((self.output / "intent.json").exists())
        self.assertFalse(json.loads((self.output / "audit.json").read_text())["recoverable"])

    def test_missing_raw_returns_none_without_new_call(self):
        self.raw.unlink()
        with self.assertWarnsRegex(RuntimeWarning, "no saved raw PNG"):
            self.assertIsNone(self.recover())
        self.assertFalse((self.output / "execution_config.json").exists())

    def test_wrong_seed_is_not_recovered(self):
        metadata = json.loads(self.metadata.read_text())
        metadata["seed"] += 1
        common.write_json(self.metadata, metadata)
        self.assertIsNone(self.recover())

    def test_matching_sidecar_with_wrong_canvas_refuses_recovery(self):
        metadata = json.loads(self.metadata.read_text())
        metadata["raw_dimensions"] = [31, 24]
        common.write_json(self.metadata, metadata)
        with self.assertRaisesRegex(ValueError, "dimensions or canvas"):
            self.recover()

    def test_recovery_is_idempotent_and_checks_existing_image_hashes(self):
        row = self.recover()
        before = {str(path): path.read_bytes() for path in self.output.rglob("*") if path.is_file()}
        self.assertEqual(self.recover(), row)
        self.assertEqual(before, {str(path): path.read_bytes() for path in self.output.rglob("*") if path.is_file()})
        Path(row["output_path"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            self.recover()

    def test_crash_before_generated_commit_resumes_from_intent_and_copied_bytes(self):
        original = common.write_jsonl
        def fail_generated(path, rows):
            if Path(path).name == "generated.jsonl":
                raise OSError("simulated power interruption before journal commit")
            return original(path, rows)
        with patch.object(common, "write_jsonl", side_effect=fail_generated):
            with self.assertRaisesRegex(OSError, "power interruption"):
                self.recover()
        self.assertTrue((self.output / "intent.json").exists())
        self.assertFalse((self.output / "generated.jsonl").exists())
        # Copies are independently owned: recovery remains possible if service
        # originals disappear after copying but before generated journal commit.
        self.raw.unlink()
        self.metadata.unlink()
        row = self.recover()
        self.assertEqual(recovery.validate_recovery(self.output)[0], row)
        self.assertEqual(len(common.read_jsonl(self.output / "generated.jsonl")), 1)

    def test_crash_after_first_copy_does_not_overwrite_or_duplicate_raw(self):
        original = recovery._copy_atomic
        calls = []
        def fail_metadata(source, destination, digest):
            calls.append(destination)
            if destination.suffix == ".json":
                raise OSError("interrupted metadata copy")
            return original(source, destination, digest)
        with patch.object(recovery, "_copy_atomic", side_effect=fail_metadata):
            with self.assertRaisesRegex(OSError, "metadata copy"):
                self.recover()
        copied_raw = self.output / "raw" / f"{self.job['sample_id']}.png"
        raw_bytes = copied_raw.read_bytes()
        self.assertEqual(Path(self.recover()["raw_output_path"]).read_bytes(), raw_bytes)

    def test_no_fabricated_attempts_and_quality_result_is_validated(self):
        row = self.recover()
        execution = json.loads((self.output / "execution_config.json").read_text())
        class Verifier:
            def evaluate(self, source, generated, condition):
                return {"status": "passed", "passed": True, "eligible_for_training": True}
        accepted = run.verify(row, Verifier(), execution)
        common.write_jsonl(self.output / "results.jsonl", [accepted])
        self.assertEqual(recovery.validate_recovery(self.output)[0], row)
        common.write_jsonl(self.output / "attempts.jsonl", [self.attempt])
        with self.assertRaisesRegex(ValueError, "fabricated POST"):
            recovery.validate_recovery(self.output)

    def test_parent_worker_lock_prevents_recovery(self):
        with (self.directory / ".run.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "Another worker"):
                self.recover()
        self.assertFalse((self.output / "generated.jsonl").exists())

    def test_controller_holding_parent_lock_can_recover_without_reacquiring_it(self):
        with run.run_lock(self.directory):
            row = recovery.recover(self.directory, self.job, self.execution, self.attempt,
                                   self.service, self.output, parent_locked=True)
        self.assertEqual(recovery.validate_recovery(self.output)[0], row)

    def test_changed_parent_attempt_is_rejected(self):
        attempts = common.read_jsonl(self.directory / "attempts.jsonl")
        attempts[0]["call_number"] = 99
        common.write_jsonl(self.directory / "attempts.jsonl", attempts)
        with self.assertRaisesRegex(ValueError, "durable call ledger"):
            self.recover()

    def test_changed_original_source_is_rejected_before_recovery(self):
        Image.new("RGB", (20, 16), (120, 10, 5)).save(self.job["source_path"])
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            self.recover()

    def test_recorded_error_reference_cannot_be_removed_after_recovery(self):
        self.recover()
        common.write_jsonl(self.directory / "results.jsonl", [])
        with self.assertRaisesRegex(ValueError, "recorded generation failure changed"):
            recovery.validate_recovery(self.output)

    def test_matching_sidecar_cannot_copy_an_unowned_raw(self):
        metadata = json.loads(self.metadata.read_text())
        metadata["result_path"] = self.job["source_path"]
        common.write_json(self.metadata, metadata)
        with self.assertRaisesRegex(ValueError, "owned paths"):
            self.recover()


if __name__ == "__main__":
    unittest.main()
