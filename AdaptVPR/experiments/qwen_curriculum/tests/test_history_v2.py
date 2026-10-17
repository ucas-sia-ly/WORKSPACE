from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.qwen_curriculum import common, history_v2 as history


class HistoryAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / "stage"
        self.directory.mkdir()
        self.source = self.root / "source.png"
        Image.new("RGB", (8, 6), (17, 22, 33)).save(self.source)
        self.job = history.seal({"sample_id": "qwen_" + "a" * 24,
            "source_path": str(self.source), "source_sha256": common.file_sha256(self.source),
            "source_dimensions": [8, 6], "condition": "night", "city": "Test", "place_id": 4,
            "prompt": "Conservative night edit", "negative_prompt": "", "seed": 42}, "record_sha256")
        common.write_jsonl(self.directory / "plan.jsonl", [self.job])
        self.plan = history.seal({"num_images": 1,
            "plan_sha256": common.file_sha256(self.directory / "plan.jsonl")})
        common.write_json(self.directory / "plan_config.json", self.plan)
        self.execution = history.seal({"run_dir": str(self.directory), "stage": "generate",
            "max_calls": 1, "plan_fingerprint": self.plan["fingerprint"],
            "quality": {"version": 1, "thresholds": {"min_matches": 64}},
            "implementation_sha256": {}, "service": {"model_id": "Qwen/Qwen-Image-Edit-2511",
                "source_modified": False, "canvas_policy": "source_aspect_v1",
                "sampling": {"infer_steps": 4, "guidance_scale": 1.0}}})
        common.write_json(self.directory / "execution_config.json", self.execution)
        self.attempt = history.seal({"sample_id": self.job["sample_id"], "call_number": 1,
            "record_sha256": self.job["record_sha256"], "execution_fingerprint": self.execution["fingerprint"]}, "attempt_sha256")
        common.write_jsonl(self.directory / "attempts.jsonl", [self.attempt])
        raw, output = self.directory / "raw.png", self.directory / "output.png"
        Image.new("RGB", (16, 12), (1, 2, 3)).save(raw)
        Image.new("RGB", (8, 6), (4, 5, 6)).save(output)
        self.generated = history.seal({**self.job, "execution_fingerprint": self.execution["fingerprint"],
            "output_path": str(output), "output_sha256": common.file_sha256(output),
            "raw_output_path": str(raw), "raw_output_sha256": common.file_sha256(raw),
            "sampling": {"raw_dimensions": [16, 12]}}, "result_sha256")
        self.result = history.seal({**self.generated, "status": "passed", "passed": True,
            "eligible_for_training": True, "weather_ok": True}, "result_sha256")
        common.write_jsonl(self.directory / "generated.jsonl", [self.generated])
        common.write_jsonl(self.directory / "results.jsonl", [self.result])

    def auditor(self):
        return history.Auditor(self.root)

    def test_original_sealed_stage_is_audited_without_current_quality_substitution(self):
        info = self.auditor().audit_stage(self.directory, complete=True)
        self.assertEqual(info["generated"], [self.generated])
        self.assertEqual(info["execution"]["quality"]["version"], 1)
        self.assertEqual(info["execution"]["quality"]["thresholds"]["min_matches"], 64)
        # The returned record remains signed by its original execution.
        self.assertEqual(info["results"][0]["execution_fingerprint"], self.execution["fingerprint"])

    def test_same_path_archived_python_is_proved_once_and_never_executed(self):
        source = self.root / "AdaptVPR/old_verifier.py"
        source.parent.mkdir()
        source.write_text('raise RuntimeError("old source must never execute")\n')
        old_hash = common.file_sha256(source)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        subprocess.run(["git", "-C", str(self.root), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.root), "-c", "user.email=test@example.invalid",
            "-c", "user.name=Test", "commit", "-qm", "archived implementation"], check=True)
        source.write_text("# current v2 implementation\n")
        audit = self.auditor()
        proof = audit.audit_file(source, old_hash)
        self.assertEqual(proof["proof"], "git_archived_source_bytes")
        self.assertEqual(proof["expected_sha256"], old_hash)
        self.assertEqual(proof["git_path"], "AdaptVPR/old_verifier.py")
        with patch.object(history.subprocess, "run") as git:
            self.assertEqual(audit.audit_file(source, old_hash), proof)
            git.assert_not_called()

    def test_csv_and_weights_cannot_be_rescued_by_git_archive(self):
        for name in ("labels.csv", "checkpoint.ckpt", "raw.png", "results.jsonl"):
            path = self.root / name
            path.write_bytes(b"changed data")
            with patch.object(history.subprocess, "run") as git:
                with self.assertRaisesRegex(ValueError, "data missing or changed"):
                    self.auditor().audit_file(path, "0" * 64)
                git.assert_not_called()

    def test_unsigned_changes_and_resigned_wrong_jobs_are_rejected(self):
        wrong = {**self.result, "prompt": "Different prompt"}
        common.write_jsonl(self.directory / "results.jsonl", [wrong])
        with self.assertRaisesRegex(ValueError, "seal"):
            self.auditor().audit_stage(self.directory)
        common.write_jsonl(self.directory / "results.jsonl", [history.seal(wrong, "result_sha256")])
        with self.assertRaisesRegex(ValueError, "execution/job"):
            self.auditor().audit_stage(self.directory)

    def test_image_mutation_is_rejected_even_when_journal_seals_are_intact(self):
        Path(self.generated["raw_output_path"]).write_bytes(b"corrupt raw")
        with self.assertRaisesRegex(ValueError, "image hash changed"):
            self.auditor().audit_stage(self.directory)

    def test_resigned_wrong_or_missing_reservation_is_rejected(self):
        wrong = history.seal({**self.attempt, "call_number": 2}, "attempt_sha256")
        common.write_jsonl(self.directory / "attempts.jsonl", [wrong])
        with self.assertRaisesRegex(ValueError, "reservation"):
            self.auditor().audit_stage(self.directory)
        common.write_jsonl(self.directory / "attempts.jsonl", [])
        with self.assertRaisesRegex(ValueError, "no durable reservation"):
            self.auditor().audit_stage(self.directory)

    def test_verification_cannot_replace_saved_raw_and_stage_budget_is_enforced(self):
        other = self.root / "other.png"
        Image.new("RGB", (8, 6), (77, 88, 99)).save(other)
        wrong = history.seal({**self.result, "output_path": str(other),
            "output_sha256": common.file_sha256(other)}, "result_sha256")
        common.write_jsonl(self.directory / "results.jsonl", [wrong])
        with self.assertRaisesRegex(ValueError, "verification changed generated"):
            self.auditor().audit_stage(self.directory)
        execution = history.seal({**self.execution, "max_calls": 1001})
        common.write_json(self.directory / "execution_config.json", execution)
        with self.assertRaisesRegex(ValueError, "plan/budget"):
            self.auditor().audit_stage(self.directory)

    def test_unfinished_failed_reservation_is_preserved_as_history(self):
        error = history.seal({**self.job, "execution_fingerprint": self.execution["fingerprint"],
            "status": "generation_error", "passed": False, "eligible_for_training": False,
            "error": "interrupted"}, "result_sha256")
        common.write_jsonl(self.directory / "generated.jsonl", [])
        common.write_jsonl(self.directory / "results.jsonl", [error])
        info = self.auditor().audit_stage(self.directory)
        self.assertEqual(info["attempts"], [self.attempt])
        self.assertEqual(info["generated"], [])
        self.assertEqual(info["results"], [error])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.auditor().audit_stage(self.directory, complete=True)


if __name__ == "__main__":
    unittest.main()
