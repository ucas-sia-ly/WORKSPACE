"""Independent Stage3 input contract tests. No services or model dependencies."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

from generation.targeted_inputs import TargetedInputError, read_targeted_tasks, run_targeted

ROOT = Path(__file__).resolve().parents[1]


class TargetedInputsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.artifact = self.root / "artifact"
        (self.artifact / "masks").mkdir(parents=True)
        (self.root / "source/City").mkdir(parents=True)
        self.manifest = self.artifact / "targets.jsonl"
        self.output = self.root / "output"
        self.rows = []
        for i, target in enumerate(("attention", "fused")):
            source = self.root / f"source/City/{i}.png"
            Image.new("RGB", (37, 23), (i * 70, 40, 90)).save(source)
            mask = Image.new("L", (37, 23), 0)
            mask.paste(1, (5, 3, 11, 9))
            mask.save(self.artifact / f"masks/{i}.png")
            self.rows.append(dict(
                schema_version=1, sample_id=f"sample_{i}", image_key=f"City/{i}.png", place_key=f"City:{i}",
                source_path=str(source), source_width=37, source_height=23, mask_original_path=f"masks/{i}.png",
                target_type=target, target_role="primary" if i == 0 else "supplementary", mask_ratio=.15,
                clean_margin=.25 - i / 2, mask_mode="connected_topk", mask_token_count=38,
                checkpoint_path="/stage2/model.ckpt", checkpoint_sha256="a" * 64,
                stage2_commit="b" * 40, bag_of_queries_head="c" * 40, seed=7,
                source_role="SOURCE", stage2_query_index=i, stage2_mask_key=f"query_{i:03d}_{target}",
            ))
        self.write_rows()

    def write_rows(self):
        self.manifest.write_text("".join(json.dumps(r) + "\n" for r in self.rows), encoding="utf-8")

    def assert_invalid(self, message):
        with self.assertRaisesRegex(TargetedInputError, message):
            run_targeted(self.manifest, self.output)
        self.assertFalse(self.output.exists())

    def test_normalized_local_task_and_preserved_stage2_metadata(self):
        original = copy.deepcopy(self.rows)
        tasks = read_targeted_tasks(self.manifest, seed=42)
        self.assertEqual(self.rows, original)
        for task, raw in zip(tasks, self.rows):
            self.assertEqual(task["route"], "local")
            self.assertEqual(task["targeting"], "provided_mask")
            self.assertTrue(task["dry_run"])
            self.assertFalse(task["generated"])
            self.assertEqual(task["seed"], 42)
            self.assertEqual(task["stage2_metadata"]["seed"], 7)
            for field in ("stage2_commit", "checkpoint_sha256", "checkpoint_path", "mask_mode", "stage2_mask_key"):
                self.assertEqual(task["stage2_metadata"][field], raw[field])
            for field in ("sample_id", "image_key", "place_key", "target_type", "target_role", "clean_margin", "mask_ratio"):
                self.assertEqual(task[field], raw[field])
            self.assertEqual(task["actual_pixel_area_original"], 36)
            self.assertTrue(Path(task["mask_original_path"]).is_absolute())
            self.assertNotIn("prompt", task)
            self.assertNotIn("position", task)

    def test_relative_source_and_one_bit_mask(self):
        self.rows[0]["source_path"] = "../source/City/0.png"
        self.write_rows()
        mask = Image.new("1", (37, 23), 0)
        mask.paste(1, (5, 3, 11, 9))
        mask.save(self.artifact / "masks/0.png")
        task = read_targeted_tasks(self.manifest, limit=1)[0]
        self.assertEqual(task["source_path"], str(self.root / "source/City/0.png"))
        self.assertEqual(task["actual_pixel_area_original"], 36)

    def test_check_only_has_no_outputs(self):
        result = run_targeted(self.manifest, self.output, check_only=True)
        self.assertEqual(result["validated"], 2)
        self.assertEqual(result["written"], 0)
        self.assertFalse(self.output.exists())

    def test_missing_source_and_mask(self):
        for field in ("source_path", "mask_original_path"):
            with self.subTest(field=field):
                saved = self.rows[0][field]
                self.rows[0][field] = "missing.png"
                self.write_rows()
                self.assert_invalid("file does not exist")
                self.rows[0][field] = saved

    def test_mask_size_binary_empty_and_rgb_rejected(self):
        cases = [
            (Image.new("L", (37, 22), 1), "mask size"),
            (Image.new("L", (37, 23), 255), "binary"),
            (Image.new("L", (37, 23), 0), "nonempty"),
            (Image.new("RGB", (37, 23), (1, 1, 1)), "single-channel"),
        ]
        for mask, message in cases:
            with self.subTest(message=message):
                mask.save(self.artifact / "masks/0.png")
                self.assert_invalid(message)

    def test_corrupt_source_is_decoded_not_just_stat_checked(self):
        Path(self.rows[0]["source_path"]).write_bytes(b"not an image")
        self.assert_invalid("sample sample_0")

    def test_duplicate_id_beyond_limit_is_rejected(self):
        self.rows[1]["sample_id"] = self.rows[0]["sample_id"]
        self.write_rows()
        with self.assertRaisesRegex(TargetedInputError, "duplicate sample_id"):
            read_targeted_tasks(self.manifest, limit=1)

    def test_invalid_schema_type_or_missing_metadata(self):
        for key, value in (("schema_version", 2), ("schema_version", True), ("target_type", "random"),
                           ("target_type", []), ("mask_ratio", 0), ("clean_margin", float("nan")),
                           ("seed", -1), ("target_role", "supplementary")):
            with self.subTest(key=key, value=value):
                saved = self.rows[0][key]
                self.rows[0][key] = value
                self.write_rows()
                self.assert_invalid(".")
                self.rows[0][key] = saved
        del self.rows[0]["stage2_commit"]
        self.write_rows()
        self.assert_invalid("missing required fields: stage2_commit")

    def test_bad_schema_beyond_limit_rejected(self):
        self.rows[1]["schema_version"] = 999
        self.write_rows()
        with self.assertRaisesRegex(TargetedInputError, "schema_version"):
            read_targeted_tasks(self.manifest, limit=1)

    def test_wrong_source_or_declared_size_rejected(self):
        saved = self.rows[0]["source_path"]
        self.rows[0]["source_path"] = self.rows[1]["source_path"]
        self.write_rows()
        self.assert_invalid("source_path does not match image_key")
        self.rows[0]["source_path"] = saved
        self.rows[0]["source_width"] = 999
        self.write_rows()
        self.assert_invalid("source_width mismatch")

    def test_optional_hashes_checked(self):
        for key in ("source_sha256", "mask_original_sha256"):
            with self.subTest(key=key):
                self.rows[0][key] = "0" * 64
                self.write_rows()
                self.assert_invalid(f"{key} mismatch")
                del self.rows[0][key]

    def test_companion_manifest_schema_and_integrity(self):
        companion = self.artifact / "export_manifest.json"
        good = dict(schema_version=1, record_count=2,
                    targets_sha256=hashlib.sha256(self.manifest.read_bytes()).hexdigest())
        companion.write_text(json.dumps(good))
        self.assertEqual(len(read_targeted_tasks(self.manifest)), 2)
        for field, value in (("schema_version", 2), ("record_count", 3), ("targets_sha256", "bad")):
            companion.write_text(json.dumps({**good, field: value}))
            self.assert_invalid("manifest")

    def test_limit_and_byte_identical_seed(self):
        first = run_targeted(self.manifest, self.output, limit=1, seed=12)
        self.assertEqual(first["written"], 1)
        contents = (self.output / "tasks.jsonl").read_bytes()
        other = self.root / "other"
        run_targeted(self.manifest, other, limit=1, seed=12)
        self.assertEqual(contents, (other / "tasks.jsonl").read_bytes())
        self.assertEqual(list(self.output.iterdir()), [self.output / "tasks.jsonl"])

    def test_resume_expands_prefix_without_duplicates(self):
        run_targeted(self.manifest, self.output, limit=1)
        result = run_targeted(self.manifest, self.output, resume=True)
        self.assertEqual((result["resumed"], result["written"]), (1, 1))
        before = (self.output / "tasks.jsonl").read_bytes()
        result = run_targeted(self.manifest, self.output, resume=True)
        self.assertEqual((result["resumed"], result["written"]), (2, 0))
        self.assertEqual(before, (self.output / "tasks.jsonl").read_bytes())

    def test_resume_rejects_changed_seed_input_and_task(self):
        run_targeted(self.manifest, self.output)
        path = self.output / "tasks.jsonl"
        before = path.read_bytes()
        with self.assertRaisesRegex(TargetedInputError, "identical prefix"):
            run_targeted(self.manifest, self.output, resume=True, seed=99)
        self.assertEqual(before, path.read_bytes())
        self.rows[0]["clean_margin"] += .1
        self.write_rows()
        with self.assertRaisesRegex(TargetedInputError, "identical prefix"):
            run_targeted(self.manifest, self.output, resume=True)
        self.assertEqual(before, path.read_bytes())
        self.rows[0]["clean_margin"] -= .1
        self.write_rows()
        path.write_bytes(before.replace(b'"route":"local"', b'"route":"global"'))
        with self.assertRaisesRegex(TargetedInputError, "identical prefix"):
            run_targeted(self.manifest, self.output, resume=True)

    def test_resume_revalidates_missing_inputs(self):
        run_targeted(self.manifest, self.output)
        (self.artifact / "masks/0.png").unlink()
        with self.assertRaisesRegex(TargetedInputError, "mask file does not exist"):
            run_targeted(self.manifest, self.output, resume=True)

    def test_existing_output_not_overwritten_and_check_only_is_read_only(self):
        run_targeted(self.manifest, self.output)
        before = (self.output / "tasks.jsonl").read_bytes()
        with self.assertRaisesRegex(TargetedInputError, "already exists"):
            run_targeted(self.manifest, self.output)
        run_targeted(self.manifest, self.output, check_only=True, seed=55)
        self.assertEqual(before, (self.output / "tasks.jsonl").read_bytes())

    def test_late_failure_publishes_nothing(self):
        (self.artifact / "masks/1.png").unlink()
        self.assert_invalid("mask file does not exist")

    def test_malformed_empty_and_duplicate_json_keys(self):
        for data in ("", "\n", "[1]\n", '{"sample_id":"x","sample_id":"y"}\n', '{"sample_id":'):
            with self.subTest(data=data):
                self.manifest.write_text(data)
                self.assert_invalid(".")

    def test_cli_modes_block_model_planner_scheduler_verifier_imports(self):
        # Run the actual entrypoint in an isolated process, rejecting heavy imports.
        guard = '''
import importlib.abc, runpy, sys
class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ('numpy', 'torch', 'transformers', 'diffusers', 'openai', 'requests', 'prompts', 'verification',
                   'generation.agent', 'generation.router', 'generation.batch', 'generation.reflection_controller',
                   'generation.llm_client', 'generation.iclight', 'generation.lightx2v', 'run')
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('Forbidden import: ' + fullname)
sys.meta_path.insert(0, BlockModels())
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
'''
        for check_only in (True, False):
            with self.subTest(check_only=check_only):
                command = [sys.executable, "-c", guard, str(ROOT / "scripts/run_targeted.py"),
                           str(self.manifest), "--output", str(self.output), "--limit", "1", "--seed", "19"]
                if check_only:
                    command.append("--check-only")
                result = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)["generated"], 0)
                if check_only:
                    self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
