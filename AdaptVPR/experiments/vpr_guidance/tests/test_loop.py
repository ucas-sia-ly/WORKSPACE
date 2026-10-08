"""Regression tests for stage recovery, matched arms, and feedback control flow."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import write_json, write_jsonl
from run_loop import Runner, _input_signature, audit_pools, loop, parse_args, salad_flags, salad_train


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.script = self.root / "stage.py"
        self.script.write_text("# test stage\n")
        self.source = self.root / "input.txt"
        self.source.write_text("input")
        self.output = self.root / "result.txt"
        self.marker = self.root / "complete.json"

    def run_stage(self, runner, **kwargs):
        return runner(self.script, "--seed", 42, complete=self.marker,
                      outputs=[self.output], inputs=[self.source], **kwargs)

    def test_completion_requires_artifacts_and_does_not_mark_failure(self):
        with patch("run_loop.subprocess.run"):
            with self.assertRaisesRegex(RuntimeError, "required artifacts"):
                self.run_stage(Runner())
        self.assertFalse(self.marker.exists())

    def test_resume_success_then_skip_validates_artifacts(self):
        def complete(*args, **kwargs):
            self.output.write_text("finished")
        with patch("run_loop.subprocess.run", side_effect=complete) as process:
            self.run_stage(Runner(), resume_arguments=["--resume", "state.pt"])
            self.assertEqual(process.call_args.args[0][-2:], ["--resume", "state.pt"])
            self.run_stage(Runner())
            self.assertEqual(process.call_count, 1)
            self.output.write_text("corrupted")
            with self.assertRaisesRegex(ValueError, "changed artifacts"):
                self.run_stage(Runner())

    def test_changed_inputs_or_request_are_rejected_before_execution(self):
        self.output.write_text("finished")
        self.run_stage(Runner(), already_finished=True)
        self.source.write_text("changed")
        with patch("run_loop.subprocess.run") as process:
            with self.assertRaisesRegex(ValueError, "changed"):
                self.run_stage(Runner())
            process.assert_not_called()

    def test_already_finished_checkpoint_can_publish_completion_marker(self):
        self.output.write_text("final checkpoint")
        with patch("run_loop.subprocess.run") as process:
            self.run_stage(Runner(), already_finished=True)
            process.assert_not_called()
        self.assertTrue(self.marker.exists())

    def test_completed_recovery_rejects_incomplete_training_state(self):
        args = self.args()
        output = self.root / 'invalid_student'
        output.mkdir()
        (output / 'checkpoint.pt').write_bytes(b'invalid fixture')
        from common import use_salad
        use_salad()
        with patch('workflow.model.read_checkpoint', return_value={'format_version': 0, 'epoch': 1}):
            with self.assertRaisesRegex(ValueError, 'full training checkpoint'):
                salad_train(Runner(), args, output, 1)
        self.assertFalse((output / 'training_complete.json').exists())

    def test_dry_run_does_not_read_missing_inputs_or_write_files(self):
        self.source.unlink()
        with patch("run_loop.subprocess.run") as process:
            self.run_stage(Runner(dry_run=True))
            process.assert_not_called()
        self.assertFalse(self.marker.exists())
        self.assertFalse(self.marker.with_name("complete_request.json").exists())

    def test_dataset_inventory_detects_added_and_changed_images(self):
        directory = self.root / 'images'
        directory.mkdir()
        first = _input_signature(directory)
        image = directory / 'one.jpg'
        image.write_bytes(b'one')
        second = _input_signature(directory)
        self.assertNotEqual(first, second)
        image.write_bytes(b'changed image')
        self.assertNotEqual(second, _input_signature(directory))

    def test_local_backbone_weights_only_apply_to_fresh_training(self):
        args = self.args('--salad-args=--backbone-weights /tmp/local.pth --batch-size 3')
        fresh = salad_flags(args, self.root / 'model', 1)
        fine_tune = salad_flags(args, self.root / 'model', 1, init=self.root / 'student.pt')
        self.assertIn('--backbone-weights', fresh)
        self.assertNotIn('--backbone-weights', fine_tune)
        self.assertIn('--batch-size', fine_tune)

    def test_runtime_cache_files_do_not_change_experimental_source_signature(self):
        directory = self.root / 'backbone'
        directory.mkdir()
        (directory / 'model.py').write_text('model')
        before = _input_signature(directory, ignore_runtime=True)
        (directory / '__pycache__').mkdir()
        (directory / '__pycache__/model.pyc').write_bytes(b'cache')
        (directory / '.git').mkdir()
        (directory / '.git/index').write_bytes(b'git cache')
        self.assertEqual(before, _input_signature(directory, ignore_runtime=True))
        (directory / 'model.py').write_text('different source')
        self.assertNotEqual(before, _input_signature(directory, ignore_runtime=True))

    def args(self, *extra):
        return parse_args(["loop", "--root", str(self.root / "experiment"), "--arm", "full",
                           "--gsv-root", str(self.root / "gsv"), "--cities", "Bangkok",
                           "--prompts", str(self.root / "prompts.jsonl"), "--rounds", "2", *extra])

    def test_dry_run_propagates_salad_batch_and_miner_to_scoring(self):
        args = self.args("--salad-args", "--batch-size 3 --images-per-place 2 --miner-margin 0.2")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            loop(args, Runner(dry_run=True))
        lines = [line for line in out.getvalue().splitlines() if "score_candidates.py" in line]
        self.assertEqual(len(lines), 2)
        self.assertIn("--train-batch-size 3 --images-per-place 2", lines[0])
        self.assertIn("--miner-margin 0.2", lines[0])
        self.assertFalse(args.root.exists())

    def test_managed_salad_flags_cannot_be_overridden_or_abbreviated(self):
        for raw in ("--epochs 99", "--epo=99", "--check-data", "--synthetic-places-only"):
            args = self.args("--salad-args=" + raw)
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, "managed"):
                salad_flags(args, self.root / "model", 1)

    def test_adaptive_loop_uses_online_outputs_and_keeps_arm_local_candidates(self):
        args = self.args('--generation-mode', 'adaptive',
                         '--salad-args=--batch-size 3 --images-per-place 2 --miner-margin 0.2',
                         '--adaptive-args=--min-candidates 2')
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            loop(args, Runner(dry_run=True))
        generated = out.getvalue()
        lines = [line for line in generated.splitlines() if 'adaptive_candidates.py' in line]
        self.assertEqual(len(lines), 2)
        self.assertIn('--train-batch-size 3 --images-per-place 2', lines[0])
        self.assertIn('--miner-margin 0.2', lines[0])
        self.assertIn('--min-candidates 2', lines[0])
        self.assertIn('full/round_0/scoring', lines[0])
        self.assertNotIn('score_candidates.py', generated)
        self.assertNotIn('generate_candidates.py', generated)
        self.assertFalse(args.root.exists())

    def test_adaptive_extras_cannot_override_managed_inputs(self):
        for flags in ('--checkpoint x', '--check=x', '--score-args=x', '--plan-only'):
            args = self.args('--generation-mode', 'adaptive', '--adaptive-args=' + flags)
            with patch('run_loop.salad_train') as train:
                with self.assertRaisesRegex(ValueError, 'managed'):
                    loop(args, Runner(dry_run=True))
                train.assert_not_called()

    def test_synthetic_fraction_that_exposes_no_synthetic_images_is_rejected_early(self):
        args = self.args("--salad-args", "--images-per-place 2 --synthetic-fraction 0.1")
        with patch("run_loop.salad_train") as train:
            with self.assertRaisesRegex(ValueError, "synthetic view"):
                loop(args, Runner(dry_run=True))
            train.assert_not_called()

    def test_no_mined_positives_skips_lora_and_keeps_released_generator(self):
        args = self.args()
        args.prompts.write_text('{}\n')
        (args.gsv_root / 'Dataframes').mkdir(parents=True)
        (args.gsv_root / 'Dataframes/Bangkok.csv').write_text('place_id\n')
        calls = []

        class FakeRunner:
            dry_run = False

            def __call__(self, script, *flags, **kwargs):
                calls.append((script.name, flags))
                if script.name == "score_candidates.py":
                    target = Path(flags[flags.index("--output-dir") + 1])
                    write_jsonl(target / "selected.jsonl", [])

        def train(*a, **kw):
            target = a[2]
            target.mkdir(parents=True, exist_ok=True)
            (target / "checkpoint.pt").write_text("fake checkpoint")

        with patch("run_loop.salad_train", side_effect=train):
            loop(args, FakeRunner())
        self.assertFalse(any(name == "train_lora.py" for name, _ in calls))
        generations = [flags for name, flags in calls if name == "generate_candidates.py"]
        self.assertEqual(len(generations), 2)
        self.assertTrue(all("--lora" not in flags for flags in generations))
        report = json.loads((args.root / "full/round_0/feedback.json").read_text())
        self.assertEqual(report["status"], "skipped_no_mined_positives")
        self.assertEqual((args.root / "full/final_pool.jsonl").read_text(), "")

    def test_nonfinite_lora_threshold_and_check_data_are_rejected_before_training(self):
        for raw in ('--min-utility nan', '--min-utility inf', '--check-data'):
            args = self.args('--lora-args=' + raw)
            with patch('run_loop.salad_train') as train:
                with self.assertRaises(ValueError):
                    loop(args, Runner(dry_run=True))
                train.assert_not_called()

    def row(self, key, arm):
        return {"sample_id": key, "source_path": str(self.root / f"{key}.jpg"),
                "output_path": str(self.root / arm / f"{key}.jpg"), "condition": "rain",
                "prompt": "rain", "passed": True, "eligible_for_training": True}

    def test_matched_pools_use_common_verified_groups(self):
        for arm, keys in {"random": ["a", "b"], "select": ["a", "b"], "full": ["b", "c"]}.items():
            write_jsonl(self.root / arm / "final_pool.jsonl", [self.row(k, arm) for k in keys])
        report = audit_pools(self.root, match=True)
        self.assertFalse(report["composition_matched"])
        self.assertEqual(report["matched_pool_groups"], 1)
        for arm in ("random", "select", "full"):
            selected = (self.root / arm / "matched_pool.jsonl").read_text()
            self.assertEqual(json.loads(selected)["sample_id"], "b")

    def test_matching_rejects_missing_arms_and_conflicting_source_identity(self):
        write_jsonl(self.root / "random/final_pool.jsonl", [self.row("a", "random")])
        with self.assertRaisesRegex(ValueError, "all three"):
            audit_pools(self.root, match=True)
        for arm in ("select", "full"):
            write_jsonl(self.root / arm / "final_pool.jsonl", [self.row("a", arm)])
        conflicting = self.row("a", "full")
        conflicting["condition"] = "snow"
        write_jsonl(self.root / "full/final_pool.jsonl", [conflicting])
        with self.assertRaisesRegex(ValueError, "disagree"):
            audit_pools(self.root, match=True)


if __name__ == "__main__":
    unittest.main()
