import argparse
from collections import Counter
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("strict_demo_entrypoint", ROOT / "tests/run_demo_10.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


class StrictDemoTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="strict-demo-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sources = demo.load_sources()
        self.args = argparse.Namespace(
            strict=True, mode="qwen4b", reflection="off", max_reflections=0,
            seed=0, resume=True, prompts_jsonl=demo.RELEASED_PROMPTS,
            gsvcities_root=self.root / "data", output=self.root / "output",
        )
        self.args.output.mkdir()
        for row in self.sources:
            path = self.args.gsvcities_root / row["gsvcities_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 24), "gray").save(path)

    def test_strict_selects_bundled_published_prompts(self):
        argv = ["demo", "--strict", "--gsvcities-root", str(self.args.gsvcities_root), "--output", str(self.args.output)]
        with patch.object(sys, "argv", argv):
            args = demo.parse_args()
        self.assertEqual(args.prompts_jsonl, demo.RELEASED_PROMPTS)

    def test_exact_source_set_prompt_integrity_and_route_coverage(self):
        _, subset = demo.prepare_inputs(self.args, self.sources)
        rows = demo.load_jsonl(subset)
        self.assertEqual([row["source_id"] for row in rows], [row["source_id"] for row in self.sources])
        self.assertEqual(Counter(row["route"] for row in rows), {"global": 2, "local": 4, "dual": 4})
        self.assertEqual(rows, demo.load_jsonl(demo.RELEASED_PROMPTS))
        metadata = json.loads(demo.RELEASED_PROVENANCE.read_text())
        self.assertEqual(hashlib.sha256(demo.RELEASED_PROMPTS.read_bytes()).hexdigest(), metadata["sha256"])

    def test_strict_invokes_upstream_prompt_mode_in_its_own_output_directory(self):
        _, subset = demo.prepare_inputs(self.args, self.sources)
        with patch.object(demo, "execute") as execute, patch.object(demo, "validate"), contextlib.redirect_stdout(io.StringIO()):
            output = demo.run_one(self.args, "qwen4b", subset)
        command = execute.call_args.args[0]
        self.assertEqual(command[command.index("--mode") + 1], "prompt")
        self.assertIn("--require-generated", command)
        self.assertIn("--resume", command)
        self.assertEqual(command[2], str(subset))
        self.assertEqual(output.name, "strict_released_reflection_off")
        self.assertNotEqual(output.name, "qwen4b_reflection_off")

    def test_missing_or_skip_prompt_fails_before_generation(self):
        rows = demo.load_jsonl(demo.RELEASED_PROMPTS)
        custom = self.root / "bad.jsonl"
        self.args.prompts_jsonl = custom
        custom.write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]))
        with self.assertRaisesRegex(RuntimeError, "missing demo sources"):
            demo.prepare_inputs(self.args, self.sources)
        rows[0] = dict(rows[0], route="skip")
        custom.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "must use global, local, or dual"):
            demo.prepare_inputs(self.args, self.sources)

    def test_ten_sources_reach_generators_even_if_planner_would_skip(self):
        from generation.agent import SceneAugmentAgent

        _, subset = demo.prepare_inputs(self.args, self.sources)
        rows = demo.load_jsonl(subset)
        records = []
        with patch.dict(os.environ, {"ADAPTVPR_DISABLE_MOCK": "0"}), contextlib.redirect_stdout(io.StringIO()):
            agent = SceneAugmentAgent(mock=True, planning_enabled=False, reflection_enabled=False)
            def render(ref_image, *args, **kwargs):
                return ref_image.copy()

            with patch.object(agent, "plan_image", side_effect=AssertionError("Strict must bypass planner Skip")), patch.object(
                agent.iclight, "generate", side_effect=render
            ) as global_generator, patch.object(
                agent.lightx2v, "generate_local", side_effect=render
            ) as local_generator, patch.object(
                agent.lightx2v, "generate_dual", side_effect=render
            ) as dual_generator:
                for row in rows:
                    image = self.args.gsvcities_root / "Images" / row["city"] / row["source_id"]
                    record = agent.run_path(image, self.args.output, entry=dict(row, bad_image=True),
                                            frozen_prompt=True, sample_id=row["sample_id"])
                    self.assertTrue(record["generated"])
                    self.assertNotEqual(record["status"], "skipped")
                    self.assertEqual(record["input_prompt"], row["prompt"])
                    self.assertTrue(Path(record["output_path"]).is_file())
                    records.append(record)
                self.assertEqual((global_generator.call_count, local_generator.call_count, dual_generator.call_count), (2, 4, 4))
                for route, generator in [("global", global_generator), ("local", local_generator), ("dual", dual_generator)]:
                    actual = [call.kwargs.get("prompt", call.args[1] if len(call.args) > 1 else None)
                              for call in generator.call_args_list]
                    self.assertEqual(actual, [row["prompt"] for row in rows if row["route"] == route])
        (self.args.output / "records.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(demo.validate(self.args.output, strict=True)["generated"], 10)

    def test_strict_still_rejects_skipped_or_missing_outputs(self):
        rows = [dict(sample_id=str(i), route="skip", status="skipped", generated=False) for i in range(10)]
        (self.args.output / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "strict coverage failed"):
            demo.validate(self.args.output, strict=True)


if __name__ == "__main__":
    unittest.main()
