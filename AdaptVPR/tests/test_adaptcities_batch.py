import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/adaptcities.py"
spec = importlib.util.spec_from_file_location("adaptcities", SCRIPT)
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = SimpleNamespace(output=self.root, reserve_gib=30)
        self.image = self.root / "image.jpg"
        Image.new("RGB", (32, 24), "red").save(self.image)
        self.row = dict(sample_id="adapt_000001", source_id="same.jpg", city="Bangkok",
                        route="global", condition="rain", prompt="Exact initial prompt.\n")
        self.record = dict(self.row, config_fingerprint="config", input_fingerprint=batch.fingerprint(self.row),
                           input_prompt=self.row["prompt"], status="passed", generated=True, passed=True,
                           eligible_for_training=True, s_geo=.9, s_div=.2, tau_geo=.78, tau_div=.15,
                           source_size=[32, 24], rounds_used=1, output_path=str(self.image),
                           source_path=str(self.image), source_sha256=batch.digest(self.image), elapsed_seconds=2., annotation_origin="local_regeneration", example_only=False,
                           reflection_rounds=[dict(prompt=self.row["prompt"], image_path=str(self.image))],
                           artifacts={str(self.image): batch.image_info(self.image)})

    def test_resume_checks_prompt_config_and_decodable_images(self):
        self.assertTrue(batch.valid_record(self.record, self.row, "config"))
        self.assertFalse(batch.valid_record(self.record, self.row, "different_config"))
        changed = dict(self.row, prompt="changed")
        self.assertFalse(batch.valid_record(self.record, changed, "config"))
        self.image.write_bytes(b"truncated jpeg")
        self.assertFalse(batch.valid_record(self.record, self.row, "config"))

    def test_global_cannot_reflect_and_false_acceptance_is_rejected(self):
        r = copy.deepcopy(self.record)
        r["reflection_rounds"] *= 2
        r["rounds_used"] = 2
        self.assertFalse(batch.valid_record(r, self.row, "config"))
        r = dict(self.record, s_geo=.1)
        self.assertFalse(batch.valid_record(r, self.row, "config"))

    def test_rejected_is_completed_but_excluded_from_training_and_shared_source_survives(self):
        rejected_row = dict(self.row, sample_id="adapt_000002")
        rejected = dict(self.record, sample_id="adapt_000002", input_fingerprint=batch.fingerprint(rejected_row),
                        s_geo=.1, passed=False, eligible_for_training=False, status="failed")
        batch.atomic_json(self.root / "records/adapt_000001.json", self.record)
        batch.atomic_json(self.root / "records/adapt_000002.json", rejected)
        # A crash left an incomplete third record: it must not count as complete.
        batch.atomic_json(self.root / "records/adapt_000003.json", {"status": "error"})
        rows = [self.row, rejected_row, dict(self.row, sample_id="adapt_000003")]
        completed = batch.read_completed(self.args, rows, "config")
        self.assertEqual(len(completed), 2)
        batch.export_data(self.args, {"requested": 3}, rows, completed)
        train = list(batch.read_jsonl(self.root / "train_manifest.jsonl"))
        self.assertEqual([r["sample_id"] for r in train], ["adapt_000001"])
        self.assertEqual(len(list(batch.read_jsonl(self.root / "annotations.jsonl"))), 2)
        self.assertFalse(json.loads((self.root / "summary.json").read_text())["complete"])

    def test_output_lock_prevents_concurrent_writers(self):
        with batch.lock(self.root):
            with self.assertRaises(RuntimeError):
                with batch.lock(self.root):
                    pass

    def test_infrastructure_failure_retries_three_times_and_exports_incomplete(self):
        self.args.image_root = self.root
        generator = mock.Mock()
        generator.evaluator.matcher_name = "superpoint-lightglue"
        generator.run_path.side_effect = ConnectionError("service unavailable")
        manifest = {"config_fingerprint": batch.fingerprint({}), "requested": 1, "pilot_count": 1}
        with mock.patch.object(batch, "provenance", return_value={}), \
             mock.patch.object(batch, "service_snapshot"), \
             mock.patch.object(batch, "pilot_budget"), \
             mock.patch.object(batch.time, "sleep"), \
             mock.patch.object(batch.shutil, "disk_usage", return_value=SimpleNamespace(free=1000 * batch.GIB)), \
             mock.patch.object(batch.signal, "signal"), \
             mock.patch("resource_limits.require_limits"), \
             mock.patch("generation.preflight.check_environment", return_value=[]), \
             mock.patch.dict("sys.modules", {"generation.agent": SimpleNamespace(SceneAugmentAgent=mock.Mock(return_value=generator))}):
            with self.assertRaises(ConnectionError):
                batch.run(self.args, manifest, [self.row])
        self.assertEqual(generator.run_path.call_count, 4)
        state = json.loads((self.root / "progress.json").read_text())
        self.assertEqual(state["state"], "stopped_incomplete")
        self.assertEqual(state["completed"], 0)
        self.assertFalse(json.loads((self.root / "summary.json").read_text())["complete"])

    def test_archived_resource_profile_is_explicitly_required_for_resume(self):
        self.assertFalse(batch.valid_record(self.record, self.row, "new_config"))
        self.assertTrue(batch.valid_record(self.record, self.row, {"new_config", "config"}))

    def test_memory_pressure_and_hard_limit_checks(self):
        import resource_limits as limits
        self.assertFalse(limits.pressure_requires_stop(7 * limits.GIB, 10))
        self.assertFalse(limits.pressure_requires_stop(5 * limits.GIB, 2))
        self.assertTrue(limits.pressure_requires_stop(5 * limits.GIB, 3))
        self.assertTrue(limits.pressure_requires_stop(2 * limits.GIB, 0))
        with mock.patch.object(limits, "cgroup_path"), \
             mock.patch.object(limits, "group_stats", return_value={"memory.max": "max", "memory.swap.max": 0}):
            with self.assertRaises(RuntimeError):
                limits.require_limits()
        with mock.patch.object(limits, "cgroup_path"), \
             mock.patch.object(limits, "group_stats", return_value={"memory.max": limits.HOST_MAX, "memory.swap.max": 0}), \
             mock.patch.object(limits, "available_memory", return_value=7 * limits.GIB), \
             mock.patch.dict("os.environ", {"TORCH_COMPILE_DISABLE": "1", "TORCHINDUCTOR_COMPILE_THREADS": "1"}):
            self.assertEqual(limits.require_limits()["memory.max"], limits.HOST_MAX)


if __name__ == "__main__":
    unittest.main()
