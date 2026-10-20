"""CPU-only edge cases for honest zero-gradient and incomplete-run summaries."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from summarize_ccr_diagnostics import (CHECKPOINTS, angle_summary, distribution,
                                       load_completed, ratio_summary)


class DiagnosticSummaryTests(unittest.TestCase):
    def test_quantiles_exclude_undefined_instead_of_inserting_zero(self):
        result = distribution([None, 1., 3., 5.])
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["undefined_count"], 1)
        self.assertEqual(result["median"], 3.)
        self.assertAlmostEqual(result["p05"], 1.2)
        self.assertEqual(distribution([None])["mean"], None)
        with self.assertRaises(ValueError):
            distribution([float("nan")])

    def test_zero_real_ratios_and_generator_angles_remain_defined_only_when_valid(self):
        result = ratio_summary([0., 2., 3., 0.], [0., 0., 1., 2.])
        self.assertEqual(result["both_zero_count"], 1)
        self.assertEqual(result["nonzero_numerator_zero_real_count"], 1)
        self.assertEqual(result["zero_real_count"], 2)
        self.assertEqual(result["finite_positive_denominator"]["count"], 2)
        self.assertEqual(result["finite_positive_denominator"]["mean"], 1.5)
        angle = angle_summary(value for value in [
            {"value": None, "angle_degrees": None},
            {"value": -.5, "angle_degrees": 120.},
            {"value": .5, "angle_degrees": 60.},
        ])
        self.assertEqual(angle["defined_count"], 2)
        self.assertEqual(angle["undefined_count"], 1)
        self.assertEqual(angle["negative_cosine_fraction_defined"], .5)
        self.assertEqual(angle["angle_degrees"]["mean"], 90.)

    def test_completed_run_requires_all_checkpoints_and_unchanged_states(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaisesRegex(ValueError, "诊断尚未完成"):
                load_completed(directory)
            payload = {"state_sha256_before": "state", "state_sha256_after": "state",
                       "parameter_versions_unchanged": True,
                       "gradients": [{}], "mining": [{}], "local": {"rows": [{}]}}
            for name in CHECKPOINTS:
                (directory / f"{name}.json").write_text(json.dumps(payload))
            protocol = {"optimizer_created": False, "optimizer_steps": 0,
                        "parameter_state_unchanged": True,
                        "checkpoint_sha256_before": {"checkpoint": "hash"},
                        "checkpoint_sha256_after": {"checkpoint": "hash"}}
            (directory / "protocol.json").write_text(json.dumps(protocol))
            loaded, _ = load_completed(directory)
            self.assertEqual(set(loaded), set(CHECKPOINTS))
            changed = copy.deepcopy(payload)
            changed["state_sha256_after"] = "different"
            (directory / "A.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "A.*内存参数"):
                load_completed(directory)


if __name__ == "__main__":
    unittest.main()
