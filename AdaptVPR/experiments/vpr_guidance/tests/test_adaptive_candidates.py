"""Adaptive budget, gating and interruption semantics without model downloads."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adaptive_candidates import (finish, parse_args, qualifies, seal_row,
                                 stopping_reason, validated_prefixes)
from common import write_jsonl


class AdaptiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = SimpleNamespace(sampling_policy="adaptive", num_candidates=4, min_candidates=1,
                                    stop_utility=0., stop_mining_probability=.25, selection="hardness", num_sources=1, seed=42)

    def row(self, index=0, **overrides):
        return {"sample_id": "a", "candidate_index": index, "passed": True, "plausible": True,
                "eligible_for_training": True, "weather_ok": True, "score_status": "scored",
                "utility": .2, "mining_probability": .5, "mean_positive_similarity": .4, **overrides}

    def test_stop_requires_quality_plausibility_and_strict_utility(self):
        for overrides in ({"eligible_for_training": False}, {"plausible": False},
                          {"utility": 0.}, {"mining_probability": .125}):
            self.assertFalse(qualifies(self.row(**overrides), self.args))
        self.assertEqual(stopping_reason([self.row()], self.args), "target_reached")
        self.args.min_candidates = 2
        self.assertIsNone(stopping_reason([self.row()], self.args))
        self.assertEqual(stopping_reason([self.row(), self.row(1, utility=0.)], self.args), "target_reached")

    def test_fixed_policy_uses_full_budget_even_with_good_first_candidate(self):
        self.args.sampling_policy = "fixed"
        self.assertIsNone(stopping_reason([self.row()], self.args))
        self.assertEqual(stopping_reason([self.row(i) for i in range(4)], self.args), "budget_exhausted")

    def test_exhaustion_keeps_best_eligible_fallback_without_claiming_target(self):
        rows = [self.row(i, utility=0.) for i in range(4)]
        rows[1] = self.row(1, eligible_for_training=False, weather_ok=False, utility=10.)
        write_jsonl(self.root / "candidates.jsonl", rows)
        finish(self.root, {"fingerprint": "f", "sample_ids": ["a"]}, {"a": rows}, self.args, {}, 0.)
        selected = json.loads((self.root / "selected.jsonl").read_text())
        self.assertNotEqual(selected["candidate_index"], 1)
        summary = json.loads((self.root / "summary.json").read_text())
        self.assertEqual(summary["stop_reasons"], {"budget_exhausted": 1})
        self.assertFalse(summary["groups"][0]["target_found"])

    def test_corrupt_image_discards_later_records_for_the_same_group(self):
        rows = [seal_row(self.row(i, utility=0.)) for i in range(3)]
        with patch("adaptive_candidates._validated_rows", return_value=[rows[0], rows[2]]):
            groups = validated_prefixes(rows, {"sample_ids": ["a"]}, self.root, self.args)
        self.assertEqual([row["candidate_index"] for row in groups["a"]], [0])

    def test_checksum_and_rows_after_stop_are_rejected(self):
        row = seal_row(self.row())
        row["utility"] = .9
        with self.assertRaisesRegex(ValueError, "checksum"):
            validated_prefixes([row], {"sample_ids": ["a"]}, self.root, self.args)
        rows = [seal_row(self.row(i)) for i in range(2)]
        with patch("adaptive_candidates._validated_rows", return_value=rows):
            with self.assertRaisesRegex(ValueError, "after adaptive stop"):
                validated_prefixes(rows, {"sample_ids": ["a"]}, self.root, self.args)

    def test_invalid_budget_and_nonfinite_thresholds_fail_at_parse(self):
        base = ["--prompts", "p", "--image-root", "i", "--real-data", "r", "--checkpoint", "c",
                "--output-dir", "o", "--num-sources", "1"]
        for flags in (["--min-candidates", "5"], ["--weather-min-shift", "nan"],
                      ["--stop-utility", "inf"], ["--stop-mining-probability", "nan"]):
            with self.assertRaises(SystemExit):
                parse_args(base + flags)


if __name__ == "__main__":
    unittest.main()
