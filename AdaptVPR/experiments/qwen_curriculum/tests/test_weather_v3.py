"""CPU checks for the v3 prompt bank, verifier and journal reader."""

from __future__ import annotations

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image

ADAPTVPR_ROOT = Path(__file__).resolve().parents[3]
if str(ADAPTVPR_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTVPR_ROOT))

from experiments.qwen_curriculum import prompt_bank, quality, weather_signal, weather_io
from experiments.qwen_curriculum.quality_v3 import QwenQualityVerifierV3, THRESHOLDS
from test_quality import Evaluator, Matcher, Weather, raw_result


class PromptBankTests(unittest.TestCase):
    def test_variants_are_deterministic_and_cover_every_domain(self):
        for domain, variants in prompt_bank.VARIANTS.items():
            picks = Counter(prompt_bank.choose_variant(domain, f"source{i}") for i in range(400))
            self.assertEqual(set(picks), set(variants))
            self.assertEqual(prompt_bank.choose_variant(domain, "x"), prompt_bank.choose_variant(domain, "x"))
            for name in variants:
                text = prompt_bank.prompt(domain, name)
                self.assertTrue(text.startswith(prompt_bank.HEADER) and text.endswith(prompt_bank.FOOTER))

    def test_night_variants_always_name_a_light_source(self):
        for domain in prompt_bank.LOW_LIGHT:
            for text in prompt_bank.VARIANTS[domain].values():
                self.assertRegex(text.lower(), "lamp|light|window|sign")


class VerifierTests(unittest.TestCase):
    def verify(self, condition, result, weather=None):
        verifier = QwenQualityVerifierV3(device="cpu", matcher=Matcher(result), evaluator=Evaluator(),
                                         weather_signal=weather or Weather(target=condition))
        return verifier, verifier.evaluate(Image.new("RGB", (40, 30), "blue"), Image.new("RGB", (40, 30), "gray"),
                                           condition)

    def test_new_domains_and_relaxed_geometry(self):
        for condition in ("sun", "overcast", "rainy_night"):
            _, result = self.verify(condition, raw_result(matches=80, inliers=58, displacement=24))
            self.assertTrue(result["passed"], result["rejection_reasons"])
            self.assertEqual(result["quality_version"], "3")
        _, result = self.verify("night", raw_result(matches=80, inliers=50, displacement=40))
        self.assertIn("geometry_inlier_ratio_below_minimum_or_invalid", result["rejection_reasons"])
        self.assertIn("geometry_homography_displacement_above_maximum", result["rejection_reasons"])

    def test_sun_needs_shift_not_positive_contrast(self):
        _, result = self.verify("sun", raw_result(), Weather(target="sun", source=-3.0, generated=-1.0))
        self.assertTrue(result["passed"], result["rejection_reasons"])
        _, result = self.verify("sun", raw_result(), Weather(target="sun", source=-3.0, generated=-2.0))
        self.assertIn("weather_shift_not_above_minimum", result["rejection_reasons"])
        _, result = self.verify("overcast", raw_result(), Weather(target="overcast", source=-3.0, generated=-1.0))
        self.assertIn("weather_generated_contrast_not_above_minimum", result["rejection_reasons"])

    def test_definition_records_bank_and_v2_globals_are_restored(self):
        verifier, _ = self.verify("sun", raw_result())
        self.assertEqual(verifier.definition["thresholds"]["min_s_geo"], THRESHOLDS["min_s_geo"])
        self.assertEqual(verifier.definition["weather_signal"]["neutral_text"]["sun"],
                         prompt_bank.CLIP_NEUTRAL["sun"])
        self.assertNotIn("sun", quality.WEATHER_TEXT)
        self.assertNotIn("sun", weather_signal.WEATHER_TEXT)
        self.assertNotIn("sun", quality.QUALITY_DEFAULTS["weather_thresholds"])
        self.assertEqual(quality.QUALITY_DEFAULTS["min_s_geo"], 0.78)


class JournalTests(unittest.TestCase):
    def test_torn_final_line_is_dropped(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.jsonl"
            path.write_text('{"a": 1}\n{"b": ')
            self.assertEqual(weather_io.read_rows(path), [{"a": 1}])
            path.write_text('{"a": \n{"b": 2}\n')
            with self.assertRaises(ValueError):
                weather_io.read_rows(path)


if __name__ == "__main__":
    unittest.main()
