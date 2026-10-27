"""CPU-only prompt contract and reproducibility tests for opt-in v5."""
from collections import Counter
from pathlib import Path
import random
import sys
import unittest

ADAPTVPR_ROOT = Path(__file__).resolve().parents[3]
if str(ADAPTVPR_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTVPR_ROOT))

from experiments.qwen_curriculum import prompt_bank_v4, prompt_bank_v5


class NightPromptV5Tests(unittest.TestCase):
    def test_weighted_draw_is_order_independent_and_rng_neutral(self):
        before = random.getstate()
        keys = [f"source-{index}" for index in range(20000)]
        forward = [prompt_bank_v5.choose_variant("night", key) for key in keys]
        reverse = [prompt_bank_v5.choose_variant("night", key) for key in reversed(keys)]
        self.assertEqual(forward, list(reversed(reverse)))
        self.assertEqual(random.getstate(), before)
        counts = Counter(forward)
        weights = prompt_bank_v5.VARIANT_WEIGHTS["night"]
        for variant, weight in weights.items():
            self.assertAlmostEqual(counts[variant] / len(keys), weight / sum(weights.values()), delta=0.015)

    def test_strict_night_excludes_twilight_and_shopfront_generation(self):
        variants = prompt_bank_v5.VARIANTS["night"]
        self.assertEqual(set(variants), set(prompt_bank_v5.VARIANT_WEIGHTS["night"]))
        self.assertNotIn("blue_hour", variants)
        self.assertNotIn("shopfronts", variants)
        with self.assertRaises(KeyError):
            prompt_bank_v5.prompt("night", "blue_hour")

    def test_every_night_prompt_relights_low_sky_scene_and_preserves_content(self):
        for variant in prompt_bank_v5.VARIANTS["night"]:
            text = prompt_bank_v5.prompt("night", variant)
            for required in (
                "little or no sky", "walls, facades, roads, pavements and vegetation",
                "solar cast shadows", "A dark sky alone is insufficient",
                "same positions and shapes", "already present", "If no light source is present",
                "Reduced detail visibility", "do not remove objects", "Dry road and dry pavement",
            ):
                self.assertIn(required, text, variant)
            self.assertNotIn("keep facade edges and road layout readable", text.lower())
            self.assertNotIn("motion blur", text.lower())
            self.assertNotIn("high-iso", text.lower())

    def test_existing_vehicle_variant_has_no_vehicle_fallback(self):
        text = prompt_bank_v5.prompt("night", "existing_headlights")
        self.assertIn("If no vehicle is present", text)
        self.assertIn("do not add traffic or beams", text)

    def test_old_records_cannot_be_relabelled_without_exact_prompt(self):
        row = {"condition": "night", "prompt_variant": "led_lamps",
               "prompt_policy": prompt_bank_v4.POLICY_VERSION,
               "prompt": prompt_bank_v4.prompt("night", "led_lamps")}
        self.assertFalse(prompt_bank_v5.matches_record(row))
        row["prompt_policy"] = prompt_bank_v5.POLICY_VERSION
        self.assertFalse(prompt_bank_v5.matches_record(row))
        row["prompt"] = prompt_bank_v5.prompt("night", "led_lamps")
        self.assertTrue(prompt_bank_v5.matches_record(row))
        row["prompt"] += " extra instruction"
        self.assertFalse(prompt_bank_v5.matches_record(row))

    def test_v4_and_other_weather_are_preserved_and_definition_is_detached(self):
        before = prompt_bank_v4.definition()
        snapshot = prompt_bank_v5.definition()
        snapshot["variants"]["night"]["led_lamps"] = "mutation"
        snapshot["inherited_non_night_policy"]["variants"]["rain"].clear()
        for domain, variants in prompt_bank_v4.VARIANTS.items():
            if domain != "night":
                for variant in variants:
                    self.assertEqual(prompt_bank_v5.prompt(domain, variant), prompt_bank_v4.prompt(domain, variant))
        self.assertEqual(before, prompt_bank_v4.definition())
        self.assertNotEqual(prompt_bank_v5.VARIANTS["night"]["led_lamps"], "mutation")


if __name__ == "__main__":
    unittest.main()
