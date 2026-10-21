"""CPU checks for the v3 prompt bank, verifier and planner."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

ADAPTVPR_ROOT = Path(__file__).resolve().parents[3]
if str(ADAPTVPR_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTVPR_ROOT))

from experiments.qwen_curriculum import prompt_bank, quality, weather_signal, weather_v3
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


class PlannerTests(unittest.TestCase):
    def test_plan_quotas_distinct_domains_and_skip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gsv = root / "gsv"
            (gsv / "Dataframes").mkdir(parents=True)
            (gsv / "Images/City").mkdir(parents=True)
            rows = []
            for place in range(12):
                for view in range(4):
                    row = dict(place_id=str(place), year="2020", month="3", northdeg=str(view), city_id="City",
                               lat="1.5", lon="2.5", panoid=f"p{place}v{view}")
                    rows.append(row)
                    Image.new("RGB", (8, 6), (place * 9, view * 40, 0)).save(
                        gsv / "Images/City" / f"City_{place:07d}_2020_03_{view:03d}_1.5_2.5_p{place}v{view}.jpg")
            with (gsv / "Dataframes/City.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            skip = root / "skip.jsonl"
            skip.write_text(json.dumps(dict(city="City", place_id=0)) + "\n")
            exclude = root / "exclude.json"
            exclude.write_text(json.dumps([["City", 1]]))
            args = SimpleNamespace(run_dir=root / "run", exclude_places=exclude, skip_manifest=skip, real_data=gsv,
                                   min_images=4, places=10, sources_per_place=2, domains_per_source=2, seed=3)
            weather_v3.plan(args)
            jobs = weather_v3.read_rows(args.run_dir / "plan.jsonl")
            self.assertEqual(len(jobs), 40)
            self.assertFalse({j["place_id"] for j in jobs} & {0, 1})
            self.assertEqual(Counter(j["condition"] for j in jobs), Counter(weather_v3.quotas(40, prompt_bank.DOMAIN_WEIGHTS)))
            by_source = {}
            for job in jobs:
                by_source.setdefault(job["source_path"], []).append(job["condition"])
            self.assertEqual(len(by_source), 20)
            self.assertGreaterEqual(sum(len(set(v)) == 2 for v in by_source.values()), 18)
            config, reloaded = weather_v3.load_plan(args.run_dir)
            self.assertEqual(reloaded, jobs)
            with self.assertRaises(ValueError):
                weather_v3.plan(args)

    def test_torn_final_line_is_dropped(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.jsonl"
            path.write_text('{"a": 1}\n{"b": ')
            self.assertEqual(weather_v3.read_rows(path), [{"a": 1}])
            path.write_text('{"a": \n{"b": 2}\n')
            with self.assertRaises(ValueError):
                weather_v3.read_rows(path)


if __name__ == "__main__":
    unittest.main()
