"""CPU checks for distinct geometry/weather failures and strict matcher choice."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

ADAPTVPR_ROOT = Path(__file__).resolve().parents[3]
if str(ADAPTVPR_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTVPR_ROOT))

from experiments.qwen_curriculum.quality import QUALITY_DEFAULTS, QwenQualityVerifier, quality_definition
from experiments.qwen_curriculum.weather_signal import WEATHER_TEXT


def raw_result(*, matches=80, inliers=72, displacement=0, concentrated=False, side=512):
    axis = np.array([side * 0.125, side * 0.375, side * 0.625, side * 0.875])
    grid = np.array([(x, y) for y in axis for x in axis])
    if concentrated:
        grid = np.array([[10.0, 10.0]])
    matched = np.resize(grid, (matches, 2))
    filtered = np.resize(grid, (inliers, 2))
    homography = np.eye(3)
    homography[0, 2] = displacement
    return {
        "matched_kpts0": matched, "matched_kpts1": matched + [displacement, 0],
        "inlier_kpts0": filtered, "inlier_kpts1": filtered + [displacement, 0],
        "all_kpts0": matched, "all_kpts1": matched, "num_inliers": inliers, "H": homography,
    }


class Matcher:
    def __init__(self, result):
        self.result, self.calls = result, 0

    def __call__(self, image0, image1):
        self.calls += 1
        return self.result


class Evaluator:
    mock = False
    matcher = None

    def __init__(self, diversity=0.01):
        self.diversity = diversity

    def _compute_s_geo(self, source, generated):
        result = self.matcher(source, generated)
        matches = len(result["matched_kpts0"])
        return result["num_inliers"] / matches if matches else 0.0

    def _compute_s_div(self, source, generated):
        return self.diversity


class Weather:
    def __init__(self, *, target="rain", source=-2.0, generated=1.0, competitor=0.5):
        self.target, self.source, self.generated, self.competitor = target, source, generated, competitor
        self.calls = []

    def measure(self, source, generated, condition):
        self.calls.append(condition)
        logit = self.generated if condition == self.target else self.competitor
        return {"weather_source_logit": self.source, "weather_generated_logit": logit,
                "weather_shift": logit - self.source, "weather_signal_model": "fake/clip"}


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.source = Image.new("RGB", (400, 300), "blue")
        self.generated = Image.new("RGB", (400, 300), "gray")

    def verify(self, result=None, weather=None, *, thresholds=None, img_size=512, diversity=0.01):
        verifier = QwenQualityVerifier(device="cpu", matcher=Matcher(result if result is not None else raw_result()),
                                       evaluator=Evaluator(diversity), weather_signal=weather or Weather(),
                                       thresholds=thresholds, img_size=img_size)
        return verifier.evaluate(self.source, self.generated, "rain")

    def test_low_cosine_diversity_can_pass_structure_and_weather(self):
        result = self.verify(diversity=0.001)
        self.assertTrue(result["geometry_ok"])
        self.assertTrue(result["weather_ok"])
        self.assertTrue(result["eligible_for_training"])
        self.assertLess(result["s_div"], 0.15)
        self.assertEqual(result["rejection_reasons"], [])

    def test_high_inlier_ratio_with_few_matches_is_rejected(self):
        result = self.verify(raw_result(matches=32, inliers=32))
        self.assertEqual(result["s_geo"], 1)
        self.assertFalse(result["geometry_ok"])
        self.assertTrue(result["weather_ok"])
        self.assertIn("geometry_too_few_matches", result["rejection_reasons"])

    def test_concentrated_inlier_evidence_is_rejected(self):
        result = self.verify(raw_result(concentrated=True))
        self.assertEqual(result["source_inlier_coverage"]["grid_coverage"], 1 / 16)
        self.assertIn("geometry_source_inlier_coverage_below_minimum", result["rejection_reasons"])

    def test_near_identity_h_boundary_is_closed_and_scales_with_resolution(self):
        self.assertTrue(self.verify(raw_result(displacement=16))["geometry_ok"])
        result = self.verify(raw_result(displacement=16.01))
        self.assertIn("geometry_homography_displacement_above_maximum", result["rejection_reasons"])
        self.assertTrue(self.verify(raw_result(displacement=8, side=256), img_size=256)["geometry_ok"])
        self.assertFalse(self.verify(raw_result(displacement=8.01, side=256), img_size=256)["geometry_ok"])

    def test_missing_singular_nonfinite_and_frame_horizon_h_are_rejected(self):
        for homography in (None, np.zeros((3, 3)), np.full((3, 3), np.nan),
                           np.array([[1.0, 0, 0], [0, 1, 0], [1, 0, -256]])):
            result = raw_result()
            result["H"] = homography
            with self.subTest(homography=homography):
                verdict = self.verify(result)
                self.assertFalse(verdict["geometry_ok"])
                self.assertIn("geometry_homography_missing_or_invalid", verdict["rejection_reasons"])

    def test_homography_scale_does_not_change_mapping_or_acceptance(self):
        result = raw_result(displacement=10)
        result["H"] *= 1e-20
        verdict = self.verify(result)
        self.assertTrue(verdict["geometry_ok"])
        self.assertAlmostEqual(verdict["estimated_H_grid_displacement_median_px"], 10)

    def test_strong_weather_does_not_override_geometry_failure(self):
        result = self.verify(raw_result(matches=80, inliers=40), Weather(generated=20))
        self.assertFalse(result["passed"])
        self.assertFalse(result["geometry_ok"])
        self.assertTrue(result["weather_ok"])

    def test_both_absolute_generated_weather_and_source_relative_shift_required(self):
        for weather, reason in ((Weather(source=-2, generated=0), "weather_generated_contrast_not_above_minimum"),
                                (Weather(source=2, generated=2), "weather_shift_not_above_minimum"),
                                (Weather(source=3, generated=2), "weather_shift_not_above_minimum")):
            with self.subTest(reason=reason):
                result = self.verify(weather=weather)
                self.assertTrue(result["geometry_ok"])
                self.assertFalse(result["weather_ok"])
                self.assertIn(reason, result["rejection_reasons"])

    def test_competing_weather_margin_is_descriptive_and_not_confidence(self):
        weather = Weather(competitor=2)
        result = self.verify(weather=weather)
        self.assertTrue(result["passed"])
        self.assertEqual(result["weather_target_vs_competing_margin"], -1)
        self.assertEqual(set(result["weather_generated_condition_logits"]), set(WEATHER_TEXT))
        self.assertEqual(len(weather.calls), len(WEATHER_TEXT))

    def test_condition_threshold_override_does_not_change_other_conditions(self):
        overrides = {"weather_thresholds": {"rain": {"min_shift": 4.0}}}
        result = self.verify(thresholds=overrides)
        self.assertFalse(result["weather_ok"])
        definition = quality_definition(overrides)
        self.assertEqual(definition["thresholds"]["weather_thresholds"]["fog"]["min_shift"], 0)
        self.assertEqual(QUALITY_DEFAULTS["weather_thresholds"]["rain"]["min_shift"], 0)

    def test_definition_is_model_free_and_independent(self):
        with patch.object(QwenQualityVerifier, "_ensure_weather", side_effect=AssertionError("must not load")):
            verifier = QwenQualityVerifier(device="cpu")
            definition = quality_definition()
        self.assertIsNone(verifier.evaluator.model)
        self.assertIsNone(verifier.capture)
        definition["thresholds"]["min_s_geo"] = 0
        self.assertEqual(QUALITY_DEFAULTS["min_s_geo"], 0.78)
        self.assertEqual(verifier.definition["s_div_policy"], "descriptive only; no minimum cosine distance")

    def test_invalid_settings_rejected_before_loading_models(self):
        for thresholds in ({"unknown": 1}, {"min_s_geo": float("nan")}, {"min_matches": True},
                           {"min_matches": 3}, {"min_source_inlier_grid_coverage": 1.1},
                           {"max_H_grid_median_px_at_512": -1},
                           {"weather_thresholds": {"clear": {"min_shift": 0}}},
                           {"weather_thresholds": {"rain": {"confidence": 1}}}):
            with self.subTest(thresholds=thresholds), self.assertRaises(ValueError):
                QwenQualityVerifier(thresholds=thresholds)
        with self.assertRaises(ValueError):
            quality_definition(img_size=0)

    def test_strict_matcher_initialization_error_has_no_fallback(self):
        fake_vismatch = SimpleNamespace(get_matcher=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("requested matcher failed")))
        verifier = QwenQualityVerifier(device="cpu", evaluator=Evaluator(), weather_signal=Weather())
        with patch.dict(sys.modules, {"vismatch": fake_vismatch}):
            with self.assertRaisesRegex(RuntimeError, "requested matcher failed"):
                verifier.evaluate(self.source, self.generated, "rain")
        self.assertIsNone(verifier.capture)

    def test_canvas_mismatch_rejected_and_unknown_condition_raises(self):
        verifier = QwenQualityVerifier(device="cpu", matcher=Matcher(raw_result()),
                                       evaluator=Evaluator(), weather_signal=Weather())
        result = verifier.evaluate(self.source, self.generated.resize((300, 300)), "rain")
        self.assertIn("geometry_canvas_dimensions_differ", result["rejection_reasons"])
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            verifier.evaluate(self.source, self.generated, "clear")

    def test_inconsistent_inlier_counts_are_not_accepted(self):
        result = raw_result()
        result["num_inliers"] = 75
        verdict = self.verify(result)
        self.assertIn("geometry_invalid_match_evidence", verdict["rejection_reasons"])

    def test_nonfinite_inliers_reject_and_remain_json_serializable(self):
        result = raw_result()
        result["inlier_kpts0"][0, 0] = float("nan")
        verdict = self.verify(result)
        self.assertFalse(verdict["geometry_ok"])
        self.assertIn("geometry_invalid_match_evidence", verdict["rejection_reasons"])
        json.dumps(verdict, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
