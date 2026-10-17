"""CPU-only metric migration compared with the current real gate implementation."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.qwen_curriculum import common, quality, requalify_v2 as migration
from experiments.qwen_curriculum.weather_signal import WEATHER_TEXT


class Matcher:
    def __init__(self, result):
        self.result = result

    def __call__(self, source, generated):
        return self.result


class Evaluator:
    mock = False
    matcher = None

    def _compute_s_geo(self, source, generated):
        result = self.matcher(source, generated)
        return result["num_inliers"] / len(result["matched_kpts0"]) if len(result["matched_kpts0"]) else 0

    def _compute_s_div(self, source, generated):
        return 0.001


class Weather:
    def __init__(self, source=-2.0, generated=1.0, competitor=0.5):
        self.source, self.generated, self.competitor = source, generated, competitor

    def measure(self, source, generated, condition):
        contrast = self.generated if condition == "rain" else self.competitor
        return {"weather_source_logit": self.source, "weather_generated_logit": contrast,
                "weather_shift": contrast - self.source, "weather_signal_model": "fake/clip"}


def sealed(row):
    body = {key: value for key, value in row.items() if key != "result_sha256"}
    return {**body, "result_sha256": common.fingerprint(body)}


def frozen(body):
    body = {key: value for key, value in body.items() if key != "fingerprint"}
    return {**body, "fingerprint": common.fingerprint(body)}


class RequalifyTests(unittest.TestCase):
    def make(self, matches=80, inliers=None, displacement=0.0, side=512, coverage=1.0,
             weather=None, raw_change=None, thresholds=None, mismatched_canvas=False):
        inliers = matches if inliers is None else inliers
        axis = np.array([0.125, 0.375, 0.625, 0.875]) * side
        points = np.array([(x, y) for y in axis for x in axis])
        homography = np.eye(3)
        homography[0, 2] = displacement
        raw = {"matched_kpts0": np.resize(points, (matches, 2)),
            "matched_kpts1": np.resize(points, (matches, 2)) + [displacement, 0],
            "inlier_kpts0": np.resize(points, (inliers, 2)),
            "inlier_kpts1": np.resize(points, (inliers, 2)) + [displacement, 0],
            "all_kpts0": np.resize(points, (max(2048, matches), 2)),
            "all_kpts1": np.resize(points, (max(2048, matches), 2)), "num_inliers": inliers, "H": homography}
        if raw_change is not None:
            raw_change(raw)
        verifier = quality.QwenQualityVerifier(device="cpu", matcher=Matcher(raw), evaluator=Evaluator(),
            weather_signal=weather or Weather(), clip_model_name="fake/clip", img_size=side, thresholds=thresholds)
        source = Image.new("RGB", (400, 300))
        generated = Image.new("RGB", (300, 300) if mismatched_canvas else (400, 300))
        verdict = verifier.evaluate(source, generated, "rain")
        old_definition = copy.deepcopy(verifier.definition)
        old_definition.update(version=1, structure_gate=migration.LEGACY_STRUCTURE_GATE)
        old_definition["thresholds"].update(min_matches=64, min_source_inlier_grid_coverage=0.35)
        reasons = list(verdict["rejection_reasons"])
        if matches < 64 and "geometry_too_few_matches" not in reasons:
            reasons.append("geometry_too_few_matches")
        if coverage < 0.35:
            reasons.append("geometry_source_inlier_coverage_below_minimum")
        canonical = ["geometry_canvas_dimensions_differ", "geometry_invalid_match_evidence",
            "geometry_inlier_ratio_below_minimum_or_invalid", "geometry_too_few_matches",
            "geometry_source_inlier_coverage_below_minimum", "geometry_homography_missing_or_invalid",
            "geometry_homography_displacement_above_maximum", "weather_shift_not_above_minimum",
            "weather_generated_contrast_not_above_minimum"]
        reasons = [reason for reason in canonical if reason in reasons]
        geometry_ok = not any(reason.startswith("geometry_") for reason in reasons)
        weather_ok = not any(reason.startswith("weather_") for reason in reasons)
        passed = geometry_ok and weather_ok
        job = {"sample_id": "qwen_" + common.fingerprint("job")[:24], "source_path": "original.jpg",
            "source_sha256": "source hash", "source_dimensions": [400, 300], "condition": "rain",
            "prompt": "unchanged original prompt", "negative_prompt": "", "seed": 123, "city": "City", "place_id": 3}
        job["record_sha256"] = common.fingerprint(job)
        row = sealed({**job, "execution_fingerprint": "authenticated-old-execution",
            "output_path": "original-normalized.png", "output_sha256": "normalized hash",
            "raw_output_path": "original-raw.png", "raw_output_sha256": "raw hash",
            "sampling": {"canvas_policy": "source_aspect_v1", "raw_dimensions": [1472, 1104]},
            "origin": "new_hard_source_qwen", **verdict,
            "source_inlier_coverage": {"grid_coverage": coverage, "grid_cells": 16},
            "control_inlier_coverage": {"grid_coverage": coverage, "grid_cells": 16},
            "geometry_ok": geometry_ok, "weather_ok": weather_ok, "passed": passed, "eligible_for_training": passed,
            "status": "passed" if passed else "rejected", "rejection_reasons": reasons})
        execution = frozen({"stage": "quality_v2_import", "quality": verifier.definition,
            "matcher_device": "cpu", "historical_quality_definitions": {row["execution_fingerprint"]: old_definition}})
        return row, execution, verdict

    def assert_same_verdict(self, migrated, actual):
        self.assertIsNotNone(migrated)
        for key in ("geometry_ok", "weather_ok", "passed", "eligible_for_training", "status", "rejection_reasons",
                    "weather_strongest_competitor", "weather_dominant_condition", "weather_target_vs_competing_margin"):
            self.assertEqual(migrated[key], actual[key], key)

    def test_matches_16_to_63_now_pass_and_coverage_is_descriptive(self):
        for matches in (16, 32, 63, 64):
            with self.subTest(matches=matches):
                row, execution, actual = self.make(matches=matches, coverage=0.0)
                self.assertFalse(row["passed"])
                migrated = migration.requalify(row, execution)
                self.assert_same_verdict(migrated, actual)
                self.assertTrue(migrated["passed"])
                self.assertNotIn("geometry_source_inlier_coverage_below_minimum", migrated["rejection_reasons"])
                self.assertEqual(migrated["source_inlier_coverage"], row["source_inlier_coverage"])

    def test_fewer_than_16_matches_still_rejects(self):
        row, execution, actual = self.make(matches=15)
        result = migration.requalify(row, execution)
        self.assert_same_verdict(result, actual)
        self.assertIn("geometry_too_few_matches", result["rejection_reasons"])

    def test_inlier_ratio_boundary_is_unchanged(self):
        for inliers in (77, 78, 79):
            with self.subTest(inliers=inliers):
                row, execution, actual = self.make(matches=100, inliers=inliers)
                self.assert_same_verdict(migration.requalify(row, execution), actual)

    def test_homography_boundary_and_scaled_matcher_size_match_actual_evaluator(self):
        for side, displacement in ((512, 16), (512, 16.001), (256, 8), (256, 8.001)):
            with self.subTest(side=side, displacement=displacement):
                row, execution, actual = self.make(side=side, displacement=displacement)
                self.assert_same_verdict(migration.requalify(row, execution), actual)

    def test_absolute_weather_and_shift_boundaries_are_strict(self):
        for source, generated in ((-2, 0), (2, 2), (3, 2), (-2, 1e-12)):
            with self.subTest(source=source, generated=generated):
                row, execution, actual = self.make(weather=Weather(source=source, generated=generated))
                self.assert_same_verdict(migration.requalify(row, execution), actual)

    def test_competing_margin_and_s_div_are_not_acceptance_gates(self):
        row, execution, actual = self.make(weather=Weather(competitor=20))
        result = migration.requalify(row, execution)
        self.assert_same_verdict(result, actual)
        self.assertTrue(result["passed"])
        self.assertLess(result["weather_target_vs_competing_margin"], 0)
        self.assertEqual(result["s_div"], 0.001)

    def test_known_invalid_homography_stays_rejected_without_inference(self):
        for matrix in (None, np.zeros((3, 3)), np.array([[1., 0, 0], [0, 1, 0], [1, 0, -256]])):
            with self.subTest(matrix=matrix):
                row, execution, actual = self.make(raw_change=lambda raw: raw.update(H=matrix))
                result = migration.requalify(row, execution)
                self.assert_same_verdict(result, actual)
                self.assertIn("geometry_homography_missing_or_invalid", result["rejection_reasons"])

    def test_original_invalid_match_evidence_cannot_be_silently_promoted(self):
        row, execution, actual = self.make(inliers=72, raw_change=lambda raw: raw.update(num_inliers=75))
        self.assertIn("geometry_invalid_match_evidence", row["rejection_reasons"])
        result = migration.requalify(row, execution)
        self.assert_same_verdict(result, actual)
        self.assertFalse(result["passed"])

    def test_original_canvas_mismatch_is_retained(self):
        row, execution, actual = self.make(mismatched_canvas=True)
        self.assert_same_verdict(migration.requalify(row, execution), actual)

    def test_migration_preserves_original_job_bytes_paths_sampling_and_provenance(self):
        row, execution, _ = self.make(matches=32)
        original = copy.deepcopy(row)
        with patch.object(quality.QwenQualityVerifier, "_ensure_matcher", side_effect=AssertionError("no matcher")), \
                patch.object(quality.QwenQualityVerifier, "_ensure_weather", side_effect=AssertionError("no CLIP")), \
                patch("requests.Session", side_effect=AssertionError("no HTTP")):
            result = migration.requalify(row, execution)
        self.assertEqual(row, original)
        for key in ("sample_id", "record_sha256", "source_path", "source_sha256", "prompt", "seed", "sampling",
                    "output_path", "output_sha256", "raw_output_path", "raw_output_sha256", "origin"):
            self.assertEqual(result[key], original[key])
        proof = result["quality_migration"]
        self.assertEqual(proof["original_result_sha256"], row["result_sha256"])
        self.assertEqual(proof["original_execution_fingerprint"], row["execution_fingerprint"])
        self.assertEqual(proof["quality_definition_fingerprint"], common.fingerprint(execution["quality"]))
        self.assertEqual(proof["mode"], "computed_from_recorded_metrics")
        self.assertEqual((proof["model_calls"], proof["http_calls"]), (0, 0))
        self.assertEqual(result["execution_fingerprint"], execution["fingerprint"])
        self.assertEqual(result, sealed(result))
        self.assertEqual(migration.requalify(row, execution), result)

    def test_missing_or_inconsistent_diagnostics_require_model_review(self):
        changes = [lambda row: row.pop("num_inliers"), lambda row: row.update(num_matched=True),
            lambda row: row.update(num_inliers=row["num_matched"] + 1), lambda row: row.update(s_geo=0.2),
            lambda row: row.update(estimated_H_grid_displacement_median_px=3.0),
            lambda row: row.pop("estimated_H"), lambda row: row.update(estimated_H_grid_displacement_median_px=None),
            lambda row: row.update(weather_shift=42), lambda row: row["weather_generated_condition_logits"].pop("snow"),
            lambda row: row.update(weather_signal_model="different/clip"),
            lambda row: row.pop("output_sha256"), lambda row: row.pop("quality_threshold_status")]
        for change in changes:
            row, execution, _ = self.make()
            change(row)
            row = sealed(row)
            with self.subTest(change=change):
                self.assertIsNone(migration.requalify(row, execution))
                self.assertIsInstance(migration.requalification_issue(row, execution), str)

    def test_unknown_reasons_or_incomplete_original_verdict_require_model_review(self):
        for field, value in (("rejection_reasons", ["unknown_model_rule"]), ("passed", False),
                             ("geometry_ok", None), ("status", "generation_error")):
            row, execution, _ = self.make()
            row[field] = value
            self.assertIsNone(migration.requalify(sealed(row), execution))

    def test_unproven_original_execution_or_changed_measurement_definition_requires_model_review(self):
        row, execution, _ = self.make()
        for mutation in (lambda definitions: definitions.clear(),
                         lambda definitions: definitions[row["execution_fingerprint"]]["weather_signal"].update(scale=99),
                         lambda definitions: definitions[row["execution_fingerprint"]].update(matcher_input="different"),
                         lambda definitions: definitions[row["execution_fingerprint"]]["thresholds"].pop("min_matches")):
            changed = copy.deepcopy(execution)
            mutation(changed["historical_quality_definitions"])
            self.assertIsNone(migration.requalify(row, frozen(changed)))

    def test_prior_current_v2_rows_can_be_migrated_with_their_authenticated_definition(self):
        row, execution, actual = self.make(matches=64)
        changed = copy.deepcopy(execution)
        changed["historical_quality_definitions"][row["execution_fingerprint"]] = changed["quality"]
        self.assert_same_verdict(migration.requalify(row, frozen(changed)), actual)

    def test_custom_current_weather_thresholds_use_actual_strict_comparisons(self):
        thresholds = {"weather_thresholds": {"rain": {"min_shift": 3.0, "min_generated_logit": 1.0}}}
        row, execution, actual = self.make(thresholds=thresholds)
        self.assert_same_verdict(migration.requalify(row, execution), actual)
        self.assertFalse(actual["passed"])

    def test_tampered_original_result_or_target_execution_raises(self):
        row, execution, _ = self.make()
        row["s_geo"] = 0.2
        with self.assertRaisesRegex(ValueError, "checksum"):
            migration.requalify(row, execution)
        row, execution, _ = self.make()
        execution["quality"]["thresholds"]["min_matches"] = 4
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            migration.requalify(row, execution)


if __name__ == "__main__":
    unittest.main()
