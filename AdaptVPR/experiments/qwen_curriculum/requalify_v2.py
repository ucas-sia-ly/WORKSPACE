"""Reapply current Qwen quality gates to authenticated historical measurements.

No model, HTTP request, image, or historical journal is opened or modified.
The caller must first authenticate the original result/execution and archived
verifier implementation.  In particular, the old evaluator's explicit invalid
match-evidence rejection is the surviving proof of checks on keypoint arrays
which were not saved.  Missing or contradictory evidence requires model review
and returns ``None``; a signed, explicit invalid-H result remains a rejection.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from experiments.qwen_curriculum import common, quality
from experiments.qwen_curriculum.weather_signal import WEATHER_TEXT


LEGACY_STRUCTURE_GATE = (
    "s_geo >= min_s_geo AND matches >= min_matches AND source 4x4 inlier coverage >= minimum "
    "AND fitted-H 5x5 median displacement <= maximum scaled from 512 pixels"
)
KNOWN_REASONS = {
    "geometry_canvas_dimensions_differ", "geometry_invalid_match_evidence",
    "geometry_inlier_ratio_below_minimum_or_invalid", "geometry_too_few_matches",
    "geometry_source_inlier_coverage_below_minimum", "geometry_homography_missing_or_invalid",
    "geometry_homography_displacement_above_maximum", "weather_shift_not_above_minimum",
    "weather_generated_contrast_not_above_minimum",
}


class _NeedsModelReview(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise _NeedsModelReview(message)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _close(left, right):
    return _number(left) and _number(right) and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10)


def _target(execution):
    body = {key: value for key, value in execution.items() if key != "fingerprint"}
    if execution.get("fingerprint") != common.fingerprint(body):
        raise ValueError("Invalid target quality execution fingerprint")
    definition = execution.get("quality")
    if not isinstance(definition, dict) or definition.get("version") != 2:
        raise ValueError("Metric requalification requires a current quality-v2 execution")
    try:
        expected = quality.quality_definition(definition["thresholds"],
            matcher_name=definition["matcher_name"], img_size=definition["matcher_image_size"],
            n_kpts=definition["max_num_keypoints"], clip_model_name=definition["weather_signal"]["weather_signal_model"],
            clip_device=definition["clip_device"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Incomplete target quality-v2 definition") from exc
    if expected != definition:
        raise ValueError("Target quality definition differs from the current quality-v2 implementation")
    return definition


def _historical_definition(row, execution, current):
    definitions = execution.get("historical_quality_definitions")
    _require(isinstance(definitions, dict), "missing authenticated historical quality definitions")
    old = definitions.get(row.get("execution_fingerprint"))
    _require(isinstance(old, dict), "original execution has no authenticated historical quality definition")
    _require(old.get("version") in (1, 2), "unknown historical quality version")
    expected_structure = LEGACY_STRUCTURE_GATE if old["version"] == 1 else current["structure_gate"]
    _require(old.get("structure_gate") == expected_structure, "historical structure policy is not the audited v1/v2 policy")
    old_measurement = {key: value for key, value in old.items() if key not in {"version", "thresholds", "structure_gate"}}
    new_measurement = {key: value for key, value in current.items() if key not in {"version", "thresholds", "structure_gate"}}
    _require(old_measurement == new_measurement, "historical matcher or weather measurement definition changed")
    limits = copy.deepcopy(old.get("thresholds"))
    _require(isinstance(limits, dict), "missing historical thresholds")
    if old["version"] == 1:
        coverage = limits.pop("min_source_inlier_grid_coverage", None)
        _require(_number(coverage) and 0 <= coverage <= 1, "invalid historical coverage threshold")
    _require(set(limits) == set(current["thresholds"]), "incomplete historical threshold definition")
    weather_limits = limits.get("weather_thresholds")
    _require(isinstance(weather_limits, dict) and set(weather_limits) == set(WEATHER_TEXT)
             and all(isinstance(value, dict) and set(value) == {"min_shift", "min_generated_logit"}
                     for value in weather_limits.values()), "incomplete historical weather thresholds")
    try:
        quality.quality_definition(limits)
    except (ValueError, TypeError):
        raise _NeedsModelReview("invalid historical thresholds") from None
    return old


def _old_verdict(row, old):
    _require(row.get("status") in {"passed", "rejected"}, "row has not completed original quality review")
    for field in ("geometry_ok", "weather_ok", "passed", "eligible_for_training"):
        _require(type(row.get(field)) is bool, f"missing original {field} verdict")
    reasons = row.get("rejection_reasons")
    _require(isinstance(reasons, list) and all(type(reason) is str for reason in reasons),
             "missing complete original rejection reasons")
    _require(len(set(reasons)) == len(reasons) and not set(reasons) - KNOWN_REASONS,
             "unknown or duplicate historical rejection diagnostics")
    _require(row["geometry_ok"] == (not any(reason.startswith("geometry_") for reason in reasons)),
             "original geometry verdict contradicts its recorded diagnostics")
    _require(row["weather_ok"] == (not any(reason.startswith("weather_") for reason in reasons)),
             "original weather verdict contradicts its recorded diagnostics")
    passed = row["geometry_ok"] and row["weather_ok"]
    _require(row["passed"] == row["eligible_for_training"] == passed and row["status"] == ("passed" if passed else "rejected"),
             "original combined verdict is inconsistent")
    _require(row.get("quality_threshold_status") == old["threshold_status"], "missing original quality-policy marker")
    return reasons


def _metrics(row, execution, current, old_reasons):
    for field in ("sample_id", "record_sha256", "source_path", "source_sha256", "output_path", "output_sha256",
                  "raw_output_path", "raw_output_sha256", "execution_fingerprint"):
        _require(isinstance(row.get(field), str) and bool(row[field]), f"missing original sealed-image identity: {field}")
    condition = row.get("condition")
    _require(condition in WEATHER_TEXT, "missing supported target condition")
    _require(row.get("matcher_name") == current["matcher_name"], "matcher identity differs")
    _require(row.get("matcher_image_size") == current["matcher_image_size"], "matcher image size differs")
    _require(row.get("matcher_device") == execution.get("matcher_device") and isinstance(row.get("matcher_device"), str),
             "matcher device differs or is missing")
    _require(row.get("clip_device") == current["clip_device"], "CLIP device differs")
    _require(row.get("weather_signal_model") == current["weather_signal"]["weather_signal_model"], "CLIP model differs")
    dimensions = row.get("source_dimensions")
    _require(isinstance(dimensions, list) and len(dimensions) == 2
             and all(type(value) is int and value > 0 for value in dimensions), "missing original image dimensions")
    for field in ("s_geo", "s_div", "num_matched", "num_inliers", "num_detected_source", "num_detected_control"):
        _require(field in row, f"missing original {field} diagnostic")
    for field in ("num_matched", "num_inliers", "num_detected_source", "num_detected_control"):
        _require(type(row[field]) is int and row[field] >= 0, f"invalid original {field} count")
    matches, inliers = row["num_matched"], row["num_inliers"]
    _require(0 <= inliers <= matches and _number(row["s_geo"]) and _number(row["s_div"]),
             "invalid original inlier ratio or match counts")
    invalid_counts = "geometry_invalid_match_evidence" in old_reasons
    if not invalid_counts:
        _require(_close(row["s_geo"], inliers / matches if matches else 0.0), "inlier ratio contradicts saved counts")
    for field, numerator in (("source_matched_fraction", matches), ("source_inlier_fraction", inliers)):
        expected = numerator / row["num_detected_source"] if row["num_detected_source"] else 0.0
        _require(_close(row.get(field), expected), f"inconsistent or missing original {field}")
    _require("estimated_H" in row and "estimated_H_grid_displacement_median_px" in row
             and "estimated_H_grid_displacement_max_px" in row, "missing original homography diagnostics")
    homography = row["estimated_H"]
    valid_h = quality.QwenQualityVerifier._valid_homography({"H": homography}, current["matcher_image_size"])
    old_invalid_h = "geometry_homography_missing_or_invalid" in old_reasons
    _require(valid_h != old_invalid_h, "stored homography validity contradicts original evaluator diagnostics")
    if not valid_h:
        _require(homography is None and row["estimated_H_grid_displacement_median_px"] is None
                 and row["estimated_H_grid_displacement_max_px"] is None,
                 "invalid homography was not saved in the audited diagnostic format")
    else:
        side = current["matcher_image_size"]
        matrix = np.asarray(homography, dtype=np.float64)
        axis = np.linspace(0, side - 1, 5)
        grid = np.array([(x, y) for y in axis for x in axis], dtype=np.float64)
        mapped_h = np.c_[grid, np.ones(len(grid))] @ matrix.T
        mapped = mapped_h[:, :2] / mapped_h[:, 2:]
        displacement = np.linalg.norm(mapped - grid, axis=1)
        for field, expected in (("estimated_H_grid_displacement_median_px", float(np.median(displacement))),
                                ("estimated_H_grid_displacement_max_px", float(np.max(displacement)))):
            _require(_close(row[field], expected) and row[field] >= 0,
                     f"inconsistent recorded {field}; model review required")
    generated_logits = row.get("weather_generated_condition_logits")
    _require(isinstance(generated_logits, dict) and set(generated_logits) == set(WEATHER_TEXT)
             and all(_number(value) for value in generated_logits.values()), "incomplete or invalid competing-weather diagnostics")
    for field in ("weather_source_logit", "weather_generated_logit", "weather_shift"):
        _require(_number(row.get(field)), f"missing or invalid original {field}")
    _require(_close(row["weather_shift"], row["weather_generated_logit"] - row["weather_source_logit"]),
             "weather shift contradicts original source/generated measurements")
    _require(row["weather_generated_logit"] == generated_logits[condition],
             "target weather contrast contradicts condition logits")
    return invalid_counts, valid_h, {key: generated_logits[key] for key in WEATHER_TEXT}


def _requalify(row, execution):
    current = _target(execution)
    body = {key: value for key, value in row.items() if key != "result_sha256"}
    try:
        digest = common.fingerprint(body)
    except (TypeError, ValueError) as exc:
        raise ValueError("Original quality row is not finite, JSON-serializable sealed evidence") from exc
    if row.get("result_sha256") != digest:
        raise ValueError("Original quality result checksum mismatch")
    old = _historical_definition(row, execution, current)
    old_reasons = _old_verdict(row, old)
    invalid_counts, valid_h, generated_logits = _metrics(row, execution, current, old_reasons)
    thresholds, condition = current["thresholds"], row["condition"]
    reasons = []
    if "geometry_canvas_dimensions_differ" in old_reasons:
        reasons.append("geometry_canvas_dimensions_differ")
    if invalid_counts:
        reasons.append("geometry_invalid_match_evidence")
    if not thresholds["min_s_geo"] <= row["s_geo"] <= 1:
        reasons.append("geometry_inlier_ratio_below_minimum_or_invalid")
    if row["num_matched"] < thresholds["min_matches"]:
        reasons.append("geometry_too_few_matches")
    if not valid_h:
        reasons.append("geometry_homography_missing_or_invalid")
    elif row["estimated_H_grid_displacement_median_px"] > thresholds["max_H_grid_median_px_at_512"] * current["matcher_image_size"] / 512:
        reasons.append("geometry_homography_displacement_above_maximum")
    geometry_ok = not reasons
    limits = thresholds["weather_thresholds"][condition]
    if not row["weather_shift"] > limits["min_shift"]:
        reasons.append("weather_shift_not_above_minimum")
    if not row["weather_generated_logit"] > limits["min_generated_logit"]:
        reasons.append("weather_generated_contrast_not_above_minimum")
    weather_ok = not any(reason.startswith("weather_") for reason in reasons)
    passed = geometry_ok and weather_ok
    competitor = max((key for key in WEATHER_TEXT if key != condition), key=generated_logits.get)
    migrated = {**row, "execution_fingerprint": execution["fingerprint"],
        "weather_generated_condition_logits": generated_logits,
        "weather_strongest_competitor": competitor,
        "weather_target_vs_competing_margin": row["weather_generated_logit"] - generated_logits[competitor],
        "weather_dominant_condition": max(generated_logits, key=generated_logits.get),
        "geometry_ok": geometry_ok, "weather_ok": weather_ok,
        "passed": passed, "eligible_for_training": passed, "status": "passed" if passed else "rejected",
        "rejection_reasons": reasons, "quality_threshold_status": current["threshold_status"],
        "quality_migration": {"version": 1, "mode": "computed_from_recorded_metrics",
            "original_result_sha256": row["result_sha256"],
            "original_execution_fingerprint": row["execution_fingerprint"],
            "original_quality_definition_fingerprint": common.fingerprint(old),
            "quality_definition_fingerprint": common.fingerprint(current),
            "model_calls": 0, "http_calls": 0,
            "geometry_validity_evidence": "authenticated original evaluator rejection diagnostics; saved counts and ratio; rechecked fitted homography",
            "legacy_coverage_statistics": "retained as historical descriptions; excluded from all quality-v2 gates",
            "statement": "computed_from_recorded_metrics; original keypoint and CLIP inference were not rerun"}}
    migrated.pop("result_sha256", None)
    return {**migrated, "result_sha256": common.fingerprint(migrated)}


def requalify(row, execution):
    """Return a sealed v2 row, or ``None`` when fresh model review is required.

    ``execution`` is a new frozen import execution containing the current
    ``quality`` definition and ``historical_quality_definitions`` mapping each
    authenticated original execution fingerprint to its archived definition.
    Corrupt seals or invalid target settings raise instead of invoking fallback.
    """
    try:
        return _requalify(row, execution)
    except _NeedsModelReview:
        return None


def requalification_issue(row, execution):
    """Explain a ``None`` result for audit output without loading any models."""
    try:
        _requalify(row, execution)
    except _NeedsModelReview as exc:
        return str(exc)
    return None
