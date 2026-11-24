"""V4.2 local joint-content screen using immutable v4.1 distance caches.

This version removes v4.1's unnecessary global DINO-area conjunction. A local
region must already disagree in BOTH DINO and aligned structural contrast.
Rejection limits remain the maximum benign FIT-control score; no limit is
relaxed to make Qwen images pass. Only FIT controls determine scan geometry,
pass/reject limits and coverage floors. Held-out controls are evaluated once
after the rule is frozen, and can veto the entire detector.

It remains an experimental safety screen, not semantic truth or a classifier
of real night-camera realism. Missing evidence and unvalidated rules abstain.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from experiments.qwen_curriculum import content_audit_v4 as legacy

VERSION = "content_safety_v4.2"
FEATURE_VERSION = "content_safety_v4.1"


def compatible_feature_pipeline(pipeline):
    """Explicitly accept only the frozen v4.1 signal schema, not arbitrary maps."""
    expected = {"version": FEATURE_VERSION, "mapping": "inverse_full_patch_footprint",
                "structure": "aligned_local_standardized_gray_cosine_v1",
                "ocr": "unmatched_boxes_diagnostic_only_v1"}
    if any(pipeline.get(key) != value for key, value in expected.items()):
        raise ValueError("Only explicitly compatible frozen v4.1 feature caches can be reused")
    provider = pipeline.get("provider", {})
    for key in ("checkpoint_sha256", "repository_python_sha256"):
        value = provider.get(key, "")
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Cached DINO model/code SHA256 is required")
    if (provider.get("normalization") != "ImageNet" or not provider.get("architecture") or
            not provider.get("torch_version") or not isinstance(pipeline.get("side"), int) or
            pipeline["side"] <= 0 or pipeline["side"] % 14 or
            not isinstance(pipeline.get("search_radius"), int) or pipeline["search_radius"] < 0):
        raise ValueError("Frozen DINO preprocessing/provider metadata is incomplete")


def _validated_signal(signals):
    if not isinstance(signals, dict) or signals.get("error"):
        raise ValueError("Required frozen-model signals missing/failed")
    compatible_feature_pipeline(signals.get("pipeline", {}))
    if signals.get("pipeline_sha256") != legacy.digest(signals["pipeline"]):
        raise ValueError("Feature pipeline fingerprint mismatch")
    arrays = legacy._arrays(signals)
    expected = signals["pipeline"]["side"] // 14
    if arrays[0].shape != (expected, expected) or signals.get("grid_shape") != [expected, expected]:
        raise ValueError("Patch grid differs from frozen DINO resize/patch size")
    return arrays


def _signal_matches_row(row, signals):
    _validated_signal(signals)
    for key in ("source_sha256", "output_sha256"):
        if not row.get(key) or signals.get(key) != row[key]:
            raise ValueError("Cached signal image hashes differ from the exact pair")
    geometry = {"estimated_H": row.get("estimated_H"), "matcher_image_size": row.get("matcher_image_size")}
    if signals.get("geometry_sha256") != legacy.digest(geometry):
        raise ValueError("Cached signal geometry differs from the pair")


def _window_mean(array, window):
    summed = np.pad(array, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return (summed[window:, window:] - summed[:-window, window:] -
            summed[window:, :-window] + summed[:-window, :-window]) / (window * window)


def local_joint_summary(signals, window_fractions):
    """Maximum local mean of min(DINO-distance, structural-distance).

    The minimum requires both signals to support a patch. Invalid areas
    contribute zero; insufficient overall overlap yields UNKNOWN separately.
    OCR novelty never enters this score. A global repaint can also score high;
    no median subtraction would erase its evidence.
    """
    dino, structure, valid, structural_valid = _validated_signal(signals)
    if not window_fractions or any(not 0 < fraction <= 1 for fraction in window_fractions):
        raise ValueError("Scan window geometry must come from FIT-control insertion sizes")
    joint = np.where(valid & structural_valid, np.minimum(dino, structure), 0.).astype(np.float64)
    side = min(joint.shape)
    windows = sorted({max(1, min(side, int(round(side * fraction)))) for fraction in window_fractions})
    scores = [float(_window_mean(joint, size).max()) for size in windows]
    return {"score": max(scores), "window_sizes": windows, "scores_by_window": scores,
            "valid_fraction": float(valid.mean()), "structure_fraction": float(structural_valid.mean()),
            "valid_patches": int(valid.sum()), "structural_patches": int(structural_valid.sum()),
            "dino_mean": float(dino[valid].mean()) if valid.any() else None,
            "structure_mean": float(structure[structural_valid].mean()) if structural_valid.any() else None,
            "ocr": signals.get("ocr", {"available": False}),
            "score_interpretation": "joint_local_safety_outlier_not_hallucination_truth"}


def _partition(key):
    return "heldout" if int(legacy.digest(key)[:8], 16) % 4 == 0 else "fit"


def _pair_controls(rows_and_signals):
    places, pipeline_ids, pipeline = {}, set(), None
    for row, signals in rows_and_signals:
        legacy.require_training_calibration(row)
        _signal_matches_row(row, signals)
        pipeline_ids.add(signals["pipeline_sha256"])
        pipeline = signals["pipeline"]
        group = places.setdefault(row["place_key"], {})
        if row["control"] in group:
            raise ValueError("One benign and one controlled_negative control per training place required")
        group[row["control"]] = (row, signals)
    if len(pipeline_ids) != 1:
        raise ValueError("Exactly one frozen feature pipeline is required")
    for group in places.values():
        if set(group) != {"benign", "controlled_negative"}:
            raise ValueError("Complete paired machine controls required for every training place")
        b, n = group["benign"][0], group["controlled_negative"][0]
        if any(b.get(key) != n.get(key) for key in
               ("source_path", "source_sha256", "estimated_H", "matcher_image_size")):
            raise ValueError("Paired controls require identical source and geometry")
        if any(n["control_recipe"].get(key) != value for key, value in b["control_recipe"].items()):
            raise ValueError("Insertion control must retain the same benign-transform recipe")
    return places, pipeline, next(iter(pipeline_ids))


def _fit_window_geometry(fit_groups):
    widths = []
    for group in fit_groups:
        row, _ = group["controlled_negative"]
        box = np.asarray(row["control_recipe"].get("insertion_xyxy"), dtype=float)
        side = row.get("matcher_image_size")
        if (box.shape != (4,) or not np.isfinite(box).all() or not side or
                box[2] <= box[0] or box[3] <= box[1] or (box < 0).any() or (box > side).any()):
            raise ValueError("FIT controls require the programmatically known insertion geometry")
        # An interior half-width avoids boundary interpolation. Quantiles cover
        # FIT insertion scales; neither Qwen pairs nor heldout masks tune scans.
        widths.append(min(box[2] - box[0], box[3] - box[1]) / side / 2)
    return sorted(set(float(value) for value in np.quantile(widths, [0, .25, .5, .75, 1])))


def _decision(summary, thresholds):
    if (not summary["valid_patches"] or not summary["structural_patches"] or
            summary["valid_fraction"] < thresholds["min_valid_fraction"] or
            summary["structure_fraction"] < thresholds["min_structure_fraction"]):
        return "unknown"
    if summary["score"] > thresholds["reject_above"]:
        return "reject"
    if summary["score"] <= thresholds["pass_at_most"]:
        return "pass"
    return "unknown"


def _paired_test(decisions, target):
    """Exact single-sided binomial discordance test at place level."""
    desired_only = opposite_only = both = neither = 0
    for pair in decisions:
        b = pair["benign"] == target
        n = pair["controlled_negative"] == target
        desired, opposite = (n, b) if target == "reject" else (b, n)
        if desired and opposite:
            both += 1
        elif desired:
            desired_only += 1
        elif opposite:
            opposite_only += 1
        else:
            neither += 1
    discordant = desired_only + opposite_only
    p = (sum(math.comb(discordant, k) for k in range(desired_only, discordant + 1)) /
         2 ** discordant) if discordant else 1.
    return {"name": "exact_paired_discordance_one_sided", "target": target,
            "desired_only": desired_only, "opposite_only": opposite_only,
            "both": both, "neither": neither, "p": p}


def calibrate_controls_v42(rows_and_signals, false_alarm_rate, *, min_pairs=20):
    if not 0 < false_alarm_rate < 1 or min_pairs < 2:
        raise ValueError("Explicit operating alpha in (0,1) and min_pairs>=2 required")
    places, pipeline, pipeline_id = _pair_controls(list(rows_and_signals))
    split = {name: [group for key, group in places.items() if _partition(key) == name]
             for name in ("fit", "heldout")}
    if any(len(groups) < min_pairs for groups in split.values()):
        raise ValueError("Insufficient complete training-place pairs in FIT/heldout partitions")
    windows = _fit_window_geometry(split["fit"])
    fit = [{label: local_joint_summary(signals, windows) for label, (_, signals) in group.items()}
           for group in split["fit"]]
    benign = [pair["benign"] for pair in fit]
    if any(not summary["valid_patches"] or not summary["structural_patches"] for summary in benign):
        raise ValueError("Every FIT benign control requires valid local evidence")
    benign_scores = np.array([pair["benign"]["score"] for pair in fit])
    negative_scores = np.array([pair["controlled_negative"]["score"] for pair in fit])
    # Reject is STRICTLY above the most extreme benign FIT control. Pass is
    # conservative: central benign envelope AND below every known FIT insertion.
    thresholds = {"reject_above": float(benign_scores.max()),
                  "pass_at_most": float(min(np.quantile(benign_scores, 1 - false_alarm_rate),
                                            np.nextafter(negative_scores.min(), -np.inf))),
                  "min_valid_fraction": min(summary["valid_fraction"] for summary in benign),
                  "min_structure_fraction": min(summary["structure_fraction"] for summary in benign)}
    rules = {"version": VERSION, "window_fractions": windows, "thresholds": thresholds,
             "score": "max_local_mean_min_dino_structure", "fit_places": len(fit)}
    # Nothing below this point is allowed to modify the frozen rule.
    frozen_rules_sha = legacy.digest(rules)
    heldout_summaries = [{label: local_joint_summary(signals, windows) for label, (_, signals) in group.items()}
                        for group in split["heldout"]]
    decisions = [{label: _decision(summary, thresholds) for label, summary in pair.items()}
                 for pair in heldout_summaries]
    counts = {label: {"n": len(decisions), **{name: sum(pair[label] == name for pair in decisions)
              for name in ("pass", "reject", "unknown")}} for label in ("benign", "controlled_negative")}
    reject_test, pass_test = _paired_test(decisions, "reject"), _paired_test(decisions, "pass")
    b, n = counts["benign"], counts["controlled_negative"]
    usable = (reject_test["p"] <= false_alarm_rate and pass_test["p"] <= false_alarm_rate and
              b["reject"] / b["n"] <= false_alarm_rate and n["pass"] / n["n"] <= false_alarm_rate)
    artifact = {"version": VERSION, "feature_pipeline": pipeline, "feature_pipeline_sha256": pipeline_id,
                "feature_cache_compatibility": "exact_frozen_v4.1_pipeline_no_new_forward",
                "decision_implementation_sha256": legacy.sha256(__file__), "rules": rules,
                "frozen_fit_rules_sha256": frozen_rules_sha, "alpha": false_alarm_rate,
                "fit_summary": {"benign_score_quantiles": np.quantile(benign_scores, [0, .5, .9, .95, 1]).tolist(),
                                "negative_score_quantiles": np.quantile(negative_scores, [0, .5, .9, .95, 1]).tolist(),
                                "negative_above_benign_max": int((negative_scores > benign_scores.max()).sum())},
                "heldout_controls": counts, "heldout_reject_separation_test": reject_test,
                "heldout_pass_separation_test": pass_test, "usable": bool(usable),
                "scope": "programmatic_training_controls_only_not_hallucination_truth",
                "holdout_limit": "reused_v4.1_control_split_exploratory_not_fresh_external_validation",
                "control_provenance": [{"place_key": key, "partition": _partition(key),
                    **{label: {"sample_id": row["sample_id"], "source_sha256": row["source_sha256"],
                        "output_sha256": row["output_sha256"], "signal_sha256": legacy.digest(signals),
                        "recipe_sha256": legacy.digest(row["control_recipe"])}
                        for label, (row, signals) in group.items()}} for key, group in sorted(places.items())]}
    artifact["artifact_sha256"] = legacy.digest(artifact)
    return artifact


def audit_pair_v42(row, signals, calibration, *, quarantine_variants=("shopfronts",)):
    result = {"version": VERSION, "decision": "unknown", "eligible_for_training": False,
              "reasons": [], "signals_are_semantic_truth": False}
    if row.get("prompt_variant") in quarantine_variants:
        result.update(decision="reject", reasons=["prompt_variant_quarantined_by_policy"])
        return result
    base = row.get("base_quality_passed", row.get("passed") is True and
                   row.get("geometry_ok") is True and row.get("weather_ok") is True)
    if base is not True:
        result["reasons"] = ["base_geometry_weather_not_verified"]
        return result
    try:
        if not calibration or not signals:
            raise ValueError("Required calibration/signals missing")
        if calibration.get("version") != VERSION or calibration.get("artifact_sha256") != legacy.digest(
                {key: value for key, value in calibration.items() if key != "artifact_sha256"}):
            raise ValueError("V4.2 calibration artifact signature invalid")
        if calibration.get("decision_implementation_sha256") != legacy.sha256(__file__):
            raise ValueError("Decision implementation changed after calibration")
        if calibration.get("usable") is not True:
            raise ValueError("Control detector failed heldout automatic validation")
        _signal_matches_row(row, signals)
        if (signals["pipeline_sha256"] != calibration.get("feature_pipeline_sha256") or
                signals["pipeline"] != calibration.get("feature_pipeline")):
            raise ValueError("Exact frozen feature cache pipeline differs from calibration")
        rules = calibration["rules"]
        if legacy.digest(rules) != calibration.get("frozen_fit_rules_sha256"):
            raise ValueError("Frozen FIT decision rules changed")
        summary = local_joint_summary(signals, rules["window_fractions"])
        decision = _decision(summary, rules["thresholds"])
        result.update(decision=decision, eligible_for_training=decision == "pass", diagnostics=summary,
                      calibration_sha256=calibration["artifact_sha256"])
        result["reasons"] = [{"pass": "inside_fit_control_central_safety_envelope",
                              "reject": "local_joint_content_safety_outlier_not_hallucination_proof",
                              "unknown": "ambiguous_band_or_insufficient_valid_overlap"}[decision]]
    except (TypeError, ValueError, KeyError, IndexError) as error:
        result["reasons"] = ["invalid_required_evidence: " + str(error)]
    return result
