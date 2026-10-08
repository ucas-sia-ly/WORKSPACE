"""Experimental Qwen quality checks separating structure from weather semantics.

Geometry combines the production inlier ratio with minimum evidence, spatial
coverage and proximity of the fitted homography to identity. CLIP cosine
distance is recorded but never required. Fixed-scale weather contrasts are
exploratory semantic checks, not calibrated probabilities or independent truth.
The constructor allocates no CLIP model; CPU CLIP is loaded on first evaluation.
"""

from __future__ import annotations

import copy
import math
import os
from typing import Any

import numpy as np
from PIL import Image

from experiments.generation_diagnosis.verifier_controls import CapturingMatcher, matcher_diagnostics
from experiments.qwen_curriculum.weather_signal import (
    WEATHER_TEXT, WeatherSignal, WeatherSignalEvaluator, weather_definition,
)


QUALITY_DEFAULTS = {
    "min_s_geo": 0.78,
    "min_matches": 64,
    "min_source_inlier_grid_coverage": 0.35,
    "max_H_grid_median_px_at_512": 16.0,
    "weather_thresholds": {
        condition: {"min_shift": 0.0, "min_generated_logit": 0.0}
        for condition in WEATHER_TEXT
    },
}


def _thresholds(overrides: dict | None) -> dict:
    result = copy.deepcopy(QUALITY_DEFAULTS)
    if overrides is not None:
        if not isinstance(overrides, dict) or set(overrides) - set(result):
            raise ValueError("Unknown/non-object Qwen quality thresholds")
        for key, value in overrides.items():
            if key != "weather_thresholds":
                result[key] = value
                continue
            if not isinstance(value, dict) or set(value) - set(WEATHER_TEXT):
                raise ValueError("Unknown/non-object condition in weather_thresholds")
            for condition, limits in value.items():
                if not isinstance(limits, dict) or set(limits) - {"min_shift", "min_generated_logit"}:
                    raise ValueError("Unknown/non-object per-condition weather thresholds")
                result[key][condition].update(limits)
    for key in ("min_s_geo", "min_source_inlier_grid_coverage"):
        if type(result[key]) not in (int, float) or not math.isfinite(result[key]) or not 0 <= result[key] <= 1:
            raise ValueError(f"{key} must be finite in [0, 1]")
    if type(result["min_matches"]) is not int or result["min_matches"] < 4:
        raise ValueError("min_matches must be an integer at least 4")
    displacement = result["max_H_grid_median_px_at_512"]
    if type(displacement) not in (int, float) or not math.isfinite(displacement) or displacement < 0:
        raise ValueError("max_H_grid_median_px_at_512 must be finite and nonnegative")
    for limits in result["weather_thresholds"].values():
        for key, value in limits.items():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Weather {key} must be finite")
    return result


def quality_definition(thresholds: dict | None = None, *, matcher_name: str = "superpoint-lightglue",
                       img_size: int = 512, n_kpts: int = 2048, clip_model_name: str | None = None,
                       clip_device: str = "cpu") -> dict:
    """Fingerprintable settings; no matcher/CLIP model loading occurs here."""
    if type(img_size) is not int or img_size <= 0 or type(n_kpts) is not int or n_kpts < 4:
        raise ValueError("img_size must be positive and n_kpts must be at least 4")
    if not isinstance(matcher_name, str) or not matcher_name.strip():
        raise ValueError("matcher_name must be nonempty")
    return {
        "version": 1,
        "thresholds": _thresholds(thresholds),
        "threshold_status": "experimental configurable first-run checks; not calibrated quality guarantees",
        "matcher_name": matcher_name, "matcher_image_size": img_size, "max_num_keypoints": n_kpts,
        "matcher_failure_policy": "raise; no silent matcher substitution",
        "matcher_input": "production JPEG95 temporary images; square resize",
        "structure_gate": "s_geo >= min_s_geo AND matches >= min_matches AND source 4x4 inlier coverage >= minimum AND fitted-H 5x5 median displacement <= maximum scaled from 512 pixels",
        "weather_gate": "generated condition-vs-clear contrast > per-condition minimum AND generated-minus-source contrast > per-condition minimum",
        "s_div_policy": "descriptive only; no minimum cosine distance",
        "competing_weather_policy": "target minus strongest competing-weather logit is descriptive only",
        "weather_signal": weather_definition(clip_model_name), "clip_device": clip_device,
    }


class QwenQualityVerifier:
    """One strict geometry matcher and one lazily loaded shared CLIP model.

    Optional matcher/evaluator/weather_signal injections support CPU tests and
    callers that already hold models. The evaluator must be a real evaluator
    interface, not its random mock mode. Individual instances are sequential.
    SALAD descriptors, place identity and training difficulty are separate stages.
    """

    def __init__(self, device: str = "cuda", *, thresholds: dict | None = None,
                 matcher_name: str = "superpoint-lightglue", img_size: int = 512,
                 n_kpts: int = 2048, clip_device: str = "cpu", clip_model_name: str | None = None,
                 matcher: Any = None, evaluator: Any = None, weather_signal: Any = None):
        self.definition = quality_definition(thresholds, matcher_name=matcher_name, img_size=img_size,
                                             n_kpts=n_kpts, clip_model_name=clip_model_name, clip_device=clip_device)
        self.thresholds = self.definition["thresholds"]
        self.device, self.clip_device = device, clip_device
        self.matcher_name, self.img_size, self.n_kpts = matcher_name, img_size, n_kpts
        self.clip_model_name = self.definition["weather_signal"]["weather_signal_model"]
        self._weather_signal = weather_signal
        if evaluator is None:
            # The production constructor also loads CLIP; geometry requires only
            # these fields. Its _compute_s_geo method stays unchanged.
            evaluator = WeatherSignalEvaluator.__new__(WeatherSignalEvaluator)
            evaluator.mock, evaluator.device = False, device
            evaluator.matcher_name, evaluator.img_size, evaluator.n_kpts = matcher_name, img_size, n_kpts
            evaluator.matcher = None
            evaluator.model, evaluator.processor = None, None
            evaluator.torch, evaluator.functional = None, None
            evaluator.clear_clip_cache()
        if getattr(evaluator, "mock", False):
            raise ValueError("Qwen quality requires real matching, not random mock evaluator scores")
        self.evaluator = evaluator
        evaluator.matcher_name, evaluator.img_size, evaluator.n_kpts = matcher_name, img_size, n_kpts
        self.capture = None
        if matcher is not None:
            self._set_matcher(matcher)
        elif getattr(evaluator, "matcher", None) is not None:
            self._set_matcher(evaluator.matcher)

    def _set_matcher(self, matcher: Any) -> None:
        self.capture = matcher if isinstance(matcher, CapturingMatcher) else CapturingMatcher(matcher)
        self.evaluator.matcher = self.capture

    def _ensure_matcher(self) -> None:
        if self.capture is None:
            from vismatch import get_matcher

            # Explicit loading here avoids DualTraitEvaluator._load_matcher's
            # automatic SIFT fallback, which changes the quality rule silently.
            self._set_matcher(get_matcher(self.matcher_name, device=self.device, max_num_keypoints=self.n_kpts))

    def _ensure_weather(self) -> Any:
        if self._weather_signal is not None:
            return self._weather_signal
        evaluator = self.evaluator
        if getattr(evaluator, "model", None) is None or getattr(evaluator, "processor", None) is None:
            import torch
            import torch.nn.functional as functional
            from transformers import CLIPModel, CLIPProcessor

            local_only = os.getenv("ADAPTVPR_CLIP_LOCAL_FILES_ONLY", "1").lower() in {"1", "true", "yes"}
            evaluator.torch, evaluator.functional = torch, functional
            evaluator.model = CLIPModel.from_pretrained(self.clip_model_name, local_files_only=local_only).to(self.clip_device)
            evaluator.processor = CLIPProcessor.from_pretrained(self.clip_model_name, local_files_only=local_only)
            evaluator.model.eval()
            evaluator.clear_clip_cache()
        else:
            # Reusing a caller's CLIP is explicit; align its tensors to the
            # requested CLIP device and invalidate features from its old device.
            evaluator.model.to(self.clip_device).eval()
            evaluator.clear_clip_cache()
        # The matcher keeps its own device; evaluator.device is used only for
        # CLIP processor tensors after explicit matcher creation above.
        evaluator.device = self.clip_device
        self._weather_signal = WeatherSignal(evaluator)
        return self._weather_signal

    @staticmethod
    def _valid_homography(result: dict, side: int) -> bool:
        try:
            homography = np.asarray(result.get("H"), dtype=np.float64)
            if homography.shape != (3, 3) or not np.isfinite(homography).all() or np.linalg.matrix_rank(homography) < 3:
                return False
            norm = np.linalg.norm(homography)
            if not math.isfinite(float(norm)) or norm == 0:
                return False
            homography = homography / norm
            axis = np.linspace(0, side - 1, 5)
            grid = np.array([(x, y, 1) for y in axis for x in axis])
            denominators = (grid @ homography.T)[:, 2]
            # Reject a horizon crossing the frame; the diagnostic helper would
            # otherwise report a median from only a subset of valid grid points.
            return bool(np.all(denominators > 1e-8) or np.all(denominators < -1e-8))
        except (ValueError, TypeError, np.linalg.LinAlgError):
            return False

    def evaluate(self, source: Image.Image, generated: Image.Image, condition: str) -> dict:
        if condition not in WEATHER_TEXT:
            raise ValueError(f"Unsupported weather condition: {condition!r}")
        self._ensure_matcher()
        self.capture.last_result = None
        s_geo = float(self.evaluator._compute_s_geo(source, generated))
        result = self.capture.last_result
        if not isinstance(result, dict):
            raise RuntimeError("Geometry matcher did not expose its result")
        valid_h = self._valid_homography(result, self.img_size)
        try:
            matched0 = np.asarray(result.get("matched_kpts0", []), dtype=np.float64).reshape(-1, 2)
            matched1 = np.asarray(result.get("matched_kpts1", []), dtype=np.float64).reshape(-1, 2)
            inlier0 = np.asarray(result.get("inlier_kpts0", []), dtype=np.float64).reshape(-1, 2)
            inlier1 = np.asarray(result.get("inlier_kpts1", []), dtype=np.float64).reshape(-1, 2)
            raw_inliers = result.get("num_inliers", 0)
            valid_counts = (isinstance(raw_inliers, (int, np.integer)) and not isinstance(raw_inliers, (bool, np.bool_))
                            and 0 <= raw_inliers <= len(matched0)
                            and len(matched0) == len(matched1)
                            and len(inlier0) == raw_inliers == len(inlier1)
                            and all(np.isfinite(points).all() for points in (matched0, matched1, inlier0, inlier1)))
        except (ValueError, TypeError):
            valid_counts = False
            matched0 = np.empty((0, 2))
        # The existing diagnostic expects a proper matrix. Missing/invalid H is
        # retained as an explicit rejection rather than triggering matrix math.
        diagnostic_result = result.copy()
        diagnostic_result["H"] = (np.asarray(result["H"], dtype=np.float64) / np.linalg.norm(result["H"])) if valid_h else None
        if not valid_counts:
            # Preserve usable match count while excluding malformed inliers
            # from coverage/hull math, which otherwise can emit NaN JSON values.
            diagnostic_result.update(matched_kpts0=matched0 if np.isfinite(matched0).all() else [],
                                     inlier_kpts0=[], inlier_kpts1=[], num_inliers=0,
                                     all_kpts0=[], all_kpts1=[])
        diagnostics = matcher_diagnostics(diagnostic_result, self.img_size)
        reasons = []
        if source.size != generated.size:
            reasons.append("geometry_canvas_dimensions_differ")
        matches, inliers = diagnostics["num_matched"], diagnostics["num_inliers"]
        if not valid_counts:
            reasons.append("geometry_invalid_match_evidence")
        if not math.isfinite(s_geo) or not self.thresholds["min_s_geo"] <= s_geo <= 1:
            reasons.append("geometry_inlier_ratio_below_minimum_or_invalid")
        if matches < self.thresholds["min_matches"]:
            reasons.append("geometry_too_few_matches")
        grid_coverage = diagnostics["source_inlier_coverage"]["grid_coverage"]
        if not math.isfinite(grid_coverage) or grid_coverage < self.thresholds["min_source_inlier_grid_coverage"]:
            reasons.append("geometry_source_inlier_coverage_below_minimum")
        displacement = diagnostics["estimated_H_grid_displacement_median_px"]
        max_displacement = self.thresholds["max_H_grid_median_px_at_512"] * self.img_size / 512
        if not valid_h or displacement is None or not math.isfinite(displacement):
            reasons.append("geometry_homography_missing_or_invalid")
        elif displacement > max_displacement:
            reasons.append("geometry_homography_displacement_above_maximum")
        geometry_reasons = reasons.copy()

        signal = self._ensure_weather()
        # Capturing evaluator shares both normalized embeddings with WeatherSignal.
        s_div = float(self.evaluator._compute_s_div(source, generated))
        measurements = {weather: signal.measure(source, generated, weather) for weather in WEATHER_TEXT}
        weather = measurements[condition]
        limits = self.thresholds["weather_thresholds"][condition]
        shift, generated_logit = weather["weather_shift"], weather["weather_generated_logit"]
        if not math.isfinite(shift) or not shift > limits["min_shift"]:
            reasons.append("weather_shift_not_above_minimum")
        if not math.isfinite(generated_logit) or not generated_logit > limits["min_generated_logit"]:
            reasons.append("weather_generated_contrast_not_above_minimum")
        generated_logits = {key: value["weather_generated_logit"] for key, value in measurements.items()}
        if any(not math.isfinite(value) for value in generated_logits.values()):
            raise ValueError("Non-finite competing weather contrast")
        strongest_competitor = max((key for key in WEATHER_TEXT if key != condition), key=generated_logits.get)
        geometry_ok = not geometry_reasons
        weather_ok = len(reasons) == len(geometry_reasons)
        passed = geometry_ok and weather_ok
        return {
            "s_geo": s_geo, "s_div": s_div, **weather, **diagnostics,
            "condition": condition, "matcher_name": self.matcher_name, "matcher_device": self.device,
            "matcher_image_size": self.img_size, "clip_device": self.clip_device,
            "weather_generated_condition_logits": generated_logits,
            "weather_strongest_competitor": strongest_competitor,
            "weather_target_vs_competing_margin": generated_logit - generated_logits[strongest_competitor],
            "weather_dominant_condition": max(generated_logits, key=generated_logits.get),
            "geometry_ok": geometry_ok, "weather_ok": weather_ok,
            "passed": passed, "eligible_for_training": passed,
            "status": "passed" if passed else "rejected", "rejection_reasons": reasons,
            "quality_threshold_status": self.definition["threshold_status"],
        }
