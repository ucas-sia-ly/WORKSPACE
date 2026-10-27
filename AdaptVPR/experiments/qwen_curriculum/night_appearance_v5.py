"""Opt-in, regional CLIP *contradiction* checks for night generation.

Center/lower crops are spatial proxies, never inferred sky/foreground masks.
The ranking rule has no fitted photometric cutoff: a daylight interpretation
winning or tying night leaves a candidate Unknown. Passing is only absence of
this particular contradiction, not proof of night realism or preserved content.
No model is loaded here; the caller supplies its existing CLIP evaluator.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping

from PIL import Image


VERSION = "night_appearance_v5.1"
LOGIT_SCALE = 100.0
PROMPT_ENSEMBLE = {
    "night": (
        "a real street photograph at night, with surfaces illuminated by artificial lights",
        "a nighttime photograph of a building facade, with uneven artificial illumination and dark unlit areas",
        "a street photograph after dark, with no daylight illuminating the scene",
    ),
    "daylight": (
        "a real street photograph in daylight, with surfaces illuminated by the sun",
        "a daytime photograph of a building facade, with sunlight and daytime cast shadows",
        "a street photograph during the day, with bright natural light illuminating the scene",
    ),
    "overcast_daylight": (
        "a real street photograph on an overcast day, with surfaces illuminated by diffuse daylight",
        "a daytime photograph of a building facade, with even cloudy daylight and soft shadows",
        "a street photograph during the day, with grey natural daylight illuminating the scene",
    ),
    "day_for_night": (
        "a daylight street photograph artificially darkened to look like night, with daytime lighting still visible",
        "a photograph of a building facade with a dark sky but surfaces still illuminated as in daytime",
        "a street photograph with reduced brightness but preserved sunlight and daytime shadows",
    ),
}
CROP_FRACTIONS = {
    "full": (0.0, 0.0, 1.0, 1.0),
    "center_proxy": (0.125, 0.25, 0.875, 0.875),
    "lower_proxy": (0.0, 0.5, 1.0, 1.0),
}


def appearance_definition(model_name=None):
    return {
        "version": VERSION,
        "model": model_name or "openai/clip-vit-base-patch32",
        "prompt_ensemble": {key: list(value) for key, value in PROMPT_ENSEMBLE.items()},
        "crop_fractions_xyxy": {key: list(value) for key, value in CROP_FRACTIONS.items()},
        "crop_interpretation": "spatial proxies, not sky masks or semantic segmentation",
        "score": "mean over prompts of 100 * cosine(image_embedding, text_embedding)",
        "decision": "Unknown if any generated crop has night <= max(other class means); otherwise candidate",
        "photometric_thresholds": None,
        "calibrated": False,
        "interpretation": "ranking contradiction screen only; not calibrated probabilities or semantic proof",
        "content_audit_required": True,
        "precision": "FP32 model inputs, embeddings and scores; no_grad; autocast disabled",
    }


def crop_boxes(size):
    """Deterministic PIL-exclusive pixel boxes for any nonempty image size."""
    width, height = (int(value) for value in size)
    if width <= 0 or height <= 0:
        raise ValueError("Night appearance requires nonempty images")
    result = {}
    for name, (x0, y0, x1, y1) in CROP_FRACTIONS.items():
        left, top = min(width - 1, math.floor(x0 * width)), min(height - 1, math.floor(y0 * height))
        right, bottom = max(left + 1, math.ceil(x1 * width)), max(top + 1, math.ceil(y1 * height))
        result[name] = [left, top, min(width, right), min(height, bottom)]
    return result


def classify_scores(regions: Mapping[str, Mapping[str, float]]):
    """Classify generated crop class means; expose every contradicting region."""
    if set(regions) != set(CROP_FRACTIONS):
        raise ValueError("Expected exactly full, center_proxy and lower_proxy scores")
    reasons, margins = [], {}
    for region, scores in regions.items():
        if set(scores) != set(PROMPT_ENSEMBLE):
            raise ValueError("Missing or unexpected night appearance classes")
        if not all(math.isfinite(float(value)) for value in scores.values()):
            raise ValueError("Non-finite night appearance scores")
        competitor = max((name for name in scores if name != "night"), key=lambda name: scores[name])
        margin = float(scores["night"] - scores[competitor])
        margins[region] = margin
        if margin <= 0:
            reasons.append({"region": region, "competing_class": competitor,
                            "night_margin": margin,
                            "reason": "night_does_not_outrank_daylight_interpretations"})
    return {
        "appearance_status": "unknown" if reasons else "candidate",
        "appearance_contradiction": bool(reasons),
        "training_eligible": False,
        "content_audit_required": True,
        "night_margins": margins,
        "reasons": reasons,
    }


class NightAppearanceSignalV5:
    """Use the verifier's CLIP in batches, keeping only one pair in memory.

    Call verifier._ensure_weather() first if the evaluator is lazy. A loaded
    FP32 eval model is required; this class never changes its dtype or mode.
    The verifier's exact full-image cached embeddings are reused when present.
    """

    def __init__(self, evaluator, *, batch_size=6):
        if (getattr(evaluator, "mock", False) or evaluator.model is None
                or evaluator.processor is None):
            raise ValueError("Night appearance needs the verifier's loaded real CLIP evaluator")
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        if getattr(evaluator.model, "training", False):
            raise ValueError("Shared CLIP model must already be in eval mode")
        self.evaluator, self.batch_size = evaluator, int(batch_size)
        self.torch, self.functional = evaluator.torch, evaluator.functional
        for parameter in evaluator.model.parameters():
            if parameter.is_floating_point() and parameter.dtype != self.torch.float32:
                raise ValueError("Night appearance requires an FP32 shared CLIP model")
        config = getattr(evaluator.model, "config", None)
        name = (getattr(config, "_name_or_path", None)
                or getattr(evaluator.model, "name_or_path", None))
        self.definition = appearance_definition(name)
        self._texts, self._pair_cache = None, {}

    def _inputs(self, **kwargs):
        inputs = self.evaluator.processor(return_tensors="pt", **kwargs)
        return {key: value.to(device=self.evaluator.device,
                              dtype=self.torch.float32 if value.is_floating_point() else value.dtype)
                for key, value in inputs.items()}

    def _features(self, value, expected_count, kind):
        if not self.torch.is_tensor(value):
            value = next((getattr(value, name, None) for name in (kind + "_embeds", "pooler_output")
                          if self.torch.is_tensor(getattr(value, name, None))), None)
        if value is None or value.ndim != 2 or value.shape[0] != expected_count:
            raise ValueError(f"Invalid projected {kind} embeddings")
        dim = getattr(getattr(self.evaluator.model, "config", None), "projection_dim", None)
        if dim is not None and value.shape[1] != dim:
            raise ValueError(f"Invalid projected {kind} dimension")
        value = value.detach().float()
        if not bool(self.torch.isfinite(value).all()) or bool((value.norm(dim=-1) == 0).any()):
            raise ValueError(f"Non-finite or empty projected {kind} embeddings")
        return self.functional.normalize(value, dim=-1)

    def _encode(self, kind, **kwargs):
        inputs = self._inputs(**kwargs)
        device_type = self.torch.device(self.evaluator.device).type
        with self.torch.no_grad(), self.torch.autocast(device_type=device_type, enabled=False):
            value = getattr(self.evaluator.model, "get_" + kind + "_features")(**inputs)
        count = len(kwargs["images"] if kind == "image" else kwargs["text"])
        return self._features(value, count, kind)

    def _text_features(self):
        if self._texts is None:
            self._texts = self._encode("text", text=[text for texts in PROMPT_ENSEMBLE.values()
                                                    for text in texts], padding=True)
        return self._texts

    def _key(self, image):
        # Includes dimensions; the same bytes laid out differently are distinct.
        digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        return digest

    def _full_cached(self, image):
        evaluator = self.evaluator
        if not hasattr(evaluator, "_clip_cache_key"):
            return None
        key = evaluator._clip_cache_key(image)
        pair = getattr(evaluator, "_weather_pair", None)
        if pair is not None:
            for offset in (0, 1):
                if pair[offset] == key:
                    return self._features(pair[offset + 2], 1, "image")
        if key == getattr(evaluator, "_clip_reference_key", None):
            feature = getattr(evaluator, "_clip_reference_feature", None)
            if feature is not None:
                return self._features(feature, 1, "image")
        return None

    def measure(self, source: Image.Image, generated: Image.Image):
        images = {"source": source.convert("RGB"), "generated": generated.convert("RGB")}
        boxes = {side: crop_boxes(image.size) for side, image in images.items()}
        keys = {side: self._key(image) for side, image in images.items()}
        features, pending, pending_keys = {}, [], []
        for side, image in images.items():
            for region, box in boxes[side].items():
                key = (keys[side], region)
                if key in features:
                    continue
                value = self._pair_cache.get(key)
                if value is None and region == "full":
                    value = self._full_cached(image)
                if value is not None:
                    features[key] = value
                elif key not in pending_keys:
                    pending.append(image.crop(box))
                    pending_keys.append(key)
        for start in range(0, len(pending), self.batch_size):
            batch = self._encode("image", images=pending[start:start + self.batch_size])
            for key, value in zip(pending_keys[start:start + self.batch_size], batch):
                features[key] = value[None]
        self._pair_cache = features
        texts = self._text_features()
        regions, generated_scores = {}, {}
        with self.torch.no_grad(), self.torch.autocast(device_type=self.torch.device(self.evaluator.device).type,
                                                     enabled=False):
            for region in CROP_FRACTIONS:
                records = {}
                for side in images:
                    logits = (LOGIT_SCALE * features[(keys[side], region)].float() @ texts.float().T)[0]
                    if not bool(self.torch.isfinite(logits).all()):
                        raise ValueError("Non-finite night appearance scores")
                    offset, classes, prompts = 0, {}, {}
                    for name, ensemble in PROMPT_ENSEMBLE.items():
                        values = logits[offset:offset + len(ensemble)]
                        classes[name] = float(values.mean().item())
                        prompts[name] = [float(value) for value in values.tolist()]
                        offset += len(ensemble)
                    records[side] = {"image_size_wh": list(images[side].size), "crop_box_xyxy": boxes[side][region],
                                     "class_scores": classes, "prompt_scores": prompts}
                generated_scores[region] = records["generated"]["class_scores"]
                records["class_score_change"] = {
                    name: records["generated"]["class_scores"][name] - records["source"]["class_scores"][name]
                    for name in PROMPT_ENSEMBLE}
                source_scores = records["source"]["class_scores"]
                generated_values = records["generated"]["class_scores"]
                records["night_vs_daylight_margin_change"] = (
                    generated_values["night"] - max(value for key, value in generated_values.items() if key != "night")
                    - source_scores["night"] + max(value for key, value in source_scores.items() if key != "night"))
                regions[region] = records
        result = classify_scores(generated_scores)
        result.update(version=VERSION, regions=regions, definition=copy.deepcopy(self.definition),
                      definition_sha256=hashlib.sha256(json.dumps(self.definition, sort_keys=True,
                                                                 separators=(",", ":")).encode()).hexdigest())
        return result
