"""Experimental CLIP weather contrast, sharing the verifier's loaded model.

The fixed 100 * cosine contrast is a surrogate logit, not calibrated log odds
or a weather probability. The shift subtracts the source image's contrast.
Use WeatherSignalEvaluator.evaluate() before WeatherSignal.measure() to reuse
both image embeddings. No production evaluator or acceptance gate is changed.
"""

from __future__ import annotations

import copy
import math
import os
from typing import Any

from PIL import Image

from verification.evaluator import DualTraitEvaluator


WEATHER_TEXT = {
    "overcast": "a street photo on an overcast cloudy day",
    "rain": "a street photo in the rain with wet road",
    "snow": "a street photo in snow",
    "fog": "a street photo in thick fog",
    "night": "a street photo at night",
}
WEATHER_NEUTRAL_TEXT = "a street photo on a clear sunny day"
WEATHER_LOGIT_SCALE = 100.0
EXPLORATORY_MIN_WEATHER_SHIFT = 6.0
SIGNAL_DEFINITION = {
    "version": 1,
    "weather_signal_model": "openai/clip-vit-base-patch32",
    "weather_text": WEATHER_TEXT.copy(),
    "neutral_text": WEATHER_NEUTRAL_TEXT,
    "scale": WEATHER_LOGIT_SCALE,
    "formula": "scale * (cos(image, condition_text) - cos(image, neutral_text))",
    "shift_formula": "generated_logit - source_logit",
    "interpretation": "fixed-scale surrogate logits; not calibrated probabilities",
    "exploratory_min_weather_shift": EXPLORATORY_MIN_WEATHER_SHIFT,
    "threshold_status": "exploratory small-sample IC-Light diagnosis",
}


def weather_definition(clip_model_name: str | None = None) -> dict:
    """Return fingerprintable signal settings without importing/loading CLIP."""
    definition = copy.deepcopy(SIGNAL_DEFINITION)
    definition["weather_signal_model"] = clip_model_name or os.getenv(
        "ADAPTVPR_CLIP_MODEL_NAME", SIGNAL_DEFINITION["weather_signal_model"]
    )
    return definition


class WeatherSignalEvaluator(DualTraitEvaluator):
    """Capture the exact features produced by the unchanged verifier methods.

    The cache is one image pair, matched using decoded RGB SHA-256 keys rather
    than object identity or file names. Changing either image forces extraction.
    Like DualTraitEvaluator, one instance is intended for sequential use.
    """

    def _extract_clip_feature(self, image: Image.Image) -> Any:
        feature = super()._extract_clip_feature(image)
        self._weather_last_image_key = self._clip_cache_key(image)
        self._weather_last_image_feature = feature
        return feature

    def _compute_s_div(self, ref_image: Image.Image, gen_image: Image.Image) -> float:
        self._weather_pair = None
        score = super()._compute_s_div(ref_image, gen_image)
        source_key = self._clip_cache_key(ref_image)
        generated_key = self._clip_cache_key(gen_image)
        if (self._clip_reference_key == source_key
                and getattr(self, "_weather_last_image_key", None) == generated_key):
            self._weather_pair = (
                source_key, generated_key, self._clip_reference_feature,
                self._weather_last_image_feature,
            )
        return score

    def clear_clip_cache(self) -> None:
        super().clear_clip_cache()
        self._weather_pair = None
        self._weather_last_image_key = None
        self._weather_last_image_feature = None


class WeatherSignal:
    """Measure an uncalibrated weather shift with the evaluator's CLIP model.

    A plain DualTraitEvaluator is supported; its reference cache is reused, but
    its generated embedding must be extracted once because it is not exposed.
    Mock evaluators are rejected so synthetic signals cannot enter real runs.
    """

    def __init__(self, evaluator: DualTraitEvaluator):
        if evaluator.mock or evaluator.model is None or evaluator.processor is None:
            raise ValueError("WeatherSignal requires a real, loaded CLIP evaluator")
        self.evaluator = evaluator
        self._text_features: dict[str, Any] = {}
        self._generated_key = None
        self._generated_feature = None
        config = getattr(evaluator.model, "config", None)
        self.weather_signal_model = (
            getattr(config, "_name_or_path", None)
            or getattr(evaluator.model, "name_or_path", None)
            or weather_definition()["weather_signal_model"]
        )

    @property
    def definition(self) -> dict:
        return weather_definition(self.weather_signal_model)

    def _condition_features(self, condition: str) -> Any:
        if condition in self._text_features:
            return self._text_features[condition]
        evaluator = self.evaluator
        inputs = evaluator.processor(
            text=[WEATHER_TEXT[condition], WEATHER_NEUTRAL_TEXT],
            return_tensors="pt", padding=True,
        ).to(evaluator.device)
        with evaluator.torch.no_grad():
            feature = evaluator.model.get_text_features(**inputs)
        if not evaluator.torch.is_tensor(feature):
            # transformers versions return either Tensor or projected embeddings
            # wrapped in text_embeds/pooler_output, matching the image helper.
            candidates = (getattr(feature, "text_embeds", None),
                          getattr(feature, "pooler_output", None))
            feature = next((value for value in candidates
                            if evaluator.torch.is_tensor(value)), None)
            if feature is None:
                raise TypeError("CLIPModel.get_text_features() returned no supported projected embedding")
        if feature.ndim != 2 or feature.shape[0] != 2:
            raise ValueError(f"Expected two 2D CLIP text embeddings, got shape={tuple(feature.shape)}")
        projection_dim = getattr(getattr(evaluator.model, "config", None), "projection_dim", None)
        if projection_dim is not None and feature.shape[-1] != int(projection_dim):
            raise ValueError("CLIP text embedding dimension does not match model projection_dim")
        feature = evaluator.functional.normalize(feature.float(), dim=-1)
        if not bool(evaluator.torch.isfinite(feature).all()):
            raise ValueError("CLIP text embeddings contain non-finite values")
        self._text_features[condition] = feature
        return feature

    def _image_features(self, source: Image.Image, generated: Image.Image) -> tuple[Any, Any]:
        evaluator = self.evaluator
        source_key = evaluator._clip_cache_key(source)
        generated_key = evaluator._clip_cache_key(generated)
        pair = getattr(evaluator, "_weather_pair", None)
        if pair is not None and pair[:2] == (source_key, generated_key):
            return pair[2], pair[3]
        if source_key != evaluator._clip_reference_key or evaluator._clip_reference_feature is None:
            evaluator._clip_reference_feature = evaluator._extract_clip_feature(source)
            evaluator._clip_reference_key = source_key
        source_feature = evaluator._clip_reference_feature
        if source_key == generated_key:
            return source_feature, source_feature
        if generated_key != self._generated_key or self._generated_feature is None:
            self._generated_feature = evaluator._extract_clip_feature(generated)
            self._generated_key = generated_key
        return source_feature, self._generated_feature

    def measure(self, source: Image.Image, generated: Image.Image, condition: str) -> dict:
        if condition not in WEATHER_TEXT:
            raise ValueError(f"Unsupported weather condition {condition!r}; expected {sorted(WEATHER_TEXT)}")
        texts = self._condition_features(condition)
        source_feature, generated_feature = self._image_features(source, generated)

        def logit(feature):
            scores = (WEATHER_LOGIT_SCALE * feature @ texts.T)[0]
            value = float((scores[0] - scores[1]).item())
            if not math.isfinite(value):
                raise ValueError("CLIP weather contrast is not finite")
            return value

        source_logit, generated_logit = logit(source_feature), logit(generated_feature)
        return {
            "weather_source_logit": source_logit,
            "weather_generated_logit": generated_logit,
            "weather_shift": generated_logit - source_logit,
            "weather_signal_model": self.weather_signal_model,
        }
