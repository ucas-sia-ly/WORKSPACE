"""Quality checks for the v3 prompt bank, delegating to the frozen v2 verifier.

The v2 modules (quality.py, weather_signal.py) stay byte-identical because the
paused 23-city campaign freezes their hashes. This verifier changes:

* CLIP texts come from ``prompt_bank`` (new sun/overcast/rainy_night domains),
  with a per-domain neutral text instead of one shared "clear sunny day".
* Geometry limits are relaxed to s_geo >= 0.70 and fitted-H median displacement
  <= 32 px. v2 rejections at 0.78 / 16 px were mostly faithful edits that Qwen had
  slightly reframed or relit (lit lamps and headlights lower the inlier ratio).
  The recovered images also move SALAD descriptors more than v2-accepted ones
  (median drop: rain 0.037 vs 0.021, night 0.041 vs 0.035), so v2 was discarding
  the harder examples. Equalizing luminance before matching was tried and did not
  change outcomes (night outputs were equally dark whether accepted or rejected).
* Sun accepts on a CLIP shift >= 1.5 without a positive absolute contrast (see
  THRESHOLDS). Overcast keeps the v2 rule, so edits of already-grey sources,
  which barely change, are rejected.
* Mean luminance of source and output is recorded for difficulty analysis.
"""
from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

from PIL import Image, ImageStat

from experiments.qwen_curriculum import prompt_bank, quality, weather_signal

# Sun: GSV sources are often sunny already, so the absolute "harsh sunlight vs
# soft daylight" contrast stays near zero even for clear glare/shadow edits
# (pilot: 4 of 17 sun edits with visible shadows/flare failed only this check).
# Require a real shift instead of a positive absolute contrast.
THRESHOLDS = {"min_s_geo": 0.70, "max_H_grid_median_px_at_512": 32.0,
              "weather_thresholds": {"sun": {"min_shift": 1.5, "min_generated_logit": -100.0}}}


def mean_luminance(image: Image.Image) -> float:
    return float(ImageStat.Stat(image.convert("L")).mean[0])


class BankWeatherSignal(weather_signal.WeatherSignal):
    def _condition_features(self, condition):
        if condition not in self._text_features:
            with patch.object(weather_signal, "WEATHER_NEUTRAL_TEXT", prompt_bank.CLIP_NEUTRAL[condition]):
                return super()._condition_features(condition)
        return self._text_features[condition]

    @property
    def definition(self):
        return bank_signal_definition(self.weather_signal_model)


def bank_signal_definition(model_name=None):
    definition = weather_signal.weather_definition(model_name)
    definition.update(version="bank_v3", weather_text=dict(prompt_bank.CLIP_TEXT),
                      neutral_text=dict(prompt_bank.CLIP_NEUTRAL),
                      formula="scale * (cos(image, condition_text) - cos(image, neutral_text[condition]))")
    return definition


def _bank_texts():
    stack = ExitStack()
    for module in (quality, weather_signal):
        stack.enter_context(patch.object(module, "WEATHER_TEXT", dict(prompt_bank.CLIP_TEXT)))
    stack.enter_context(patch.object(quality, "WeatherSignal", BankWeatherSignal))
    defaults = {**quality.QUALITY_DEFAULTS, "weather_thresholds": {
        domain: {"min_shift": 0.0, "min_generated_logit": 0.0} for domain in prompt_bank.CLIP_TEXT}}
    stack.enter_context(patch.object(quality, "QUALITY_DEFAULTS", defaults))
    return stack


def quality_definition_v3(**kwargs) -> dict:
    kwargs.setdefault("thresholds", THRESHOLDS)
    with _bank_texts():
        definition = quality.quality_definition(**kwargs)
    definition.update(version="3", weather_signal=bank_signal_definition(kwargs.get("clip_model_name")),
                      prompt_bank=prompt_bank.POLICY_VERSION)
    return definition


class QwenQualityVerifierV3(quality.QwenQualityVerifier):
    def __init__(self, *args, thresholds=None, **kwargs):
        thresholds = THRESHOLDS if thresholds is None else thresholds
        with _bank_texts():
            super().__init__(*args, thresholds=thresholds, **kwargs)
        self.definition = quality_definition_v3(
            thresholds=thresholds, matcher_name=self.matcher_name, img_size=self.img_size,
            n_kpts=self.n_kpts, clip_device=self.clip_device, clip_model_name=self.clip_model_name)

    def evaluate(self, source: Image.Image, generated: Image.Image, condition: str) -> dict:
        with _bank_texts():
            result = super().evaluate(source, generated, condition)
        result.update(luminance_source=mean_luminance(source), luminance_generated=mean_luminance(generated),
                      quality_version="3")
        return result
