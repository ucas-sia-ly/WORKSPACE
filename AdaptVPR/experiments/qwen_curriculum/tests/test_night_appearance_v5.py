"""Deterministic CPU tests; no model downloads, generation or training."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from PIL import Image

ADAPTVPR_ROOT = Path(__file__).resolve().parents[3]
if str(ADAPTVPR_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTVPR_ROOT))

from experiments.qwen_curriculum.night_appearance_v5 import (
    CROP_FRACTIONS, PROMPT_ENSEMBLE, NightAppearanceSignalV5,
    appearance_definition, classify_scores, crop_boxes,
)


class Processor:
    def __call__(self, *, text=None, images=None, **kwargs):
        if text is not None:
            return {"input_ids": torch.stack([torch.eye(4)[index]
                    for index, prompts in enumerate(PROMPT_ENSEMBLE.values()) for _ in prompts])}
        pixels = []
        for image in images:
            values = torch.tensor(list(image.tobytes()), dtype=torch.float32).reshape(-1, 3).mean(0)
            pixels.append(torch.stack((values[2], values[0], values[1], values.new_zeros(()))))
        return {"pixel_values": torch.stack(pixels)}


class Model(torch.nn.Module):
    def __init__(self, return_wrapper=False):
        super().__init__()
        self.placeholder = torch.nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(projection_dim=4, _name_or_path="test/deterministic-clip")
        self.image_calls, self.text_calls, self.batch_sizes = 0, 0, []
        self.return_wrapper = return_wrapper
        self.eval()

    def get_image_features(self, pixel_values):
        assert not torch.is_grad_enabled()
        assert not torch.is_autocast_enabled("cpu")
        assert pixel_values.dtype == torch.float32
        self.image_calls += 1
        self.batch_sizes.append(len(pixel_values))
        value = pixel_values * self.placeholder
        return SimpleNamespace(image_embeds=value) if self.return_wrapper else value

    def get_text_features(self, input_ids):
        assert not torch.is_grad_enabled()
        assert not torch.is_autocast_enabled("cpu")
        self.text_calls += 1
        return SimpleNamespace(pooler_output=input_ids) if self.return_wrapper else input_ids


def evaluator(**kwargs):
    return SimpleNamespace(mock=False, model=Model(**kwargs), processor=Processor(),
                           torch=torch, functional=F, device="cpu")


class NightAppearanceTests(unittest.TestCase):
    def setUp(self):
        self.day = Image.new("RGB", (24, 32), (255, 0, 0))
        self.night = Image.new("RGB", (24, 32), (0, 0, 255))

    def test_all_night_is_only_candidate_with_required_content_audit(self):
        instance = evaluator()
        signal = NightAppearanceSignalV5(instance)
        result = signal.measure(self.day, self.night)
        self.assertEqual(result["appearance_status"], "candidate")
        self.assertFalse(result["training_eligible"])
        self.assertTrue(result["content_audit_required"])
        self.assertEqual(instance.model.batch_sizes, [6])
        self.assertIs(signal.evaluator.model, instance.model)
        for record in result["regions"].values():
            self.assertEqual(record["night_vs_daylight_margin_change"], 200)
            self.assertEqual(record["class_score_change"]["night"], 100)
        self.assertIsNone(instance.model.placeholder.grad)

    def test_daylight_lower_crop_blocks_full_image_night_candidate(self):
        generated = Image.new("RGB", (100, 100), (0, 0, 255))
        generated.paste((255, 0, 0), (0, 60, 100, 100))
        result = NightAppearanceSignalV5(evaluator()).measure(self.day, generated)
        self.assertGreater(result["night_margins"]["full"], 0)
        self.assertLess(result["night_margins"]["lower_proxy"], 0)
        self.assertEqual(result["appearance_status"], "unknown")
        self.assertTrue(any(reason["region"] == "lower_proxy" for reason in result["reasons"]))

    def test_no_hard_brightness_rule(self):
        signal = NightAppearanceSignalV5(evaluator())
        dim_day = signal.measure(self.day, Image.new("RGB", (4, 5), (1, 0, 0)))
        bright_night = signal.measure(self.day, self.night)
        self.assertEqual(dim_day["appearance_status"], "unknown")
        self.assertEqual(bright_night["appearance_status"], "candidate")
        self.assertIsNone(appearance_definition()["photometric_thresholds"])

    def test_ties_unknown_and_all_regions_required(self):
        scores = {region: {name: 0.0 for name in PROMPT_ENSEMBLE} for region in CROP_FRACTIONS}
        self.assertEqual(classify_scores(scores)["appearance_status"], "unknown")
        with self.assertRaises(ValueError):
            classify_scores({"full": scores["full"]})
        scores["full"]["night"] = float("nan")
        with self.assertRaises(ValueError):
            classify_scores(scores)

    def test_cache_bounded_reuses_source_and_invalidates_changed_pixels(self):
        instance = evaluator()
        signal = NightAppearanceSignalV5(instance)
        first = signal.measure(self.day, self.night)
        self.assertEqual(signal.measure(self.day.copy(), self.night.copy()), first)
        self.assertEqual(instance.model.image_calls, 1)
        self.assertEqual(instance.model.text_calls, 1)
        changed = self.night.copy()
        changed.putpixel((0, 0), (255, 0, 0))
        signal.measure(self.day, changed)
        self.assertEqual(instance.model.batch_sizes, [6, 3])
        self.assertEqual(len(signal._pair_cache), 6)

    def test_shared_evaluator_exact_full_image_features_reused(self):
        instance = evaluator()
        instance._clip_cache_key = lambda image: (image.size, image.tobytes())
        instance._weather_pair = (
            instance._clip_cache_key(self.day), instance._clip_cache_key(self.night),
            torch.tensor([[0., 1., 0., 0.]]), torch.tensor([[1., 0., 0., 0.]]))
        NightAppearanceSignalV5(instance).measure(self.day, self.night)
        self.assertEqual(instance.model.batch_sizes, [4])

    def test_identity_has_zero_shift_and_deduplicated_batch(self):
        instance = evaluator()
        result = NightAppearanceSignalV5(instance).measure(self.night, self.night.copy())
        self.assertEqual(instance.model.batch_sizes, [3])
        self.assertTrue(all(region["night_vs_daylight_margin_change"] == 0
                            for region in result["regions"].values()))

    def test_different_sizes_and_boundary_boxes_recorded(self):
        for size in ((1, 1), (2, 7), (31, 19)):
            boxes = crop_boxes(size)
            for left, top, right, bottom in boxes.values():
                self.assertTrue(0 <= left < right <= size[0])
                self.assertTrue(0 <= top < bottom <= size[1])
        result = NightAppearanceSignalV5(evaluator()).measure(self.day, self.night.resize((31, 19)))
        self.assertEqual(result["regions"]["full"]["source"]["image_size_wh"], [24, 32])
        self.assertEqual(result["regions"]["lower_proxy"]["generated"]["crop_box_xyxy"], [0, 9, 31, 19])
        self.assertIn("not sky masks", result["definition"]["crop_interpretation"])

    def test_fp32_no_grad_under_ambient_amp_and_projected_wrappers(self):
        instance = evaluator(return_wrapper=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = NightAppearanceSignalV5(instance, batch_size=2).measure(self.day, self.night)
        self.assertEqual(result["night_margins"]["full"], 100)
        self.assertEqual(instance.model.batch_sizes, [2, 2, 2])
        self.assertIsNone(instance.model.placeholder.grad)

    def test_nonfinite_or_zero_embeddings_fail_closed(self):
        for value in (float("nan"), 0.0):
            instance = evaluator()
            instance.model.get_image_features = lambda pixel_values: torch.full_like(pixel_values, value)
            with self.assertRaises(ValueError):
                NightAppearanceSignalV5(instance).measure(self.day, self.night)

    def test_shared_model_mode_and_precision_are_not_silently_mutated(self):
        instance = evaluator()
        instance.model.train()
        with self.assertRaises(ValueError):
            NightAppearanceSignalV5(instance)
        self.assertTrue(instance.model.training)
        instance.model.eval().half()
        with self.assertRaises(ValueError):
            NightAppearanceSignalV5(instance)
        self.assertEqual(instance.model.placeholder.dtype, torch.float16)

    def test_definition_independent_copies(self):
        original = copy.deepcopy(PROMPT_ENSEMBLE)
        definition = appearance_definition()
        definition["prompt_ensemble"]["night"] = ("changed",)
        self.assertEqual(PROMPT_ENSEMBLE, original)


if __name__ == "__main__":
    unittest.main()
