"""CPU regressions for embedding reuse, hash joins and source-disjoint selection."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from PIL import Image

ADAPTVPR_ROOT = Path(__file__).resolve().parents[3]
if str(ADAPTVPR_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTVPR_ROOT))

from experiments.generation_diagnosis import calibrate_weather_signal as calibration
from experiments.qwen_curriculum.common import file_sha256
from experiments.qwen_curriculum.weather_signal import (
    SIGNAL_DEFINITION, WEATHER_NEUTRAL_TEXT, WEATHER_TEXT, WeatherSignal,
    WeatherSignalEvaluator, weather_definition,
)
from verification.evaluator import DualTraitEvaluator


class Inputs(dict):
    def to(self, device):
        return self


class Processor:
    def __call__(self, *, images=None, text=None, **kwargs):
        if images is not None:
            return Inputs(pixel_values=torch.tensor([list(images.getpixel((0, 0)))], dtype=torch.float32))
        return Inputs(input_ids=torch.tensor([[1, 0, 0], [0, 0, 1]], dtype=torch.float32))


class Model:
    def __init__(self, return_form="tensor"):
        self.config = SimpleNamespace(projection_dim=3, _name_or_path="test/clip")
        self.image_calls, self.text_calls, self.return_form = 0, 0, return_form

    def get_image_features(self, pixel_values):
        self.image_calls += 1
        return pixel_values

    def get_text_features(self, input_ids):
        self.text_calls += 1
        if self.return_form == "tensor":
            return input_ids
        return SimpleNamespace(**{self.return_form: input_ids})


def evaluator(cls=WeatherSignalEvaluator, return_form="tensor"):
    instance = cls.__new__(cls)
    instance.mock, instance.device = False, "cpu"
    instance.model, instance.processor = Model(return_form), Processor()
    instance.torch, instance.functional = torch, F
    instance.clear_clip_cache()
    instance._compute_s_geo = lambda source, generated: 0.9
    return instance


class WeatherSignalTests(unittest.TestCase):
    def setUp(self):
        self.source = Image.new("RGB", (2, 2), (0, 0, 255))
        self.generated = Image.new("RGB", (2, 2), (255, 0, 0))

    def test_reuses_exact_evaluator_features_and_texts(self):
        verifier = evaluator()
        signal = WeatherSignal(verifier)
        score = verifier.evaluate(self.source, self.generated, route="global")
        measurement = signal.measure(self.source.copy(), self.generated.copy(), "rain")
        self.assertEqual(verifier.model.image_calls, 2)
        self.assertEqual(measurement["weather_source_logit"], -100)
        self.assertEqual(measurement["weather_generated_logit"], 100)
        self.assertEqual(measurement["weather_shift"], 200)
        self.assertEqual(score.s_div, 1)
        signal.measure(self.source, self.generated, "rain")
        self.assertEqual(verifier.model.text_calls, 1)
        self.assertEqual(signal.definition["weather_signal_model"], "test/clip")

    def test_changed_pixels_invalidate_generated_feature(self):
        verifier = evaluator()
        verifier.evaluate(self.source, self.generated, route="global")
        signal = WeatherSignal(verifier)
        self.generated.paste((0, 255, 0), (0, 0, 2, 2))
        measurement = signal.measure(self.source, self.generated, "fog")
        self.assertEqual(verifier.model.image_calls, 3)
        self.assertEqual(measurement["weather_generated_logit"], 0)
        signal.measure(self.source, self.generated, "snow")
        self.assertEqual(verifier.model.image_calls, 3)

    def test_reference_cache_reused_for_next_candidate(self):
        verifier = evaluator()
        signal = WeatherSignal(verifier)
        verifier.evaluate(self.source, self.generated, route="global")
        signal.measure(self.source, self.generated, "rain")
        alternative = Image.new("RGB", (2, 2), (0, 255, 0))
        verifier.evaluate(self.source, alternative, route="global")
        signal.measure(self.source, alternative, "rain")
        self.assertEqual(verifier.model.image_calls, 3)
        verifier.clear_clip_cache()
        self.assertIsNone(verifier._weather_pair)

    def test_plain_evaluator_shares_model_and_reference_cache(self):
        verifier = evaluator(DualTraitEvaluator)
        signal = WeatherSignal(verifier)
        verifier.evaluate(self.source, self.generated, route="global")
        signal.measure(self.source, self.generated, "night")
        self.assertIs(signal.evaluator.model, verifier.model)
        self.assertEqual(verifier.model.image_calls, 3)
        signal.measure(self.source, self.generated, "overcast")
        self.assertEqual(verifier.model.image_calls, 3)

    def test_identity_shift_and_all_conditions(self):
        verifier = evaluator()
        signal = WeatherSignal(verifier)
        for condition in WEATHER_TEXT:
            with self.subTest(condition=condition):
                self.assertEqual(signal.measure(self.source, self.source.copy(), condition)["weather_shift"], 0)
        self.assertEqual(verifier.model.image_calls, 1)
        self.assertEqual(verifier.model.text_calls, len(WEATHER_TEXT))

    def test_projected_text_return_forms(self):
        for return_form in ("tensor", "text_embeds", "pooler_output"):
            with self.subTest(return_form=return_form):
                measurement = WeatherSignal(evaluator(return_form=return_form)).measure(self.source, self.generated, "rain")
                self.assertEqual(measurement["weather_shift"], 200)

    def test_text_projection_dim_and_return_shape_are_checked(self):
        for value in (torch.ones(1, 3), torch.ones(2, 4), torch.full((2, 3), float("nan"))):
            verifier = evaluator()
            verifier.model.get_text_features = lambda **inputs: value
            with self.subTest(shape=tuple(value.shape)), self.assertRaises(ValueError):
                WeatherSignal(verifier).measure(self.source, self.generated, "rain")
        with self.assertRaises(TypeError):
            WeatherSignal(evaluator(return_form="last_hidden_state")).measure(self.source, self.generated, "rain")

    def test_mock_and_unknown_conditions_fail(self):
        verifier = evaluator()
        verifier.mock = True
        with self.assertRaisesRegex(ValueError, "real"):
            WeatherSignal(verifier)
        verifier.mock = False
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            WeatherSignal(verifier).measure(self.source, self.generated, "clear")
        self.assertEqual(verifier.model.text_calls, 0)

    def test_definition_is_independent_and_matches_legacy_texts(self):
        definition = weather_definition("test/clip")
        definition["weather_text"]["rain"] = "modified"
        self.assertEqual(WEATHER_TEXT["rain"], "a street photo in the rain with wet road")
        self.assertEqual(SIGNAL_DEFINITION["scale"], 100)
        self.assertEqual(WEATHER_NEUTRAL_TEXT, "a street photo on a clear sunny day")


def reviewed(source, label, shift, condition="rain"):
    return {"source_sha256": source, "output_sha256": f"{source}-{label}", "condition": condition,
            "weather_present": label, "metric_passed": True, "weather_shift": shift,
            "weather_source_logit": -3.0, "weather_generated_logit": shift - 3.0}


class CalibrationTests(unittest.TestCase):
    def test_auc_ties_and_single_class_are_explicit(self):
        self.assertEqual(calibration.auc([True, False], [1, 1]), 0.5)
        self.assertIsNone(calibration.auc([True, True], [1, 2]))

    def test_source_fold_threshold_uses_only_other_source_labels_and_scores(self):
        rows = [reviewed("a", "yes", 8), reviewed("a", "no", 2),
                reviewed("b", "weak", 7), reviewed("b", "no", 3),
                reviewed("c", "yes", 10), reviewed("c", "no", 1)]
        baseline = calibration.summarize(rows, [6])
        changed = [{**row, "weather_shift": -1000 if row["weather_present"] == "yes" else 1000}
                   if row["source_sha256"] == "a" else row for row in rows]
        alternative = calibration.summarize(changed, [6])
        fold = baseline["leave_one_source_out"]["folds"][0]
        changed_fold = alternative["leave_one_source_out"]["folds"][0]
        self.assertEqual(fold["held_out_source_sha256"], "a")
        self.assertEqual(fold["selected_threshold"], changed_fold["selected_threshold"])
        self.assertNotIn("a", fold["train_source_sha256"])
        self.assertEqual(baseline["leave_one_source_out"]["held_out_predictions_n"], len(rows))

    def test_strict_shift_threshold_and_weak_label_meanings(self):
        rows = [reviewed("a", "yes", 6), reviewed("b", "weak", 7), reviewed("c", "no", 5)]
        summary = calibration.group_summary(rows, [6])["thresholds"][0]
        self.assertEqual(summary["yes_or_weak_vs_no"]["tp"], 1)
        self.assertEqual(summary["yes_vs_weak_or_no"]["fp"], 1)
        self.assertEqual(summary["yes_or_weak_vs_no"]["keep_rate_by_label"]["yes"], 0)

    def test_review_hash_join_rejects_changed_actual_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            source, output = root / "source.png", run / "output.png"
            Image.new("RGB", (2, 2), "blue").save(source)
            Image.new("RGB", (2, 2), "red").save(output)
            config = {"sources": [{"src": 0, "source_path": str(source), "source_sha256": file_sha256(source)}]}
            config["fingerprint"] = calibration.fingerprint(config)
            (run / "generation_config.json").write_text(json.dumps(config))
            row = {"source_path": str(source), "output_path": str(output), "src": 0, "cond": "rain", "method": "test",
                   "seed": 1, "status": "ok", "passed": True, "config_fingerprint": config["fingerprint"],
                   "output_sha256": file_sha256(output)}
            (run / "results.jsonl").write_text(json.dumps(row) + "\n")
            review = {"source_path": str(source), "output_path": str(output), "source_index": 0, "condition": "rain",
                      "method": "test", "seed": 1, "run": "run", "metric_passed": True, "weather_present": "yes",
                      "config_fingerprint": config["fingerprint"], "output_sha256": file_sha256(output)}
            reviews = root / "reviews.jsonl"
            reviews.write_text(json.dumps(review) + "\n")
            joined, metadata = calibration.validate_reviews(reviews)
            self.assertEqual(joined[0]["source_sha256"], file_sha256(source))
            self.assertEqual(metadata["image_count"], 2)
            Image.new("RGB", (2, 2), "green").save(output)
            with self.assertRaisesRegex(ValueError, "checksum"):
                calibration.validate_reviews(reviews)

    def test_legacy_probe_order_and_provenance_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            probe = Path(directory) / "probe.json"
            reviews = [reviewed("a", "yes", 7), reviewed("b", "uncertain", 2)]
            probe.write_text(json.dumps([{"cond": "rain", "label": "yes", "passed": True, "src": -3, "gen": 4}]))
            rows, metadata = calibration.join_scores(reviews, legacy_probe=probe)
            self.assertEqual(len(rows), 1)
            self.assertIn("unverified", rows[0]["weather_score_provenance"])
            self.assertIn("cannot verify", metadata["limitation"])
            probe.write_text(json.dumps([{"cond": "rain", "label": "no", "passed": True, "src": -3, "gen": 4}]))
            with self.assertRaisesRegex(ValueError, "order"):
                calibration.join_scores(reviews, legacy_probe=probe)

    def test_historical_threshold_metadata_does_not_change_hash_cached_embeddings(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "scores.jsonl"
            row = reviewed("a", "yes", 7)
            definition = weather_definition()
            definition.update(exploratory_min_weather_shift=6.0,
                              threshold_status="exploratory small-sample IC-Light diagnosis")
            cached = {**row, "weather_signal_model": definition["weather_signal_model"],
                      "weather_signal_definition": definition,
                      "weather_score_provenance": "recomputed_on_hash_verified_images"}
            cache.write_text(json.dumps(cached) + "\n")
            scores, _ = calibration.join_scores([row], scores_path=cache)
            self.assertEqual(scores[0]["weather_shift"], 7)
            cached["weather_signal_definition"]["scale"] = 1.0
            cache.write_text(json.dumps(cached) + "\n")
            with self.assertRaisesRegex(ValueError, "definition differs"):
                calibration.join_scores([row], scores_path=cache)


if __name__ == "__main__":
    unittest.main()
