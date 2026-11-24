"""Reliability OT opt-in initialization, portable inference, and CPU resume."""

import contextlib
import copy
import io
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import test_pretrained_initialization as fixtures
from train_salad import train
from workflow.model import (
    SALADModel, checkpoint_state_and_config, load_checkpoint_model,
    load_model_state, read_checkpoint, validate_model_config,
)


class ReliabilityWorkflowTests(unittest.TestCase):
    def setUp(self):
        # Keep all generated fixture data/checkpoints within the permitted tree.
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        with patch("test_pretrained_initialization.tempfile.TemporaryDirectory", return_value=temporary):
            fixtures.PretrainedInitializationTests.setUp(self)

    def enabled_config(self):
        config = copy.deepcopy(self.config)
        config["agg_config"].update(
            reliability_ot=True, reliability_lambda=0.7, reliability_hidden_dim=5,
        )
        return config

    def model(self, config):
        return SALADModel(config, pretrained_backbone=False, backbone_repo=self.repo)

    def arguments(self, output, *extra):
        return fixtures.PretrainedInitializationTests.arguments(self, output, *extra)

    def reliability_arguments(self, output, *extra):
        return self.arguments(
            output, "--reliability-ot", "--reliability-lambda", "0.7",
            "--reliability-head-lr", "1e-4", "--reliability-loss-weight", "0.1",
            "--reliability-real-prior-weight", "0.01",
            "--reliability-coverage-weight", "0.1",
            "--reliability-coverage-floor", "0.5", "--synthetic-places-only",
            *extra,
        )

    def test_legacy_opt_in_initialization_preserves_every_original_weight_and_descriptor(self):
        original = self.model(self.config).eval()
        load_model_state(original, self.state)
        enabled = self.model(self.enabled_config()).eval()
        load_model_state(enabled, self.state, allow_new_reliability=True)
        for name, value in self.state.items():
            torch.testing.assert_close(enabled.state_dict()[name], value, rtol=0, atol=0)
        images = torch.randn(1, 3, 28, 28)
        with torch.no_grad():
            old_descriptor = original(images)
            new_descriptor, auxiliary = enabled(images, return_aux=True)
        torch.testing.assert_close(new_descriptor, old_descriptor, rtol=1e-5, atol=1e-6)
        self.assertEqual(tuple(auxiliary["reliability"].shape), (1, 1, 2, 2))
        torch.testing.assert_close(auxiliary["reliability"], torch.full((1, 1, 2, 2), 0.9))

    def test_legacy_head_initialization_exception_remains_strict_for_original_weights(self):
        enabled = self.model(self.enabled_config())
        with self.assertRaisesRegex(RuntimeError, "reliability"):
            load_model_state(enabled, self.state)
        incomplete = {name: value for name, value in self.state.items() if name != "aggregator.dust_bin"}
        with self.assertRaisesRegex(RuntimeError, "every original SALAD weight"):
            load_model_state(enabled, incomplete, allow_new_reliability=True)
        partial = {
            **self.state, "aggregator.reliability_head.0.weight": enabled.aggregator.reliability_head[0].weight.detach(),
        }
        with self.assertRaisesRegex(RuntimeError, "Partial reliability weights"):
            load_model_state(enabled, partial, allow_new_reliability=True)
        with self.assertRaisesRegex(RuntimeError, "reliability"):
            load_model_state(self.model(self.config), enabled.state_dict())

    def test_native_and_raw_enabled_checkpoints_support_single_image_inference(self):
        config = self.enabled_config()
        model = self.model(config).eval()
        load_model_state(model, self.state, allow_new_reliability=True)
        # Persist a nonconstant, learned-looking head to detect accidental reset.
        with torch.no_grad():
            model.aggregator.reliability_head[-1].weight.fill_(0.02)
            model.aggregator.reliability_head[-1].bias.fill_(1.7)
        images = torch.randn(1, 3, 28, 28)
        with torch.no_grad():
            expected = model(images)
        for kind, payload in (("native", {"state_dict": model.state_dict(), "model_config": config}),
                              ("raw", model.state_dict())):
            with self.subTest(kind=kind):
                path = self.root / f"{kind}.pt"
                torch.save(payload, path)
                state, inferred = checkpoint_state_and_config(read_checkpoint(path))
                self.assertTrue(inferred["agg_config"]["reliability_ot"])
                self.assertEqual(inferred["agg_config"]["reliability_hidden_dim"], 5)
                self.assertAlmostEqual(inferred["agg_config"]["reliability_lambda"], 0.7, places=6)
                loaded = load_checkpoint_model(path, "cpu", backbone_repo=self.repo)
                with torch.no_grad():
                    actual = loaded(images)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(tuple(actual.shape), (1, 24))
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(state[name], value, rtol=0, atol=0)

    def test_raw_legacy_checkpoint_does_not_add_reliability_config_or_parameters(self):
        _, config = checkpoint_state_and_config(self.state)
        self.assertFalse(any(key.startswith("reliability_") for key in config["agg_config"]))
        model = self.model(self.config).eval()
        load_model_state(model, self.state)
        self.assertFalse(any("reliability" in key for key in model.state_dict()))
        with torch.no_grad():
            descriptor = model(torch.randn(1, 3, 28, 28))
        self.assertIsInstance(descriptor, torch.Tensor)

    def test_default_training_keeps_original_config_dataset_schema_and_optimizer(self):
        initial = self.root / "default_pretrained.pt"
        torch.save({"state_dict": self.state, "model_config": self.config}, initial)
        output = self.root / "default_training"
        with contextlib.redirect_stdout(io.StringIO()):
            train(self.arguments(output, "--init-checkpoint", str(initial)))
        checkpoint = read_checkpoint(output / "checkpoint.pt")
        self.assertEqual(checkpoint["model_config"], self.config)
        self.assertFalse(any("reliability" in key for key in checkpoint["state_dict"]))
        self.assertFalse(any(key.startswith("reliability") for key in checkpoint["training_config"]))
        self.assertNotIn("reliability_pairs", checkpoint["dataset_summary"])
        self.assertEqual(len(checkpoint["optimizer_state_dict"]["param_groups"]), 1)

    def test_malformed_head_strength_and_config_are_rejected(self):
        config = self.enabled_config()
        enabled = self.model(config)
        load_model_state(enabled, self.state, allow_new_reliability=True)
        state = enabled.state_dict()
        for strength in (float("nan"), float("inf"), -0.1):
            with self.subTest(strength=strength):
                broken = {**state, "aggregator.reliability_ot_lambda": torch.tensor(strength)}
                with self.assertRaisesRegex(ValueError, "valid head/strength"):
                    checkpoint_state_and_config(broken)
                bad_config = self.enabled_config()
                bad_config["agg_config"]["reliability_lambda"] = strength
                with self.assertRaisesRegex(ValueError, "reliability_lambda"):
                    validate_model_config(bad_config)
        for missing in ("aggregator.reliability_ot_lambda", "aggregator.reliability_head.0.weight"):
            with self.subTest(missing=missing):
                broken = {key: value for key, value in state.items() if key != missing}
                with self.assertRaises(ValueError):
                    checkpoint_state_and_config(broken)
        for key, value in (("reliability_ot", False), ("reliability_lambda", 0.9),
                           ("reliability_hidden_dim", 6)):
            with self.subTest(config_key=key):
                broken_config = self.enabled_config()
                broken_config["agg_config"][key] = value
                with self.assertRaisesRegex(ValueError, "weights and saved model configuration"):
                    checkpoint_state_and_config({"state_dict": state, "model_config": broken_config})

    def test_enabled_training_updates_head_and_resumes_exactly_without_original_checkpoint(self):
        initial = self.root / "pretrained.pt"
        torch.save({"state_dict": self.state, "model_config": self.config,
                    "optimizer_state_dict": {"invalid_for_optimizer": True}, "epoch": 99}, initial)
        output = self.root / "enabled_training"
        with contextlib.redirect_stdout(io.StringIO()):
            train(self.reliability_arguments(output, "--init-checkpoint", str(initial)))
        checkpoint = read_checkpoint(output / "checkpoint.pt")
        self.assertTrue(checkpoint["model_config"]["agg_config"]["reliability_ot"])
        self.assertEqual(checkpoint["epoch"], 2)
        self.assertEqual(checkpoint["global_step"], 2)
        self.assertTrue(checkpoint["dataset_summary"]["reliability_pairs"])
        saved_options = checkpoint["training_config"]["reliability"]
        for name, value in (("loss_weight", 0.1), ("real_prior_weight", 0.01),
                            ("coverage_weight", 0.1), ("coverage_floor", 0.5),
                            ("head_learning_rate", 1e-4)):
            self.assertEqual(saved_options[name], value)
        metrics = checkpoint["metrics"]["reliability"]
        for name, value in metrics.items():
            self.assertTrue(math.isfinite(value), name)
        self.assertEqual(metrics["paired_synthetic_exposure"], 2)
        self.assertGreater(metrics["mean_reliability"], 0)
        self.assertLessEqual(metrics["mean_reliability"], 1)
        for name, value in checkpoint["state_dict"].items():
            self.assertTrue(bool(torch.isfinite(value).all()), name)
        final_bias = checkpoint["state_dict"]["aggregator.reliability_head.2.bias"]
        self.assertFalse(torch.equal(final_bias, torch.full_like(final_bias, math.log(9.0))))
        groups = checkpoint["optimizer_state_dict"]["param_groups"]
        self.assertEqual(len(groups), 2)
        head_group = next(group for group in groups if math.isclose(group["initial_lr"], 1e-4))
        self.assertEqual(len(head_group["params"]), 4)
        for parameter in head_group["params"]:
            moments = checkpoint["optimizer_state_dict"]["state"][parameter]
            self.assertGreater(float(moments["step"]), 0)
            self.assertTrue(bool(torch.isfinite(moments["exp_avg"]).all()))
            self.assertTrue(bool(torch.isfinite(moments["exp_avg_sq"]).all()))
        initial.unlink()
        resumed_dir = self.root / "resumed"
        with contextlib.redirect_stdout(io.StringIO()):
            train(self.reliability_arguments(resumed_dir, "--resume", str(output / "checkpoint_epoch_001.pt")))
        resumed = read_checkpoint(resumed_dir / "checkpoint.pt")
        self.assertEqual(resumed["training_config"], checkpoint["training_config"])
        self.assertEqual(resumed["global_step"], checkpoint["global_step"])
        for name, value in checkpoint["state_dict"].items():
            torch.testing.assert_close(resumed["state_dict"][name], value, rtol=0, atol=0)
        self.assertEqual(resumed["optimizer_state_dict"]["param_groups"], groups)
        for parameter, values in checkpoint["optimizer_state_dict"]["state"].items():
            for name, value in values.items():
                torch.testing.assert_close(resumed["optimizer_state_dict"]["state"][parameter][name],
                                           value, rtol=0, atol=0)

    def test_resume_rejects_changed_reliability_options_and_missing_opt_in(self):
        initial = self.root / "pretrained.pt"
        torch.save({"state_dict": self.state, "model_config": self.config}, initial)
        output = self.root / "trained"
        with contextlib.redirect_stdout(io.StringIO()):
            train(self.reliability_arguments(output, "--init-checkpoint", str(initial)))
        resume = str(output / "checkpoint_epoch_001.pt")
        for flag, value in (("--reliability-lambda", "0.9"),
                            ("--reliability-head-lr", "0.0002"),
                            ("--reliability-loss-weight", "0.2"),
                            ("--reliability-real-prior-weight", "0.02"),
                            ("--reliability-coverage-weight", "0.2"),
                            ("--reliability-coverage-floor", "0.6")):
            with self.subTest(flag=flag), contextlib.redirect_stdout(io.StringIO()):
                args = self.reliability_arguments(self.root / flag[2:], "--resume", resume, flag, value)
                with self.assertRaisesRegex(ValueError, "Resume|reliability"):
                    train(args)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "reliability|Resume"):
                train(self.arguments(self.root / "disabled", "--resume", resume, "--synthetic-places-only"))


if __name__ == "__main__":
    unittest.main()
