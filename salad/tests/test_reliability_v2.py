"""End-to-end contracts for cached, contextual reliability and its ablations."""

import contextlib
import copy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image

import test_pretrained_initialization as fixtures
from cache_reliability_targets import build_cache, parse_args as cache_arguments
from train_salad import parse_args, resolve_model_initialization, train
from workflow.model import (
    DINOBackbone, SALADModel, checkpoint_state_and_config, load_checkpoint_model,
    load_model_state, read_checkpoint, validate_model_config,
)
from workflow.reliability import reliability_losses
from workflow.reliability_cache import ReliabilityTargetCache, file_sha256


class ReliabilityV2Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        with patch("test_pretrained_initialization.tempfile.TemporaryDirectory", return_value=temporary):
            fixtures.PretrainedInitializationTests.setUp(self)
        self.config["image_size"] = [56, 56]
        # Copy/paste must actually change pixels, unlike a constant-color view.
        import numpy as np
        for position, path in enumerate(sorted(self.data.glob("Images/Test/*.jpg"))):
            generator = np.random.default_rng(100 + position)
            Image.fromarray(generator.integers(0, 256, (56, 56, 3), dtype=np.uint8)).save(path)
        for position, record in enumerate(self.records()):
            generator = np.random.default_rng(200 + position)
            Image.fromarray(generator.integers(0, 256, (56, 56, 3), dtype=np.uint8)).save(record["output_path"])
        self.initial = self.root / "pretrained.pt"
        torch.save({"state_dict": self.state, "model_config": self.config}, self.initial)

    def records(self):
        return [json.loads(line) for line in self.manifest.read_text().splitlines()]

    def enabled_config(self, **extra):
        config = copy.deepcopy(self.config)
        config["agg_config"].update(reliability_ot=True, reliability_lambda=0.7,
                                   reliability_hidden_dim=5, **extra)
        return config

    def arguments(self, output, *extra):
        return fixtures.PretrainedInitializationTests.arguments(
            self, output, "--image-size", "56", "56", "--synthetic-places-only",
            "--reliability-ot", "--reliability-lambda", "0.7",
            "--reliability-hidden-dim", "5", "--reliability-head-lr", "0.001", *extra)

    def run_training(self, output, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            train(self.arguments(output, *extra))
        return read_checkpoint(output / "checkpoint.pt")

    def build_targets(self):
        destination = self.root / "targets"
        args = cache_arguments([
            "--real-data", str(self.data), "--synthetic-manifest", str(self.manifest),
            "--checkpoint", str(self.initial), "--output-dir", str(destination),
            "--backbone-repo", str(self.repo), "--device", "cpu",
            "--image-size", "56", "56", "--batch-size", "2",
        ])
        with contextlib.redirect_stdout(io.StringIO()):
            index = build_cache(args)
        return destination, index

    def test_cached_context_corruption_training_updates_head_and_resumes_exactly(self):
        cache, index = self.build_targets()
        # The 4x4 fixture has no complete radius-2 structural neighborhood.
        # These weak labels stay unknown, without manufacturing negatives.
        self.assertEqual(index["summary"]["positive_patches"], 0)
        self.assertEqual(index["summary"]["negative_patches"], 0)
        self.assertEqual(index["summary"]["unknown_patches"], 32)
        extra = ["--reliability-context", "--reliability-detach-features",
                 "--reliability-target-cache", str(cache),
                 "--reliability-corruption-weight", "0.1",
                 "--reliability-corruption-max-images", "2"]
        output = self.root / "cached_training"
        calls = []
        original_forward = DINOBackbone.forward
        def capture_forward(model, images):
            calls.append(len(images))
            return original_forward(model, images)
        with patch.object(DINOBackbone, "forward", capture_forward), patch(
                "workflow.reliability.build_local_targets", side_effect=AssertionError("live teacher used")):
            checkpoint = self.run_training(output, "--init-checkpoint", str(self.initial), *extra)
        self.assertEqual(calls, [4, 2, 4, 2])  # Main views + known corruption copies; no source teacher.
        metrics = checkpoint["metrics"]["reliability"]
        self.assertEqual(metrics["negative_patches"], 0)
        self.assertEqual(metrics["unknown_patches"], 32)
        self.assertEqual(metrics["paired_synthetic_exposure"], 2)
        self.assertGreater(metrics["corruption_negative_patches"], 0)
        self.assertGreater(metrics["corruption_loss"], 0)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        aggregator = checkpoint["model_config"]["agg_config"]
        self.assertTrue(aggregator["reliability_context"])
        self.assertTrue(aggregator["reliability_detach_features"])
        options = checkpoint["training_config"]["reliability"]
        self.assertEqual(options["target_cache_integrity"], ReliabilityTargetCache(cache, (56, 56)).integrity_digest)
        head_group = checkpoint["optimizer_state_dict"]["param_groups"][1]
        self.assertEqual(len(head_group["params"]), 6)  # Both depthwise parameters are optimized.
        for parameter in head_group["params"]:
            moments = checkpoint["optimizer_state_dict"]["state"][parameter]
            self.assertEqual(float(moments["step"]), 2)
            self.assertTrue(bool(torch.isfinite(moments["exp_avg"]).all()))
        for parameter in head_group["params"][-2:]:
            self.assertGreater(float(checkpoint["optimizer_state_dict"]["state"][parameter]["exp_avg"].abs().sum()), 0)
        self.assertFalse(torch.equal(checkpoint["state_dict"]["aggregator.reliability_head.2.bias"],
                                     torch.full((1,), math.log(9.0))))
        self.assertGreater(float(checkpoint["state_dict"]["aggregator.reliability_head.2.weight"].abs().sum()), 0)
        self.initial.unlink()
        resumed = self.run_training(self.root / "resumed", "--resume",
                                    str(output / "checkpoint_epoch_001.pt"), *extra)
        self.assertEqual(checkpoint["training_config"], resumed["training_config"])
        self.assertEqual(checkpoint["optimizer_state_dict"]["param_groups"], resumed["optimizer_state_dict"]["param_groups"])
        self.assertEqual(resumed["global_step"], 2)
        for name, value in checkpoint["state_dict"].items():
            torch.testing.assert_close(resumed["state_dict"][name], value, rtol=0, atol=0)
        for parameter, moments in checkpoint["optimizer_state_dict"]["state"].items():
            for name, value in moments.items():
                torch.testing.assert_close(resumed["optimizer_state_dict"]["state"][parameter][name],
                                           value, rtol=0, atol=0)
        torch.testing.assert_close(resumed["rng_state"]["torch"], checkpoint["rng_state"]["torch"], rtol=0, atol=0)
        for flag in (("--no-reliability-context",), ("--no-reliability-detach-features",),
                     ("--reliability-corruption-weight", "0.2")):
            with self.subTest(changed_option=flag), self.assertRaisesRegex(ValueError, "Resume|reliability"):
                self.run_training(self.root / ("reject_" + flag[0][2:]), "--resume",
                                  str(output / "checkpoint_epoch_001.pt"), *extra, *flag)

    def test_raw_and_native_context_fixed_checkpoints_infer_and_predict_single_images(self):
        images = torch.randn(1, 3, 56, 56)
        for mode in ("learned", "fixed"):
            config = self.enabled_config(reliability_context=True, reliability_mode=mode,
                                         reliability_detach_features=True)
            model = SALADModel(config, pretrained_backbone=False, backbone_repo=self.repo).eval()
            load_model_state(model, self.state, allow_new_reliability=True)
            with torch.no_grad():
                model.aggregator.reliability_head[-1].weight.fill_(0.03)
                expected = model(images)
            for kind, payload in (("raw", model.state_dict()),
                                  ("native", {"state_dict": model.state_dict(), "model_config": config})):
                with self.subTest(mode=mode, kind=kind):
                    path = self.root / f"{mode}_{kind}.pt"
                    torch.save(payload, path)
                    _, inferred = checkpoint_state_and_config(read_checkpoint(path))
                    self.assertTrue(inferred["agg_config"]["reliability_context"])
                    self.assertEqual(inferred["agg_config"].get("reliability_mode", "learned"), mode)
                    loaded = load_checkpoint_model(path, "cpu", backbone_repo=self.repo)
                    with torch.no_grad():
                        actual = loaded(images)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(tuple(actual.shape), (1, 24))

    def test_context_input_detach_keeps_auxiliary_gradient_out_of_backbone(self):
        for detached in (False, True):
            config = self.enabled_config(reliability_context=True, reliability_detach_features=detached)
            model = SALADModel(config, pretrained_backbone=False, backbone_repo=self.repo)
            with torch.no_grad():
                model.aggregator.reliability_head[-1].weight.fill_(0.1)
                model.aggregator.reliability_context.weight.fill_(0.02)
                model.aggregator.reliability_context.bias.fill_(0.1)
            features = torch.randn(2, 384, 4, 4, requires_grad=True)
            logits = model.aggregator.predict_reliability(features)
            logits.square().mean().backward()
            if detached:
                self.assertIsNone(features.grad)
            else:
                self.assertGreater(float(features.grad.abs().sum()), 0)
            for parameter in model.aggregator.reliability_parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_task_only_and_fixed_ablation_skip_companion_teacher_and_fixed_never_updates_head(self):
        original_forward = DINOBackbone.forward
        for mode in ("learned", "fixed"):
            calls = []
            initial_predictor = {}
            original_initializer = SALADModel.__init__
            def capture_init(model, *args, **kwargs):
                original_initializer(model, *args, **kwargs)
                initial_predictor.update({name: value.clone() for name, value in model.state_dict().items()
                                          if "reliability_head" in name or "reliability_context" in name})
            def capture_forward(model, images):
                calls.append(len(images))
                return original_forward(model, images)
            options = ["--reliability-mode", mode, "--reliability-context",
                       "--reliability-loss-weight", "0", "--reliability-real-prior-weight", "0",
                       "--reliability-coverage-weight", "0", "--reliability-corruption-weight", "0"]
            with patch.object(SALADModel, "__init__", capture_init), patch.object(DINOBackbone, "forward", capture_forward):
                checkpoint = self.run_training(self.root / mode, "--init-checkpoint", str(self.initial), *options)
            self.assertEqual(calls, [4, 4])
            self.assertEqual(checkpoint["metrics"]["reliability"]["total"], 0)
            if mode == "fixed":
                for name, value in initial_predictor.items():
                    torch.testing.assert_close(checkpoint["state_dict"][name], value, rtol=0, atol=0)
                for parameter in checkpoint["optimizer_state_dict"]["param_groups"][1]["params"]:
                    self.assertNotIn(parameter, checkpoint["optimizer_state_dict"]["state"])

    def test_cached_full_and_compact_targets_are_detached_and_unknowns_produce_no_negative_loss(self):
        features = torch.randn(3, 384, 4, 4, requires_grad=True)
        logits = torch.zeros(3, 1, 4, 4, requires_grad=True)
        mask = torch.tensor([False, True, True])
        targets = torch.full_like(logits, 0.5)
        targets[1, :, :2] = 0.95
        targets[2, :, 2:] = 0.05
        targets.requires_grad_(True)
        confidence = (targets.detach() != 0.5).float().requires_grad_(True)
        options = dict(auxiliary_weight=1.0, real_prior_weight=0, coverage_weight=0)
        full = reliability_losses(logits, features, None, mask, mask,
                                  cached_targets=targets, cached_confidence=confidence, **options)
        compact = reliability_losses(logits, features, None, mask, mask,
                                     cached_targets=targets[mask], cached_confidence=confidence[mask], **options)
        for name, value in full.items():
            torch.testing.assert_close(value, compact[name], rtol=0, atol=0)
        self.assertEqual(float(full["positive_patches"]), 8)
        self.assertEqual(float(full["negative_patches"]), 8)
        full["total"].backward()
        self.assertIsNone(features.grad)
        self.assertIsNone(targets.grad)
        self.assertIsNone(confidence.grad)
        self.assertGreater(float(logits.grad.abs().sum()), 0)
        logits = torch.zeros_like(logits, requires_grad=True)
        unknown = reliability_losses(logits, features, None, mask, mask,
                                     cached_targets=torch.full_like(logits, 0.5),
                                     cached_confidence=torch.zeros_like(logits), **options)
        unknown["total"].backward()
        self.assertEqual(float(unknown["negative_patches"]), 0)
        self.assertEqual(float(unknown["unknown_patches"]), 32)
        self.assertEqual(float(logits.grad.abs().sum()), 0)

    def test_invalid_cached_loss_labels_fail_instead_of_becoming_supervision(self):
        logits = torch.zeros(2, 1, 4, 4, requires_grad=True)
        features = torch.randn(2, 384, 4, 4)
        mask = torch.tensor([False, True])
        for targets, confidence in (
                (torch.full((1, 1, 4, 4), float("nan")), torch.ones(1, 1, 4, 4)),
                (torch.full((1, 1, 4, 4), 0.5), torch.ones(1, 1, 4, 4)),
                (torch.full((1, 1, 4, 4), 0.95), torch.full((1, 1, 4, 4), -0.1)),
                (torch.ones(1, 1, 3, 4), torch.ones(1, 1, 3, 4))):
            with self.subTest(shape=targets.shape, first=float(targets.flatten()[0])):
                with self.assertRaisesRegex(ValueError, "cached"):
                    reliability_losses(logits, features, None, mask, mask,
                                       cached_targets=targets, cached_confidence=confidence)
        with self.assertRaisesRegex(ValueError, "together"):
            reliability_losses(logits, features, None, mask, mask, cached_targets=torch.ones_like(logits))

    def test_strict_context_mode_validation_and_legacy_schema_compatibility(self):
        for config in (self.config, self.enabled_config()):
            model = SALADModel(config, pretrained_backbone=False, backbone_repo=self.repo)
            payload = {"state_dict": model.state_dict(), "model_config": config}
            _, recovered = checkpoint_state_and_config(payload)
            self.assertEqual(recovered, config)
            self.assertNotIn("reliability_context", recovered["agg_config"])
            self.assertNotIn("reliability_mode", recovered["agg_config"])
        config = self.enabled_config(reliability_context=True, reliability_mode="fixed")
        model = SALADModel(config, pretrained_backbone=False, backbone_repo=self.repo)
        state = model.state_dict()
        for broken in ({name: value for name, value in state.items() if name != "aggregator.reliability_context.bias"},
                       {**state, "aggregator.reliability_fixed_mode": torch.tensor(False)},
                       {**state, "aggregator.reliability_fixed_mode": torch.tensor(1)}):
            with self.assertRaisesRegex(ValueError, "context|fixed"):
                checkpoint_state_and_config(broken)
        for name, value in (("reliability_context", False), ("reliability_mode", "learned")):
            broken_config = copy.deepcopy(config)
            broken_config["agg_config"][name] = value
            with self.assertRaisesRegex(ValueError, "context/mode"):
                checkpoint_state_and_config({"state_dict": state, "model_config": broken_config})
        malformed = self.enabled_config(reliability_mode="unrecognized")
        with self.assertRaisesRegex(ValueError, "reliability_mode"):
            validate_model_config(malformed)
        with self.assertRaisesRegex(ValueError, "auxiliary loss weights"):
            resolve_model_initialization(self.arguments(self.root / "bad_fixed", "--init-checkpoint",
                                                       str(self.initial), "--reliability-mode", "fixed"))

    def test_new_head_keeps_original_initialization_and_next_training_rng_draw(self):
        torch.manual_seed(91)
        original = SALADModel(self.config, pretrained_backbone=False, backbone_repo=self.repo)
        original_next_draw = torch.rand(32)
        torch.manual_seed(91)
        enabled = SALADModel(self.enabled_config(reliability_context=True), pretrained_backbone=False,
                             backbone_repo=self.repo)
        enabled_next_draw = torch.rand(32)
        for name, value in original.state_dict().items():
            torch.testing.assert_close(enabled.state_dict()[name], value, rtol=0, atol=0)
        torch.testing.assert_close(enabled_next_draw, original_next_draw, rtol=0, atol=0)

    def test_cache_rejects_mutated_payload_and_cli_requires_unaugmented_training(self):
        cache, _ = self.build_targets()
        payload = torch.load(cache / "targets.pt", weights_only=True)
        payload["confidence"].fill_(float("nan"))
        torch.save(payload, cache / "targets.pt")
        index = json.loads((cache / "index.json").read_text())
        index["tensor_sha256"] = file_sha256(cache / "targets.pt")
        (cache / "index.json").write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, "Invalid reliability cache confidence"):
            self.run_training(self.root / "bad_cache", "--init-checkpoint", str(self.initial),
                              "--reliability-target-cache", str(cache))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--real-data", str(self.data), "--output-dir", str(self.root / "bad_augment"),
                        "--reliability-ot", "--reliability-target-cache", str(cache)])


if __name__ == "__main__":
    unittest.main()
