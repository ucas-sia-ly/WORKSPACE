"""Fixed-ratio V2 cache/context/corruption integration without CUDA or DINO.

Only the backbone and its model construction are small CPU fixtures. Dataset
replacement, cache validation, SALAD, losses, optimizer, checkpoint handling and
the four experiment arms run through their production implementations.
"""

from __future__ import annotations

import contextlib
import copy
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

ADAPTVPR_ROOT = Path(__file__).resolve().parents[3]
if str(ADAPTVPR_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTVPR_ROOT))

from experiments.qwen_curriculum import common, test_700_ratio_experiment as experiment
from experiments.qwen_curriculum import fixed_reliability_cache
from experiments.qwen_curriculum.tests import test_fixed_reliability_experiment as fixtures

common.use_salad()
from models.aggregators.salad import SALAD
from workflow.model import default_model_config
from workflow.reliability_cache import (
    IMAGE_PREPROCESSING, TARGET_ALGORITHM, ReliabilityTargetCache,
)


class FakeBackbone(fixtures.FakeBackbone):
    """Preserve a 16x16 grid matching the real 224px/14px cache contract."""

    def forward(self, images):
        features = self.projection(F.adaptive_avg_pool2d(images, (16, 16)))
        return features, features.mean(dim=(-2, -1))


class FakeSALADModel(nn.Module):
    def __init__(self, config, pretrained_backbone=True, backbone_repo=None,
                 backbone_weights=None):
        super().__init__()
        self.config = config
        self.image_size = tuple(config["image_size"])
        self.backbone = FakeBackbone()
        if pretrained_backbone:
            self.backbone.load_state_dict(torch.load(backbone_weights, weights_only=True))
        self.aggregator = SALAD(**config["agg_config"], dropout=0)

    def forward(self, images, return_aux=False):
        return self.aggregator(self.backbone(images), return_aux=return_aux)


class FixedReliabilityV2ExperimentTests(unittest.TestCase):
    def setUp(self):
        fixtures.FixedReliabilityExperimentTests.setUp(self)
        self.cache_path = self.root / "targets"
        self.cache_path.mkdir()
        # Preserve the same source slots while making rejection explicit. The
        # fixed experiment uses all physical outputs, including rejected ones.
        for index, row in enumerate(self.rows):
            row.update(sample_id=f"fixture_{index}", passed=index != 1,
                       eligible_for_training=index != 1,
                       status="rejected" if index == 1 else "passed",
                       source_sha256=common.file_sha256(Path(row["source_path"])),
                       generated_sha256=common.file_sha256(Path(row["output_path"])))
            row["output_sha256"] = row["generated_sha256"]
        common.write_jsonl(self.root / "generated_700.jsonl", self.rows)
        targets = torch.full((4, 1, 16, 16), 0.5)
        confidence = torch.zeros_like(targets)
        # Positives have nonuniform support; unknown cells stay unlabelled.
        for index in range(4):
            targets[index, :, 2:10 + index, 3:12] = 0.95
            confidence[index, :, 2:10 + index, 3:12] = 0.8
        torch.save({"targets": targets, "confidence": confidence}, self.cache_path / "targets.pt")
        index = {
            "format_version": 1, "target_algorithm": TARGET_ALGORITHM,
            "image_preprocessing": IMAGE_PREPROCESSING, "patch_stride": 14,
            "image_size": [224, 224], "tensor_file": "targets.pt",
            "teacher_checkpoint": str(self.backbone_weights),
            "teacher_checkpoint_sha256": common.file_sha256(self.backbone_weights),
            "teacher_model_config": fixed_reliability_cache.TEACHER_MODEL_CONFIG,
            "manifest": str(self.root / "generated_700.jsonl"),
            "manifest_sha256": common.file_sha256(self.root / "generated_700.jsonl"),
            "tensor_sha256": common.file_sha256(self.cache_path / "targets.pt"),
            "entries": [
                {"source_path": row["source_path"], "output_path": row["output_path"],
                 "source_sha256": row["source_sha256"],
                 "output_sha256": row["generated_sha256"]}
                for row in self.rows
            ],
        }
        common.write_json(self.cache_path / "index.json", index)
        self.targets, self.confidence = targets, confidence
        self.cache_index = index
        self.cache = ReliabilityTargetCache(self.cache_path, (224, 224))
        self.args.reliability_version = 2
        self.args.reliability_context = True
        self.args.reliability_detach_features = True
        self.args.reliability_mode = "learned"
        self.args.reliability_target_cache = self.cache_path
        self.args.reliability_corruption_weight = 0.05
        self.args.reliability_corruption_max_images = 4
        self.config.pop("fingerprint")
        self.config["request"] = {"num_images": 4, "backbone_weights": str(self.backbone_weights)}
        self.config["reliability"].update(
            version=2, context=True, detach_features=True, mode="learned",
            target_cache=str(self.cache_path), corruption_weight=0.05,
            corruption_max_images=4,
        )
        self.config["files_sha256"]["generated_700.jsonl"] = index["manifest_sha256"]
        self.config = experiment.seal(self.config)

    image_tensor = staticmethod(fixtures.FixedReliabilityExperimentTests.image_tensor)

    def model_config(self, enabled=True):
        config = default_model_config(experiment.BACKBONE, (224, 224), 4, 2, 8, 8)
        if enabled:
            config["agg_config"].update(
                reliability_ot=True, reliability_lambda=2.0, reliability_hidden_dim=64,
                reliability_context=True, reliability_detach_features=True,
                reliability_mode="learned",
            )
        return config

    def resolve_initialization(self, args, resume=None):
        return self.model_config(), None, {"init_checkpoint": None, "init_checkpoint_sha256": None}

    def test_cached_targets_keep_exact_nested_source_slots_and_rejected_output(self):
        self.assertEqual(self.groups[4], self.groups[8][:5])
        generated_sources = {row["source_path"] for row in self.rows}
        for ratio in (8, 4):
            real = experiment.FixedPairedDataset(self.root, self.config, f"true_{ratio}to1")
            generated = experiment.FixedPairedDataset(self.root, self.config, f"generated_{ratio}to1")
            self.assertIsInstance(generated.reliability_target_cache, ReliabilityTargetCache)
            sources, real_count, generated_count, rejected_count = [], 0, 0, 0
            for group_index, bag in enumerate(self.groups[ratio]):
                baseline_images, baseline_labels, baseline_flags, baseline_targets, baseline_confidence = real[group_index]
                decoded = []
                original_open = Image.open
                def capture_open(path, *args, **kwargs):
                    decoded.append(str(Path(path)))
                    return original_open(path, *args, **kwargs)
                with patch("PIL.Image.open", side_effect=capture_open):
                    images, labels, flags, targets, confidence = generated[group_index]
                self.assertEqual(decoded, [path for path, _ in generated.selected(group_index)])
                self.assertEqual(len(decoded), 4)
                self.assertTrue(torch.equal(baseline_labels, labels))
                self.assertFalse(bool(baseline_flags.any()))
                self.assertTrue(torch.equal(baseline_targets, torch.full_like(baseline_targets, 0.5)))
                self.assertEqual(float(baseline_confidence.sum()), 0)
                self.assertEqual(tuple(targets.shape), (4, 1, 16, 16))
                self.assertEqual(tuple(confidence.shape), (4, 1, 16, 16))
                for slot, source in enumerate(bag["sources"]):
                    sources.append(source)
                    torch.testing.assert_close(baseline_images[slot], self.image_tensor(source), rtol=0, atol=0)
                    if source in generated_sources:
                        self.assertTrue(bool(flags[slot]))
                        self.assertNotIn(source, decoded)
                        row = generated.generated[source]
                        expected_target, expected_confidence = self.cache.fetch(row["output_path"], source)
                        torch.testing.assert_close(targets[slot], expected_target, rtol=0, atol=0)
                        torch.testing.assert_close(confidence[slot], expected_confidence, rtol=0, atol=0)
                        torch.testing.assert_close(images[slot], self.image_tensor(row["output_path"]), rtol=0, atol=0)
                        rejected_count += int(not row["passed"])
                    else:
                        self.assertFalse(bool(flags[slot]))
                        torch.testing.assert_close(images[slot], baseline_images[slot], rtol=0, atol=0)
                        self.assertTrue(torch.equal(targets[slot], torch.full_like(targets[slot], 0.5)))
                        self.assertEqual(float(confidence[slot].sum()), 0)
                generated_count += int(flags.sum())
                real_count += flags.numel() - int(flags.sum())
            self.assertEqual((real_count, generated_count, rejected_count), (4 * ratio, 4, 1))
            self.assertEqual(len(sources), 4 * (ratio + 1))
            self.assertEqual(len(sources), len(set(sources)))

    def test_dataset_rejects_manifest_change_and_modified_generated_bytes(self):
        manifest = self.root / "generated_700.jsonl"
        original_manifest = manifest.read_bytes()
        manifest.write_bytes(original_manifest + b"\n")
        with self.assertRaisesRegex(ValueError, "manifest"):
            experiment.FixedPairedDataset(self.root, self.config, "generated_8to1")
        manifest.write_bytes(original_manifest)
        for name in ("output_path", "source_path"):
            with self.subTest(changed_image=name):
                image = Path(self.rows[0][name])
                original_image = image.read_bytes()
                image.write_bytes(original_image + b"changed image bytes")
                with self.assertRaisesRegex(ValueError, "image content changed|image hash"):
                    experiment.FixedPairedDataset(self.root, self.config, "generated_8to1")
                image.write_bytes(original_image)

    def test_dataset_rejects_missing_cache_pair_and_altered_tensor(self):
        payload = {"targets": self.targets[:3], "confidence": self.confidence[:3]}
        torch.save(payload, self.cache_path / "targets.pt")
        index = copy.deepcopy(self.cache_index)
        index["entries"] = index["entries"][:3]
        index["tensor_sha256"] = common.file_sha256(self.cache_path / "targets.pt")
        common.write_json(self.cache_path / "index.json", index)
        with self.assertRaisesRegex(ValueError, "no cached|missing|pair"):
            experiment.FixedPairedDataset(self.root, self.config, "generated_8to1")
        torch.save({"targets": self.targets, "confidence": self.confidence}, self.cache_path / "targets.pt")
        index = copy.deepcopy(self.cache_index)
        index["tensor_sha256"] = common.file_sha256(self.cache_path / "targets.pt")
        common.write_json(self.cache_path / "index.json", index)
        tensor_path = self.cache_path / "targets.pt"
        tensor_path.write_bytes(tensor_path.read_bytes() + b"tampered tensor")
        with self.assertRaisesRegex(ValueError, "tensor hash mismatch"):
            experiment.FixedPairedDataset(self.root, self.config, "generated_8to1")

    def test_production_train_arm_requires_published_cache_binding(self):
        self.args.command = "train-arm"
        self.args.arm = "generated_8to1"
        # An internally valid cache may be inspected before its publication
        # binding; it is still forbidden as a production training input.
        descriptor = fixed_reliability_cache.validate_cache(self.config, self.root)
        self.assertEqual(descriptor["source_pair_count"], 4)
        with patch.object(experiment, "parse_args", return_value=self.args), patch.object(
                experiment, "prepare", return_value=self.config), patch.object(experiment, "train_arm") as trainer:
            with self.assertRaisesRegex(FileNotFoundError, "cache_binding"):
                experiment.main([])
            trainer.assert_not_called()
        common.write_json(self.root / "cache_binding.json", descriptor)
        validator = fixed_reliability_cache.validate_cache
        with patch.object(experiment, "parse_args", return_value=self.args), patch.object(
                experiment, "prepare", return_value=self.config), patch.object(experiment, "train_arm") as trainer, patch.object(
                fixed_reliability_cache, "validate_cache", wraps=validator) as checked:
            experiment.main([])
            checked.assert_called_once_with(self.config, self.root, required=True)
            trainer.assert_called_once_with(self.args, self.config)

    def test_actual_four_cpu_trainers_update_context_and_preserve_main_exposure(self):
        torch.manual_seed(self.args.seed)
        standard = FakeSALADModel(self.model_config(False), backbone_weights=self.backbone_weights)
        standard_sha = experiment.tensor_state_sha256(standard.aggregator.state_dict())
        torch.manual_seed(self.args.seed)
        fresh = FakeSALADModel(self.model_config(), backbone_weights=self.backbone_weights)
        initial_context = {name: parameter.detach().clone() for name, parameter
                           in fresh.aggregator.reliability_context.named_parameters()}
        initial_head = {name: parameter.detach().clone() for name, parameter
                        in fresh.aggregator.reliability_head.named_parameters()}
        original_state = {name: value for name, value in fresh.aggregator.state_dict().items()
                          if not name.startswith("reliability_")}
        self.assertEqual(experiment.tensor_state_sha256(original_state), standard_sha)
        initializations = []
        # At least two optimizer steps are needed: the zero-initialized final
        # head first opens the gradient path into the neighborhood branch.
        original_threads = torch.set_num_threads
        def cpu_threads(count):
            original_threads(min(count, 2))
        with patch("workflow.model.SALADModel", FakeSALADModel), patch(
                "train_salad.resolve_model_initialization", side_effect=self.resolve_initialization), patch(
                "workflow.reliability.build_local_targets", side_effect=AssertionError("live teacher used")), patch(
                "torch.set_num_threads", side_effect=cpu_threads), contextlib.redirect_stdout(io.StringIO()):
            for arm, (ratio, replace) in experiment.ARMS.items():
                self.args.arm = arm
                forwards = []
                original_forward = FakeBackbone.forward
                def capture_forward(backbone, images):
                    forwards.append(len(images))
                    return original_forward(backbone, images)
                with patch.object(FakeBackbone, "forward", capture_forward):
                    experiment.train_arm(self.args, self.config)
                # Clean metric views and capped corruption copies are the
                # only backbone forwards. Exact sources are never reencoded.
                self.assertEqual(forwards, [12, 4, 12, 4, 12, 4] if ratio == 8 else [12, 4, 8, 4])
                saved = experiment.verify_checkpoint(self.args, self.config, arm)
                experiment.check_exposure(self.args, self.config, arm)
                self.assertEqual(saved["global_step"], 3 if ratio == 8 else 2)
                self.assertEqual(saved["epoch"], 1)
                training = saved["training_config"]
                self.assertIsNone(training["init_checkpoint"])
                self.assertEqual(training["synthetic_manifest"], str(self.root / "generated_700.jsonl"))
                self.assertEqual(training["reliability"]["teacher"], "offline_frozen_local_self_similarity_v1")
                self.assertEqual(training["reliability"]["target_cache"], str(self.cache_path))
                self.assertEqual(training["reliability"]["target_cache_integrity"], self.cache.integrity_digest)
                self.assertEqual(training["reliability"]["controlled_corruption"], {
                    "weight": 0.05, "max_images": 4,
                    "method": "real_only_local_copy_paste_v1", "rng": "independent_per_step_generator_v1",
                })
                aggregator = saved["model_config"]["agg_config"]
                for key, expected in {"reliability_ot": True, "reliability_context": True,
                                      "reliability_detach_features": True, "reliability_mode": "learned"}.items():
                    self.assertEqual(aggregator[key], expected)
                metrics = saved["metrics"]
                self.assertEqual((metrics["real_exposure"], metrics["synthetic_exposure"]),
                                 (4 * ratio, 4) if replace else (4 * (ratio + 1), 0))
                reliability = metrics["reliability"]
                self.assertEqual(reliability["paired_synthetic_exposure"], 4 if replace else 0)
                self.assertEqual(reliability["negative_patches"], 0)
                self.assertGreater(reliability["corruption_negative_patches"], 0)
                self.assertGreater(reliability["corruption_loss"], 0)
                self.assertTrue(all(np.isfinite(value) for value in reliability.values()))
                self.assertTrue(np.isfinite(metrics["loss"]))
                if replace:
                    self.assertGreater(reliability["positive_patches"], 0)
                    self.assertGreater(reliability["unknown_patches"], 0)
                else:
                    self.assertEqual(reliability["positive_patches"], 0)
                self.assertTrue(any(not torch.equal(saved["state_dict"][f"aggregator.reliability_context.{name}"], value)
                                    for name, value in initial_context.items()))
                self.assertTrue(any(not torch.equal(saved["state_dict"][f"aggregator.reliability_head.{name}"], value)
                                    for name, value in initial_head.items()))
                optimizer = saved["optimizer_state_dict"]
                head_ids = optimizer["param_groups"][1]["params"]
                self.assertEqual(len(head_ids), 6)
                for parameter in head_ids[-2:]:
                    moments = optimizer["state"][parameter]
                    self.assertEqual(float(moments["step"]), saved["global_step"])
                    self.assertGreater(float(moments["exp_avg"].abs().sum()), 0)
                initializations.append(experiment.verify_initialization(self.args, self.config, arm))
                self.assertEqual(initializations[-1]["standard_aggregator_state_sha256"], standard_sha)
                loaded = FakeSALADModel(saved["model_config"], pretrained_backbone=False)
                loaded.load_state_dict(saved["state_dict"], strict=True)
                loaded.eval()
                image = experiment.FixedPairedDataset(self.root, self.config, arm)[0][0][0:1]
                with torch.no_grad():
                    descriptor = loaded(image)
                self.assertEqual(tuple(descriptor.shape), (1, 24))
                torch.testing.assert_close(descriptor.norm(dim=1), torch.ones(1), rtol=1e-5, atol=1e-5)
                checkpoint = self.root / arm / "checkpoint.pt"
                original_hash = common.file_sha256(checkpoint)
                experiment.train_arm(self.args, self.config)
                self.assertEqual(common.file_sha256(checkpoint), original_hash)
                # Reject silently changed offline-teacher metadata on resume.
                damaged = copy.deepcopy(saved)
                damaged["training_config"]["reliability"]["target_cache_integrity"] = "0" * 64
                torch.save(damaged, checkpoint)
                with self.assertRaisesRegex(ValueError, "module|reliability|cache"):
                    experiment.verify_checkpoint(self.args, self.config, arm)
                torch.save(saved, checkpoint)
        for key in ("initial_backbone_state_sha256", "initial_aggregator_state_sha256",
                    "standard_aggregator_state_sha256"):
            self.assertEqual(len({row[key] for row in initializations}), 1)
        shared = experiment.check_shared_initialization(self.args, self.config)
        self.assertEqual(shared["standard_aggregator_state_sha256"], standard_sha)
        self.assertEqual(shared["initial_backbone_state_sha256"], experiment.tensor_state_sha256(
            torch.load(self.backbone_weights, weights_only=True)))


if __name__ == "__main__":
    unittest.main()
