"""CPU integration coverage for fixed source replacement with reliability OT.

Only the DINO backbone is replaced by a small local fixture. The actual fixed
dataset, SALAD reliability head, weak loss, training loop and checkpoint logic
remain in use; no model download, generation request or CUDA work is needed.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
from pathlib import Path
import sys
import tempfile
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

common.use_salad()
from models.aggregators.salad import SALAD
from workflow.model import default_model_config


RELIABILITY = {
    "enabled": True, "lambda": 2.0, "hidden_dim": 64,
    "head_learning_rate": 6e-5, "loss_weight": 0.1,
    "real_prior_weight": 0.01, "coverage_weight": 0.1,
    "coverage_floor": 0.5,
}


class FakeBackbone(nn.Module):
    """A spatial RGB projection retaining enough structure for real weak BCE."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 768, 1)

    def forward(self, images):
        features = self.projection(F.adaptive_avg_pool2d(images, (8, 8)))
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


class FixedReliabilityExperimentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, self.previous_threads)
        self.real = self.root / "real"
        (self.real / "Dataframes").mkdir(parents=True)
        for city in experiment.CITIES:
            (self.real / "Dataframes" / f"{city}.csv").write_text("fixture metadata\n")
        images = self.root / "images"
        images.mkdir()
        self.places = {}
        self.rows = []
        for place in range(12):
            paths = []
            for view in range(8):
                # Random spatial structure makes the companion check meaningful
                # and yields structural positive anchors in the actual weak loss.
                rng = np.random.default_rng(1000 + place * 8 + view)
                pixels = rng.integers(25, 180, (32, 32, 3), dtype=np.uint8)
                source = images / f"source_{place}_{view}.png"
                Image.fromarray(pixels).save(source)
                paths.append(str(source))
                if place < 4 and view == 0:
                    output = images / f"generated_{place}.png"
                    Image.fromarray((pixels.astype(np.float32) * 0.7 + 35).astype(np.uint8)).save(output)
                    self.rows.append({"city": experiment.CITIES[0], "place_id": place,
                                      "source_path": str(source), "output_path": str(output)})
            self.places[(experiment.CITIES[0], place)] = paths
        self.groups = experiment.build_groups(self.places, self.rows, 4, 42)
        for ratio, bags in self.groups.items():
            for index, bag in enumerate(bags):
                bag.update(group_id=index, label=bag["place_id"])
            common.write_jsonl(self.root / f"groups_{ratio}to1.jsonl", bags)
        common.write_jsonl(self.root / "generated_700.jsonl", self.rows)
        inventory = [{"path": path, "sha256": common.file_sha256(Path(path)), "kind": "source"}
                     for path in sorted({p for bag in self.groups[8] for p in bag["sources"]})]
        inventory += [{"path": row["output_path"], "sha256": common.file_sha256(Path(row["output_path"])),
                       "kind": "generated"} for row in self.rows]
        common.write_jsonl(self.root / "image_inventory.jsonl", inventory)
        self.backbone_weights = self.root / "backbone_fixture.pt"
        torch.manual_seed(73)
        torch.save(FakeBackbone().state_dict(), self.backbone_weights)
        self.args = argparse.Namespace(
            output_dir=self.root, real_data=self.real, backbone_weights=self.backbone_weights,
            backbone_repo=self.root, epochs=1, batch_size=3, trainable_blocks=4,
            learning_rate=6e-5, seed=42, device="cpu", arm=None,
            reliability_ot=True, reliability_lambda=2.0, reliability_hidden_dim=64,
            reliability_head_lr=6e-5, reliability_loss_weight=0.1,
            reliability_real_prior_weight=0.01, reliability_coverage_weight=0.1,
            reliability_coverage_floor=0.5, baseline_output_dir=None,
        )
        self.config = experiment.seal({
            "backbone_weights_sha256": common.file_sha256(self.backbone_weights),
            "reliability": copy.deepcopy(RELIABILITY),
            "arms": {
                arm: {"ratio": ratio, "replace": replace, "source_slots": 4 * (ratio + 1),
                      "true_per_epoch": 4 * (ratio if replace else ratio + 1),
                      "generated_per_epoch": 4 if replace else 0,
                      "schedule": f"groups_{ratio}to1.jsonl"}
                for arm, (ratio, replace) in experiment.ARMS.items()
            },
            "files_sha256": {name: common.file_sha256(self.root / name) for name in (
                "groups_4to1.jsonl", "groups_8to1.jsonl", "generated_700.jsonl", "image_inventory.jsonl")},
        })

    @staticmethod
    def image_tensor(path):
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR),
                                dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(pixels).permute(2, 0, 1)
        return (tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor(
            [0.229, 0.224, 0.225]).view(3, 1, 1)

    def test_exact_companions_and_original_exposure_for_both_ratio_pairs(self):
        self.assertEqual(self.groups[4], self.groups[8][:5])
        generated_sources = {row["source_path"] for row in self.rows}
        for ratio in (8, 4):
            real = experiment.FixedPairedDataset(self.root, self.config, f"true_{ratio}to1")
            generated = experiment.FixedPairedDataset(self.root, self.config, f"generated_{ratio}to1")
            sources, real_exposure, generated_exposure, paired_exposure = [], 0, 0, 0
            for index, bag in enumerate(self.groups[ratio]):
                real_images, real_labels, real_kinds, real_companions, real_valid = real[index]
                images, labels, kinds, companions, valid = generated[index]
                self.assertTrue(torch.equal(real_labels, labels))
                self.assertFalse(bool(real_kinds.any()))
                self.assertFalse(bool(real_valid.any()))
                torch.testing.assert_close(real_companions, real_images, rtol=0, atol=0)
                self.assertTrue(torch.equal(valid, kinds))
                for slot, source in enumerate(bag["sources"]):
                    sources.append(source)
                    expected_source = self.image_tensor(source)
                    torch.testing.assert_close(companions[slot], expected_source, rtol=0, atol=0)
                    torch.testing.assert_close(real_images[slot], expected_source, rtol=0, atol=0)
                    if source in generated_sources:
                        self.assertTrue(bool(kinds[slot]))
                        output = generated.generated[source]["output_path"]
                        torch.testing.assert_close(images[slot], self.image_tensor(output), rtol=0, atol=0)
                        self.assertFalse(torch.equal(images[slot], companions[slot]))
                    else:
                        self.assertFalse(bool(kinds[slot]))
                        torch.testing.assert_close(images[slot], real_images[slot], rtol=0, atol=0)
                generated_exposure += int(kinds.sum())
                real_exposure += kinds.numel() - int(kinds.sum())
                paired_exposure += int(valid.sum())
            self.assertEqual(len(sources), len(set(sources)))
            self.assertEqual(len(sources), 4 * (ratio + 1))
            self.assertEqual((real_exposure, generated_exposure, paired_exposure), (4 * ratio, 4, 4))

    def test_disabled_module_preserves_original_three_item_dataset(self):
        config = copy.deepcopy(self.config)
        config["reliability"]["enabled"] = False
        for arm in experiment.ARMS:
            dataset = experiment.FixedPairedDataset(self.root, config, arm)
            self.assertEqual(len(dataset[0]), 3)

    def model_config(self, enabled=True):
        config = default_model_config(experiment.BACKBONE, (224, 224), 4, 2, 8, 8)
        if enabled:
            config["agg_config"].update(reliability_ot=True, reliability_lambda=2.0,
                                        reliability_hidden_dim=64)
        return config

    def resolve_initialization(self, args, resume=None):
        # Keep the real trainer's provenance semantics; only shrink aggregator
        # dimensions and replace DINO construction for an inexpensive CPU test.
        return self.model_config(), None, {"init_checkpoint": None, "init_checkpoint_sha256": None}

    def test_four_actual_cpu_trainers_save_head_and_exact_paired_exposure(self):
        common.use_salad()
        torch.manual_seed(self.args.seed)
        standard = FakeSALADModel(self.model_config(False), backbone_weights=self.backbone_weights)
        expected_standard = experiment.tensor_state_sha256(standard.aggregator.state_dict())
        torch.manual_seed(self.args.seed)
        fresh = FakeSALADModel(self.model_config(), backbone_weights=self.backbone_weights)
        initial_head = fresh.aggregator.reliability_head[-1].weight.detach().clone()
        initial_head_bias = fresh.aggregator.reliability_head[-1].bias.detach().clone()
        stripped = {name: tensor for name, tensor in fresh.aggregator.state_dict().items()
                    if not name.startswith("reliability_")}
        self.assertEqual(experiment.tensor_state_sha256(stripped), expected_standard)
        initializations = []
        with patch("workflow.model.SALADModel", FakeSALADModel), patch(
                "train_salad.resolve_model_initialization", side_effect=self.resolve_initialization), \
                contextlib.redirect_stdout(io.StringIO()):
            for arm, (ratio, replaced) in experiment.ARMS.items():
                self.args.arm = arm
                experiment.train_arm(self.args, self.config)
                saved = experiment.verify_checkpoint(self.args, self.config, arm)
                experiment.check_exposure(self.args, self.config, arm)
                self.assertEqual(saved["epoch"], 1)
                training = saved["training_config"]
                self.assertIsNone(training["init_checkpoint"])
                self.assertIsNone(training["init_checkpoint_sha256"])
                self.assertTrue(saved["model_config"]["agg_config"]["reliability_ot"])
                self.assertEqual(saved["model_config"]["agg_config"]["reliability_lambda"], 2.0)
                self.assertEqual(saved["model_config"]["agg_config"]["reliability_hidden_dim"], 64)
                for name, value in {"loss_weight": 0.1, "real_prior_weight": 0.01,
                                    "coverage_weight": 0.1, "coverage_floor": 0.5,
                                    "head_learning_rate": 6e-5}.items():
                    self.assertEqual(training["reliability"][name], value)
                metrics = saved["metrics"]
                expected = (4 * ratio, 4) if replaced else (4 * (ratio + 1), 0)
                self.assertEqual((metrics["real_exposure"], metrics["synthetic_exposure"]), expected)
                self.assertEqual(metrics["reliability"]["paired_synthetic_exposure"], 4 if replaced else 0)
                self.assertTrue(np.isfinite(metrics["loss"]))
                self.assertTrue(np.isfinite(metrics["reliability"]["total"]))
                if replaced:
                    self.assertGreater(metrics["reliability"]["positive_patches"], 0)
                    self.assertGreater(metrics["reliability"]["supervised_fraction"], 0)
                keys = {key for key in saved["state_dict"] if key.startswith("aggregator.reliability_")}
                self.assertEqual(keys, {"aggregator.reliability_ot_lambda",
                                        "aggregator.reliability_head.0.weight",
                                        "aggregator.reliability_head.0.bias",
                                        "aggregator.reliability_head.2.weight",
                                        "aggregator.reliability_head.2.bias"})
                changed_weight = not torch.equal(saved["state_dict"]["aggregator.reliability_head.2.weight"],
                                                 initial_head)
                changed_bias = not torch.equal(saved["state_dict"]["aggregator.reliability_head.2.bias"],
                                               initial_head_bias)
                self.assertTrue(changed_weight or changed_bias)
                self.assertEqual(len(saved["optimizer_state_dict"]["param_groups"]), 2)
                initialization = experiment.verify_initialization(self.args, self.config, arm)
                initializations.append(initialization)
                self.assertEqual(initialization["standard_aggregator_state_sha256"], expected_standard)
                # The saved model works with a single image and strict weights;
                # there is no inference companion or training-only input.
                loaded = FakeSALADModel(saved["model_config"], pretrained_backbone=False)
                loaded.load_state_dict(saved["state_dict"], strict=True)
                loaded.eval()
                image = experiment.FixedPairedDataset(self.root, self.config, arm)[0][0][0:1]
                with torch.no_grad():
                    descriptor = loaded(image)
                self.assertEqual(tuple(descriptor.shape), (1, 24))
                torch.testing.assert_close(descriptor.norm(dim=1), torch.ones(1), rtol=1e-5, atol=1e-5)
                # A finished arm must skip training and leave the checkpoint intact.
                digest = common.file_sha256(self.root / arm / "checkpoint.pt")
                experiment.train_arm(self.args, self.config)
                self.assertEqual(common.file_sha256(self.root / arm / "checkpoint.pt"), digest)
        for name in ("initial_backbone_state_sha256", "initial_aggregator_state_sha256",
                     "standard_aggregator_state_sha256"):
            self.assertEqual(len({row[name] for row in initializations}), 1)
        shared = experiment.check_shared_initialization(self.args, self.config)
        self.assertEqual(shared["standard_aggregator_state_sha256"], expected_standard)
        self.assertEqual(shared["initial_backbone_state_sha256"],
                         experiment.tensor_state_sha256(torch.load(self.backbone_weights, weights_only=True)))


if __name__ == "__main__":
    unittest.main()
