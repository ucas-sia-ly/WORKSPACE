"""CPU contracts for original pretrained SALAD with accepted-only replacement.

These tests keep the original aggregation architecture, exercise the complete
checkpoint initialization branch and retain real-only places in the full pool.
No pretrained downloads or CUDA devices are needed.
"""

import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

SALAD_ROOT = Path(__file__).resolve().parents[1]
if str(SALAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SALAD_ROOT))

from train_salad import parse_args, resolve_model_initialization, train
from workflow.model import SALADModel, default_model_config, load_model_state, read_checkpoint
from workflow.training_data import MixedGSVCitiesDataset


HUBCONF = '''
import torch
from torch import nn

class FakeDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 384, 14, 14)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 384))
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(384, 384), nn.GELU()) for _ in range(4)])
        self.norm = nn.LayerNorm(384)

    def prepare_tokens_with_masks(self, images):
        patches = self.patch(images).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(len(images), -1, -1) + patches.mean(1, keepdim=True)
        return torch.cat([cls, patches], dim=1)

def dinov2_vits14(pretrained=True):
    if pretrained:
        raise AssertionError('Complete SALAD initialization must avoid backbone pretrained loading')
    return FakeDINO()
'''


class OriginalPretrainedFinetuneTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        self.repo = self.root / "dinov2"
        self.repo.mkdir()
        (self.repo / "hubconf.py").write_text(HUBCONF)
        torch.manual_seed(11)
        self.config = default_model_config("dinov2_vits14", (28, 28), 2, 2, 8, 8)
        model = SALADModel(self.config, pretrained_backbone=False, backbone_repo=self.repo)
        self.state = {key: value.clone() for key, value in model.state_dict().items()}
        self.initial = self.root / "original_salad.ckpt"
        torch.save(self.state, self.initial)

        self.data = self.root / "gsv"
        (self.data / "Dataframes").mkdir(parents=True)
        (self.data / "Images" / "Test").mkdir(parents=True)
        rows, self.sources = [], []
        rng = np.random.default_rng(17)
        for place in range(3):
            for view in range(4):
                panoid = f"p{place}v{view}"
                rows.append([place, 2020, 1, 0, "Test", "1.0", "2.0", panoid])
                source = self.data / "Images" / "Test" / f"Test_{place:07d}_2020_01_000_1.0_2.0_{panoid}.jpg"
                pixels = rng.integers(0, 256, (28, 28, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(source)
                self.sources.append(source)
        with (self.data / "Dataframes" / "Test.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["place_id", "year", "month", "northdeg", "city_id", "lat", "lon", "panoid"])
            writer.writerows(rows)
        accepted, rejected = self.root / "accepted.png", self.root / "rejected.png"
        Image.new("RGB", (28, 28), (20, 30, 40)).save(accepted)
        Image.new("RGB", (28, 28), (80, 90, 100)).save(rejected)
        self.manifest = self.root / "accepted_only.jsonl"
        self.manifest.write_text("".join(json.dumps(entry) + "\n" for entry in [
            {"source_path": str(self.sources[0]), "output_path": str(accepted),
             "condition": "night", "passed": True, "eligible_for_training": True},
            {"source_path": str(self.sources[4]), "output_path": str(rejected),
             "condition": "snow", "passed": False, "eligible_for_training": False},
        ]))

    def arguments(self, output, *extra):
        return parse_args([
            "--real-data", str(self.data), "--synthetic-manifest", str(self.manifest),
            "--output-dir", str(output), "--epochs", "2", "--batch-size", "3",
            "--images-per-place", "4", "--min-images-per-place", "4",
            "--image-size", "28", "28", "--num-trainable-blocks", "2",
            "--learning-rate", "1e-5", "--weight-decay", "1e-4",
            "--backbone-repo", str(self.repo), "--num-workers", "0",
            "--synthetic-mode", "replace", "--synthetic-fraction", "1.0",
            "--device", "cpu", "--precision", "32", "--no-augment", *extra,
        ])

    def test_raw_original_weights_override_fresh_defaults_without_adding_a_head(self):
        args = self.arguments(self.root / "unused", "--init-checkpoint", str(self.initial))
        config, state, provenance = resolve_model_initialization(args)
        # The CLI default is ViT-B, while the checkpoint proves that dimensions
        # must be inferred from its complete original state rather than flags.
        self.assertEqual(args.backbone, "dinov2_vitb14")
        self.assertEqual(config["backbone_arch"], "dinov2_vits14")
        self.assertNotIn("reliability_ot", config["agg_config"])
        model = SALADModel(config, pretrained_backbone=False, backbone_repo=self.repo)
        load_model_state(model, state)
        self.assertFalse(model.aggregator.reliability_ot)
        self.assertEqual(set(model.state_dict()), set(self.state))
        for key, original in self.state.items():
            torch.testing.assert_close(model.state_dict()[key], original, rtol=0, atol=0)
        self.assertEqual(provenance["init_checkpoint_sha256"], hashlib.sha256(self.initial.read_bytes()).hexdigest())

    def test_full_pool_keeps_real_only_places_and_never_replaces_with_rejected_rows(self):
        dataset = MixedGSVCitiesDataset(
            self.data, self.manifest, images_per_place=4, min_images_per_place=4,
            synthetic_mode="replace", synthetic_fraction=1.0,
            image_size=(28, 28), augment=False,
        )
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.summary["num_synthetic_images"], 1)
        self.assertEqual(dataset.summary["manifest"]["ignored_rows"], 1)
        observed = [int(dataset[index][2].sum()) for index in range(3)]
        self.assertEqual(observed, [1, 0, 0])

    def test_finetune_updates_original_head_and_last_blocks_but_preserves_frozen_weights(self):
        output = self.root / "trained"
        args = self.arguments(output, "--init-checkpoint", str(self.initial))
        initialized = []
        def observe_initialization(model, state, **kwargs):
            result = load_model_state(model, state, **kwargs)
            initialized.append({key: value.detach().clone() for key, value in model.state_dict().items()})
            return result
        with patch("workflow.model.load_model_state", side_effect=observe_initialization), contextlib.redirect_stdout(io.StringIO()):
            train(args)
        self.assertEqual(len(initialized), 1)
        for key, original in self.state.items():
            torch.testing.assert_close(initialized[0][key], original, rtol=0, atol=0)
        checkpoint = read_checkpoint(output / "checkpoint.pt")
        trained = checkpoint["state_dict"]
        self.assertFalse(any("reliability" in key for key in trained))
        self.assertEqual(checkpoint["dataset_summary"]["num_places"], 3)
        self.assertEqual(checkpoint["metrics"]["synthetic_exposure"], 1)
        self.assertEqual(checkpoint["metrics"]["real_exposure"], 11)
        frozen = [key for key in self.state if key.startswith(("backbone.model.patch", "backbone.model.cls_token", "backbone.model.blocks.0.", "backbone.model.blocks.1."))]
        self.assertTrue(frozen)
        for key in frozen:
            torch.testing.assert_close(trained[key], self.state[key], rtol=0, atol=0)
        for prefix in ("aggregator.", "backbone.model.blocks.2.", "backbone.model.blocks.3."):
            self.assertTrue(any(not torch.equal(trained[key], value) for key, value in self.state.items() if key.startswith(prefix)), prefix)
        self.assertEqual(checkpoint["training_config"]["init_checkpoint"], str(self.initial.resolve()))
        self.assertNotIn("reliability", checkpoint["training_config"])


if __name__ == "__main__":
    unittest.main()
