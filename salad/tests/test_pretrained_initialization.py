"""Complete SALAD initialization and synthetic-place fine-tuning contracts."""

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

import torch
from PIL import Image

SALAD_ROOT = Path(__file__).resolve().parents[1]
if str(SALAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SALAD_ROOT))

from train_salad import parse_args, resolve_model_initialization, train
from workflow.model import SALADModel, default_model_config, read_checkpoint


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
        raise RuntimeError("Full checkpoint initialization must not request pretrained backbone weights")
    return FakeDINO()
'''


class PretrainedInitializationTests(unittest.TestCase):
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
        self.data = self.root / "gsv"
        (self.data / "Dataframes").mkdir(parents=True)
        (self.data / "Images" / "Test").mkdir(parents=True)
        rows = []
        synthetic = []
        for place in range(3):
            for view in range(2):
                panoid = f"p{place}v{view}"
                rows.append([place, 2020, 1, 0, "Test", "1.0", "2.0", panoid])
                image = self.data / "Images" / "Test" / f"Test_{place:07d}_2020_01_000_1.0_2.0_{panoid}.jpg"
                Image.new("RGB", (28, 28), (40 + place * 60, 50 + view * 50, 100)).save(image)
                if place > 0 and view == 0:
                    output = self.root / f"aug_{place}.jpg"
                    Image.new("RGB", (28, 28), (70, place * 40, 80)).save(output)
                    synthetic.append({"source_path": str(image), "output_path": str(output),
                                      "passed": True, "eligible_for_training": True})
        with (self.data / "Dataframes" / "Test.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["place_id", "year", "month", "northdeg", "city_id", "lat", "lon", "panoid"])
            writer.writerows(rows)
        self.manifest = self.root / "synthetic.jsonl"
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in synthetic))
        self.config = default_model_config("dinov2_vits14", (28, 28), 0, 2, 8, 8)
        torch.manual_seed(17)
        model = SALADModel(self.config, pretrained_backbone=False, backbone_repo=self.repo)
        self.state = {name: value.clone() for name, value in model.state_dict().items()}

    def arguments(self, output, *extra):
        return parse_args([
            "--real-data", str(self.data), "--synthetic-manifest", str(self.manifest),
            "--output-dir", str(output), "--epochs", "2", "--batch-size", "2",
            "--images-per-place", "2", "--min-images-per-place", "2",
            "--image-size", "28", "28", "--num-trainable-blocks", "0",
            "--backbone-repo", str(self.repo), "--num-workers", "0",
            "--weight-decay", "0", "--device", "cpu", "--precision", "32",
            "--no-augment", *extra,
        ])

    def test_raw_lightning_and_native_weights_infer_shapes(self):
        payloads = {
            "raw": self.state,
            "lightning": {"state_dict": self.state, "hyper_parameters": {
                "backbone_config": {"num_trainable_blocks": 4, "norm_layer": True}}},
            "native": {"state_dict": self.state, "model_config": self.config},
        }
        for name, payload in payloads.items():
            with self.subTest(format=name):
                initial = self.root / f"{name}.pt"
                torch.save(payload, initial)
                args = self.arguments(self.root / name, "--init-checkpoint", str(initial))
                config, state, provenance = resolve_model_initialization(args)
                self.assertEqual(config["backbone_arch"], "dinov2_vits14")
                self.assertEqual(config["agg_config"]["num_clusters"], 2)
                self.assertEqual(config["image_size"], [28, 28])
                self.assertEqual(config["backbone_config"]["num_trainable_blocks"], 0)
                model = SALADModel(config, pretrained_backbone=False, backbone_repo=self.repo)
                model.load_state_dict(state, strict=True)
                for key, value in self.state.items():
                    torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
                self.assertEqual(provenance["init_checkpoint"], str(initial.resolve()))
                self.assertEqual(provenance["init_checkpoint_sha256"], hashlib.sha256(initial.read_bytes()).hexdigest())

    def test_fine_tune_preserves_full_initialization_and_resumes_without_original(self):
        initial = self.root / "pretrained.pt"
        # Training state in an initialization checkpoint must be ignored. Only
        # --resume restores that state; fine-tuning starts a fresh optimizer.
        torch.save({"state_dict": self.state, "model_config": self.config,
                    "optimizer_state_dict": {"invalid_for_optimizer": True}, "epoch": 99}, initial)
        output = self.root / "trained"
        args = self.arguments(output, "--init-checkpoint", str(initial), "--synthetic-places-only")
        # Zero loss makes initialization observable after a real loader,
        # backward pass, optimizer step and checkpoint save, without updates.
        with patch("workflow.metric_loss.multi_similarity_loss", side_effect=lambda descriptors, labels, **kw: descriptors.sum() * 0), contextlib.redirect_stdout(io.StringIO()):
            train(args)
        checkpoint = read_checkpoint(output / "checkpoint.pt")
        self.assertEqual(checkpoint["epoch"], 2)
        self.assertEqual(checkpoint["global_step"], 2)
        self.assertEqual(checkpoint["dataset_summary"]["num_places"], 2)
        self.assertEqual(checkpoint["dataset_summary"]["num_real_images"], 4)
        self.assertEqual(checkpoint["dataset_summary"]["num_synthetic_images"], 2)
        self.assertEqual(checkpoint["dataset_summary"]["excluded_places_without_synthetic"], 1)
        self.assertEqual(checkpoint["metrics"]["real_exposure"], 2)
        self.assertEqual(checkpoint["metrics"]["synthetic_exposure"], 2)
        for key, value in self.state.items():
            torch.testing.assert_close(checkpoint["state_dict"][key], value, rtol=0, atol=0)
        provenance = checkpoint["training_config"]
        self.assertEqual(provenance["init_checkpoint"], str(initial.resolve()))
        self.assertEqual(provenance["init_checkpoint_sha256"], hashlib.sha256(initial.read_bytes()).hexdigest())
        self.assertEqual(json.loads((output / "training_config.json").read_text())["init_policy"], "pretrained_salad_checkpoint")
        initial.unlink()
        resumed_dir = self.root / "resumed"
        resume_args = self.arguments(resumed_dir, "--resume", str(output / "checkpoint_epoch_001.pt"),
                                     "--synthetic-places-only")
        with patch("workflow.metric_loss.multi_similarity_loss", side_effect=lambda descriptors, labels, **kw: descriptors.sum() * 0), contextlib.redirect_stdout(io.StringIO()):
            train(resume_args)
        resumed = read_checkpoint(resumed_dir / "checkpoint.pt")
        self.assertEqual(resumed["training_config"], provenance)
        self.assertEqual(resumed["global_step"], 2)

    def test_incomplete_full_weights_are_rejected_strictly(self):
        initial = self.root / "incomplete.pt"
        torch.save({key: value for key, value in self.state.items() if key != "aggregator.dust_bin"}, initial)
        args = self.arguments(self.root / "rejected", "--init-checkpoint", str(initial))
        with self.assertRaisesRegex(RuntimeError, "dust_bin"), contextlib.redirect_stdout(io.StringIO()):
            train(args)

    def test_filter_requires_two_augmented_places(self):
        self.manifest.write_text(self.manifest.read_text().splitlines()[0] + "\n")
        # Data-only validation exercises filtering without requiring a model.
        args = self.arguments(self.root / "check", "--synthetic-places-only", "--check-data",
                              "--num-clusters", "2")
        with self.assertRaisesRegex(ValueError, "two eligible places"):
            train(args)

    def test_init_and_resume_are_mutually_exclusive(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.arguments(self.root / "invalid", "--init-checkpoint", "pretrained.pt", "--resume", "latest.pt")


if __name__ == "__main__":
    unittest.main()
