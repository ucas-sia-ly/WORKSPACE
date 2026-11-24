"""CPU checks for metric learning, portable checkpoints, and the two CLI entries."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from PIL import Image

SALAD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SALAD_ROOT))

from workflow.metric_loss import multi_similarity_loss
from workflow.model import load_checkpoint_model, read_checkpoint


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
    return FakeDINO()
'''


class WorkflowTests(unittest.TestCase):
    def test_hard_pair_loss_is_finite_and_has_gradients(self):
        descriptors = torch.tensor([[1., 0.], [0., 1.], [0.8, 0.2], [0.2, 0.8]], requires_grad=True)
        loss = multi_similarity_loss(descriptors, torch.tensor([0, 0, 1, 1]))
        self.assertGreater(float(loss.detach()), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(descriptors.grad).all())
        self.assertGreater(float(descriptors.grad.abs().sum()), 0)

    def test_no_valid_pairs_has_differentiable_zero(self):
        for labels in ([0, 0, 0], [0, 1, 2]):
            descriptors = torch.randn(3, 8, requires_grad=True)
            loss = multi_similarity_loss(descriptors, torch.tensor(labels))
            self.assertEqual(float(loss.detach()), 0)
            loss.backward()
            self.assertTrue(torch.equal(descriptors.grad, torch.zeros_like(descriptors)))

    def test_loss_stays_float32_inside_autocast(self):
        torch.manual_seed(7)
        descriptors = torch.randn(8, 16, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        reference = multi_similarity_loss(descriptors, labels)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = multi_similarity_loss(descriptors, labels)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_model_checkpoint_and_cli_epoch_resume(self):
        # A tiny local test backbone avoids downloads. The SALAD aggregator,
        # optimizer, dataset, loss, checkpoint and evaluation entries are real.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "dinov2"
            repo.mkdir()
            (repo / "hubconf.py").write_text(HUBCONF)
            data = root / "gsv"
            (data / "Dataframes").mkdir(parents=True)
            (data / "Images" / "Test").mkdir(parents=True)
            rows = []
            sources = []
            for place in range(2):
                for view in range(2):
                    panoid = f"p{place}v{view}"
                    rows.append([place, 2020, 1, 0, "Test", "1.0", "2.0", panoid])
                    path = data / "Images" / "Test" / f"Test_{place:07d}_2020_01_000_1.0_2.0_{panoid}.jpg"
                    Image.new("RGB", (28, 28), (place * 150, 40 + view * 80, 100)).save(path)
                    sources.append(path)
            with (data / "Dataframes" / "Test.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["place_id", "year", "month", "northdeg", "city_id", "lat", "lon", "panoid"])
                writer.writerows(rows)
            manifest = root / "synthetic.jsonl"
            generated = root / "aug.jpg"
            Image.new("RGB", (28, 28), (25, 60, 80)).save(generated)
            manifest.write_text(json.dumps({"source_path": str(sources[0]), "output_path": str(generated),
                                            "passed": True, "eligible_for_training": True}) + "\n")
            full = root / "full"
            common = ["--real-data", str(data), "--synthetic-manifest", str(manifest),
                      "--epochs", "2", "--batch-size", "2", "--images-per-place", "2",
                      "--min-images-per-place", "2", "--backbone", "dinov2_vits14",
                      "--backbone-repo", str(repo), "--image-size", "28", "28",
                      "--num-clusters", "2", "--cluster-dim", "8", "--token-dim", "8",
                      "--num-trainable-blocks", "0", "--num-workers", "0", "--no-augment",
                      "--device", "cpu", "--precision", "32"]
            env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

            def run(script, arguments):
                result = subprocess.run([sys.executable, str(SALAD_ROOT / script), *arguments],
                                        text=True, capture_output=True, env=env, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            run("train_salad.py", [*common, "--output-dir", str(full)])
            checkpoint = read_checkpoint(full / "checkpoint.pt")
            self.assertEqual(checkpoint["epoch"], 2)
            self.assertEqual(checkpoint["metrics"]["real_exposure"], 3)
            self.assertEqual(checkpoint["metrics"]["synthetic_exposure"], 1)
            self.assertEqual(checkpoint["global_step"], 2)
            first = read_checkpoint(full / "checkpoint_epoch_001.pt")
            self.assertTrue(any(not torch.equal(value, first["state_dict"][key])
                                for key, value in checkpoint["state_dict"].items() if key.startswith("aggregator.")))
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("backbone."):
                    torch.testing.assert_close(value, first["state_dict"][key], rtol=0, atol=0)

            resumed_dir = root / "resumed"
            run("train_salad.py", [*common, "--output-dir", str(resumed_dir),
                                   "--resume", str(full / "checkpoint_epoch_001.pt")])
            resumed = read_checkpoint(resumed_dir / "checkpoint.pt")
            for key, value in checkpoint["state_dict"].items():
                torch.testing.assert_close(value, resumed["state_dict"][key], rtol=0, atol=0)

            model = load_checkpoint_model(full / "checkpoint.pt", "cpu", backbone_repo=repo)
            self.assertEqual(model.image_size, (28, 28))
            self.assertFalse(any(p.requires_grad for p in model.backbone.parameters()))
            raw_path = root / "raw.pt"
            torch.save(checkpoint["state_dict"], raw_path)
            legacy = load_checkpoint_model(raw_path, "cpu", backbone_repo=repo)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, legacy.state_dict()[key], rtol=0, atol=0)

            eval_manifest = root / "evaluation.json"
            eval_manifest.write_text(json.dumps({
                "references": [{"id": "db0", "path": str(sources[0])},
                               {"id": "db1", "path": str(sources[2])}],
                "queries": [{"id": "query", "path": str(generated), "positives": ["db1"],
                             "source_id": str(sources[0].relative_to(data / "Images"))}],
            }))
            output = root / "results.json"
            run("evaluate_salad.py", ["--checkpoint", str(full / "checkpoint.pt"),
                                      "--dataset", "manifest", "--dataset-root", str(root),
                                      "--eval-manifest", str(eval_manifest), "--output", str(output),
                                      "--backbone-repo", str(repo), "--device", "cpu",
                                      "--num-workers", "0", "--save-hard-cases"])
            results = json.loads(output.read_text())
            self.assertEqual(set(results["recall"]), {"R@1", "R@5", "R@10"})
            self.assertEqual(results["rank_base"], 1)
            if results["error_queries"]:
                self.assertEqual(results["error_queries"][0]["source_id"], "Test/" + sources[0].name)


if __name__ == "__main__":
    unittest.main()
