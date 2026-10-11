"""CPU cache integration: complete cohorts, exact bytes and immutable recovery."""

from __future__ import annotations

import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.qwen_curriculum import common, fixed_reliability_cache as fixed

common.use_salad()
from workflow.reliability_cache import ReliabilityTargetCache


class FakeDINO(nn.Module):
    constructions = []

    def __init__(self, architecture, config, pretrained, repo, weights):
        super().__init__()
        self.constructions.append((architecture, deepcopy(config), pretrained, repo, weights))
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, images):
        if self.training or self.scale.requires_grad:
            raise AssertionError("The offline teacher must stay frozen and in evaluation mode")
        grid = F.adaptive_avg_pool2d(images, (16, 16)) * self.scale
        return grid, grid.mean((-2, -1))


class FixedReliabilityCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        self.weights = self.root / "raw_dino.pth"
        self.weights.write_bytes(b"Raw DINO-only fixture, never a SALAD checkpoint")
        self.rows = []
        for i in range(4):
            pixels = np.random.default_rng(i).integers(0, 255, (32, 32, 3), dtype=np.uint8)
            source, output = self.root / f"source{i}.png", self.root / f"generated{i}.png"
            Image.fromarray(pixels).save(source)
            Image.fromarray((pixels.astype(np.float32) * 0.8).astype(np.uint8)).save(output)
            self.rows.append({
                "source_path": str(source), "source_sha256": common.file_sha256(source),
                "output_path": str(output), "output_sha256": common.file_sha256(output),
                "status": "passed" if i % 2 else "rejected", "passed": bool(i % 2),
                "eligible_for_training": bool(i % 2),
            })
        self.manifest = self.root / "generated_700.jsonl"
        common.write_jsonl(self.manifest, self.rows)
        self.original_manifest = self.manifest.read_bytes()
        self.config = self.seal({
            "request": {"num_images": 4, "backbone_weights": str(self.weights)},
            "backbone_weights_sha256": common.file_sha256(self.weights),
            "files_sha256": {"generated_700.jsonl": common.file_sha256(self.manifest)},
            "reliability": {"enabled": True, "target_cache": str(self.root / "targets")},
        })
        self.args = SimpleNamespace(output_dir=self.root, backbone_weights=self.weights,
                                    backbone_repo=self.root, device="cpu")
        FakeDINO.constructions = []

    @staticmethod
    def seal(value):
        return {**value, "fingerprint": common.fingerprint({k: v for k, v in value.items() if k != "fingerprint"})}

    def build(self, config=None):
        with patch("workflow.model.DINOBackbone", FakeDINO), contextlib.redirect_stdout(io.StringIO()):
            return fixed.ensure_cache(self.args, config or self.config)

    def test_all_passed_and_rejected_pairs_use_raw_frozen_dino_without_manifest_changes(self):
        descriptor = self.build()
        self.assertEqual(descriptor["source_pair_count"], 4)
        self.assertEqual(self.manifest.read_bytes(), self.original_manifest)
        self.assertEqual(len(FakeDINO.constructions), 1)
        architecture, config, pretrained, repo, weights = FakeDINO.constructions[0]
        self.assertEqual(architecture, "dinov2_vitb14")
        self.assertEqual(config, {"num_trainable_blocks": 0, "norm_layer": True, "return_token": True})
        self.assertTrue(pretrained)
        self.assertEqual(weights, self.weights)
        self.assertEqual(repo, self.root)
        cache = ReliabilityTargetCache(Path(descriptor["cache_path"]), (224, 224))
        self.assertEqual(len(cache.index["entries"]), 4)
        self.assertEqual(cache.index["teacher_model_config"], fixed.TEACHER_MODEL_CONFIG)
        for row in self.rows:
            target, confidence = cache.fetch(Path(row["output_path"]), Path(row["source_path"]))
            self.assertEqual(tuple(target.shape), (1, 16, 16))
            self.assertEqual(tuple(confidence.shape), (1, 16, 16))
        self.assertEqual(fixed.validate_cache(self.config, self.root, required=True), descriptor)

    def test_completed_cache_reuses_without_teacher_or_teacher_file_and_repairs_missing_binding(self):
        descriptor = self.build()
        target_path = self.root / "targets/targets.pt"
        target_bytes = target_path.read_bytes()
        self.weights.unlink()
        with patch("workflow.model.DINOBackbone", side_effect=AssertionError("Teacher must not load")):
            self.assertEqual(fixed.ensure_cache(self.args, self.config), descriptor)
            (self.root / "cache_binding.json").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "lacks its experiment"):
                fixed.validate_cache(self.config, self.root, required=True)
            self.assertEqual(fixed.ensure_cache(self.args, self.config), descriptor)
        self.assertEqual(target_path.read_bytes(), target_bytes)

    def test_image_and_tensor_hash_changes_are_rejected(self):
        self.build()
        source = Path(self.rows[0]["source_path"])
        original = source.read_bytes()
        source.write_bytes(b"changed source")
        with self.assertRaisesRegex(ValueError, "source image hash changed"):
            fixed.validate_cache(self.config, self.root, required=True)
        source.write_bytes(original)
        target_path = self.root / "targets/targets.pt"
        target_path.write_bytes(b"changed targets")
        with self.assertRaisesRegex(ValueError, "tensor hash mismatch"):
            fixed.ensure_cache(self.args, self.config)
        self.assertEqual(target_path.read_bytes(), b"changed targets")

    def test_wrong_exact_source_or_incomplete_cohort_is_rejected_before_binding(self):
        self.build()
        binding = self.root / "cache_binding.json"
        binding.unlink()
        index_path = self.root / "targets/index.json"
        original = json.loads(index_path.read_text())
        changed = deepcopy(original)
        changed["entries"][0]["source_path"] = self.rows[1]["source_path"]
        changed["entries"][0]["source_sha256"] = self.rows[1]["source_sha256"]
        common.write_json(index_path, changed)
        with self.assertRaisesRegex(ValueError, "source pairs/hashes differ"):
            fixed.ensure_cache(self.args, self.config)
        # A valid-looking 3-pair cache still cannot stand in for all 4 edits.
        changed = deepcopy(original)
        changed["entries"] = changed["entries"][:3]
        payload = torch.load(self.root / "targets/targets.pt", weights_only=True)
        torch.save({key: tensor[:3] for key, tensor in payload.items()}, self.root / "targets/targets.pt")
        changed["tensor_sha256"] = common.file_sha256(self.root / "targets/targets.pt")
        common.write_json(index_path, changed)
        with self.assertRaisesRegex(ValueError, "source pairs/hashes differ"):
            fixed.ensure_cache(self.args, self.config)
        self.assertFalse(binding.exists())

    def test_binding_changes_and_changed_experiment_are_rejected(self):
        self.build()
        binding_path = self.root / "cache_binding.json"
        original = json.loads(binding_path.read_text())
        changed = deepcopy(original)
        changed["source_pair_count"] = 3
        common.write_json(binding_path, self.seal(changed))
        with self.assertRaisesRegex(ValueError, "binding differs"):
            fixed.validate_cache(self.config, self.root)
        common.write_json(binding_path, original)
        config = deepcopy(self.config)
        config["request"]["seed"] = 73
        config = self.seal(config)
        with self.assertRaisesRegex(ValueError, "binding differs"):
            fixed.ensure_cache(self.args, config)

    def test_dangling_binding_is_never_silently_repaired(self):
        self.assertIsNone(fixed.validate_cache(self.config, self.root))
        with self.assertRaises(FileNotFoundError):
            fixed.validate_cache(self.config, self.root, required=True)
        self.build()
        (self.root / "targets/index.json").unlink()
        for action in (lambda: fixed.validate_cache(self.config, self.root),
                       lambda: fixed.ensure_cache(self.args, self.config)):
            with self.assertRaisesRegex(ValueError, "binding exists without"):
                action()

    def test_only_unbound_incomplete_targets_can_be_overwritten(self):
        path = self.root / "targets"
        path.mkdir()
        (path / "targets.pt").write_bytes(b"interrupted target write")
        self.assertIsNone(fixed.validate_cache(self.config, self.root))
        self.build()
        (path / "unrelated.txt").write_text("must survive")
        with self.assertRaisesRegex(ValueError, "Unexpected reliability cache files"):
            fixed.ensure_cache(self.args, self.config)
        self.assertEqual((path / "unrelated.txt").read_text(), "must survive")

    def test_teacher_identity_and_manifest_bytes_are_strictly_bound(self):
        self.weights.write_bytes(b"another raw teacher")
        with self.assertRaisesRegex(ValueError, "teacher weights differ"):
            self.build()
        self.weights.write_bytes(b"Raw DINO-only fixture, never a SALAD checkpoint")
        self.build()
        self.manifest.write_bytes(self.original_manifest + b"\n")
        with self.assertRaisesRegex(ValueError, "manifest bytes changed"):
            fixed.validate_cache(self.config, self.root)


if __name__ == "__main__":
    unittest.main()
