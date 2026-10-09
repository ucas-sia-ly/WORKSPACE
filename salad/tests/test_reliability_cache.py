"""Static frozen-teacher targets, provenance, and zero companion decoding."""

import contextlib
import csv
import io
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

SALAD_ROOT = Path(__file__).resolve().parents[1]
if str(SALAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SALAD_ROOT))

from cache_reliability_targets import build_cache, parse_args
from workflow.reliability_cache import ReliabilityTargetCache, file_sha256, load_teacher_image
from workflow.training_data import MixedGSVCitiesDataset


class FakeTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.config = {"test_teacher": True}
        self.calls = 0

    def backbone(self, images):
        assert not self.training and not torch.is_grad_enabled()
        assert not any(parameter.requires_grad for parameter in self.parameters())
        self.calls += 1
        generator = torch.Generator().manual_seed(81)
        features = torch.randn(1, 24, images.shape[-2] // 14, images.shape[-1] // 14,
                               generator=generator).to(images.device)
        return features.expand(images.shape[0], -1, -1, -1).clone(), torch.zeros(images.shape[0], 24)


class ReliabilityCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.gsv = self.root / "gsv"
        (self.gsv / "Dataframes").mkdir(parents=True)
        images = self.gsv / "Images" / "City"
        images.mkdir(parents=True)
        rows = []
        self.sources = []
        for view in range(4):
            row = {"place_id": "1", "year": "2020", "month": "1", "northdeg": str(view),
                   "city_id": "City", "lat": "1.0", "lon": "2.0", "panoid": f"p{view}"}
            source = images / f"City_0000001_2020_01_{view:03d}_1.0_2.0_p{view}.jpg"
            Image.new("RGB", (17, 19), (20 + 20 * view, 60, 100)).save(source)
            self.sources.append(source)
            rows.append(row)
        with (self.gsv / "Dataframes" / "City.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        self.outputs = []
        entries = []
        for view in range(2):
            output = self.root / f"generated{view}.jpg"
            output.write_bytes(self.sources[view].read_bytes())
            self.outputs.append(output)
            entries.append({"passed": True, "eligible_for_training": True, "weather_ok": True,
                            "source_path": str(self.sources[view]), "output_path": str(output)})
        # Rejected rows must not have their nonexistent paths resolved.
        entries.append({"passed": False, "eligible_for_training": True, "source_path": "missing", "output_path": "missing"})
        self.manifest = self.root / "manifest.jsonl"
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in entries))
        self.checkpoint = self.root / "teacher.ckpt"
        self.checkpoint.write_bytes(b"frozen teacher fixture")
        self.cache_path = self.root / "cache"
        self.teacher = FakeTeacher()
        self.args = parse_args(["--real-data", str(self.gsv), "--synthetic-manifest", str(self.manifest),
                                "--checkpoint", str(self.checkpoint), "--output-dir", str(self.cache_path),
                                "--device", "cpu", "--batch-size", "1"])

    def build(self):
        with patch("cache_reliability_targets.load_checkpoint_model", return_value=self.teacher):
            with contextlib.redirect_stdout(io.StringIO()):
                return build_cache(self.args)

    def dataset(self, cache=None, **kwargs):
        return MixedGSVCitiesDataset(self.gsv, self.manifest, images_per_place=4,
                                     min_images_per_place=4, synthetic_fraction=0.5,
                                     image_size=(224, 224), augment=False,
                                     reliability_pairs=True, reliability_target_cache=cache, **kwargs)

    def rewrite_index(self, mutate):
        index_path = self.cache_path / "index.json"
        index = json.loads(index_path.read_text())
        mutate(index)
        index_path.write_text(json.dumps(index))

    def rewrite_payload(self, mutate):
        tensor_path = self.cache_path / "targets.pt"
        payload = torch.load(tensor_path, weights_only=True)
        mutate(payload)
        torch.save(payload, tensor_path)
        self.rewrite_index(lambda index: index.update(tensor_sha256=file_sha256(tensor_path)))

    def test_builder_frozen_teacher_exact_acceptance_and_compact_payload(self):
        index = self.build()
        self.assertEqual(self.teacher.calls, 2)
        self.assertEqual(index["summary"]["pairs"], 2)
        self.assertGreater(index["summary"]["positive_patches"], 0)
        self.assertEqual(index["summary"]["negative_patches"], 0)
        self.assertEqual(index["teacher_checkpoint_sha256"], file_sha256(self.checkpoint))
        self.assertEqual(index["manifest_sha256"], file_sha256(self.manifest))
        payload = torch.load(self.cache_path / "targets.pt", weights_only=True)
        self.assertEqual(set(payload), {"targets", "confidence"})
        self.assertEqual(tuple(payload["targets"].shape), (2, 1, 16, 16))
        self.assertFalse(payload["targets"].requires_grad)

    def test_fetch_exact_image_identity_cpu_grid_and_safe_copy(self):
        self.build()
        cache = ReliabilityTargetCache(self.cache_path, (224, 224))
        target, confidence = cache.fetch(self.outputs[0], self.sources[0])
        self.assertEqual(tuple(target.shape), (1, 16, 16))
        self.assertEqual(target.device.type, "cpu")
        self.assertGreater(float(confidence.sum()), 0)
        target.fill_(0)
        self.assertGreater(float(cache.fetch(self.outputs[0], self.sources[0])[0].sum()), 0)
        with self.assertRaisesRegex(ValueError, "exact source mismatch"):
            cache.fetch(self.outputs[0], self.sources[1])
        with self.assertRaisesRegex(ValueError, "no cached"):
            cache.fetch(self.sources[0], self.sources[0])

    def test_dataset_cache_preserves_sampler_pixels_rng_and_skips_companion_decode(self):
        self.build()
        live = self.dataset()
        cached = self.dataset(self.cache_path)
        random.seed(79)
        old = live[0]
        after_live = random.getstate()
        paths = []
        actual_open = Image.open
        def capture(path, *args, **kwargs):
            paths.append(Path(path))
            return actual_open(path, *args, **kwargs)
        random.seed(79)
        with patch("PIL.Image.open", side_effect=capture):
            new = cached[0]
        self.assertEqual(len(paths), 4)
        self.assertEqual(random.getstate(), after_live)
        for a, b in zip(old[:3], new[:3]):
            self.assertTrue(torch.equal(a, b))
        _, _, kinds, targets, confidence = new
        self.assertEqual(tuple(targets.shape), (4, 1, 16, 16))
        self.assertTrue(torch.equal(targets[~kinds], torch.full_like(targets[~kinds], 0.5)))
        self.assertEqual(float(confidence[~kinds].sum()), 0)
        self.assertGreater(float(confidence[kinds].sum()), 0)
        batch = next(iter(torch.utils.data.DataLoader(cached, batch_size=1)))
        self.assertEqual(tuple(batch[3].shape), (1, 4, 1, 16, 16))

    def test_static_cache_requires_paired_unaugmented_exact_manifest(self):
        self.build()
        for paired, augment in ((False, False), (True, True)):
            with self.assertRaisesRegex(ValueError, "requires reliability_pairs=True and augment=False"):
                MixedGSVCitiesDataset(self.gsv, self.manifest, augment=augment,
                                     reliability_pairs=paired, reliability_target_cache=self.cache_path)
        self.manifest.write_text(self.manifest.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            self.dataset(self.cache_path)

    def test_cache_hash_detects_tensor_mutation(self):
        self.build()
        path = self.cache_path / "targets.pt"
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "tensor hash mismatch"):
            ReliabilityTargetCache(self.cache_path, (224, 224))

    def test_cache_grid_and_finite_ranges_are_validated(self):
        for mutate in (lambda p: p.update(targets=p["targets"][:, :, :-1]),
                       lambda p: p["confidence"].fill_(float("nan")),
                       lambda p: p["targets"].fill_(1.1)):
            with self.subTest(mutation=mutate):
                if not (self.cache_path / "index.json").exists():
                    self.build()
                self.rewrite_payload(mutate)
                with self.assertRaisesRegex(ValueError, "Invalid reliability cache"):
                    ReliabilityTargetCache(self.cache_path, (224, 224))
                (self.cache_path / "index.json").unlink()

    def test_image_changes_are_detected_initially_and_after_fetch(self):
        self.build()
        cache = ReliabilityTargetCache(self.cache_path, (224, 224))
        cache.fetch(self.outputs[0], self.sources[0])
        self.outputs[0].write_bytes(self.outputs[0].read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "image content changed"):
            cache.fetch(self.outputs[0], self.sources[0])
        with self.assertRaisesRegex(ValueError, "image content changed"):
            self.dataset(self.cache_path)

    def test_cache_image_size_duplicate_and_missing_pairs_fail(self):
        self.build()
        with self.assertRaisesRegex(ValueError, "image_size differs"):
            ReliabilityTargetCache(self.cache_path, (210, 224))
        self.rewrite_index(lambda index: index["entries"][1].update(index["entries"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate output"):
            ReliabilityTargetCache(self.cache_path, (224, 224))

    def test_completed_cache_cannot_be_overwritten_but_incomplete_can_retry(self):
        self.build()
        with self.assertRaisesRegex(FileExistsError, "Completed"):
            self.build()
        (self.cache_path / "index.json").unlink()
        self.build()
        self.assertTrue((self.cache_path / "index.json").is_file())

    def test_dataset_rejects_cache_missing_an_eligible_generated_image(self):
        self.build()
        # A different cache can be internally valid yet omit a training view.
        self.rewrite_payload(lambda p: p.update(targets=p["targets"][:1], confidence=p["confidence"][:1]))
        self.rewrite_index(lambda index: index.update(entries=index["entries"][:1]))
        with self.assertRaisesRegex(ValueError, "no cached reliability"):
            self.dataset(self.cache_path)

    def test_live_pair_mode_summary_stays_unchanged_without_cache(self):
        dataset = self.dataset()
        self.assertNotIn("reliability_target_cache", dataset.summary)
        self.assertEqual(dataset.summary["reliability_pair_augmentation"], "shared_spatial_and_color_jitter")

    def test_integrity_digest_changes_with_teacher_or_tensor_provenance(self):
        self.build()
        first = ReliabilityTargetCache(self.cache_path, (224, 224)).integrity_digest
        self.rewrite_index(lambda index: index.update(teacher_checkpoint_sha256="1" * 64))
        second = ReliabilityTargetCache(self.cache_path, (224, 224)).integrity_digest
        self.assertNotEqual(first, second)
        self.rewrite_payload(lambda p: p["confidence"].mul_(0.5))
        third = ReliabilityTargetCache(self.cache_path, (224, 224)).integrity_digest
        self.assertNotEqual(second, third)

    def test_teacher_preprocessing_equals_nonaugmented_dataset(self):
        dataset = self.dataset()
        actual_open = Image.open
        paths = []
        def capture(path, *args, **kwargs):
            paths.append(Path(path))
            return actual_open(path, *args, **kwargs)
        random.seed(45)
        with patch("PIL.Image.open", side_effect=capture):
            images, _, kinds, _, _ = dataset[0]
        cursor = 0
        for index, synthetic in enumerate(kinds.tolist()):
            expected = load_teacher_image(paths[cursor], (224, 224))
            self.assertTrue(torch.equal(images[index], expected))
            cursor += 1 + int(synthetic)


if __name__ == "__main__":
    unittest.main()
