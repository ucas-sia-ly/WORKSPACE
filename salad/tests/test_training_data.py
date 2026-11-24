"""Small local fixtures for the real/synthetic place grouping contract."""

from __future__ import annotations

import csv
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from workflow.training_data import MixedGSVCitiesDataset


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.gsv = self.root / "gsv"
        (self.gsv / "Dataframes").mkdir(parents=True)
        self.paths = {}
        self.csv_rows = {}
        for city in ("Alpha", "Beta"):
            rows = []
            (self.gsv / "Images" / city).mkdir(parents=True)
            for place_id in (1, 2):
                for view in range(4):
                    row = {
                        "place_id": str(place_id), "year": "2020", "month": "1",
                        "northdeg": "12", "city_id": city, "lat": "1.2300",
                        "lon": "4.5600", "panoid": f"p{place_id}_{view}",
                    }
                    filename = f"{city}_{place_id:07d}_2020_01_012_1.2300_4.5600_p{place_id}_{view}.jpg"
                    path = self.gsv / "Images" / city / filename
                    Image.new("RGB", (12, 10), color=(20 + view * 30, place_id * 30, 80)).save(path)
                    self.paths[(city, place_id, view)] = path
                    rows.append(row)
            self.csv_rows[city] = rows
            self._write_csv(city, rows)
        self.manifest = self.root / "outputs" / "manifest.jsonl"
        self.manifest.parent.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_csv(self, city, rows):
        fields = ["place_id", "year", "month", "northdeg", "city_id", "lat", "lon", "panoid"]
        with (self.gsv / "Dataframes" / f"{city}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            writer.writerows(rows)

    def _candidate(self, name="generated.jpg", source_key=("Alpha", 1, 0)):
        output = self.manifest.parent / name
        Image.new("RGB", (12, 10), color=(230, 180, 20)).save(output)
        return {
            "passed": True, "eligible_for_training": True,
            "source_path": str(self.paths[source_key]), "output_path": str(output),
        }

    def _manifest(self, entries):
        self.manifest.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
        return self.manifest

    def _dataset(self, **kwargs):
        return MixedGSVCitiesDataset(self.gsv, augment=False, image_size=(8, 14), **kwargs)

    def test_city_place_id_collisions_get_distinct_labels(self):
        dataset = self._dataset()
        self.assertEqual(len(dataset), 4)
        alpha = next(p for p in dataset.places if (p.city, p.place_id) == ("Alpha", 1))
        beta = next(p for p in dataset.places if (p.city, p.place_id) == ("Beta", 1))
        self.assertNotEqual(alpha.label, beta.label)
        self.assertEqual(len(set(p.label for p in dataset.places)), 4)
        self.assertEqual(dataset.summary["num_real_images"], 16)
        self.assertIn("1.2300_4.5600", alpha.real_paths[0].name)
        json.dumps(dataset.summary)

    def test_accepted_only_deduplicated_and_exact_source_inheritance(self):
        valid = self._candidate()
        self._manifest([
            valid, valid.copy(), {"passed": False, "eligible_for_training": True},
            {"passed": True, "eligible_for_training": False},
            {"passed": "true", "eligible_for_training": True},
        ])
        dataset = self._dataset(synthetic_manifest=self.manifest)
        counts = dataset.summary["manifest"]
        self.assertEqual(counts["rows"], 5)
        self.assertEqual(counts["accepted_rows"], 2)
        self.assertEqual(counts["ignored_rows"], 3)
        self.assertEqual(counts["duplicate_outputs"], 1)
        self.assertEqual(dataset.summary["num_synthetic_images"], 1)
        self.assertEqual(len(dataset.places[0].synthetic_paths), 1)
        self.assertTrue(all(not p.synthetic_paths for p in dataset.places[1:]))

    def test_sampling_shapes_dtype_unique_views_and_actual_exposure(self):
        import torch

        self._manifest([self._candidate(f"synthetic_{i}.jpg") for i in range(3)])
        dataset = self._dataset(synthetic_manifest=self.manifest, synthetic_fraction=0.5)
        random.seed(7)
        images, labels, kinds = dataset[0]
        self.assertEqual(tuple(images.shape), (4, 3, 8, 14))
        self.assertEqual(images.dtype, torch.float32)
        self.assertEqual(labels.dtype, torch.long)
        self.assertEqual(kinds.dtype, torch.bool)
        self.assertEqual(kinds.sum().item(), 2)
        self.assertEqual(labels.unique().numel(), 1)
        self.assertTrue(torch.isfinite(images).all())
        # Real views have distinguishable red-channel intensity; no replacement.
        real_pixels = images[~kinds, 0, 0, 0]
        self.assertEqual(real_pixels.unique().numel(), 2)
        batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
        self.assertEqual(tuple(batch[0].shape), (2, 4, 3, 8, 14))
        self.assertEqual(batch[1][:, 0].unique().numel(), 2)

    def test_fraction_one_keeps_real_and_synthetic_shortage_falls_back(self):
        self._manifest([self._candidate(f"synthetic_{i}.jpg") for i in range(4)])
        dataset = self._dataset(synthetic_manifest=self.manifest, synthetic_fraction=1)
        self.assertEqual(dataset[0][2].sum().item(), 3)
        self.assertEqual(dataset[1][2].sum().item(), 0)
        dataset = self._dataset(synthetic_manifest=self.manifest, synthetic_fraction=0)
        self.assertEqual(dataset[0][2].sum().item(), 0)

    def test_direct_sampling_is_reproducible_when_seeded(self):
        import torch

        dataset = self._dataset()
        random.seed(22)
        first = dataset[0]
        random.seed(22)
        second = dataset[0]
        for left, right in zip(first, second):
            self.assertTrue(torch.equal(left, right))

    def test_relative_manifest_project_and_source_image_root_paths(self):
        candidate = self._candidate()
        candidate["source_path"] = str(self.paths[("Alpha", 1, 0)].relative_to(self.gsv / "Images"))
        candidate["output_path"] = "generated.jpg"
        self._manifest([candidate])
        dataset = self._dataset(synthetic_manifest=self.manifest)
        self.assertEqual(dataset.summary["num_synthetic_images"], 1)
        candidate["output_path"] = "outputs/generated.jpg"
        self._manifest([candidate])
        # Resolve the AdaptVPR project-relative path via a manifest ancestor,
        # even when the command is invoked in another working directory.
        with patch("workflow.training_data.Path.cwd", return_value=self.gsv):
            dataset = self._dataset(synthetic_manifest=self.manifest)
        self.assertEqual(dataset.summary["num_synthetic_images"], 1)

    def test_ambiguous_relative_paths_are_rejected(self):
        candidate = self._candidate()
        Image.new("RGB", (8, 8)).save(self.root / "generated.jpg")
        candidate["output_path"] = "generated.jpg"
        self._manifest([candidate])
        with self.assertRaisesRegex(ValueError, "ambiguous output_path"):
            self._dataset(synthetic_manifest=self.manifest)

    def test_missing_output_or_source_metadata_is_actionable(self):
        candidate = self._candidate()
        candidate["output_path"] = str(self.root / "missing.jpg")
        self._manifest([candidate])
        with self.assertRaisesRegex(FileNotFoundError, "output_path does not resolve"):
            self._dataset(synthetic_manifest=self.manifest)
        unknown_source = self.root / "unknown_source.jpg"
        Image.new("RGB", (8, 8)).save(unknown_source)
        candidate = self._candidate()
        candidate["source_path"] = str(unknown_source)
        self._manifest([candidate])
        with self.assertRaisesRegex(ValueError, "not a selected GSV metadata image"):
            self._dataset(synthetic_manifest=self.manifest)

    def test_conflicting_synthetic_labels_are_rejected(self):
        first = self._candidate()
        second = first.copy()
        second["source_path"] = str(self.paths[("Beta", 1, 0)])
        self._manifest([first, second])
        with self.assertRaisesRegex(ValueError, "conflicting source place labels"):
            self._dataset(synthetic_manifest=self.manifest)

    def test_city_filter_excludes_other_manifest_city(self):
        self._manifest([self._candidate(source_key=("Beta", 1, 0))])
        dataset = self._dataset(synthetic_manifest=self.manifest, cities=["Alpha"])
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.summary["manifest"]["excluded_city_rows"], 1)
        self.assertEqual(dataset.summary["num_synthetic_images"], 0)

    def test_synthetic_views_do_not_rescue_places_missing_real_views(self):
        rows = [r for r in self.csv_rows["Alpha"] if r["place_id"] != "2" or r["panoid"] != "p2_3"]
        self._write_csv("Alpha", rows)
        self._manifest([self._candidate(source_key=("Alpha", 2, 0))])
        dataset = self._dataset(synthetic_manifest=self.manifest)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.summary["excluded_places"], 1)
        self.assertEqual(dataset.summary["num_synthetic_images"], 0)

    def test_bad_configuration_and_malformed_accepted_manifest(self):
        for kwargs in (
            {"images_per_place": 1}, {"min_images_per_place": 3},
            {"synthetic_fraction": 1.2}, {"synthetic_fraction": float("nan")},
            {"image_size": (0, 10)},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MixedGSVCitiesDataset(self.gsv, **kwargs)
        self._manifest([{"passed": True, "eligible_for_training": True}])
        with self.assertRaisesRegex(ValueError, "source_path must be a non-empty"):
            self._dataset(synthetic_manifest=self.manifest)
        self.manifest.write_text("{bad json}\n")
        with self.assertRaisesRegex(ValueError, "malformed JSON"):
            self._dataset(synthetic_manifest=self.manifest)

    def test_missing_metadata_image_fails_before_training(self):
        self.paths[("Alpha", 1, 0)].unlink()
        with self.assertRaisesRegex(FileNotFoundError, "missing real image"):
            self._dataset()

    def test_corrupt_image_raises_instead_of_fabricating_training_input(self):
        candidate = self._candidate()
        Path(candidate["output_path"]).write_text("not an image")
        self._manifest([candidate])
        dataset = self._dataset(synthetic_manifest=self.manifest)
        with self.assertRaisesRegex(RuntimeError, "Cannot decode training image"):
            dataset[0]


if __name__ == "__main__":
    unittest.main()
