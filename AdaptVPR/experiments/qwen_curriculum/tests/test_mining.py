"""Training-only source difficulty, quotas and disk cache without model downloads."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mine_sources import (allocate_city_quotas, build_place_context, extract_descriptor_cache,
                         image_inventory, load_training_sources, score_source_difficulty,
                         select_hard_sources, sequence_fingerprint, fingerprint, validate_descriptors)


class DifficultyTests(unittest.TestCase):
    def setUp(self):
        self.descriptors = np.array([[1., 0.], [.8, .6], [.9, np.sqrt(.19)],
                                     [.9, np.sqrt(.19)], [0., 1.], [0., 1.]], dtype=np.float32)
        self.labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        self.coords = np.array([[0., 0.], [0., 0.], [1., 1.], [1., 1.], [2., 2.], [2., 2.]])

    def score(self, **kwargs):
        return score_source_difficulty(self.descriptors, self.labels, self.coords,
                                       query_chunk_size=2, reference_chunk_size=1, **kwargs)

    def test_positive_is_leave_one_out_and_own_centroid_is_not_negative(self):
        scores = self.score(near_negative_radius_m=0)
        self.assertAlmostEqual(float(scores["own_loo_positive_similarity"][0]), .8, places=6)
        self.assertAlmostEqual(float(scores["hardest_negative_similarity"][0]), .9, places=6)
        self.assertAlmostEqual(float(scores["hardness"][0]), .1, places=6)
        self.assertEqual(int(scores["centroid_positive_rank"][0]), 2)
        self.assertEqual(int(scores["num_negative_places"][0]), 2)
        self.assertTrue(scores["reliable"][0])

    def test_nearby_other_place_is_excluded_but_remote_place_is_retained(self):
        self.coords[2:4] = [0., .0001]  # About eleven metres away.
        scores = self.score()
        self.assertAlmostEqual(float(scores["hardest_negative_similarity"][0]), 0., places=6)
        self.assertEqual(int(scores["num_negative_places"][0]), 1)
        self.assertEqual(int(scores["centroid_positive_rank"][0]), 1)
        self.coords[4:] = [0., .00015]
        scores = self.score()
        self.assertFalse(scores["reliable"][0])
        self.assertEqual(int(scores["num_negative_places"][0]), 0)
        self.assertEqual(int(scores["centroid_positive_rank"][0]), 0)
        self.assertTrue(np.isfinite(scores["hardness"]).all())

    def test_chunked_scores_match_and_place_sums_can_be_memmapped(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = self.score(near_negative_radius_m=0)
            actual = score_source_difficulty(self.descriptors, self.labels, self.coords,
                query_chunk_size=3, reference_chunk_size=3, near_negative_radius_m=0, cache_dir=directory)
            for name in expected:
                np.testing.assert_allclose(actual[name], expected[name], atol=2e-6)
            sums = np.load(Path(directory) / "place_sums.npy", mmap_mode="r")
            self.assertIsInstance(sums, np.memmap)
            np.testing.assert_allclose(sums[0], [1.8, .6], atol=1e-6)

    def test_weak_real_positive_is_unreliable_and_bad_vectors_labels_fail(self):
        scores = self.score(min_positive_similarity=.85)
        self.assertFalse(scores["reliable"][0])
        for descriptors in (np.zeros((2, 2)), np.array([[1., np.nan]]), np.array([[1., np.inf]]),
                            np.array([[1, 2]], dtype=np.int64), np.empty((0, 2))):
            with self.assertRaises(ValueError):
                validate_descriptors(descriptors)
        for labels in (np.array([0., 0., 1., 1., 2., 2.]), np.array([0, 0, 4, 4, 5, 5]),
                       np.array([0, 0, 1, 1, 2, -1]), np.array([0, 0, 0, 0, 0, 1])):
            with self.assertRaises(ValueError):
                score_source_difficulty(self.descriptors, labels, self.coords)
        invalid_coords = self.coords.copy()
        invalid_coords[0, 0] = 91
        with self.assertRaises(ValueError):
            build_place_context(self.descriptors, self.labels, invalid_coords)


class SelectionTests(unittest.TestCase):
    def make_inputs(self):
        records = [{"city": city, "place_id": i // 3, "source_path": f"/{city}/{i}.jpg",
                    "source_id": f"{city}/{i}.jpg", "source_index": offset + i}
                   for offset, city in ((0, "A"), (12, "B")) for i in range(12)]
        scores = {"hardness": np.array(list(range(12)) * 2, dtype=np.float32),
                  "own_loo_positive_similarity": np.full(24, .5),
                  "hardest_negative_similarity": np.full(24, .6),
                  "centroid_positive_rank": np.ones(24, dtype=np.int64),
                  "num_negative_places": np.full(24, 7, dtype=np.int64),
                  "reliable": np.ones(24, dtype=bool)}
        return records, scores

    def test_even_quota_and_per_place_cap_are_reproducible(self):
        records, scores = self.make_inputs()
        one, audit = select_hard_sources(records, scores, 8, min_quantile=0, max_quantile=1, max_sources_per_place=2)
        two, _ = select_hard_sources(records, scores, 8, min_quantile=0, max_quantile=1, max_sources_per_place=2)
        self.assertEqual(one, two)
        self.assertEqual(audit["quotas"], {"A": 4, "B": 4})
        counts = {}
        for row in one:
            key = row["city"], row["place_id"]
            counts[key] = counts.get(key, 0) + 1
        self.assertLessEqual(max(counts.values()), 2)
        self.assertEqual(len({row["source_path"] for row in one}), 8)

    def test_quantile_tail_band_and_insufficient_capacity_are_explicit(self):
        records, scores = self.make_inputs()
        selected, audit = select_hard_sources(records, scores, 2, min_quantile=.70, max_quantile=.975)
        self.assertTrue(all(7.7 <= row["hardness"] <= 10.725 for row in selected))
        self.assertEqual(audit["cities"]["A"]["band_sources"], 3)
        with self.assertRaisesRegex(ValueError, "Reduce --num-sources"):
            select_hard_sources(records, scores, 8, min_quantile=.70, max_quantile=.975)
        scores["reliable"][:] = False
        with self.assertRaisesRegex(ValueError, "Only 0 eligible"):
            select_hard_sources(records, scores, 2)

    def test_capped_quota_redistribution_and_proportions(self):
        self.assertEqual(allocate_city_quotas({"A": 1, "B": 10}, 5), {"A": 1, "B": 4})
        self.assertEqual(allocate_city_quotas({"A": 10, "B": 10}, 2, {"A": 1, "B": 2}), {"A": 1, "B": 1})
        self.assertEqual(allocate_city_quotas({"A": 10, "B": 10}, 8, {"A": 1, "B": 3}), {"A": 2, "B": 6})
        self.assertEqual(allocate_city_quotas({"A": 0, "B": 10}, 3), {"A": 0, "B": 3})

    def test_duplicate_source_and_invalid_place_or_scores_are_rejected(self):
        records, scores = self.make_inputs()
        records[1]["source_path"] = records[0]["source_path"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            select_hard_sources(records, scores, 2)
        records, scores = self.make_inputs()
        records[0]["place_id"] = "0"
        with self.assertRaisesRegex(ValueError, "place_id"):
            select_hard_sources(records, scores, 2)
        records, scores = self.make_inputs()
        scores["hardness"][0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            select_hard_sources(records, scores, 2)


class MetadataCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "Dataframes").mkdir()
        (self.root / "Images" / "City").mkdir(parents=True)
        self.rows = []
        for place in range(2):
            for view in range(2):
                row = {"place_id": str(place), "year": "2020", "month": "01", "northdeg": str(view),
                       "city_id": "City", "lat": str(place), "lon": "0", "panoid": f"p{place}v{view}"}
                self.rows.append(row)
                name = f"City_{place:07d}_2020_01_{view:03d}_{place}_0_p{place}v{view}.jpg"
                Image.new("RGB", (9, 7), (50 + place * 60, 80 + view * 20, 130)).save(self.root / "Images" / "City" / name)
        self.write_csv(self.rows)

    def write_csv(self, rows):
        with (self.root / "Dataframes" / "City.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_exact_metadata_paths_deduplicate_and_invalid_labels_fail(self):
        self.write_csv(self.rows + [self.rows[0]])
        records, cities = load_training_sources(self.root, ["City"], 2)
        self.assertEqual(cities, ["City"])
        self.assertEqual(len(records), 4)
        self.assertEqual([row["place_index"] for row in records], [0, 0, 1, 1])
        self.assertEqual(records[0]["source_id"], "City/City_0000000_2020_01_000_0_0_p0v0.jpg")
        self.assertEqual(image_inventory(records), image_inventory(records))
        self.assertEqual(sequence_fingerprint(records), fingerprint(records))
        bad = [dict(row) for row in self.rows]
        bad[0]["place_id"] = "wrong"
        self.write_csv(bad)
        with self.assertRaisesRegex(ValueError, "integers"):
            load_training_sources(self.root, ["City"], 2)

    def test_streamed_cache_resume_skips_model_and_partial_marker_recomputes(self):
        records, _ = load_training_sources(self.root, ["City"], 2)
        calls = []

        class TinyModel(torch.nn.Module):
            image_size = (4, 5)

            def forward(self, batch):
                calls.append(tuple(batch.shape))
                return batch.mean(dim=(2, 3))

        model_calls = []

        def factory(*args, **kwargs):
            model_calls.append(1)
            return TinyModel()

        args = SimpleNamespace(checkpoint=self.root / "unused.pt", device="cpu", backbone_repo=None,
                               batch_size=3, num_workers=0)
        request = {"fingerprint": "test-config"}
        cache = self.root / "cache"
        descriptors, metadata = extract_descriptor_cache(args, records, request, cache, model_factory=factory)
        self.assertIsInstance(descriptors, np.memmap)
        self.assertEqual(descriptors.shape, (4, 3))
        self.assertEqual(calls, [(3, 3, 4, 5), (1, 3, 4, 5)])
        np.testing.assert_allclose(np.linalg.norm(descriptors, axis=1), 1., atol=1e-6)
        resumed, resumed_meta = extract_descriptor_cache(args, records, request, cache, model_factory=factory)
        self.assertEqual(len(model_calls), 1)
        self.assertEqual(metadata, resumed_meta)
        np.testing.assert_array_equal(descriptors, resumed)
        (cache / "descriptor_complete.json").unlink()
        rebuilt, _ = extract_descriptor_cache(args, records, request, cache, model_factory=factory)
        self.assertEqual(len(model_calls), 2)
        np.testing.assert_array_equal(descriptors, rebuilt)


if __name__ == "__main__":
    unittest.main()
