"""Coverage sampling regression checks; CPU only, no model or network needed."""

import csv
import io
import math
import random
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from train_salad import parse_args
from workflow.training_data import (
    CoveragePlaceBatchSampler, MixedGSVCitiesDataset, SamplingCoverage, coverage_pk_indices,
)


class CoveragePermutationTests(unittest.TestCase):
    def test_boundaries_keep_every_permutation_and_each_group_unique(self):
        for n in range(2, 26):
            for k in range(2, n + 1):
                epochs = 3 * n // math.gcd(n, k)
                groups = [coverage_pk_indices(n, k, seed=42, city="Alpha", place_id=1, epoch=e)
                          for e in range(epochs)]
                with self.subTest(n=n, k=k):
                    self.assertTrue(all(len(group) == len(set(group)) == k for group in groups))
                    stream = [i for group in groups for i in group]
                    for start in range(0, len(stream), n):
                        self.assertEqual(sorted(stream[start:start + n]), list(range(n)))
                    # Direct random access, without replaying previous epochs.
                    for epoch in reversed(range(epochs)):
                        self.assertEqual(groups[epoch], coverage_pk_indices(
                            n, k, seed=42, city="Alpha", place_id=1, epoch=epoch))

    def test_stable_identity_and_global_rng_independence(self):
        options = dict(seed=13, city="Alpha", place_id=7, epoch=100003)
        random.seed(99)
        state = random.getstate()
        baseline = coverage_pk_indices(101, 4, **options)
        self.assertEqual(state, random.getstate())
        for name, value in (("seed", 14), ("city", "Beta"), ("place_id", 8)):
            self.assertNotEqual(baseline, coverage_pk_indices(101, 4, **{**options, name: value}))

    def test_one_visit_per_place_with_no_singleton_batches(self):
        for n in range(2, 26):
            for p in range(2, 10):
                sampler = CoveragePlaceBatchSampler(n, p, seed=42)
                sampler.set_epoch(7)
                batches = list(sampler)
                with self.subTest(n=n, p=p):
                    self.assertEqual(len(batches), len(sampler))
                    self.assertTrue(all(len(batch) >= 2 for batch in batches))
                    self.assertEqual(sorted(index for batch in batches for _, index in batch), list(range(n)))
                    self.assertEqual({epoch for batch in batches for epoch, _ in batch}, {7})
                    self.assertEqual(batches, list(sampler))


class CoverageDatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "Dataframes").mkdir()
        for city in ("Alpha", "Beta"):
            image_dir = self.root / "Images" / city
            image_dir.mkdir(parents=True)
            rows = []
            for place in range(3):
                for view in reversed(range(5 + place)):
                    panoid = f"p{place}v{view}"
                    rows.append([place, 2020, 1, 0, city, "1.0", "2.0", panoid])
                    path = image_dir / f"{city}_{place:07d}_2020_01_000_1.0_2.0_{panoid}.jpg"
                    Image.new("RGB", (8, 8), (view * 30, place * 30, 100)).save(path)
            with (self.root / "Dataframes" / f"{city}.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["place_id", "year", "month", "northdeg", "city_id", "lat", "lon", "panoid"])
                writer.writerows(rows)

    def dataset(self, **options):
        return MixedGSVCitiesDataset(self.root, images_per_place=4, min_images_per_place=4,
                                     augment=False, image_size=(8, 8), return_sampling_metadata=True,
                                     **options)

    def test_random_default_retains_original_rng_calls(self):
        dataset = self.dataset()
        self.assertEqual(dataset.real_sampling, "random_pk")
        random.seed(21)
        expected = [(path, False) for path in random.sample(dataset.places[0].real_paths, 4)]
        random.sample([], 0)
        random.shuffle(expected)
        expected_state = random.getstate()
        random.seed(21)
        self.assertEqual(expected, dataset.sample_paths(0))
        self.assertEqual(expected_state, random.getstate())
        args = parse_args(["--real-data", str(self.root), "--output-dir", str(self.root / "out")])
        self.assertEqual(args.real_sampling, "random_pk")

    def test_full_pool_stable_order_and_direct_epoch(self):
        dataset = self.dataset(real_sampling="coverage_pk", seed=17)
        self.assertEqual([len(p.real_paths) for p in dataset.places], [5, 6, 7, 5, 6, 7])
        for place in dataset.places:
            self.assertEqual(place.real_paths, sorted(place.real_paths, key=lambda p: p.name))
        dataset.set_epoch(13)
        self.assertEqual(dataset.sample_paths(0), dataset.sample_paths(0, epoch=13))
        first = dataset[(13, 0)][-1]
        dataset.set_epoch(99)
        self.assertTrue(torch.equal(first["real_indices"], dataset[(13, 0)][-1]["real_indices"]))

    def collect(self, workers, epochs, persistent=False):
        dataset = self.dataset(real_sampling="coverage_pk", seed=17)
        sampler = CoveragePlaceBatchSampler(len(dataset), 2, seed=17)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=workers,
                            persistent_workers=persistent)
        result = {}
        # Deliberately leave dataset.epoch stale: keys must convey epoch to all workers.
        for epoch in epochs:
            sampler.set_epoch(epoch)
            result[epoch] = [(batch[-1]["place_index"].tolist(), batch[-1]["real_indices"].tolist())
                             for batch in loader]
        return result

    def test_workers_persistent_workers_and_direct_resume(self):
        baseline = self.collect(0, range(4))
        for workers in (2, 4):
            with self.subTest(workers=workers):
                self.assertEqual(baseline, self.collect(workers, range(4), persistent=True))
                self.assertEqual({3: baseline[3]}, self.collect(workers, [3]))
        self.assertEqual({3: baseline[3]}, self.collect(0, [3]))

    def test_actual_exposure_statistics_and_checkpoint_resume(self):
        dataset = self.dataset(real_sampling="coverage_pk")
        coverage = SamplingCoverage(dataset)
        metadata = {"place_index": torch.tensor([0, 1]),
                    "real_indices": torch.tensor([[0, 1, -1, -1], [0, 1, 2, 3]])}
        coverage.update(metadata)
        self.assertEqual(coverage.summary()["epoch_unique_real_images"], 6)
        buffer = io.BytesIO()
        torch.save(coverage.state_dict(), buffer)
        buffer.seek(0)
        restored = SamplingCoverage(dataset, state=torch.load(buffer, weights_only=True))
        coverage.begin_epoch()
        for tracker in (coverage, restored):
            tracker.update(metadata)
            tracker.update({"place_index": torch.tensor([0]),
                            "real_indices": torch.tensor([[0, 2, -1, -1]])})
        self.assertEqual(coverage.state_dict(), restored.state_dict())
        self.assertEqual(coverage.summary(), restored.summary())
        summary = coverage.summary()
        self.assertEqual(summary["epoch_unique_real_images"], 7)
        self.assertEqual(summary["epoch_repeated_real_exposures"], 1)
        self.assertEqual(summary["epoch_previously_seen_real_exposures"], 7)
        self.assertEqual(summary["cumulative_unique_real_images"], 7)
        self.assertEqual(summary["cumulative_repeated_real_exposures"], 7)
        self.assertEqual(summary["cumulative_real_coverage"], 7 / 36)
        places = coverage.place_statistics()
        self.assertEqual(places[0]["epoch_visits"], 2)
        self.assertEqual(places[0]["cumulative_visits"], 3)
        self.assertEqual(places[0]["epoch_sampling_frequency"], 2 / 3)
        self.assertEqual(places[2]["epoch_sampling_frequency"], 0)


if __name__ == "__main__":
    unittest.main()
