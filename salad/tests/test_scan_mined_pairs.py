"""Bounded input-replay and summary checks for the forward-only scan."""
import random
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from scan_mined_pairs import FixedScanDataset, make_schedule, summarize, tensor_digest, run_worker
from workflow.mining_diagnostics import source_aware_mined_pair_statistics, common_anchor_statistics


class FakeDataset:
    def __init__(self, *args, **kwargs):
        self.places = [SimpleNamespace(city="city", place_id=i, synthetic_paths=[Path(f"/{i}/synthetic.jpg")]) for i in range(64)]
        self.synthetic_fraction = kwargs.get("synthetic_fraction", 0.)

    def __len__(self):
        return len(self.places)

    def sample_paths(self, index, epoch=0):
        return [(Path(f"/{index}/{epoch}/{i}.jpg"), i == 3 and self.synthetic_fraction > 0) for i in range(4)]

    def __getitem__(self, key):
        epoch, index = key
        return (torch.full((4, 3, 2, 2), random.random()), torch.full((4,), index),
                torch.tensor([flag for _, flag in self.sample_paths(index, epoch)]))


class TinyModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.layer = torch.nn.Linear(12, 4)

    def forward(self, images):
        return self.layer(images.flatten(1))


class ScanReplayTest(unittest.TestCase):
    def test_schedule_unique_places_reproducible_exhaustion(self):
        first = make_schedule(65, 5, 42)
        self.assertEqual(first, make_schedule(65, 5, 42))
        self.assertNotEqual(first, make_schedule(65, 5, 43))
        for batch in first:
            self.assertEqual(len(batch), 32)
            self.assertEqual(len(set(index for _, index in batch)), 32)
        self.assertEqual([batch[0][0] for batch in first], [0, 0, 1, 1, 2])
        with self.assertRaises(ValueError):
            make_schedule(31, 1, 42)

    def test_tensor_replay_independent_of_workers_and_external_random(self):
        dataset = FixedScanDataset(FakeDataset(), 43)
        schedule = make_schedule(len(dataset), 2, 44)
        hashes = []
        for workers in (0, 2):
            random.seed(workers + 5)
            before = random.getstate()
            loader = DataLoader(dataset, batch_sampler=schedule, num_workers=workers,
                                generator=torch.Generator().manual_seed(1))
            hashes.append([tensor_digest(images) for images, _, _, _ in loader])
            self.assertEqual(random.getstate(), before)
        self.assertEqual(hashes[0], hashes[1])

    def test_source_counts_survive_summary_and_weight_anchors(self):
        descriptors = torch.tensor([[1., 0.], [.8, .6], [.6, .8], [0., 1.]]).repeat(8, 1)
        labels = torch.arange(16).repeat_interleave(2)
        flags = torch.tensor([False, True]).repeat(16)
        stats = source_aware_mined_pair_statistics(descriptors, labels, flags)
        stats["common_8_place_anchors"] = common_anchor_statistics(stats, 32)
        row = dict(checkpoint="initial", cohort="training_subset", composition="mix", mode="eval_fp32", P=8, **stats)
        aggregate, = summarize([row, row])
        self.assertEqual(aggregate["anchors"], 64)
        self.assertEqual(aggregate["real_anchors"], 32)
        self.assertEqual(aggregate["synthetic_anchors"], 32)
        self.assertEqual(aggregate["real_synthetic_mined_positive"], 2 * stats["source_pairs"]["real_synthetic"]["mined_positive"])
        self.assertEqual(aggregate["mean_synthetic_fraction"], .5)

    def test_complete_cpu_scan_preserves_checkpoint_and_excludes_reserved_places(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "initial.pt"
            torch.save(TinyModel().state_dict(), checkpoint)
            manifest = root / "train_manifest.jsonl"
            manifest.write_text("{}\n")
            plan = dict(config=dict(checkpoint=str(checkpoint), real_data="unused", cities=["city"],
                backbone_repo="unused", num_trainable_blocks=1, mix_fraction=.5),
                train_manifest=str(manifest), heldout_places=[["city", 0]],
                calibration_pairs=[dict(city="city", place_id=1)])
            (root / "plan.json").write_text(json.dumps(plan))
            out = root / "scan"
            out.mkdir()
            args = SimpleNamespace(experiment_dir=root, device="cpu", modes=["eval_fp32"],
                torch_threads=1, checkpoints=["initial"], training_seed=42, seed=42,
                no_augmentation=False, cohorts=["training_subset", "full_real"], batches=1,
                compositions=["real", "mix"], num_workers=0, forward_batch_size=128, save_per_anchor=False)
            before = checkpoint.read_bytes()
            with patch("workflow.model.SALADModel", TinyModel), patch(
                "workflow.model.checkpoint_state_and_config", side_effect=lambda state: (state, {"backbone_config": {}})), patch(
                "workflow.training_data.MixedGSVCitiesDataset", FakeDataset):
                run_worker(args, out)
            self.assertEqual(checkpoint.read_bytes(), before)
            protocol = json.loads((out / "protocol.json").read_text())
            self.assertTrue(protocol["parameter_state_unchanged"])
            self.assertTrue(protocol["parameter_gradients_absent"])
            self.assertEqual(protocol["optimizer_steps"], 0)
            self.assertEqual(protocol["backward_passes"], 0)
            self.assertEqual(protocol["excluded_places"], 2)
            self.assertEqual(protocol["places"]["full_real"], 62)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(len(summary), 12)
            self.assertEqual(len((out / "batch_manifest.jsonl").read_text().splitlines()), 4)


if __name__ == "__main__":
    unittest.main()
