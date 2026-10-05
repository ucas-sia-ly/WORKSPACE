from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms as T

from AdaptVPR.experiments.vpr_guidance.data import place_key_from_name, read_global_prompts
from AdaptVPR.experiments.vpr_guidance.mixed_salad import MixedGSVCitiesDataset, load_synthetic_manifest
from AdaptVPR.experiments.vpr_guidance.data import gsv_image_name


def make_mixed_fixture(directory, places=9):
    root = Path(directory) / "gsv"
    (root / "Dataframes").mkdir(parents=True)
    (root / "Images" / "Bangkok").mkdir(parents=True)
    rows, sources, synthetic = [], {}, []
    for place_id in range(1, places + 1):
        for i in range(4):
            row = {"city_id": "Bangkok", "place_id": place_id, "panoid": f"p{place_id}_{i}",
                   "year": 2017, "month": 1, "northdeg": i, "lat": 13.0, "lon": 100.0}
            rows.append(row)
            source = root / "Images" / "Bangkok" / gsv_image_name(row)
            Image.new("RGB", (8, 8), (255, place_id * 10, 0)).save(source)
            sources[place_id] = source
        gen = Path(directory) / f"s{place_id}.png"
        Image.new("RGB", (8, 8), (0, place_id * 10, 255)).save(gen)
        synthetic.append({"city": "Bangkok", "place_id": f"{place_id:07d}",
                          "source_path": str(sources[place_id]), "generated_path": str(gen),
                          "route": "global", "passed": True, "eligible_for_training": True})
    pd.DataFrame(rows).to_csv(root / "Dataframes" / "Bangkok.csv", index=False)
    manifest = Path(directory) / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in synthetic), encoding="utf-8")
    return root, manifest, synthetic


class DataPipelineTests(unittest.TestCase):
    def test_global_prompt_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.jsonl"
            rows = [
                {"sample_id": "g1", "source_id": "Bangkok_0000001_2017_01_000_0_0_p.jpg", "route": "global", "condition": "snow", "prompt": "snow"},
                {"sample_id": "l1", "source_id": "Bangkok_0000001_2017_01_000_0_0_p.jpg", "route": "local", "condition": "occlusion", "prompt": "bus"},
                {"sample_id": "g2", "source_id": "Bangkok_0000002_2017_01_000_0_0_q.jpg", "route": "global", "condition": "night", "prompt": "night"},
            ]
            path.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
            got = read_global_prompts(path, ["snow"])
            self.assertEqual([x["sample_id"] for x in got], ["g1"])

    def test_place_key(self):
        self.assertEqual(
            place_key_from_name("Bangkok_0000123_2019_06_090_13.1_100.1_abc.jpg"),
            ("Bangkok", "0000123"),
        )

    def test_manifest_only_keeps_verified_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted = root / "accepted.png"
            rejected = root / "rejected.png"
            Image.new("RGB", (8, 8)).save(accepted)
            Image.new("RGB", (8, 8)).save(rejected)
            manifest = root / "manifest.jsonl"
            rows = [
                {"city": "Bangkok", "place_id": "0000001", "generated_path": str(accepted), "source_path": str(accepted), "route": "global", "passed": True, "eligible_for_training": True},
                {"city": "Bangkok", "place_id": "0000002", "generated_path": str(rejected), "passed": False, "eligible_for_training": False},
            ]
            manifest.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
            got = load_synthetic_manifest(manifest)
            self.assertIn(("Bangkok", "0000001"), got)
            self.assertNotIn(("Bangkok", "0000002"), got)

    def test_epoch_quota_respects_unique_synthetic_capacity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "gsv"
            (root / "Dataframes").mkdir(parents=True)
            (root / "Images" / "Bangkok").mkdir(parents=True)

            rows = []
            for place_id in (1, 2):
                for i in range(4):
                    rows.append({
                        "city_id": "Bangkok",
                        "place_id": place_id,
                        "panoid": f"p{place_id}_{i}",
                        "year": 2017,
                        "month": 1,
                        "northdeg": i,
                        "lat": 13.0 + place_id,
                        "lon": 100.0 + i,
                    })
                    Image.new("RGB", (8, 8)).save(
                        root / "Images" / "Bangkok" / gsv_image_name(rows[-1])
                    )
            pd.DataFrame(rows).to_csv(root / "Dataframes" / "Bangkok.csv", index=False)

            synth_dir = Path(tmp) / "synth"
            synth_dir.mkdir()
            synth_paths = []
            # Only two unique synthetic images exist, one for each place.
            for place_id in (1, 2):
                path = synth_dir / f"s{place_id}.png"
                Image.new("RGB", (8, 8)).save(path)
                synth_paths.append(path)
            manifest = Path(tmp) / "manifest.jsonl"
            manifest.write_text(
                "\n".join(
                    json.dumps({
                        "city": "Bangkok",
                        "place_id": f"{place_id:07d}",
                        "generated_path": str(path),
                        "source_path": str(root / "Images" / "Bangkok" / gsv_image_name(rows[(place_id - 1) * 4])),
                        "route": "global",
                        "passed": True,
                        "eligible_for_training": True,
                    })
                    for place_id, path in zip((1, 2), synth_paths)
                ),
                encoding="utf-8",
            )

            # Request an unrealistically high synthetic share. The planner must
            # stop at two unique synthetic slots, not duplicate files to reach it.
            ds = MixedGSVCitiesDataset(
                root,
                manifest,
                cities=["Bangkok"],
                img_per_place=4,
                min_img_per_place=4,
                real_to_synth=(1, 1),
                seed=7,
            )
            stats = ds.mix_stats
            self.assertEqual(stats["synthetic_capacity_slots"], 2)
            self.assertEqual(stats["planned_synthetic_slots"], 2)
            self.assertTrue(stats["coverage_limited"])
            self.assertTrue(all(q <= 1 for q in ds._synthetic_quota.values()))

    def test_real_only_needs_no_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "gsv"
            (root / "Dataframes").mkdir(parents=True)
            (root / "Images" / "Bangkok").mkdir(parents=True)
            rows = []
            for i in range(4):
                rows.append({
                    "city_id": "Bangkok", "place_id": 1, "panoid": f"p{i}",
                    "year": 2017, "month": 1, "northdeg": i, "lat": 13.0, "lon": 100.0,
                })
            pd.DataFrame(rows).to_csv(root / "Dataframes" / "Bangkok.csv", index=False)
            ds = MixedGSVCitiesDataset(
                root, None, cities=["Bangkok"], img_per_place=4,
                min_img_per_place=4, real_to_synth=(1, 0), seed=1,
            )
            self.assertEqual(ds.mix_stats["planned_synthetic_slots"], 0)

    def test_repeated_manifest_file_cannot_inflate_unique_capacity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, manifest, records = make_mixed_fixture(tmp, places=2)
            manifest.write_text("\n".join(json.dumps(records[0]) for _ in range(4)), encoding="utf-8")
            ds = MixedGSVCitiesDataset(root, manifest, ["Bangkok"], real_to_synth=(1, 1))
            self.assertEqual(ds.mix_stats["synthetic_capacity_slots"], 1)
            self.assertEqual(ds.mix_stats["planned_synthetic_slots"], 1)

    def test_manifest_rejects_false_strings_non_global_and_wrong_source_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, manifest, records = make_mixed_fixture(tmp, places=2)
            for changes in ({"route": "local"}, {"place_id": "0000002"}):
                record = dict(records[0], **changes)
                manifest.write_text(json.dumps(record), encoding="utf-8")
                with self.assertRaises(ValueError):
                    MixedGSVCitiesDataset(root, manifest, ["Bangkok"])
            manifest.write_text(json.dumps(dict(records[0], passed="false")), encoding="utf-8")
            ds = MixedGSVCitiesDataset(root, manifest, ["Bangkok"])
            self.assertEqual(ds.mix_stats["planned_synthetic_slots"], 0)

    def test_actual_eight_to_one_exposure_and_place_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, manifest, _ = make_mixed_fixture(tmp)
            ds = MixedGSVCitiesDataset(root, manifest, ["Bangkok"], return_mix_metadata=True)
            ds.transform = T.ToTensor()
            actual_synthetic = actual_real = 0
            for images, labels, meta in DataLoader(ds, batch_size=3, num_workers=0):
                self.assertEqual(tuple(images.shape[1:]), (4, 3, 8, 8))
                self.assertTrue(torch.all(labels == labels[:, :1]))
                blue = (images[:, :, 2].mean(dim=(-1, -2)) > .9).sum(dim=1)
                self.assertTrue(torch.equal(blue, meta["synthetic_slots"]))
                for label, count in zip(labels[:, 0].tolist(), blue.tolist()):
                    self.assertEqual(count, ds._synthetic_quota[ds.keys[label]])
                for place_images, label in zip(images, labels[:, 0].tolist()):
                    expected_green = int(ds.keys[label][1]) * 10 / 255
                    # JPEG real images have small quantization error; synthetic
                    # PNGs carry an exact, distinct marker for every place.
                    self.assertTrue(torch.all((place_images[:, 1] - expected_green).abs() < 3 / 255))
                actual_synthetic += int(blue.sum())
                actual_real += int(meta["real_slots"].sum())
            self.assertEqual((actual_real, actual_synthetic), (32, 4))

    def test_shared_epoch_reaches_persistent_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, manifest, _ = make_mixed_fixture(tmp)
            ds = MixedGSVCitiesDataset(root, manifest, ["Bangkok"], return_mix_metadata=True)
            ds.transform = T.ToTensor()
            loader = DataLoader(ds, batch_size=3, num_workers=2, persistent_workers=True)
            list(loader)
            ds.set_epoch(7)
            for _, labels, meta in loader:
                self.assertTrue(torch.all(meta["epoch"] == 7))
                for label, count in zip(labels[:, 0].tolist(), meta["synthetic_slots"].tolist()):
                    self.assertEqual(count, ds._synthetic_quota[ds.keys[label]])
            del loader

    def test_distributed_padding_is_in_actual_batch_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, manifest, records = make_mixed_fixture(tmp)
            manifest.write_text("\n".join(json.dumps(records[0]) for _ in range(4)), encoding="utf-8")
            ds = MixedGSVCitiesDataset(root, manifest, ["Bangkok"], return_mix_metadata=True)
            ds.transform = T.ToTensor()
            synthetic = real = 0
            for rank in range(2):
                sampler = DistributedSampler(ds, num_replicas=2, rank=rank, shuffle=False)
                for _, _, meta in DataLoader(ds, batch_size=2, sampler=sampler):
                    synthetic += int(meta["synthetic_slots"].sum())
                    real += int(meta["real_slots"].sum())
            self.assertEqual(ds.mix_stats["planned_synthetic_slots"], 1)
            self.assertEqual((synthetic, real), (2, 38))

    def test_real_only_api_ignores_missing_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _ = make_mixed_fixture(tmp, places=2)
            ds = MixedGSVCitiesDataset(root, Path(tmp) / "missing.jsonl", ["Bangkok"], real_to_synth=(1, 0))
            self.assertEqual(ds.mix_stats["planned_synthetic_slots"], 0)

    def test_same_synthetic_file_cannot_have_two_valid_source_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, manifest, records = make_mixed_fixture(tmp, places=2)
            records[1]["generated_path"] = records[0]["generated_path"]
            manifest.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "conflicting place labels"):
                MixedGSVCitiesDataset(root, manifest, ["Bangkok"])

    def test_same_place_number_in_different_cities_stays_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, manifest, records = make_mixed_fixture(tmp, places=1)
            rome = pd.read_csv(root / "Dataframes" / "Bangkok.csv")
            rome["city_id"] = "Rome"
            rome.to_csv(root / "Dataframes" / "Rome.csv", index=False)
            (root / "Images" / "Rome").mkdir()
            for _, row in rome.iterrows():
                source = root / "Images" / "Rome" / gsv_image_name(row)
                Image.new("RGB", (8, 8), (255, 200, 0)).save(source)
            generated = Path(tmp) / "rome.png"
            Image.new("RGB", (8, 8), (0, 200, 255)).save(generated)
            records.append(dict(records[0], city="Rome", source_path=str(source),
                                generated_path=str(generated)))
            manifest.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
            ds = MixedGSVCitiesDataset(root, manifest, ["Bangkok", "Rome"], real_to_synth=(1, 1))
            ds.transform = T.ToTensor()
            self.assertEqual(len(set(ds.label_map.values())), 2)
            for index, key in enumerate(ds.keys):
                images, labels = ds[index]
                self.assertTrue(torch.all(labels == ds.label_map[key]))
                expected_green = (10 if key[0] == "Bangkok" else 200) / 255
                self.assertTrue(torch.all((images[:, 1] - expected_green).abs() < 3 / 255))


if __name__ == "__main__":
    unittest.main()
