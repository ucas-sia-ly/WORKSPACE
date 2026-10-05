from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from AdaptVPR.experiments.vpr_guidance.data import place_key_from_name, read_global_prompts
from AdaptVPR.experiments.vpr_guidance.mixed_salad import MixedGSVCitiesDataset, load_synthetic_manifest


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
                {"city": "Bangkok", "place_id": "0000001", "generated_path": str(accepted), "passed": True, "eligible_for_training": True},
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


if __name__ == "__main__":
    unittest.main()
