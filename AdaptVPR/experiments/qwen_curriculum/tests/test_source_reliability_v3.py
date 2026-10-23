"""Small CPU proofs for the archived source-reliability refinement."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.qwen_curriculum import common, source_reliability_v3 as reliability, source_screen_v3


class ReliabilityTests(unittest.TestCase):
    def fixture(self, root):
        gsv = root / "gsv"
        (gsv / "Dataframes").mkdir(parents=True)
        (gsv / "Images/City").mkdir(parents=True)
        rows, records = [], []
        for place in range(2):
            for view in range(4):
                row = dict(city_id="City", place_id=place, year=2020, month=1, northdeg=view,
                           lat="1.5", lon="2.5", panoid=f"p{place}v{view}")
                rows.append(row)
                filename = f"City_{place:07d}_2020_01_{view:03d}_1.5_2.5_p{place}v{view}.jpg"
                path = gsv / "Images/City" / filename
                image = np.random.default_rng(place * 10 + view).integers(20, 235, (150, 200, 3), dtype=np.uint8)
                Image.fromarray(image).save(path)
                records.append(dict(source_path=str(path), source_id=f"City/{filename}", city="City", place_id=place,
                                    latitude=1.5, longitude=2.5, place_index=place, source_index=len(records)))
        metadata = gsv / "Dataframes/City.csv"
        with metadata.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, rows[0])
            writer.writeheader()
            writer.writerows(rows)
        inventory = hashlib.sha256()
        for record in records:
            stat = Path(record["source_path"]).stat()
            inventory.update(json.dumps([record["source_path"], stat.st_size, stat.st_mtime_ns],
                                        separators=(",", ":")).encode() + b"\0")
        request = dict(real_data=str(gsv), cities=["City"], min_images_per_place=4,
                       metadata_sha256={"City": common.file_sha256(metadata)},
                       source_order_sha256=common.fingerprint(records),
                       image_inventory=dict(method="metadata_image_path_size_mtime", files=8, sha256=inventory.hexdigest()))
        request["fingerprint"] = common.fingerprint(request)
        scoring = dict(descriptor_fingerprint=request["fingerprint"], min_positive_similarity=0.15)
        scoring["fingerprint"] = common.fingerprint(scoring)
        config = dict(descriptor_request=request, scoring=scoring)
        config["fingerprint"] = common.fingerprint(config)
        positive = np.array([.08, .6, .6, .6, .6, .6, .6, .6], dtype=np.float32)
        buffer = io.BytesIO()
        np.savez(buffer, own_loo_positive_similarity=positive, hardest_negative_similarity=np.ones(8, np.float32) * .2,
                 hardness=np.ones(8, np.float32) * -.4, centroid_positive_rank=np.ones(8, np.int64),
                 num_negative_places=np.ones(8, np.int64), reliable=positive >= .15)
        score = buffer.getvalue()
        scope = "mining_23city/cities/City"
        cache = f"{scope}/cache/{request['fingerprint'][:20]}"
        entries = {f"{scope}/mining_config.json": json.dumps(config).encode(),
                   f"{cache}/source_scores.npz": score,
                   f"{cache}/scores_complete.json": json.dumps(dict(config_fingerprint=scoring["fingerprint"],
                                                                  sha256=hashlib.sha256(score).hexdigest())).encode()}
        archive = root / "legacy.tar.gz"
        self.archive(archive, entries)
        return gsv, archive, records, entries

    def archive(self, path, entries):
        with tarfile.open(path, "w:gz") as stream:
            for name, content in entries.items():
                member = tarfile.TarInfo(name)
                member.size = len(content)
                stream.addfile(member, io.BytesIO(content))

    def test_all_view_mapping_and_same_place_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gsv, archive, records, _ = self.fixture(root)
            mapping, places, report = reliability.verified_score_map(gsv, archive)
            self.assertEqual(len(mapping), 8)
            self.assertEqual(len(places), 2)
            self.assertFalse(mapping[records[0]["source_path"]]["reliable"])
            self.assertEqual(report["places_with_reliable_views"], 2)
            selected = [{**source_screen_v3.screen_image(records[index]["source_path"]),
                         "city": "City", "place_id": records[index]["place_id"]} for index in (0, 4)]
            original = reliability.verified_score_map
            with patch.object(reliability, "verified_score_map", side_effect=lambda data: original(data, archive)):
                refined, audit = reliability.refine_sources(selected, gsv, root / "refined")
            self.assertEqual(audit["replaced_views"], 1)
            self.assertEqual({r["place_id"] for r in refined}, {0, 1})
            self.assertNotEqual(refined[0]["source_path"], selected[0]["source_path"])
            self.assertEqual(refined[1]["source_path"], selected[1]["source_path"])
            self.assertTrue(all(r["source_reliability"]["reliable"] for r in refined))
            self.assertEqual(audit["output_file_sha256"], common.file_sha256(root / "refined/refined_sources.jsonl"))
            self.assertTrue((root / "refined/reliability_config.json").is_file())

    def test_inventory_and_csv_changes_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gsv, archive, records, _ = self.fixture(root)
            path = Path(records[0]["source_path"])
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
            with self.assertRaisesRegex(ValueError, "inventory changed"):
                reliability.verified_score_map(gsv, archive)
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            csv_path = gsv / "Dataframes/City.csv"
            csv_path.write_bytes(csv_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "CSV changed"):
                reliability.verified_score_map(gsv, archive)

    def test_score_bytes_and_missing_city_proofs_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gsv, archive, _, entries = self.fixture(root)
            name = next(k for k in entries if k.endswith("source_scores.npz"))
            original = entries[name]
            entries[name] += b"broken"
            self.archive(archive, entries)
            with self.assertRaisesRegex(ValueError, "checksum/completion"):
                reliability.verified_score_map(gsv, archive)
            entries[name] = original
            self.archive(archive, entries)
            (gsv / "Dataframes/Uncovered.csv").write_text("city_id,place_id\n")
            with self.assertRaisesRegex(ValueError, "exactly all current cities"):
                reliability.verified_score_map(gsv, archive)

    def test_unusable_place_is_explicitly_excluded_without_spending_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gsv, archive, records, _ = self.fixture(root)
            selected = [{**source_screen_v3.screen_image(records[0]["source_path"]), "city": "City", "place_id": 0}]
            original = reliability.verified_score_map
            with patch.object(reliability, "verified_score_map", side_effect=lambda data: original(data, archive)), \
                 patch.object(source_screen_v3, "screen_image", return_value={"passed": False, "reasons": ["unusable"]}):
                refined, report = reliability.refine_sources(selected, gsv, root / "refined")
            self.assertEqual(refined, [])
            self.assertEqual(report["skipped_places"], 1)
            self.assertEqual(report["skipped"][0]["place_id"], 0)
            self.assertEqual(report["skipped"][0]["reason"], "no_reliable_cpu_suitable_distinct_view")
            self.assertEqual((root / "refined/refined_sources.jsonl").read_text(), "")


if __name__ == "__main__":
    unittest.main()
