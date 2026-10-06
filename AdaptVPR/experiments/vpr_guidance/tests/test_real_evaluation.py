"""Protocol and exact-retrieval regression tests; no synthetic VPR benchmark."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image
import torch
from torch import nn

from AdaptVPR.experiments.vpr_guidance.evaluate_real import (
    retrieval_ranks, recall_metrics, official_pose_metrics, load_checkpoint,
)
from AdaptVPR.experiments.vpr_guidance.real_data import (
    metadata_split, svox_split, nordland_split, robotcar_public_pose_split, read_colmap_reference_poses,
)


def image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(path)


class RealEvaluationTests(unittest.TestCase):
    def test_svox_uses_documented_utm_and_only_existing_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image(root / "images/test/gallery/@100@200@2012@p@F.jpg")
            image(root / "images/test/gallery/@140@200@2012@q@F.jpg")
            image(root / "images/test/queries_night/@105@200@timestamp@.jpg")
            split = svox_split(root)
            self.assertEqual(split.conditions, ["night"])
            self.assertEqual(split.positives, [[0]])
            image(root / "images/test/queries_night/unknown.jpg")
            with self.assertRaises(ValueError):
                svox_split(root)

    def test_nordland_uses_numeric_frame_metadata_not_lexical_filename_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.txt").write_text("A positive is within 10 frames.")
            for i in (12, 2, 20):
                image(root / f"images/test/database/@0@{i*2.3}@@@@@{i}@@@@@@@@.jpg")
            image(root / "images/test/queries/@0@27.6@@@@@12@@@@@@@@.jpg")
            split = nordland_split(root, root)
            self.assertEqual(split.positives, [[0, 1, 2]])
            self.assertIn("@2@", split.references[0].name)

    def test_explicit_metadata_positive_paths_and_no_positive_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image(root / "ref.jpg"); image(root / "query.jpg")
            metadata = root / "evaluation.json"
            payload = {"protocol": {"name": "official_mapping", "source": "official_fixture"},
                       "references": [{"path": "ref.jpg"}],
                       "queries": [{"path": "query.jpg", "condition": "dusk", "positives": ["ref.jpg"]}]}
            metadata.write_text(json.dumps(payload))
            self.assertEqual(metadata_split(root, metadata).positives, [[0]])
            payload["queries"][0]["positives"] = []
            metadata.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                metadata_split(root, metadata)

    def test_real_full_rank_is_not_truncated_at_ten(self):
        refs = np.eye(12, dtype=np.float32)
        query = np.arange(12, 0, -1, dtype=np.float32)[None]
        ranks, top1 = retrieval_ranks(refs, query, [[11]])
        self.assertEqual(ranks.tolist(), [12])
        self.assertEqual(top1.tolist(), [0])
        self.assertEqual(recall_metrics(ranks)["mean_rank"], 12)
        self.assertEqual(recall_metrics(ranks)["R@10"], 0)

    def test_pose_metric_requires_joint_translation_and_rotation(self):
        split = SimpleNamespace(reference_poses=[{"center_m": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]}],
                                query_poses=[{"center_m": [.1, 0, 0], "quaternion_wxyz": [0, 0, 0, 1]}])
        result = official_pose_metrics(split, [0], [0])
        self.assertTrue(all(value == 0 for value in result["localization_accuracy_percent"].values()))
        split.query_poses[0]["quaternion_wxyz"] = [-1, 0, 0, 0]
        result = official_pose_metrics(split, [0], [0])
        self.assertTrue(all(value == 100 for value in result["localization_accuracy_percent"].values()))

    def test_native_public_pose_and_colmap_archive_without_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "3D-models/all-merged"
            directory.mkdir(parents=True)
            # COLMAP world->camera translation -10 -> world camera center +10.
            with zipfile.ZipFile(directory / "001_aligned.zip", "w") as archive:
                archive.writestr("001_aligned/images.txt", "# COLMAP\n1 1 0 0 0 -10 0 0 1 overcast-reference/rear/1.png\n\n")
            image(root / "images/overcast-reference/rear/1.jpg")
            image(root / "images/night/rear/2.jpg")
            (root / "metadata").mkdir()
            (root / "metadata/robotcar_v2_train.txt?utm_source=chatgpt.com").write_text(
                "night/rear/2.jpg 1 0 0 11 0 1 0 0 0 0 1 0 0 0 0 1\n")
            split = robotcar_public_pose_split(root)
            self.assertEqual(split.reference_poses[0]["center_m"], [10., 0., 0.])
            self.assertEqual(split.query_poses[0]["center_m"], [11., 0., 0.])
            self.assertEqual(split.positives, [[0]])
            self.assertFalse(split.protocol["official_hidden_test"])
            self.assertFalse((directory / "001_aligned").exists())

    def test_fresh_lightning_and_raw_state_load_without_manual_conversion(self):
        class FakeVPR(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(1))
                self.architecture = kwargs
        FakeVPR.__module__ = "vpr_model"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module = SimpleNamespace(__file__=str(root / "vpr_model.py"), VPRModel=FakeVPR)
            with patch.dict("sys.modules", {"vpr_model": module}):
                path = root / "fresh.ckpt"
                torch.save({"state_dict": {"weight": torch.tensor([3.])},
                            "hyper_parameters": {"backbone_arch": "declared_arch"}}, path)
                model, arch = load_checkpoint(path, root, "cpu")
                self.assertEqual(arch["backbone_arch"], "declared_arch")
                self.assertEqual(float(model.weight), 3.)
                torch.save({"weight": torch.tensor([4.])}, path)
                model, _ = load_checkpoint(path, root, "cpu")
                self.assertEqual(float(model.weight), 4.)


if __name__ == "__main__":
    unittest.main()
