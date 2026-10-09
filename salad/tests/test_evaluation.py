"""Meaningful metadata/protocol and exact-retrieval checks; no model downloads."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workflow.evaluation import (EvaluationImage, EvaluationSet, build_results,
                                 coordinate_positives, exact_retrieval,
                                 extract_descriptors, load_evaluation_set, load_manifest)
from evaluate_salad import validate_image_size


class EvaluationMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def touch(self, name):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def manifest(self, data):
        path = self.root / "eval.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_manifest_uses_explicit_ids_and_preserves_source(self):
        reference = self.touch("references/db.jpg")
        query = self.touch("generated/query.jpg")
        manifest = self.manifest({
            "dataset": "GSV-derived",
            "references": [{"id": "r", "path": "references/db.jpg"}],
            "queries": [{"id": "generated-q", "path": "generated/query.jpg",
                         "positives": ["r", "r"], "source_id": "Bangkok/source.jpg"}],
        })
        dataset = load_manifest(manifest, self.root)
        dataset.validate()
        self.assertEqual(dataset.name, "GSV-derived")
        self.assertEqual(dataset.references[0].path, reference)
        self.assertEqual(dataset.queries[0].path, query)
        self.assertEqual(dataset.queries[0].source_id, "Bangkok/source.jpg")
        self.assertEqual(dataset.positives, [[0]])

    def test_manifest_rejects_unknown_positive_and_duplicate_ids(self):
        data = {"references": [{"id": "r", "path": "db.jpg"}],
                "queries": [{"id": "q", "path": "q.jpg", "positives": ["missing"]}]}
        with self.assertRaisesRegex(ValueError, "unknown positives"):
            load_manifest(self.manifest(data), self.root)
        data["queries"][0]["positives"] = ["r"]
        data["references"].append({"id": "r", "path": "other.jpg"})
        with self.assertRaisesRegex(ValueError, "Duplicate reference"):
            load_manifest(self.manifest(data), self.root)

    def test_manifest_rejects_source_outside_gsv_images(self):
        data = {"references": [{"id": "r", "path": "db.jpg"}],
                "queries": [{"id": "q", "path": "q.jpg", "positives": ["r"],
                             "source_id": "../other.jpg"}]}
        with self.assertRaisesRegex(ValueError, "relative path within GSV"):
            load_manifest(self.manifest(data), self.root)

    def test_external_query_id_never_infers_source(self):
        path = self.manifest({
            "references": [{"id": "r", "path": "db.jpg"}],
            "queries": [{"id": "night/rear/q.jpg", "path": "q.jpg", "positives": ["r"]}],
        })
        dataset = load_manifest(path, self.root, "RobotCar")
        self.assertIsNone(dataset.queries[0].source_id)

    def test_svox_radius_includes_boundary_and_skips_unscorable(self):
        self.touch("images/test/gallery/@0@0@reference@.jpg")
        self.touch("images/test/gallery/@3@4@reference@.jpg")
        self.touch("images/test/gallery/@50@50@reference@.jpg")
        self.touch("images/test/queries_night/@0@0@query@.jpg")
        self.touch("images/test/queries_night/@100@100@query@.jpg")
        dataset = load_evaluation_set("SVOX", self.root, query_subdirs=["queries_night"],
                                      positive_radius=5)
        self.assertEqual(dataset.positives, [[0, 1], []])
        self.assertEqual(dataset.summary()["num_evaluated_queries"], 1)
        self.assertEqual(dataset.summary()["num_queries_without_positives"], 1)
        self.assertTrue(all(query.source_id is None for query in dataset.queries))

    def test_coordinate_grid_matches_brute_force_negative_coordinates(self):
        def image(index, x, y):
            return EvaluationImage(str(index), self.root / f"@{x}@{y}@{index}@.jpg")
        coordinates = [(-8, -4), (-3, -4), (0, 0), (3, 4), (5, 0), (5.1, 0), (20, 20)]
        references = [image(i, x, y) for i, (x, y) in enumerate(coordinates)]
        queries = [image("q1", 0, 0), image("q2", -5, -4)]
        expected = [[i for i, (x, y) in enumerate(coordinates)
                     if (x - qx) ** 2 + (y - qy) ** 2 <= 25] for qx, qy in [(0, 0), (-5, -4)]]
        self.assertEqual(coordinate_positives(references, queries, 5), expected)

    def test_nordland_uses_frame_ids_instead_of_lexical_sort_positions(self):
        for frame in (0, 1, 10, 11, 20):
            self.touch(f"images/test/database/@0@{frame * 2.3:05.1f}@@@@@{frame}@@@@@@@@.jpg")
        self.touch("images/test/queries/@0@00000.0@@@@@0@@@@@@@@.jpg")
        dataset = load_evaluation_set("Nordland", self.root, frame_window=10)
        positive_names = [dataset.references[index].path.name for index in dataset.positives[0]]
        self.assertEqual({int(name.split("@")[7]) for name in positive_names}, {0, 1, 10})
        self.assertEqual(dataset.protocol["ground_truth"], "frame_window")

    def test_robotcar_requires_ground_truth_manifest(self):
        with self.assertRaisesRegex(ValueError, "requires --eval-manifest"):
            load_evaluation_set("RobotCar", self.root)

    def test_dataset_checks_missing_images(self):
        self.touch("db.jpg")
        path = self.manifest({
            "references": [{"id": "r", "path": "db.jpg"}],
            "queries": [{"id": "q", "path": "missing.jpg", "positives": ["r"]}],
        })
        with self.assertRaisesRegex(FileNotFoundError, "Missing 1 evaluation images"):
            load_evaluation_set("manifest", self.root, manifest=path)

    def test_dataset_rejects_overlapping_reference_and_query_paths(self):
        self.touch("same.jpg")
        path = self.manifest({
            "references": [{"id": "r", "path": "same.jpg"}],
            "queries": [{"id": "q", "path": "same.jpg", "positives": ["r"]}],
        })
        with self.assertRaisesRegex(ValueError, "self-matches"):
            load_evaluation_set("manifest", self.root, manifest=path)

    def test_image_size_requires_valid_patches_and_enough_clusters(self):
        config = {"agg_config": {"num_clusters": 64}}
        validate_image_size((322, 322), config)
        with self.assertRaisesRegex(ValueError, "multiples"):
            validate_image_size((320, 322), config)
        with self.assertRaisesRegex(ValueError, "more than 64"):
            validate_image_size((112, 112), config)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class ExactRetrievalTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("PIL") and importlib.util.find_spec("numpy"),
                         "Pillow/NumPy are not installed")
    def test_descriptor_extraction_normalizes_rgb_and_preserves_order(self):
        import torch
        from PIL import Image

        class MeanModel(torch.nn.Module):
            def forward(self, batch):
                return batch.mean(dim=(2, 3))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (9, 7), (255, 0, 0)).save(root / "red.png")
            Image.new("L", (3, 5), 255).save(root / "white.png")
            images = [EvaluationImage("red", root / "red.png"),
                      EvaluationImage("white", root / "white.png")]
            descriptors = extract_descriptors(MeanModel(), images, (14, 28), "cpu",
                                              batch_size=1, num_workers=0)
            expected = (torch.tensor([[1., 0., 0.], [1., 1., 1.]])
                        - torch.tensor([.485, .456, .406])) / torch.tensor([.229, .224, .225])
            torch.testing.assert_close(descriptors, expected)

    def test_exact_rank_beyond_top10_and_multiple_positives(self):
        import torch

        references = torch.arange(20, dtype=torch.float32)[:, None]
        queries = torch.tensor([[0.0], [19.0], [4.0]])
        result = exact_retrieval(references, queries, [[18, 15], [19], []],
                                 query_chunk_size=2, reference_chunk_size=3)
        self.assertEqual(result[0]["rank"], 16)
        self.assertEqual(result[0]["ground_truth_index"], 15)
        self.assertEqual(result[0]["predicted_index"], 0)
        self.assertEqual(result[0]["distance_gt"], 15)
        self.assertEqual(result[0]["distance_pred"], 0)
        self.assertEqual(result[1]["rank"], 1)
        self.assertIsNone(result[2])

    def test_equal_distance_ties_use_reference_order(self):
        result = exact_retrieval([[-1.0], [1.0], [1.0]], [[0.0]], [[2]],
                                 reference_chunk_size=1)
        self.assertEqual(result[0]["rank"], 3)
        self.assertEqual(result[0]["predicted_index"], 0)
        result = exact_retrieval([[-1.0], [1.0], [1.0]], [[0.0]], [[2, 1]],
                                 reference_chunk_size=2)
        self.assertEqual(result[0]["rank"], 2)
        self.assertEqual(result[0]["ground_truth_index"], 1)

    def test_chunked_ranks_match_full_sort_oracle(self):
        import torch

        generator = torch.Generator().manual_seed(123)
        references = torch.randn(37, 9, generator=generator)
        queries = torch.randn(11, 9, generator=generator)
        positives = [[index, (index + 13) % 37] for index in range(11)]
        distances = (queries.square().sum(1, keepdim=True)
                     + references.square().sum(1)[None, :] - 2 * queries @ references.T).clamp_min(0)
        order = distances.argsort(dim=1, stable=True)
        expected = []
        for row, indices in enumerate(positives):
            ranking = order[row].tolist()
            best = min(indices, key=ranking.index)
            expected.append((ranking.index(best) + 1, best, ranking[0]))
        for query_chunk, reference_chunk in ((1, 1), (3, 7), (30, 100)):
            result = exact_retrieval(references, queries, positives,
                                     query_chunk_size=query_chunk, reference_chunk_size=reference_chunk)
            self.assertEqual([(row["rank"], row["ground_truth_index"], row["predicted_index"])
                              for row in result], expected)

    def test_results_recall_denominator_and_hard_case_provenance(self):
        dataset = EvaluationSet("derived", [EvaluationImage(str(i), Path(f"db{i}.jpg")) for i in range(3)],
                                [EvaluationImage("aug1", Path("q1.jpg"), "Bangkok/source.jpg"),
                                 EvaluationImage("q2", Path("q2.jpg")),
                                 EvaluationImage("no_gt", Path("q3.jpg"))], [[2], [0], []])
        retrieval = exact_retrieval([[0.0], [1.0], [2.0]], [[0.0], [0.0], [0.0]], dataset.positives)
        result = build_results(dataset, retrieval, save_hard_cases=True)
        self.assertEqual(result["recall"], {"R@1": 0.5, "R@5": 1.0, "R@10": 1.0})
        self.assertEqual(result["num_evaluated_queries"], 2)
        self.assertEqual(len(result["error_queries"]), 1)
        case = result["error_queries"][0]
        self.assertEqual(case["source_id"], "Bangkok/source.jpg")
        self.assertEqual(case["ground_truth"], "2")
        self.assertEqual(case["predicted"], "0")
        self.assertEqual(case["rank"], 3)


if __name__ == "__main__":
    unittest.main()
