"""CPU regressions for batch expectations and the scorer/trainer manifest contract."""

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from PIL import Image

GUIDANCE_ROOT = Path(__file__).resolve().parents[1]
if str(GUIDANCE_ROOT) not in sys.path:
    sys.path.insert(0, str(GUIDANCE_ROOT))

import score_candidates as scoring  # noqa: E402
from feedback import expected_hardest_negative, ms_positive_utility, score_candidates, select_per_group  # noqa: E402
from workflow.training_data import MixedGSVCitiesDataset  # noqa: E402


def unit(similarity):
    return torch.tensor([similarity, math.sqrt(1 - similarity ** 2)], dtype=torch.float32)


class BatchExpectationTests(unittest.TestCase):
    def test_expected_utility_mines_rare_hard_batches(self):
        candidate = unit(1)[None]
        positives = [unit(0.4)[None]]
        pool = torch.stack([unit(0.8), unit(-0.8)])
        result = score_candidates(candidate, positives, pool, torch.zeros(1, 2, dtype=torch.bool),
                                  negatives_per_batch=1, draws=512, seed=3)[0]
        probability = result["mining_probability"]
        self.assertAlmostEqual(probability, 0.5, delta=0.08)
        self.assertAlmostEqual(result["utility"], probability * math.log1p(math.exp(-0.4)), places=6)
        self.assertAlmostEqual(result["mined_pairs"], probability, places=6)
        old_utility, _ = ms_positive_utility(torch.tensor([[0.4]]),
                                           torch.tensor([result["expected_hardest_negative"]]))
        self.assertEqual(float(old_utility), 0.0)
        self.assertGreater(result["utility"], 0)

    def test_grouped_pool_samples_places_then_uses_all_k_views(self):
        candidate = unit(1)[None]
        pool = torch.stack([torch.stack([unit(1), unit(0)]), torch.stack([unit(0), unit(0)])])
        hardest = expected_hardest_negative(candidate, pool, torch.zeros(1, 2, dtype=torch.bool),
                                            negatives_per_batch=2, draws=512,
                                            generator=torch.Generator().manual_seed(7))
        self.assertAlmostEqual(float(hardest), 0.5, delta=0.08)
        own = torch.tensor([[False, True]])
        self.assertEqual(float(expected_hardest_negative(candidate, pool, own, 2)), 1.0)
        with self.assertRaisesRegex(ValueError, "other places"):
            expected_hardest_negative(candidate, pool, own, 4)
        with self.assertRaisesRegex(ValueError, "divisible"):
            expected_hardest_negative(candidate, pool, own, 3)

    def test_shared_draws_do_not_create_fake_candidate_differences(self):
        candidate = unit(1).repeat(2, 1)
        positives = [torch.stack([unit(0.3), unit(0.7)])] * 2
        pool = torch.stack([unit(0.8), unit(-0.8), unit(0)])
        scores = score_candidates(candidate, positives, pool, torch.zeros(2, 3, dtype=torch.bool),
                                  1, draws=32, seed=19, positives_per_batch=1)
        self.assertEqual(scores[0], scores[1])
        self.assertEqual(scores, score_candidates(candidate, positives, pool,
                                                 torch.zeros(2, 3, dtype=torch.bool),
                                                 1, draws=32, seed=19, positives_per_batch=1))

    def test_positive_draws_use_training_batch_count(self):
        candidate = unit(1)[None]
        positives = [torch.stack([unit(0.2), unit(0.4), unit(0.6)])]
        scores = score_candidates(candidate, positives, unit(1)[None], torch.zeros(1, 1, dtype=torch.bool),
                                  1, draws=512, positives_per_batch=1)[0]
        expected = sum(math.log1p(math.exp(-s)) for s in (0.2, 0.4, 0.6)) / 3
        self.assertAlmostEqual(scores["utility"], expected, delta=0.02)
        self.assertEqual(scores["positive_pairs"], 1)
        self.assertEqual(scores["available_real_views"], 3)
        self.assertEqual(scores["mined_pairs"], 1)
        self.assertEqual(scores["mining_probability"], 1)

    def test_miner_uses_official_strict_boundary(self):
        utility, mined = ms_positive_utility(torch.tensor([[0.5]]), torch.tensor([0.375]), epsilon=0.125)
        self.assertEqual(float(utility), 0.0)
        self.assertEqual(int(mined), 0)

    def test_invalid_tensor_contracts_fail_before_computation(self):
        candidate, pool, mask = unit(1)[None], unit(0)[None], torch.zeros(1, 1, dtype=torch.bool)
        for bad in (torch.zeros(1, 2), torch.tensor([[float("nan"), 1]]), torch.ones(1, 3)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                score_candidates(candidate, [candidate], bad, mask, 1)
        for bad_mask in (mask.float(), torch.zeros(2, 1, dtype=torch.bool)):
            with self.subTest(mask=bad_mask), self.assertRaises(ValueError):
                score_candidates(candidate, [candidate], pool, bad_mask, 1)
        with self.assertRaises(ValueError):
            score_candidates(candidate, [candidate[:0]], pool, mask, 1)
        with self.assertRaises(ValueError):
            score_candidates(candidate, [candidate], pool, mask, 1, positives_per_batch=2)
        for arguments in ({"alpha": 0}, {"epsilon": -1}, {"base": float("inf")},
                          {"positive_mask": mask.float()}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                ms_positive_utility(torch.ones(1, 1), torch.ones(1), **arguments)
        with self.assertRaises(ValueError):
            expected_hardest_negative(candidate, pool, mask, negatives_per_batch=True)
        self.assertEqual(score_candidates(candidate[:0], [], pool, mask[:0], 1), [])

    def test_invalid_selection_even_for_empty_input(self):
        with self.assertRaises(ValueError):
            select_per_group([], "unknown")
        with self.assertRaises(ValueError):
            select_per_group([{"passed": True, "sample_id": "x", "utility": float("nan"),
                               "mean_positive_similarity": 0.2}], "hardness")
        self.assertEqual(select_per_group([{"passed": 1}], "random"), [])


class ManifestScoringTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.real_data = self.root / "gsv"
        dataframe_dir = self.real_data / "Dataframes"
        image_dir = self.real_data / "Images" / "City"
        dataframe_dir.mkdir(parents=True)
        image_dir.mkdir(parents=True)
        fields = ["place_id", "year", "month", "northdeg", "city_id", "lat", "lon", "panoid"]
        self.real_paths = {}
        with (dataframe_dir / "City.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for place in range(3):
                self.real_paths[place] = []
                for view in range(2):
                    row = {"place_id": str(place), "year": "2020", "month": "1", "northdeg": str(view),
                           "city_id": "City", "lat": "1.0", "lon": "2.0", "panoid": f"p{place}v{view}"}
                    writer.writerow(row)
                    filename = f"City_{place:07d}_2020_01_{view:03d}_1.0_2.0_p{place}v{view}.jpg"
                    path = image_dir / filename
                    Image.new("RGB", (2, 2)).save(path)
                    self.real_paths[place].append(path.resolve())
        self.generated = self.root / "generated"
        self.generated.mkdir()
        self.manifest = self.generated / "candidates.jsonl"
        self.rows = []
        for index in range(2):
            output = self.generated / f"candidate{index}.jpg"
            Image.new("RGB", (2, 2)).save(output)
            self.rows.append({"sample_id": "source0-rain", "candidate_index": index,
                              "source_path": str(self.real_paths[0][0]), "output_path": str(output),
                              "prompt": "rainy street", "condition": "rain", "passed": True})
        self.checkpoint = self.root / "student.pt"
        self.checkpoint.write_bytes(b"mock student")
        self.write_rows(self.rows)
        self.dataset = MixedGSVCitiesDataset(self.real_data, images_per_place=2, min_images_per_place=2)
        self.place_of = {path: (p.city, p.place_id) for p in self.dataset.places for path in p.real_paths}

    def tearDown(self):
        self.temporary.cleanup()

    def write_rows(self, rows):
        self.manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    def arguments(self):
        return ["--candidates", str(self.manifest), "--checkpoint", str(self.checkpoint),
                "--real-data", str(self.real_data), "--output-dir", str(self.root / "scored"),
                "--selection", "hardness", "--images-per-place", "2", "--min-images-per-place", "2",
                "--train-batch-size", "3", "--negative-pool-size", "3", "--negative-draws", "8",
                "--device", "cpu", "--num-workers", "0"]

    def test_resolver_and_dedup_match_training_paths(self):
        row = {**self.rows[0], "source_path": f"City/{self.real_paths[0][0].name}",
               "output_path": "candidate0.jpg"}
        self.write_rows([row, row, {"passed": False}])
        verified, counts, total = scoring.load_verified_candidates([self.manifest], self.real_data, self.place_of)
        self.assertEqual(total, 3)
        self.assertEqual(len(verified), 1)
        self.assertEqual(Path(verified[0]["source_path"]), self.real_paths[0][0])
        self.assertEqual(Path(verified[0]["output_path"]), self.generated / "candidate0.jpg")
        self.assertEqual(counts, {"duplicate_candidates": 1, "rejected_by_verifier": 1})

    def test_conflicting_duplicates_and_group_identity_are_rejected(self):
        alternatives = [
            {**self.rows[1], "output_path": self.rows[0]["output_path"]},
            {**self.rows[1], "candidate_index": 0},
            {**self.rows[1], "source_path": str(self.real_paths[1][0])},
            {**self.rows[1], "condition": "snow"},
        ]
        for other in alternatives:
            with self.subTest(other=other), self.assertRaises(ValueError):
                self.write_rows([self.rows[0], other])
                scoring.load_verified_candidates([self.manifest], self.real_data, self.place_of)

    def test_real_output_and_invalid_verified_rows_are_rejected(self):
        for bad in ({**self.rows[0], "output_path": str(self.real_paths[1][0])},
                    {**self.rows[0], "candidate_index": True},
                    {**self.rows[0], "prompt": ""}, {**self.rows[0], "source_path": "missing.jpg"}):
            with self.subTest(bad=bad), self.assertRaises((ValueError, FileNotFoundError)):
                self.write_rows([bad])
                scoring.load_verified_candidates([self.manifest], self.real_data, self.place_of)

    def test_pool_has_k_distinct_views_per_place(self):
        real_views = {(p.city, p.place_id): p.real_paths for p in self.dataset.places}
        pool = scoring.sample_negative_pool(real_views, 3, 2, 9)
        self.assertEqual(len(pool), 3)
        self.assertEqual(len({key for key, _ in pool}), 3)
        self.assertTrue(all(len(set(views)) == 2 for _, views in pool))
        self.assertEqual(pool, scoring.sample_negative_pool(real_views, 3, 2, 9))

    def test_end_to_end_cpu_scoring_manifest_loads_in_training(self):
        def extract(model, images, *args):
            return torch.stack([unit(0) if image.path.parent == self.generated else
                                unit(1) if image.path in self.real_paths[0] else unit(0)
                                for image in images])
        with patch("workflow.model.load_checkpoint_model", return_value=SimpleNamespace(image_size=(2, 2))) as load:
            with patch("workflow.evaluation.extract_descriptors", side_effect=extract):
                scoring.main(self.arguments())
        load.assert_called_once()
        output = self.root / "scored"
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["groups_selected"], 1)
        self.assertEqual(summary["negative_places_per_batch"], 2)
        self.assertEqual(summary["negative_pool_images"], 6)
        self.assertEqual(summary["mined_rate_all_verified"], 1)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["score_settings"]["positive_views_per_draw"], 1)
        self.assertEqual(len(summary["candidate_manifests"][0]["sha256"]), 64)
        mixed = MixedGSVCitiesDataset(self.real_data, output / "selected.jsonl",
                                      images_per_place=2, min_images_per_place=2)
        self.assertEqual(mixed.summary["num_synthetic_images"], 1)

    def test_no_verified_candidates_writes_empty_result_without_model(self):
        self.write_rows([{**row, "passed": False} for row in self.rows])
        with patch("workflow.model.load_checkpoint_model") as load:
            scoring.main(self.arguments())
        load.assert_not_called()
        output = self.root / "scored"
        self.assertEqual((output / "selected.jsonl").read_text(), "")
        self.assertEqual((output / "scored.jsonl").read_text(), "")
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["status"], "no_verified_training_candidates")
        self.assertEqual(summary["mined_rate_selected"], 0)

    def test_effective_batch_size_matches_training_small_dataset(self):
        arguments = self.arguments()
        arguments[arguments.index("--train-batch-size") + 1] = "32"
        def extract(model, images, *args):
            return unit(0).repeat(len(images), 1)
        with patch("workflow.model.load_checkpoint_model", return_value=SimpleNamespace(image_size=(2, 2))):
            with patch("workflow.evaluation.extract_descriptors", side_effect=extract):
                scoring.main(arguments)
        summary = json.loads((self.root / "scored" / "summary.json").read_text())
        self.assertEqual(summary["score_settings"]["train_batch_size"], 32)
        self.assertEqual(summary["score_settings"]["effective_train_batch_size"], 3)
        self.assertEqual(summary["negative_places_per_batch"], 2)

    def test_real_output_from_an_excluded_place_is_rejected(self):
        dataframe = self.real_data / "Dataframes" / "City.csv"
        with dataframe.open("a", newline="") as handle:
            csv.writer(handle).writerow([3, 2020, 1, 0, "City", "1.0", "2.0", "excluded"])
        excluded = self.real_data / "Images" / "City" / "City_0000003_2020_01_000_1.0_2.0_excluded.jpg"
        Image.new("RGB", (2, 2)).save(excluded)
        self.write_rows([{**self.rows[0], "output_path": str(excluded)}])
        with patch("workflow.model.load_checkpoint_model") as load:
            with self.assertRaisesRegex(ValueError, "generated image"):
                scoring.main(self.arguments())
        load.assert_not_called()

    def test_small_negative_pool_fails_before_model_load(self):
        arguments = self.arguments()
        arguments[arguments.index("--negative-pool-size") + 1] = "1"
        with patch("workflow.model.load_checkpoint_model") as load:
            with self.assertRaisesRegex(ValueError, "Negative pool"):
                scoring.main(arguments)
        load.assert_not_called()

    def test_cli_rejects_invalid_training_batch_config(self):
        for flag, value in (("--train-batch-size", "1"), ("--images-per-place", "3"),
                            ("--negative-draws", "0"), ("--alpha", "0"), ("--miner-margin", "nan")):
            arguments = self.arguments()
            if flag in arguments:
                arguments[arguments.index(flag) + 1] = value
            else:
                arguments.extend([flag, value])
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                with patch("sys.stderr"):
                    scoring.parse_args(arguments)


if __name__ == "__main__":
    unittest.main()
