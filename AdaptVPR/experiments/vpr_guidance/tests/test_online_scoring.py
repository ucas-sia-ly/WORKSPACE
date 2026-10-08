"""CPU coverage for persistent online scoring and standalone score parity."""

import csv
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from PIL import Image

GUIDANCE_ROOT = Path(__file__).resolve().parents[1]
if str(GUIDANCE_ROOT) not in sys.path:
    sys.path.insert(0, str(GUIDANCE_ROOT))

import online_scoring  # noqa: E402
import score_candidates as standalone  # noqa: E402
from common import read_jsonl, write_jsonl  # noqa: E402
from feedback import score_candidates  # noqa: E402
from online_scoring import OnlineScorer  # noqa: E402
from workflow.evaluation import extract_descriptors  # noqa: E402
from workflow.training_data import MixedGSVCitiesDataset  # noqa: E402


class TinyStudent(torch.nn.Module):
    image_size = (14, 28)

    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, images):
        self.inputs.append(images.detach().clone())
        means = images.mean(dim=(-1, -2))
        return torch.cat((means, torch.ones_like(means)), dim=1)


class OnlineScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.real_data = self.root / "gsv"
        dataframes = self.real_data / "Dataframes"
        images = self.real_data / "Images" / "City"
        dataframes.mkdir(parents=True)
        images.mkdir(parents=True)
        self.real_paths = []
        # A heterogeneous view count exposes positive RNG dependence on batch
        # padding. The final place is excluded from eligible training metadata.
        counts = (4, 7, 5, 6, 1)
        with (dataframes / "City.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["place_id", "year", "month", "northdeg", "city_id", "lat", "lon", "panoid"])
            for place, count in enumerate(counts):
                views = []
                for view in range(count):
                    writer.writerow([place, 2020, 1, view, "City", "1.0", "2.0", f"p{place}v{view}"])
                    path = images / f"City_{place:07d}_2020_01_{view:03d}_1.0_2.0_p{place}v{view}.jpg"
                    Image.new("RGB", (31, 19), (30 + place * 35, 15 + view * 30, 210 - place * 20)).save(path)
                    views.append(path)
                self.real_paths.append(views)
        self.checkpoint = self.root / "student.ckpt"
        self.checkpoint.write_bytes(b"dummy checkpoint for injected CPU student")
        self.generated = self.root / "generated"
        self.generated.mkdir()
        self.rows = []
        for index, place in enumerate((0, 1, 2, 0)):
            path = self.generated / f"candidate{index}.jpg"
            Image.new("RGB", (25, 37), (220 - index * 40, 100, 30 + index * 50)).save(path)
            self.rows.append({
                "sample_id": f"sample{index}", "candidate_index": 0, "passed": True,
                "prompt": "frozen prompt", "condition": "rain", "s_geo": 0.9,
                "source_path": str(self.real_paths[place][0]), "output_path": str(path),
            })
        self.manifest = self.root / "candidates.jsonl"
        write_jsonl(self.manifest, self.rows)
        self.arguments = [
            "--candidates", str(self.manifest), "--checkpoint", str(self.checkpoint),
            "--real-data", str(self.real_data), "--selection", "hardness",
            "--output-dir", str(self.root / "scored"), "--train-batch-size", "3",
            "--images-per-place", "4", "--min-images-per-place", "4",
            "--negative-pool-size", "4", "--negative-draws", "17",
            "--calibration-places", "4", "--plausibility-quantile", "0.25",
            "--device", "cpu", "--num-workers", "0", "--seed", "13",
        ]
        self.args = standalone.parse_args(self.arguments)
        generator = torch.Generator().manual_seed(88)
        paths = [path for views in self.real_paths for path in views]
        paths.extend(Path(row["output_path"]) for row in self.rows)
        self.descriptors = {path: torch.randn(9, generator=generator) for path in paths}
        self.extractions = []
        self.model = TinyStudent()
        self.load_model = Mock(return_value=self.model)

    def extract(self, model, images, image_size, device, batch_size, num_workers):
        paths = [image.path for image in images]
        self.extractions.append(paths)
        self.assertIs(model, self.model)
        self.assertEqual(image_size, self.model.image_size)
        self.assertEqual(device, "cpu")
        self.assertEqual(batch_size, self.args.batch_size)
        self.assertEqual(num_workers, 0)
        return torch.stack([self.descriptors[path] for path in paths])

    def create_scorer(self, sources=None, *, extractor=None):
        return OnlineScorer(
            self.args, sources or [row["source_path"] for row in self.rows],
            model_factory=self.load_model, descriptor_extractor=extractor or self.extract,
        )

    def assert_row_equal(self, first, second):
        self.assertEqual(set(first), set(second))
        for field in first:
            if isinstance(first[field], float):
                self.assertAlmostEqual(first[field], second[field], places=6, msg=field)
            else:
                self.assertEqual(first[field], second[field], field)

    def test_loads_student_once_and_extracts_real_context_once(self):
        scorer = self.create_scorer()
        self.load_model.assert_called_once_with(self.checkpoint, "cpu", backbone_repo=None)
        self.assertEqual(len(self.extractions), 1)
        self.assertEqual(len(self.extractions[0]), len(set(self.extractions[0])))
        self.assertEqual(set(self.extractions[0]), {p for views in self.real_paths[:4] for p in views})
        self.assertEqual(scorer.calibration["real_anchors"], 22)
        self.assertEqual(scorer.score_settings["positive_context_width"], 7)
        calibration = scorer.metadata()["plausibility_calibration"]
        for row in reversed(self.rows):
            scorer.score(row)
        self.assertEqual(len(self.extractions), 5)
        self.assertTrue(all(len(paths) == 1 for paths in self.extractions[1:]))
        self.assertEqual(scorer.metadata()["plausibility_calibration"], calibration)
        self.load_model.assert_called_once()
        metadata = scorer.metadata()
        json.dumps(metadata, allow_nan=False)
        metadata["score_settings"]["seed"] = -1
        self.assertEqual(scorer.score_settings["seed"], 13)

    def test_matches_standalone_scoring_and_calibration_with_heterogeneous_views(self):
        scorer = self.create_scorer()
        online = {row["sample_id"]: scorer.score(row) for row in reversed(self.rows)}
        with patch("workflow.model.load_checkpoint_model", side_effect=self.load_model):
            with patch("workflow.evaluation.extract_descriptors", side_effect=self.extract):
                standalone.main(self.arguments)
        offline = read_jsonl(self.args.output_dir / "scored.jsonl")
        for row in offline:
            self.assert_row_equal(online[row["sample_id"]], row)
        summary = json.loads((self.args.output_dir / "summary.json").read_text())
        self.assertEqual(scorer.score_settings, summary["score_settings"])
        self.assertEqual(scorer.calibration, summary["plausibility_calibration"])

    def test_scores_ignore_batch_membership_and_completion_order(self):
        scorer = self.create_scorer()
        expected = {row["sample_id"]: scorer.score(row) for row in self.rows}
        pool = torch.stack([torch.stack([self.descriptors[path] for path in paths]) for _, paths in scorer.pool])
        for order in ([self.rows[0]], list(reversed(self.rows)), [self.rows[2], self.rows[0]]):
            places = [scorer.place_of[Path(row["source_path"])] for row in order]
            anchors = torch.stack([self.descriptors[Path(row["output_path"])] for row in order])
            positives = [torch.stack([self.descriptors[p] for p in scorer.real_views[key]]) for key in places]
            same = torch.tensor([[place == key for key in scorer.pool_keys] for place in places])
            scores = score_candidates(
                anchors, positives, pool, same, scorer.negatives_per_batch,
                draws=self.args.negative_draws, seed=self.args.seed,
                positives_per_batch=3, positive_context_width=7,
            )
            for row, score in zip(order, scores):
                for field, value in score.items():
                    self.assertAlmostEqual(value, expected[row["sample_id"]][field], places=6, msg=field)
                self.assert_row_equal(scorer.score(row), expected[row["sample_id"]])

    def test_calibration_is_leave_one_out_and_excludes_own_place_from_negatives(self):
        with patch.object(online_scoring, "score_candidates", wraps=score_candidates) as score:
            scorer = self.create_scorer()
        score.assert_called_once()
        anchors, positives, pool, same = score.call_args.args[:4]
        self.assertEqual(len(anchors), 22)
        for anchor, positive_set in zip(anchors, positives):
            self.assertFalse(bool((positive_set == anchor).all(dim=1).any()))
        places = [key for key in scorer.calibration_keys for _ in scorer.real_views[key]]
        for i, place in enumerate(places):
            self.assertEqual(len(positives[i]), len(scorer.real_views[place]) - 1)
            self.assertEqual(same[i].tolist(), [place == key for key in scorer.pool_keys])
        self.assertEqual(pool.shape, (4, 4, 9))

    def test_missing_or_ineligible_sources_fail_before_model_load(self):
        for source in (self.root / "unknown.jpg", self.real_paths[4][0]):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "eligible training metadata"):
                self.create_scorer([source])
        self.load_model.assert_not_called()
        self.assertEqual(self.extractions, [])

    def test_negative_pool_validation_precedes_model_loading(self):
        self.args.negative_pool_size = 1
        with self.assertRaisesRegex(ValueError, "Negative pool"):
            self.create_scorer()
        self.load_model.assert_not_called()

    def test_unverified_and_real_image_outputs_are_rejected_without_extraction(self):
        scorer = self.create_scorer([self.rows[0]["source_path"]])
        bad_rows = [
            {**self.rows[0], "passed": False},
            {**self.rows[0], "output_path": str(self.real_paths[1][0])},
            {**self.rows[0], "output_path": str(self.real_paths[4][0])},
            self.rows[1], {**self.rows[0], "candidate_index": True},
        ]
        for row in bad_rows:
            with self.subTest(row=row), self.assertRaises(ValueError):
                scorer.score(row)
        self.assertEqual(len(self.extractions), 1)

    def test_disabled_gate_does_not_calibrate_or_extract_unneeded_real_views(self):
        self.args.plausibility_quantile = 0
        self.args.negative_pool_size = 3
        scorer = self.create_scorer([self.rows[0]["source_path"]])
        required = set(self.real_paths[0])
        required.update(path for _, paths in scorer.pool for path in paths)
        self.assertEqual(set(self.extractions[0]), required)
        self.assertIsNone(scorer.calibration)
        self.assertIsNone(scorer.floor)
        self.assertTrue(scorer.score(self.rows[0])["plausible"])

    def test_final_jpeg_matches_training_preprocessing_and_is_not_cached(self):
        scorer = self.create_scorer(extractor=extract_descriptors)
        result = scorer.score(self.rows[0])
        candidate_tensor = self.model.inputs[-1][0]
        training = MixedGSVCitiesDataset(
            self.real_data, images_per_place=4, min_images_per_place=4,
            image_size=self.model.image_size, synthetic_fraction=0.25, augment=False,
        )
        training.places[0].synthetic_paths.append(Path(self.rows[0]["output_path"]))
        random.seed(37)
        tensors, _, synthetic = training[0]
        self.assertTrue(torch.equal(tensors[synthetic][0], candidate_tensor))
        self.assertEqual(tuple(candidate_tensor.shape), (3, 14, 28))
        Image.new("RGB", (25, 37), (5, 230, 140)).save(self.rows[0]["output_path"])
        replaced = scorer.score(self.rows[0])
        self.assertFalse(torch.equal(candidate_tensor, self.model.inputs[-1][0]))
        self.assertNotEqual(result["mean_positive_similarity"], replaced["mean_positive_similarity"])
        self.assert_row_equal(scorer.score(self.rows[0]), replaced)


if __name__ == "__main__":
    unittest.main()
