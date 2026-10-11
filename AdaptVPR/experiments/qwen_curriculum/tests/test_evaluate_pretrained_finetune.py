"""CPU contract tests for six-domain evaluation, integrity and safe resume."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.qwen_curriculum import common, evaluate_pretrained_finetune as target

common.use_salad()
from workflow import evaluation


class TinyOriginalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.aggregator = SimpleNamespace(reliability_ot=False)


class SixDomainEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkpoint = self.root / "original.ckpt"
        self.checkpoint.write_bytes(b"original pretrained weights")
        self.dataset_root = self.root / "svox"
        self.dataset_root.mkdir()
        self.output = self.root / "evaluation"
        self.repo = self.root / "dinov2"
        self.repo.mkdir()
        (self.repo / "hubconf.py").write_text("# local backbone factory\n")
        self.references = []
        for index in range(2):
            path = self.dataset_root / f"reference_{index}.png"
            path.write_bytes(bytes([index + 1]))
            self.references.append(evaluation.EvaluationImage(f"reference_{index}", path))
        self.datasets = {}
        for domain, folder in target.DOMAINS.items():
            path = self.dataset_root / f"query_{domain}.png"
            path.write_bytes(b"query")
            query = evaluation.EvaluationImage(domain, path)
            self.datasets[domain] = evaluation.EvaluationSet("SVOX", list(self.references), [query], [[0]], {
                "ground_truth": "utm_radius", "positive_radius_meters": 25.0,
                "split": "test", "query_subdirs": [folder],
            })
        self.config = {"agg_config": {"num_clusters": 1, "cluster_dim": 1, "token_dim": 1}}
        self.model_backend = SimpleNamespace(
            read_checkpoint=Mock(return_value={}),
            checkpoint_state_and_config=Mock(side_effect=lambda unused: ({}, self.config)),
            load_checkpoint_model=Mock(side_effect=lambda *args, **kwargs: TinyOriginalModel()),
        )
        self.extractions = []

        def extract(model, images, image_size, device, batch_size, num_workers):
            self.extractions.append([image.id for image in images])
            if len(images) == 2:
                return torch.tensor([[1., 0.], [0., 1.]])
            return torch.tensor([[1., 0.]])

        self.evaluation_backend = SimpleNamespace(
            load_evaluation_set=Mock(side_effect=lambda *args, **kwargs: self.datasets[
                next(domain for domain, folder in target.DOMAINS.items()
                     if kwargs["query_subdirs"] == [folder])]),
            extract_descriptors=Mock(side_effect=extract),
            exact_retrieval=evaluation.exact_retrieval, build_results=evaluation.build_results,
        )
        counts = {"gallery": 2, **{folder: 1 for folder in target.DOMAINS.values()}}
        for mocker in (patch.object(target, "_backend", return_value=(self.evaluation_backend, self.model_backend)),
                       patch.object(target, "COUNTS", counts)):
            mocker.start()
            self.addCleanup(mocker.stop)

    def run_evaluation(self, **kwargs):
        return target.evaluate(self.checkpoint, self.dataset_root, self.output, self.repo,
                               device="cpu", **kwargs)

    def test_full_six_domains_extract_gallery_once_and_resume_without_loading_model(self):
        results = self.run_evaluation()
        self.assertEqual(list(results), list(target.DOMAINS))
        self.assertEqual(self.extractions[0], [image.id for image in self.references])
        self.assertEqual(len(self.extractions), 7)
        for domain, row in results.items():
            self.assertEqual(row["image_size"], [322, 322])
            self.assertEqual(row["descriptor_dtype"], "float32")
            self.assertEqual(row["num_queries"], 1)
            self.assertEqual(row["recall"], {"R@1": 1, "R@5": 1, "R@10": 1})
            self.assertEqual(row["checkpoint_sha256"], common.file_sha256(self.checkpoint))
            self.assertEqual(target._read_sealed(self.output / f"SVOX_{domain}.json"), row)
        self.assertEqual(self.run_evaluation(), results)
        self.assertEqual(len(self.extractions), 7)
        self.assertEqual(self.model_backend.load_checkpoint_model.call_count, 1)
        self.assertEqual(target._read_sealed(self.output / "summary.json")["state"], "complete")

    def test_interruption_keeps_completed_domain_and_resume_finishes_remaining_five(self):
        normal = self.evaluation_backend.extract_descriptors.side_effect
        calls = 0

        def interrupted(*args):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("fixture interrupted after day")
            return normal(*args)

        self.evaluation_backend.extract_descriptors.side_effect = interrupted
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.run_evaluation()
        old_day = (self.output / "SVOX_day.json").read_bytes()
        self.assertFalse((self.output / "summary.json").exists())
        self.evaluation_backend.extract_descriptors.side_effect = normal
        self.assertEqual(len(self.run_evaluation()), 6)
        self.assertEqual((self.output / "SVOX_day.json").read_bytes(), old_day)
        self.assertEqual(sum(len(images) == 2 for images in self.extractions), 2)

    def test_changed_checkpoint_refuses_cached_results(self):
        self.run_evaluation()
        self.checkpoint.write_bytes(b"different weights")
        with self.assertRaisesRegex(ValueError, "frozen protocol changed"):
            self.run_evaluation()

    def test_changed_image_inventory_refuses_cached_results(self):
        self.run_evaluation()
        self.references[0].path.write_bytes(b"different image")
        with self.assertRaisesRegex(ValueError, "input inventory"):
            self.run_evaluation()

    def test_changed_resolution_refuses_322_results(self):
        self.run_evaluation()
        with self.assertRaisesRegex(ValueError, "frozen protocol changed"):
            self.run_evaluation(image_size=(224, 224))

    def test_tampered_cached_recall_cannot_resume(self):
        self.run_evaluation()
        path = self.output / "SVOX_night.json"
        row = json.loads(path.read_text())
        row["recall"]["R@1"] = 0
        common.write_json(path, row)
        with self.assertRaisesRegex(ValueError, "checksum differs"):
            self.run_evaluation()

    def test_resealed_wrong_checkpoint_results_cannot_resume(self):
        self.run_evaluation()
        path = self.output / "SVOX_night.json"
        row = json.loads(path.read_text())
        row.pop("record_sha256")
        row["checkpoint_sha256"] = "different"
        common.write_json(path, target._seal(row))
        with self.assertRaisesRegex(ValueError, "different weights"):
            self.run_evaluation()

    def test_gallery_order_mismatch_refused_before_model_loading(self):
        self.datasets["night"].references.reverse()
        with self.assertRaisesRegex(ValueError, "same ordered gallery"):
            self.run_evaluation()
        self.model_backend.load_checkpoint_model.assert_not_called()

    def test_manifest_override_and_partial_counts_refused(self):
        manifest = self.dataset_root / "evaluation_manifest.json"
        manifest.write_text("{}")
        with self.assertRaisesRegex(ValueError, "manifest override"):
            self.run_evaluation()
        manifest.unlink()
        self.datasets["night"].queries.clear()
        self.datasets["night"].positives.clear()
        with self.assertRaisesRegex(ValueError, "counts differ"):
            self.run_evaluation()

    def test_reliability_config_or_head_weights_refused_even_for_cached_runs(self):
        self.config["agg_config"]["reliability_ot"] = True
        with self.assertRaisesRegex(ValueError, "original SALAD"):
            self.run_evaluation()
        del self.config["agg_config"]["reliability_ot"]
        self.model_backend.checkpoint_state_and_config.side_effect = None
        self.model_backend.checkpoint_state_and_config.return_value = (
            {"aggregator.reliability_head.0.weight": torch.zeros(1)}, self.config)
        with self.assertRaisesRegex(ValueError, "head weights"):
            self.run_evaluation()

    def test_bad_descriptor_shape_or_dtype_refused_without_false_completion(self):
        self.evaluation_backend.extract_descriptors.side_effect = lambda *args: torch.zeros(1, 2)
        with self.assertRaisesRegex(ValueError, "Descriptor count"):
            self.run_evaluation()
        self.assertFalse((self.output / "summary.json").exists())
        self.evaluation_backend.extract_descriptors.side_effect = lambda *args: torch.zeros(2, 2, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "FP32 dtype"):
            self.run_evaluation()


if __name__ == "__main__":
    unittest.main()
