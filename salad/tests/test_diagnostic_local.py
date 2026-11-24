"""Read-only dropout controls distinguish backbone, head and checkpoint drift."""
import copy
import json
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from workflow.diagnostic_local import local_consistency_probe
from workflow.pair_correspondence import CorrespondenceCalibration


class TinyLocalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Conv2d(3, 8, 1)
        self.score = nn.Conv2d(8, 3, 1)
        self.content = nn.Conv2d(8, 4, 1)
        self.score_dropout = nn.Dropout(.5)
        self.content_dropout = nn.Dropout(.5)
        self.register_buffer("forward_count", torch.zeros((), dtype=torch.long))

    def forward(self, images, return_aux=False):
        self.forward_count.add_(1)
        features = self.backbone(F.adaptive_avg_pool2d(images, (2, 2)))
        scores = self.score(self.score_dropout(features)).flatten(2)
        content = self.content(self.content_dropout(features)).flatten(2)
        routing = torch.cat((scores, torch.zeros_like(scores[:, :1])), dim=1).log_softmax(1)
        descriptor = content.mean(-1)
        aux = dict(backbone_local_features=features, score_logits=scores, log_assignment=routing)
        return (descriptor, aux) if return_aux else descriptor


class DiagnosticLocalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(8)
        self.student = TinyLocalModel()
        self.teacher = copy.deepcopy(self.student).eval().requires_grad_(False)
        # Unique local RGB texture avoids the repeated-patch ambiguity veto.
        image = np.random.default_rng(18).integers(10, 240, (28, 28, 3), dtype=np.uint8)
        tensor = torch.tensor(image).permute(2, 0, 1).float()[None] / 255
        generated = np.clip(image * .8 + 8, 0, 255).astype(np.uint8)
        self.pair = dict(source_tensor=tensor, generated_tensor=tensor * .8 + 8 / 255,
            source_image=Image.fromarray(image), generated_image=Image.fromarray(generated),
            source_path="source.jpg", generated_path="night.jpg", condition="night",
            quality=dict(passed=True, eligible_for_training=True))
        self.calibration = CorrespondenceCalibration(-.9, 0., 2., .01, 1000, 1, {})

    def probe(self, **kwargs):
        return local_consistency_probe(self.student, self.teacher, [self.pair], self.calibration,
                                       device="cpu", repeats=3, seed=91, **kwargs)

    def test_same_checkpoint_eval_zero_backbone_train_zero_head_train_nonzero(self):
        result = self.probe(precisions=("32",))
        rows = [r for r in result["rows"] if r["view"] == "source_self"]
        baseline = next(r for r in rows if r["student_mode"] == "eval")
        for key in ("feature", "score", "ot"):
            self.assertLess(abs(baseline[key]), 1e-6)
        stochastic = [r for r in rows if r["student_mode"] == "train"]
        self.assertTrue(all(abs(r["feature"]) < 1e-6 for r in stochastic))
        self.assertGreater(sum(r["score"] for r in stochastic), 0.)
        self.assertGreater(sum(r["ot"] for r in stochastic), 0.)
        self.assertGreater(len({r["score"] for r in stochastic}), 1)
        self.assertEqual(result["protocol"]["optimizer_steps"], 0)

    def test_state_buffers_rng_grads_and_mixed_modes_are_restored(self):
        self.student.train()
        self.student.score_dropout.eval()
        self.student.backbone.weight.grad = torch.ones_like(self.student.backbone.weight)
        before = {k: v.clone() for k, v in self.student.state_dict().items()}
        teacher_before = {k: v.clone() for k, v in self.teacher.state_dict().items()}
        flags = [m.training for m in self.student.modules()]
        parameter_versions = [p._version for p in self.student.parameters()]
        rng = torch.get_rng_state().clone()
        result = self.probe(precisions=("32",))
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for k, value in self.student.state_dict().items():
            torch.testing.assert_close(value, before[k], rtol=0, atol=0)
        for k, value in self.teacher.state_dict().items():
            torch.testing.assert_close(value, teacher_before[k], rtol=0, atol=0)
        self.assertEqual(flags, [m.training for m in self.student.modules()])
        self.assertEqual(parameter_versions, [p._version for p in self.student.parameters()])
        self.assertTrue((self.student.backbone.weight.grad == 1).all())
        self.assertTrue(all(p.grad is None for p in self.teacher.parameters()))
        self.assertEqual(result, self.probe(precisions=("32",)))
        json.dumps(result, allow_nan=False)

    def test_own_eval_reference_separates_drift_from_randomness(self):
        with torch.no_grad():
            self.student.backbone.weight.add_(.7)
            self.student.score.weight.mul_(-2)
        rows = self.probe(precisions=("32",))["rows"]
        same = next(r for r in rows if r["view"] == "source_self" and r["student_mode"] == "eval")
        drift = next(r for r in rows if r["view"] == "source_copy" and r["student_mode"] == "eval")
        self.assertLess(same["feature"], 1e-6)
        self.assertLess(same["score"], 1e-6)
        self.assertGreater(drift["feature"], 0.)
        self.assertGreater(drift["score"], 0.)

    def test_verified_support_fixed_for_all_modes_precisions_and_unknown_not_zero_evidence(self):
        self.calibration = None
        result = self.probe()
        rows = [r for r in result["rows"] if r["correspondence"] == "verified"]
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(r["positive_patches"] == 0 for r in rows))
        self.assertTrue(all(r["feature"] == r["score"] == r["ot"] == 0 for r in rows))
        self.assertTrue(all(s["distributions"]["feature"]["n"] == 0
                            for k, s in result["summaries"].items() if "|verified|" in k))
        self.assertEqual(result["correspondence"][0]["positive_a_patches"], 0)

    def test_bf16_controls_finite_and_precision_labels_explicit(self):
        # A known copy must really exercise positive weighted gathering, rather
        # than accidentally passing every mode because all q are zero.
        self.pair["generated_tensor"] = self.pair["source_tensor"].clone()
        self.pair["generated_image"] = self.pair["source_image"].copy()
        result = self.probe()
        self.assertEqual({r["precision"] for r in result["rows"]}, {"32", "bf16"})
        for row in result["rows"]:
            self.assertTrue(all(np.isfinite(row[k]) for k in ("feature", "score", "ot", "feature_cosine")))
        supports = [r["positive_patches"] for r in result["rows"] if r["correspondence"] == "verified"]
        self.assertEqual(len(set(supports)), 1)
        self.assertGreater(supports[0], 0)
        json.dumps(result, allow_nan=False)

    def test_failures_preserve_rng_and_modes(self):
        self.student.train()
        self.teacher.eval()
        self.pair["source_tensor"] = torch.zeros(1, 4, 28, 28)
        rng = torch.get_rng_state().clone()
        with self.assertRaisesRegex(ValueError, "must be"):
            self.probe()
        self.assertTrue(self.student.training)
        self.assertFalse(self.teacher.training)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
