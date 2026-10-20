"""CPU checks for non-mutating diagnostics and exact MS mining statistics."""

import json
import math
import unittest

import torch

from workflow.diagnostic_gradients import (collect_loss_gradients, default_parameter_groups,
    gradient_vector_summary, mined_pair_statistics)
from workflow.metric_loss import multi_similarity_loss


class MinedPairDiagnosticsTest(unittest.TestCase):
    def test_masks_match_direct_reference_and_ms_loss(self):
        descriptors = torch.tensor([[1., 0.], [.8, .6], [.6, .8], [0., 1.]], requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])
        stats = mined_pair_statistics(descriptors, labels)
        similarity = torch.nn.functional.normalize(descriptors.detach(), dim=1)
        similarity = similarity @ similarity.T
        positive_count = negative_count = 0
        for i, anchor in enumerate(stats["per_anchor"]):
            pos = [j for j in range(4) if j != i and labels[i] == labels[j]]
            neg = [j for j in range(4) if labels[i] != labels[j]]
            mp = sum(bool(similarity[i, j] < max(similarity[i, k] for k in neg) + .1) for j in pos)
            mn = sum(bool(similarity[i, j] > min(similarity[i, k] for k in pos) - .1) for j in neg)
            self.assertEqual(anchor["mined_positive"], mp)
            self.assertEqual(anchor["mined_negative"], mn)
            positive_count += mp
            negative_count += mn
        self.assertEqual(stats["positive_pairs"], 4)
        self.assertEqual(stats["negative_pairs"], 8)
        self.assertEqual(stats["mined_positive_pairs"], positive_count)
        self.assertEqual(stats["mined_negative_pairs"], negative_count)
        self.assertEqual(stats["loss"], multi_similarity_loss(descriptors, labels).item())
        self.assertIsNone(descriptors.grad)
        json.dumps(stats, allow_nan=False)

    def test_easy_batches_have_no_mined_pairs_and_zero_loss(self):
        stats = mined_pair_statistics(torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]]),
                                      torch.tensor([1, 1, 2, 2]))
        self.assertEqual(stats["mined_positive_pairs"], 0)
        self.assertEqual(stats["mined_negative_pairs"], 0)
        self.assertEqual(stats["active_anchor_ratio"], 0)
        self.assertEqual(stats["loss"], 0)

    def test_absent_positive_and_negative_are_explicit(self):
        only_one_place = mined_pair_statistics(torch.ones(2, 3), torch.tensor([1, 1]))
        all_unique = mined_pair_statistics(torch.eye(2), torch.tensor([1, 2]))
        self.assertIsNone(only_one_place["per_anchor"][0]["hardest_negative_similarity"])
        self.assertIsNone(all_unique["per_anchor"][0]["hardest_positive_similarity"])
        self.assertEqual(only_one_place["loss"], 0)
        self.assertEqual(all_unique["loss"], 0)
        with self.assertRaises(ValueError):
            mined_pair_statistics(torch.empty(0, 3), torch.empty(0, dtype=torch.long))


class GradientDiagnosticsTest(unittest.TestCase):
    def test_shared_graph_gradients_do_not_change_weights_or_existing_grad(self):
        layer = torch.nn.Linear(2, 2)
        parameters = list(layer.named_parameters())
        before = {name: parameter.detach().clone() for name, parameter in parameters}
        layer.weight.grad = torch.ones_like(layer.weight) * 17
        before_grad = layer.weight.grad.clone()
        y = layer(torch.tensor([[1., 2.]]))
        gradients = collect_loss_gradients({"real": y.square().sum(), "syn": y.sum(),
                                           "absent": torch.tensor(0.)}, parameters)
        for name, parameter in parameters:
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        torch.testing.assert_close(layer.weight.grad, before_grad, rtol=0, atol=0)
        self.assertIsNone(layer.bias.grad)
        self.assertTrue(all(value is None for value in gradients["absent"]))
        self.assertTrue(all(value.device.type == "cpu" and not value.requires_grad
                            for values in gradients.values() for value in values if value is not None))

    def test_norm_angles_weighting_and_clipping_match_explicit_vectors(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        named = [("aggregator.score.weight", parameter)]
        grads = {"real": (torch.tensor([3., 0.]),),
                 "syn": (torch.tensor([0., 4.]),), "score": (torch.tensor([5., 5.]),)}
        stats = gradient_vector_summary(grads, named, {"real": 1, "syn": .5, "score": 0})
        total = stats["groups"]["all"]
        self.assertEqual(total["losses"]["real"]["raw_norm"], 3)
        self.assertEqual(total["losses"]["syn"]["weighted_norm"], 2)
        self.assertAlmostEqual(total["losses"]["score"]["raw_norm"], math.sqrt(50))
        self.assertEqual(total["losses"]["score"]["weighted_norm"], 0)
        self.assertEqual(total["cosines"]["real|syn"]["value"], 0)
        self.assertEqual(total["cosines"]["real|syn"]["angle_degrees"], 90)
        self.assertAlmostEqual(total["combined"]["norm"], math.sqrt(13))
        self.assertAlmostEqual(total["combined"]["vs_real"]["value"], 3 / math.sqrt(13))
        self.assertTrue(total["combined"]["would_clip"])
        self.assertAlmostEqual(total["combined"]["clip_scale"], 1 / (math.sqrt(13) + 1e-6))
        json.dumps(stats, allow_nan=False)

    def test_opposing_and_zero_gradients_have_undefined_total_angle(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        stats = gradient_vector_summary({"real": (torch.tensor([1., -2.]),),
            "syn": (torch.tensor([-1., 2.]),), "feature": (None,)}, [("x", parameter)])
        all_stats = stats["groups"]["all"]
        self.assertEqual(all_stats["cosines"]["real|syn"]["value"], -1)
        self.assertFalse(all_stats["cosines"]["real|feature"]["defined"])
        self.assertEqual(all_stats["combined"]["norm"], 0)
        self.assertFalse(all_stats["combined"]["vs_real"]["defined"])
        self.assertEqual(all_stats["combined"]["clip_scale"], 1)

    def test_salad_parameter_groups_and_error_guards(self):
        names = ["backbone.model.blocks.11.attn.qkv.weight", "backbone.model.norm.weight",
                 "aggregator.score.0.weight", "aggregator.cluster_features.0.weight",
                 "aggregator.token_features.0.weight", "aggregator.dust_bin"]
        named = [(name, torch.nn.Parameter(torch.ones(2))) for name in names]
        groups = default_parameter_groups(named)
        self.assertEqual(groups["backbone"], [0, 1])
        self.assertEqual(groups["backbone_block_11"], [0])
        self.assertEqual(groups["backbone_norm"], [1])
        self.assertEqual(groups["score"], [2])
        self.assertEqual(groups["content"], [3])
        self.assertEqual(groups["token"], [4])
        self.assertEqual(groups["dustbin"], [5])
        gradients = {"real": (None,) * len(named)}
        with self.assertRaises(ValueError):
            gradient_vector_summary(gradients, named, {"real": -1})
        with self.assertRaises(ValueError):
            gradient_vector_summary(gradients, named, groups={"bad": [0, 0]})
        with self.assertRaises(RuntimeError):
            collect_loss_gradients({"real": torch.tensor(float("nan"))}, named)


if __name__ == "__main__":
    unittest.main()
