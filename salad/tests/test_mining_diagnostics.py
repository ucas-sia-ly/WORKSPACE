"""CPU checks for no-update saturation certificates and source attribution."""
import unittest

import torch

from workflow.mining_diagnostics import (common_anchor_statistics,
                                         source_aware_mined_pair_statistics)
from workflow.metric_loss import multi_similarity_loss


class SourceAwareMiningTest(unittest.TestCase):
    def test_directed_source_pairs_partition_masks(self):
        x = torch.tensor([[1., 0.], [.8, .6], [.6, .8], [0., 1.]], requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])
        flags = torch.tensor([False, True, False, True])
        result = source_aware_mined_pair_statistics(x, labels, flags)
        self.assertIsNone(x.grad)
        for name, total in (("positive", "positive_pairs"), ("negative", "negative_pairs"),
                            ("mined_positive", "mined_positive_pairs"),
                            ("mined_negative", "mined_negative_pairs")):
            self.assertEqual(sum(row[name] for row in result["source_pairs"].values()), result[total])
        self.assertEqual(result["source_pairs"]["real_synthetic"]["positive"], 2)
        self.assertEqual(result["source_pairs"]["synthetic_real"]["positive"], 2)
        self.assertEqual(result["source_pairs"]["real_real"]["positive"], 0)
        self.assertFalse(result["zero_ms_gradient_certified_by_empty_mask"])
        common = common_anchor_statistics(result, 2)
        self.assertEqual(common["active_anchors"], sum(a["ms_active"] for a in result["per_anchor"][:2]))

    def test_empty_mask_certifies_exact_zero_ms_gradient(self):
        x = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]], requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])
        result = source_aware_mined_pair_statistics(x, labels, torch.zeros(4, dtype=torch.bool))
        self.assertTrue(result["zero_ms_gradient_certified_by_empty_mask"])
        self.assertFalse(result["numerically_zero_loss_with_nonempty_mask"])
        gradient, = torch.autograd.grad(multi_similarity_loss(x, labels), (x,))
        self.assertEqual(torch.count_nonzero(gradient), 0)

    def test_all_real_counts_and_validation(self):
        x, labels = torch.eye(4), torch.tensor([0, 0, 1, 1])
        result = source_aware_mined_pair_statistics(x, labels, torch.zeros(4, dtype=torch.bool))
        self.assertEqual(result["source_anchors"]["synthetic"]["anchors"], 0)
        self.assertEqual(result["source_pairs"]["real_real"]["mined_positive"], result["mined_positive_pairs"])
        with self.assertRaises(ValueError):
            source_aware_mined_pair_statistics(x, labels, torch.zeros(4))
        with self.assertRaises(ValueError):
            common_anchor_statistics(result, 5)

    def test_fast_monitor_counts_match_detailed_diagnostic(self):
        generator = torch.Generator().manual_seed(42)
        x = torch.randn(32, 8, generator=generator)
        labels = torch.arange(8).repeat_interleave(4)
        flags = torch.arange(32) % 3 == 0
        detailed = source_aware_mined_pair_statistics(x, labels, flags)
        fast = source_aware_mined_pair_statistics(x, labels, flags, include_per_anchor=False,
                                                loss_value=detailed["loss"])
        self.assertNotIn("per_anchor", fast)
        for key, value in fast.items():
            if isinstance(value, float):
                self.assertAlmostEqual(value, detailed[key], places=7)
            else:
                self.assertEqual(value, detailed[key])


if __name__ == "__main__":
    unittest.main()
