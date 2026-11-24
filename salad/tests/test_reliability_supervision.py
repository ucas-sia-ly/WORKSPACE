"""CPU tests for structural evidence, unknown masking, and auxiliary gradients."""

import unittest
from unittest.mock import patch

import torch

from workflow.reliability import build_local_targets, reliability_losses


class ReliabilitySupervisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def features(self, batch=1, size=20):
        return torch.randn(batch, 32, size, size, generator=torch.Generator().manual_seed(21))

    def warped(self):
        source = self.features()
        generated = source.clone()
        generated[..., 5:16, 5:16] = source[..., 5:16, 7:18]
        return source, generated

    def test_same_structure_has_positive_anchors_and_unknown_border(self):
        source = self.features()
        targets, confidence, info = build_local_targets(source, source)
        self.assertEqual(tuple(targets.shape), (1, 1, 20, 20))
        self.assertEqual(int(info["negative_patches"]), 0)
        self.assertGreater(int(info["positive_patches"]), 200)
        self.assertTrue((confidence[..., :2, :] == 0).all())
        self.assertTrue((targets[confidence == 0] == 0.5).all())
        self.assertTrue(torch.isfinite(confidence).all())

    def test_uniform_style_rotation_scale_and_offset_do_not_create_errors(self):
        source = self.features()
        matrix, _ = torch.linalg.qr(torch.randn(32, 32, generator=torch.Generator().manual_seed(7)))
        offset = torch.randn(1, 32, 1, 1, generator=torch.Generator().manual_seed(8)) * 10
        generated = 3 * torch.einsum("ij,bjhw->bihw", matrix, source) + offset
        expected_targets, expected_confidence, _ = build_local_targets(source, source)
        targets, confidence, info = build_local_targets(source, generated)
        torch.testing.assert_close(targets, expected_targets)
        torch.testing.assert_close(confidence, expected_confidence, atol=2e-5, rtol=2e-5)
        self.assertEqual(int(info["negative_patches"]), 0)

    def test_low_cross_image_cosine_alone_does_not_label_error(self):
        source = self.features()
        # Every corresponding channel vector has cosine -1; self-similarity
        # structure is unchanged, so no patch may be labeled an error.
        _, _, info = build_local_targets(source, -source)
        self.assertEqual(int(info["negative_patches"]), 0)
        self.assertGreater(int(info["positive_patches"]), 0)

    def test_local_structural_displacement_has_supported_negative_anchors(self):
        source, generated = self.warped()
        targets, confidence, info = build_local_targets(source, generated)
        self.assertGreater(int(info["negative_patches"]), 20)
        self.assertGreater(int(info["positive_patches"]), 0)
        negative = targets[0, 0] < 0.5
        allowed = torch.zeros_like(negative)
        allowed[5:16, 5:16] = True
        self.assertFalse((negative & ~allowed).any())
        self.assertTrue((confidence[targets < 0.5] > 0).all())

    def test_flat_or_tiny_grids_remain_unknown_and_finite(self):
        for source in (torch.ones(2, 8, 7, 7), self.features(size=3)):
            targets, confidence, info = build_local_targets(source, source)
            self.assertTrue((targets == 0.5).all())
            self.assertTrue((confidence == 0).all())
            self.assertEqual(int(info["positive_patches"] + info["negative_patches"]), 0)
            self.assertTrue(torch.isfinite(confidence).all())

    def test_teacher_targets_are_detached_from_both_feature_grids(self):
        source, generated = self.warped()
        targets, confidence, _ = build_local_targets(source.requires_grad_(), generated.requires_grad_())
        self.assertFalse(targets.requires_grad)
        self.assertFalse(confidence.requires_grad)

    def test_compact_and_full_aligned_companions_produce_same_losses(self):
        features = self.features(batch=3)
        logits = torch.zeros(3, 1, 20, 20, requires_grad=True)
        paired = torch.tensor([False, True, False])
        kinds = torch.tensor([False, True, True])
        compact = reliability_losses(logits, features, features[paired], paired, kinds)
        full = reliability_losses(logits, features, features, paired, kinds)
        for key in compact:
            torch.testing.assert_close(compact[key], full[key])

    def test_losses_propagate_head_gradients_but_not_detached_teacher_gradients(self):
        source, generated = self.warped()
        source.requires_grad_()
        generated.requires_grad_()
        logits = torch.zeros(1, 1, 20, 20, requires_grad=True)
        result = reliability_losses(logits, generated, source, torch.tensor([True]), torch.tensor([True]))
        result["total"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0)
        self.assertIsNone(source.grad)
        self.assertIsNone(generated.grad)
        self.assertGreater(int(result["negative_patches"]), 0)

    def test_unknown_teacher_regions_have_no_supervision_gradient(self):
        features = torch.ones(1, 8, 7, 7)
        logits = torch.randn(1, 1, 7, 7, requires_grad=True)
        result = reliability_losses(logits, features, features, torch.tensor([True]), torch.tensor([True]),
                                    real_prior_weight=0, coverage_weight=0)
        result["total"].backward()
        self.assertEqual(float(result["supervision"].detach()), 0)
        self.assertTrue((logits.grad == 0).all())

    def test_coverage_floor_pushes_collapsed_reliability_up(self):
        logits = torch.full((2, 1, 7, 7), -10.0, requires_grad=True)
        result = reliability_losses(logits, torch.ones(2, 8, 7, 7), None,
                                    torch.tensor([False, False]), torch.tensor([True, True]),
                                    auxiliary_weight=0, real_prior_weight=0)
        result["total"].backward()
        self.assertGreater(float(result["coverage"].detach()), 0)
        self.assertTrue((logits.grad < 0).all())
        self.assertEqual(float(result["below_floor_fraction"]), 1)

    def test_real_prior_is_soft_and_has_no_forced_synthetic_negative_quota(self):
        logits = torch.full((2, 1, 7, 7), 10.0, requires_grad=True)
        result = reliability_losses(logits, torch.ones(2, 8, 7, 7), None,
                                    torch.tensor([False, False]), torch.tensor([False, True]))
        result["total"].backward()
        self.assertTrue((logits.grad[0] > 0).all())
        self.assertTrue((logits.grad[1] == 0).all())
        self.assertEqual(float(result["coverage"].detach()), 0)
        self.assertEqual(float(result["supervision"].detach()), 0)

    def test_positive_negative_class_mass_is_balanced(self):
        targets = torch.full((1, 1, 10, 10), 0.95)
        targets[..., :1, :1] = 0.05
        confidence = torch.ones_like(targets)
        diagnostics = {"positive_patches": torch.tensor(99), "negative_patches": torch.tensor(1),
                       "unknown_patches": torch.tensor(0), "supervised_fraction": torch.tensor(1.)}
        logits = torch.zeros_like(targets, requires_grad=True)
        with patch("workflow.reliability.build_local_targets", return_value=(targets, confidence, diagnostics)):
            result = reliability_losses(logits, torch.ones(1, 8, 10, 10), torch.ones(1, 8, 10, 10),
                                        torch.tensor([True]), torch.tensor([True]), auxiliary_weight=1,
                                        real_prior_weight=0, coverage_weight=0)
        result["total"].backward()
        torch.testing.assert_close(logits.grad[targets < .5].sum(),
                                   -logits.grad[targets > .5].sum())

    def test_mixed_precision_inputs_produce_finite_float32_loss(self):
        for dtype in (torch.float16, torch.bfloat16):
            features = self.features().to(dtype)
            logits = torch.zeros(1, 1, 20, 20, dtype=dtype, requires_grad=True)
            result = reliability_losses(logits, features, features, torch.tensor([True]), torch.tensor([True]))
            self.assertEqual(result["total"].dtype, torch.float32)
            self.assertTrue(torch.isfinite(result["total"]))
            result["total"].backward()
            self.assertTrue(torch.isfinite(logits.grad).all())

    def test_invalid_pairing_and_nonfinite_data_fail_explicitly(self):
        features = self.features()
        logits = torch.zeros(1, 1, 20, 20)
        with self.assertRaisesRegex(ValueError, "only synthetic"):
            reliability_losses(logits, features, features, torch.tensor([True]), torch.tensor([False]))
        with self.assertRaisesRegex(ValueError, "require source"):
            reliability_losses(logits, features, None, torch.tensor([True]), torch.tensor([True]))
        with self.assertRaisesRegex(ValueError, "boolean"):
            reliability_losses(logits, features, None, torch.tensor([0]), torch.tensor([True]))
        with self.assertRaisesRegex(ValueError, "finite"):
            build_local_targets(features, features * float("nan"))
        with self.assertRaisesRegex(ValueError, "finite"):
            reliability_losses(logits + float("inf"), features, None,
                                torch.tensor([False]), torch.tensor([True]))
        with self.assertRaisesRegex(ValueError, "weights"):
            reliability_losses(logits, features, None, torch.tensor([False]), torch.tensor([True]),
                                auxiliary_weight=-1)


if __name__ == "__main__":
    unittest.main()
