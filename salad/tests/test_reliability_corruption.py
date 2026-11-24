"""Local damage, context/detach branches, and ablation compatibility tests."""

import math
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

SALAD_ROOT = Path(__file__).resolve().parents[1]
if str(SALAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SALAD_ROOT))

from models.aggregators.salad import SALAD
from workflow.reliability_corruption import corrupted_patch_loss, make_local_corruptions


class LocalCorruptionTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.manual_seed(23)

    @staticmethod
    def images(batch=5, grid_h=8, grid_w=8, patch=2):
        tiles = torch.arange(grid_h * grid_w, dtype=torch.float32).reshape(1, 1, grid_h, grid_w)
        pixels = tiles.repeat_interleave(patch, -2).repeat_interleave(patch, -1)
        return pixels.repeat(batch, 3, 1, 1) + torch.arange(batch)[:, None, None, None] * 1000

    def test_only_real_views_and_bounded_patch_aligned_copy(self):
        for grid_h, grid_w in ((4, 4), (8, 8), (5, 7)):
            with self.subTest(grid=(grid_h, grid_w)):
                images = self.images(grid_h=grid_h, grid_w=grid_w)
                original = images.clone()
                kinds = torch.tensor([False, True, False, False, True])
                damaged, mask, indices = make_local_corruptions(images, kinds, max_images=2, patch_size=2)
                self.assertEqual(damaged.shape[0], 2)
                self.assertEqual(mask.shape, (2, 1, grid_h, grid_w))
                self.assertEqual(mask.dtype, torch.bool)
                self.assertEqual(indices.unique().numel(), 2)
                self.assertFalse(kinds[indices].any())
                self.assertTrue(torch.equal(original, images))
                pixel_mask = mask.repeat_interleave(2, -2).repeat_interleave(2, -1).expand_as(damaged)
                # Unique numbered tiles make every copied destination differ.
                self.assertTrue(torch.equal(damaged != images[indices], pixel_mask))
                self.assertTrue((mask.flatten(1).sum(1) >= 4).all())
                self.assertTrue((mask.flatten(1).sum(1) <= grid_h * grid_w / 4).all())
                for row, index in zip(damaged, indices):
                    self.assertTrue(torch.isin(row.unique(), images[index].unique()).all())

    def test_reproducible_and_rng_state_can_resume(self):
        images = self.images()
        kinds = torch.tensor([False, True, False, False, True])
        state = torch.get_rng_state()
        first = make_local_corruptions(images, kinds, patch_size=2)
        next_draw = torch.rand(5)
        torch.set_rng_state(state)
        second = make_local_corruptions(images, kinds, patch_size=2)
        for left, right in zip(first, second):
            self.assertTrue(torch.equal(left, right))
        self.assertTrue(torch.equal(next_draw, torch.rand(5)))

    def test_explicit_generator_is_reproducible_and_preserves_global_rng(self):
        images = self.images()
        kinds = torch.tensor([False, True, False, False, True])
        state = torch.get_rng_state()
        generator = torch.Generator(device=images.device).manual_seed(907)
        first = make_local_corruptions(images, kinds, patch_size=2, generator=generator)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        # Unrelated global draws cannot affect a separately reseeded generator.
        torch.rand(13)
        later_state = torch.get_rng_state()
        other = torch.Generator(device=images.device).manual_seed(907)
        second = make_local_corruptions(images, kinds, patch_size=2, generator=other)
        self.assertTrue(torch.equal(later_state, torch.get_rng_state()))
        for left, right in zip(first, second):
            self.assertTrue(torch.equal(left, right))

    def test_uniform_images_create_no_fake_negative_anchors(self):
        images = torch.full((2, 3, 16, 16), 6.)
        corrupted, mask, indices = make_local_corruptions(
            images, torch.zeros(2, dtype=torch.bool), patch_size=2,
        )
        self.assertTrue(torch.equal(corrupted, images[indices]))
        self.assertFalse(mask.any())
        logits = torch.randn(mask.shape, requires_grad=True)
        loss = corrupted_patch_loss(logits, mask)
        self.assertEqual(int(loss["corruption_negative_patches"]), 0)
        self.assertEqual(float(loss["corruption_loss"].detach()), 0.)

    def test_identical_nonuniform_regions_create_no_fake_negative_anchors(self):
        # With a 4x4 grid, vertical halves each contain these same two rows;
        # every source/target rectangle is identical despite internal texture.
        tiles = torch.tensor([0., 1., 0., 1.]).reshape(1, 1, 4, 1).repeat(2, 3, 1, 4)
        images = tiles.repeat_interleave(2, -2).repeat_interleave(2, -1)
        self.assertGreater(float(images.var()), 0.)
        corrupted, mask, indices = make_local_corruptions(
            images, torch.zeros(2, dtype=torch.bool), patch_size=2,
        )
        self.assertTrue(torch.equal(corrupted, images[indices]))
        self.assertFalse(mask.any())

    def test_empty_selection_does_not_consume_rng(self):
        images = self.images(batch=2)
        for kinds, cap in ((torch.ones(2, dtype=torch.bool), 4),
                           (torch.zeros(2, dtype=torch.bool), 0)):
            before = torch.get_rng_state()
            damaged, mask, indices = make_local_corruptions(images, kinds, max_images=cap, patch_size=2)
            self.assertEqual(damaged.shape, (0, 3, 16, 16))
            self.assertEqual(mask.shape, (0, 1, 8, 8))
            self.assertEqual(indices.shape, (0,))
            self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_corruption_input_gradients_are_detached(self):
        images = self.images(batch=1).requires_grad_()
        corrupted, _, _ = make_local_corruptions(images, torch.zeros(1, dtype=torch.bool), patch_size=2)
        self.assertFalse(corrupted.requires_grad)
        self.assertIsNone(images.grad)

    def test_masked_bce_has_zero_gradient_outside_known_damage(self):
        logits = torch.full((2, 1, 8, 8), math.log(9.0), requires_grad=True)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[:, :, 2:4, 4:6] = True
        losses = corrupted_patch_loss(logits, mask)
        self.assertEqual(int(losses["corruption_negative_patches"]), 8)
        self.assertAlmostEqual(float(losses["corruption_mean_reliability"]), .9, places=6)
        losses["corruption_loss"].backward()
        self.assertTrue(torch.equal(logits.grad[~mask], torch.zeros_like(logits.grad[~mask])))
        self.assertTrue((logits.grad[mask] > 0).all())
        torch.testing.assert_close(logits.grad[mask], torch.full((8,), (.9 - .05) / 8))

    def test_empty_mask_and_batch_are_differentiable_zero(self):
        for shape in ((2, 1, 8, 8), (0, 1, 8, 8)):
            logits = torch.zeros(shape, requires_grad=True)
            losses = corrupted_patch_loss(logits, torch.zeros(shape, dtype=torch.bool))
            self.assertEqual(float(losses["corruption_loss"].detach()), 0.)
            self.assertEqual(int(losses["corruption_negative_patches"]), 0)
            losses["corruption_loss"].backward()
            self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_amp_and_extreme_half_logits_are_finite(self):
        for dtype in (torch.float16, torch.bfloat16):
            logits = torch.tensor([1000., -1000., 0., 1.], dtype=dtype).reshape(1, 1, 2, 2)
            logits.requires_grad_()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                losses = corrupted_patch_loss(logits, torch.ones_like(logits, dtype=torch.bool))
            self.assertEqual(losses["corruption_loss"].dtype, torch.float32)
            self.assertTrue(torch.isfinite(losses["corruption_loss"]))
            losses["corruption_loss"].backward()
            self.assertTrue(torch.isfinite(logits.grad).all())

    def test_clean_reference_detached_and_outside_consistency_is_localized(self):
        logits = torch.zeros(1, 1, 8, 8, requires_grad=True)
        reference = torch.full_like(logits, math.log(9.0), requires_grad=True)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[:, :, 2:4, 4:6] = True
        losses = corrupted_patch_loss(logits, mask, reference_logits=reference)
        self.assertGreater(float(losses["corruption_consistency"].detach()), 0.)
        torch.testing.assert_close(losses["corruption_loss"],
                                   losses["corruption_negative_loss"] + .1 * losses["corruption_consistency"])
        losses["corruption_loss"].backward()
        self.assertIsNone(reference.grad)
        self.assertTrue((logits.grad[mask] > 0).all())
        self.assertTrue((logits.grad[~mask] < 0).all())

    def test_matching_clean_reference_has_zero_outside_gradient(self):
        logits = torch.randn(1, 1, 8, 8, requires_grad=True)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[:, :, 2:4, 4:6] = True
        losses = corrupted_patch_loss(logits, mask, reference_logits=logits)
        self.assertEqual(float(losses["corruption_consistency"].detach()), 0.)
        losses["corruption_loss"].backward()
        self.assertTrue(torch.equal(logits.grad[~mask], torch.zeros_like(logits.grad[~mask])))

    def test_invalid_shapes_parameters_and_nonfinite_logits_rejected(self):
        images = self.images(batch=2)
        kinds = torch.zeros(2, dtype=torch.bool)
        for cap in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                make_local_corruptions(images, kinds, max_images=cap, patch_size=2)
        for patch in (0, True, 1.5, 3):
            with self.assertRaises(ValueError):
                make_local_corruptions(images, kinds, patch_size=patch)
        with self.assertRaisesRegex(ValueError, "4x4"):
            make_local_corruptions(self.images(batch=2, grid_h=3), kinds, patch_size=2)
        with self.assertRaises(ValueError):
            make_local_corruptions(images, kinds.float(), patch_size=2)
        with self.assertRaises(ValueError):
            corrupted_patch_loss(torch.zeros(1, 2, 4, 4), torch.zeros(1, 2, 4, 4, dtype=torch.bool))
        with self.assertRaises(ValueError):
            corrupted_patch_loss(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))
        with self.assertRaisesRegex(ValueError, "finite"):
            corrupted_patch_loss(torch.full((1, 1, 4, 4), float("nan")),
                                 torch.ones(1, 1, 4, 4, dtype=torch.bool))


class ReliabilityContextTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.manual_seed(31)

    @staticmethod
    def model(**kwargs):
        kwargs.setdefault("reliability_hidden_dim", 8)
        return SALAD(num_channels=12, num_clusters=4, cluster_dim=6,
                     token_dim=8, dropout=0, **kwargs)

    @staticmethod
    def inputs():
        return torch.randn(2, 12, 4, 4), torch.randn(2, 12)

    def test_original_weights_and_next_rng_state_identical(self):
        torch.manual_seed(731)
        original = self.model()
        expected_rng = torch.get_rng_state()
        expected_next = torch.randn(9)
        for context in (False, True):
            for mode in ("learned", "fixed"):
                torch.manual_seed(731)
                enabled = self.model(reliability_ot=True, reliability_context=context,
                                     reliability_mode=mode)
                self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                self.assertTrue(torch.equal(expected_next, torch.randn(9)))
                for key, value in original.state_dict().items():
                    self.assertTrue(torch.equal(value, enabled.state_dict()[key]), key)

    def test_context_adds_only_ten_parameters_per_hidden_channel(self):
        base = self.model(reliability_ot=True)
        context = self.model(reliability_ot=True, reliability_context=True)
        self.assertEqual(sum(p.numel() for p in context.reliability_parameters())
                         - sum(p.numel() for p in base.reliability_parameters()), 80)
        self.assertEqual(context.reliability_context.groups, 8)
        self.assertEqual(context.reliability_context.kernel_size, (3, 3))
        self.assertEqual(len(list(self.model().reliability_parameters())), 0)
        self.assertEqual(set(context.state_dict()) - set(base.state_dict()),
                         {"reliability_context.weight", "reliability_context.bias"})
        logits = context.predict_reliability(self.inputs()[0])
        torch.testing.assert_close(logits.sigmoid(), torch.full_like(logits, .9))

    def test_context_has_neighbor_support_but_legacy_head_is_pointwise(self):
        pointwise = self.model(reliability_ot=True)
        context = self.model(reliability_ot=True, reliability_context=True)
        for model in (pointwise, context):
            with torch.no_grad():
                model.reliability_head[0].weight.fill_(.1)
                model.reliability_head[0].bias.zero_()
                model.reliability_head[2].weight.fill_(.1)
                model.reliability_head[2].bias.zero_()
                if model.reliability_context_enabled:
                    model.reliability_context.weight.fill_(.1)
                    model.reliability_context.bias.zero_()
        x = torch.ones(1, 12, 4, 4)
        changed = x.clone()
        changed[:, :, 1, 1] += 1.
        self.assertEqual(float(pointwise.predict_reliability(x)[0, 0, 1, 2].detach()),
                         float(pointwise.predict_reliability(changed)[0, 0, 1, 2].detach()))
        self.assertGreater(float(context.predict_reliability(changed)[0, 0, 1, 2].detach()),
                           float(context.predict_reliability(x)[0, 0, 1, 2].detach()))

    def test_detach_stops_auxiliary_backbone_gradient_and_keeps_head_gradient(self):
        for detach in (False, True):
            with self.subTest(detach=detach):
                model = self.model(reliability_ot=True, reliability_context=True,
                                   reliability_detach_features=detach)
                nn.init.normal_(model.reliability_head[2].weight, std=.05)
                x = self.inputs()[0].requires_grad_()
                model.predict_reliability(x).square().mean().backward()
                if detach:
                    self.assertIsNone(x.grad)
                else:
                    self.assertGreater(float(x.grad.abs().sum()), 0.)
                for parameter in model.reliability_parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(model.reliability_context.weight.grad.abs().sum()), 0.)

    def test_detach_does_not_disable_descriptor_backbone_gradient(self):
        model = self.model(reliability_ot=True, reliability_context=True,
                           reliability_detach_features=True)
        x, token = self.inputs()
        x.requires_grad_()
        token.requires_grad_()
        descriptor = model((x, token))
        (descriptor * torch.randn_like(descriptor)).sum().backward()
        self.assertGreater(float(x.grad.abs().sum()), 0.)
        self.assertGreater(float(token.grad.abs().sum()), 0.)
        self.assertGreater(float(model.reliability_head[2].weight.grad.abs().sum()), 0.)

    def test_fixed_mode_ignores_head_and_context_and_preserves_metric_gradient(self):
        model = self.model(reliability_ot=True, reliability_context=True, reliability_mode="fixed")
        for parameter in model.reliability_parameters():
            nn.init.normal_(parameter)
        x, token = self.inputs()
        x.requires_grad_()
        descriptor, aux = model((x, token), return_aux=True)
        self.assertFalse(aux["reliability_logits"].requires_grad)
        torch.testing.assert_close(aux["reliability"], torch.full_like(aux["reliability"], .9))
        (descriptor * torch.randn_like(descriptor)).sum().backward()
        self.assertGreater(float(x.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in model.reliability_parameters()))

    def test_fixed_mode_state_round_trip_and_invalid_marker_rejected(self):
        model = self.model(reliability_ot=True, reliability_context=True, reliability_mode="fixed")
        restored = self.model(reliability_ot=True, reliability_context=True, reliability_mode="fixed")
        restored.load_state_dict(model.state_dict(), strict=True)
        inputs = self.inputs()
        self.assertTrue(torch.equal(model(inputs), restored(inputs)))
        self.assertEqual(model.state_dict()["reliability_fixed_mode"].dtype, torch.bool)
        for marker in (torch.tensor(False), torch.tensor(1), torch.tensor([True])):
            state = dict(model.state_dict())
            state["reliability_fixed_mode"] = marker
            with self.assertRaisesRegex(RuntimeError, "scalar boolean True"):
                restored.load_state_dict(state, strict=True)
        learned = self.model(reliability_ot=True)
        self.assertNotIn("reliability_fixed_mode", learned.state_dict())

    def test_context_amp_logits_and_gradients_finite(self):
        # Some CPU DNNL builds cannot backpropagate depthwise BF16. Native
        # convolution lets this verify model AMP behavior independent of ISA.
        with torch.backends.mkldnn.flags(enabled=False):
            model = self.model(reliability_ot=True, reliability_context=True)
            nn.init.normal_(model.reliability_head[2].weight, std=.03)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                descriptor, aux = model(self.inputs(), return_aux=True)
                loss = descriptor.square().sum() + aux["reliability_logits"].square().mean()
            self.assertEqual(aux["reliability_logits"].dtype, torch.float32)
            self.assertTrue(torch.isfinite(descriptor).all())
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.reliability_parameters()))

    def test_invalid_mode_and_disabled_predictor_fail_explicitly(self):
        with self.assertRaises(ValueError):
            self.model(reliability_ot=True, reliability_mode="unknown")
        with self.assertRaisesRegex(RuntimeError, "reliability_ot"):
            self.model().predict_reliability(self.inputs()[0])


if __name__ == "__main__":
    unittest.main()
