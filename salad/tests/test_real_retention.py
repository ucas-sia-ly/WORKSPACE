"""Numerical, gradient, mode and streaming checks for real-domain retention."""
import copy
import unittest

import torch
from torch import nn
import torch.nn.functional as F

from workflow.real_retention import (descriptor_retention_loss, local_eval_with_grad,
                                      backward_real_retention)


class TinyDescriptor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(12, 7)
        self.dropout = nn.Dropout(.8)
        self.calls = 0

    def forward(self, x, return_aux=False):
        self.calls += 1
        return F.normalize(self.dropout(self.linear(x.flatten(1))), dim=1)


class RealRetentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(53)

    def test_cosine_formula_fp32_and_teacher_has_no_gradient(self):
        student = torch.randn(5, 11, requires_grad=True)
        teacher = torch.randn(5, 11, requires_grad=True)
        loss = descriptor_retention_loss(student, teacher)
        expected = (1 - F.cosine_similarity(student, teacher.detach(), dim=1)).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertGreater(float(student.grad.norm()), 0)
        self.assertIsNone(teacher.grad)

    def test_identical_and_zero_vectors_exact_zero_including_gradient(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for source in (torch.randn(3, 9), torch.zeros(3, 9)):
                student = source.to(dtype).clone().requires_grad_()
                teacher = source.to(dtype).clone().requires_grad_()
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    loss = descriptor_retention_loss(student, teacher)
                self.assertEqual(loss.dtype, torch.float32)
                self.assertEqual(float(loss.detach()), 0.)
                loss.backward()
                self.assertEqual(float(student.grad.abs().sum()), 0.)
                self.assertIsNone(teacher.grad)

    def test_empty_and_zero_teacher_rows_are_safe_and_reduce_over_batch(self):
        student = torch.empty(0, 5, requires_grad=True)
        loss = descriptor_retention_loss(student, torch.empty_like(student))
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertEqual(student.grad.shape, student.shape)
        student = torch.randn(3, 5, requires_grad=True)
        teacher = torch.randn(3, 5)
        teacher[0].zero_()
        expected = descriptor_retention_loss(student[1:], teacher[1:]) * (2 / 3)
        torch.testing.assert_close(descriptor_retention_loss(student, teacher), expected)

    def test_nonfinite_and_shape_inputs_rejected(self):
        for value in (float("nan"), float("inf")):
            a, b = torch.randn(2, 4), torch.randn(2, 4)
            a[0, 0] = value
            with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
                descriptor_retention_loss(a, b)
            with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
                descriptor_retention_loss(b, a)
        for a, b in ((torch.randn(2, 4), torch.randn(2, 3)),
                     (torch.randn(2, 4, 1), torch.randn(2, 4, 1)),
                     (torch.ones(2, 4, dtype=torch.long), torch.ones(2, 4, dtype=torch.long))):
            with self.assertRaises(ValueError):
                descriptor_retention_loss(a, b)

    def test_extreme_finite_vectors_normalize_without_overflow_under_amp(self):
        for dtype, magnitude in ((torch.float16, 60000.), (torch.bfloat16, 2e38), (torch.float32, 2e38)):
            student = torch.tensor([[magnitude, 0]], dtype=dtype, requires_grad=True)
            teacher = torch.tensor([[0, magnitude]], dtype=dtype, requires_grad=True)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss = descriptor_retention_loss(student, teacher)
            self.assertAlmostEqual(float(loss.detach()), 1.)
            loss.backward()
            self.assertTrue(torch.isfinite(student.grad).all())
            self.assertIsNone(teacher.grad)

    def test_context_preserves_modes_buffers_rng_and_allows_backward(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout(.5))
        model.train()
        model[0].eval()
        modes = [m.training for m in model.modules()]
        buffers = {k: v.clone() for k, v in model.named_buffers()}
        rng = torch.random.get_rng_state().clone()
        with local_eval_with_grad(model):
            self.assertTrue(all(not m.training for m in model.modules()))
            torch.rand(4)
            output = model(torch.ones(3, 4))
        output.sum().backward()
        self.assertGreater(float(model[0].weight.grad.abs().sum()), 0)
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for k, v in model.named_buffers():
            self.assertTrue(torch.equal(buffers[k], v))
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with local_eval_with_grad(model):
                model[1].running_mean.add_(3)
                torch.rand(2)
                raise RuntimeError("intentional")
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for k, v in model.named_buffers():
            self.assertTrue(torch.equal(buffers[k], v))

    def test_same_weights_eval_retention_zero_then_drift_restorative_gradient(self):
        student = TinyDescriptor().train()
        teacher = copy.deepcopy(student).eval()
        images = torch.randn(5, 3, 2, 2)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        metrics = backward_real_retention(student, teacher, images, scaler, .7, chunk_size=2)
        self.assertEqual(metrics["images"], 5)
        self.assertEqual(metrics["chunks"], 3)
        self.assertEqual(metrics["loss"], 0.)
        self.assertEqual(float(student.linear.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertTrue(student.training)
        with torch.no_grad():
            student.linear.weight.add_(torch.randn_like(student.linear.weight) * .2)
        student.zero_grad(set_to_none=True)
        metrics = backward_real_retention(student, teacher, images, scaler, .7, chunk_size=2)
        self.assertGreater(metrics["loss"], 0.)
        self.assertGreater(float(student.linear.weight.grad.abs().sum()), 0.)
        # A small gradient step should move the same input descriptors toward
        # the teacher. This checks a restorative effect, not merely a gradient.
        with torch.no_grad():
            for p in student.parameters():
                p.add_(p.grad, alpha=-.01)
        after = backward_real_retention(student, teacher, images, scaler, .7, chunk_size=2)
        self.assertLess(after["loss"], metrics["loss"])

    def test_streaming_reduction_and_accumulation_match_full_batch(self):
        student = TinyDescriptor().train()
        teacher = TinyDescriptor().eval()
        expected_model = copy.deepcopy(student)
        images = torch.randn(5, 3, 2, 2)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        # Simulate an existing main backward; the helper must accumulate.
        sum(p.sum() * .03 for p in student.parameters()).backward()
        sum(p.sum() * .03 for p in expected_model.parameters()).backward()
        with local_eval_with_grad(expected_model):
            expected = descriptor_retention_loss(expected_model(images), teacher(images))
            (.4 * expected).backward()
        metrics = backward_real_retention(student, teacher, images, scaler, .4, chunk_size=2)
        self.assertAlmostEqual(metrics["loss"], float(expected.detach()), places=6)
        self.assertAlmostEqual(metrics["weighted_total"], .4 * float(expected.detach()), places=6)
        for actual, wanted in zip(student.parameters(), expected_model.parameters()):
            torch.testing.assert_close(actual.grad, wanted.grad, atol=2e-7, rtol=2e-6)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_disabled_retention_has_no_forward_and_bad_settings_rejected(self):
        model, teacher = TinyDescriptor(), TinyDescriptor()
        images = torch.randn(3, 3, 2, 2)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        self.assertEqual(backward_real_retention(model, teacher, images, scaler, 0)["chunks"], 0)
        self.assertEqual(model.calls + teacher.calls, 0)
        for weight in (-1, float("nan")):
            with self.assertRaises(ValueError):
                backward_real_retention(model, teacher, images, scaler, weight)
        with self.assertRaises(ValueError):
            backward_real_retention(model, teacher, images, scaler, 1, chunk_size=0)
        with torch.no_grad():
            teacher.linear.weight.zero_()
            teacher.linear.bias.zero_()
        with self.assertRaisesRegex(RuntimeError, "Zero teacher descriptor"):
            backward_real_retention(model, teacher, images, scaler, 1)


if __name__ == "__main__":
    unittest.main()
