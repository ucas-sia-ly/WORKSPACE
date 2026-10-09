"""Small CPU tests for optional reliability-conditioned SALAD transport."""

import math
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
import torch.nn.functional as F

SALAD_ROOT = Path(__file__).resolve().parents[1]
if str(SALAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SALAD_ROOT))

from models.aggregators.salad import SALAD, get_matching_probs, log_otp_solver


def legacy_transport(scores, dustbin):
    """Frozen pre-change computation: verifies arithmetic, not only shape."""
    b, m, n = scores.shape
    matrix = torch.empty(b, m + 1, n, dtype=scores.dtype, device=scores.device)
    matrix[:, :m, :] = scores
    matrix[:, m, :] = dustbin
    norm = -torch.tensor(math.log(n + m), device=scores.device)
    log_a, log_b = norm.expand(m + 1).contiguous(), norm.expand(n).contiguous()
    log_a[-1] += math.log(n - m)
    log_a, log_b = log_a.expand(b, -1), log_b.expand(b, -1)
    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)
    for _ in range(3):
        u = log_a - torch.logsumexp(matrix + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(matrix + u.unsqueeze(2), dim=1).squeeze()
    return matrix + u.unsqueeze(2) + v.unsqueeze(1) - norm


def legacy_descriptor(model, features, token):
    local = model.cluster_features(features).flatten(2)
    scores = model.score(features).flatten(2)
    token = model.token_features(token)
    assignment = legacy_transport(scores, model.dust_bin).exp()[:, :-1, :]
    weights = assignment.unsqueeze(1).repeat(1, model.cluster_dim, 1, 1)
    local = local.unsqueeze(2).repeat(1, 1, model.num_clusters, 1)
    return F.normalize(torch.cat([
        F.normalize(token, dim=-1),
        F.normalize((local * weights).sum(-1), dim=1).flatten(1),
    ], dim=-1), dim=-1)


class ReliabilityOTTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.manual_seed(17)

    def model(self, **kwargs):
        kwargs.setdefault("reliability_hidden_dim", 8)
        return SALAD(num_channels=12, num_clusters=4, cluster_dim=6,
                     token_dim=8, dropout=0, **kwargs)

    def inputs(self, batch=2, dtype=torch.float32):
        return (torch.randn(batch, 12, 4, 4, dtype=dtype),
                torch.randn(batch, 12, dtype=dtype))

    def test_disabled_state_keys_are_exactly_legacy(self):
        model = self.model()
        expected = {"dust_bin"}
        for module in ("token_features", "cluster_features", "score"):
            for layer in (0, 2 if module == "token_features" else 3):
                expected.update({f"{module}.{layer}.weight", f"{module}.{layer}.bias"})
        self.assertEqual(set(model.state_dict()), expected)
        self.assertFalse(hasattr(model, "reliability_head"))
        restored = self.model()
        restored.load_state_dict(model.state_dict(), strict=True)
        x = self.inputs()
        self.assertTrue(torch.equal(model(x), restored(x)))

    def test_disabled_forward_is_bitwise_legacy(self):
        model = self.model().eval()
        x, token = self.inputs()
        expected = legacy_descriptor(model, x, token)
        actual = model((x, token))
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.shape, (2, 32))

    def test_fresh_enabled_and_disabled_share_original_initialization(self):
        torch.manual_seed(731)
        original = self.model().eval()
        torch.manual_seed(731)
        enabled = self.model(reliability_ot=True).eval()
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, enabled.state_dict()[key]), key)
        inputs = self.inputs()
        torch.testing.assert_close(enabled(inputs), original(inputs), atol=3e-7, rtol=3e-6)

    def test_scalar_transport_is_bitwise_legacy(self):
        scores = torch.randn(2, 4, 16)
        dust = torch.tensor(1.0)
        self.assertTrue(torch.equal(get_matching_probs(scores, dust), legacy_transport(scores, dust)))

    def test_single_image_has_no_batch_broadcast(self):
        for enabled in (False, True):
            model = self.model(reliability_ot=enabled).eval()
            x = self.inputs(1)
            result = model(x)
            batch_result = model(tuple(item.repeat(2, *([1] * (item.ndim - 1))) for item in x))
            self.assertEqual(result.shape, (1, 32))
            torch.testing.assert_close(result, batch_result[:1], atol=2e-7, rtol=2e-6)
            self.assertTrue(torch.isfinite(result).all())

    def test_head_initializes_high_and_returns_aux_without_cache(self):
        model = self.model(reliability_ot=True)
        x = self.inputs()
        descriptor, aux = model(x, return_aux=True)
        self.assertEqual(aux["local_features"].shape, x[0].shape)
        self.assertIs(aux["local_features"], x[0])
        self.assertEqual(aux["reliability_logits"].shape, (2, 1, 4, 4))
        torch.testing.assert_close(aux["reliability"], torch.full((2, 1, 4, 4), .9))
        self.assertEqual(aux["assignment_dustbin"].shape, (2, 16))
        torch.testing.assert_close(aux["assignment_keep"] + aux["assignment_dustbin"],
                                   torch.ones(2, 16), atol=2e-6, rtol=1e-5)
        self.assertIsInstance(model(x), torch.Tensor)
        self.assertEqual(descriptor.shape, (2, 32))
        self.assertFalse(any(isinstance(value, torch.Tensor)
                             for value in vars(model).values()))

    def test_zero_lambda_and_high_trust_warm_start_match_legacy(self):
        base = self.model().eval()
        x = self.inputs()
        expected = base(x)
        for strength in (0., 2.):
            enabled = self.model(reliability_ot=True, reliability_lambda=strength).eval()
            incompatible = enabled.load_state_dict(base.state_dict(), strict=False)
            self.assertEqual(set(incompatible.missing_keys), {
                "reliability_ot_lambda",
                "reliability_head.0.weight", "reliability_head.0.bias",
                "reliability_head.2.weight", "reliability_head.2.bias",
            })
            self.assertEqual(incompatible.unexpected_keys, [])
            torch.testing.assert_close(enabled(x), expected, atol=3e-7, rtol=3e-6)

    def test_enabled_state_dict_strict_round_trip(self):
        model = self.model(reliability_ot=True, reliability_lambda=.7)
        restored = self.model(reliability_ot=True)
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertAlmostEqual(restored.reliability_lambda, .7, places=6)
        x = self.inputs()
        self.assertTrue(torch.equal(model(x), restored(x)))

    def test_raw_checkpoint_rejects_invalid_strength(self):
        model = self.model(reliability_ot=True)
        for strength in (-1., float("nan"), float("inf")):
            state = dict(model.state_dict())
            state["reliability_ot_lambda"] = torch.tensor(strength)
            with self.assertRaisesRegex(RuntimeError, "finite nonnegative scalar"):
                model.load_state_dict(state, strict=True)

    def test_low_reliability_steers_the_dustbin_for_that_patch(self):
        scores = torch.zeros(1, 3, 12)
        reliability = torch.ones(1, 12)
        reliability[:, 0] = 0.
        dustbin = 1. + 4. * (1. - reliability)
        assignment = get_matching_probs(scores, dustbin, num_iters=100).exp()
        self.assertGreater(assignment[0, -1, 0], assignment[0, -1, 1])
        torch.testing.assert_close(assignment.sum(dim=1), torch.ones(1, 12))
        torch.testing.assert_close(assignment.sum(dim=2), torch.tensor([[1., 1., 1., 9.]]),
                                   atol=2e-5, rtol=1e-5)

    def test_constant_dustbin_offset_is_a_gauge_not_a_collapse_switch(self):
        scores = torch.randn(2, 4, 16)
        base = get_matching_probs(scores, torch.tensor(1.)).exp()
        all_low = get_matching_probs(scores, torch.full((2, 16), 5.)).exp()
        torch.testing.assert_close(base, all_low, atol=1e-7, rtol=2e-6)

    def test_transport_and_descriptor_gradients_reach_head_and_backbone_features(self):
        model = self.model(reliability_ot=True)
        # The final head is zero-initialized; after one update gradients also
        # reach its first convolution. Simulate that nonconstant state here.
        nn.init.normal_(model.reliability_head[-1].weight, std=.03)
        x, token = self.inputs()
        x.requires_grad_()
        token.requires_grad_()
        descriptor, aux = model((x, token), return_aux=True)
        weights = torch.randn_like(descriptor)
        loss = (descriptor * weights).sum() + .01 * aux["reliability_logits"].square().mean()
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(model.reliability_head[0].weight.grad.abs().sum(), 0.)
        self.assertGreater(model.reliability_head[-1].weight.grad.abs().sum(), 0.)
        self.assertGreater(x.grad.abs().sum(), 0.)
        self.assertGreater(token.grad.abs().sum(), 0.)

    def test_metric_gradient_can_train_constant_initialized_head(self):
        model = self.model(reliability_ot=True)
        descriptor = model(self.inputs())
        (descriptor * torch.randn_like(descriptor)).sum().backward()
        final_gradient = model.reliability_head[-1].weight.grad
        self.assertTrue(torch.isfinite(final_gradient).all())
        self.assertGreater(final_gradient.abs().sum(), 0.)

    def test_per_patch_transport_passes_double_precision_gradcheck(self):
        scores = torch.randn(1, 2, 5, dtype=torch.float64, requires_grad=True)
        dustbin = torch.randn(1, 5, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda s, z: get_matching_probs(s, z, num_iters=5).exp(),
            (scores, dustbin), eps=1e-6, atol=1e-5, rtol=1e-3,
        ))

    def test_log_transport_extreme_scores_and_small_regularization_are_finite(self):
        scores = (torch.randn(1, 4, 16) * 1000).requires_grad_()
        dustbin = torch.randn(1, 16, requires_grad=True)
        log_assignment = get_matching_probs(scores, dustbin, num_iters=20, reg=.01)
        self.assertTrue(torch.isfinite(log_assignment).all())
        log_assignment.exp().square().mean().backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertTrue(torch.isfinite(dustbin.grad).all())

    def test_enabled_half_precision_uses_float32_transport_and_finite_gradients(self):
        for dtype in (torch.float16, torch.bfloat16):
            # Some CPU DNNL builds support half forward but not backward.
            # Use native convolution so this tests transport, not CPU ISA.
            with self.subTest(dtype=dtype), torch.backends.mkldnn.flags(enabled=False):
                model = self.model(reliability_ot=True).to(dtype=dtype)
                x = self.inputs(dtype=dtype)
                descriptor, aux = model(x, return_aux=True)
                self.assertEqual(aux["assignment_dustbin"].dtype, torch.float32)
                self.assertEqual(descriptor.dtype, torch.float32)
                self.assertTrue(torch.isfinite(descriptor).all())
                descriptor.square().sum().add(aux["reliability_logits"].mean()).backward()
                for parameter in model.parameters():
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_enabled_amp_uses_float32_transport(self):
        model = self.model(reliability_ot=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            descriptor, aux = model(self.inputs(), return_aux=True)
        self.assertEqual(aux["assignment_dustbin"].dtype, torch.float32)
        self.assertEqual(descriptor.dtype, torch.float32)
        self.assertTrue(torch.isfinite(descriptor).all())

    def test_invalid_ot_inputs_fail_explicitly(self):
        for strength in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.model(reliability_ot=True, reliability_lambda=strength)
        for hidden in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                self.model(reliability_ot=True, reliability_hidden_dim=hidden)
        with self.assertRaises(ValueError):
            get_matching_probs(torch.zeros(1, 4, 4))
        for reg in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                get_matching_probs(torch.zeros(1, 4, 8), reg=reg)
        with self.assertRaises(ValueError):
            get_matching_probs(torch.zeros(1, 4, 8), num_iters=0)


if __name__ == "__main__":
    unittest.main()
