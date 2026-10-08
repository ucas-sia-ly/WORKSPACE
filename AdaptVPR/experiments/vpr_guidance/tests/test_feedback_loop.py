"""CPU tests for the SALAD-feedback scoring, selection, and LoRA mechanics."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

GUIDANCE_ROOT = Path(__file__).resolve().parents[1]
if str(GUIDANCE_ROOT) not in sys.path:
    sys.path.insert(0, str(GUIDANCE_ROOT))

from feedback import (  # noqa: E402
    expected_hardest_negative,
    ms_positive_utility,
    score_candidates,
    select_per_group,
)
from lora_utils import (  # noqa: E402
    freeze_non_lora_parameters,
    get_lora_parameters,
    inject_lora_into_unet,
    load_lora_checkpoint,
    report_trainable_parameters,
    save_lora_checkpoint,
    unfreeze_lora_parameters,
)

SALAD_ROOT = GUIDANCE_ROOT.parents[2] / "salad"


def unit(*values):
    return F.normalize(torch.tensor(values, dtype=torch.float32), dim=-1)


class UtilityTests(unittest.TestCase):
    def test_easy_positive_is_not_mined(self):
        utility, mined = ms_positive_utility(torch.tensor([[0.9]]), torch.tensor([0.3]))
        self.assertEqual(float(utility), 0.0)
        self.assertEqual(int(mined), 0)

    def test_hard_positive_is_mined_and_harder_is_worth_more(self):
        sims = torch.tensor([[0.35], [0.1]])
        utility, mined = ms_positive_utility(sims, torch.tensor([0.3, 0.3]))
        self.assertTrue(bool((mined == 1).all()))
        self.assertGreater(float(utility[1]), float(utility[0]))
        self.assertGreater(float(utility[0]), 0.0)

    def test_matches_salad_metric_loss_positive_term(self):
        """Positive utility plus the negative term matches SALAD's actual loss."""
        try:
            sys.path.insert(0, str(SALAD_ROOT))
            from workflow import metric_loss
        except ImportError:
            self.skipTest("SALAD checkout not available")
        finally:
            sys.path.remove(str(SALAD_ROOT))
        torch.manual_seed(0)
        descriptors = F.normalize(torch.randn(6, 8), dim=1)
        labels = torch.tensor([0, 0, 0, 1, 2, 3])
        similarity = descriptors @ descriptors.T
        hardest = similarity[0, 3:].max().reshape(1)
        utility, _ = ms_positive_utility(similarity[0:1, 1:3], hardest)
        # Recompute the reference anchor-0 positive term with the official mining rule.
        with torch.no_grad():
            mined = similarity[0, 1:3] - 0.1 < hardest
        reference = torch.logsumexp(torch.cat((torch.zeros(1), -similarity[0, 1:3][mined])), 0)
        self.assertAlmostEqual(float(utility), float(reference), places=6)
        same = labels[:, None] == labels[None, :]
        positives = same & ~torch.eye(len(labels), dtype=torch.bool)
        negatives = ~same
        hardest_negatives = similarity.masked_fill(~negatives, -torch.inf).max(dim=1).values
        positive_utility, _ = ms_positive_utility(similarity, hardest_negatives, positives)
        hardest_positives = similarity.masked_fill(~positives, torch.inf).min(dim=1).values
        negative_mask = negatives & (similarity + 0.1 > hardest_positives[:, None])
        negative_terms = (50 * similarity).masked_fill(~negative_mask, -torch.inf)
        negative_utility = torch.logsumexp(torch.cat((torch.zeros(len(labels), 1), negative_terms), dim=1), dim=1) / 50
        expected_loss = (positive_utility + negative_utility).mean()
        actual_loss = metric_loss.multi_similarity_loss(descriptors, labels)
        self.assertAlmostEqual(float(actual_loss), float(expected_loss), places=6)

    def test_hardest_negative_ignores_own_place(self):
        candidates = unit(1.0, 0.0)[None]
        pool = torch.stack([unit(1.0, 0.0), unit(0.0, 1.0), unit(0.0, 1.0)])
        same = torch.tensor([[True, False, False]])
        hardest = expected_hardest_negative(candidates, pool, same, negatives_per_batch=2, draws=4)
        self.assertAlmostEqual(float(hardest), 0.0, places=6)

    def test_negative_set_must_cover_a_batch(self):
        with self.assertRaises(ValueError):
            expected_hardest_negative(unit(1.0, 0.0)[None], unit(0.0, 1.0)[None], torch.tensor([[False]]),
                                      negatives_per_batch=2)

    def test_score_candidates_ranks_off_place_candidate_higher(self):
        place = torch.stack([unit(1.0, 0.0, 0.0), unit(0.95, 0.05, 0.0)])
        pool = torch.stack([unit(0.0, 1.0, 0.0), unit(0.0, 0.0, 1.0), unit(0.0, 0.7, 0.7)])
        candidates = torch.stack([unit(0.98, 0.02, 0.0), unit(0.4, 0.9, 0.0)])
        same = torch.zeros(2, 3, dtype=torch.bool)
        easy, hard = score_candidates(candidates, [place, place], pool, same, negatives_per_batch=3)
        self.assertEqual(easy["utility"], 0.0)
        self.assertGreater(hard["utility"], 0.0)
        self.assertLess(hard["mean_positive_similarity"], easy["mean_positive_similarity"])


class SelectionTests(unittest.TestCase):
    rows = [
        {"sample_id": "a", "candidate_index": 0, "passed": True, "utility": 0.2, "mean_positive_similarity": 0.5},
        {"sample_id": "a", "candidate_index": 1, "passed": True, "utility": 0.7, "mean_positive_similarity": 0.3},
        {"sample_id": "a", "candidate_index": 2, "passed": False, "utility": 9.0, "mean_positive_similarity": 0.0},
        {"sample_id": "b", "candidate_index": 0, "passed": True, "utility": 0.0, "mean_positive_similarity": 0.8},
        {"sample_id": "b", "candidate_index": 1, "passed": True, "utility": 0.0, "mean_positive_similarity": 0.6},
        {"sample_id": "c", "candidate_index": 0, "passed": False, "utility": 1.0, "mean_positive_similarity": 0.1},
    ]

    def test_hardness_takes_max_verified_utility_then_lowest_similarity(self):
        chosen = {r["sample_id"]: r["candidate_index"] for r in select_per_group(self.rows, "hardness")}
        self.assertEqual(chosen, {"a": 1, "b": 1})

    def test_random_is_matched_and_deterministic(self):
        first = select_per_group(self.rows, "random", seed=3)
        second = select_per_group(self.rows, "random", seed=3)
        self.assertEqual(first, second)
        self.assertEqual({r["sample_id"] for r in first}, {"a", "b"})
        self.assertTrue(all(r["passed"] for r in first))


class TinyIcLightUNet(torch.nn.Module):
    """8-channel input via concat_conds, like the adapter's hooked IC-Light UNet."""

    def __init__(self):
        super().__init__()
        self.conv_in = torch.nn.Conv2d(8, 16, 3, padding=1)
        self.to_q = torch.nn.Linear(16, 16)
        self.to_k = torch.nn.Linear(16, 16)
        self.to_v = torch.nn.Linear(16, 16)
        self.to_out = torch.nn.Sequential(torch.nn.Linear(16, 16))
        self.conv_out = torch.nn.Conv2d(16, 4, 3, padding=1)

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs):
        features = self.conv_in(torch.cat([sample, cross_attention_kwargs["concat_conds"]], dim=1))
        b, c, h, w = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        attended = F.scaled_dot_product_attention(self.to_q(tokens), self.to_k(tokens), self.to_v(tokens))
        tokens = tokens + self.to_out(attended)
        return SimpleNamespace(sample=self.conv_out(tokens.transpose(1, 2).reshape(b, c, h, w)))


class LoRATests(unittest.TestCase):
    def test_lora_is_registered_and_keeps_fp32_on_half_base(self):
        unet = TinyIcLightUNet().to(torch.bfloat16)
        freeze_non_lora_parameters(unet)
        layers = inject_lora_into_unet(unet, rank=2, dtype=torch.float32)
        unfreeze_lora_parameters(layers)
        self.assertEqual(set(layers), {"to_q", "to_k", "to_v", "to_out.0"})
        ids = {id(p) for p in unet.parameters()}
        self.assertTrue(all(id(p) in ids for p in get_lora_parameters(layers)))
        self.assertTrue(all(p.dtype == torch.float32 for p in get_lora_parameters(layers)))
        report_trainable_parameters(unet, layers)
        out = unet.to_q(torch.randn(3, 16, dtype=torch.bfloat16))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_double_injection_is_rejected(self):
        unet = TinyIcLightUNet()
        inject_lora_into_unet(unet, rank=2)
        with self.assertRaises(RuntimeError):
            inject_lora_into_unet(unet, rank=2)

    def test_denoising_step_updates_only_lora(self):
        torch.manual_seed(0)
        unet = TinyIcLightUNet()
        freeze_non_lora_parameters(unet)
        layers = inject_lora_into_unet(unet, rank=2, dtype=torch.float32)
        unfreeze_lora_parameters(layers)
        params = get_lora_parameters(layers)
        lora_ids = {id(p) for p in params}
        base_before = {n: p.detach().clone() for n, p in unet.named_parameters() if id(p) not in lora_ids}
        lora_before = [p.detach().clone() for p in params]
        optimizer = torch.optim.AdamW(params, lr=1e-2)
        latent, cond = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8)
        noise = torch.randn_like(latent)
        abar = torch.tensor([0.5, 0.9]).view(-1, 1, 1, 1)
        noisy = abar.sqrt() * latent + (1 - abar).sqrt() * noise
        for _ in range(2):  # B starts at zero, so A only moves from the second step.
            pred = unet(noisy, None, None, cross_attention_kwargs={"concat_conds": cond}).sample
            loss = F.mse_loss(pred, noise)
            optimizer.zero_grad()
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params))
            optimizer.step()
        self.assertTrue(all(not torch.equal(b, p) for b, p in zip(lora_before, params)))
        for name, param in unet.named_parameters():
            if id(param) not in lora_ids:
                self.assertIsNone(param.grad, name)
                self.assertTrue(torch.equal(base_before[name], param), name)

    def test_checkpoint_round_trip_uses_adapter_metadata(self):
        source = TinyIcLightUNet()
        layers = inject_lora_into_unet(source, rank=2, alpha=4.0)
        with torch.no_grad():
            for p in get_lora_parameters(layers):
                p.normal_()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lora.safetensors"
            save_lora_checkpoint(layers, path)
            target = TinyIcLightUNet()
            target_layers = inject_lora_into_unet(target, rank=2, alpha=4.0)
            metadata = load_lora_checkpoint(target_layers, path)
        self.assertEqual((metadata["lora_rank"], metadata["lora_alpha"]), ("2", "4.0"))
        for name in layers:
            self.assertTrue(torch.equal(layers[name].lora_B, target_layers[name].lora_B))

    def test_strict_load_rejects_mismatched_layers(self):
        layers = inject_lora_into_unet(TinyIcLightUNet(), rank=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lora.safetensors"
            save_lora_checkpoint({k: v for k, v in layers.items() if k != "to_q"}, path)
            with self.assertRaises(KeyError):
                load_lora_checkpoint(inject_lora_into_unet(TinyIcLightUNet(), rank=2), path)


if __name__ == "__main__":
    unittest.main()


class PlausibilityTests(unittest.TestCase):
    def test_floor_is_quantile_of_real_margins(self):
        from feedback import identity_margin, plausibility_floor
        scores = [{"mean_positive_similarity": m + 0.1, "expected_hardest_negative": 0.1}
                  for m in (0.0, 0.1, 0.2, 0.3, 0.4)]
        self.assertAlmostEqual(identity_margin(scores[2]), 0.2)
        self.assertAlmostEqual(plausibility_floor(scores, 0.25), 0.1)
        with self.assertRaises(ValueError):
            plausibility_floor(scores, 0.0)

    def test_implausible_rows_are_excluded_for_every_method(self):
        rows = [
            {"sample_id": "a", "candidate_index": 0, "passed": True, "plausible": False,
             "utility": 1.2, "mean_positive_similarity": 0.1},
            {"sample_id": "a", "candidate_index": 1, "passed": True, "plausible": True,
             "utility": 0.3, "mean_positive_similarity": 0.3},
            {"sample_id": "b", "candidate_index": 0, "passed": True, "plausible": False,
             "utility": 0.9, "mean_positive_similarity": 0.1},
        ]
        for method in ("hardness", "random"):
            chosen = select_per_group(rows, method)
            self.assertEqual([(r["sample_id"], r["candidate_index"]) for r in chosen], [("a", 1)])
