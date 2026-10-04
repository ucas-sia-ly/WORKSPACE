"""CPU checks for gradient flow, truncation, and leakage-free retrieval."""
from pathlib import Path
import unittest
import torch
from diffusers import DDIMScheduler
from AdaptVPR.experiments.vpr_guidance.ddim import Guidance, denoise, normalized_gradient_step
from AdaptVPR.experiments.vpr_guidance.gsv_pairs import parse_filename, positive_indices
from AdaptVPR.experiments.vpr_guidance.generate import read_prompts, released_global_negative_prompt
from AdaptVPR.experiments.vpr_guidance.vpr import Descriptor, enable_salad_image_gradients, freeze
from AdaptVPR.experiments.vpr_guidance.evaluate_retrieval import retrieval_metrics, success_check

A = 'Bangkok_0000002_2017_05_577_13.715_100.485_pano_with_underscores.jpg'
B = 'Bangkok_0000002_2020_09_394_13.715_100.485_other.jpg'
C = 'Bangkok_0000003_2020_09_394_13.715_100.485_distractor.jpg'
D = 'London_0000002_2020_09_394_51.715_-0.485_other.jpg'


class TokenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(3, 3) for _ in range(3)])
        self.norm = torch.nn.LayerNorm(3)

    def prepare_tokens_with_masks(self, x):
        tokens = x.mean((2, 3))[:, None]
        return tokens.repeat(1, 5, 1)  # class + four 14x14 patches


class DetachedBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TokenModel()
        self.norm_layer = True
        self.return_token = True
        self.num_channels = 3

    def forward(self, image):
        tokens = self.model.prepare_tokens_with_masks(image)
        with torch.no_grad():
            for block in self.model.blocks[:-1]:
                tokens = block(tokens)
        tokens = tokens.detach()
        tokens = self.model.norm(self.model.blocks[-1](tokens))
        return tokens[:, 1:].reshape(-1, 2, 2, 3).permute(0, 3, 1, 2), tokens[:, 0]


class Tests(unittest.TestCase):
    def test_place_and_capture_parsing(self):
        self.assertEqual(parse_filename(A).place_key, parse_filename(B).place_key)
        self.assertEqual(parse_filename(A).panorama_id, 'pano_with_underscores')
        self.assertEqual(positive_indices(A, [A, B, C, D]), [1])
        self.assertEqual(parse_filename('BuenosAires_0000066_2014_05_-02_-34.62_-58.38_pano.jpg').heading, -2)
        with self.assertRaises(ValueError):
            parse_filename('not_gsv.jpg')

    def test_demo_has_only_two_global_snow_samples(self):
        demo = Path(__file__).resolve().parents[3] / 'tests/demo_10_prompts.jsonl'
        rows = read_prompts(demo, ['snow'])
        self.assertEqual([r['sample_id'] for r in rows], ['adapt_000066', 'adapt_000071'])
        self.assertIn('changed camera viewpoint', released_global_negative_prompt())

    def test_salad_patch_preserves_values_and_restores_image_gradient(self):
        model = torch.nn.Module()
        model.backbone = DetachedBackbone()
        freeze(model)
        image = torch.rand(1, 3, 28, 28, requires_grad=True)
        expected_features, expected_token = model.backbone(image)
        self.assertFalse(expected_features.requires_grad)
        enable_salad_image_gradients(model)
        features, token = model.backbone(image)
        torch.testing.assert_close(features, expected_features, rtol=0, atol=0)
        torch.testing.assert_close(token, expected_token, rtol=0, atol=0)
        grad, = torch.autograd.grad(features.square().mean() + token[:, 0].mean(), image)
        self.assertGreater(grad.abs().sum(), 0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))

    def test_descriptor_preprocessing_keeps_gradients(self):
        class MeanModel(torch.nn.Module):
            def forward(self, image):
                return image.mean((2, 3))
        descriptor = Descriptor(MeanModel(), device='cpu')
        image = torch.rand(1, 3, 32, 40, requires_grad=True)
        desc = descriptor(image)
        torch.testing.assert_close(desc.norm(dim=-1), torch.ones(1))
        grad, = torch.autograd.grad(desc[:, 0].sum(), image)
        self.assertGreater(grad.abs().sum(), 0)

    def test_guidance_schedule_and_normalization(self):
        self.assertEqual([i + 1 for i in range(44) if Guidance(.03, 5, 10).selected(i, 44)], [35, 40])
        self.assertFalse(Guidance(0).selected(4, 44))
        x = torch.zeros(1, 4, 2, 2)
        updated, _ = normalized_gradient_step(x, torch.ones_like(x) * 7, .03)
        self.assertAlmostEqual(float(updated.square().mean().sqrt()), .03, places=6)
        with self.assertRaises(FloatingPointError):
            normalized_gradient_step(x, torch.full_like(x, float('nan')), .03)

    def test_explicit_ddim_gradient_and_zero_scale_baseline(self):
        torch.manual_seed(42)
        scheduler = DDIMScheduler(clip_sample=False, prediction_type='epsilon')
        scheduler.set_timesteps(5)
        unet = freeze(torch.nn.Conv2d(3, 3, 1))
        vae = freeze(torch.nn.Conv2d(3, 3, 1))
        class MeanModel(torch.nn.Module):
            def forward(self, image):
                return image.mean((2, 3))
        descriptor = Descriptor(MeanModel(), device='cpu')
        source = torch.nn.functional.normalize(torch.tensor([[1., .2, .4]]), dim=-1)
        inputs = []
        def predict(x, t):
            inputs.append((x.is_leaf, x.requires_grad))
            return unet(x) * .01
        predict.dtype = torch.float32
        decode = lambda x: vae(x).sigmoid()
        initial = torch.randn(1, 3, 4, 4)
        with torch.no_grad():
            reference = initial.clone()
            for t in scheduler.timesteps:
                reference = scheduler.step(predict(reference, t), t, reference, eta=0).prev_sample
        baseline, empty = denoise(initial, scheduler.timesteps, scheduler, predict, decode,
                                  descriptor, source, Guidance(0))
        torch.testing.assert_close(baseline, reference, rtol=0, atol=0)
        self.assertEqual(empty, [])
        inputs.clear()
        # Also works when caller is inside no_grad: selected steps explicitly enable grads.
        with torch.no_grad():
            guided, trace = denoise(initial, scheduler.timesteps, scheduler, predict, decode,
                                    descriptor, source, Guidance(.03, 1))
        self.assertEqual(len(trace), 5)
        self.assertTrue(all(leaf for leaf, enabled in inputs if enabled))
        self.assertFalse(guided.requires_grad)
        self.assertGreater((guided - baseline).abs().sum(), 0)
        self.assertTrue(all(p.grad is None for m in (unet, vae) for p in m.parameters()))
        baseline_loss = 1 - (descriptor(decode(baseline)) * source).sum()
        guided_loss = 1 - (descriptor(decode(guided)) * source).sum()
        self.assertLess(guided_loss, baseline_loss)

    def test_retrieval_excludes_source_from_ranking_and_positives(self):
        # The exact source would rank first; after exclusion alternate capture ranks second.
        database = torch.tensor([[1., 0.], [.6, .8], [.8, .6]])
        metrics = retrieval_metrics(torch.tensor([[1., 0.]]), database, [A], [A, B, C])
        self.assertEqual(metrics['recall_at_1'], 0)
        self.assertEqual(metrics['recall_at_5'], 1)
        self.assertEqual(metrics['median_rank'], 2)
        self.assertAlmostEqual(metrics['mean_positive_cosine_similarity'], .6, places=6)
        missing = retrieval_metrics(torch.tensor([[1., 0.]]), database[[0, 2]], [A], [A, C])
        self.assertEqual(missing['num_queries'], 0)
        self.assertIsNone(missing['recall_at_1'])

    def test_success_requires_heldout_and_geometry(self):
        base = {'num_queries': 2, 'recall_at_1': .5, 'recall_at_5': 1.,
                'median_rank': 1.5, 'mean_positive_cosine_similarity': .7}
        guided = dict(base, recall_at_1=1.)
        report = {'salad': {'baseline': base, 'guided': guided}}
        geo = {'num_pairs': 2, 'baseline_mean': .4, 'guided_mean': .4}
        self.assertEqual(success_check(report, geo)['status'], 'insufficient_evidence')
        report['boq'] = {'baseline': base, 'guided': guided}
        self.assertEqual(success_check(report, geo)['status'], 'pass')
        report['boq']['guided'] = dict(guided, mean_positive_cosine_similarity=.6)
        self.assertEqual(success_check(report, geo)['status'], 'fail')


if __name__ == '__main__':
    unittest.main()
