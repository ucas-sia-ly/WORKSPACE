"""CPU-only second-order graph, GSV episode, factory and optimizer regressions."""
import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F
from torch.func import functional_call

from AdaptVPR.experiments.vpr_guidance.bilevel import (
    GSVRealIndex, construct_episode, MetaEpisode, EpisodeUnavailable, bilevel_objective,
)
from AdaptVPR.experiments.vpr_guidance.salad_factory import (
    make_metric_learning, build_fresh_salad, fresh_salad_config, preprocess_tensor,
    validate_meta_args, OBJECTIVE_VERSION, metric_loss,
)
from AdaptVPR.experiments.vpr_guidance.train_generator import (
    assert_lora_optimizer, train_bilevel_step, differentiable_accepted_prediction,
)
from AdaptVPR.experiments.vpr_guidance.train_online_generator import validate_bilevel_checkpoint
from AdaptVPR.experiments.vpr_guidance.tests.test_data_pipeline import make_mixed_fixture
from AdaptVPR.experiments.vpr_guidance.teacher import file_sha256, model_sha256


class TinyEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(3, 5)
        self.loss_fn, self.miner = make_metric_learning()

    def forward(self, x):
        if x.ndim == 4:
            x = x.mean((-1, -2))
        return F.normalize(torch.tanh(self.embed(x)), dim=-1)


class TinyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_phi = nn.Parameter(torch.tensor([.11, -.23, .37]))

    def forward(self, x):
        return x + self.lora_phi


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(4, 12)
        self.proj = nn.Linear(4, 4)
        self.proj_drop = nn.Identity()
        self.num_heads, self.scale, self.attn_drop = 1, .5, 0.

    def forward(self, x):
        raise AssertionError("meta should install eager attention")


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = TinyAttention()

    def forward(self, x):
        return x + self.attn(x)


class TinyDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 4, 14, stride=14)
        self.blocks = nn.ModuleList([TinyBlock() for _ in range(6)])
        self.norm = nn.LayerNorm(4)

    def prepare_tokens_with_masks(self, image):
        patches = self.patch(image).flatten(2).transpose(1, 2)
        return torch.cat([patches.mean(1, keepdim=True), patches], dim=1)


class TinyAgg(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 5)

    def forward(self, pair):
        features, token = pair
        return F.normalize(self.proj(torch.cat([features.mean((-1, -2)), token], -1)), dim=-1)


class TinyFreshSALAD(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.config = config
        self.backbone = nn.Module()
        self.backbone.model = TinyDino()
        self.backbone.num_trainable_blocks = config['backbone_config']['num_trainable_blocks']
        self.backbone.norm_layer = self.backbone.return_token = True
        self.backbone.num_channels = 4
        self.aggregator = TinyAgg()
        self.loss_fn, self.miner = make_metric_learning()

    def forward(self, x):
        return self.aggregator(self.backbone(x))


def meta_args(**overrides):
    return SimpleNamespace(**{**dict(meta_inner_lr=.1, meta_inner_steps=1, meta_places=2,
        meta_support_real_per_place=1, meta_query_real_per_place=2, meta_image_size=126,
        meta_train_backbone_blocks=0, generator_grad_scale=65536., lambda_meta=1., lambda_vpr=None, lambda_diff=0.,
        lambda_keep=0., grad_clip=10.), **overrides})


class BilevelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(2)

    def inputs(self):
        generator, model = TinyGenerator(), TinyEmbedding().eval()
        support = generator(torch.randn(4, 3))
        query = torch.randn(4, 3)  # independent held-out REAL surrogate
        labels = torch.tensor([0, 0, 1, 1])
        return generator, model, support, query, labels

    def test_bilevel_hypergradient_reaches_generator(self):
        generator, model, support, query, labels = self.inputs()
        snapshot = copy.deepcopy(model.state_dict())
        result = bilevel_objective(model, support, labels, query, labels, inner_lr=.15)
        result.outer_loss.backward()
        self.assertIsNotNone(generator.lora_phi.grad)
        self.assertGreater(float(generator.lora_phi.grad.norm()), 1e-7)
        self.assertTrue(torch.isfinite(generator.lora_phi.grad).all())
        self.assertGreater(result.metrics['meta/mined_inner_pairs'], 0)
        self.assertGreater(result.metrics['meta/mined_outer_pairs'], 0)
        self.assertNotEqual(float(result.inner_loss_before.detach()), float(result.inner_loss_after))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, snapshot[name], rtol=0, atol=0)

    def test_detached_inner_update_breaks_expected_meta_gradient(self):
        generator, model, support, query, labels = self.inputs()
        params, buffers = dict(model.named_parameters()), dict(model.named_buffers())
        inner, _ = metric_loss(functional_call(model, (params, buffers), (support,)),
                               labels, model.loss_fn, model.miner)
        grads = torch.autograd.grad(inner, list(params.values()), create_graph=False)
        fast = {name: p - .15 * g.detach() for (name, p), g in zip(params.items(), grads)}
        outer, _ = metric_loss(functional_call(model, (fast, buffers), (query,)),
                               labels, model.loss_fn, model.miner)
        meta_grad, = torch.autograd.grad(outer, generator.lora_phi, allow_unused=True)
        self.assertIsNone(meta_grad)

    def test_multistep_restarts_fixed_initialization_and_matches_finite_difference(self):
        generator, model, support, query, labels = self.inputs()
        inputs = support - generator.lora_phi
        def objective(phi):
            return bilevel_objective(model, inputs.detach() + phi, labels, query, labels,
                                      inner_lr=.15, inner_steps=2).outer_loss
        loss = objective(generator.lora_phi)
        grad, = torch.autograd.grad(loss, generator.lora_phi)
        delta = .001
        numeric = []
        for i in range(3):
            direction = torch.zeros(3); direction[i] = delta
            numeric.append((objective(generator.lora_phi + direction).item()
                            - objective(generator.lora_phi - direction).item()) / (2 * delta))
        torch.testing.assert_close(grad, torch.tensor(numeric), rtol=.015, atol=3e-4)
        torch.testing.assert_close(objective(generator.lora_phi), loss, rtol=0, atol=0)

    def test_tensor_preprocessing_retains_second_order_input_path(self):
        images = torch.rand(2, 3, 32, 40, requires_grad=True)
        norm = preprocess_tensor(images, 126)
        first, = torch.autograd.grad(norm.square().mean(), images, create_graph=True)
        second, = torch.autograd.grad(first.square().sum(), images)
        self.assertGreater(float(second.norm()), 0)
        self.assertEqual(norm.dtype, torch.float32)

    def test_fresh_salad_factory_matches_downstream_config(self):
        from AdaptVPR.experiments.vpr_guidance.train_salad import build_fresh_salad as downstream
        self.assertIs(downstream, build_fresh_salad)
        meta = build_fresh_salad(model_class=TinyFreshSALAD, meta=True, seed=42, train_backbone_blocks=0)
        final = downstream(model_class=TinyFreshSALAD, seed=42, train_backbone_blocks=4)
        self.assertEqual(meta.config, fresh_salad_config(train_backbone_blocks=0))
        self.assertEqual(meta.config['agg_config'], final.config['agg_config'])
        self.assertEqual(meta.config['backbone_arch'], 'dinov2_vitb14')
        self.assertEqual(model_sha256(meta), model_sha256(final))
        self.assertTrue(all(n.startswith('aggregator.') for n, p in meta.named_parameters() if p.requires_grad))
        self.assertEqual((meta.loss_fn.alpha, meta.loss_fn.beta, meta.loss_fn.base, meta.miner.epsilon),
                         (1., 50., 0., .1))
        self.assertEqual(type(meta.loss_fn.distance).__name__, 'DotProductSimilarity')
        self.assertEqual(type(meta.miner.distance).__name__, 'CosineSimilarity')

    def test_metric_helper_matches_official_salad_factory(self):
        import importlib.util
        path = Path(__file__).resolve().parents[4] / 'salad' / 'utils' / 'losses.py'
        spec = importlib.util.spec_from_file_location('official_salad_losses_test', path)
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        loss, miner = make_metric_learning()
        upstream_loss = official.get_loss('MultiSimilarityLoss')
        upstream_miner = official.get_miner('MultiSimilarityMiner', .1)
        desc = F.normalize(torch.randn(8, 5), dim=-1)
        labels = torch.arange(4).repeat_interleave(2)
        for mine, upstream in zip(miner(desc, labels), upstream_miner(desc, labels)):
            torch.testing.assert_close(mine, upstream, rtol=0, atol=0)
        actual, _ = metric_loss(desc, labels, loss, miner)
        expected = upstream_loss(desc, labels, upstream_miner(desc, labels))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_meta_backbone_policy_keeps_pixel_graph_and_never_silently_falls_back(self):
        for blocks in (0, 1, 4):
            model = build_fresh_salad(model_class=TinyFreshSALAD, meta=True, seed=42,
                                      train_backbone_blocks=blocks)
            image = torch.rand(4, 3, 126, 126, requires_grad=True)
            result = bilevel_objective(model, image, torch.tensor([0, 0, 1, 1]),
                        torch.rand_like(image), torch.tensor([0, 0, 1, 1]), inner_lr=.1)
            grad, = torch.autograd.grad(result.outer_loss, image)
            self.assertGreater(float(grad.norm()), 0)
            active = {n for n, p in model.named_parameters() if p.requires_grad}
            if blocks:
                self.assertIn('backbone.model.blocks.5.attn.qkv.weight', active)
                self.assertIn('backbone.model.norm.weight', active)
            else:
                self.assertFalse(any(n.startswith('backbone.') for n in active))
        with self.assertRaises(ValueError):
            build_fresh_salad(model_class=TinyFreshSALAD, meta=True, train_backbone_blocks=5)

    def test_generator_optimizer_contains_only_lora(self):
        generator, model = TinyGenerator(), TinyEmbedding()
        opt = torch.optim.AdamW(generator.parameters())
        assert_lora_optimizer(generator, opt, list(generator.parameters()))
        opt.add_param_group({'params': list(model.parameters())})
        with self.assertRaisesRegex(RuntimeError, 'ONLY'):
            assert_lora_optimizer(generator, opt, list(generator.parameters()))

    def test_teacher_loss_not_in_bilevel_objective(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, rows = make_mixed_fixture(tmp, places=2)
            for i, row in enumerate(rows):
                row['sample_id'] = f's{i}'
            episode = construct_episode(rows, [], GSVRealIndex(root, ['Bangkok']), random.Random(42), places=2)
            generator, model = TinyGenerator(), TinyEmbedding().eval()
            before = copy.deepcopy(model.state_dict())
            opt = torch.optim.SGD(generator.parameters(), lr=.01)
            def prediction(row, **kwargs):
                offset = generator.lora_phi.new_tensor([.15, -.2, .1]) * int(row['place_id'])
                image = torch.sigmoid(generator.lora_phi + offset)[None, :, None, None].expand(1, 3, 8, 8)
                return image, image.square().mean(), image.abs().mean(), 1
            with patch('AdaptVPR.experiments.vpr_guidance.train_generator.differentiable_accepted_prediction', prediction):
                record = train_bilevel_step(episode, step=1, args=meta_args(), t2i=None, vae=None,
                    unet=generator, meta_salad=model, scheduler=None, timesteps=[1], opt=opt,
                    trainable=list(generator.parameters()), rng=random.Random(42))
            self.assertGreater(record['generator/meta_only_lora_grad_norm'], 0)
            self.assertGreater(record['generator/lora_grad_norm'], 0)
            self.assertEqual(record['generator/loss_total'], record['generator/loss_meta'])
            self.assertNotIn('diagnostic/teacher_salad_cosine', record)
            self.assertFalse(any(p.grad is not None for p in model.parameters()))
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_actual_lora_unet_x0_vae_meta_only_gradient(self):
        from AdaptVPR.experiments.vpr_guidance.tests.test_gradients import tiny_unet, TinyVAE
        from AdaptVPR.experiments.vpr_guidance.iclight import attach_lora
        from diffusers import DDIMScheduler
        with tempfile.TemporaryDirectory() as tmp:
            root, _, rows = make_mixed_fixture(tmp, places=2)
            for i, row in enumerate(rows):
                row.update(sample_id=f's{i}', prompt='snow',
                           generated_sha256=file_sha256(row['generated_path']),
                           source_sha256=file_sha256(row['source_path']))
            episode = construct_episode(rows, [], GSVRealIndex(root, ['Bangkok']), random.Random(42), places=2)
            unet = tiny_unet()
            original_forward = unet.forward
            def iclight_forward(*args, **kwargs):
                # Real IC-Light consumes concat_conds in its UNet wrapper.
                kwargs.pop('cross_attention_kwargs', None)
                return original_forward(*args, **kwargs)
            unet.forward = iclight_forward
            trainable = attach_lora(unet, rank=2, alpha=2)
            unet.enable_gradient_checkpointing()
            unet.train()
            vae = TinyVAE().requires_grad_(False).eval()
            model = TinyEmbedding().eval()
            opt = torch.optim.SGD(trainable, lr=.01)
            scheduler = DDIMScheduler(beta_schedule='scaled_linear', beta_start=.00085,
                                      beta_end=.012, clip_sample=False)
            from contextlib import ExitStack
            with ExitStack() as stack:
                stack.enter_context(patch('AdaptVPR.experiments.vpr_guidance.train_generator.encode_image_latent',
                                          lambda *a: torch.randn(1, 4, 8, 8) * .1))
                stack.enter_context(patch('AdaptVPR.adapters.iclight_sd15_fc._concat_condition',
                                          lambda *a: torch.zeros(1, 4, 8, 8)))
                stack.enter_context(patch('AdaptVPR.experiments.vpr_guidance.train_generator.encode_prompt',
                                          lambda *a: torch.randn(1, 5, 8)))
                record = train_bilevel_step(episode, step=1, args=meta_args(), t2i=None, vae=vae,
                    unet=unet, meta_salad=model, scheduler=scheduler, timesteps=[161], opt=opt,
                    trainable=trainable, rng=random.Random(42))
            self.assertGreater(record['generator/meta_only_lora_grad_norm'], 0)
            self.assertGreater(record['generator/lora_grad_norm'], 0)
            self.assertFalse(any(p.grad is not None for p in model.parameters()))
            self.assertFalse(any(p.grad is not None for p in vae.parameters()))
            self.assertFalse(any(p.grad is not None for n,p in unet.named_parameters() if 'lora_' not in n))

    def test_teacher_diagnostic_cannot_change_lora_gradient(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, rows = make_mixed_fixture(tmp, places=2)
            for i, row in enumerate(rows):
                row['sample_id'] = f's{i}'
            episode = construct_episode(rows, [], GSVRealIndex(root, ['Bangkok']), random.Random(42), places=2)
            base_generator, base_model = TinyGenerator(), TinyEmbedding().eval()
            class DiagnosticTeacher:
                device = 'cpu'
                def __call__(self, image):
                    assert not torch.is_grad_enabled()
                    return F.normalize(image.mean((-1, -2)), dim=-1)
            descriptors = {r['sample_id']: torch.ones(1, 3, requires_grad=True) for r in rows}
            gradients = []
            for teacher in (None, DiagnosticTeacher()):
                generator, model = copy.deepcopy(base_generator), copy.deepcopy(base_model)
                def prediction(row, **kwargs):
                    x = torch.sigmoid(generator.lora_phi + int(row['place_id']) * .1)
                    image = x[None, :, None, None].expand(1, 3, 8, 8)
                    return image, image.square().mean(), image.abs().mean(), 1
                with patch('AdaptVPR.experiments.vpr_guidance.train_generator.differentiable_accepted_prediction', prediction):
                    record = train_bilevel_step(episode, step=1, args=meta_args(), t2i=None, vae=None,
                        unet=generator, meta_salad=model, scheduler=None, timesteps=[1],
                        opt=torch.optim.SGD(generator.parameters(), lr=.01),
                        trainable=list(generator.parameters()), rng=random.Random(42),
                        teacher=teacher, source_descriptors=descriptors)
                gradients.append(generator.lora_phi.grad.clone())
                self.assertEqual('diagnostic/teacher_salad_cosine' in record, teacher is not None)
            torch.testing.assert_close(*gradients, rtol=0, atol=0)
            self.assertFalse(any(d.grad is not None for d in descriptors.values()))

    def test_old_objective_checkpoint_and_retired_cli_are_rejected(self):
        for payload in ({}, {'extra': {'config': {'lambda_vpr': .1}}},
                        {'extra': {'config': {'objective_version': 'teacher_cosine_v1'}}}):
            with self.assertRaisesRegex(ValueError, 'old frozen-teacher'):
                validate_bilevel_checkpoint(payload)
        validate_bilevel_checkpoint({'extra': {'config': {'objective_version': OBJECTIVE_VERSION}}})
        validate_meta_args(meta_args())
        for name, value in [('lambda_vpr', .1), ('meta_places', 1), ('meta_inner_steps', 0),
                            ('meta_query_real_per_place', 1), ('meta_image_size', 128),
                            ('meta_train_backbone_blocks', 5), ('meta_inner_lr', float('nan'))]:
            with self.assertRaises(ValueError):
                validate_meta_args(meta_args(**{name: value}))

    def test_full_pipeline_forwards_all_meta_settings(self):
        from AdaptVPR.experiments.vpr_guidance.train_full import args_parser, generator_training_flags
        from AdaptVPR.experiments.vpr_guidance.train_online_generator import args_parser as online_parser
        from AdaptVPR.experiments.vpr_guidance.salad_factory import META_FIELDS
        shared = ['--prompts', 'prompts.jsonl', '--gsv-root', 'gsv', '--salad-root', 'salad',
                  '--output-dir', 'out']
        args = args_parser().parse_args(shared + ['--meta-inner-steps', '2', '--meta-inner-lr', '.02',
                '--meta-places', '3', '--meta-support-real-per-place', '2',
                '--meta-query-real-per-place', '3', '--meta-image-size', '140',
                '--meta-train-backbone-blocks', '4', '--lambda-meta', '.7', '--generator-grad-scale', '1024'])
        flags = list(map(str, generator_training_flags(args)))
        online = online_parser().parse_args(shared + flags)
        for field in META_FIELDS:
            self.assertEqual(getattr(args, field), getattr(online, field))
        self.assertNotIn('--lambda-vpr', flags)

    def test_rejected_targets_fail_before_prediction(self):
        with self.assertRaisesRegex(ValueError, 'rejected'):
            differentiable_accepted_prediction({'passed': False}, step=1, t2i=None, vae=None,
                unet=None, scheduler=None, timesteps=None, rng=None)


class EpisodeTests(unittest.TestCase):
    def test_support_query_are_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, rows = make_mixed_fixture(tmp, places=4)
            index = GSVRealIndex(root, ['Bangkok'])
            first = construct_episode(rows[:1], [rows[1:]], index, random.Random(42))
            second = construct_episode(rows[:1], [rows[1:]], index, random.Random(42))
            self.assertEqual(first, second)
            self.assertEqual(len(first.places), 4)
            self.assertIn(rows[0], [p.synthetic_row for p in first.places])
            for place in first.places:
                self.assertFalse(set(place.support_real) & set(place.query_real))
                self.assertEqual(len(place.query_real), 2)
                for path in place.support_real + place.query_real:
                    self.assertEqual(index.labels.key_for_source(Path(path)), place.key)
                    self.assertNotEqual(path, place.synthetic_row['generated_path'])

    def test_meta_batch_requires_multiple_places(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, rows = make_mixed_fixture(tmp, places=1)
            index = GSVRealIndex(root, ['Bangkok'])
            with self.assertRaises(EpisodeUnavailable):
                construct_episode(rows, [], index, random.Random(42), places=2)
            with self.assertRaises(ValueError):
                construct_episode(rows, [], index, random.Random(42), places=1)
            with self.assertRaisesRegex(ValueError, 'multiple distinct'):
                MetaEpisode(tuple()).validate()

    def test_replay_needs_current_and_wrong_labels_or_rejected_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, rows = make_mixed_fixture(tmp, places=2)
            index = GSVRealIndex(root, ['Bangkok'])
            with self.assertRaisesRegex(EpisodeUnavailable, 'no current'):
                construct_episode([], [rows], index, random.Random(42), places=2)
            with self.assertRaisesRegex(ValueError, 'place_id disagrees'):
                construct_episode([{**rows[0], 'place_id': '2'}], [rows[1:]], index,
                                   random.Random(42), places=2)
            with self.assertRaisesRegex(ValueError, 'rejected'):
                construct_episode([{**rows[0], 'passed': False}], [rows[1:]], index,
                                   random.Random(42), places=2)
            # Multiple synthetic conditions for one place cannot become negatives.
            with self.assertRaises(EpisodeUnavailable):
                construct_episode([rows[0], dict(rows[0])], [], index, random.Random(42), places=2)

    def test_insufficient_real_query_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, rows = make_mixed_fixture(tmp, places=2)
            for path in list((root / 'Images' / 'Bangkok').glob('*.jpg')):
                if not any(str(path) == r['source_path'] for r in rows):
                    path.unlink()
            with self.assertRaisesRegex(EpisodeUnavailable, 'enough distinct real'):
                construct_episode(rows, [], GSVRealIndex(root, ['Bangkok']), random.Random(42), places=2)


if __name__ == '__main__':
    unittest.main()
