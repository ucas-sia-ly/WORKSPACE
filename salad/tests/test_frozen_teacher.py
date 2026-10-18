"""Stage 2A interfaces, finite-iteration OT, and strictly frozen teachers."""

import contextlib
import copy
import hashlib
import io
import unittest
from unittest.mock import patch

import torch

import test_pretrained_initialization as fixture
from test_reliability_ot import legacy_descriptor
from models.aggregators.salad import SALAD, get_matching_probs
from train_salad import train
from workflow.metric_loss import multi_similarity_loss
from workflow.model import (FrozenSALADTeacher, SALADModel, input_preprocessing,
                            load_checkpoint_model, profile_teacher_targets, read_checkpoint,
                            validate_teacher_compatibility)


class SALADAuxInterfaceTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        torch.manual_seed(39)

    def test_default_and_rectangular_shapes_match_plain_descriptor(self):
        for reliability in (False, True):
            model = SALAD(num_channels=768, reliability_ot=reliability).eval()
            for h, w in ((16, 16), (14, 18)):
                with self.subTest(reliability=reliability, h=h, w=w), torch.no_grad():
                    x, token = torch.randn(2, 768, h, w), torch.randn(2, 768)
                    plain = model((x, token))
                    descriptor, aux = model((x, token), return_aux=True)
                    torch.testing.assert_close(descriptor, plain, rtol=0, atol=0)
                    self.assertIs(aux['backbone_local_features'], x)
                    self.assertIs(aux['local_features'], x)
                    self.assertEqual(aux['content_features'].shape, (2, 128, h*w))
                    self.assertEqual(aux['score_logits'].shape, (2, 64, h*w))
                    self.assertEqual(aux['log_assignment'].shape, (2, 65, h*w))
                    self.assertTrue(torch.isfinite(aux['log_assignment']).all())
                    torch.testing.assert_close(aux['content_features'], model.cluster_features(x).flatten(2))
                    torch.testing.assert_close(aux['score_logits'], model.score(x).flatten(2))
                    assignment = aux['log_assignment'].exp()
                    torch.testing.assert_close(assignment[:, -1], aux['assignment_dustbin'])
                    torch.testing.assert_close(assignment[:, :-1].sum(1), aux['assignment_keep'])
                    self.assertEqual('reliability_logits' in aux, reliability)
                    if not reliability:
                        torch.testing.assert_close(plain, legacy_descriptor(model, x, token), rtol=0, atol=0)

    def test_original_parameter_keys_order_and_strict_loading(self):
        expected = ['dust_bin']
        for name, layers in (('token_features', (0, 2)), ('cluster_features', (0, 3)), ('score', (0, 3))):
            for layer in layers:
                expected.extend([f'{name}.{layer}.weight', f'{name}.{layer}.bias'])
        model = SALAD(num_channels=12, num_clusters=4, cluster_dim=6, token_dim=8)
        self.assertEqual(list(model.state_dict()), expected)
        self.assertEqual([name for name, _ in model.named_parameters()], expected)
        restored = SALAD(num_channels=12, num_clusters=4, cluster_dim=6, token_dim=8)
        incompatible = restored.load_state_dict(model.state_dict(), strict=True)
        self.assertFalse(incompatible.missing_keys or incompatible.unexpected_keys)

    def test_finite_iteration_ot_mass_constraints(self):
        # Moderate logits permit a useful 3-iteration row-marginal tolerance.
        # Arbitrarily extreme scores do NOT have a universal 3-iteration bound.
        for n in (81, 256, 322):
            scores = torch.randn(2, 64, n) * 0.5
            assignment = get_matching_probs(scores, torch.tensor(1.), num_iters=3).exp()
            converged = get_matching_probs(scores, torch.tensor(1.), num_iters=100).exp()
            target = torch.tensor([1.] * 64 + [float(n - 64)]).expand(2, -1)
            self.assertEqual(assignment.shape, (2, 65, n))
            torch.testing.assert_close(assignment.sum(1), torch.ones(2, n), atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(assignment.sum((1, 2)), torch.full((2,), float(n)), atol=1e-4, rtol=2e-6)
            torch.testing.assert_close(assignment.sum(2), target, atol=2e-3, rtol=2e-3)
            torch.testing.assert_close(converged.sum(2), target, atol=1e-4, rtol=2e-5)
            self.assertLessEqual((converged.sum(2) - target).abs().max(),
                                 (assignment.sum(2) - target).abs().max() + 1e-5)

    def test_aux_gradients_reach_student_content_and_score_without_reordering_dropout(self):
        model = SALAD(num_channels=12, num_clusters=4, cluster_dim=6, token_dim=8).train()
        x, token = torch.randn(2, 12, 4, 5, requires_grad=True), torch.randn(2, 12)
        state = torch.get_rng_state()
        plain = model((x, token))
        after_plain = torch.get_rng_state()
        torch.set_rng_state(state)
        descriptor, aux = model((x, token), return_aux=True)
        torch.testing.assert_close(descriptor, plain, rtol=0, atol=0)
        self.assertTrue(torch.equal(after_plain, torch.get_rng_state()))
        # Probe the interfaces with a linear functional, not a distillation loss.
        loss = sum((aux[name] * torch.randn_like(aux[name])).sum()
                   for name in ('content_features', 'score_logits', 'log_assignment'))
        loss.backward()
        for branch in (model.cluster_features, model.score):
            for parameter in branch.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_sharp_scores_need_more_iterations_for_row_marginals(self):
        scores = torch.randn(2, 64, 256) * 4
        short = get_matching_probs(scores, torch.tensor(1.), num_iters=3).exp()
        long = get_matching_probs(scores, torch.tensor(1.), num_iters=200).exp()
        target = torch.tensor([1.] * 64 + [192.])
        torch.testing.assert_close(short.sum(1), torch.ones(2, 256), rtol=2e-6, atol=2e-6)
        self.assertGreater((short.sum(2) - target).abs().max().item(), 1e-3)
        torch.testing.assert_close(long.sum(2), target.expand(2, -1), rtol=2e-5, atol=2e-4)


class FrozenTeacherTests(unittest.TestCase):
    arguments = fixture.PretrainedInitializationTests.arguments

    def setUp(self):
        fixture.PretrainedInitializationTests.setUp(self)
        # Produce a genuinely updated tiny checkpoint, not random weights
        # relabeled as a trained teacher.
        model = SALADModel(self.config, pretrained_backbone=False, backbone_repo=self.repo).train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss = multi_similarity_loss(model(torch.randn(4, 3, 28, 28)), torch.tensor([0, 0, 1, 1]))
        loss.backward()
        optimizer.step()
        self.payload = {'state_dict': model.state_dict(), 'model_config': self.config,
                        'global_step': 1, 'epoch': 1, 'image_preprocessing': input_preprocessing((28, 28))}
        self.teacher_path = self.root / 'trained_teacher.pt'
        torch.save(self.payload, self.teacher_path)

    def teacher(self, path=None, config=None):
        return FrozenSALADTeacher(path or self.teacher_path, config or self.config, backbone_repo=self.repo)

    def test_frozen_eval_no_grad_fp32_and_initialization_rng_isolation(self):
        rng = torch.get_rng_state()
        teacher = self.teacher()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(teacher.provenance['checkpoint_sha256'], hashlib.sha256(self.teacher_path.read_bytes()).hexdigest())
        teacher.train().requires_grad_(True)
        self.assertTrue(all(not m.training for m in teacher.modules()))
        self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
        images = torch.randn(2, 3, 28, 28, requires_grad=True)
        with torch.enable_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
            descriptor, aux = teacher(images)
        self.assertFalse(descriptor.requires_grad)
        self.assertTrue(all(not value.requires_grad for value in aux.values()))
        self.assertEqual(aux['log_assignment'].dtype, torch.float32)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        plain = teacher(images, return_aux=False)
        torch.testing.assert_close(descriptor, plain, rtol=0, atol=0)
        self.assertIsNone(images.grad)

    def test_raw_random_zero_step_and_partial_teacher_rejected_but_legacy_student_loads(self):
        raw = self.root / 'raw.pt'
        torch.save(self.payload['state_dict'], raw)
        student = load_checkpoint_model(raw, 'cpu', backbone_repo=self.repo)
        self.assertEqual(set(student.state_dict()), set(self.payload['state_dict']))
        with self.assertRaisesRegex(ValueError, 'full trained'):
            self.teacher(raw)
        for step in (0, -1, None):
            invalid = self.root / f'untrained_{step}.pt'
            torch.save({**self.payload, 'global_step': step}, invalid)
            with self.assertRaisesRegex(ValueError, 'full trained'):
                self.teacher(invalid)
        missing = {k: v for k, v in self.payload['state_dict'].items() if k != 'aggregator.score.3.weight'}
        partial = self.root / 'partial.pt'
        torch.save({**self.payload, 'state_dict': missing}, partial)
        with self.assertRaisesRegex(RuntimeError, 'score.3.weight'):
            self.teacher(partial)

    def test_compatibility_checks_every_feature_dimension_and_preprocessing(self):
        for key in ('num_clusters', 'cluster_dim', 'token_dim'):
            config = copy.deepcopy(self.config)
            config['agg_config'][key] += 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self.teacher(config=config)
        config = copy.deepcopy(self.config)
        config['backbone_arch'] = 'dinov2_vitb14'
        config['agg_config']['num_channels'] = 768
        with self.assertRaisesRegex(ValueError, 'backbone architectures'):
            self.teacher(config=config)
        config = copy.deepcopy(self.config)
        config['image_size'] = [28, 42]
        with self.assertRaisesRegex(ValueError, 'preprocessing'):
            self.teacher(config=config)
        preprocessing = input_preprocessing((28, 28))
        preprocessing['mean'] = [0.5, 0.5, 0.5]
        with self.assertRaisesRegex(ValueError, 'preprocessing'):
            validate_teacher_compatibility(self.config, self.config, preprocessing)

    def test_non_square_online_targets_and_chunk_equivalence(self):
        config = copy.deepcopy(self.config)
        config['image_size'] = [28, 42]
        path = self.root / 'rectangle.pt'
        torch.save({**self.payload, 'model_config': config,
                    'image_preprocessing': input_preprocessing((28, 42))}, path)
        teacher = self.teacher(path, config)
        images = torch.randn(5, 3, 28, 42)
        whole, whole_aux = teacher(images)
        pieces = list(teacher.iter_targets(images, batch_size=2))
        self.assertEqual([offset for offset, _, _ in pieces], [0, 2, 4])
        torch.testing.assert_close(torch.cat([descriptor for _, descriptor, _ in pieces]), whole, rtol=1e-5, atol=1e-6)
        for name, value in whole_aux.items():
            torch.testing.assert_close(torch.cat([aux[name] for _, _, aux in pieces]), value, rtol=1e-5, atol=1e-6)
        self.assertEqual(whole_aux['log_assignment'].shape, (5, 3, 6))
        report = profile_teacher_targets(teacher, images, batch_size=2)
        self.assertEqual(report['images'], 5)
        self.assertEqual(report['chunks'], 3)
        self.assertIsNone(report['cuda_peak_allocated_bytes'])
        self.assertGreater(report['seconds'], 0)

    def test_lightning_resolution_and_nested_preprocessing_are_checked(self):
        path = self.root / 'lightning.pt'
        torch.save({'state_dict': self.payload['state_dict'], 'global_step': 1,
                    'hyper_parameters': {'image_size': [28, 28], 'backbone_config': self.config['backbone_config']}}, path)
        teacher = self.teacher(path)
        self.assertEqual(teacher.provenance['model_config']['image_size'], [28, 28])
        incompatible = input_preprocessing((28, 28))
        incompatible['color_space'] = 'BGR'
        torch.save({**self.payload, 'training_config': {'image_preprocessing': incompatible}}, path)
        with self.assertRaisesRegex(ValueError, 'conflicting preprocessing'):
            self.teacher(path)

    def test_oom_halves_chunks_and_single_image_failure_is_actionable(self):
        teacher = self.teacher()
        original = teacher.forward
        images = torch.randn(5, 3, 28, 28)
        def limited(batch, **kwargs):
            if len(batch) > 2:
                raise torch.cuda.OutOfMemoryError('simulated bounded-memory GPU')
            return original(batch, **kwargs)
        with patch.object(teacher, 'forward', side_effect=limited):
            pieces = list(teacher.iter_targets(images))
        self.assertEqual([len(descriptor) for _, descriptor, _ in pieces], [2, 2, 1])
        self.assertEqual(teacher.oom_retries, 1)
        with patch.object(teacher, 'forward', side_effect=torch.cuda.OutOfMemoryError('simulated')):
            with self.assertRaisesRegex(RuntimeError, 'one image'):
                list(teacher.iter_targets(images, batch_size=1))

    def test_training_probe_does_not_change_student_and_teacher_stays_out_of_optimizer(self):
        def args(output, with_teacher):
            options = ['--teacher-checkpoint', str(self.teacher_path), '--teacher-batch-size', '2'] if with_teacher else []
            return self.arguments(output, '--init-checkpoint', str(self.teacher_path), '--synthetic-fraction', '0',
                                  '--batch-size', '3', '--num-trainable-blocks', '1', *options)
        with contextlib.redirect_stdout(io.StringIO()):
            train(args(self.root / 'baseline', False))
        teachers = []
        def factory(*a, **kw):
            teacher = FrozenSALADTeacher(*a, **kw)
            teachers.append(teacher)
            return teacher
        original_step = torch.optim.AdamW.step
        def checked_step(optimizer, *a, **kw):
            teacher_ids = {id(p) for teacher in teachers for p in teacher.parameters()}
            self.assertFalse(teacher_ids & {id(p) for group in optimizer.param_groups for p in group['params']})
            return original_step(optimizer, *a, **kw)
        with patch('workflow.model.FrozenSALADTeacher', side_effect=factory), \
                patch.object(torch.optim.AdamW, 'step', checked_step), contextlib.redirect_stdout(io.StringIO()):
            train(args(self.root / 'probed', True))
        baseline = read_checkpoint(self.root / 'baseline' / 'checkpoint.pt')
        actual = read_checkpoint(self.root / 'probed' / 'checkpoint.pt')
        for name, value in baseline['state_dict'].items():
            torch.testing.assert_close(value, actual['state_dict'][name], rtol=0, atol=0)
        self.assertEqual(actual['metrics']['teacher']['images'], 6)
        self.assertEqual(actual['metrics']['teacher']['chunks'], 3)
        self.assertEqual(actual['training_config']['teacher']['checkpoint_sha256'], teachers[0].provenance['checkpoint_sha256'])
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in teachers[0].parameters()))
        for name, value in teachers[0].model.state_dict().items():
            torch.testing.assert_close(value, self.payload['state_dict'][name], rtol=0, atol=0)
        resume_args = args(self.root / 'resumed', True)
        resume_args.init_checkpoint = None
        resume_args.resume = self.root / 'probed' / 'checkpoint_epoch_001.pt'
        with contextlib.redirect_stdout(io.StringIO()):
            train(resume_args)
        resumed = read_checkpoint(resume_args.output_dir / 'checkpoint.pt')
        for name, value in actual['state_dict'].items():
            torch.testing.assert_close(value, resumed['state_dict'][name], rtol=0, atol=0)
        torch.save({**self.payload, 'global_step': 2}, self.teacher_path)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'training_config'):
            train(resume_args)


if __name__ == '__main__':
    unittest.main()
