"""Stage 3 evidence, gradient isolation, OT constraints and joint training."""
import contextlib
import copy
import hashlib
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from models.aggregators.salad import SALAD, get_matching_probs
from workflow.reliability import (tristate_reliability_loss, reliability_region_statistics,
                                  summarize_reliability_regions)
from workflow.reliability_corruption import make_generated_occlusions
from workflow.model import SALADModel, input_preprocessing
import test_local_distillation as fixtures


class TriStateTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.manual_seed(13)

    def test_independent_targets_unknown_zero_and_detached_confidence(self):
        logits = torch.zeros(1, 1, 2, 3, requires_grad=True)
        pos = torch.tensor([[[[1., .4, 0], [0, 0, 0]]]], requires_grad=True)
        neg = torch.tensor([[[[0., 0, .7], [0, 0, 0]]]], requires_grad=True)
        result = tristate_reliability_loss(logits, pos, neg)
        result['loss'].backward()
        self.assertTrue((logits.grad[pos > 0] < 0).all())
        self.assertTrue((logits.grad[neg > 0] > 0).all())
        self.assertTrue((logits.grad[(pos + neg) == 0] == 0).all())
        self.assertIsNone(pos.grad); self.assertIsNone(neg.grad)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            tristate_reliability_loss(logits, pos, pos)

    def test_unknown_dark_or_low_similarity_never_creates_negative(self):
        logits = torch.full((2, 1, 3, 5), -20., requires_grad=True)
        zero = torch.zeros_like(logits)
        loss = tristate_reliability_loss(logits, zero, zero)['loss']
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertFalse(logits.grad.any())
        report = summarize_reliability_regions(reliability_region_statistics(logits, zero, zero,
                                               torch.ones(2, 15)))
        self.assertIsNone(report['negative_mean_reliability'])
        self.assertEqual(report['unknown_fraction'], 1.)

    def test_amp_extreme_logits_and_finite_validation(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            logits = torch.tensor([[[[1000., -1000.]]]], dtype=dtype, requires_grad=True)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                loss = tristate_reliability_loss(logits, torch.tensor([[[[1., 0.]]]]),
                                                 torch.tensor([[[[0., 1.]]]]))['loss']
            self.assertEqual(loss.dtype, torch.float32)
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
        with self.assertRaises(ValueError):
            tristate_reliability_loss(logits * float('nan'), torch.zeros_like(logits), torch.zeros_like(logits))

    def test_detach_switch_blocks_only_predictor_gradient_not_local_feature(self):
        for detach in (False, True):
            model = SALAD(num_channels=8, num_clusters=2, cluster_dim=4, token_dim=4,
                          reliability_ot=True, reliability_context=True, reliability_detach_features=detach)
            # Fresh head has zero final weights; a trained/nonconstant head is
            # necessary to test whether the path is blocked rather than zero-init.
            with torch.no_grad():
                model.reliability_head[2].weight.normal_(std=.1)
            features = torch.randn(1, 8, 4, 4, requires_grad=True)
            logits = model.predict_reliability(features)
            pos = torch.zeros_like(logits); pos[..., :2, :] = 1
            neg = torch.zeros_like(logits); neg[..., 2:, :] = 1
            tristate_reliability_loss(logits, pos, neg)['loss'].backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.reliability_parameters()))
            if detach:
                self.assertIsNone(features.grad)
            else:
                self.assertGreater(float(features.grad.abs().sum()), 0.)
            features.grad = None
            features.square().mean().backward()
            self.assertGreater(float(features.grad.abs().sum()), 0.)

    def test_dustbin_redistributes_fixed_mass_and_zero_r_is_not_hard_rejection(self):
        scores = torch.randn(1, 4, 20) * .3
        constant = get_matching_probs(scores, torch.tensor(1.), num_iters=100).exp()
        shifted = get_matching_probs(scores, torch.tensor(3.), num_iters=100).exp()
        torch.testing.assert_close(constant, shifted, atol=2e-6, rtol=2e-6)
        reliability = torch.ones(1, 20); reliability[:, :5] = 0
        varied = get_matching_probs(scores, 1 + 2 * (1 - reliability), num_iters=100).exp()
        torch.testing.assert_close(varied.sum(1), torch.ones(1, 20), atol=2e-6, rtol=2e-6)
        self.assertAlmostEqual(float(varied[:, -1].sum()), 16., places=4)
        self.assertTrue((varied[:, :-1, :5].sum(1) > 0).all())
        self.assertGreater(float(varied[:, -1, :5].mean()), float(constant[:, -1, :5].mean()))

    def test_controlled_occlusion_exact_mask_no_source_mutation_and_independent_rng(self):
        images = torch.randn(2, 3, 56, 84)
        original = images.clone(); rng = torch.get_rng_state()
        changed, mask = make_generated_occlusions(images, generator=torch.Generator().manual_seed(11))
        self.assertTrue(torch.equal(images, original)); self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        pixel_mask = mask.repeat_interleave(14, -2).repeat_interleave(14, -1).expand_as(images)
        self.assertTrue(torch.equal(changed != original, pixel_mask))
        self.assertEqual(int(mask.sum()), 8)
        again = make_generated_occlusions(images, generator=torch.Generator().manual_seed(11))
        self.assertTrue(torch.equal(changed, again[0]))
        _, empty = make_generated_occlusions(torch.zeros_like(images), generator=torch.Generator().manual_seed(11))
        self.assertFalse(empty.any())


class Stage3WorkflowTests(unittest.TestCase):
    arguments = fixtures.LocalWorkflowTests.arguments
    run_training = fixtures.LocalWorkflowTests.run_training

    def setUp(self):
        fixtures.LocalWorkflowTests.setUp(self)
        self.payload['model_config']['image_size'] = [56, 56]
        self.payload['image_preprocessing'] = input_preprocessing((56, 56))
        torch.save(self.payload, self.teacher_path)
        calibration = json.loads(self.calibration_path.read_text())
        calibration['provenance']['teacher'] = dict(
            checkpoint_sha256=hashlib.sha256(self.teacher_path.read_bytes()).hexdigest(),
            image_preprocessing=self.payload['image_preprocessing'])
        self.calibration_path.write_text(json.dumps(calibration))
        for i, path in enumerate(sorted((self.data/'Images/Test').glob('*.jpg'))):
            Image.fromarray(np.random.default_rng(i).integers(10, 240, (56, 56, 3), dtype=np.uint8)).save(path)
        for row in map(json.loads, self.manifest.read_text().splitlines()):
            with Image.open(row['source_path']) as image:
                Image.fromarray((np.asarray(image)*.8).astype(np.uint8)).save(row['output_path'])

    def args(self, output, *extra):
        return fixtures.LocalWorkflowTests.args(self, output, '--image-size', '56', '56',
            '--reliability-ot', '--reliability-context', '--reliability-detach-features',
            '--reliability-head-lr', '.001', '--reliability-corruption-weight', '.05',
            '--reliability-generated-weight', '.1', *extra)

    def test_joint_training_real_copy_paste_and_explicit_generated_occlusion(self):
        args = self.args(self.root/'joint', '--epochs', '1', '--reliability-generated-occlusion-weight', '.05')
        events=[]
        backward, step = torch.Tensor.backward, torch.optim.AdamW.step
        def capture_backward(loss, *a, **kw):
            events.append('backward'); return backward(loss, *a, **kw)
        def capture_step(opt, *a, **kw):
            self.assertTrue(all(torch.isfinite(p.grad).all() for g in opt.param_groups for p in g['params'] if p.grad is not None))
            events.append('step'); return step(opt, *a, **kw)
        with patch.object(torch.Tensor, 'backward', capture_backward), patch.object(torch.optim.AdamW, 'step', capture_step):
            result = self.run_training(args)
        self.assertEqual(events, ['backward']*6 + ['step'])
        metrics = result['metrics']
        self.assertGreater(metrics['reliability']['corruption_negative_patches'], 0)
        self.assertEqual(metrics['generated_reliability']['negative_count'], 0)
        self.assertGreater(metrics['generated_reliability']['positive_count'], 0)
        self.assertGreater(metrics['controlled_generated_occlusion']['negative_count'], 0)
        self.assertIsNone(metrics['generated_reliability']['negative_mean_reliability'])

    def test_resume_exact_and_new_configuration_checked(self):
        args = self.args(self.root/'full', '--reliability-generated-occlusion-weight', '.05')
        full = self.run_training(args)
        args.init_checkpoint=None; args.resume=self.root/'full/checkpoint_epoch_001.pt'; args.output_dir=self.root/'resume'
        resumed = self.run_training(args)
        for key, value in full['state_dict'].items():
            torch.testing.assert_close(value, resumed['state_dict'][key], rtol=0, atol=0)
        self.assertEqual(full['rng_state']['synthetic_aux'], resumed['rng_state']['synthetic_aux'])
        args.reliability_generated_occlusion_weight=.04
        with self.assertRaisesRegex(ValueError, 'training_config'):
            self.run_training(args)

    def test_explicit_manifest_occlusion_has_no_positive_overlap(self):
        rows = list(map(json.loads, self.manifest.read_text().splitlines()))
        for i, row in enumerate(rows):
            with Image.open(row['output_path']) as handle:
                pixels = np.array(handle)
            pixels[:14,:14] = 0
            Image.fromarray(pixels).save(row['output_path'])
            mask = np.zeros((56,56), dtype=np.uint8); mask[:14,:14]=255
            path=self.root/f'mask{i}.png'; Image.fromarray(mask).save(path)
            row['occlusion_mask_path']=str(path)
        self.manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        result=self.run_training(self.args(self.root/'mask', '--epochs','1'))
        self.assertEqual(result['metrics']['generated_reliability']['negative_count'], 2)

    def test_empty_auxiliary_has_no_fabricated_labels(self):
        result=self.run_training(self.args(self.root/'empty','--epochs','1','--synthetic-exposure-prob','0'))
        self.assertNotIn('generated_reliability',result['metrics'])
        self.assertGreater(result['metrics']['reliability']['corruption_negative_patches'],0)


if __name__ == '__main__':
    unittest.main()
