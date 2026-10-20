"""Registered legacy pretrained teachers retain strict loading and provenance."""

import copy
import hashlib
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import torch

import test_frozen_teacher as fixture
import run_ccr_feasibility as feasibility
import workflow.model as models


class RegisteredTeacherTests(unittest.TestCase):
    def setUp(self):
        # The existing fixture actually updates its tiny model once; registering
        # its raw export therefore does not relabel random weights as trained.
        fixture.FrozenTeacherTests.setUp(self)
        self.raw = self.root / 'pretrained_raw.pt'
        torch.save(self.payload['state_dict'], self.raw)

    @staticmethod
    def file_sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def registration(self, path=None):
        return patch.dict(models.REGISTERED_PRETRAINED_TEACHERS, {
            self.file_sha(path or self.raw): {
                'name': 'genuinely_stepped_tiny_pretrained_fixture',
                'source_url': 'https://example.invalid/trained_fixture.pt',
            },
        })

    def teacher(self, path=None, config=None):
        return models.FrozenSALADTeacher(
            path or self.raw, config or self.config, backbone_repo=self.repo)

    def test_registered_raw_strictly_loads_frozen_targets_without_invented_steps(self):
        rng = torch.get_rng_state()
        with self.registration():
            teacher = self.teacher()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        teacher.train().requires_grad_(True)
        self.assertTrue(all(not module.training for module in teacher.modules()))
        self.assertTrue(all(not parameter.requires_grad for parameter in teacher.parameters()))
        for name, value in self.payload['state_dict'].items():
            torch.testing.assert_close(teacher.model.state_dict()[name], value, rtol=0, atol=0)
        self.assertEqual(teacher.provenance['checkpoint_sha256'], self.file_sha(self.raw))
        self.assertIsNone(teacher.provenance['checkpoint_global_step'])
        self.assertEqual(teacher.provenance['training_evidence']['kind'],
                         'registered_pretrained_artifact')
        images = torch.randn(2, 3, 28, 28, requires_grad=True)
        with torch.enable_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
            descriptor, aux = teacher(images)
        self.assertFalse(descriptor.requires_grad)
        self.assertTrue(all(not tensor.requires_grad for tensor in aux.values()))
        self.assertTrue(torch.isfinite(descriptor).all())
        self.assertTrue(torch.isfinite(aux['log_assignment']).all())
        self.assertEqual(aux['log_assignment'].dtype, torch.float32)
        self.assertIsNone(images.grad)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))

    def test_registration_follows_bytes_when_pretrained_file_is_renamed(self):
        renamed = self.root / 'arbitrary_new_filename.ckpt'
        shutil.copyfile(self.raw, renamed)
        with self.registration():
            teacher = self.teacher(renamed)
        self.assertEqual(teacher.provenance['checkpoint'], str(renamed.resolve()))
        self.assertEqual(teacher.provenance['checkpoint_sha256'], self.file_sha(self.raw))

    def test_modified_raw_weights_do_not_inherit_registration(self):
        changed = copy.deepcopy(self.payload['state_dict'])
        changed['aggregator.cluster_features.3.weight'].flatten()[0] += 0.125
        mutated = self.root / 'modified_pretrained.pt'
        torch.save(changed, mutated)
        self.assertNotEqual(self.file_sha(mutated), self.file_sha(self.raw))
        with self.registration(), self.assertRaisesRegex(ValueError, 'full trained|registered'):
            self.teacher(mutated)

    def test_unregistered_raw_and_random_aggregator_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'full trained|registered'):
            self.teacher()
        random_path = self.root / 'dino_salad.ckpt'
        # A familiar filename must never count as training evidence.
        torch.save(self.state, random_path)
        with self.registration(), self.assertRaisesRegex(ValueError, 'full trained|registered'):
            self.teacher(random_path)

    def test_explicit_invalid_native_step_is_rejected_even_if_hash_registered(self):
        for step in (0, -1, None, True):
            path = self.root / f'explicit_invalid_step_{step}.pt'
            torch.save({**self.payload, 'global_step': step}, path)
            with self.subTest(step=step), self.registration(path), \
                    self.assertRaisesRegex(ValueError, 'full trained|global_step'):
                self.teacher(path)

    def test_registered_partial_aggregator_still_fails_strict_loading(self):
        # Retain the shape-inference tensors, so only strict state loading can
        # detect this omitted trained aggregator parameter.
        state = {name: value for name, value in self.payload['state_dict'].items()
                 if name != 'aggregator.cluster_features.3.bias'}
        partial = self.root / 'partial_registered.pt'
        torch.save(state, partial)
        with self.registration(partial), self.assertRaisesRegex(
                RuntimeError, 'cluster_features.3.bias'):
            self.teacher(partial)

    def test_raw_resolution_is_explicit_student_contract_or_legacy_default(self):
        rectangular = copy.deepcopy(self.config)
        rectangular['image_size'] = [28, 42]
        with self.registration():
            state, default_config, default_provenance = models.load_teacher_checkpoint(self.raw)
            self.assertEqual(default_config['image_size'], [224, 224])
            self.assertEqual(default_provenance['image_preprocessing'],
                             models.input_preprocessing((224, 224)))
            state, config, provenance = models.load_teacher_checkpoint(
                self.raw, student_config=rectangular)
            teacher = self.teacher(config=rectangular)
        self.assertEqual(set(state), set(self.payload['state_dict']))
        self.assertEqual(config['image_size'], [28, 42])
        self.assertEqual(provenance['image_preprocessing'],
                         models.input_preprocessing((28, 42)))
        descriptor, aux = teacher(torch.randn(1, 3, 28, 42))
        self.assertEqual(descriptor.shape[0], 1)
        self.assertEqual(aux['log_assignment'].shape, (1, 3, 6))
        # Native metadata remains authoritative; adapting legacy raw resolution
        # must not silently override an explicitly saved preprocessing contract.
        with self.assertRaisesRegex(ValueError, 'preprocessing|image_size'):
            self.teacher(self.teacher_path, config=rectangular)

    def test_registered_raw_still_checks_architecture_and_cluster_dimensions(self):
        different_backbone = copy.deepcopy(self.config)
        different_backbone['backbone_arch'] = 'dinov2_vitb14'
        different_backbone['agg_config']['num_channels'] = 768
        with self.registration(), self.assertRaisesRegex(ValueError, 'backbone architectures'):
            models.load_teacher_checkpoint(self.raw, student_config=different_backbone)
        for dimension in ('num_clusters', 'cluster_dim', 'token_dim'):
            incompatible = copy.deepcopy(self.config)
            incompatible['agg_config'][dimension] += 1
            with self.subTest(dimension=dimension), self.registration(), \
                    self.assertRaisesRegex(ValueError, dimension):
                models.load_teacher_checkpoint(self.raw, student_config=incompatible)

    def test_runner_preparation_uses_the_same_teacher_loader(self):
        class LoaderReached(Exception):
            pass

        config = {'checkpoint': str(self.teacher_path), 'teacher_checkpoint': str(self.raw)}
        with patch('workflow.model.load_teacher_checkpoint', side_effect=LoaderReached) as load, \
                patch('workflow.training_data.MixedGSVCitiesDataset') as dataset, \
                self.assertRaises(LoaderReached):
            feasibility.prepare(config, self.root / 'not_created')
        load.assert_called_once()
        dataset.assert_not_called()
        args, kwargs = load.call_args
        self.assertEqual(Path(args[0]), self.raw)
        student_config = kwargs.get('student_config', args[1] if len(args) > 1 else None)
        self.assertIsNotNone(student_config)
        self.assertEqual(student_config['image_size'], [224, 224])


if __name__ == '__main__':
    unittest.main()
