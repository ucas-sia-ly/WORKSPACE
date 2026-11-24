"""Pure-real baseline split, objective, selection and independent confirmation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_real_baseline as baseline
from train_salad import parse_args
import test_training_data as fixture


class RealBaselineTests(unittest.TestCase):
    _write_csv = fixture.TrainingDataTests._write_csv

    def setUp(self):
        fixture.TrainingDataTests.setUp(self)
        self.addCleanup(self.temporary.cleanup)
        self.config = baseline.load_config(baseline.ccr.ROOT / 'configs/real_baseline.json')
        self.checkpoint = self.root / 'initial.ckpt'
        self.checkpoint.write_bytes(b'initial model fixture; prepare does not load models')
        self.split = self.root / 'reserved.json'
        self.split.write_text(json.dumps(dict(version=1, groups=dict(validation=[['Alpha', 1]], calibration=[['Beta', 1]]))))
        self.out = self.root / 'baseline'
        self.config.update(real_data=str(self.gsv), checkpoint=str(self.checkpoint),
                           reserved_places=str(self.split), output_dir=str(self.out))
        with patch.object(baseline, 'validation_inventory', return_value={'fixture': True}):
            self.plan = baseline.prepare(self.config, self.out)

    def metrics(self, **changes):
        result = dict(epoch=1, global_step=1, batches=1, loss=.1,
            real_exposure=self.plan['real_exposures_per_epoch'], synthetic_exposure=0,
            mining=dict(active_anchor_fraction=.12, no_mined_pair_batch_fraction=.01,
                        real_real_mined_positive=12, real_real_mined_negative=25, clip_batch_fraction=0))
        result.update(changes)
        return result

    def evaluation(self, directory, value=.96, checkpoint=None):
        directory.mkdir(parents=True, exist_ok=True)
        result = dict(recall={'R@1': value, 'R@5': .98, 'R@10': .99},
                      query_results=[dict(query_id='one', rank=1), dict(query_id='two', rank=2)])
        path = directory / 'svox_val.json'
        baseline.ccr.write_json(path, result)
        baseline.ccr.write_json(path.with_suffix('.receipt.json'), dict(
            checkpoint_sha256=baseline.ccr.digest(checkpoint or self.checkpoint),
            result_sha256=baseline.ccr.digest(path), protocol=baseline.evaluation_plan(self.plan)['evaluations'][0]))

    def completed_run(self, directory, epoch, value=.96, **changes):
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint = directory / 'checkpoint.pt'
        checkpoint.write_bytes(b'run fixture')
        for n in range(1, epoch + 1):
            baseline.ccr.write_json(directory / f'metrics_epoch_{n:03d}.json', self.metrics(epoch=n, **changes))
        self.evaluation(directory / 'evaluations' / str(epoch), value, checkpoint)

    def test_full_pool_excludes_every_view_of_reserved_places_without_synthetic_dependency(self):
        self.assertEqual(self.plan['train_places'], 2)
        self.assertEqual(self.plan['train_real_images'], 8)
        self.assertEqual(self.plan['real_exposures_per_epoch'], 8)
        report = json.loads((self.out / 'exclusion_report.json').read_text())
        self.assertEqual(report['counts_by_reason'], {'calibration': 1, 'validation': 1})
        self.assertTrue(all(row['real_images'] == 4 for row in report['records']))
        self.assertFalse(any('synthetic' in key for key in self.plan['input_hashes']))

    def test_duplicate_unknown_or_overlapping_reserved_identities_are_rejected(self):
        for groups in (dict(validation=[['Alpha', 1]], calibration=[['Alpha', 1]]),
                       dict(validation=[['Alpha', True]], calibration=[])):
            with self.assertRaises(ValueError):
                baseline.reserved_identities(dict(version=1, groups=groups))
        self.split.write_text(json.dumps(dict(version=1, groups=dict(validation=[['missing', 1]], calibration=[]))))
        with self.assertRaisesRegex(ValueError, 'missing from GSV'):
            baseline.prepare(self.config, self.root / 'bad')

    def test_commands_are_pure_real_and_resume_keeps_the_full_cosine_schedule(self):
        for candidate in self.config['candidates']:
            directory = self.out / 'screen' / candidate['name']
            command = baseline.training_command(self.plan, candidate, 42, directory, 1)
            args = parse_args(command[2:])
            self.assertIsNone(args.synthetic_manifest)
            self.assertIsNone(args.teacher_checkpoint)
            self.assertFalse(args.synthetic_aux or args.reliability_ot or args.synthetic_places_only)
            self.assertEqual(args.synthetic_fraction, 0)
            self.assertEqual(args.epochs, 3)
            self.assertEqual(args.stop_after_epoch, 1)
            self.assertEqual(args.max_batches_per_epoch, 0)
            self.assertEqual(args.max_steps, 0)
            self.assertEqual(args.miner_margin, .1)
            self.assertTrue(args.log_mined_pairs)
            resumed = parse_args(baseline.training_command(self.plan, candidate, 43, directory, 2, True)[2:])
            self.assertIsNotNone(resumed.resume)
            self.assertEqual(resumed.epochs, args.epochs)

    def test_signal_checks_do_not_equate_nonzero_loss_with_sufficient_pairs(self):
        self.assertEqual(baseline.signal_failures(self.metrics(), self.plan), [])
        bad = self.metrics()
        bad['mining'].update(active_anchor_fraction=.001, no_mined_pair_batch_fraction=.8)
        failures = baseline.signal_failures(bad, self.plan)
        self.assertIn('insufficient_active_anchors', failures)
        self.assertIn('too_many_empty_mining_batches', failures)
        self.assertTrue(baseline.signal_failures(self.metrics(synthetic_exposure=1), self.plan))
        self.assertTrue(baseline.signal_failures(self.metrics(real_exposure=4), self.plan))
        self.assertTrue(baseline.signal_failures(self.metrics(loss=float('nan')), self.plan))

    def test_candidate_selection_waits_for_all_and_never_reads_weather_test(self):
        rows = [dict(candidate=c['name'], recall={'R@1': .96 + i / 10000}, failures=[])
                for i, c in enumerate(self.config['candidates'])]
        self.assertIsNone(baseline.select_candidate(self.plan, rows[:-1]))
        rows[-1]['failures'] = ['too_many_empty_mining_batches']
        self.assertEqual(baseline.select_candidate(self.plan, rows), rows[-2]['candidate'])
        self.assertTrue(all(e['split'] == 'val' for e in baseline.evaluation_plan(self.plan)['evaluations']))
        self.assertEqual({e['name'] for e in baseline.evaluation_plan(self.plan, True)['evaluations']},
                         {'svox_default', 'svox_night', 'svox_rain', 'svox_snow', 'svox_sun', 'svox_overcast'})

    def test_baseline_requires_all_independent_seeds_and_fixed_final_epoch(self):
        self.evaluation(self.out / 'initial')
        for c in self.config['candidates']:
            self.completed_run(self.out / 'screen' / c['name'], 1)
        selected = self.config['candidates'][0]['name']
        for seed in self.config['confirmation_seeds'][:-1]:
            self.completed_run(self.out / 'confirm' / selected / str(seed), 3)
        report = baseline.summarize(self.plan, self.out)
        self.assertEqual(report['status'], 'confirmation_required')
        self.assertFalse((self.out / 'baseline.json').exists())
        self.completed_run(self.out / 'confirm' / selected / str(self.config['confirmation_seeds'][-1]), 3)
        report = baseline.summarize(self.plan, self.out)
        self.assertEqual(report['status'], 'passed_operational_checks')
        self.assertEqual(report['seed_std_pp'], 0)
        saved = json.loads((self.out / 'baseline.json').read_text())
        self.assertEqual(set(saved['checkpoint_by_seed']), {'43', '44', '45'})
        self.assertTrue(all(row['epoch'] == 3 for row in report['confirmation']))

    def test_confirmed_seeds_can_fail_despite_high_mean_or_lower_training_loss(self):
        self.evaluation(self.out / 'initial')
        for c in self.config['candidates']:
            self.completed_run(self.out / 'screen' / c['name'], 1)
        selected = self.config['candidates'][0]['name']
        for seed, value in zip(self.config['confirmation_seeds'], (.959, .960, .970)):
            self.completed_run(self.out / 'confirm' / selected / str(seed), 3, value, loss=.01)
        self.assertEqual(baseline.summarize(self.plan, self.out)['status'], 'baseline_not_stable')
        self.assertFalse((self.out / 'baseline.json').exists())

    def test_resumed_plan_and_evaluation_receipts_reject_mutation(self):
        with patch.object(baseline, 'validation_inventory', return_value={'fixture': True}):
            baseline.validate_plan(self.plan)
        self.evaluation(self.out / 'initial')
        candidate = self.config['candidates'][0]
        directory = self.out / 'screen' / candidate['name']
        self.completed_run(directory, 1)
        (directory / 'checkpoint.pt').write_bytes(b'wrong checkpoint')
        with self.assertRaisesRegex(ValueError, 'receipt/checkpoint mismatch'):
            baseline.summarize(self.plan, self.out)
        self.split.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'input_hashes changed'):
            baseline.validate_plan(self.plan)

    def test_interrupted_initial_evaluation_without_receipt_remains_resumable(self):
        self.evaluation(self.out / 'initial')
        (self.out / 'initial/svox_val.receipt.json').unlink()
        self.assertEqual(baseline.summarize(self.plan, self.out)['status'], 'prepared_not_trained')

    def test_checkpoint_objective_schedule_and_initialization_must_match(self):
        candidate = self.config['candidates'][0]
        training = dict(synthetic_manifest=None, synthetic_fraction=0, synthetic_places_only=False,
            batch_size=candidate['batch_size'], learning_rate=candidate['learning_rate'],
            weight_decay=0, miner_margin=.1, real_sampling='coverage_pk', max_batches_per_epoch=0,
            seed=43, augment=True, images_per_place=4, real_data=self.config['real_data'],
            model_config=dict(backbone_config=dict(num_trainable_blocks=2), agg_config={}),
            init_checkpoint_sha256=self.plan['input_hashes'][self.config['checkpoint']],
            excluded_places=dict(sha256=self.plan['input_hashes'][self.plan['excluded_places']]))
        saved=dict(format_version=1, optimizer_state_dict={}, training_config=training,
                   epoch=1, global_step=1, total_epochs=3, precision='bf16')
        baseline.checked_state(saved, self.plan, candidate, 43)
        for key, value in (('synthetic_manifest', '/anything'), ('teacher', {}), ('batch_size', 2)):
            altered=copy.deepcopy(saved)
            altered['training_config'][key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):
                baseline.checked_state(altered, self.plan, candidate, 43)


class RealBaselineResumeTests(unittest.TestCase):
    def test_cpu_epoch_resume_matches_uninterrupted_pure_real_training(self):
        import contextlib
        import io
        import torch
        import test_frozen_teacher as teacher_fixture
        from train_salad import train
        from workflow.model import read_checkpoint

        teacher_fixture.FrozenTeacherTests.setUp(self)
        config = baseline.load_config(baseline.ccr.ROOT / 'configs/real_baseline.json')
        config.update(real_data=str(self.data), checkpoint=str(self.teacher_path), backbone_repo=str(self.repo),
                      device='cpu', precision='32', num_workers=0, images_per_place=2, epochs=2)
        excluded = self.root / 'reserved.json'
        excluded.write_text('[]')
        candidate = dict(name='tiny', batch_size=2, learning_rate=1e-5, num_trainable_blocks=1)
        config['candidates'] = [candidate]
        plan = dict(config=config, excluded_places=str(excluded), steps_per_epoch={'tiny': 1},
                    input_hashes={str(self.teacher_path): baseline.ccr.digest(self.teacher_path),
                                  str(excluded): baseline.ccr.digest(excluded)})
        calls = []
        def child(command, *args, **kwargs):
            calls.append(command)
            train(parse_args(command[2:]))
        directory = self.root / 'resumed'
        with contextlib.redirect_stdout(io.StringIO()), patch.object(baseline.ccr, 'run_child', side_effect=child), \
                patch.object(baseline.ccr, 'evaluate_checkpoint'):
            baseline.run_candidate(plan, candidate, 43, directory, 1)
            baseline.run_candidate(plan, candidate, 43, directory, 2)
            full = self.root / 'full'
            train(parse_args(baseline.training_command(plan, candidate, 43, full, 2)[2:]))
        self.assertEqual(len(calls), 2)
        self.assertNotIn('--resume', calls[0])
        self.assertIn('--resume', calls[1])
        resumed, expected = read_checkpoint(directory / 'checkpoint.pt'), read_checkpoint(full / 'checkpoint.pt')
        for name, value in expected['state_dict'].items():
            torch.testing.assert_close(value, resumed['state_dict'][name], rtol=0, atol=0)
        self.assertIsNone(resumed['training_config']['synthetic_manifest'])
        self.assertNotIn('teacher', resumed['training_config'])
        self.assertEqual(resumed['metrics']['synthetic_exposure'], 0)
        self.assertEqual(resumed['global_step'], 2)


if __name__ == '__main__':
    unittest.main()
