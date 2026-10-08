"""CPU-only queued generation gates, complete training proof and stage aggregation."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

ADAPTVPR_ROOT = Path(__file__).resolve().parents[3]
if str(ADAPTVPR_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTVPR_ROOT))
from experiments.qwen_curriculum import campaign, common, recover_interrupted


ARMS = {'generated_8to1', 'true_8to1', 'generated_4to1', 'true_4to1'}


def seal(row, field='fingerprint'):
    body = {key: value for key, value in row.items() if key != field}
    return {**body, field: common.fingerprint(body)}


class Waiting(Exception):
    """End a mocked sleep without making an active wait look complete."""


class CampaignTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.training = self.root / 'training'
        self.training.mkdir()
        self.output = self.root / 'campaign'
        self.output.mkdir()
        self.training_config = seal({'protocol': 'four fixed arms then all evaluations'})
        common.write_json(self.training / 'experiment_config.json', self.training_config)
        self.args = SimpleNamespace(output_dir=self.output, training_dir=self.training,
                                    training_unit='training.service')
        self.coordinator = {'training_fingerprint': self.training_config['fingerprint'],
                            'implementation_sha256': {}}
        self.stages = []
        self.jobs = []
        self.saved = []
        for stage_index in range(2):
            directory = self.root / f'original_stage_{stage_index}'
            directory.mkdir()
            jobs, generated, results, attempts = [], [], [], []
            for index in range(2):
                serial = stage_index * 2 + index
                source = self.root / f'source_{serial}.png'
                output = directory / f'original_output_{serial}.png'
                raw = directory / f'original_raw_{serial}.png'
                Image.new('RGB', (20, 16), (10 + serial, 40, 60)).save(source)
                Image.new('RGB', (20, 16), (50 + serial, 80, 100)).save(output)
                Image.new('RGB', (30, 24), (50 + serial, 80, 100)).save(raw)
                job = seal({'sample_id': f'qwen_{serial:024x}', 'source_path': str(source),
                    'source_sha256': common.file_sha256(source), 'source_dimensions': [20, 16],
                    'city': 'City', 'place_id': serial, 'seed': serial + 1,
                    'condition': 'night' if index == 0 else 'snow',
                    'prompt': 'conservative weather edit', 'negative_prompt': ''}, 'record_sha256')
                jobs.append(job)
            common.write_jsonl(directory / 'plan.jsonl', jobs)
            plan = seal({'num_images': 2, 'plan_sha256': common.file_sha256(directory / 'plan.jsonl'),
                         'implementation_sha256': {}})
            common.write_json(directory / 'plan_config.json', plan)
            execution = seal({'stage': 'generate', 'plan_fingerprint': plan['fingerprint'], 'max_calls': 2,
                'implementation_sha256': {}, 'qwen_url': 'http://127.0.0.1:8001/generate',
                'request_timeout': 600, 'matcher_device': 'cpu', 'cpu_threads': 1})
            common.write_json(directory / 'execution_config.json', execution)
            for index, job in enumerate(jobs):
                serial = stage_index * 2 + index
                output = directory / f'original_output_{serial}.png'
                raw = directory / f'original_raw_{serial}.png'
                row = seal({**job, 'execution_fingerprint': execution['fingerprint'],
                    'output_path': str(output), 'output_sha256': common.file_sha256(output),
                    'raw_output_path': str(raw), 'raw_output_sha256': common.file_sha256(raw)}, 'result_sha256')
                generated.append(row)
                results.append(seal({**row, 'status': 'passed' if index == 0 else 'rejected',
                    'passed': index == 0, 'eligible_for_training': index == 0}, 'result_sha256'))
                attempts.append(seal({'sample_id': job['sample_id'], 'call_number': index + 1,
                    'record_sha256': job['record_sha256'], 'execution_fingerprint': execution['fingerprint']},
                    'attempt_sha256'))
            common.write_jsonl(directory / 'generated.jsonl', generated)
            common.write_jsonl(directory / 'results.jsonl', results)
            common.write_jsonl(directory / 'attempts.jsonl', attempts)
            common.write_json(directory / 'summary.json', {'state': 'complete', 'generation_errors': 0})
            self.stages.append({'run_dir': str(directory), 'num_images': 2, 'max_calls': 2,
                                'plan_fingerprint': plan['fingerprint'], 'plan_sha256': plan['plan_sha256']})
            self.jobs.extend(jobs)
            self.saved.extend(results)
        self.config = seal({'total_images': 4, 'parent_run_dir': self.stages[0]['run_dir'],
                            'stages': self.stages})
        self.full_report()

    def full_report(self):
        checkpoints = {}
        for arm in ARMS:
            directory = self.training / arm
            directory.mkdir(exist_ok=True)
            checkpoint = directory / 'checkpoint.pt'
            checkpoint.write_bytes(f'final checkpoint {arm}'.encode())
            checkpoints[arm] = common.file_sha256(checkpoint)
        self.report = seal({'state': 'complete', 'experiment_fingerprint': self.training_config['fingerprint'],
                            'checkpoint_sha256': checkpoints})
        common.write_json(self.training / 'comparison.json', self.report)
        common.write_json(self.training / 'progress.json', {'stage': 'complete'})

    def finish_proof(self):
        return campaign.training_finished(self.training, self.training_config['fingerprint'])

    def wait(self, state='inactive', code=0):
        with patch.object(campaign, 'unit_state', return_value=state), \
             patch.object(campaign, 'unit_exit_code', return_value=code), redirect_stdout(io.StringIO()):
            return campaign.wait_for_training(self.args, self.config, self.coordinator)

    def test_complete_training_requires_all_four_unchanged_checkpoints_and_progress(self):
        self.assertTrue(self.finish_proof())
        self.wait()
        common.write_json(self.training / 'progress.json', {'stage': 'evaluation', 'arm': 'true_4to1'})
        self.assertFalse(self.finish_proof())
        with self.assertRaisesRegex(RuntimeError, 'before full completion'):
            self.wait()

    def test_missing_completion_report_keeps_generation_stopped(self):
        (self.training / 'comparison.json').unlink()
        self.assertFalse(self.finish_proof())
        with self.assertRaisesRegex(RuntimeError, 'before full completion'):
            self.wait()

    def test_partial_arm_report_and_changed_checkpoint_are_rejected(self):
        report = dict(self.report)
        report['checkpoint_sha256'] = dict(report['checkpoint_sha256'])
        report['checkpoint_sha256'].pop('true_4to1')
        common.write_json(self.training / 'comparison.json', seal(report))
        with self.assertRaisesRegex(ValueError, 'all four training arms'):
            self.finish_proof()
        common.write_json(self.training / 'comparison.json', self.report)
        (self.training / 'true_4to1/checkpoint.pt').write_bytes(b'changed checkpoint')
        with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
            self.finish_proof()

    def test_completion_report_seal_and_experiment_identity_are_checked(self):
        report = dict(self.report)
        report['state'] = 'incomplete'
        common.write_json(self.training / 'comparison.json', report)
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.finish_proof()
        report = seal({**self.report, 'experiment_fingerprint': 'another experiment'})
        common.write_json(self.training / 'comparison.json', report)
        with self.assertRaisesRegex(ValueError, 'another experiment'):
            self.finish_proof()

    def test_unsuccessful_training_exit_stops_even_with_complete_report(self):
        with self.assertRaisesRegex(RuntimeError, 'exited unsuccessfully'):
            self.wait(state='failed', code=1)
        with self.assertRaisesRegex(RuntimeError, 'exited unsuccessfully'):
            self.wait(state='inactive', code=2)

    def test_changed_training_config_is_rejected_during_wait(self):
        common.write_json(self.training / 'experiment_config.json', seal({'protocol': 'changed'}))
        with self.assertRaisesRegex(ValueError, 'experiment changed'):
            self.wait(state='active')

    def test_active_training_never_recovers_starts_qwen_or_launches_child(self):
        # Even an existing complete report does not release an active training unit.
        with ExitStack() as stack:
            stack.enter_context(patch.object(campaign, 'freeze_coordinator', return_value=self.coordinator))
            stack.enter_context(patch.object(campaign, 'check_frozen_artifacts'))
            stack.enter_context(patch.object(campaign, 'unit_state', return_value='active'))
            exit_code = stack.enter_context(patch.object(campaign, 'unit_exit_code'))
            stack.enter_context(patch.object(campaign.time, 'sleep', side_effect=Waiting))
            recover = stack.enter_context(patch.object(recover_interrupted, 'recover_interrupted'))
            qwen = stack.enter_context(patch.object(campaign, 'start_qwen'))
            child = stack.enter_context(patch.object(campaign.subprocess, 'Popen'))
            stack.enter_context(redirect_stdout(io.StringIO()))
            with self.assertRaises(Waiting):
                campaign.execute(self.args, self.config)
        recover.assert_not_called()
        qwen.assert_not_called()
        child.assert_not_called()
        exit_code.assert_not_called()
        summary = json.loads((self.output / 'summary.json').read_text())
        self.assertEqual(summary['state'], 'waiting_for_training')
        self.assertEqual(summary['generated_images'], 4)

    def test_wait_unlocks_only_after_unit_stops_with_valid_full_report(self):
        with patch.object(campaign, 'unit_state', side_effect=['active', 'inactive']), \
             patch.object(campaign, 'unit_exit_code', return_value=0), \
             patch.object(campaign.time, 'sleep') as sleep, redirect_stdout(io.StringIO()):
            campaign.wait_for_training(self.args, self.config, self.coordinator)
        sleep.assert_called_once_with(60)

    def test_aggregate_retains_original_paths_exact_records_and_counts(self):
        stage_bytes = {Path(stage['run_dir']) / name: (Path(stage['run_dir']) / name).read_bytes()
                       for stage in self.stages for name in ('results.jsonl', 'generated.jsonl', 'attempts.jsonl')}
        original_images = {Path(row['output_path']): Path(row['output_path']).read_bytes() for row in self.saved}
        frozen_provenance = b'{"immutable_stage_plan": true}\n'
        (self.output / 'stage_provenance.json').write_bytes(frozen_provenance)
        summary = campaign.refresh(self.output, self.config, 'waiting_for_training')
        self.assertEqual((self.output / 'stage_provenance.json').read_bytes(), frozen_provenance)
        self.assertEqual(summary['target_images'], 4)
        self.assertEqual(summary['generated_images'], 4)
        self.assertEqual(summary['quality_evaluated'], 4)
        self.assertEqual(summary['calls_reserved'], 4)
        self.assertEqual(summary['accepted'], 2)
        self.assertEqual(summary['rejected'], 2)
        self.assertEqual(summary['generation_errors'], 0)
        self.assertEqual(summary['new_images_remaining'], 0)
        self.assertEqual(summary['conditions_generated'], {'night': 2, 'snow': 2})
        combined = common.read_jsonl(self.output / 'results.jsonl')
        self.assertEqual(combined, self.saved)
        self.assertEqual([row['output_path'] for row in combined], [row['output_path'] for row in self.saved])
        accepted = common.read_jsonl(self.output / 'training_manifest.jsonl')
        self.assertEqual(accepted, [row for row in self.saved if row['passed'] and row['eligible_for_training']])
        provenance = json.loads((self.output / 'runtime_stage_provenance.json').read_text())
        self.assertEqual([row['results_offset'] for row in provenance], [0, 2])
        self.assertEqual([row['results_count'] for row in provenance], [2, 2])
        self.assertFalse(list(self.output.rglob('*.png')))
        for path, expected in {**stage_bytes, **original_images}.items():
            self.assertEqual(path.read_bytes(), expected)

    def test_partial_aggregate_counts_reserved_pending_work(self):
        directory = Path(self.stages[1]['run_dir'])
        for name in ('generated.jsonl', 'results.jsonl'):
            common.write_jsonl(directory / name, common.read_jsonl(directory / name)[:1])
        summary = campaign.refresh(self.output, self.config, 'waiting_for_training')
        self.assertEqual(summary['generated_images'], 3)
        self.assertEqual(summary['quality_evaluated'], 3)
        self.assertEqual(summary['calls_reserved'], 4)
        self.assertEqual(summary['new_images_remaining'], 1)

    def test_duplicate_result_generated_id_or_source_content_is_refused(self):
        directory = Path(self.stages[1]['run_dir'])
        original_results = (directory / 'results.jsonl').read_bytes()
        original_generated = (directory / 'generated.jsonl').read_bytes()
        first_directory = Path(self.stages[0]['run_dir'])
        first_result = common.read_jsonl(first_directory / 'results.jsonl')[0]
        first_generated = common.read_jsonl(first_directory / 'generated.jsonl')[0]
        rows = common.read_jsonl(directory / 'results.jsonl')
        rows[0] = first_result
        common.write_jsonl(directory / 'results.jsonl', rows)
        with self.assertRaisesRegex(ValueError, 'Duplicate result'):
            campaign.refresh(self.output, self.config, 'running')
        (directory / 'results.jsonl').write_bytes(original_results)
        for field in ('sample_id', 'source_sha256'):
            rows = common.read_jsonl(directory / 'generated.jsonl')
            rows[0][field] = first_generated[field]
            common.write_jsonl(directory / 'generated.jsonl', rows)
            with self.assertRaisesRegex(ValueError, 'Duplicate generated source'):
                campaign.refresh(self.output, self.config, 'running')
            (directory / 'generated.jsonl').write_bytes(original_generated)
        self.assertFalse((self.output / 'summary.json').exists())

    def test_final_completion_validates_real_job_seals_images_and_ledger(self):
        campaign.validate_completed_campaign(self.config)
        directory = Path(self.stages[1]['run_dir'])
        original = (directory / 'generated.jsonl').read_bytes()
        common.write_jsonl(directory / 'generated.jsonl', common.read_jsonl(directory / 'generated.jsonl')[:1])
        with self.assertRaisesRegex(ValueError, 'Not every planned source'):
            campaign.validate_completed_campaign(self.config)
        (directory / 'generated.jsonl').write_bytes(original)
        ledger = (directory / 'attempts.jsonl').read_bytes()
        common.write_jsonl(directory / 'attempts.jsonl', common.read_jsonl(directory / 'attempts.jsonl')[:1])
        with self.assertRaisesRegex(ValueError, 'call ledger differs'):
            campaign.validate_completed_campaign(self.config)
        (directory / 'attempts.jsonl').write_bytes(ledger)
        Path(self.saved[-1]['output_path']).write_bytes(b'changed original stage image')
        with self.assertRaisesRegex(ValueError, 'missing or changed'):
            campaign.validate_completed_campaign(self.config)

    def test_execute_rechecks_artifacts_after_wait_before_recovery(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(campaign, 'freeze_coordinator', return_value=self.coordinator))
            check = stack.enter_context(patch.object(campaign, 'check_frozen_artifacts',
                                                    side_effect=[None, ValueError('changed while queued')]))
            stack.enter_context(patch.object(campaign, 'wait_for_training'))
            recover = stack.enter_context(patch.object(recover_interrupted, 'recover_interrupted'))
            qwen = stack.enter_context(patch.object(campaign, 'start_qwen'))
            with self.assertRaisesRegex(ValueError, 'changed while queued'):
                campaign.execute(self.args, self.config)
        self.assertEqual(check.call_count, 2)
        recover.assert_not_called()
        qwen.assert_not_called()

    def test_stage_subprocess_failure_never_publishes_campaign_complete(self):
        child = Mock(returncode=1)
        child.poll.return_value = 1
        with ExitStack() as stack:
            stack.enter_context(patch.object(campaign, 'freeze_coordinator', return_value=self.coordinator))
            stack.enter_context(patch.object(campaign, 'check_frozen_artifacts'))
            stack.enter_context(patch.object(campaign, 'wait_for_training'))
            stack.enter_context(patch.object(recover_interrupted, 'recover_interrupted', return_value={}))
            stack.enter_context(patch.object(campaign, 'start_qwen'))
            stack.enter_context(patch.object(campaign.subprocess, 'Popen', return_value=child))
            stack.enter_context(redirect_stdout(io.StringIO()))
            with self.assertRaisesRegex(RuntimeError, 'Generation stage 1 stopped'):
                campaign.execute(self.args, self.config)
        self.assertNotEqual(json.loads((self.output / 'summary.json').read_text())['state'], 'complete')

    def test_complete_execution_stops_qwen_only_after_exact_total_validation(self):
        child = Mock(returncode=0)
        child.poll.return_value = 0
        with ExitStack() as stack:
            stack.enter_context(patch.object(campaign, 'freeze_coordinator', return_value=self.coordinator))
            checks = stack.enter_context(patch.object(campaign, 'check_frozen_artifacts'))
            wait = stack.enter_context(patch.object(campaign, 'wait_for_training'))
            recover = stack.enter_context(patch.object(recover_interrupted, 'recover_interrupted', return_value={}))
            qwen = stack.enter_context(patch.object(campaign, 'start_qwen'))
            children = stack.enter_context(patch.object(campaign.subprocess, 'Popen', return_value=child))
            systemctl = stack.enter_context(patch.object(campaign.subprocess, 'run'))
            stack.enter_context(redirect_stdout(io.StringIO()))
            campaign.execute(self.args, self.config)
        wait.assert_called_once()
        self.assertEqual(checks.call_count, 3)
        self.assertEqual(recover.call_count, 2)
        qwen.assert_called_once()
        self.assertEqual(children.call_count, 2)
        self.assertEqual([call.args[0][call.args[0].index('--run-dir') + 1] for call in children.call_args_list],
                         [stage['run_dir'] for stage in self.stages])
        systemctl.assert_called_once_with(['systemctl', '--user', 'stop', 'adaptvpr-lightx2v.service'], check=True)
        summary = json.loads((self.output / 'summary.json').read_text())
        self.assertEqual(summary['state'], 'complete')
        self.assertEqual(summary['generated_images'], 4)
        self.assertEqual(summary['quality_evaluated'], 4)

    def test_campaign_count_mismatch_cannot_publish_complete(self):
        wrong_target = seal({**self.config, 'total_images': 5})
        child = Mock(returncode=0)
        child.poll.return_value = 0
        with ExitStack() as stack:
            stack.enter_context(patch.object(campaign, 'freeze_coordinator', return_value=self.coordinator))
            stack.enter_context(patch.object(campaign, 'check_frozen_artifacts'))
            stack.enter_context(patch.object(campaign, 'wait_for_training'))
            stack.enter_context(patch.object(recover_interrupted, 'recover_interrupted', return_value={}))
            stack.enter_context(patch.object(campaign, 'start_qwen'))
            stack.enter_context(patch.object(campaign.subprocess, 'Popen', return_value=child))
            stack.enter_context(redirect_stdout(io.StringIO()))
            with self.assertRaisesRegex(ValueError, 'cumulative target'):
                campaign.execute(self.args, wrong_target)
        self.assertNotEqual(json.loads((self.output / 'summary.json').read_text())['state'], 'complete')



if __name__ == '__main__':
    unittest.main()
