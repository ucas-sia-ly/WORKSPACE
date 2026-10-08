"""CPU-only interrupted-output recovery, provenance and immutable-record checks."""
import fcntl
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

ADAPTVPR_ROOT = Path(__file__).resolve().parents[3]
if str(ADAPTVPR_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTVPR_ROOT))
from experiments.qwen_curriculum import common, recover_interrupted as recovery


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / 'run'
        self.directory.mkdir()
        self.service = self.root / 'service'
        self.service.mkdir()
        self.code = self.root / 'code'
        self.code.mkdir()
        (self.code / 'frozen.py').write_text('frozen implementation\n')
        self.addCleanup(patch.stopall)
        patch.object(common, 'ADAPTVPR_ROOT', self.code).start()
        self.jobs = []
        for index in range(2):
            source = self.root / f'source{index}.png'
            Image.new('RGB', (20, 16), (25 + index, 80, 45)).save(source)
            job = {'sample_id': 'qwen_' + common.fingerprint(index)[:24],
                   'source_path': str(source), 'source_sha256': common.file_sha256(source),
                   'source_dimensions': [20, 16], 'condition': 'night',
                   'city': 'City', 'place_id': index, 'seed': 123 + index,
                   'prompt': 'night conservative edit', 'negative_prompt': ''}
            self.jobs.append(recovery._sealed(job, 'record_sha256'))
        common.write_jsonl(self.directory / 'plan.jsonl', self.jobs)
        plan_config = recovery._sealed({'num_images': 2,
            'plan_sha256': common.file_sha256(self.directory / 'plan.jsonl')}, 'fingerprint')
        common.write_json(self.directory / 'plan_config.json', plan_config)
        self.config = recovery._sealed({'stage': 'generate', 'run_dir': str(self.directory),
            'plan_fingerprint': plan_config['fingerprint'], 'max_calls': 2,
            'implementation_sha256': {'frozen.py': common.file_sha256(self.code / 'frozen.py')},
            'service': {'model_id': 'Qwen/Qwen-Image-Edit-2511', 'source_modified': False,
                        'canvas_policy': 'source_aspect_v1', 'sampling': {'infer_steps': 4, 'guidance_scale': 1.0},
                        'canvas': {'target_shape_order': 'height,width', 'target_pixels': 720, 'multiple': 1,
                                   'min_side': 1, 'max_side': 128, 'rounding': 'nearest_multiple',
                                   'unsupported_aspect_ratio': 'reject'}}}, 'fingerprint')
        common.write_json(self.directory / 'execution_config.json', self.config)
        attempts = [recovery._sealed({'call_number': index + 1, 'sample_id': job['sample_id'],
            'record_sha256': job['record_sha256'], 'execution_fingerprint': self.config['fingerprint']},
            'attempt_sha256') for index, job in enumerate(self.jobs)]
        common.write_jsonl(self.directory / 'attempts.jsonl', attempts)
        (self.directory / 'images').mkdir()
        (self.directory / 'raw').mkdir()
        old_image = self.directory / 'images' / f"{self.jobs[0]['sample_id']}.png"
        old_raw = self.directory / 'raw' / f"{self.jobs[0]['sample_id']}.png"
        Image.new('RGB', (20, 16), (60, 70, 80)).save(old_image)
        Image.new('RGB', (30, 24), (60, 70, 80)).save(old_raw)
        first = recovery._sealed({**self.jobs[0], 'execution_fingerprint': self.config['fingerprint'],
            'output_path': str(old_image), 'output_sha256': common.file_sha256(old_image),
            'raw_output_path': str(old_raw), 'raw_output_sha256': common.file_sha256(old_raw)}, 'result_sha256')
        common.write_jsonl(self.directory / 'generated.jsonl', [first])
        common.write_jsonl(self.directory / 'results.jsonl', [recovery._sealed({**first,
            'status': 'passed', 'passed': True, 'eligible_for_training': True}, 'result_sha256')])
        self.raw, self.metadata = self.make_output('orphan')
        self.immutable = {name: (self.directory / name).read_bytes() for name in
            ('results.jsonl', 'attempts.jsonl', 'execution_config.json', 'plan.jsonl', 'plan_config.json')}

    def make_output(self, name):
        raw = self.service / f'{name}.png'
        Image.new('RGB', (30, 24), (80, 90, 100)).save(raw)
        metadata = raw.with_suffix('.json')
        common.write_json(metadata, {'source_path': str(self.directory / 'source_inputs' / f"{self.jobs[1]['sample_id']}.png"),
            'seed': self.jobs[1]['seed'], 'result_path': str(raw), 'metadata_path': str(metadata),
            'canvas_policy': 'source_aspect_v1', 'source_dimensions': [20, 16],
            'raw_dimensions': [30, 24], 'target_shape': [24, 30]})
        return raw, metadata

    def run_recovery(self, apply=False):
        return recovery.recover_interrupted(self.directory, self.service, apply=apply)

    def assert_immutable(self):
        for name, expected in self.immutable.items():
            self.assertEqual((self.directory / name).read_bytes(), expected, name)

    def test_audit_is_read_only_and_reports_exact_orphan(self):
        before = {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        report = self.run_recovery()
        self.assertEqual(report['recoverable'], [self.jobs[1]['sample_id']])
        self.assertEqual(report['http_calls'], 0)
        self.assertEqual(report['quality_checks'], 0)
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_recovery_appends_only_unverified_generated_row(self):
        raw_bytes, metadata_bytes = self.raw.read_bytes(), self.metadata.read_bytes()
        report = self.run_recovery(True)
        self.assertEqual(report['recovered'], [self.jobs[1]['sample_id']])
        rows = common.read_jsonl(self.directory / 'generated.jsonl')
        self.assertEqual(len(rows), 2)
        row = rows[-1]
        self.assertEqual(row['execution_fingerprint'], self.config['fingerprint'])
        recovery._check_seal(row, 'result_sha256')
        self.assertNotIn('passed', row)
        self.assertNotIn('sampling', row)
        self.assertNotIn('generation_seconds', row)
        self.assertEqual(row['recovery']['sidecar_does_not_prove'], ['prompt', 'service_identity'])
        self.assertEqual(Path(row['raw_output_path']).read_bytes(), raw_bytes)
        self.assertEqual(Path(row['raw_output_path']).with_suffix('.json').read_bytes(), metadata_bytes)
        self.assertFalse(self.raw.exists())
        self.assertFalse(self.metadata.exists())
        with Image.open(Path(row['raw_output_path'])) as raw, Image.open(row['output_path']) as normalized:
            expected = raw.convert('RGB').resize((20, 16), Image.Resampling.LANCZOS)
            self.assertEqual(expected.tobytes(), normalized.tobytes())
        self.assert_immutable()

    def test_second_recovery_skips_and_checks_saved_hashes(self):
        self.run_recovery(True)
        before = (self.directory / 'generated.jsonl').read_bytes()
        self.assertEqual(self.run_recovery(True)['recovered'], [])
        self.assertEqual((self.directory / 'generated.jsonl').read_bytes(), before)
        row = common.read_jsonl(self.directory / 'generated.jsonl')[-1]
        Path(row['output_path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Image missing or changed'):
            self.run_recovery(True)

    def test_wrong_seed_is_not_a_matching_candidate(self):
        meta = json.loads(self.metadata.read_text())
        meta['seed'] += 1
        common.write_json(self.metadata, meta)
        self.assertEqual(self.run_recovery(True)['recoverable'], [])
        self.assertTrue(self.raw.exists())
        self.assert_immutable()

    def test_two_matching_sidecars_are_ambiguous(self):
        self.make_output('duplicate')
        before = (self.directory / 'generated.jsonl').read_bytes()
        with self.assertRaisesRegex(ValueError, 'Multiple matching'):
            self.run_recovery(True)
        self.assertEqual((self.directory / 'generated.jsonl').read_bytes(), before)
        self.assertTrue(self.raw.exists())
        self.assert_immutable()

    def test_sidecar_dimension_mismatch_refuses_recovery(self):
        meta = json.loads(self.metadata.read_text())
        meta['raw_dimensions'] = [31, 24]
        common.write_json(self.metadata, meta)
        with self.assertRaisesRegex(ValueError, 'dimensions or canvas'):
            self.run_recovery(True)
        self.assertTrue(self.raw.exists())
        self.assert_immutable()

    def test_changed_frozen_implementation_refuses_recovery(self):
        (self.code / 'frozen.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Frozen implementation changed'):
            self.run_recovery(True)
        self.assertTrue(self.raw.exists())
        self.assert_immutable()

    def test_changed_original_source_refuses_recovery(self):
        Image.new('RGB', (20, 16), (100, 20, 20)).save(self.jobs[1]['source_path'])
        with self.assertRaisesRegex(ValueError, 'Image missing or changed'):
            self.run_recovery(True)
        self.assertTrue(self.raw.exists())
        self.assert_immutable()

    def test_changed_durable_ledger_refuses_recovery(self):
        rows = common.read_jsonl(self.directory / 'attempts.jsonl')
        rows[-1]['call_number'] = 500
        common.write_jsonl(self.directory / 'attempts.jsonl', rows)
        with self.assertRaisesRegex(ValueError, 'attempt_sha256'):
            self.run_recovery(True)
        self.assertTrue(self.raw.exists())

    def test_journal_failure_after_file_moves_resumes_from_hashed_intent(self):
        original = common.write_jsonl
        def fail_generated(path, rows):
            if Path(path).name == 'generated.jsonl':
                raise OSError('interrupted journal write')
            return original(path, rows)
        with patch.object(common, 'write_jsonl', side_effect=fail_generated):
            with self.assertRaisesRegex(OSError, 'interrupted journal'):
                self.run_recovery(True)
        self.assertFalse(self.raw.exists())
        self.assertFalse(self.metadata.exists())
        self.assertEqual(len(common.read_jsonl(self.directory / 'generated.jsonl')), 1)
        self.assertEqual(self.run_recovery(True)['recovered'], [self.jobs[1]['sample_id']])
        self.assertEqual(len(common.read_jsonl(self.directory / 'generated.jsonl')), 2)
        self.assert_immutable()

    def test_service_sidecar_cannot_move_unowned_image(self):
        meta = json.loads(self.metadata.read_text())
        meta['result_path'] = self.jobs[1]['source_path']
        common.write_json(self.metadata, meta)
        with self.assertRaisesRegex(ValueError, 'owned image'):
            self.run_recovery(True)
        self.assertTrue(self.raw.exists())
        self.assert_immutable()

    def test_worker_lock_blocks_apply(self):
        with (self.directory / '.run.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, 'worker owns'):
                self.run_recovery(True)
        self.assertTrue(self.raw.exists())
        self.assert_immutable()


if __name__ == '__main__':
    unittest.main()
