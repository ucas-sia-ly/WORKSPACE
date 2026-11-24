from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from experiments.qwen_curriculum import common, run
from experiments.qwen_curriculum import night_revision_planner as module


class RevisionPlannerTests(unittest.TestCase):
    def fixture(self, directory):
        source = directory/'source.png'
        Image.new('RGB', (20, 10), 'white').save(source)
        sources = [dict(sample_id='a', source_path=str(source),
                        source_sha256=common.file_sha256(source))]
        cfg = dict(fingerprint='revision-protocol', seed=42,
                   analysis_tokens=2400, compose_tokens=1700)
        planner = module.Planner(directory/'unused-model',
                                 policy=SimpleNamespace(SYSTEM='revised compose system'))
        return planner, sources, ['Inspect source evidence'], cfg

    def test_saved_response_reused_without_loading_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            planner, sources, requests, cfg = self.fixture(root)
            with patch.object(planner, '_infer', return_value=['saved']) as infer:
                self.assertEqual(planner.ask(root, sources, requests, 'analysis', 0, cfg), ['saved'])
                self.assertEqual(planner.ask(root, sources, requests, 'analysis', 0, cfg), ['saved'])
            self.assertEqual(infer.call_count, 1)
            self.assertIsNone(planner._backend)
            self.assertFalse((root/'inference_recovery').exists())

    def test_interrupted_text_inference_is_archived_and_retried_with_same_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            planner, sources, requests, cfg = self.fixture(root)
            seeds = []
            def interrupted(sources, requests, identity, seed):
                seeds.append(seed)
                raise KeyboardInterrupt()
            with patch.object(planner, '_infer', side_effect=interrupted):
                for _ in range(2):
                    with self.assertRaises(KeyboardInterrupt):
                        planner.ask(root, sources, requests, 'compose', 0, cfg)
            def completed(sources, requests, identity, seed):
                seeds.append(seed)
                return ['done']
            with patch.object(planner, '_infer', side_effect=completed):
                self.assertEqual(planner.ask(root, sources, requests, 'compose', 0, cfg), ['done'])
            self.assertEqual(len(set(seeds)), 1)
            archives = list((root/'inference_recovery').rglob('*.request.json'))
            self.assertEqual(len(archives), 2)
            self.assertEqual(sorted(module.sealed_read(p)['inference_attempt'] for p in archives), [0, 1])
            response = module.sealed_read(next(p for p in (root/'batches').glob('*.json')
                                               if not p.name.endswith('.request.json')))
            self.assertEqual(response['inference_attempt'], 2)
            self.assertTrue(response['no_image_generation'])

    def test_saved_response_finishes_after_missing_group_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            planner, sources, requests, cfg = self.fixture(root)
            with patch.object(planner, '_infer', return_value=['ready']):
                planner.ask(root, sources, requests, 'analysis', 0, cfg)
            cached = module.Planner(root/'unused-model', policy=planner.policy)
            with patch.object(cached, 'ensure_model', side_effect=AssertionError('Must reuse response')):
                self.assertEqual(cached.ask(root, sources, requests, 'analysis', 0, cfg), ['ready'])

    def test_request_and_response_mismatch_is_never_recomputed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            planner, sources, requests, cfg = self.fixture(root)
            with patch.object(planner, '_infer', return_value=['ready']):
                planner.ask(root, sources, requests, 'analysis', 0, cfg)
            request = next((root/'batches').glob('*.request.json'))
            value = module.sealed_read(request)
            common.write_json(request, run.sealed({**value, 'inference_attempt': 99}))
            with patch.object(planner, '_infer', side_effect=AssertionError('Must not recompute')):
                with self.assertRaisesRegex(ValueError, 'reservation disagree'):
                    planner.ask(root, sources, requests, 'analysis', 0, cfg)

    def test_corrupt_or_different_prior_reservation_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            planner, sources, requests, cfg = self.fixture(root)
            with patch.object(planner, '_infer', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    planner.ask(root, sources, requests, 'analysis', 0, cfg)
            path = next((root/'batches').glob('*.request.json'))
            common.write_json(path, run.sealed({**module.sealed_read(path), 'seed': -1}))
            with patch.object(planner, '_infer', side_effect=AssertionError('Must not retry')):
                with self.assertRaisesRegex(ValueError, 'exact inputs'):
                    planner.ask(root, sources, requests, 'analysis', 0, cfg)
            self.assertFalse((root/'inference_recovery').exists())

    def test_system_token_budget_and_source_bytes_are_part_of_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            planner, sources, requests, cfg = self.fixture(root)
            first = planner._identity(sources, requests, 'compose', 0, cfg)
            self.assertEqual(first['system'], 'revised compose system')
            self.assertEqual(first['max_tokens'], 1700)
            analysis = planner._identity(sources, requests, 'analysis', 0, cfg)
            self.assertEqual(analysis['system'], module.ANALYSIS_SYSTEM)
            planner.policy.SYSTEM = 'changed system'
            self.assertNotEqual(first, planner._identity(sources, requests, 'compose', 0, cfg))
            Path(sources[0]['source_path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'source bytes changed'):
                planner._identity(sources, requests, 'compose', 0, cfg)

    def test_missing_reservation_with_saved_valid_response_is_recoverable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            planner, sources, requests, cfg = self.fixture(root)
            with patch.object(planner, '_infer', return_value=['ready']):
                planner.ask(root, sources, requests, 'analysis', 0, cfg)
            next((root/'batches').glob('*.request.json')).unlink()
            with patch.object(planner, '_infer', side_effect=AssertionError('Must not infer')):
                self.assertEqual(planner.ask(root, sources, requests, 'analysis', 0, cfg), ['ready'])

    def test_invalid_input_batch_and_response_cardinality_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            planner, sources, requests, cfg = self.fixture(root)
            for bad_sources, bad_requests in (([], []), (sources*2, requests*2), (sources, [])):
                with self.assertRaises(ValueError):
                    planner.ask(root, bad_sources, bad_requests, 'analysis', 0, cfg)
            with patch.object(planner, '_infer', return_value=[]):
                with self.assertRaisesRegex(RuntimeError, 'correspondence'):
                    planner.ask(root, sources, requests, 'analysis', 0, cfg)
            self.assertEqual(len(list((root/'batches').glob('*.request.json'))), 1)
            self.assertEqual(len(list((root/'batches').glob('*.json'))), 1)

    def test_close_is_lazy_and_idempotent(self):
        planner = module.Planner('/unused')
        planner.close()
        backend = SimpleNamespace(close=unittest.mock.Mock())
        planner._backend = backend
        planner.close(); planner.close()
        backend.close.assert_called_once()

    def test_invalid_budgets_fail_before_gpu_load(self):
        for amount in (0, -1, 43):
            with self.assertRaises(ValueError):
                module.Planner('/unused', allocator_gib=amount)

    def test_runtime_supplement_is_bound_to_cached_response_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);planner,sources,requests,cfg=self.fixture(root)
            before=planner._identity(sources,requests,'analysis',0,cfg)
            cfg['execution_patch_sha256']='repair-digest'
            after=planner._identity(sources,requests,'analysis',0,cfg)
            self.assertNotEqual(before,after)
            self.assertEqual(after['execution_patch_sha256'],'repair-digest')


if __name__ == '__main__':
    unittest.main()
