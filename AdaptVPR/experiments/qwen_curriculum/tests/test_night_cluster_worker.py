import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments.qwen_curriculum import night_cluster_worker as worker


class ClusterWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.node = worker.sealed(dict(version=1, node='local', machine_id='this-machine',
            backend_unit='night-cluster-local.service', python=sys.executable,
            service_identity={'profile': 'frozen'}, limits=dict(max_gpu_gib=44,
                max_rss_gib=20, reserve_ram_gib=4, reserve_disk_gib=20)))
        self.write('node.json', self.node)

    def write(self, filename, value):
        worker.common.write_json(self.root / filename, value)

    def entry(self, number=1, remote=False):
        state = self.root / f'shard{number}'
        state.mkdir(exist_ok=True)
        (state / 'plan_config.json').write_text(json.dumps({'id': number}))
        digest = worker.common.file_sha256(state / 'plan_config.json')
        return dict(state=str(state), config_sha256=digest,
                    original_state=f'/original/shard{number}' if remote else str(state),
                    original_config_sha256='c'*64 if remote else digest)

    def queue(self, entries, node='local'):
        self.write('queue.json', worker.sealed(dict(version=1, node=node, entries=entries)))

    def test_node_is_bound_to_machine_and_code(self):
        with patch.object(worker, 'machine_id', return_value='this-machine'):
            self.assertEqual(worker.load_node(self.root), self.node)
        with patch.object(worker, 'machine_id', return_value='another-machine'):
            with self.assertRaisesRegex(ValueError, 'another machine'):
                worker.load_node(self.root)
        self.write('node.json', worker.sealed({**self.node, 'worker_sha256': '0'*64}))
        with patch.object(worker, 'machine_id', return_value='this-machine'):
            with self.assertRaisesRegex(ValueError, 'implementation changed'):
                worker.load_node(self.root)

    def test_invalid_limits_and_unsealed_files_are_rejected(self):
        self.write('node.json', {**self.node, 'node': 'gpu2'})
        with self.assertRaisesRegex(ValueError, 'checksum'):
            worker.read_sealed(self.root / 'node.json')
        node = copy.deepcopy(self.node)
        node['limits']['max_gpu_gib'] = -1
        self.write('node.json', worker.sealed(node))
        with patch.object(worker, 'machine_id', return_value='this-machine'):
            with self.assertRaisesRegex(ValueError, 'resource limit'):
                worker.load_node(self.root)

    def test_queue_allows_only_prefix_append(self):
        first, second = self.entry(), self.entry(2)
        self.queue([first, second])
        self.assertEqual(worker.load_queue(self.root, self.node, [first]), [first, second])
        for invalid in ([second, first], [], [second]):
            with self.subTest(invalid=invalid):
                self.queue(invalid)
                with self.assertRaisesRegex(ValueError, 'prefix'):
                    worker.load_queue(self.root, self.node, [first])

    def test_queue_rejects_duplicates_wrong_owner_and_malformed_paths(self):
        first = self.entry()
        self.queue([first, first])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            worker.load_queue(self.root, self.node, [])
        self.queue([first], node='gpu2')
        with self.assertRaisesRegex(ValueError, 'owner'):
            worker.load_queue(self.root, self.node, [])
        self.queue([{**first, 'state': '../relative'}])
        with self.assertRaisesRegex(ValueError, 'absolute'):
            worker.load_queue(self.root, self.node, [])

    def test_original_shard_and_generator_identity_remain_frozen(self):
        entry = self.entry()
        config = dict(service=self.node['service_identity'])
        engine = SimpleNamespace(load_plan=Mock(return_value=(config, [{'condition': 'night'}])))
        self.assertEqual(worker.validate_entry(entry, self.node, engine)[0], config)
        with self.assertRaisesRegex(ValueError, 'original frozen'):
            worker.validate_entry({**entry, 'original_state': '/other'}, self.node, engine)
        engine.load_plan.return_value = ({'service': {'profile': 'other'}}, [])
        with self.assertRaisesRegex(ValueError, 'identity'):
            worker.validate_entry(entry, self.node, engine)
        (Path(entry['state']) / 'plan_config.json').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            worker.validate_entry(entry, self.node, engine)

    def test_remote_requires_exact_original_provenance(self):
        entry = self.entry(remote=True)
        node = {**self.node, 'node': 'gpu2'}
        config = dict(service=node['service_identity'])
        engine = SimpleNamespace(load_plan=Mock(return_value=(config, [{'condition': 'night'}])))
        with self.assertRaisesRegex(ValueError, 'origin provenance'):
            worker.validate_entry(entry, node, engine)
        config['cluster_origin'] = dict(node='gpu2', original_config_sha256=entry['original_config_sha256'])
        worker.validate_entry(entry, node, engine)
        config['cluster_origin']['original_config_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'origin differs'):
            worker.validate_entry(entry, node, engine)

    def test_pause_finishes_reservation_and_commits_first_recovered_image(self):
        pause = worker.threading.Event()
        pause.set()
        publish, append = Mock(return_value={'state': 'generating'}), Mock()
        engine = SimpleNamespace(publish=publish, _append=append)
        args = (self.root, {}, [], {}, {}, [], 'generating', 'night')
        with worker.commit_pause(engine, pause, self.root / 'pause_requested'):
            engine._append(self.root / 'attempts.jsonl', {'id': 'reservation'})
            engine.publish(*args)  # An active HTTP request must not be interrupted.
            engine._append(self.root / 'generated.jsonl', {'id': 'saved'})
            engine.publish(*args)
            engine._append(self.root / 'results.jsonl', {'id': 'deferred_verification'})
            with self.assertRaises(worker.PauseRequested):
                engine.publish(*args)
        self.assertIs(engine.publish, publish)
        self.assertIs(engine._append, append)
        self.assertEqual(append.call_count, 3)

    def test_pause_file_is_checked_at_complete_boundary(self):
        pause = worker.threading.Event()
        (self.root / 'pause_requested').touch()
        engine = SimpleNamespace(publish=Mock(), _append=Mock())
        with worker.commit_pause(engine, pause, self.root / 'pause_requested'):
            with self.assertRaises(worker.PauseRequested):
                engine.publish(self.root, {}, [], {}, {}, [], 'domain_complete', 'night')
        self.assertTrue(pause.is_set())

    def test_limits_refuse_busy_gpu_and_low_reserves(self):
        sample = dict(available_gib=12, owned_rss_gib=8, gpu_used_gib=35, disk_free_gib=100)
        limits = self.node['limits']
        self.assertIsNone(worker.violation(sample, limits))
        self.assertIn('already in use', worker.violation(sample, limits, before_start=True))
        for metric, value in (('available_gib', 3), ('owned_rss_gib', 21),
                              ('gpu_used_gib', 45), ('disk_free_gib', 19)):
            self.assertIsNotNone(worker.violation({**sample, metric: value}, limits))

    def test_backend_cannot_adopt_unowned_or_stop_replacement_service(self):
        backend = worker.Backend(self.root, self.node)
        with patch.object(worker, 'service_pid', return_value=123):
            with self.assertRaisesRegex(RuntimeError, 'ownership'):
                backend.existing()
        backend.pid = 123
        self.write('backend_owner.json', worker.sealed(backend.owner()))
        with patch.object(worker, 'service_pid', return_value=123):
            self.assertEqual(backend.existing(), 123)
        with patch.object(worker, 'service_pid', return_value=456), patch.object(worker, 'systemctl') as systemctl:
            backend.stop()
            systemctl.assert_not_called()

    def test_resource_guard_stops_only_owned_backend_on_violation(self):
        backend = SimpleNamespace(pid=123, stop=Mock())
        pause = worker.threading.Event()
        guard = worker.ResourceGuard(self.root, self.node, backend, pause)
        guard.done = Mock()
        guard.done.wait.side_effect = [False, True]
        with patch.object(guard, 'sample', return_value=dict(available_gib=2, owned_rss_gib=4,
             gpu_used_gib=30, disk_free_gib=100)):
            guard.monitor()
        backend.stop.assert_called_once()
        self.assertTrue(pause.is_set())
        self.assertIn('RAM reserve', guard.error)

    def test_system_backend_scope_limits_sudo_to_start_stop(self):
        completed = SimpleNamespace(stdout='123\n')
        with patch.object(worker.subprocess, 'run', return_value=completed) as execute:
            self.assertEqual(worker.service_pid('night-gpu2.service', 'system'), 123)
            self.assertEqual(execute.call_args.args[0], [
                'systemctl', 'show', 'night-gpu2.service', '-p', 'MainPID', '--value'])
            worker.systemctl('start', 'night-gpu2.service', scope='system')
            self.assertEqual(execute.call_args.args[0], ['sudo', '-n', 'systemctl', 'start', 'night-gpu2.service'])
            worker.systemctl('stop', 'night-gpu2.service', scope='system')
            self.assertEqual(execute.call_args.args[0], ['sudo', '-n', 'systemctl', 'stop', 'night-gpu2.service'])
            worker.service_pid('night-local.service')
            self.assertEqual(execute.call_args.args[0][:2], ['systemctl', '--user'])

    def test_worker_uses_single_backend_and_ledger_survives_pause(self):
        from experiments.qwen_curriculum import balanced_night
        entry = self.entry()
        self.queue([entry])
        backend = SimpleNamespace(pid=0, existing=Mock(), start=Mock(), health=Mock(), stop=Mock())
        guard = SimpleNamespace(error=None, paths=[], thread=Mock(ident=1), sample=Mock(return_value={}), close=Mock())
        config = dict(service=self.node['service_identity'], image_root=str(self.root / 'images'))
        engine = SimpleNamespace(load_plan=Mock(return_value=(config, [{'condition': 'night'}])),
                                 publish=Mock(), _append=Mock())

        def generate(args):
            self.assertEqual(args.max_calls, 0)
            self.assertEqual(args.device, 'cpu')
            self.assertEqual(args.request_timeout, 1800)
            (self.root / 'pause_requested').touch()
            return {'state': 'domain_complete'}

        engine.generate = Mock(side_effect=generate)
        module = 'experiments.qwen_curriculum.night_cluster_worker.'
        with patch(module+'machine_id', return_value='this-machine'), \
             patch(module+'Backend', return_value=backend), patch(module+'ResourceGuard', return_value=guard), \
             patch.object(balanced_night, 'runtime', return_value=SimpleNamespace(engine=lambda: engine)):
            result = worker.run_worker(self.root, poll_seconds=.001)
        self.assertEqual(result['state'], 'paused')
        self.assertEqual(result['completed_entries'], 1)
        backend.start.assert_called_once()
        backend.health.assert_called_once()
        backend.stop.assert_called_once()
        self.assertEqual(worker.read_sealed(self.root / 'completed.json')['entries'], [entry])
        self.assertEqual(worker.read_sealed(self.root / 'observed_queue.json')['entries'], [entry])
        (self.root / 'pause_requested').unlink()
        engine.generate.reset_mock(); backend.start.reset_mock()
        load_queue = worker.load_queue

        def finish_idle(directory, node, observed):
            entries = load_queue(directory, node, observed)
            (self.root / 'pause_requested').touch()
            return entries

        with patch(module+'machine_id', return_value='this-machine'), \
             patch(module+'Backend', return_value=backend), patch(module+'ResourceGuard', return_value=guard), \
             patch(module+'load_queue', side_effect=finish_idle), \
             patch.object(balanced_night, 'runtime', return_value=SimpleNamespace(engine=lambda: engine)):
            resumed = worker.run_worker(self.root, poll_seconds=.001)
        self.assertEqual(resumed['completed_entries'], 1)
        engine.generate.assert_not_called()
        backend.start.assert_not_called()


if __name__ == '__main__':
    unittest.main()
