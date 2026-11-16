"""CPU-only checks for durable cluster ownership and remote publication.

Every subprocess/transfer entry point is mocked. These tests never contact SSH,
systemd, a generator, or a GPU, and only write inside TemporaryDirectory.
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from experiments.qwen_curriculum import common
from experiments.qwen_curriculum import night_cluster as cluster


def result_sealed(value):
    body = {key: item for key, item in value.items() if key != 'result_sha256'}
    return {**body, 'result_sha256': common.fingerprint(body)}


class ClusterControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.root = self.workspace / 'outputs/cluster'
        self.cohort = self.workspace / 'outputs/cohort/state'
        self.cohort.mkdir(parents=True)
        for attribute, value in [('ROOT', self.workspace), ('COHORT', self.cohort)]:
            context = patch.object(cluster, attribute, value)
            context.start()
            self.addCleanup(context.stop)
        self.calls = patch.object(cluster, 'call', side_effect=AssertionError('Unexpected external command'))
        self.calls.start()
        self.addCleanup(self.calls.stop)

    def shard(self, number, jobs=2, saved=0, attempted=False):
        state = self.workspace / f'outputs/cohort/shards/s{number:05d}/state'
        state.mkdir(parents=True)
        cluster.save(state / 'plan_config.json', {'version': 1, 'shard': number})
        if saved:
            common.write_jsonl(state / 'results.jsonl', [{'sample_id': f'{number}-{i}'} for i in range(saved)])
        if attempted:
            common.write_jsonl(state / 'attempts.jsonl', [{'reserved': True}])
        return {'original_state': str(state),
                'original_config_sha256': common.file_sha256(state / 'plan_config.json'),
                'jobs': jobs, 'saved_at_creation': saved}

    def marker(self, descriptors):
        value = result_sealed({'ready': sum(item['jobs'] for item in descriptors), 'shards': [
            {'state': item['original_state'], 'jobs': item['jobs'],
             'config_sha256': item['original_config_sha256']} for item in descriptors]})
        common.write_json(self.cohort / 'prompts_complete.json', value)
        return value

    def queues(self, descriptors, owners=None, entries=None, partial=()):
        self.root.mkdir(parents=True, exist_ok=True)
        config = cluster.freeze(self.root / 'cluster.json', {
            'version': 1, 'shards': descriptors, 'partial_local_only': list(partial)})
        cluster.save(self.root / 'ownership.json', {
            'version': 1, 'cluster_fingerprint': config['fingerprint'], 'owners': owners or {}})
        for node in ('local', 'gpu2'):
            cluster.save(self.root / 'nodes' / node / 'queue.json', {
                'version': 1, 'node': node, 'entries': (entries or {}).get(node, [])})
        return config

    def test_disconnected_owner_is_never_reassigned(self):
        first, second = self.shard(0), self.shard(1)
        config = {'shards': [first, second], 'partial_local_only': []}
        owners = {'owners': {first['original_state']: 'gpu2'}}
        queues = {'local': {'entries': []}, 'gpu2': {'entries': [dict(first)]}}
        self.assertEqual(cluster.select_assignment(config, owners, queues, 'local'), second)
        config['shards'] = [first]
        self.assertIsNone(cluster.select_assignment(config, owners, queues, 'local'))

    def test_resume_one_node_after_pause_all_keeps_other_node_paused(self):
        self.queues([])
        (self.root/'pause_requested').touch()
        (self.root/'nodes/local/pause_requested').touch()
        def command(node,*args,**kwargs):
            if node=='local' and args[:2]==('rm','-f'):
                Path(args[2]).unlink(missing_ok=True)
        with patch.object(cluster,'command_node',side_effect=command),patch.object(cluster,'call'):
            cluster.controls(self.root,'resume','local')
        self.assertFalse((self.root/'pause_requested').exists())
        self.assertFalse((self.root/'nodes/local/pause_requested').exists())
        self.assertTrue((self.root/'nodes/gpu2/pause_requested').exists())

    def test_verified_deployment_receipt_survives_transient_unit_disappearance(self):
        directory=self.root/'deployment';directory.mkdir(parents=True)
        manifest=directory/'weight_bundle.json';manifest.write_text('trusted bundle manifest')
        cluster.save(directory/'remote_ready.json',{'manifest_sha256':common.file_sha256(manifest)})
        self.assertTrue(cluster.transfer_complete(self.root))
        manifest.write_text('different bundle')
        self.assertFalse(cluster.transfer_complete(self.root))

    def test_reserved_but_not_staged_work_returns_only_to_its_owner(self):
        first = self.shard(0)
        config = {'shards': [first], 'partial_local_only': []}
        owners = {'owners': {first['original_state']: 'gpu2'}}
        queues = {node: {'entries': []} for node in ('local', 'gpu2')}
        self.assertIsNone(cluster.select_assignment(config, owners, queues, 'local'))
        self.assertEqual(cluster.select_assignment(config, owners, queues, 'gpu2'), first)

    def test_partial_original_shard_is_local_only_even_without_owner_entry(self):
        partial, fresh = self.shard(0, attempted=True), self.shard(1)
        config = {'shards': [partial, fresh], 'partial_local_only': [partial['original_state']]}
        queues = {node: {'entries': []} for node in ('local', 'gpu2')}
        self.assertEqual(cluster.select_assignment(config, {'owners': {}}, queues, 'gpu2'), fresh)
        self.assertEqual(cluster.select_assignment(config, {'owners': {}}, queues, 'local'), partial)

    def test_grant_reserves_ownership_before_staging_and_recovers_transfer_failure(self):
        descriptor = self.shard(0)
        self.queues([descriptor])

        def interrupted_stage(root, selected, node):
            self.assertEqual(cluster.read_sealed(root / 'ownership.json')['owners'],
                             {descriptor['original_state']: 'gpu2'})
            raise OSError('simulated disconnected transfer')

        with patch.object(cluster, 'stage_assignment', side_effect=interrupted_stage):
            with self.assertRaisesRegex(OSError, 'disconnected'):
                cluster.grant(self.root, 'gpu2')
        self.assertEqual(cluster.queue(self.root, 'gpu2')['entries'], [])
        self.assertIsNone(cluster.grant(self.root, 'local'))
        entry = {**descriptor, 'state': str(self.root / 'nodes/gpu2/shards/s00000/state'),
                 'config_sha256': 'a' * 64}
        with patch.object(cluster, 'stage_assignment', return_value=entry) as stage, \
                patch.object(cluster, 'rsync_files'):
            self.assertEqual(cluster.grant(self.root, 'gpu2'), entry)
            stage.assert_called_once_with(self.root, descriptor, 'gpu2')
        self.assertEqual(cluster.queue(self.root, 'gpu2')['entries'], [entry])

    def test_unfinished_queued_shard_prevents_second_grant(self):
        first, second = self.shard(0), self.shard(1)
        entry = {**first, 'state': first['original_state']}
        self.queues([first, second], owners={first['original_state']: 'local'}, entries={'local': [entry]})
        with patch.object(cluster, 'stage_assignment') as stage:
            self.assertIsNone(cluster.grant(self.root, 'local'))
            stage.assert_not_called()

    def test_remote_pause_is_mirrored_for_controller_scheduling(self):
        (self.root / 'nodes/gpu2').mkdir(parents=True)
        with patch.object(cluster, 'command_node') as command:
            cluster.controls(self.root, 'pause', 'gpu2')
        self.assertTrue((self.root / 'nodes/gpu2/pause_requested').is_file())
        command.assert_called_once_with(
            'gpu2', 'touch', str(self.root / 'nodes/gpu2/pause_requested'))

    def test_prepare_accepts_result_sha_marker_and_is_idempotent_after_progress(self):
        descriptor = self.shard(0, attempted=True)
        self.marker([descriptor])
        initial = cluster.prepare(self.root)
        ownership = (self.root / 'ownership.json').read_bytes()
        common.write_jsonl(Path(descriptor['original_state']) / 'results.jsonl', [{'sample_id': 'new'}])
        repeated = cluster.prepare(self.root)
        self.assertEqual(initial, repeated)
        self.assertEqual((self.root / 'ownership.json').read_bytes(), ownership)
        self.assertEqual(initial['historical_saved'], 0)
        self.assertEqual(cluster.read_sealed(self.root / 'ownership.json')['owners'],
                         {descriptor['original_state']: 'local'})

    def test_prepare_recovers_crash_after_cluster_freeze(self):
        descriptor = self.shard(0, attempted=True)
        self.marker([descriptor])
        with patch.object(cluster, 'save', side_effect=OSError('interrupted initialization')):
            with self.assertRaises(OSError):
                cluster.prepare(self.root)
        self.assertTrue((self.root / 'cluster.json').exists())
        cluster.prepare(self.root)
        self.assertEqual(cluster.read_sealed(self.root / 'ownership.json')['owners'],
                         {descriptor['original_state']: 'local'})
        for node in ('local', 'gpu2'):
            self.assertEqual(cluster.queue(self.root, node)['entries'], [])
        self.assertTrue((self.cohort / 'cluster_ownership.json').exists())

    def test_prepare_rejects_changed_cohort_marker(self):
        descriptor = self.shard(0)
        self.marker([descriptor])
        cluster.prepare(self.root)
        value = json.loads((self.cohort / 'prompts_complete.json').read_text())
        value['ready'] += 1
        common.write_json(self.cohort / 'prompts_complete.json', result_sealed(value))
        with self.assertRaises(ValueError):
            cluster.prepare(self.root)

    def test_prepare_rejects_changed_original_config_on_reentry(self):
        descriptor = self.shard(0)
        self.marker([descriptor])
        cluster.prepare(self.root)
        cluster.save(Path(descriptor['original_state']) / 'plan_config.json', {'modified': True})
        with self.assertRaises(ValueError):
            cluster.prepare(self.root)

    def sync_fixture(self):
        descriptor = self.shard(0, jobs=1)
        remote_root = self.root / 'nodes/gpu2'
        state = remote_root / 'shards/s00000/state'
        state.mkdir(parents=True)
        staging = self.root / 'incoming/gpu2'
        incoming = staging / 'shards/s00000/state'
        incoming.mkdir(parents=True)
        source = self.workspace / 'source.png'
        Image.new('RGB', (8, 6), 'white').save(source)
        body = {'sample_id': 'night-sample', 'condition': 'night', 'city': 'city', 'place_id': 'place',
                'source_path': str(source), 'source_sha256': common.file_sha256(source),
                'source_dimensions': [8, 6], 'prompt': 'Preserve this place at night.', 'seed': 42}
        job = {**body, 'record_sha256': common.fingerprint(body)}
        common.write_jsonl(state / 'plan.jsonl', [job])
        config = cluster.freeze(state / 'plan_config.json', {
            'image_root': str(remote_root / 'images'), 'plan_sha256': common.file_sha256(state / 'plan.jsonl'),
            'request_budget': 1, 'service': {'canvas': {
                'target_shape_order': 'height,width', 'rounding': 'nearest_multiple',
                'unsupported_aspect_ratio': 'reject', 'target_pixels': 192,
                'multiple': 1, 'min_side': 1, 'max_side': 32}}})
        entry = {**descriptor, 'state': str(state),
                 'config_sha256': common.file_sha256(state / 'plan_config.json')}
        self.queues([descriptor], owners={descriptor['original_state']: 'gpu2'}, entries={'gpu2': [entry]})
        output = remote_root / 'images/night/night-sample.png'
        raw = remote_root / 'images/raw/night/night-sample.png'
        for path, size in [(output, (8, 6)), (raw, (16, 12))]:
            staged = staging / path.relative_to(remote_root)
            staged.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', size, 'black').save(staged)
        sidecar = staging / raw.with_suffix('.json').relative_to(remote_root)
        service_raw = (common.ADAPTVPR_ROOT / 'tmp/service_outputs/lightx2v/test-raw.png').resolve()
        common.write_json(sidecar, {
            'raw_dimensions': [16, 12], 'seed': 42, 'source_dimensions': [8, 6],
            'source_path': str(state / 'requests/a00000001/source_inputs/night-sample.png'),
            'result_path': str(service_raw), 'metadata_path': str(service_raw.with_suffix('.json')),
            'canvas_policy': 'source_aspect_v1', 'target_shape': [12, 16]})
        attempt_body = {'sample_id': job['sample_id'], 'record_sha256': job['record_sha256'],
                        'workspace': str(state / 'requests/a00000001'), 'call_number': 1, 'seed': job['seed'],
                        'prompt': job['prompt'], 'source_sha256': job['source_sha256'],
                        'plan_sha256': config['plan_sha256'], 'reserved_utc': '2026-10-31T00:00:00Z'}
        attempt = {**attempt_body, 'attempt_sha256': common.fingerprint(attempt_body)}
        generated = result_sealed({**job, 'output_path': str(output),
                                  'output_sha256': common.file_sha256(staging / output.relative_to(remote_root)),
                                  'raw_output_path': str(raw),
                                  'raw_output_sha256': common.file_sha256(staging / raw.relative_to(remote_root)),
                                  'metadata_sha256': common.file_sha256(sidecar),
                                  'attempt_sha256': attempt['attempt_sha256'], 'origin': 'reserved_qwen_request',
                                  'sampling': {}})
        result = result_sealed({**generated, 'passed': False, 'eligible_for_training': False,
                                'status': 'pending_verification', 'verification_pending': True,
                                'quality_version': 'deferred-no-model'})
        common.write_jsonl(incoming / 'attempts.jsonl', [attempt])
        common.write_jsonl(incoming / 'generated.jsonl', [generated])
        common.write_jsonl(incoming / 'results.jsonl', [result])
        common.write_json(incoming / 'summary.json', {'state': 'domain_complete', 'saved': 1, 'evaluated': 1})
        return {'entry': entry, 'state': state, 'staging': staging, 'incoming': incoming,
                'remote_root': remote_root, 'job': job, 'generated': generated, 'result': result,
                'attempt': attempt, 'output': output, 'raw': raw, 'sidecar': sidecar}

    def sync(self):
        with patch.object(cluster, 'call', return_value=''):
            return cluster.sync(self.root)

    def test_sync_valid_result_is_pending_and_idempotent(self):
        case = self.sync_fixture()
        self.assertEqual(self.sync(), 1)
        original = (case['state'] / 'results.jsonl').read_bytes()
        self.assertEqual(self.sync(), 1)
        self.assertEqual((case['state'] / 'results.jsonl').read_bytes(), original)
        self.assertFalse(common.read_jsonl(case['state'] / 'results.jsonl')[0]['eligible_for_training'])
        self.assertTrue(case['output'].is_file())
        self.assertTrue(case['raw'].is_file())

    def test_sync_rejects_result_tampering(self):
        case = self.sync_fixture()
        row = {**case['result'], 'prompt': 'Changed prompt'}
        common.write_jsonl(case['incoming'] / 'results.jsonl', [row])
        with self.assertRaises(ValueError):
            self.sync()

    def test_sync_rejects_resealed_different_frozen_job(self):
        case = self.sync_fixture()
        row = result_sealed({**case['result'], 'prompt': 'Changed prompt'})
        common.write_jsonl(case['incoming'] / 'results.jsonl', [row])
        with self.assertRaises(ValueError):
            self.sync()

    def test_sync_rejects_training_eligible_result(self):
        case = self.sync_fixture()
        row = result_sealed({**case['result'], 'eligible_for_training': True})
        common.write_jsonl(case['incoming'] / 'results.jsonl', [row])
        with self.assertRaises(ValueError):
            self.sync()

    def test_sync_rejects_output_outside_assigned_directory(self):
        case = self.sync_fixture()
        row = result_sealed({**case['result'], 'output_path': str(self.workspace / 'escaped.png')})
        common.write_jsonl(case['incoming'] / 'results.jsonl', [row])
        with self.assertRaises(ValueError):
            self.sync()

    def test_sync_rejects_incoming_output_symlink(self):
        case = self.sync_fixture()
        staged = case['staging'] / case['output'].relative_to(case['remote_root'])
        external = self.workspace / 'external.png'
        staged.replace(external)
        staged.symlink_to(external)
        with self.assertRaises(ValueError):
            self.sync()

    def test_sync_rejects_changed_image_bytes(self):
        case = self.sync_fixture()
        staged = case['staging'] / case['output'].relative_to(case['remote_root'])
        Image.new('RGB', (8, 6), 'red').save(staged)
        with self.assertRaises(ValueError):
            self.sync()

    def test_sync_rejects_changed_sidecar(self):
        case = self.sync_fixture()
        common.write_json(case['sidecar'], {'changed': True})
        with self.assertRaises(ValueError):
            self.sync()

    def test_sync_rejects_resealed_sidecar_for_different_request(self):
        case = self.sync_fixture()
        metadata = json.loads(case['sidecar'].read_text())
        common.write_json(case['sidecar'], {**metadata, 'seed': 999})
        digest = common.file_sha256(case['sidecar'])
        for name in ('generated', 'result'):
            row = result_sealed({**case[name], 'metadata_sha256': digest})
            common.write_jsonl(case['incoming'] / ('results.jsonl' if name == 'result' else 'generated.jsonl'), [row])
        with self.assertRaises(ValueError):
            self.sync()

    def test_sync_rejects_non_png_payload_even_with_matching_hash(self):
        case = self.sync_fixture()
        staged = case['staging'] / case['output'].relative_to(case['remote_root'])
        Image.new('RGB', (8, 6), 'black').save(staged, format='JPEG')
        digest = common.file_sha256(staged)
        for name in ('generated', 'result'):
            row = result_sealed({**case[name], 'output_sha256': digest})
            common.write_jsonl(case['incoming'] / ('results.jsonl' if name == 'result' else 'generated.jsonl'), [row])
        with self.assertRaises(ValueError):
            self.sync()

    def test_sync_rejects_unsealed_generated_row(self):
        case = self.sync_fixture()
        generated = {**case['generated'], 'prompt': 'corrupted generated journal'}
        common.write_jsonl(case['incoming'] / 'generated.jsonl', [generated])
        with self.assertRaises(ValueError):
            self.sync()

    def test_sync_rejects_duplicate_results(self):
        case = self.sync_fixture()
        common.write_jsonl(case['incoming'] / 'results.jsonl', [case['result'], case['result']])
        with self.assertRaises(ValueError):
            self.sync()

    def test_sync_rejects_invalid_request_reservation(self):
        case = self.sync_fixture()
        common.write_jsonl(case['incoming'] / 'attempts.jsonl', [{**case['attempt'], 'seed': 999}])
        with self.assertRaises(ValueError):
            self.sync()

    def test_sync_rejects_generated_image_without_reservation(self):
        case = self.sync_fixture()
        common.write_jsonl(case['incoming'] / 'attempts.jsonl', [])
        with self.assertRaises(ValueError):
            self.sync()

    def test_sync_does_not_regress_already_committed_prefix(self):
        case = self.sync_fixture()
        self.sync()
        saved = (case['state'] / 'results.jsonl').read_bytes()
        common.write_jsonl(case['incoming'] / 'results.jsonl', [])
        with self.assertRaises(ValueError):
            self.sync()
        self.assertEqual((case['state'] / 'results.jsonl').read_bytes(), saved)

    def test_sync_torn_final_json_line_waits_without_committing_tail(self):
        case = self.sync_fixture()
        path = case['incoming'] / 'results.jsonl'
        path.write_text(path.read_text() + '{"sample_id":')
        self.assertEqual(self.sync(), 1)
        self.assertEqual(common.read_jsonl(case['state'] / 'results.jsonl'), [case['result']])

    def test_rsync_input_rejects_symlinks_and_workspace_escape(self):
        (self.root / 'deployment').mkdir(parents=True)
        regular = self.workspace / 'regular'
        regular.write_text('source')
        link = self.workspace / 'link'
        link.symlink_to(regular)
        for path in (link, self.workspace.parent / 'outside'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                cluster.rsync_files(self.root, [path])


if __name__ == '__main__':
    unittest.main()
