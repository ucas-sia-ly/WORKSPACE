from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.qwen_curriculum import night_generation_completion as finish


class CompletionTests(unittest.TestCase):
    def test_remote_completion_must_match_node_and_owned_prefix(self):
        queue = [dict(state='one'), dict(state='two')]
        good = finish.base.worker.sealed(dict(node_fingerprint='node', entries=queue[:1]))
        with patch.object(finish.cluster, 'remote', return_value=finish.json.dumps(good)), \
                patch.object(finish.cluster, 'queue', return_value=dict(entries=queue)), \
                patch.object(finish.base.worker, 'read_sealed', return_value=dict(fingerprint='node')):
            self.assertEqual(finish.completion_entries(Path('/root'), 'gpu2'), queue[:1])
            for body in (dict(node_fingerprint='wrong', entries=queue[:1]),
                         dict(node_fingerprint='node', entries=[queue[1]]),
                         dict(node_fingerprint='node', entries=queue+[dict(state='foreign')])):
                with patch.object(finish.cluster, 'remote', return_value=finish.json.dumps(finish.base.worker.sealed(body))):
                    with self.assertRaises(ValueError):
                        finish.completion_entries(Path('/root'), 'gpu2')

    def test_outstanding_image_shard_blocks_second_assignment(self):
        with patch.object(finish.base, 'load', return_value={}), \
                patch.object(finish.recovery, 'reconcile'), \
                patch.object(finish.cluster, 'queue', return_value=dict(entries=[dict(state='busy')])), \
                patch.object(finish.cluster, 'stage_assignment') as stage:
            self.assertFalse(finish.grant(Path('/root'), 'gpu2', []))
            stage.assert_not_called()

    def test_remote_dispatch_does_not_require_primary_image_copy(self):
        entry = dict(original_state='/assigned', state='/remote')
        descriptor = dict(original_state='/next')
        cfg = dict(fingerprint='protocol')
        feed = dict(entries=[descriptor])
        owner = dict(cluster_fingerprint='protocol', owners={})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'source'; source.mkdir()
            target = root/'remote'; target.mkdir()
            (source/'prompt_receipts.jsonl').write_text('exact proof\n')
            descriptor['original_state'] = str(source)
            new_entry = dict(original_state=str(source), state=str(target))
            with patch.object(finish.base, 'load', return_value=cfg), \
                    patch.object(finish.recovery, 'reconcile'), \
                    patch.object(finish.cluster, 'queue', return_value=dict(entries=[entry])), \
                    patch.object(finish.base.worker, 'read_sealed', side_effect=[feed, owner]), \
                    patch.object(finish.cluster, 'select_assignment', return_value=descriptor), \
                    patch.object(finish.cluster, 'stage_assignment', return_value=new_entry), \
                    patch.object(finish.cluster, 'save') as save, \
                    patch.object(finish.cluster, 'rsync_files'):
                self.assertTrue(finish.grant(root, 'gpu2', [entry]))
                self.assertEqual(owner['owners'], {str(source): 'gpu2'})
                self.assertEqual((target/'prompt_receipts.jsonl').read_text(), 'exact proof\n')
                self.assertEqual(save.call_args_list[0].args[0], root/'ownership.json')

    def test_mercury_worker_reuses_existing_memory_limited_backend_adapter(self):
        for node in ('local', 'gpu2'):
            unit = finish.worker_unit(node)
            self.assertIn('MemoryMax=2G', unit)
            self.assertIn('MemorySwapMax=0', unit)
            self.assertIn('night_mercury_images.py worker --node '+node, unit)
            self.assertNotIn('train', unit)

    def test_unpublished_owned_shard_is_recovered_before_campaign_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(finish, 'ROOT', root), \
                    patch.object(finish, 'campaign', return_value=root), \
                    patch.object(finish.base, 'load'), \
                    patch.object(finish, 'completion_entries', return_value=[]), \
                    patch.object(finish.base.worker, 'read_sealed', side_effect=[
                        dict(entries=[dict(original_state='reserved')]),
                        dict(owners={'reserved': 'gpu2'})]), \
                    patch.object(finish.cluster, 'queue', return_value=dict(entries=[])), \
                    patch.object(finish, 'pause_marker'), \
                    patch.object(finish, 'grant', return_value=True) as grant, \
                    patch.object(finish.base, 'active', return_value='active'):
                value = finish.dispatch('original', 'gpu2')
                self.assertEqual(value['state'], 'dispatched')
                grant.assert_called_once()

    def test_bulk_collection_preserves_partial_files_and_long_transfer_budget(self):
        with patch.object(finish, 'campaign', return_value=Path('/root')), \
                patch.object(finish.cluster, 'call', return_value='done') as call, \
                patch.object(finish.base, 'status'), \
                patch.object(finish.base, 'prune_confirmed', return_value=0), \
                patch.object(finish.cluster, 'sync', side_effect=lambda root: finish.cluster.call(
                    ['rsync', '-a', 'remote:/images/', '/root/incoming/gpu2/'], timeout=600)):
            previous = finish.cluster.call
            try:
                finish.collect('original', once=True)
                self.assertIn('--partial-dir=.rsync-partial', call.call_args.args[0])
                self.assertEqual(call.call_args.kwargs['timeout'], 7200)
            finally:
                finish.cluster.call = previous


if __name__ == '__main__':
    unittest.main()
