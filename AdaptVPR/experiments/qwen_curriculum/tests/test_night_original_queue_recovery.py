from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.qwen_curriculum import night_original_queue_recovery as recovery


class QueueRecoveryTests(unittest.TestCase):
    def queue(self,n):
        return recovery.cluster.sealed(dict(version=1,node='gpu2',entries=[dict(state=str(i)) for i in range(n)]))

    def test_equal_queue_does_not_transfer_or_validate_images(self):
        queue=self.queue(2)
        with patch.object(recovery.cluster,'queue',return_value=queue),patch.object(recovery,'remote_queue',return_value=queue),\
             patch.object(recovery,'verify_staged') as verify,patch.object(recovery.cluster,'rsync_files') as transfer:
            self.assertEqual(recovery.reconcile(Path('/unused')),0)
            verify.assert_not_called();transfer.assert_not_called()

    def test_lost_delivery_republishes_exact_queue_and_is_idempotent(self):
        local=self.queue(3);remote=self.queue(2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(recovery.cluster,'queue',return_value=local),\
                 patch.object(recovery,'remote_queue',side_effect=[remote,local,local]),\
                 patch.object(recovery,'verify_staged') as verify,patch.object(recovery.cluster,'rsync_files') as transfer:
                self.assertEqual(recovery.reconcile(root),1)
                self.assertEqual(recovery.reconcile(root),0)
                verify.assert_called_once_with(local['entries'][2:])
                transfer.assert_called_once_with(root,[root/'nodes/gpu2/queue.json'])
            self.assertEqual(len(list((root/'deployment/queue_deliveries').glob('*.json'))),1)
            self.assertFalse((root/'ownership.json').exists())

    def test_divergent_or_ahead_remote_queue_is_never_overwritten(self):
        local=self.queue(2)
        for remote in (self.queue(3),{**self.queue(1),'entries':[dict(state='different')]}):
            with patch.object(recovery.cluster,'queue',return_value=local),patch.object(recovery,'remote_queue',return_value=remote),\
                 patch.object(recovery.cluster,'rsync_files') as transfer:
                with self.assertRaises(ValueError):recovery.reconcile(Path('/unused'))
                transfer.assert_not_called()

    def test_missing_staged_inputs_block_publication(self):
        with patch.object(recovery.cluster,'queue',return_value=self.queue(2)),\
             patch.object(recovery,'remote_queue',return_value=self.queue(1)),\
             patch.object(recovery,'verify_staged',side_effect=ValueError('missing input')),\
             patch.object(recovery.cluster,'rsync_files') as transfer:
            with self.assertRaises(ValueError):recovery.reconcile(Path('/unused'))
            transfer.assert_not_called()

    def test_lost_confirmation_does_not_cause_duplicate_publication_or_assignment(self):
        local=self.queue(2);remote=self.queue(1)
        with patch.object(recovery.cluster,'queue',return_value=local),\
             patch.object(recovery,'remote_queue',side_effect=[remote,RuntimeError('SSH lost'),local]),\
             patch.object(recovery,'verify_staged'),patch.object(recovery.cluster,'rsync_files') as transfer:
            with self.assertRaises(RuntimeError):recovery.reconcile(Path('/unused'))
            self.assertEqual(recovery.reconcile(Path('/unused')),0)
            self.assertEqual(transfer.call_count,1)


if __name__=='__main__':unittest.main()
