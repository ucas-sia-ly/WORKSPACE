import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock,patch
from types import SimpleNamespace

from experiments.qwen_curriculum import night_cluster_resources as r,night_cluster_worker as w


def policy():
    return dict(version=1,ram_soft_gib=2,ram_emergency_gib=.75,ram_grace_seconds=60,
                gpu_reserve_gib=1.5,startup_gpu_free_gib=42,startup_ram_gib=10,
                retry_seconds=120,stable_seconds=30,fingerprint='policy')


def sample(**changes):
    return dict(available_gib=12,owned_rss_gib=11,gpu_free_gib=45,gpu_used_gib=3,
                disk_free_gib=100,**{}) | changes


class ResourcePolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy=policy();self.limits=dict(max_rss_gib=20,reserve_disk_gib=20,
                                           max_gpu_gib=44,reserve_ram_gib=4)

    def test_desktop_above_2gb_allowed_when_profile_fits(self):
        self.assertIsNone(r.startup_reason(sample(gpu_used_gib=4,gpu_free_gib=44),self.limits,self.policy))

    def test_busy_gpu_uses_actual_free_memory(self):
        self.assertIn('VRAM',r.startup_reason(sample(gpu_free_gib=30),self.limits,self.policy))

    def test_model_loading_needs_own_startup_ram_budget(self):
        self.assertIn('RAM',r.startup_reason(sample(available_gib=9),self.limits,self.policy))

    def test_old_4gb_threshold_is_not_new_stop_threshold(self):
        p=r.Pressure(self.policy,self.limits)
        self.assertEqual(p.check(sample(available_gib=3.5),100,backend_active=True),(None,False))

    def test_short_low_ram_recovers_without_stopping(self):
        p=r.Pressure(self.policy,self.limits)
        self.assertEqual(p.check(sample(available_gib=1.8),100,backend_active=True),(None,False))
        self.assertEqual(p.check(sample(available_gib=1.9),150,backend_active=True),(None,False))
        self.assertEqual(p.check(sample(available_gib=3),155,backend_active=True),(None,False))
        self.assertEqual(p.check(sample(available_gib=1.8),200,backend_active=True),(None,False))

    def test_sustained_low_ram_requests_graceful_commit(self):
        p=r.Pressure(self.policy,self.limits)
        p.check(sample(available_gib=1.8),0,backend_active=True)
        reason,emergency=p.check(sample(available_gib=1.8),60,backend_active=True)
        self.assertIn('sustained',reason);self.assertFalse(emergency)

    def test_emergency_ram_and_vram_are_immediate(self):
        for s in (sample(available_gib=.5),sample(gpu_free_gib=1)):
            reason,emergency=r.Pressure(self.policy,self.limits).check(s,0,backend_active=True)
            self.assertTrue(reason);self.assertTrue(emergency)

    def test_waiting_without_backend_does_not_trigger_low_ram_shutdown(self):
        self.assertEqual(r.Pressure(self.policy,self.limits).check(sample(available_gib=.5),0,
                         backend_active=False),(None,False))

    def test_disk_finishes_current_image_and_rss_has_hard_limit(self):
        self.assertEqual(r.Pressure(self.policy,self.limits).check(sample(disk_free_gib=19),0,
                         backend_active=True),('output disk reserve reached',False))
        self.assertEqual(r.Pressure(self.policy,self.limits).check(sample(owned_rss_gib=21),0,
                         backend_active=True),('owned RSS limit reached',True))

    def test_nonfinite_sample_is_not_accepted(self):
        self.assertEqual(r.Pressure(self.policy,self.limits).check(sample(gpu_free_gib=float('nan')),0,
                         backend_active=True),('nonfinite resource sample',True))

    def test_only_explicit_resource_errors_retry(self):
        self.assertTrue(r.recoverable(r.ResourceWait('loading budget')))
        self.assertTrue(r.recoverable(RuntimeError('Resource wait: sustained low host RAM')))
        for error in (ValueError('checksum'),RuntimeError('unknown POST state'),RuntimeError('Resource accounting failed')):
            self.assertFalse(r.recoverable(error))

    def test_uncertain_reservation_cannot_be_automatically_reposted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'attempts.jsonl';engine=SimpleNamespace(_append=Mock())
            w.common.write_jsonl(path,[{'sample_id':'old'}])
            r.protect_reservations(engine)
            with self.assertRaisesRegex(RuntimeError,'duplicate POST'):
                engine._append(path,{'sample_id':'old'})
            engine._append(path,{'sample_id':'new'})
            engine._append(path.parent/'results.jsonl',{'sample_id':'old'})

    def test_policy_is_bound_to_node_and_implementation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'policy.json';node={'fingerprint':'node'}
            body=policy() | dict(node_fingerprint='node',implementation_sha256=w.common.file_sha256(r.__file__))
            w.common.write_json(path,w.sealed(body))
            self.assertEqual(r.load_policy(path,node)['ram_soft_gib'],2)
            for changes in ({'node_fingerprint':'different'},{'implementation_sha256':'0'*64},
                            {'ram_emergency_gib':3},{'stable_seconds':0},{'retry_seconds':1}):
                w.common.write_json(path,w.sealed(body|changes))
                with self.assertRaises(ValueError):r.load_policy(path,node)

    def test_wait_state_preserves_progress_and_is_not_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            w.common.write_json(path/'status.json',dict(state='failed',active_state='shard',completed_entries=4))
            r.waiting_status(path,self.policy,'low RAM',sample(),123)
            value=json.loads((path/'status.json').read_text())
            self.assertEqual(value['state'],'waiting_for_resources')
            self.assertIsNone(value['error']);self.assertEqual(value['completed_entries'],4)

    def test_soft_guard_never_stops_backend_before_commit(self):
        self._guard(emergency=False)

    def test_hard_guard_stops_only_its_backend(self):
        self._guard(emergency=True)

    def test_recovery_reenters_original_worker_only_after_resource_wait(self):
        self._main([r.ResourceWait('RAM budget'),None],expected_calls=2)

    def test_integrity_failure_does_not_auto_restart(self):
        with self.assertRaisesRegex(ValueError,'checksum'):
            self._main([ValueError('checksum')],expected_calls=1)

    def test_manual_pause_is_respected_before_any_model_load(self):
        self._main([],expected_calls=0,paused=True)

    def _main(self,effects,expected_calls,paused=False):
        import itertools
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            if paused:(directory/'pause_requested').touch()
            clock=itertools.count(0,100)
            event=Mock();event.is_set.return_value=False
            with patch.object(w,'load_node',return_value={'limits':self.limits}), \
                    patch.object(r,'load_policy',return_value=self.policy), \
                    patch.object(r.subprocess,'check_output',return_value='49140'), \
                    patch.object(r,'install'),patch.object(r.signal,'signal'), \
                    patch.object(r,'protect_reservations'), \
                    patch.object(r.threading,'Event',return_value=event), \
                    patch.object(r.time,'monotonic',side_effect=lambda:next(clock)), \
                    patch.object(r.time,'time',side_effect=lambda:next(clock)), \
                    patch.object(w,'resource_sample',return_value=sample()), \
                    patch.object(w,'run_worker',side_effect=effects) as run:
                try:r.main(['--node-dir',tmp,'--policy',str(directory/'policy.json')])
                finally:self.assertEqual(run.call_count,expected_calls)

    def _guard(self,emergency):
        with tempfile.TemporaryDirectory() as tmp:
            originals=(w.Backend,w.ResourceGuard,w.resource_sample)
            self.addCleanup(lambda:setattr(w,'Backend',originals[0]))
            self.addCleanup(lambda:setattr(w,'ResourceGuard',originals[1]))
            self.addCleanup(lambda:setattr(w,'resource_sample',originals[2]))
            r.install(self.policy,48)
            backend=Mock(pid=123);pause=threading.Event()
            guard=w.ResourceGuard(Path(tmp),{'limits':self.limits},backend,pause)
            guard.done=Mock();guard.done.wait.return_value=False
            guard.sample=Mock(return_value=sample())
            guard.pressure.check=Mock(return_value=('pressure',emergency))
            guard.monitor()
            self.assertTrue(pause.is_set());self.assertTrue(guard.error.startswith('Resource wait: '))
            self.assertEqual(backend.stop.call_count,int(emergency))


if __name__=='__main__':unittest.main()
