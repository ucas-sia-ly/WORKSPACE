"""Versioned operational resource policy; frozen jobs and worker stay unchanged.

Resource pressure is a recoverable wait. Integrity, ownership and unknown POST
errors still fail closed. Only this node's backend may be stopped.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import common, night_cluster_worker as worker
from experiments.qwen_curriculum import weather_io


class ResourceWait(RuntimeError):
    pass


def load_policy(path, node):
    policy = worker.read_sealed(path)
    if (policy.get('version') != 1 or policy.get('node_fingerprint') != node['fingerprint']
            or policy.get('implementation_sha256') != common.file_sha256(__file__)):
        raise ValueError('Resource policy version, node or implementation changed')
    for key in ('ram_soft_gib', 'ram_emergency_gib', 'ram_grace_seconds', 'gpu_reserve_gib',
                'startup_gpu_free_gib', 'startup_ram_gib', 'retry_seconds', 'stable_seconds'):
        if type(policy.get(key)) not in (float, int) or not math.isfinite(policy[key]) or policy[key] <= 0:
            raise ValueError('Invalid operational resource policy: ' + key)
    if not policy['ram_emergency_gib'] < policy['ram_soft_gib'] < policy['startup_ram_gib']:
        raise ValueError('RAM recovery thresholds require hysteresis')
    if policy['retry_seconds'] < 30 or policy['stable_seconds'] < 10:
        raise ValueError('Recovery must not busy-loop or repeatedly reload the model')
    return policy


def startup_reason(sample, limits, policy):
    if sample['available_gib'] < policy['startup_ram_gib']:
        return 'waiting for host RAM before model loading'
    if sample['gpu_free_gib'] < policy['startup_gpu_free_gib']:
        return 'waiting for enough free VRAM for this model profile'
    if sample['disk_free_gib'] < limits['reserve_disk_gib']:
        return 'waiting for output disk space'
    return None


class Pressure:
    def __init__(self, policy, limits):
        self.policy, self.limits, self.low_since = policy, limits, None

    def check(self, sample, now, *, backend_active):
        """Return (reason, emergency). Ordinary low RAM must persist first."""
        if any(not math.isfinite(sample[k]) for k in
               ('available_gib', 'owned_rss_gib', 'gpu_free_gib', 'disk_free_gib')):
            return 'nonfinite resource sample', True
        if sample['owned_rss_gib'] > self.limits['max_rss_gib']:
            return 'owned RSS limit reached', True
        if not backend_active:
            self.low_since = None
            return None, False
        if sample['available_gib'] < self.policy['ram_emergency_gib']:
            return 'emergency host RAM shortage', True
        if sample['gpu_free_gib'] < self.policy['gpu_reserve_gib']:
            return 'GPU safety reserve reached', True
        if sample['disk_free_gib'] < self.limits['reserve_disk_gib']:
            return 'output disk reserve reached', False
        if sample['available_gib'] < self.policy['ram_soft_gib']:
            if self.low_since is None:
                self.low_since = now
            if now - self.low_since >= self.policy['ram_grace_seconds']:
                return 'sustained low host RAM', False
        else:
            self.low_since = None
        return None, False


def install(policy, gpu_total):
    """Adapt operational classes, never model inference or frozen sampling."""
    original_sample, original_backend, original_guard = worker.resource_sample, worker.Backend, worker.ResourceGuard

    def sample(pid, paths):
        value = original_sample(pid, paths)
        value.update(gpu_total_gib=gpu_total, gpu_free_gib=gpu_total-value['gpu_used_gib'])
        if any(not math.isfinite(v) for v in value.values()):
            raise ValueError('Nonfinite hardware accounting')
        return value

    class Backend(original_backend):
        def start(self, value):
            pid = self.existing()
            if not pid:
                reason = startup_reason(value, self.node['limits'], policy)
                if reason:
                    raise ResourceWait(reason)
                # The original >2GiB desktop-usage check is replaced by actual
                # free VRAM above. Retain its ownership/service-start handling.
                safe = dict(value, gpu_used_gib=0,
                            available_gib=max(value['available_gib'], self.node['limits']['reserve_ram_gib']))
                return super().start(safe)
            reason, _ = Pressure(policy, self.node['limits']).check(value,time.monotonic(),backend_active=True)
            if reason:
                raise ResourceWait(reason)

    class Guard(original_guard):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.pressure = Pressure(policy, self.node['limits'])

        def monitor(self):
            while not self.done.wait(2):
                try:
                    reason, emergency = self.pressure.check(self.sample(), time.monotonic(),
                                                           backend_active=bool(self.backend.pid))
                    if reason:
                        self.error = 'Resource wait: ' + reason
                        self.pause.set()
                        common.write_json(self.directory/'telemetry.json', dict(sampled_unix=time.time(),
                            current=self.last_sample, peaks=self.peaks, error=self.error,
                            action='stop_owned_backend' if emergency else 'finish_image_then_wait',
                            policy_fingerprint=policy['fingerprint']))
                        # Soft pressure completes/commits the current image.
                        if emergency:
                            self.backend.stop()
                        return
                except Exception as exc:
                    # Monitoring failures are NOT silently retried as pressure.
                    self.error = f'Resource accounting failed: {type(exc).__name__}: {exc}'
                    self.pause.set()
                    self.backend.stop()
                    return

    worker.resource_sample, worker.Backend, worker.ResourceGuard = sample, Backend, Guard


def protect_reservations(engine):
    """Recovery runs before a new reservation; never POST an uncertain job twice."""
    original=engine._append

    def append(path,row):
        path=Path(path)
        if path.name=='attempts.jsonl':
            prior=weather_io.read_rows(path)
            if any(value['sample_id']==row['sample_id'] for value in prior):
                raise RuntimeError('Unresolved reserved request; automatic duplicate POST forbidden: '+row['sample_id'])
        return original(path,row)

    engine._append=append


def waiting_status(directory, policy, reason, value, retry_at):
    path=directory/'status.json'
    previous=__import__('json').loads(path.read_text()) if path.exists() else {}
    previous.update(state='waiting_for_resources', backend_pid=0, pid=__import__('os').getpid(),
        updated_unix=time.time(), error=None, resource_wait_reason=reason,
        resource_policy_fingerprint=policy['fingerprint'], next_retry_unix=retry_at, resources=value)
    common.write_json(path,previous)


def recoverable(exc):
    return isinstance(exc, ResourceWait) or (isinstance(exc, RuntimeError) and str(exc).startswith('Resource wait: '))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--node-dir',type=Path,required=True)
    p.add_argument('--policy',type=Path,required=True)
    args=p.parse_args(argv);directory=args.node_dir.resolve()
    node=worker.load_node(directory);policy=load_policy(args.policy,node)
    output=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total',
        '--format=csv,noheader,nounits','-i','0'],text=True,timeout=10)
    gpu_total=float(output.strip())/1024
    if gpu_total <= policy['startup_gpu_free_gib']+policy['gpu_reserve_gib']:
        raise ValueError('Resource policy does not fit this GPU')
    install(policy,gpu_total)
    from experiments.qwen_curriculum import balanced_night
    protect_reservations(balanced_night.runtime().engine())
    stopped=threading.Event()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopped.set())
    marker=directory/'pause_requested';reason='initial resource check';retry_at=0;stable=None;failures=0
    while not stopped.is_set() and not marker.exists():
        value=worker.resource_sample(0,[directory])
        reason_now=startup_reason(value,node['limits'],policy)
        now=time.monotonic()
        if reason_now or time.time()<retry_at:
            stable=None
            waiting_status(directory,policy,reason_now or reason,value,retry_at)
            stopped.wait(10)
            continue
        if stable is None:stable=now
        if now-stable<policy['stable_seconds']:
            waiting_status(directory,policy,'resources stabilizing before model loading',value,retry_at)
            stopped.wait(10)
            continue
        try:
            worker.run_worker(directory)
            return
        except Exception as exc:
            if not recoverable(exc):
                raise
            reason=str(exc);failures+=1;stable=None
            retry_at=time.time()+min(900,policy['retry_seconds']*2**min(failures-1,3))
            common.write_json(directory/'resource_recovery.json',dict(version=1,reason=reason,
                policy_fingerprint=policy['fingerprint'],pressure_restarts=failures,next_retry_unix=retry_at))
            print(f'{reason}; backend released; automatic recovery after cooldown and stable resources',flush=True)
    path=directory/'status.json'
    if path.exists():
        import json
        value=json.loads(path.read_text());value.update(state='paused',backend_pid=0,error=None,next_retry_unix=None)
        common.write_json(path,value)


if __name__=='__main__':main()
