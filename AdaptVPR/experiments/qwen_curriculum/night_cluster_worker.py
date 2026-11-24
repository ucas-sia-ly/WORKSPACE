"""One machine's durable, resource-bounded night generation queue.

The controller assigns disjoint frozen shards. This worker never selects sources
or rewrites prompts, and uses the existing reservation/raw-recovery engine.
SIGTERM and ``pause_requested`` finish the current image before releasing Qwen.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import requests

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import common

GIB = 1024 ** 3


def sealed(value):
    body = {k: v for k, v in value.items() if k != 'fingerprint'}
    return {**body, 'fingerprint': common.fingerprint(body)}


def read_sealed(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or value != sealed(value):
        raise ValueError(f'Configuration checksum mismatch: {path}')
    return value


def machine_id():
    return Path('/etc/machine-id').read_text().strip()


def load_node(directory):
    node = read_sealed(directory / 'node.json')
    if node.get('version') != 1 or node.get('node') not in ('local', 'gpu2'):
        raise ValueError('Unsupported cluster worker node')
    if node.get('machine_id') != machine_id():
        raise ValueError('Node belongs to another machine')
    unit = node.get('backend_unit', '')
    if not re.fullmatch(r'[A-Za-z0-9_.@-]+\.service', unit):
        raise ValueError('A specific user backend service is required')
    if node.get('backend_scope', 'user') not in ('user', 'system'):
        raise ValueError('Unsupported backend systemd scope')
    if not isinstance(node.get('service_identity'), dict):
        raise ValueError('A frozen generator service identity is required')
    for name in ('max_gpu_gib', 'max_rss_gib', 'reserve_ram_gib', 'reserve_disk_gib'):
        value = node.get('limits', {}).get(name)
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError('Invalid resource limit: ' + name)
    for path, digest in node.get('files_sha256', {}).items():
        if common.file_sha256(path) != digest:
            raise ValueError('Worker code/input changed: ' + path)
    if node.get('worker_sha256') and common.file_sha256(__file__) != node['worker_sha256']:
        raise ValueError('Worker implementation changed')
    return node


def load_queue(directory, node, observed):
    """An atomic controller append can add work, never transfer existing work."""
    value = read_sealed(directory / 'queue.json')
    if value.get('version') != 1 or value.get('node') != node['node']:
        raise ValueError('Queue owner differs from this worker')
    entries = value.get('entries')
    if not isinstance(entries, list):
        raise ValueError('Queue entries must be a list')
    originals, states = set(), set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('Malformed queue entry')
        for name in ('state', 'original_state'):
            path = entry.get(name)
            if not isinstance(path, str) or not Path(path).is_absolute():
                raise ValueError('Queue paths must be absolute')
        for name in ('config_sha256', 'original_config_sha256'):
            if not re.fullmatch('[0-9a-f]{64}', str(entry.get(name, ''))):
                raise ValueError('Queue requires immutable plan-config checksums')
        if entry['original_state'] in originals or entry['state'] in states:
            raise ValueError('Duplicate shard assignment')
        originals.add(entry['original_state']); states.add(entry['state'])
    if entries[:len(observed)] != observed:
        raise ValueError('Previously observed queue prefix was changed or removed')
    return entries


def validate_entry(entry, node, engine):
    state = Path(entry['state'])
    if common.file_sha256(state / 'plan_config.json') != entry['config_sha256']:
        raise ValueError('Assigned shard configuration changed')
    if node['node'] == 'local':
        if (entry['state'] != entry['original_state'] or
                entry['config_sha256'] != entry['original_config_sha256']):
            raise ValueError('Local worker must use the original frozen shard')
    config, jobs = engine.load_plan(state)
    if config['service'] != node['service_identity']:
        raise ValueError('Shard generator identity differs from this node')
    origin = config.get('cluster_origin')
    if node['node'] == 'gpu2' and not isinstance(origin, dict):
        raise ValueError('Remote shard requires explicit origin provenance')
    if origin is not None:
        if (origin.get('node') != node['node'] or
                origin.get('original_config_sha256') != entry['original_config_sha256'] or
                origin.get('original_state', entry['original_state']) != entry['original_state']):
            raise ValueError('Derived shard origin differs from its assignment')
    if any(job['condition'] != 'night' for job in jobs):
        raise ValueError('Only the frozen night jobs are allowed')
    return config, jobs


def systemctl(*arguments, scope='user'):
    command = ['systemctl', '--user', *arguments] if scope == 'user' else ['systemctl', *arguments]
    if scope == 'system' and arguments[0] in ('start', 'stop'):
        command = ['sudo', '-n', *command]
    return subprocess.run(command, check=True,
                          text=True, capture_output=True, timeout=35).stdout.strip()


def service_pid(unit, scope='user'):
    return int(systemctl('show', unit, '-p', 'MainPID', '--value', scope=scope) or '0')


def process_tree_rss(roots):
    """RSS of our worker/backend trees; no unrelated user applications."""
    processes = {}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            processes[int(directory.name)] = (int(fields[1]), int(fields[21]))
        except (OSError, ValueError, IndexError):
            continue
    owned = {pid for pid in roots if pid}
    while True:
        children = {pid for pid, (parent, _) in processes.items() if parent in owned}
        if children <= owned:
            break
        owned.update(children)
    return sum(processes[pid][1] * os.sysconf('SC_PAGE_SIZE') for pid in owned if pid in processes) / GIB


def resource_sample(backend_pid, output_paths):
    fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(fields['MemAvailable'].split()[0]) * 1024 / GIB
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-i', '0'],
                            check=True, text=True, capture_output=True, timeout=10)
    gpu = float(result.stdout.strip()) / 1024
    disks = []
    for path in output_paths:
        path = Path(path)
        while not path.exists():
            path = path.parent
        disks.append(shutil.disk_usage(path).free / GIB)
    return dict(available_gib=available, owned_rss_gib=process_tree_rss((os.getpid(), backend_pid)),
                gpu_used_gib=gpu, disk_free_gib=min(disks))


def violation(sample, limits, *, before_start=False):
    for metric, bound, relation, message in (
        ('available_gib', 'reserve_ram_gib', 'below', 'Host RAM reserve reached'),
        ('owned_rss_gib', 'max_rss_gib', 'above', 'Owned process RSS limit reached'),
        ('gpu_used_gib', 'max_gpu_gib', 'above', 'GPU memory limit reached'),
        ('disk_free_gib', 'reserve_disk_gib', 'below', 'Output disk reserve reached')):
        value = sample[metric]
        if not math.isfinite(value) or (value < limits[bound] if relation == 'below' else value > limits[bound]):
            return message
    if before_start and sample['gpu_used_gib'] > 2:
        return 'GPU is already in use; refusing to interfere with another task'
    return None


class Backend:
    def __init__(self, directory, node):
        self.directory, self.node, self.pid = directory, node, 0
        self.scope = node.get('backend_scope', 'user')
        self.owner_file = directory / 'backend_owner.json'

    def owner(self):
        return dict(node=self.node['node'], node_fingerprint=self.node['fingerprint'],
                    machine_id=self.node['machine_id'], backend_unit=self.node['backend_unit'],
                    backend_scope=self.scope, pid=self.pid)

    def existing(self):
        pid = service_pid(self.node['backend_unit'], self.scope)
        if pid:
            if not self.owner_file.exists():
                raise RuntimeError('Backend is active without this node\'s ownership receipt')
            owner = read_sealed(self.owner_file)
            self.pid = pid
            if owner != sealed(self.owner()):
                self.pid = 0
                raise RuntimeError('Backend belongs to another worker or has restarted')
        return pid

    def start(self, sample):
        pid = self.existing()
        reason = violation(sample, self.node['limits'], before_start=not pid)
        if reason:
            raise RuntimeError(reason)
        if not pid:
            systemctl('start', self.node['backend_unit'], scope=self.scope)
            self.pid = service_pid(self.node['backend_unit'], self.scope)
            if not self.pid:
                raise RuntimeError('Backend service did not start')
            common.write_json(self.owner_file, sealed(self.owner()))

    def health(self, pause):
        from experiments.qwen_curriculum import run
        deadline = time.monotonic() + 240
        url = self.node.get('health_url', 'http://127.0.0.1:8001/health')
        with requests.Session() as session:
            session.trust_env = False
            while True:
                if pause.is_set():
                    raise PauseRequested()
                try:
                    response = session.get(url, timeout=10)
                    response.raise_for_status()
                    identity = run.service_identity(response.json())
                except (requests.RequestException, RuntimeError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Backend health did not become ready')
                    pause.wait(2)
                    continue
                if identity != self.node['service_identity']:
                    raise ValueError('Actual backend differs from the frozen node identity')
                return

    def stop(self):
        # Never stop a replacement service started by a different task.
        if self.pid and service_pid(self.node['backend_unit'], self.scope) == self.pid:
            systemctl('stop', self.node['backend_unit'], scope=self.scope)
        self.pid = 0


class ResourceGuard:
    def __init__(self, directory, node, backend, pause):
        self.directory, self.node, self.backend, self.pause = directory, node, backend, pause
        self.paths, self.error = [directory], None
        self.peaks, self.last_sample, self.done = {}, {}, threading.Event()
        self.thread = threading.Thread(target=self.monitor, daemon=True)

    def sample(self):
        value = resource_sample(self.backend.pid, tuple(self.paths))
        self.last_sample = value
        for key in ('gpu_used_gib', 'owned_rss_gib'):
            self.peaks[key] = max(value[key], self.peaks.get(key, 0))
        for key in ('available_gib', 'disk_free_gib'):
            self.peaks[key] = min(value[key], self.peaks.get(key, value[key]))
        common.write_json(self.directory / 'telemetry.json', dict(
            sampled_unix=time.time(), current=value, peaks=self.peaks, error=self.error))
        return value

    def monitor(self):
        while not self.done.wait(2):
            try:
                reason = violation(self.sample(), self.node['limits'])
                if reason:
                    raise RuntimeError(reason)
            except Exception as exc:
                self.error = f'{type(exc).__name__}: {exc}'
                self.pause.set()
                try:
                    common.write_json(self.directory / 'telemetry.json', dict(sampled_unix=time.time(),
                        current=self.last_sample, peaks=self.peaks, error=self.error))
                except OSError:
                    pass
                try:
                    self.backend.stop()
                except Exception as stop_error:
                    self.error += f'; backend stop: {stop_error}'
                return

    def close(self):
        self.done.set()
        self.thread.join(timeout=45)


class PauseRequested(KeyboardInterrupt):
    """A safe pause after committing the active image."""


@contextmanager
def commit_pause(engine, pause, requested_file):
    """Check only after publication of a newly committed result, not a POST."""
    original, append = engine.publish, engine._append
    committed = set()

    def record(path, row):
        append(path, row)
        if Path(path).name == 'results.jsonl':
            committed.add(str(Path(path).parent))

    def publish(directory, config, jobs, generated, results, attempts, state, domain, error=None):
        value = original(directory, config, jobs, generated, results, attempts, state, domain, error)
        key = str(directory)
        if state == 'domain_complete' or (state == 'generating' and key in committed):
            committed.discard(key)
            if pause.is_set() or requested_file.exists():
                pause.set()
                raise PauseRequested()
        return value

    engine.publish, engine._append = publish, record
    try:
        yield
    finally:
        engine.publish, engine._append = original, append


def run_worker(directory, *, poll_seconds=10):
    from experiments.qwen_curriculum import balanced_night, run
    directory = Path(directory).resolve()
    node = load_node(directory)
    engine = balanced_night.runtime().engine()
    backend = Backend(directory, node)
    pause, started = threading.Event(), time.time()
    requested = directory / 'pause_requested'
    guard = ResourceGuard(directory, node, backend, pause)
    previous_handlers = {}
    observed_file = directory / 'observed_queue.json'
    observed = []
    if observed_file.exists():
        snapshot = read_sealed(observed_file)
        if snapshot.get('node_fingerprint') != node['fingerprint']:
            raise ValueError('Observed queue belongs to another node configuration')
        observed = snapshot['entries']
    completed_file = directory / 'completed.json'
    completed = []
    if completed_file.exists():
        snapshot = read_sealed(completed_file)
        if snapshot.get('node_fingerprint') != node['fingerprint']:
            raise ValueError('Completion ledger belongs to another node')
        completed = snapshot['entries']
    active, error, backend_ready = None, None, False

    def status(state):
        value = dict(version=1, node=node['node'], node_fingerprint=node['fingerprint'], state=state,
            active_state=active, queue_entries=len(observed), completed_entries=len(completed),
            pid=os.getpid(), backend_pid=backend.pid, started_unix=started, updated_unix=time.time(),
            wall_seconds=time.time()-started, error=error, verification='deferred; not approved training data')
        common.write_json(directory / 'status.json', value)
        return value

    with run.run_lock(directory):
        try:
            for name in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[name] = signal.signal(name, lambda *_: pause.set())
            backend.existing()
            guard.thread.start()
            with commit_pause(engine, pause, requested):
                while not pause.is_set() and not requested.exists():
                    entries = load_queue(directory, node, observed)
                    if entries != observed:
                        observed = entries
                        common.write_json(observed_file, sealed(dict(version=1,
                            node_fingerprint=node['fingerprint'], entries=observed)))
                    if completed != entries[:len(completed)]:
                        raise ValueError('Completion ledger differs from the assigned queue prefix')
                    cursor = len(completed)
                    if cursor == len(entries):
                        active = None
                        status('waiting_for_work')
                        pause.wait(poll_seconds)
                        continue
                    entry = entries[cursor]
                    active = entry['state']
                    config, _ = validate_entry(entry, node, engine)
                    guard.paths = [directory, Path(config['image_root']), Path(active)]
                    if not backend_ready:
                        backend.start(guard.sample())
                        backend.health(pause)
                        backend_ready = True
                    status('generating')
                    summary = engine.generate(SimpleNamespace(output_dir=Path(active), domain='night',
                        max_calls=0, qwen_url=node.get('health_url', 'http://127.0.0.1:8001/health').rsplit('/', 1)[0]+'/generate',
                        request_timeout=1800, device='cpu'))
                    if summary['state'] != 'domain_complete':
                        raise RuntimeError('Shard did not complete: '+summary['state'])
                    completed.append(entry)
                    common.write_json(completed_file, sealed(dict(version=1,
                        node_fingerprint=node['fingerprint'], entries=completed)))
                    status('shard_complete')
                if guard.error:
                    raise RuntimeError(guard.error)
        except PauseRequested:
            error = guard.error
            if error:
                raise RuntimeError(error)
        except Exception as exc:
            error = guard.error or f'{type(exc).__name__}: {exc}'
            raise
        finally:
            if guard.thread.ident:
                guard.close()
            try:
                backend.stop()
            except Exception as exc:
                error = error or f'Backend shutdown: {exc}'
            for name, handler in previous_handlers.items():
                signal.signal(name, handler)
            status('failed' if error else 'paused')
    return status('failed' if error else 'paused')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('command', choices=('run', 'status'), nargs='?', default='run')
    parser.add_argument('--node-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'status':
        print((args.node_dir / 'status.json').read_text())
    else:
        run_worker(args.node_dir)


if __name__ == '__main__':
    main()
