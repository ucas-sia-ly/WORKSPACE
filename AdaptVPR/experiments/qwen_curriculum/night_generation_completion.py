"""Finish original non-mercury work, then its 29-style mercury replacements.

Dispatch uses authenticated completion ledgers, independently of slow image
collection. Image collection still verifies every artifact before primary commit
or remote pruning. No image POST, source or historical plan is rewritten.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import night_original_queue_recovery as recovery
from experiments.qwen_curriculum import night_mercury_prompts as prompts
from experiments.qwen_curriculum import night_mercury_images as mercury

base = recovery.base
common = base.common
cluster = base.cluster
ROOT = common.WORKSPACE_ROOT/'outputs/night_generation_completion_v1'
UNIT = 'ccr-night-completion.service'


def campaign(name):
    if name == 'original':
        recovery.original.configure()
        return prompts.OLD
    mercury.configure()
    return mercury.ROOT


def completion_entries(root, node):
    path = root/'nodes'/node/'completed.json'
    if node == 'gpu2':
        raw = cluster.remote(cluster.REMOTE_PY, '-c',
            'import sys;from pathlib import Path;p=Path(sys.argv[1]);print(p.read_text() if p.exists() else "null")',
            str(path))
        value = json.loads(raw)
    else:
        value = base.worker.read_sealed(path) if path.exists() else None
    if value is None:
        return []
    node_cfg = base.worker.read_sealed(root/'nodes'/node/'node.json')
    queue = cluster.queue(root, node)['entries']
    if (value != base.worker.sealed(value) or value['node_fingerprint'] != node_cfg['fingerprint']
            or value['entries'] != queue[:len(value['entries'])]):
        raise ValueError('Authenticated completion ledger is not the owned queue prefix')
    return value['entries']


def grant(root, node, completed):
    """Same durable ownership; collection lag must not idle the remote GPU."""
    cfg = base.load(root)
    if node == 'gpu2':
        recovery.reconcile(root)
    queues = {n: cluster.queue(root, n) for n in ('local', 'gpu2')}
    if len(queues[node]['entries']) > len(completed):
        return False
    feed = base.worker.read_sealed(root/'feed.json')
    owner = base.worker.read_sealed(root/'ownership.json')
    if owner['cluster_fingerprint'] != cfg['fingerprint']:
        raise ValueError('Ownership protocol changed')
    descriptor = cluster.select_assignment(dict(shards=feed['entries'], partial_local_only=[]), owner, queues, node)
    if descriptor is None:
        return False
    owner['owners'][descriptor['original_state']] = node
    cluster.save(root/'ownership.json', owner)
    entry = cluster.stage_assignment(root, descriptor, node)
    if node == 'gpu2':
        import shutil
        source = Path(descriptor['original_state'])/'prompt_receipts.jsonl'
        target = Path(entry['state'])/'prompt_receipts.jsonl'
        if target.exists() and common.file_sha256(target) != common.file_sha256(source):
            raise ValueError('Derived execution receipts changed')
        if not target.exists():
            shutil.copy2(source, target)
        cluster.rsync_files(root, [target])
    queues[node]['entries'].append(entry)
    cluster.save(root/'nodes'/node/'queue.json', queues[node])
    if node == 'gpu2':
        cluster.rsync_files(root, [root/'nodes/gpu2/queue.json'])
    return True


def pause_marker(root, node, paused):
    path = root/'nodes'/node/'pause_requested'
    path.parent.mkdir(parents=True, exist_ok=True)
    if paused:
        path.touch()
    else:
        path.unlink(missing_ok=True)
    if node == 'gpu2':
        cluster.remote('mkdir', '-p', str(path.parent))
        cluster.remote('touch' if paused else 'rm', *([] if paused else ['-f']), str(path))


def dispatch(name, node):
    root = campaign(name); base.load(root)
    if (ROOT/('pause_'+node)).exists():
        pause_marker(root, node, True)
        return dict(state='paused')
    if node == 'local':
        pid = int(cluster.call(['systemctl', '--user', 'show', 'ccr-night-mercury-prompts.service',
                                '-p', 'MainPID', '--value']) or 0)
        if pid or not (prompts.ROOT/'state/prompts_complete.json').exists():
            return dict(state='waiting_for_prompts')
    completed = completion_entries(root, node)
    feed = base.worker.read_sealed(root/'feed.json')
    owner = base.worker.read_sealed(root/'ownership.json')
    queue = cluster.queue(root, node)['entries']
    queued_states = {e['original_state'] for e in queue}
    unpublished = any(n == node and state not in queued_states for state, n in owner['owners'].items())
    if (len(owner['owners']) == len(feed['entries'])
            and len(completed) == len(queue) and not unpublished):
        pause_marker(root, node, True)
        return dict(state='node_complete', shards=len(completed))
    pause_marker(root, node, False)
    added = grant(root, node, completed)
    state = base.active(node)
    if state == 'inactive' and len(cluster.queue(root, node)['entries']) > len(completed):
        cluster.command_node(node, 'systemctl', '--user', 'start', base.unit(node))
    elif state == 'failed':
        raise RuntimeError('Worker failed; inspect logs before explicit resume: '+base.unit(node))
    return dict(state='dispatched' if added else 'working', completed_shards=len(completed),
                queued_shards=len(cluster.queue(root, node)['entries']))


def child(*args, timeout=240):
    return cluster.call([cluster.LOCAL_PY, '-B', str(Path(__file__).resolve()), *args], timeout=timeout)


def controller():
    ROOT.mkdir(parents=True, exist_ok=True)
    stopped = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.set())
    with (ROOT/'controller.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not stopped.is_set():
            progress = dict(updated_unix=time.time(), nodes={}, errors={})
            if ((prompts.ROOT/'state/prompts_complete.json').exists()
                    and not (prompts.ROOT/'deployment/ready.json').exists()):
                try:
                    child('prepare-mercury', timeout=600)
                except Exception as exc:
                    progress['errors']['prepare'] = str(exc)
            for node in ('gpu2', 'local'):
                try:
                    value = json.loads(child('step', '--campaign', 'original', '--node', node).splitlines()[-1])
                    name = 'original'
                    if value['state'] == 'node_complete' and (prompts.ROOT/'deployment/ready.json').exists():
                        # The old worker finishes its active image and releases
                        # the shared backend before the next campaign starts.
                        campaign('original')
                        if base.active(node) == 'inactive':
                            name = 'mercury'
                            value = json.loads(child('step', '--campaign', name, '--node', node).splitlines()[-1])
                    progress['nodes'][node] = dict(campaign=name, **value)
                except Exception as exc:
                    progress['errors'][node] = str(exc)
            common.write_json(ROOT/'status.json', progress)
            print(json.dumps(progress), flush=True)
            stopped.wait(30)


def prepare_mercury():
    mercury.configure(); mercury.prepare()
    root = prompts.ROOT
    cfg = base.load(root)
    files = list(map(Path, cfg['files_sha256']))
    files += [root/'cluster.json', root/'ownership.json', root/'feed.json']
    files += [root/'nodes/gpu2'/n for n in ('node.json', 'resource_policy.json', 'queue.json', 'pause_requested')]
    cluster.rsync_files(root, files)
    unit_path = root/'deployment/ccr-night-mercury-gpu2-worker.service'
    unit_path.write_text(worker_unit('gpu2'))
    cluster.rsync_files(root, [unit_path])
    cluster.remote('cp', str(unit_path), '/home/epic/.config/systemd/user/'+unit_path.name)
    cluster.remote('systemctl', '--user', 'daemon-reload')
    local = Path.home()/'.config/systemd/user/ccr-night-mercury-local-worker.service'
    local.write_text(worker_unit('local'))
    cluster.call(['systemctl', '--user', 'daemon-reload'])
    # Separate image collection continues even if dispatch or prompting pauses.
    cluster.call(['systemctl', '--user', 'start', 'ccr-night-collect-mercury.service'])
    cluster.freeze(root/'deployment/ready.json', dict(cluster_fingerprint=cfg['fingerprint'],
        remote_worker_installed=True, local_worker_installed=True))


def worker_unit(node):
    python = cluster.LOCAL_PY if node == 'local' else cluster.REMOTE_PY
    return ('[Unit]\nDescription=Original-policy mercury replacement '+node+'\nAfter=network.target\n'
        '[Service]\nType=exec\nWorkingDirectory='+str(common.WORKSPACE_ROOT)+'\n'
        'Environment=PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2\n'
        'ExecStart='+python+' -B '+str(Path(mercury.__file__).resolve())+' worker --node '+node+'\n'
        'MemoryAccounting=yes\nMemoryMax=2G\nMemorySwapMax=0\nCPUQuota=200%\n'
        'TimeoutStopSec=1900\nKillMode=mixed\nRestart=no\n')


def collect(name, once=False):
    root = campaign(name)
    original_call = cluster.call
    def transfer_call(command, **kwargs):
        if (command and command[0] == 'rsync' and
                str(root/'incoming/gpu2')+'/' == command[-1]):
            # Interrupted bulk transfers keep partial bytes. A weak but live
            # Wi-Fi link may take much longer than the old 600s wall timeout.
            command = [*command[:1], '--partial', '--partial-dir=.rsync-partial',
                       '--timeout=90', *command[1:]]
            kwargs['timeout'] = 7200
        return original_call(command, **kwargs)
    cluster.call = transfer_call
    while True:
        try:
            value = dict(collected=cluster.sync(root), remote_removed=base.prune_confirmed(root))
            print(json.dumps(value), flush=True)
            base.status(root)
        except Exception as exc:
            # rsync retains completed files; a transient Wi-Fi failure never
            # reassigns an image or stops the independent dispatch loop.
            print(f'Collection will retry: {exc}', flush=True)
            if once:
                raise
        if once:
            return
        time.sleep(30)


def controls(action, node):
    ROOT.mkdir(parents=True, exist_ok=True)
    for n in ('local', 'gpu2') if node == 'all' else (node,):
        marker = ROOT/('pause_'+n)
        if action == 'pause':
            marker.touch()
        else:
            marker.unlink(missing_ok=True)
        for name in ('original', 'mercury'):
            root = campaign(name)
            if (root/'cluster.json').exists():
                pause_marker(root, n, True if action == 'pause' else name == 'mercury')
                # reset-failed returns 1 for healthy inactive units on some
                # systemd releases. Only invoke it for an actual failed unit.
                if action == 'resume' and base.active(n) == 'failed':
                    cluster.command_node(n, 'systemctl', '--user', 'reset-failed', base.unit(n))
    if action == 'resume':
        local_allowed = not (ROOT/'pause_local').exists()
        if local_allowed:
            (prompts.ROOT/'state/pause_requested').unlink(missing_ok=True)
        if local_allowed and not (prompts.ROOT/'state/prompts_complete.json').exists():
            cluster.call(['systemctl', '--user', 'start', 'ccr-night-mercury-prompts.service'])
        cluster.call(['systemctl', '--user', 'start', UNIT, 'ccr-night-collect-original.service'])
        if (prompts.ROOT/'prepared.json').exists():
            cluster.call(['systemctl', '--user', 'start', 'ccr-night-collect-mercury.service'])
    elif node in ('local', 'all'):
        (prompts.ROOT/'state/pause_requested').touch()


def manifest():
    seen_ids = set(); seen_places = set()
    def rows():
        paths = [prompts.SNAPSHOT/'retained_manifest.jsonl']
        for root in (prompts.OLD, prompts.ROOT):
            if not (root/'cluster.json').exists():
                continue
            for n in ('local', 'gpu2'):
                paths += [Path(e['state'])/'results.jsonl' for e in cluster.queue(root, n)['entries']]
        for path in paths:
            for row in base.weather_io.read_rows(path):
                place = row['city'], int(row['place_id'])
                if row['sample_id'] in seen_ids or place in seen_places or row['style_id'] == 'mercury_vapor':
                    raise ValueError('Duplicate image/place or excluded mercury in final manifest')
                base.run.validate_image(row, 'output_path', 'output_sha256')
                seen_ids.add(row['sample_id']); seen_places.add(place)
                yield row
    path = ROOT/'generated_manifest.jsonl'
    common.write_jsonl(path, rows())
    return dict(manifest=str(path), verified_images=len(seen_ids), target=61984,
                verification='files intact; image quality review still pending')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('controller', 'step', 'prepare-mercury', 'collect', 'sync',
                                      'status', 'pause', 'resume', 'manifest'))
    p.add_argument('--campaign', choices=('original', 'mercury'), default='original')
    p.add_argument('--node', choices=('all', 'local', 'gpu2'), default='all')
    args = p.parse_args()
    if args.command == 'controller':
        controller()
    elif args.command == 'prepare-mercury':
        prepare_mercury()
    elif args.command == 'step':
        if args.node == 'all':
            raise ValueError('Choose one dispatcher node')
        print(json.dumps(dispatch(args.campaign, args.node)))
    elif args.command in ('collect', 'sync'):
        collect(args.campaign, once=args.command == 'sync')
    elif args.command in ('pause', 'resume'):
        controls(args.command, args.node)
    elif args.command == 'manifest':
        print(json.dumps(manifest(), indent=2))
    else:
        value = json.loads((ROOT/'status.json').read_text()) if (ROOT/'status.json').exists() else {}
        value['prompts'] = json.loads((prompts.ROOT/'state/progress.json').read_text())
        value['images'] = {}
        for name in ('original', 'mercury'):
            root = campaign(name)
            if (root/'cluster.json').exists():
                value['images'][name] = base.status(root)
        print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
