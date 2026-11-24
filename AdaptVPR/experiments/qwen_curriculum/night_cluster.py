"""Manage disjoint night-generation queues over authenticated SSH.

Assignments survive disconnections. A missing heartbeat never authorizes a
second machine to repeat a request. Original frozen plans remain unchanged.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import common
from experiments.qwen_curriculum.night_cluster_worker import read_sealed, sealed

ROOT = common.WORKSPACE_ROOT
DEFAULT = ROOT / 'outputs/night_cluster_v1'
COHORT = ROOT / 'outputs/night_place_fast_v1/state'
LOCAL_PY = '/home/admin123/miniconda3/envs/AdaptVPR/bin/python'
REMOTE_PY = '/home/epic/anaconda3/envs/ccr-night/bin/python'
REMOTE = 'ccr-gpu2'
WORKER = Path(__file__).with_name('night_cluster_worker.py')


def call(command, **kwargs):
    return subprocess.run(command, check=True, text=True, capture_output=True, **kwargs).stdout.strip()


def remote(*command, timeout=60):
    return call(['ssh', '-o', 'ConnectTimeout=10', REMOTE, shlex.join(command)], timeout=timeout)


def save(path, body):
    common.write_json(path, sealed(body))


def freeze(path, body):
    value = sealed(body)
    if path.exists():
        if read_sealed(path) != value:
            raise ValueError('Frozen configuration differs: '+str(path))
    else:
        common.write_json(path, value)
    return value


def queue(root, node):
    return read_sealed(root/'nodes'/node/'queue.json')


def cohort_marker():
    value=json.loads((COHORT/'prompts_complete.json').read_text())
    if value.get('result_sha256')!=common.fingerprint({k:v for k,v in value.items() if k!='result_sha256'}):
        raise ValueError('Cohort completion marker checksum mismatch')
    return value


def command_node(node, *command, timeout=60):
    return remote(*command, timeout=timeout) if node == 'gpu2' else call(list(command), timeout=timeout)


def unit(node):
    return f'ccr-night-{node}-worker.service'


def rsync_files(root, paths):
    """Transfer only explicit workspace-relative inputs; never secrets/envs."""
    names = []
    for path in sorted(set(map(Path, paths))):
        relative = path.resolve().relative_to(ROOT).as_posix()
        if '\n' in relative or '\r' in relative or path.is_symlink():
            raise ValueError('Unsafe transfer input: '+str(path))
        names.append(relative)
    with tempfile.NamedTemporaryFile('w', dir=root/'deployment', suffix='.files') as handle:
        handle.write('\n'.join(names)+'\n'); handle.flush()
        call(['rsync', '-a', '--modify-window=-1', '--partial', '--files-from='+handle.name,
              str(ROOT)+'/', REMOTE+':'+str(ROOT)+'/'], timeout=600)


def prepare(root):
    marker = cohort_marker()
    if (root/'cluster.json').exists():
        config=read_sealed(root/'cluster.json')
        if config['cohort_marker_sha256']!=common.file_sha256(COHORT/'prompts_complete.json'):
            raise ValueError('Original cohort completion marker changed')
        for item in marker['shards']:
            if common.file_sha256(Path(item['state'])/'plan_config.json')!=item['config_sha256']:
                raise ValueError('Original shard changed: '+item['state'])
        return initialize(root,config)
    shards, historical, partial = [], 0, []
    for item in marker['shards']:
        state = Path(item['state'])
        digest = common.file_sha256(state/'plan_config.json')
        if digest != item['config_sha256']:
            raise ValueError('Original shard changed: '+str(state))
        rows = common.read_jsonl(state/'results.jsonl') if (state/'results.jsonl').exists() else []
        historical += len(rows)
        if len(rows) == item['jobs']:
            continue
        descriptor = dict(original_state=str(state), original_config_sha256=digest,
                          jobs=item['jobs'], saved_at_creation=len(rows))
        shards.append(descriptor)
        if (state/'attempts.jsonl').exists() and (state/'attempts.jsonl').stat().st_size:
            partial.append(str(state))
    (root/'deployment').mkdir(parents=True, exist_ok=True)
    config = freeze(root/'cluster.json', dict(version=1, cohort_state=str(COHORT),
        cohort_marker_sha256=common.file_sha256(COHORT/'prompts_complete.json'),
        planned=marker['ready'], historical_saved=historical, shards=shards,
        partial_local_only=partial, remote=REMOTE,
        scheduling='one in-flight shard per node; durable ownership; no timeout reassignment',
        verification='deferred; generated files are not approved training data'))
    return initialize(root,config)


def initialize(root,config):
    (root/'deployment').mkdir(parents=True,exist_ok=True)
    if not (root/'ownership.json').exists():
        save(root/'ownership.json', dict(version=1, cluster_fingerprint=config['fingerprint'],
                                       owners={state:'local' for state in config['partial_local_only']}))
    for node in ('local', 'gpu2'):
        directory = root/'nodes'/node
        directory.mkdir(parents=True, exist_ok=True)
        if not (directory/'queue.json').exists():
            save(directory/'queue.json', dict(version=1, node=node, entries=[]))
    guard = COHORT/'cluster_ownership.json'
    freeze(guard, dict(version=1, cluster_root=str(root), cluster_fingerprint=config['fingerprint']))
    return config


def select_assignment(config, ownership, queues, node):
    """An interrupted shard is never moved away from its original machine."""
    assigned = {e['original_state'] for q in queues.values() for e in q['entries']}
    pending = [s for s in config['shards'] if ownership['owners'].get(s['original_state']) == node
               and s['original_state'] not in assigned]
    if pending:
        return pending[0]
    for descriptor in config['shards']:
        state = descriptor['original_state']
        if state in ownership['owners'] or state in assigned:
            continue
        if node != 'local' and state in config['partial_local_only']:
            continue
        return descriptor
    return None


def stage_assignment(root, descriptor, node):
    original = Path(descriptor['original_state'])
    if common.file_sha256(original/'plan_config.json') != descriptor['original_config_sha256']:
        raise ValueError('Assigned original config changed')
    if node == 'local':
        return dict(**{k:v for k,v in descriptor.items() if k.startswith('original_')},
                    state=str(original), config_sha256=descriptor['original_config_sha256'])
    target = root/'nodes/gpu2/shards'/original.parent.name/'state'
    target.mkdir(parents=True, exist_ok=True)
    files = []
    for name in ('plan.jsonl','planner_config.json','coverage_snapshot.json','source_snapshot.jsonl'):
        source = original/name; destination = target/name
        if destination.exists() and common.file_sha256(destination) != common.file_sha256(source):
            raise ValueError('Derived frozen input changed')
        if not destination.exists(): shutil.copy2(source, destination)
        files.extend((source, destination))
    original_config = read_sealed(original/'plan_config.json')
    config = {k:v for k,v in original_config.items() if k != 'fingerprint'}
    config.update(service=read_sealed(root/'nodes/gpu2/node.json')['service_identity'],
        image_root=str(root/'nodes/gpu2/images'), cluster_origin=dict(node=node,
        original_state=str(original), original_config_sha256=descriptor['original_config_sha256'],
        original_fingerprint=original_config['fingerprint'], policy='same frozen jobs; explicit hardware execution profile'))
    freeze(target/'plan_config.json', config)
    files.extend((original/'plan_config.json', target/'plan_config.json'))
    files.extend(map(Path, original_config['files_sha256']))
    jobs = common.read_jsonl(original/'plan.jsonl')
    for job in jobs:
        source = Path(job['source_path'])
        if common.file_sha256(source) != job['source_sha256']:
            raise ValueError('Source changed before transfer: '+str(source))
        files.append(source)
    rsync_files(root, files)
    return dict(original_state=str(original), original_config_sha256=descriptor['original_config_sha256'],
                state=str(target), config_sha256=common.file_sha256(target/'plan_config.json'))


def transfer_complete(root=DEFAULT):
    receipt=root/'deployment/remote_ready.json'
    if receipt.exists():
        value=read_sealed(receipt)
        return value.get('manifest_sha256')==common.file_sha256(root/'deployment/weight_bundle.json')
    state = call(['systemctl','--user','show','ccr-gpu2-transfer.service','-p','ActiveState','-p','Result','-p','ExecMainStatus'])
    return all(term in state.splitlines() for term in ('ActiveState=inactive','Result=success','ExecMainStatus=0'))


def grant(root, node):
    config = read_sealed(root/'cluster.json')
    ownership = read_sealed(root/'ownership.json')
    if ownership['cluster_fingerprint'] != config['fingerprint']:
        raise ValueError('Ownership belongs to a different cluster')
    queues = {n:queue(root,n) for n in ('local','gpu2')}
    q = queues[node]
    # Limit outstanding work, so faster workers naturally receive more shards.
    complete = 0
    for entry in q['entries']:
        summary = Path(entry['state'])/'summary.json'
        if summary.exists() and json.loads(summary.read_text()).get('state') == 'domain_complete': complete += 1
    if len(q['entries'])-complete >= 1:
        return None
    descriptor = select_assignment(config, ownership, queues, node)
    if descriptor is None:
        return None
    owners = ownership['owners']
    state = descriptor['original_state']
    if state in owners and owners[state] != node:
        raise ValueError('Cannot transfer ownership of an existing assignment')
    owners[state] = node
    # Reserve ownership before any transfer; crashes cannot issue it elsewhere.
    save(root/'ownership.json', ownership)
    entry = stage_assignment(root, descriptor, node)
    q['entries'].append(entry)
    save(root/'nodes'/node/'queue.json', q)
    if node == 'gpu2':
        rsync_files(root, [root/'nodes/gpu2/queue.json'])
    return entry


def sync(root):
    with (root/'sync.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return sync_locked(root)


def sync_locked(root):
    """Read remote state through staging; publish complete verified images first."""
    q = queue(root,'gpu2')
    staging = root/'incoming/gpu2'
    staging.mkdir(parents=True, exist_ok=True)
    remote_root = root/'nodes/gpu2'
    call(['rsync','-a','--exclude=node.json','--exclude=queue.json','--exclude=*.lock',
          REMOTE+':'+str(remote_root)+'/',str(staging)+'/'], timeout=600)
    collected = 0
    for entry in q['entries']:
        state = Path(entry['state'])
        if common.file_sha256(state/'plan_config.json')!=entry['config_sha256']:
            raise ValueError('Collected shard configuration changed')
        config=read_sealed(state/'plan_config.json')
        relative = state.relative_to(remote_root)
        incoming = staging/relative
        journal = incoming/'results.jsonl'
        if not journal.exists(): continue
        # The producer appends; a concurrently copied incomplete last line waits.
        from experiments.qwen_curriculum import weather_io
        rows = weather_io.read_rows(journal)
        previous=weather_io.read_rows(state/'results.jsonl')
        if rows[:len(previous)]!=previous:
            raise ValueError('Remote committed prefix regressed or changed')
        jobs = {j['sample_id']:j for j in common.read_jsonl(state/'plan.jsonl')}
        generated_rows=weather_io.read_rows(incoming/'generated.jsonl')
        generated = {r['sample_id']:r for r in generated_rows}
        if len(generated)!=len(generated_rows) or len({r.get('sample_id') for r in rows})!=len(rows):
            raise ValueError('Duplicate remote image commit')
        attempts=weather_io.read_rows(incoming/'attempts.jsonl')
        reservations={}
        for number,attempt in enumerate(attempts,start=1):
            job=jobs.get(attempt.get('sample_id'))
            if (job is None or attempt.get('attempt_sha256')!=common.fingerprint({k:v for k,v in attempt.items() if k!='attempt_sha256'})
                or attempt.get('call_number')!=number or number>config['request_budget']
                or attempt.get('workspace')!=str(state/'requests'/f'a{number:08d}')
                or attempt.get('record_sha256')!=job['record_sha256']
                or any(attempt.get(k)!=job[k] for k in ('seed','prompt','source_sha256'))
                or attempt.get('plan_sha256')!=config['plan_sha256']):
                raise ValueError('Invalid remote request reservation')
            reservations[attempt['attempt_sha256']]=attempt
        for row in rows[len(previous):]:
            body = {k:v for k,v in row.items() if k != 'result_sha256'}
            job = jobs.get(row.get('sample_id'))
            if (row.get('result_sha256') != common.fingerprint(body) or job is None
                    or any(row.get(k) != v for k,v in job.items()) or row['sample_id'] not in generated
                    or row.get('status') != 'pending_verification' or row.get('eligible_for_training') is not False):
                raise ValueError('Remote result differs from its frozen job/provenance')
            generated_row=generated[row['sample_id']]
            if (generated_row.get('result_sha256')!=common.fingerprint({k:v for k,v in generated_row.items() if k!='result_sha256'})
                or any(row.get(k)!=v for k,v in generated_row.items() if k!='result_sha256')
                or row.get('origin')!='reserved_qwen_request'
                or row.get('attempt_sha256') not in reservations):
                raise ValueError('Invalid remote generated record or missing durable reservation')
            from PIL import Image
            publishing=[]
            for field in ('output_path','raw_output_path'):
                destination = Path(row[field])
                expected = remote_root/'images'/('raw' if field=='raw_output_path' else '')/'night'/(row['sample_id']+'.png')
                if destination != expected:
                    raise ValueError('Remote output escapes assigned image directory')
                source = staging/destination.relative_to(remote_root)
                reject_symlinks(source,staging)
                digest = row[field.replace('_path','_sha256')]
                if common.file_sha256(source) != digest:
                    raise ValueError('Remote image SHA mismatch')
                with Image.open(source) as image:
                    image.load()
                    if image.format != 'PNG' or (field=='output_path' and list(image.size)!=job['source_dimensions']):
                        raise ValueError('Remote image format/size mismatch')
                publishing.append((source,destination,digest))
            sidecar = Path(row['raw_output_path']).with_suffix('.json')
            candidate = staging/sidecar.relative_to(remote_root)
            reject_symlinks(candidate,staging)
            if common.file_sha256(candidate) != row['metadata_sha256']:
                raise ValueError('Remote sidecar SHA mismatch')
            from experiments.qwen_curriculum import resume_recovery,run
            metadata=json.loads(candidate.read_text())
            resume_recovery._validate_metadata(metadata,Path(reservations[row['attempt_sha256']]['workspace']),
                run.service_output_directory(),job,{'service':config['service']})
            with Image.open(staging/Path(row['raw_output_path']).relative_to(remote_root)) as image:
                if list(image.size)!=metadata['raw_dimensions']:raise ValueError('Remote raw canvas mismatch')
            publishing.append((candidate,sidecar,row['metadata_sha256']))
            for source,destination,digest in publishing:
                if destination.exists():
                    if common.file_sha256(destination)!=digest:raise ValueError('Conflicting collected artifact')
                else:
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    os.link(source,destination)
        # Full validated rows replace the local view; a torn tail is not committed.
        common.write_jsonl(state/'generated.jsonl',[generated[r['sample_id']] for r in rows])
        common.write_jsonl(state/'results.jsonl',rows)
        for name in ('attempts.jsonl','summary.json'):
            source=incoming/name
            if not source.exists():continue
            if name.endswith('.jsonl'):
                common.write_jsonl(state/name,weather_io.read_rows(source))
            else:
                value=json.loads(source.read_text())
                if value['saved']!=len(rows):
                    value.update(state='sync_in_progress',saved=len(rows),evaluated=len(rows))
                common.write_json(state/name,value)
        collected+=len(rows)
    for name in ('status.json','telemetry.json','completed.json','backend_owner.json'):
        path=staging/name
        if path.exists():
            common.write_json(remote_root/name,json.loads(path.read_text()))
    return collected


def reject_symlinks(path,root):
    current=root
    for part in path.relative_to(root).parts:
        current=current/part
        if current.is_symlink():raise ValueError('Symlink in remote artifact: '+str(path))


def status(root):
    config=read_sealed(root/'cluster.json'); owners=read_sealed(root/'ownership.json')['owners']
    result=dict(planned=config['planned'],saved=config['historical_saved'],nodes={},
                unassigned=len(config['shards'])-len(owners),verification=config['verification'])
    originals={e['original_state'] for e in queue(root,'gpu2')['entries']}
    marker=cohort_marker()
    states=[Path(s['state']) for s in marker['shards'] if s['state'] not in originals]
    states += [Path(e['state']) for e in queue(root,'gpu2')['entries']]
    saved=0
    for state in states:
        path=state/'summary.json'
        if path.exists():
            saved+=json.loads(path.read_text()).get('evaluated',0)
    result.update(saved=saved,remaining=config['planned']-saved)
    for node in ('local','gpu2'):
        directory=root/'nodes'/node
        result['nodes'][node]={name:json.loads((directory/name).read_text()) if (directory/name).exists() else None
            for name in ('status.json','telemetry.json')}
    revision=root/'prompt_revision_requested.json'
    if revision.exists():
        requested=read_sealed(revision);successor=Path(requested['successor_root'])/'state'
        result['prompt_revision']=dict(old_image_queue_paused=True,successor_state=str(successor),
            inventory=json.loads((successor/'inventory.json').read_text())['total'] if (successor/'inventory.json').exists() else None,
            progress=json.loads((successor/'progress.json').read_text()) if (successor/'progress.json').exists() else None,
            note='saved counts historical commits, including retired mercury; use retained_manifest and successor progress')
    common.write_json(root/'status.json',result)
    return result


def manifest(root):
    """Export all durable rows with bounded memory, only when requested."""
    from experiments.qwen_curriculum import weather_io
    remote_entries=queue(root,'gpu2')['entries']
    remote_originals={e['original_state'] for e in remote_entries}
    states=[Path(s['state']) for s in cohort_marker()['shards'] if s['state'] not in remote_originals]
    states.extend(Path(e['state']) for e in remote_entries)
    def rows():
        seen=set()
        for state in states:
            for row in weather_io.read_rows(state/'results.jsonl'):
                if row['sample_id'] in seen:raise ValueError('Duplicate generated sample across machines')
                seen.add(row['sample_id']);yield row
    output=root/'generated_manifest.jsonl'
    common.write_jsonl(output,rows())
    return {'manifest':str(output),'verification':'pending; not approved training data'}


def run_controller(root, once=False):
    import threading
    stopped=threading.Event()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopped.set())
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not stopped.is_set():
            errors={}
            try: sync(root)
            except Exception as exc: errors['gpu2_sync']=str(exc)
            for node in ('local','gpu2'):
                try:
                    directory=root/'nodes'/node
                    paused=(root/'pause_requested').exists() or (directory/'pause_requested').exists()
                    if node=='gpu2':
                        if paused:remote('touch',str(directory/'pause_requested'))
                        else:remote('rm','-f',str(directory/'pause_requested'))
                    if paused:continue
                    if node=='gpu2' and not transfer_complete(root):continue
                    grant(root,node)
                    active=command_node(node,'systemctl','--user','is-active',unit(node))
                except subprocess.CalledProcessError as exc:
                    if exc.returncode==3 and exc.stdout.strip()=='inactive':
                        command_node(node,'systemctl','--user','start',unit(node))
                    else:errors[node]=str(exc)
                except Exception as exc:errors[node]=str(exc)
            value=status(root);value['controller_errors']=errors;value['updated_unix']=time.time()
            common.write_json(root/'status.json',value)
            print(json.dumps({k:v for k,v in value.items() if k!='nodes'},ensure_ascii=False),flush=True)
            if value['remaining']==0:
                controls(root,'pause','all')
                manifest(root)
                return
            if once:return
            stopped.wait(30)


def controls(root, action, selected):
    revision=root/'prompt_revision_requested.json'
    if action=='resume' and revision.exists():
        raise ValueError('This image queue is retired for prompt revision. Keep it paused; see '+str(revision))
    nodes=('local','gpu2') if selected=='all' else (selected,)
    if action=='pause' and selected=='all':(root/'pause_requested').touch()
    if action=='resume':
        if selected!='all' and (root/'pause_requested').exists():
            for other in ('local','gpu2'):
                if other!=selected:(root/'nodes'/other/'pause_requested').touch()
        (root/'pause_requested').unlink(missing_ok=True)
    for node in nodes:
        marker=root/'nodes'/node/'pause_requested'
        if action=='pause':
            marker.touch()
            command_node(node,'touch',str(marker))
        else:
            command_node(node,'rm','-f',str(marker))
            if node=='gpu2':marker.unlink(missing_ok=True)
            if node!='gpu2' or transfer_complete(root):command_node(node,'systemctl','--user','start',unit(node))
    if action=='resume':call(['systemctl','--user','start','ccr-night-controller.service'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('command',choices=('prepare','status','sync','manifest','controller','pause','resume'))
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--node',choices=('all','local','gpu2'),default='all')
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args(argv);root=args.root.resolve()
    if args.command=='prepare':value=prepare(root)
    elif args.command=='controller':return run_controller(root,args.once)
    elif args.command in ('pause','resume'):return controls(root,args.command,args.node)
    elif args.command=='sync':value={'collected':sync(root)}
    elif args.command=='manifest':value=manifest(root)
    else:value=status(root)
    print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
