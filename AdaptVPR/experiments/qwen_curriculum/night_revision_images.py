"""Stream committed revised night prompts to disjoint, durable GPU queues.

The remote GPU starts immediately. The local GPU is admitted only after the
complete prompt/export barrier AND planner process exit. All image artifacts
are SHA-verified back to the primary host before remote copies are removed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import common, campaign, run, weather_io
from experiments.qwen_curriculum import night_revision as revision
from experiments.qwen_curriculum import night_cluster as cluster
from experiments.qwen_curriculum import night_cluster_worker as worker
from experiments.qwen_curriculum import night_cluster_resources as resources

ROOT=common.WORKSPACE_ROOT/'outputs/night_revision_images_v1'
VERSION='night_revision_images_v1'
CONTROLLER='ccr-night-revision-images-controller.service'
PLANNER='ccr-night-revision-prompts-v2.service'


def unit(node):
    return f'ccr-night-revision-{node}-worker.service'


def code_paths():
    names=('night_revision_images.py','night_revision.py','night_revision_runtime.py',
        'night_revision_planner.py','night_diversity_policy_v2.py','night_style_catalog_v2.json',
        'night_cluster.py','night_cluster_worker.py','night_cluster_resources.py','balanced_night.py',
        'night_diversity_policy.py','night_style_catalog.json','scene_aware_night.py','scene_prompting.py',
        'night_first_v3.py','common.py','run.py','campaign.py','resume_recovery.py','weather_io.py')
    return [Path(__file__).with_name(name) for name in names]


def engine():
    from experiments.qwen_curriculum import balanced_night
    value=balanced_night.runtime().engine()
    value.load_plan=load_execution_plan
    return value


def quality():
    from experiments.qwen_curriculum import balanced_night
    return balanced_night.runtime().quality_definition(clip_device='cpu')


def load_execution_plan(directory):
    directory=Path(directory);cfg=campaign.config_file(directory/'plan_config.json')
    if cfg.get('stage')!=VERSION or cfg.get('quality')!=quality():
        raise ValueError('Revised image execution policy changed')
    for name,digest in cfg['files_sha256'].items():
        if common.file_sha256(name)!=digest:raise ValueError('Frozen image dependency changed: '+name)
    if common.file_sha256(directory/'plan.jsonl')!=cfg['plan_sha256']:
        raise ValueError('Revised image plan changed')
    jobs=common.read_jsonl(directory/'plan.jsonl')
    receipts=common.read_jsonl(directory/'prompt_receipts.jsonl')
    if (len(jobs)!=cfg['num_jobs'] or len(jobs)!=len(receipts) or
            len({j['sample_id'] for j in jobs})!=len(jobs) or cfg['domain_counts']!={'night':len(jobs)}):
        raise ValueError('Revised job budget or domains changed')
    for job,receipt in zip(jobs,receipts,strict=True):
        if (receipt!=run.sealed(receipt) or receipt['status']!='ready' or receipt['job']!=job
                or not re.fullmatch(r'qnight2_[0-9a-f]{24}',job['sample_id'])
                or job['record_sha256']!=common.fingerprint({k:v for k,v in job.items() if k!='record_sha256'})
                or job['condition']!='night' or job['style_id']=='mercury_vapor'):
            raise ValueError('Only exact sealed revised ready jobs may execute')
        revision.policy.validate_plan(receipt['plan'],1)
        if job['prompt']!=revision.policy.render_prompt(receipt['plan']['variants'][0]):
            raise ValueError('Published prompt differs from validated VLM prose')
    return cfg,jobs


def prepare(root,source=revision.ROOT):
    root=Path(root).resolve();source=Path(source).resolve();cfg=revision.load(source)
    (root/'deployment').mkdir(parents=True,exist_ok=True)
    definition=dict(version=VERSION,source_root=str(source),source_fingerprint=cfg['fingerprint'],
        source_config_sha256=common.file_sha256(source/'state/revision_config.json'),
        source_total=cfg['num_places'],retained_old_images=cfg['retained_images'],
        files_sha256={str(p):common.file_sha256(p) for p in code_paths()},
        local_gate='validated complete prompt export and planner MainPID=0; stable free GPU resources',
        scheduling='append-only sealed ready feed; durable ownership; one outstanding shard per node',
        storage='all normalized, raw and sidecars collected to primary; remote completed copies pruned only after SHA acknowledgement',
        verification='deferred; never approved training data',primary_disk_reserve_gib=20)
    definition=cluster.freeze(root/'cluster.json',definition)
    if not (root/'ownership.json').exists():
        cluster.save(root/'ownership.json',dict(version=1,cluster_fingerprint=definition['fingerprint'],owners={}))
    if not (root/'feed.json').exists():cluster.save(root/'feed.json',dict(version=1,entries=[],predecessors={}))
    for node in ('local','gpu2'):
        directory=root/'nodes'/node;directory.mkdir(parents=True,exist_ok=True)
        original=worker.read_sealed(cluster.DEFAULT/'nodes'/node/'node.json')
        node_cfg={k:v for k,v in original.items() if k!='fingerprint'}
        node_cfg['files_sha256']={**original['files_sha256'],str(Path(__file__).resolve()):common.file_sha256(__file__)}
        if node=='gpu2':node_cfg['limits']={**original['limits'],'max_rss_gib':50}
        node_cfg=cluster.freeze(directory/'node.json',node_cfg)
        policy=dict(version=1,node_fingerprint=node_cfg['fingerprint'],
            implementation_sha256=common.file_sha256(resources.__file__),
            ram_soft_gib=2 if node=='local' else 8,ram_emergency_gib=.75 if node=='local' else 2,
            ram_grace_seconds=60,gpu_reserve_gib=1.5,startup_gpu_free_gib=42 if node=='local' else 22,
            startup_ram_gib=10 if node=='local' else 54,retry_seconds=120,stable_seconds=30,
            rss_soft_gib=18 if node=='local' else 46,
            description='Finish active image and recycle owned backend before gradual RSS pressure; frozen BF16 sampling unchanged')
        cluster.freeze(directory/'resource_policy.json',policy)
        if not (directory/'queue.json').exists():cluster.save(directory/'queue.json',dict(version=1,node=node,entries=[]))
    return definition


def load(root):
    cfg=worker.read_sealed(root/'cluster.json')
    if cfg['version']!=VERSION:raise ValueError('Wrong revised image protocol')
    for name,digest in cfg['files_sha256'].items():
        if common.file_sha256(name)!=digest:raise ValueError('Image controller code changed: '+name)
    return cfg


def publish_ready(root,cfg,source_cfg):
    """Only a complete committed group plus its matching prompt index is visible."""
    state=Path(cfg['source_root'])/'state';feed=worker.read_sealed(root/'feed.json');seen=feed['predecessors']
    added=0
    for tasks in revision.chunks(revision.rows(state/'tasks.jsonl'),source_cfg['batch_size']):
        key=common.fingerprint([t['sample_id'] for t in tasks])
        if not (state/'groups'/(key+'.json')).exists():break
        group,_=revision.read_group(state,tasks,source_cfg)
        consistent=True
        for record,task in zip(group['records'],tasks,strict=True):
            path=state/'prompts'/(task['sample_id']+'.json')
            if not path.exists():consistent=False;break
            current=revision.sealed_read(path)
            if current!=record:
                if current['status']=='ready':raise ValueError('An already ready prompt changed during streaming')
                consistent=False;break
        if not consistent:break
        records=[]
        for record in group['records']:
            predecessor=record['predecessor_sample_id']
            if predecessor in seen:
                if record['status']!='ready' or record['job']['record_sha256']!=seen[predecessor]:
                    raise ValueError('Previously published revised prompt changed')
            elif record['status']=='ready':records.append(record)
        if not records:continue
        directory=root/'shards'/f's{len(feed["entries"]):05d}'/'state';directory.mkdir(parents=True,exist_ok=True)
        jobs=[r['job'] for r in records]
        # A crash before feed commit reuses the same deterministic frozen shard.
        for name,values in (('plan.jsonl',jobs),('prompt_receipts.jsonl',records),
                ('source_snapshot.jsonl',[r['source'] for r in records])):
            path=directory/name
            if path.exists() and common.read_jsonl(path)!=values:raise ValueError('Uncommitted execution shard changed')
            if not path.exists():common.write_jsonl(path,values)
        campaign.frozen(directory/'planner_config.json',dict(stage=VERSION,source_group_sha256=group['result_sha256']))
        campaign.frozen(directory/'coverage_snapshot.json',source_cfg['coverage_counts'])
        files={**cfg['files_sha256'],str(root/'cluster.json'):common.file_sha256(root/'cluster.json'),
            str(directory/'prompt_receipts.jsonl'):common.file_sha256(directory/'prompt_receipts.jsonl')}
        plan=campaign.frozen(directory/'plan_config.json',dict(version=1,stage=VERSION,num_jobs=len(jobs),
            request_budget=len(jobs),domain_counts={'night':len(jobs)},image_root=str(root/'images'),
            service=worker.read_sealed(root/'nodes/local/node.json')['service_identity'],quality=quality(),
            plan_sha256=common.file_sha256(directory/'plan.jsonl'),files_sha256=files,
            source_group_sha256=group['result_sha256'],source_fingerprint=source_cfg['fingerprint']))
        feed['entries'].append(dict(original_state=str(directory),
            original_config_sha256=common.file_sha256(directory/'plan_config.json'),jobs=len(jobs),saved_at_creation=0))
        for record in records:seen[record['predecessor_sample_id']]=record['job']['record_sha256']
        cluster.save(root/'feed.json',feed);added+=len(records)
    return added


def grant(root,node):
    cfg=load(root);feed=worker.read_sealed(root/'feed.json');ownership=worker.read_sealed(root/'ownership.json')
    if ownership['cluster_fingerprint']!=cfg['fingerprint']:raise ValueError('Revised ownership changed')
    queues={n:cluster.queue(root,n) for n in ('local','gpu2')};queue=queues[node]
    complete=sum((Path(e['state'])/'summary.json').exists() and
        json.loads((Path(e['state'])/'summary.json').read_text())['state']=='domain_complete' for e in queue['entries'])
    if len(queue['entries'])-complete>=1:return None
    descriptor=cluster.select_assignment({'shards':feed['entries'],'partial_local_only':[]},ownership,queues,node)
    if descriptor is None:return None
    ownership['owners'][descriptor['original_state']]=node
    cluster.save(root/'ownership.json',ownership)
    entry=cluster.stage_assignment(root,descriptor,node)
    if node=='gpu2':
        source=Path(descriptor['original_state'])/'prompt_receipts.jsonl'
        target=Path(entry['state'])/'prompt_receipts.jsonl'
        if target.exists() and common.file_sha256(target)!=common.file_sha256(source):
            raise ValueError('Derived revised prompt receipts changed')
        if not target.exists():shutil.copy2(source,target)
        cluster.rsync_files(root,[target])
    queue['entries'].append(entry);cluster.save(root/'nodes'/node/'queue.json',queue)
    if node=='gpu2':cluster.rsync_files(root,[root/'nodes/gpu2/queue.json'])
    return entry


def planner_finished(source,source_cfg,pid):
    if pid:return False
    if not (source/'state/prompts_complete.json').exists():return False
    revision.validate_completion(source,source_cfg)
    return True


def active(node):
    result=cluster.command_node(node,'systemctl','--user','show',unit(node),'-p','ActiveState','--value')
    return result.strip()


def controls(root,action,node):
    load(root)
    for name in (('local','gpu2') if node=='all' else (node,)):
        marker=root/'nodes'/name/'pause_requested'
        if action=='pause':marker.touch()
        else:marker.unlink(missing_ok=True)
        if name=='gpu2':cluster.remote('touch' if action=='pause' else 'rm',
            *([] if action=='pause' else ['-f']),str(marker))
        if action=='resume':cluster.command_node(name,'systemctl','--user','reset-failed',unit(name))
    if action=='resume':cluster.call(['systemctl','--user','start',CONTROLLER])


def status(root):
    cfg=load(root);feed=worker.read_sealed(root/'feed.json');saved=committed=0;nodes={}
    for node in ('local','gpu2'):
        directory=root/'nodes'/node
        nodes[node]={name:json.loads((directory/name).read_text()) if (directory/name).exists() else None
            for name in ('status.json','telemetry.json')}
        for entry in cluster.queue(root,node)['entries']:
            path=Path(entry['state'])/'summary.json'
            if path.exists():
                summary=json.loads(path.read_text());saved+=summary.get('saved',0);committed+=summary.get('evaluated',0)
    value=dict(planned_source_places=cfg['source_total'],prompts_published=len(feed['predecessors']),
        new_images_saved=saved,new_images_committed=committed,retained_old_images=cfg['retained_old_images'],nodes=nodes,
        local_start_policy=cfg['local_gate'],storage=cfg['storage'],verification='pending')
    common.write_json(root/'status.json',value);return value


def manifest(root):
    def records():
        seen=set()
        for node in ('local','gpu2'):
            for entry in cluster.queue(root,node)['entries']:
                for row in weather_io.read_rows(Path(entry['state'])/'results.jsonl'):
                    if row['sample_id'] in seen:raise ValueError('Duplicate revised image across machines')
                    seen.add(row['sample_id']);yield row
    path=root/'generated_manifest.jsonl';common.write_jsonl(path,records());return str(path)


def prune_remote(root,ack_path):
    """Delete only acknowledged image copies from a completed remote shard."""
    directory=root/'nodes/gpu2';node=worker.load_node(directory)
    ack=worker.read_sealed(ack_path);entry=ack['entry']
    completed=worker.read_sealed(directory/'completed.json')
    if (node['node']!='gpu2' or ack['node_fingerprint']!=node['fingerprint'] or
            completed['node_fingerprint']!=node['fingerprint'] or entry not in completed['entries']):
        raise ValueError('Only completed owned remote shards can be pruned')
    if entry not in cluster.queue(root,'gpu2')['entries']:
        raise ValueError('Remote prune is outside the assigned queue')
    rows=weather_io.read_rows(Path(entry['state'])/'results.jsonl')
    if common.fingerprint(rows)!=ack['results_fingerprint']:
        raise ValueError('Remote commits differ from primary acknowledgement')
    expected=[]
    for row in rows:
        if not re.fullmatch(r'qnight2_[0-9a-f]{24}',row['sample_id']):raise ValueError('Wrong prune namespace')
        if (row['output_path']!=str(directory/'images/night'/(row['sample_id']+'.png')) or
                row['raw_output_path']!=str(directory/'images/raw/night'/(row['sample_id']+'.png'))):
            raise ValueError('Prune output paths differ from exact owned image names')
        expected.extend([(row['output_path'],row['output_sha256']),
            (row['raw_output_path'],row['raw_output_sha256']),
            (str(Path(row['raw_output_path']).with_suffix('.json')),row['metadata_sha256'])])
    if ack['files']!=[{'path':p,'sha256':d} for p,d in expected]:raise ValueError('Prune acknowledgement file set changed')
    for path,digest in expected:
        path=Path(path)
        if not path.is_relative_to(directory/'images'):raise ValueError('Prune escapes image directory')
        cluster.reject_symlinks(path,directory/'images')
        if path.exists() and common.file_sha256(path)!=digest:raise ValueError('Remote image changed before pruning')
    for path,_ in expected:Path(path).unlink(missing_ok=True)
    receipt=directory/'pruned'/ack_path.name
    cluster.freeze(receipt,dict(acknowledgement_sha256=common.file_sha256(ack_path),files=len(expected),
        primary_verified=True,remote_images_removed=True))
    return {'files':len(expected),'remote_copies_removed':True}


def prune_confirmed(root):
    directory=root/'nodes/gpu2';path=directory/'completed.json'
    if not path.exists():return 0
    completed=worker.read_sealed(path);queue=cluster.queue(root,'gpu2')['entries']
    node=worker.read_sealed(directory/'node.json')
    if completed['node_fingerprint']!=node['fingerprint'] or completed['entries']!=queue[:len(completed['entries'])]:
        raise ValueError('Remote completion acknowledgement changed')
    count=0
    for entry in completed['entries']:
        name=common.fingerprint(entry)+'.json';ack_path=directory/'acknowledged'/name
        done=directory/'acknowledged'/(name+'.confirmed')
        if done.exists():continue
        state=Path(entry['state']);rows=weather_io.read_rows(state/'results.jsonl')
        plan=worker.read_sealed(state/'plan_config.json')
        if len(rows)!=plan['num_jobs']:continue
        files=[]
        for row in rows:
            for field,digest in (('output_path','output_sha256'),('raw_output_path','raw_output_sha256')):
                run.validate_image(row,field,digest);files.append({'path':row[field],'sha256':row[digest]})
            sidecar=Path(row['raw_output_path']).with_suffix('.json')
            if common.file_sha256(sidecar)!=row['metadata_sha256']:raise ValueError('Primary sidecar changed')
            files.append({'path':str(sidecar),'sha256':row['metadata_sha256']})
        cluster.freeze(ack_path,dict(node_fingerprint=node['fingerprint'],entry=entry,
            results_fingerprint=common.fingerprint(rows),files=files,primary_all_artifacts_verified=True))
        cluster.rsync_files(root,[ack_path])
        cluster.remote(cluster.REMOTE_PY,'-B',str(Path(__file__).resolve()),'prune',
            '--root',str(root),'--ack',str(ack_path),timeout=180)
        common.write_json(done,worker.sealed(dict(acknowledgement_sha256=common.file_sha256(ack_path))))
        count+=len(rows)
    return count


def run_controller(root,once=False):
    cfg=load(root);source=Path(cfg['source_root']);source_cfg=revision.load(source)
    if source_cfg['fingerprint']!=cfg['source_fingerprint']:raise ValueError('Prompt source cohort changed')
    stopped=threading.Event()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopped.set())
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not stopped.is_set():
            errors={};publish_ready(root,cfg,source_cfg)
            try:
                cluster.sync(root)
                prune_confirmed(root)
            except Exception as exc:errors['sync']=str(exc)
            pid=int(cluster.call(['systemctl','--user','show',PLANNER,'-p','MainPID','--value']) or 0)
            local_ready=planner_finished(source,source_cfg,pid)
            disk_ready=shutil.disk_usage(root).free/1024**3>=cfg['primary_disk_reserve_gib']
            for node in ('gpu2','local'):
                try:
                    directory=root/'nodes'/node;paused=(directory/'pause_requested').exists()
                    if paused or not disk_ready or (node=='local' and not local_ready):continue
                    if node=='gpu2' and 'sync' in errors:continue
                    grant(root,node)
                    queue=cluster.queue(root,node)
                    if not queue['entries']:continue
                    state=active(node)
                    if state=='inactive':cluster.command_node(node,'systemctl','--user','start',unit(node))
                    elif state=='failed':errors[node]='worker failed; inspect error before explicit resume'
                except Exception as exc:errors[node]=str(exc)
            value=status(root);value.update(updated_unix=time.time(),controller_errors=errors,
                local_eligible=local_ready,primary_disk_ready=disk_ready)
            common.write_json(root/'status.json',value)
            print(json.dumps({k:v for k,v in value.items() if k!='nodes'}),flush=True)
            if local_ready and value['new_images_committed']==value['prompts_published']:
                controls(root,'pause','all');manifest(root);return
            if once:return value
            stopped.wait(30)


def run_node(directory):
    engine()
    original=resources.Pressure.check
    def pressure(self,sample,now,*,backend_active):
        reason,emergency=original(self,sample,now,backend_active=backend_active)
        if reason:return reason,emergency
        if backend_active and sample['owned_rss_gib']>=self.policy['rss_soft_gib']:
            return 'owned RSS recycle threshold reached; finish image before releasing cache',False
        return None,False
    resources.Pressure.check=pressure
    resources.main(['--node-dir',str(directory),'--policy',str(directory/'resource_policy.json')])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('command',choices=('prepare','controller','worker','status','pause','resume','sync','manifest','prune'))
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--node',choices=('all','local','gpu2'),default='all')
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--ack',type=Path)
    args=parser.parse_args(argv);root=args.root.resolve()
    if args.command=='prepare':value=prepare(root)
    elif args.command=='controller':return run_controller(root,args.once)
    elif args.command=='worker':
        if args.node=='all':raise ValueError('Choose exactly one worker machine')
        return run_node(root/'nodes'/args.node)
    elif args.command in ('pause','resume'):return controls(root,args.command,args.node)
    elif args.command=='sync':value={'collected':cluster.sync(root),'remote_copies_removed':prune_confirmed(root)}
    elif args.command=='prune':
        if args.ack is None:raise ValueError('An exact verified acknowledgement is required')
        value=prune_remote(root,args.ack.resolve())
    elif args.command=='manifest':value={'manifest':manifest(root)}
    else:value=status(root)
    print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
