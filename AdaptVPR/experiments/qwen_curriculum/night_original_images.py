"""Resume exact original night prompts, excluding every mercury-vapor job.

Existing non-mercury results remain untouched. Both machines execute only
uncommitted, never-reserved original jobs. Revised qnight2 outputs are archived
separately and do not count as completion of this original-prompt continuation.
"""
from __future__ import annotations

from collections import Counter
import fcntl
import json
from pathlib import Path
import re
import shutil
import signal
import threading
import time
import sys

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import night_revision_images as base
from experiments.qwen_curriculum import common,run,campaign,weather_io

ROOT=common.WORKSPACE_ROOT/'outputs/night_original_no_mercury_v1'
VERSION='night_original_no_mercury_v1'
SNAPSHOT=common.WORKSPACE_ROOT/'outputs/night_revision_v2/state'
SHARD_SIZE=32


def pending_jobs(snapshot):
    """Stream and verify original jobs; never recompose or alter their seed/ID."""
    cached_state=None;originals={};seen=set()
    for task in base.revision.rows(snapshot/'tasks.jsonl'):
        if task['style_id']=='mercury_vapor':continue
        if task['task_reason']!='not_committed' or task['committed'] or task['attempts']:
            raise ValueError('Only uncommitted and never-reserved non-mercury jobs may be reassigned')
        state=Path(task['original_state'])
        if str(state)!=cached_state:
            if common.file_sha256(state/'plan_config.json')!=task['original_config_sha256']:
                raise ValueError('Original frozen plan config changed')
            cfg=campaign.config_file(state/'plan_config.json')
            if common.file_sha256(state/'plan.jsonl')!=cfg['plan_sha256']:
                raise ValueError('Original frozen prompt manifest changed')
            originals={j['sample_id']:j for j in common.read_jsonl(state/'plan.jsonl')};cached_state=str(state)
        job=task['original_job']
        if (originals.get(task['sample_id'])!=job or job['sample_id'] in seen
                or job['record_sha256']!=task['original_record_sha256']
                or job['record_sha256']!=common.fingerprint({k:v for k,v in job.items() if k!='record_sha256'})
                or job['style_id']=='mercury_vapor' or job['condition']!='night'):
            raise ValueError('Original prompt, source, style or seed differs from its frozen plan')
        seen.add(job['sample_id'])
        yield task,job


def validate_proof(job,proof):
    if (proof!=run.sealed(proof) or proof.get('job')!=job
            or proof.get('original_record_sha256')!=job.get('record_sha256')
            or proof.get('status')!='original_frozen_pending'
            or not re.fullmatch(r'qnight_[0-9a-f]{24}',job.get('sample_id',''))
            or job.get('style_id')=='mercury_vapor' or job.get('condition')!='night'
            or job.get('record_sha256')!=common.fingerprint({k:v for k,v in job.items() if k!='record_sha256'})):
        raise ValueError('Only unchanged original non-mercury jobs are allowed')


def load_execution_plan(directory):
    directory=Path(directory);cfg=campaign.config_file(directory/'plan_config.json')
    if cfg.get('stage')!=VERSION or cfg.get('quality')!=base.quality():
        raise ValueError('Original-prompt execution policy changed')
    for name,digest in cfg['files_sha256'].items():
        if common.file_sha256(name)!=digest:raise ValueError('Frozen original-prompt dependency changed: '+name)
    if common.file_sha256(directory/'plan.jsonl')!=cfg['plan_sha256']:
        raise ValueError('Original-prompt continuation plan changed')
    jobs=common.read_jsonl(directory/'plan.jsonl');proofs=common.read_jsonl(directory/'prompt_receipts.jsonl')
    if (len(jobs)!=cfg['num_jobs'] or len(jobs)!=len(proofs) or
            len({j['sample_id'] for j in jobs})!=len(jobs) or cfg['domain_counts']!={'night':len(jobs)}):
        raise ValueError('Original-prompt budget or domains changed')
    for job,proof in zip(jobs,proofs,strict=True):validate_proof(job,proof)
    return cfg,jobs


def prepare(root):
    root=Path(root).resolve();(root/'deployment').mkdir(parents=True,exist_ok=True)
    inventory=base.revision.sealed_read(SNAPSHOT/'inventory.json')
    if common.file_sha256(SNAPSHOT/'tasks.jsonl')!=inventory['hashes']['tasks.jsonl']:
        raise ValueError('Original pending inventory changed')
    if inventory['total']['reserved_without_commit']:
        raise ValueError('Resolve original uncertain requests before creating a continuation')
    marker=base.cluster.cohort_marker()
    if common.file_sha256(base.cluster.COHORT/'prompts_complete.json')!=inventory['cohort_marker_sha256']:
        raise ValueError('Original completed prompt cohort changed')
    retained_ids=set();retained=0
    for row in base.revision.rows(SNAPSHOT/'retained_manifest.jsonl'):
        if row['style_id']=='mercury_vapor' or row['sample_id'] in retained_ids:
            raise ValueError('Retained non-mercury inventory is invalid')
        if not Path(row['output_path']).is_file():raise ValueError('A retained original image is missing')
        retained_ids.add(row['sample_id']);retained+=1
    if retained!=inventory['total']['retained']:raise ValueError('Retained image count changed')
    files={str(p):common.file_sha256(p) for p in base.code_paths()}
    files[str(Path(__file__).resolve())]=common.file_sha256(__file__)
    cfg=base.cluster.freeze(root/'cluster.json',dict(version=VERSION,source_total=13206,
        retained_old_images=retained,source_root=str(base.cluster.COHORT.parent),
        source_cohort_sha256=inventory['cohort_marker_sha256'],
        tasks_sha256=common.file_sha256(SNAPSHOT/'tasks.jsonl'),
        retained_manifest=str(SNAPSHOT/'retained_manifest.jsonl'),
        retained_manifest_sha256=common.file_sha256(SNAPSHOT/'retained_manifest.jsonl'),
        files_sha256=files,excluded_style='mercury_vapor',excluded_places=2066,
        local_gate='both machines now; wait only for stopped revised VLM and stable resources',
        scheduling='exact original Prompt/seed/sample ID; disjoint durable ownership; no completed jobs repeated',
        storage='all normalized images, raw images and sidecars on primary; acknowledged remote copies pruned',
        primary_disk_reserve_gib=8,verification='deferred; not automatically approved training data'))
    if not (root/'ownership.json').exists():base.cluster.save(root/'ownership.json',dict(
        version=1,cluster_fingerprint=cfg['fingerprint'],owners={}))
    if not (root/'feed.json').exists():base.cluster.save(root/'feed.json',dict(version=1,entries=[],predecessors={}))
    for node in ('local','gpu2'):
        directory=root/'nodes'/node;directory.mkdir(parents=True,exist_ok=True)
        original=base.worker.read_sealed(common.WORKSPACE_ROOT/'outputs/night_revision_images_v1/nodes'/node/'node.json')
        definition={k:v for k,v in original.items() if k!='fingerprint'}
        definition['files_sha256']={**definition['files_sha256'],str(Path(__file__).resolve()):common.file_sha256(__file__)}
        if node=='local':definition['limits']={**definition['limits'],'reserve_disk_gib':8}
        node_cfg=base.cluster.freeze(directory/'node.json',definition)
        original_policy=base.worker.read_sealed(common.WORKSPACE_ROOT/'outputs/night_revision_images_v1/nodes'/node/'resource_policy.json')
        policy={k:v for k,v in original_policy.items() if k!='fingerprint'}
        policy.update(node_fingerprint=node_cfg['fingerprint'])
        if node=='local':policy['startup_ram_gib']=8
        base.cluster.freeze(directory/'resource_policy.json',policy)
        if not (directory/'queue.json').exists():base.cluster.save(directory/'queue.json',dict(version=1,node=node,entries=[]))
    feed=base.worker.read_sealed(root/'feed.json');styles=Counter();count=0
    for index,group in enumerate(base.revision.chunks(pending_jobs(SNAPSHOT),SHARD_SIZE)):
        directory=root/'shards'/f's{index:05d}'/'state';directory.mkdir(parents=True,exist_ok=True)
        jobs=[];proofs=[]
        for task,job in group:
            if job['sample_id'] in retained_ids:raise ValueError('A completed original job must not be repeated')
            jobs.append(job);styles[job['style_id']]+=1;count+=1
            proofs.append(run.sealed(dict(status='original_frozen_pending',job=job,
                original_state=task['original_state'],original_config_sha256=task['original_config_sha256'],
                original_record_sha256=task['original_record_sha256'],previous_attempts=0)))
        for name,rows in (('plan.jsonl',jobs),('prompt_receipts.jsonl',proofs),
                ('source_snapshot.jsonl',[dict(source_path=j['source_path'],source_sha256=j['source_sha256']) for j in jobs])):
            path=directory/name
            if path.exists() and common.read_jsonl(path)!=rows:raise ValueError('Original continuation shard changed')
            if not path.exists():common.write_jsonl(path,rows)
        campaign.frozen(directory/'planner_config.json',dict(version=VERSION,original_prompts_unchanged=True))
        campaign.frozen(directory/'coverage_snapshot.json',dict(excluded_style='mercury_vapor'))
        campaign.frozen(directory/'plan_config.json',dict(version=1,stage=VERSION,num_jobs=len(jobs),request_budget=len(jobs),
            domain_counts={'night':len(jobs)},image_root=str(root/'images'),quality=base.quality(),
            service=base.worker.read_sealed(root/'nodes/local/node.json')['service_identity'],
            plan_sha256=common.file_sha256(directory/'plan.jsonl'),files_sha256={**files,
                str(root/'cluster.json'):common.file_sha256(root/'cluster.json'),
                str(directory/'prompt_receipts.jsonl'):common.file_sha256(directory/'prompt_receipts.jsonl')}))
        descriptor=dict(original_state=str(directory),original_config_sha256=common.file_sha256(directory/'plan_config.json'),
            jobs=len(jobs),saved_at_creation=0)
        if index<len(feed['entries']):
            if feed['entries'][index]!=descriptor:raise ValueError('Published continuation prefix changed')
        else:
            feed['entries'].append(descriptor)
            feed['predecessors'].update({j['sample_id']:j['record_sha256'] for j in jobs})
            base.cluster.save(root/'feed.json',feed)
    if count!=cfg['source_total'] or len(feed['predecessors'])!=count or len(styles)!=29:
        raise ValueError('Original non-mercury source budget or styles changed')
    base.cluster.freeze(root/'prepared.json',dict(planned=count,retained=retained,excluded_mercury=2066,
        original_prompt_and_seed_unchanged=True,style_counts=dict(styles),shards=len(feed['entries'])))
    return cfg


def manifest(root):
    cfg=base.load(root)
    if common.file_sha256(cfg['retained_manifest'])!=cfg['retained_manifest_sha256']:
        raise ValueError('Retained original manifest changed')
    def rows():
        seen=set()
        for row in base.revision.rows(Path(cfg['retained_manifest'])):
            if row['style_id']=='mercury_vapor' or row['sample_id'] in seen:raise ValueError('Invalid retained original row')
            seen.add(row['sample_id']);yield row
        for node in ('local','gpu2'):
            for entry in base.cluster.queue(root,node)['entries']:
                for row in weather_io.read_rows(Path(entry['state'])/'results.jsonl'):
                    if row['style_id']=='mercury_vapor' or row['sample_id'] in seen:raise ValueError('Duplicate or excluded generated job')
                    seen.add(row['sample_id']);yield row
    path=root/'generated_manifest.jsonl';common.write_jsonl(path,rows());return str(path)


def prune_remote(root,ack_path):
    directory=root/'nodes/gpu2';node=base.worker.load_node(directory)
    ack=base.worker.read_sealed(ack_path);entry=ack['entry'];completed=base.worker.read_sealed(directory/'completed.json')
    if (node['node']!='gpu2' or ack['node_fingerprint']!=node['fingerprint'] or
            completed['node_fingerprint']!=node['fingerprint'] or entry not in completed['entries'] or
            entry not in base.cluster.queue(root,'gpu2')['entries']):
        raise ValueError('Only acknowledged completed original shards may be pruned')
    rows=weather_io.read_rows(Path(entry['state'])/'results.jsonl')
    if common.fingerprint(rows)!=ack['results_fingerprint']:raise ValueError('Primary acknowledgement differs from remote commits')
    expected=[]
    for row in rows:
        if not re.fullmatch(r'qnight_[0-9a-f]{24}',row['sample_id']) or row['style_id']=='mercury_vapor':
            raise ValueError('Wrong original job namespace or style')
        if (row['output_path']!=str(directory/'images/night'/(row['sample_id']+'.png')) or
                row['raw_output_path']!=str(directory/'images/raw/night'/(row['sample_id']+'.png'))):
            raise ValueError('Original prune escapes exact owned paths')
        expected.extend([(row['output_path'],row['output_sha256']),(row['raw_output_path'],row['raw_output_sha256']),
            (str(Path(row['raw_output_path']).with_suffix('.json')),row['metadata_sha256'])])
    if ack['files']!=[dict(path=p,sha256=d) for p,d in expected]:raise ValueError('Acknowledged file set changed')
    for name,digest in expected:
        path=Path(name);base.cluster.reject_symlinks(path,directory/'images')
        if path.exists() and common.file_sha256(path)!=digest:raise ValueError('Remote original copy changed')
    for name,_ in expected:Path(name).unlink(missing_ok=True)
    base.cluster.freeze(directory/'pruned'/ack_path.name,dict(acknowledgement_sha256=common.file_sha256(ack_path),
        files=len(expected),primary_verified=True,remote_images_removed=True))
    return {'files':len(expected),'remote_copies_removed':True}


def controller(root,once=False):
    cfg=base.load(root)
    if not (root/'prepared.json').exists():raise ValueError('Complete original-prompt precheck first')
    stopped=threading.Event()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopped.set())
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not stopped.is_set():
            errors={}
            try:base.cluster.sync(root);base.prune_confirmed(root)
            except Exception as exc:errors['sync']=str(exc)
            planner_pid=int(base.cluster.call(['systemctl','--user','show',base.PLANNER,'-p','MainPID','--value']) or 0)
            disk_ready=shutil.disk_usage(root).free/1024**3>=cfg['primary_disk_reserve_gib']
            for node in ('local','gpu2'):
                try:
                    directory=root/'nodes'/node;paused=(directory/'pause_requested').exists()
                    if node=='gpu2':base.cluster.remote('touch' if paused else 'rm',
                        *([] if paused else ['-f']),str(directory/'pause_requested'))
                    if paused or not disk_ready or node=='local' and planner_pid:continue
                    if node=='gpu2' and 'sync' in errors:continue
                    base.grant(root,node)
                    if not base.cluster.queue(root,node)['entries']:continue
                    state=base.active(node)
                    if state=='inactive':base.cluster.command_node(node,'systemctl','--user','start',base.unit(node))
                    elif state=='failed':errors[node]='worker failed; inspect before explicit resume'
                except Exception as exc:errors[node]=str(exc)
            value=base.status(root);value.update(updated_unix=time.time(),controller_errors=errors,
                original_prompts=True,excluded_style='mercury_vapor',local_eligible=not planner_pid,primary_disk_ready=disk_ready)
            common.write_json(root/'status.json',value)
            print(json.dumps({k:v for k,v in value.items() if k!='nodes'}),flush=True)
            if value['new_images_committed']==cfg['source_total']:
                base.controls(root,'pause','all');manifest(root);return
            if once:return value
            stopped.wait(30)


def configure():
    # Private execution adapter: historical planners and image modules stay
    # byte-identical. Durable POST/raw recovery and resource handling are reused.
    base.ROOT=ROOT;base.VERSION=VERSION;base.__file__=__file__
    base.CONTROLLER='ccr-night-original-images-controller.service'
    base.unit=lambda node:f'ccr-night-original-{node}-worker.service'
    base.prepare=prepare;base.load_execution_plan=load_execution_plan
    base.run_controller=controller;base.manifest=manifest;base.prune_remote=prune_remote


def main(argv=None):
    configure();base.main(argv)


if __name__=='__main__':main()
