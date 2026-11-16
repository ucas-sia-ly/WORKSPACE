"""Replan retired-mercury and unfinished sources; never generate images here.

Original frozen cohorts stay immutable. A new atomic per-batch journal binds
each source, its scene evidence, eligibility and newly composed night prompt.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import campaign,common,run,weather_io
from experiments.qwen_curriculum import night_diversity_policy_v2 as policy
from experiments.qwen_curriculum.night_revision_planner import Planner
from experiments.qwen_curriculum.night_cluster_worker import resource_sample
from experiments.qwen_curriculum import night_revision_runtime as runtime

ROOT=common.WORKSPACE_ROOT/'outputs/night_revision_v2'
VERSION='night_revision_v2'


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():yield json.loads(line)


def sealed_read(path):
    value=json.loads(Path(path).read_text())
    if value!=run.sealed(value):raise ValueError('Revision receipt checksum changed: '+str(path))
    return value


def chunks(values,count):
    group=[]
    for value in values:
        group.append(value)
        if len(group)==count:yield group;group=[]
    if group:yield group


def dependencies():
    from experiments.qwen_curriculum import night_revision_planner,scene_prompting
    from experiments.qwen_curriculum import night_cluster_worker
    return {str(Path(p).resolve()):common.file_sha256(p) for p in
            (__file__,policy.__file__,policy.CATALOG_PATH,night_revision_planner.__file__,
             scene_prompting.__file__,common.__file__,run.__file__,campaign.__file__,night_cluster_worker.__file__,runtime.__file__)}


def prepare(root,model,batch_size=32,seed=42):
    state=root/'state';inventory=sealed_read(state/'inventory.json')
    if inventory['total']['reserved_without_commit']:
        raise ValueError('Resolve interrupted image reservations before replacing their jobs')
    for name,digest in inventory['hashes'].items():
        if common.file_sha256(state/name)!=digest:raise ValueError('Inventory changed: '+name)
    counts={key:dict(saved=0,pending=0,mismatch=0) for key in policy.STYLES}
    for key,value in counts.items():value['saved']=inventory['by_style'][key]['retained']
    cfg=dict(version=VERSION,condition='night',model_path=str(model.resolve()),batch_size=batch_size,
        seed=seed,num_places=inventory['total']['revision_tasks'],images_per_place=1,
        analysis_tokens=2600,compose_tokens=1800,analysis_drafts=2,compose_drafts=3,
        original_cohort_marker_sha256=inventory['cohort_marker_sha256'],
        tasks_sha256=common.file_sha256(state/'tasks.jsonl'),inventory_sha256=common.file_sha256(state/'inventory.json'),
        retained_images=inventory['total']['retained'],retired_styles=['mercury_vapor'],
        coverage_counts=counts,policy=policy.definition(),files_sha256=dependencies(),
        inference=dict(dtype='bfloat16',attention='sdpa',allocator_max_gib=42,padding='left'),
        software={name:importlib.metadata.version(name) for name in ('torch','transformers','Pillow')},
        generation_barrier='text planning only; image generation requires a new explicit launch',
        interrupted_text_call='record prior reservation lineage and recompute only absent response; never lose saved responses')
    model_files={str(p.resolve()):common.file_sha256(p) for p in sorted(model.rglob('*'))
                 if p.is_file() and '.cache' not in p.relative_to(model).parts}
    if not any(name.endswith('.safetensors') for name in model_files):
        raise ValueError('Complete local VLM checkpoint required')
    cfg['model_files_sha256']=model_files
    return campaign.frozen(state/'revision_config.json',cfg)


def load(root):
    state=root/'state';cfg=campaign.config_file(state/'revision_config.json')
    if cfg['version']!=VERSION or cfg['policy']!=policy.definition():raise ValueError('Revision policy changed')
    cfg=runtime.verify(root,cfg,dependencies(),policy.definition())
    for p,digest in cfg['model_files_sha256'].items():
        if common.file_sha256(p)!=digest:raise ValueError('Frozen revision dependency changed: '+p)
    model=Path(cfg['model_path'])
    actual={str(p.resolve()) for p in model.rglob('*') if p.is_file() and '.cache' not in p.relative_to(model).parts}
    if actual!=set(cfg['model_files_sha256']):raise ValueError('VLM checkpoint file inventory changed')
    if common.file_sha256(state/'tasks.jsonl')!=cfg['tasks_sha256']:raise ValueError('Revision task inventory changed')
    if common.file_sha256(state/'inventory.json')!=cfg['inventory_sha256']:raise ValueError('Revision baseline inventory changed')
    for name,v in cfg['software'].items():
        if importlib.metadata.version(name)!=v:raise ValueError('Planner software changed: '+name)
    return cfg


def make_job(task,plan,cfg):
    source=task['source'];variant=plan['variants'][0]
    if variant['style_id']=='mercury_vapor':raise ValueError('Mercury style is retired')
    key=common.fingerprint([VERSION,cfg['fingerprint'],source['source_sha256'],plan,task['sample_id']])
    job={k:source[k] for k in ('city','place_id','source_path','source_sha256','source_dimensions')}
    job.update(sample_id='qnight2_'+key[:24],condition='night',style_id=variant['style_id'],
        prompt_policy=policy.VERSION,prompt=policy.render_prompt(variant),negative_prompt='',
        prompt_variant=variant['style_id'],style_name=variant['style_id']+': '+variant['style_name'],
        seed=common.candidate_seed(cfg['seed'],key,0),source_sample_id=source['sample_id'],pair_id=key,
        with_transient=False,transient_policy='plausible people/vehicles; declared surface_state; preserve fixed place structure',
        scene_plan_sha256=common.fingerprint(plan),appearance=variant,
        protected_regions=plan.get('protected_regions',[]),scene_summary=plan['scene_summary'],
        authorized_transient=None,planned_region_is_verified_mask=False,
        supersedes_sample_id=task['sample_id'],supersedes_record_sha256=task['original_record_sha256'],
        revision_reason=task['task_reason'],source_real_pool_sha256=source['real_pool_sha256'])
    job['record_sha256']=common.fingerprint(job)
    return job


def plan_group(reader,directory,tasks,cfg,allocated,prior=None):
    """Only completed group receipts alter coverage on resume."""
    candidates=[];records={};remaining=[];prior=prior or {};histories={t['sample_id']:[] for t in tasks}
    for task in tasks:
        saved=prior.get(task['sample_id'],{})
        if saved.get('source_evidence'):
            evidence=policy.source_evidence(saved['source_evidence'])
            eligible,excluded=policy.eligible_styles(evidence)
            if not eligible:raise ValueError('Saved confirmed source no longer has a compatible style')
            candidates.append(dict(task=task,evidence=evidence,eligible=eligible,excluded=excluded,
                feedback='',style=saved.get('selected_style')))
        else:remaining.append(task)
    for draft in range(cfg['analysis_drafts']):
        if not remaining:break
        sources=[task['source'] for task in remaining]
        requests=[policy.analysis_request()+(' Previous response failed: '+task['_analysis_error']+
            '. Reinspect the source; reply in ENGLISH, including the summary and ALL evidence strings. '
            'Use valid JSON; paraphrase signs without quoted lettering. Return EVERY exact key, never only a repair fragment.'
            if '_analysis_error' in task else '') for task in remaining]
        replies=reader.ask(directory,sources,requests,'analysis',draft,cfg)
        retry=[]
        for task,raw in zip(remaining,replies,strict=True):
            try:
                parsed,repairs=policy.parse_json_with_diagnostics(raw)
                evidence=policy.source_evidence(parsed)
                eligible,excluded=policy.eligible_styles(evidence)
                if not eligible:raise ValueError('No confirmed source-compatible style')
                if repairs:histories[task['sample_id']].append(dict(phase='analysis',draft=draft,
                    format_repairs=repairs,raw_sha256=common.fingerprint(raw)))
                candidates.append(dict(task=task,evidence=evidence,eligible=eligible,excluded=excluded,feedback=''))
            except (ValueError,KeyError,TypeError) as exc:
                histories[task['sample_id']].append(dict(phase='analysis',draft=draft,error=str(exc),raw_sha256=common.fingerprint(raw)))
                task=dict(task,_analysis_error=str(exc));retry.append(task)
        remaining=retry
    for task in remaining:
        records[task['sample_id']]=dict(status='failed',reason='invalid_source_analysis',error=task['_analysis_error'])
    reservations=Counter()
    for c in candidates:
        counts={key:{**stats,'pending':allocated.get(key,0)+reservations[key]} for key,stats in cfg['coverage_counts'].items()}
        if c.get('style') is None:
            c['style']=policy.select_styles(c['eligible'],1,counts,seed=cfg['seed'],source_key=c['task']['sample_id'])[0]
        if c['style'] not in c['eligible']:raise ValueError('Recovered style is not supported by original source evidence')
        reservations[c['style']]+=1
    for draft in range(cfg['compose_drafts']):
        if not candidates:break
        replies=reader.ask(directory,[c['task']['source'] for c in candidates],
            [policy.compose_request(c['evidence'],c['style'],c['feedback']) for c in candidates],'compose',draft,cfg)
        retry=[]
        for c,raw in zip(candidates,replies,strict=True):
            try:
                plan=policy.parse_plan(raw,c['evidence'],c['style'])
                job=make_job(c['task'],plan,cfg)
                records[c['task']['sample_id']]=dict(status='ready',plan=plan,job=job,
                    source_evidence=c['evidence'],eligible_styles=c['eligible'],excluded_styles=c['excluded'])
            except (ValueError,KeyError,TypeError) as exc:
                histories[c['task']['sample_id']].append(dict(phase='compose',draft=draft,error=str(exc),raw_sha256=common.fingerprint(raw)))
                c['feedback']=policy.compose_feedback(raw,c['evidence'],c['style'],str(exc));retry.append(c)
        candidates=retry
    for c in candidates:
        records[c['task']['sample_id']]=dict(status='failed',reason='composer_drafts_exhausted',error=c['feedback'],
            selected_style=c['style'],source_evidence=c['evidence'],eligible_styles=c['eligible'],excluded_styles=c['excluded'])
    return [run.sealed(dict(**records[t['sample_id']],source=t['source'],
        predecessor_sample_id=t['sample_id'],cohort_fingerprint=cfg['fingerprint'],task_reason=t['task_reason'],
        attempt_errors=histories[t['sample_id']],
        **({'source_analysis_reuse':prior[t['sample_id']]['source_analysis_reuse']}
           if prior.get(t['sample_id'],{}).get('source_analysis_reuse') else {}))) for t in tasks]


def check_group(value,tasks,cfg):
    if (value['cohort_fingerprint']!=cfg['fingerprint'] or value['predecessor_ids']!=[t['sample_id'] for t in tasks]
            or len(value['records'])!=len(tasks)):raise ValueError('Atomic revision group changed')
    if value.get('execution_patch_sha256') not in (None,*cfg.get('execution_patch_history',[cfg.get('execution_patch_sha256')])):
        raise ValueError('Revision group runtime differs from the frozen repair supplement')
    for record,task in zip(value['records'],tasks,strict=True):
        if record!=run.sealed(record) or record['source']!=task['source'] or record['predecessor_sample_id']!=task['sample_id']:
            raise ValueError('Revision response/source linkage changed')
        if record['status']=='ready':
            policy.validate_plan(record['plan'],1)
            if record['job']!=make_job(task,record['plan'],cfg):raise ValueError('Revision job changed')


def group_versions(state,tasks,cfg):
    """Replay immutable repair layers; no later version may alter an earlier success."""
    key=common.fingerprint([t['sample_id'] for t in tasks])
    base=sealed_read(state/'groups'/(key+'.json'));check_group(base,tasks,cfg)
    versions=[base]
    history=cfg.get('execution_patch_history',[cfg.get('execution_patch_sha256')])
    legacy=state/'repairs'/(key+'.json')
    legacy_value=sealed_read(legacy) if legacy.exists() else None
    if legacy_value and legacy_value.get('execution_patch_sha256') not in history:
        raise ValueError('Failure repair uses an unknown execution identity')
    for digest in history:
        if not digest:continue
        path=state/'repairs'/digest/(key+'.json')
        value=sealed_read(path) if path.exists() else None
        if legacy_value and legacy_value.get('execution_patch_sha256')==digest:
            if value is not None:raise ValueError('Duplicate failure repair layer')
            value=legacy_value
        if value is None:continue
        check_group(value,tasks,cfg);previous=versions[-1]
        if (value.get('parent_group_sha256')!=base['result_sha256'] or value.get('repair_round')!=1
                or value.get('execution_patch_sha256')!=digest
                or value.get('previous_receipt_sha256',base['result_sha256'])!=previous['result_sha256']):
            raise ValueError('Failure repair lineage changed')
        for original,repaired in zip(previous['records'],value['records'],strict=True):
            if original['status']=='ready' and original!=repaired:
                raise ValueError('A failure repair must never rewrite an already ready prompt')
        versions.append(value)
    return versions


def read_group(state,tasks,cfg):
    versions=group_versions(state,tasks,cfg)
    return versions[-1],versions[0]


def repair_group(reader,state,tasks,cfg,allocated,value,base):
    """One durable additional pass per group, not an unbounded retry on each restart."""
    key=common.fingerprint([t['sample_id'] for t in tasks])
    path=state/'repairs'/str(cfg.get('execution_patch_sha256'))/(key+'.json')
    if (path.exists() or not cfg.get('execution_patch_sha256')
            or value is not base and value.get('execution_patch_sha256')==cfg.get('execution_patch_sha256')):return value
    failed={r['predecessor_sample_id']:r for r in value['records'] if r['status']=='failed'}
    if not failed:return value
    # Legacy failures did not store the assigned category. Recover it from the
    # sealed exact-source compose cache, rather than silently trying a new style.
    legacy={key:r for key,r in failed.items()
            if r.get('reason')=='composer_drafts_exhausted' and not r.get('selected_style')}
    choices={key:set() for key in legacy}
    analysis={key:r for key,r in failed.items() if not r.get('source_evidence')}
    reusable={key:[] for key in analysis}
    if legacy or analysis:
        for cached in sorted((state/'batches').glob('*.json')):
            if cached.name.endswith('.request.json'):continue
            response=sealed_read(cached);identity=response['identity']
            if identity['cohort_fingerprint']!=cfg['fingerprint']:continue
            for index,(source,raw) in enumerate(zip(identity['source_inputs'],response['responses'],strict=True)):
                if identity['phase']=='analysis' and not identity.get('execution_patch_sha256'):
                    for key,record in analysis.items():
                        if source['sample_id']!=record['source']['sample_id'] or source['source_sha256']!=record['source']['source_sha256']:
                            continue
                        try:
                            parsed,repairs=policy.parse_json_with_diagnostics(raw)
                            evidence=policy.source_evidence(parsed)
                            if not policy.eligible_styles(evidence)[0]:continue
                        except (ValueError,KeyError,TypeError):continue
                        reusable[key].append((identity['attempt'],cached.name,index,evidence,
                            dict(response_sha256=response['result_sha256'],response_file=str(cached),
                                 response_index=index,format_repairs=repairs,
                                 reason='Reuse exact-source original analysis; no status or evidence facts filled or changed')))
                if identity['phase']!='compose':continue
                for key,record in legacy.items():
                    if (source['sample_id']!=record['source']['sample_id'] or
                            source['source_sha256']!=record['source']['source_sha256']):continue
                    try:style=policy.parse_json(raw)['variants'][0]['style_id']
                    except (ValueError,KeyError,TypeError,IndexError):continue
                    if style in record['eligible_styles']:choices[key].add(style)
        for key,styles in choices.items():
            if len(styles)!=1:raise ValueError('Cannot unambiguously recover the original failed compose style')
            failed[key]={**failed[key],'selected_style':next(iter(styles))}
        for key,candidates in reusable.items():
            if candidates:
                chosen=min(candidates,key=lambda c:c[:3])
                failed[key]={**failed[key],'source_evidence':chosen[3],'source_analysis_reuse':chosen[4]}
    budget=Counter(allocated)
    budget.update(r['job']['style_id'] for r in value['records'] if r['status']=='ready')
    subset=[t for t in tasks if t['sample_id'] in failed]
    recovered=plan_group(reader,state,subset,cfg,budget,prior=failed)
    replacements={r['predecessor_sample_id']:r for r in recovered}
    merged=[replacements.get(r['predecessor_sample_id'],r) for r in value['records']]
    result=run.sealed(dict(cohort_fingerprint=cfg['fingerprint'],predecessor_ids=value['predecessor_ids'],
        records=merged,parent_group_sha256=base['result_sha256'],repair_round=1,
        previous_receipt_sha256=value['result_sha256'],
        execution_patch_sha256=cfg['execution_patch_sha256'],recovered=sum(r['status']=='ready' for r in recovered),
        attempted=len(subset)))
    check_group(result,tasks,cfg);common.write_json(path,result)
    return result


def sync_prompt(state,task,record,original,ancestors=()):
    path=state/'prompts'/(task['sample_id']+'.json')
    if path.exists():
        current=sealed_read(path)
        if current==record:return
        if not any(current==old and old['status']=='failed' for old in (original,*ancestors)):
            raise ValueError('Prompt differs from group commit or failure-repair lineage')
    common.write_json(path,record)


def prompt(root,*,max_groups=0,repair_only=False):
    state=root/'state';cfg=load(root)
    if (state/'prompts_complete.json').exists():return validate_completion(root,cfg)
    for node in ('local','gpu2'):
        value=json.loads((common.WORKSPACE_ROOT/'outputs/night_cluster_v1/nodes'/node/'status.json').read_text())
        if value['backend_pid']:raise ValueError('Stop image backends before prompt planning')
    reader=Planner(Path(cfg['model_path']));allocated=Counter();eligible_counts=Counter();failures=Counter()
    finished=ready=failed=fresh=initial_failures=recovered=0
    start=time.monotonic();done=threading.Event();errors=[];stopping=threading.Event()
    initial=resource_sample(0,[state])
    if initial['available_gib']<6 or initial['gpu_used_gib']>4:raise RuntimeError('Need >=6GiB available RAM and an idle GPU before revision planning')
    def progress(status,error=None):
        memory={}
        if reader._backend is not None:
            memory=dict(cuda_peak_allocated_gib=reader.torch.cuda.max_memory_allocated()/1024**3,
                        cuda_peak_reserved_gib=reader.torch.cuda.max_memory_reserved()/1024**3)
        common.write_json(state/'progress.json',dict(state=status,total=cfg['num_places'],processed=finished,
            ready=ready,failed=failed,elapsed_seconds=time.monotonic()-start,planned_style_counts=dict(allocated),
            eligible_style_counts=dict(eligible_counts),failure_reasons=dict(failures),error=error,
            generation_started=False,images_reused=cfg['retained_images'],
            failures_before_recovery=initial_failures,recovered_failures=recovered,
            execution_patch_sha256=cfg.get('execution_patch_sha256'),**memory))
    def watch():
        low=None
        while not done.wait(2):
            try:
                value=resource_sample(0,[state]);now=time.monotonic()
                if done.is_set():return
                common.write_json(state/'resources.json',dict(sampled_unix=time.time(),current=value))
                if value['available_gib']<2:low=now if low is None else low
                else:low=None
                if value['available_gib']<.75 or value['gpu_used_gib']>44 or value['disk_free_gib']<8 or (low is not None and now-low>=60):
                    errors.append('Revision planner resource protection reached');os.kill(os.getpid(),signal.SIGTERM);return
            except Exception as exc:
                if done.is_set():return
                errors.append(str(exc));os.kill(os.getpid(),signal.SIGTERM);return
    def stop(*_):
        stopping.set()
        if errors:raise KeyboardInterrupt(errors[0])
    old_handlers={}
    for sig in (signal.SIGTERM,signal.SIGINT):old_handlers[sig]=signal.signal(sig,stop)
    monitor=threading.Thread(target=watch,daemon=True);monitor.start()
    try:
        with run.run_lock(state):
            for tasks in chunks(rows(state/'tasks.jsonl'),cfg['batch_size']):
                key=common.fingerprint([t['sample_id'] for t in tasks]);path=state/'groups'/(key+'.json')
                if path.exists():
                    versions=group_versions(state,tasks,cfg);value,base=versions[-1],versions[0]
                else:
                    if stopping.is_set() or repair_only:break
                    for task in tasks:run.validate_image(task['source'],'source_path','source_sha256')
                    progress('prompting')
                    records=plan_group(reader,state,tasks,cfg,allocated)
                    value=run.sealed(dict(cohort_fingerprint=cfg['fingerprint'],predecessor_ids=[t['sample_id'] for t in tasks],
                        records=records,execution_patch_sha256=cfg.get('execution_patch_sha256')))
                    common.write_json(path,value);fresh+=1
                    base=value
                    versions=[base]
                check_group(value,tasks,cfg)
                if not stopping.is_set():value=repair_group(reader,state,tasks,cfg,allocated,value,base)
                initial_failures+=sum(r['status']=='failed' for r in base['records'])
                recovered+=sum(r['status']=='failed' for r in base['records'])-sum(r['status']=='failed' for r in value['records'])
                for index,(record,task,original) in enumerate(zip(value['records'],tasks,base['records'],strict=True)):
                    sync_prompt(state,task,record,original,[v['records'][index] for v in versions[1:]])
                    finished+=1
                    if record['status']=='ready':ready+=1;allocated[record['job']['style_id']]+=1
                    else:failed+=1;failures[record['reason']]+=1
                    eligible_counts.update(record.get('eligible_styles',[]))
                progress('prompting');print(json.dumps(dict(processed=finished,total=cfg['num_places'],ready=ready,failed=failed)),flush=True)
                if ready==0 and finished>=cfg['batch_size']:raise RuntimeError('No valid prompts in first batch; inspect schema before full execution')
                if stopping.is_set() or (max_groups and fresh>=max_groups):break
            if finished==cfg['num_places']:
                result=export(root,cfg);progress('prompts_complete');return result
            progress('paused',errors[0] if errors else None)
            return dict(state='paused',processed=finished,ready=ready,failed=failed)
    except BaseException as exc:
        progress('failed',f'{type(exc).__name__}: {exc}');raise
    finally:
        done.set();monitor.join(timeout=15);reader.close()
        for sig,handler in old_handlers.items():signal.signal(sig,handler)


def validate_completion(root,cfg):
    state=root/'state';value=sealed_read(state/'prompts_complete.json')
    if (value['cohort_fingerprint']!=cfg['fingerprint'] or value['selected']!=cfg['num_places']
            or value.get('execution_patch_sha256')!=cfg.get('execution_patch_sha256')
            or value['ready']+value['failed']!=cfg['num_places']
            or common.file_sha256(state/'night_prompts.jsonl')!=value['manifest_sha256']):
        raise ValueError('Revision completion barrier changed')
    for shard in value['shards']:
        if common.file_sha256(Path(shard['state'])/'plan_config.json')!=shard['config_sha256']:
            raise ValueError('Exported revision shard changed')
    return value


def export(root,cfg=None):
    cfg=cfg or load(root);state=root/'state';shards=[];jobs=[];sources=[];failed=0;ready=0
    seen=set();count=0;job_ids=set()
    for tasks in chunks(rows(state/'tasks.jsonl'),cfg['batch_size']):
        key=common.fingerprint([t['sample_id'] for t in tasks])
        group,_=read_group(state,tasks,cfg)
        for record,task in zip(group['records'],tasks,strict=True):
            if task['sample_id'] in seen:raise ValueError('Duplicate revision predecessor')
            seen.add(task['sample_id']);count+=1
            if sealed_read(state/'prompts'/(task['sample_id']+'.json'))!=record:
                raise ValueError('Export prompt differs from its atomic source/group commit')
            if record['status']=='ready':
                if record['job']['sample_id'] in job_ids:raise ValueError('Duplicate revised image job')
                job_ids.add(record['job']['sample_id'])
    if count!=cfg['num_places']:raise ValueError('Revision source count differs from frozen budget')
    def flush():
        if not jobs:return
        shard=root/'shards'/f's{len(shards):05d}'/'state'
        common.write_jsonl(shard/'plan.jsonl',jobs);common.write_jsonl(shard/'source_snapshot.jsonl',sources)
        config=campaign.frozen(shard/'plan_config.json',dict(version=1,stage=policy.VERSION,
            revision_fingerprint=cfg['fingerprint'],num_jobs=len(jobs),request_budget=len(jobs),domain_counts=dict(night=len(jobs)),
            image_root=str(root/'images'),plan_sha256=common.file_sha256(shard/'plan.jsonl'),
            files_sha256={**cfg.get('effective_files_sha256',cfg['files_sha256']),
                str(state/'revision_config.json'):common.file_sha256(state/'revision_config.json')},
            generation_status='prepared_only; explicit new image executor required',retired_styles=['mercury_vapor']))
        shards.append(dict(state=str(shard),jobs=len(jobs),config_sha256=common.file_sha256(shard/'plan_config.json')))
        jobs.clear();sources.clear()
    def all_jobs():
        nonlocal failed,ready
        for task in rows(state/'tasks.jsonl'):
            value=sealed_read(state/'prompts'/(task['sample_id']+'.json'))
            if value['cohort_fingerprint']!=cfg['fingerprint']:raise ValueError('Prompt completion identity changed')
            if value['status']!='ready':failed+=1;continue
            ready+=1;jobs.append(value['job']);sources.append(value['source']);yield value['job']
            if len(jobs)>=128:flush()
        flush()
    common.write_jsonl(state/'night_prompts.jsonl',all_jobs())
    result=run.sealed(dict(cohort_fingerprint=cfg['fingerprint'],selected=cfg['num_places'],ready=ready,failed=failed,
        shards=shards,manifest_sha256=common.file_sha256(state/'night_prompts.jsonl'),retired_styles=['mercury_vapor'],
        generation_started=False,retained_old_images=cfg['retained_images'],
        execution_patch_sha256=cfg.get('execution_patch_sha256')))
    common.write_json(state/'prompts_complete.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('command',choices=('prepare','plan','status','export'))
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--model',type=Path,default=Path('/home/admin123/github/checkpoints/Qwen3-VL-4B-Instruct'))
    p.add_argument('--batch-size',type=int,choices=(4,8,16,32,64),default=32)
    p.add_argument('--max-groups',type=int,default=0)
    p.add_argument('--repair-only',action='store_true',help='Retry failures in already committed groups; do not begin pending sources')
    p.add_argument('--execute',action='store_true')
    args=p.parse_args(argv);root=args.root.resolve()
    if args.command=='prepare':value=prepare(root,args.model,args.batch_size)
    elif args.command=='plan':
        if not args.execute:raise ValueError('GPU prompt planning requires --execute; images are never generated here')
        value=prompt(root,max_groups=args.max_groups,repair_only=args.repair_only)
    elif args.command=='export':value=export(root)
    else:
        path=root/'state/progress.json';value=json.loads(path.read_text()) if path.exists() else dict(state='prepared_not_prompted')
    print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
