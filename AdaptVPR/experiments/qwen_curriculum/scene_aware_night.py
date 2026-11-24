"""Bounded image-conditioned night/style/transient experiments, separate from v3-v5."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

from PIL import Image, ImageDraw, ImageOps

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import campaign,common,quality_v3,run,scene_prompting as planning

ROOT=common.WORKSPACE_ROOT/'outputs/scene_conditioned_night_v2'
_ENGINE=None


def quality_definition(**kwargs):
    value=quality_v3.quality_definition_v3(**kwargs)
    value.update(version='scene-aware-candidate-1',planner=planning.VERSION,
        eligibility='independent source/candidate authorization audit required',
        verification_schedule='generator first; real geometry/CLIP only after Qwen service stops',
        regional_clip='v5 regional classifier not used as a hard gate for diverse styles')
    return value


class CandidateVerifier(quality_v3.QwenQualityVerifierV3):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.definition=quality_definition(thresholds=self.thresholds,matcher_name=self.matcher_name,
            img_size=self.img_size,n_kpts=self.n_kpts,clip_model_name=self.clip_model_name,clip_device=self.clip_device)

    def evaluate(self,source,output,condition):
        value=super().evaluate(source,output,condition)
        value.update(base_quality_passed=value['passed'],eligible_for_training=False,
            content_status='pending_source_conditioned_vlm_audit',quality_version='scene-aware-candidate-1')
        return value


class DeferredVerifier:
    """Commit only explicit pending status while the large generator is loaded."""
    def __init__(self,*args,**kwargs):
        self.definition=quality_definition(clip_device='cpu')

    def evaluate(self,*args):
        return dict(passed=False,eligible_for_training=False,base_quality_passed=False,
                    status='pending_verification',content_status='pending_geometry_weather_and_vlm',
                    quality_version='deferred-no-model',verification_pending=True)


def engine():
    global _ENGINE
    if _ENGINE is None:
        name='experiments.qwen_curriculum._scene_aware_durable_engine'
        spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name('night_first_v3.py'))
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        module.load_plan=load_plan;module.DOMAINS=('night','rainy_night')
        module.QwenQualityVerifierV3=DeferredVerifier
        module._original_publish=module.publish;module.publish=publish
        module._original_append=module._append;module._append=bounded_append
        _ENGINE=module
    return _ENGINE


def bounded_append(path,row):
    if Path(path).name=='attempts.jsonl':
        cfg=campaign.config_file(Path(path).parent/'plan_config.json')
        if row['call_number']>cfg['request_budget']:
            raise RuntimeError('Cumulative image-generation budget exhausted')
    engine()._original_append(path,row)


def publish(directory,config,jobs,generated,results,attempts,state,domain,error=None):
    value=engine()._original_publish(directory,config,jobs,generated,results,attempts,state,domain,error)
    value.update(candidate_passed=value['accepted'],accepted=0,
        places=len({(j['city'],j['place_id']) for j in jobs}),
        styles=dict(Counter(r['style_name'] for r in generated.values())),
        transient_images=sum(r['with_transient'] for r in generated.values()),
        training_eligibility='none before source/candidate authorization audit; audit candidate is not proof')
    for item in value['by_domain'].values():item['candidate_passed'],item['accepted']=item['accepted'],0
    common.write_json(directory/'summary.json',value)
    return value


def check_roots(directory,image_root):
    directory,image_root=Path(directory).resolve(),Path(image_root).resolve()
    for path in (directory,image_root):
        for name in ('qwen_curriculum','weather_v3','weather_v4','weather_v5_pilot'):
            old=common.WORKSPACE_ROOT/'outputs'/name
            if path==old or old in path.parents or path in old.parents:
                raise ValueError('Use an independent new experiment directory')


def choose_sources(rows,ids,max_sources,excluded,real_root,seed):
    if type(max_sources) is not int or max_sources<=0:
        raise ValueError('A positive bounded source budget is required')
    pool={}
    for row in rows:
        if (row['city'],int(row['place_id'])) not in excluded:
            pool[row['sample_id']]=row
    if ids:
        if len(ids)>max_sources or len(set(ids))!=len(ids) or any(i not in pool for i in ids):
            raise ValueError('Explicit sources are excluded, missing, duplicate or exceed budget')
        selected=[pool[i] for i in ids]
    else:
        selected=sorted(pool.values(),key=lambda r:common.fingerprint([seed,r['sample_id']]))[:max_sources]
    if not selected or len({(r['city'],r['place_id']) for r in selected})!=len(selected):
        raise ValueError('Sources must be nonempty distinct training places')
    for row in selected:
        if row!=run.sealed(row):raise ValueError('Source record seal changed')
        run.validate_image(row,'source_path','source_sha256')
        path=Path(row['source_path']).resolve()
        if (path.parent!=real_root/'Images'/row['city'] or
                not path.name.startswith(f"{row['city']}_{int(row['place_id']):07d}_")):
            raise ValueError('Image does not belong to its declared real training place')
        with Image.open(path) as image:
            if list(image.size)!=row['source_dimensions']:raise ValueError('Source dimensions changed')
    return selected


def make_jobs(sources,plans,paired_transients,seed):
    if len(sources)!=len(plans):raise ValueError('Every source requires its own completed scene plan')
    jobs=[]
    for source,plan in zip(sources,plans):
        planning.validate_plan(plan,len(plan['variants']))
        for index,variant in enumerate(plan['variants']):
            pair_key=common.fingerprint([planning.VERSION,source['source_sha256'],plan,index,seed])
            pair_seed=common.candidate_seed(seed,pair_key,0)
            for transient in ([False,True] if paired_transients and variant.get('transient') else [False]):
                sid='qscene_'+common.fingerprint([pair_key,transient])[:24]
                job={k:source[k] for k in ('city','place_id','source_path','source_sha256','source_dimensions')}
                job.update(sample_id=sid,condition='rainy_night' if variant['weather']=='rain' else 'night',
                    prompt_policy=planning.VERSION,prompt=planning.render_prompt(variant,transient),negative_prompt='',
                    prompt_variant=variant['style_name'],style_name=variant['style_name'],seed=pair_seed,
                    source_sample_id=source['sample_id'],pair_id=pair_key,with_transient=transient,
                    scene_plan_sha256=common.fingerprint(plan),appearance=variant,
                    protected_regions=plan['protected_regions'],scene_summary=plan['scene_summary'],
                    authorized_transient=variant['transient'] if transient else None,
                    planned_region_is_verified_mask=False)
                job['record_sha256']=common.fingerprint(job);jobs.append(job)
    return jobs


def preparation(args):
    directory=args.output_dir
    check_roots(directory,args.image_root)
    cfg_path=directory/'planner_config.json'
    if cfg_path.exists():
        cfg=campaign.config_file(cfg_path)
        expected=dict(source_manifest=str(args.source_manifest),source_ids=args.source_ids,max_sources=args.max_sources,
            variants=args.variants,paired_transients=args.paired_transients,seed=args.seed,
            model_path=str(args.planner_model),image_root=str(args.image_root),max_jobs=args.max_jobs)
        if cfg['request']!=expected:raise ValueError('Frozen planning request changed')
        for name,digest in cfg['files_sha256'].items():
            if common.file_sha256(name)!=digest:raise ValueError('Frozen planner input/code/model changed: '+name)
        return cfg,common.read_jsonl(directory/'source_snapshot.jsonl')
    excluded={(c,int(p)) for c,p in json.loads(args.exclude_places.read_text())}
    sources=choose_sources(common.read_jsonl(args.source_manifest),args.source_ids,args.max_sources,excluded,args.real_data,args.seed)
    if type(args.variants) is not int or not 1<=args.variants<=8:
        raise ValueError('Declare between 1 and 8 variants per source')
    if args.max_jobs<args.max_sources*args.variants*(2 if args.paired_transients else 1):
        raise ValueError('max-jobs must cover the predeclared maximum including paired transients')
    files=[Path(__file__),Path(planning.__file__),args.exclude_places,args.service_config,
           *sorted(args.planner_model.glob('*.json')),*sorted(args.planner_model.glob('*.txt')),
           *sorted(args.planner_model.glob('*.safetensors'))]
    if not list(args.planner_model.glob('*.safetensors')):
        raise ValueError('Local vision planner weights are missing; no download or mock fallback')
    common.write_jsonl(directory/'source_snapshot.jsonl',sources)
    files.append(directory/'source_snapshot.jsonl')
    cfg=campaign.frozen(cfg_path,dict(version=1,policy=planning.definition(),
        source_manifest_sha256=common.file_sha256(args.source_manifest),
        request=dict(source_manifest=str(args.source_manifest),source_ids=args.source_ids,max_sources=args.max_sources,
            variants=args.variants,paired_transients=args.paired_transients,seed=args.seed,
            model_path=str(args.planner_model),image_root=str(args.image_root),max_jobs=args.max_jobs),
        files_sha256={str(p):common.file_sha256(p) for p in files},
        model_parameters=dict(dtype='bfloat16',attention='sdpa',temperature=.8,top_p=.9,top_k=30,max_new_tokens=2200),
        planner_requests_per_source=2))
    return cfg,sources


def prepare(args):
    with run.run_lock(args.output_dir):
        if (args.output_dir/'plan_config.json').exists():return load_plan(args.output_dir)[0]
        cfg,sources=preparation(args)
        if not args.execute:
            return dict(status='prepared_inputs_only',sources=len(sources),maximum_jobs=args.max_jobs,
                planner_calls=0,generation_calls=0,next='prepare --execute runs the real local vision planner')
        campaign.ensure_stopped()
        plans=[];recent=[];reader=None
        try:
            for source in sources:
                dest=args.output_dir/'scene_plans'/f"{common.fingerprint(source['sample_id'])[:24]}.json"
                if dest.exists():
                    record=json.loads(dest.read_text())
                    if (record!=run.sealed(record) or record['source_sha256']!=source['source_sha256'] or
                            record['planner_fingerprint']!=cfg['fingerprint']):
                        raise ValueError('Cached scene plan identity changed')
                    plan=planning.validate_plan(record['plan'],args.variants)
                else:
                    feedback='';plan=None
                    for attempt in range(cfg['planner_requests_per_source']):
                        reserve=dest.with_suffix(f'.request_{attempt+1}.json')
                        cached=dest.with_suffix(f'.raw_{attempt+1}.json')
                        if reserve.exists() and not cached.exists():
                            raise RuntimeError('Unresolved local planner request; preserve evidence and use a new bounded preparation')
                        draw=common.candidate_seed(args.seed,source['sample_id'],attempt)
                        if cached.exists():raw=json.loads(cached.read_text())['response']
                        else:
                            common.write_json(reserve,dict(source_sha256=source['source_sha256'],planner_fingerprint=cfg['fingerprint']))
                            if reader is None:reader=planning.LocalVisionPlanner(args.planner_model)
                            raw=reader.read(source['source_path'],args.variants,draw,recent,feedback)
                            common.write_json(cached,dict(response=raw,seed=draw))
                        try:
                            plan=planning.validate_plan(planning.normalize_plan(planning.parse_json(raw)),args.variants)
                            break
                        except ValueError as error:
                            feedback='Previous response failed the machine schema/authorization check: '+str(error)+'. Return a complete corrected JSON plan. Do not force unsafe transients; set them null if needed.'
                    if plan is None:raise ValueError('Bounded planner drafts exhausted: '+feedback)
                    common.write_json(dest,run.sealed(dict(source_sha256=source['source_sha256'],
                        planner_fingerprint=cfg['fingerprint'],plan=plan)))
                plans.append(plan);recent.extend(v['style_name'] for v in plan['variants'])
        finally:
            if reader is not None:reader.close()
        jobs=make_jobs(sources,plans,args.paired_transients,args.seed)
        if len(jobs)>args.max_jobs:raise ValueError('Jobs exceed frozen image budget')
        common.write_jsonl(args.output_dir/'plan.jsonl',jobs)
        from experiments.qwen_curriculum import weather_io
        runtime=run.execution_definition(SimpleNamespace(run_dir=args.output_dir,device='cuda',cpu_threads=4))
        files={Path(m.__file__).resolve() for m in
            (sys.modules[__name__],planning,quality_v3,campaign,run,common,weather_io,engine(),engine().resume_recovery)}
        files.update(common.ADAPTVPR_ROOT/p for p in runtime['implementation_sha256'])
        files.add(common.ADAPTVPR_ROOT/'prompts/rules.py')
        files.add(Path(quality_v3.prompt_bank.__file__))
        files.add(common.ADAPTVPR_ROOT/'adapters/lightx2v_qwen_image_edit.py')
        files.update(args.output_dir.glob('scene_plans/*.json'))
        frozen=campaign.frozen(args.output_dir/'plan_config.json',dict(version=1,stage=planning.VERSION,
            planner_fingerprint=cfg['fingerprint'],num_jobs=len(jobs),request_budget=len(jobs),
            domain_counts=dict(Counter(j['condition'] for j in jobs)),image_root=str(args.image_root),
            quality=quality_definition(clip_device='cpu'),service=campaign.config_file(args.service_config)['reference_execution']['service'],
            plan_sha256=common.file_sha256(args.output_dir/'plan.jsonl'),
            files_sha256={str(p):common.file_sha256(p) for p in sorted(files)}))
        engine().rebuild_manifest(args.output_dir,frozen,[])
        return publish(args.output_dir,frozen,jobs,{},{},[],'prepared','all')


def load_plan(directory):
    directory=Path(directory);cfg=campaign.config_file(directory/'plan_config.json')
    check_roots(directory,cfg['image_root'])
    if cfg['stage']!=planning.VERSION or cfg['quality']!=quality_definition(clip_device='cpu'):
        raise ValueError('Frozen scene-aware generation policy changed')
    if common.file_sha256(directory/'plan.jsonl')!=cfg['plan_sha256']:raise ValueError('Plan checksum changed')
    for name,digest in cfg['files_sha256'].items():
        if common.file_sha256(name)!=digest:raise ValueError('Frozen input/implementation changed: '+name)
    jobs=common.read_jsonl(directory/'plan.jsonl')
    if (len(jobs)!=cfg['num_jobs'] or len({j['sample_id'] for j in jobs})!=len(jobs) or
            Counter(j['condition'] for j in jobs)!=Counter(cfg['domain_counts'])):
        raise ValueError('Job budget/domain counts changed')
    for job in jobs:
        if (job['record_sha256']!=common.fingerprint({k:v for k,v in job.items() if k!='record_sha256'}) or
                job['prompt']!=planning.render_prompt(job['appearance'],job['with_transient'])):
            raise ValueError('Scene-specific prompt/authorization changed')
    return cfg,jobs


def generate(args):
    if args.max_calls<=0:raise ValueError('Explicit positive --max-calls required')
    campaign.ensure_stopped()
    with run.run_lock(args.output_dir):
        cfg,jobs=load_plan(args.output_dir)
        generated,results,attempts=engine()._read_state(args.output_dir,jobs,cfg)
        if len(results)==len(jobs):publish(args.output_dir,cfg,jobs,generated,results,attempts,'complete','all')
        else:
            before=len(attempts)
            try:
                campaign.start_qwen({'reference_execution':{'service':cfg['service']}})
                for domain in cfg['domain_counts']:
                    now=len(run.read_rows(args.output_dir,'attempts.jsonl'))
                    remaining=args.max_calls-(now-before)
                    if remaining<=0:break
                    child=SimpleNamespace(**{**vars(args),'domain':domain,'max_calls':remaining})
                    engine().generate(child,locked=True)
            finally:campaign.stop_qwen()
    return verify(args)


def quality_records(directory,generated_results):
    path=directory/'quality_results.jsonl'
    records=engine()._rows(path) if path.exists() else []
    output={}
    for row in records:
        sid=row['sample_id'];original=generated_results.get(sid)
        if (original is None or sid in output or row!=run.sealed(row) or
                row['generation_receipt_sha256']!=original['result_sha256'] or
                any(row.get(k)!=original[k] for k in ('source_sha256','output_sha256','prompt','seed','authorized_transient'))):
            raise ValueError('Quality record is duplicated or differs from its exact generation')
        output[sid]=row
    return output


def verify(args):
    cfg,jobs=load_plan(args.output_dir)
    with run.run_lock(args.output_dir):
        _,generated,_=engine()._read_state(args.output_dir,jobs,cfg)
        records=quality_records(args.output_dir,generated)
        if not args.execute:return dict(verified=len(records),pending=len(generated)-len(records),model_calls=0)
        campaign.ensure_stopped()
        verifier=None
        path=args.output_dir/'quality_results.jsonl'
        # Repair only a torn final append using the established durable reader.
        common.write_jsonl(path,records.values())
        for job in jobs:
            sid=job['sample_id']
            if sid not in generated or sid in records:continue
            if verifier is None:verifier=CandidateVerifier(device=args.device,clip_device='cpu')
            with Image.open(job['source_path']) as source,Image.open(generated[sid]['output_path']) as output:
                verdict=verifier.evaluate(source.convert('RGB'),output.convert('RGB'),job['condition'])
            entry=run.sealed({**generated[sid],**verdict,'verification_pending':False,
                'generation_receipt_sha256':generated[sid]['result_sha256']})
            engine()._original_append(path,entry);records[sid]=entry
        summary=dict(verified=len(records),base_candidates=sum(r['base_quality_passed'] for r in records.values()),
                     training_eligible=0,generator_service='stopped before verifier loaded')
        common.write_json(args.output_dir/'quality_summary.json',summary)
        return summary


def validate_audit(value,job):
    if value.get('status') not in ('candidate','reject','unknown'):
        raise ValueError('Invalid audit status')
    for field in ('authorized_edit_achieved','placement_legal','landmarks_visible'):
        if type(value.get(field)) is not bool:raise ValueError('Audit booleans must be explicit')
    for field in ('permanent_changes','unexpected_objects','observed_added_transients'):
        if not isinstance(value.get(field),list) or any(not isinstance(x,str) for x in value[field]):
            raise ValueError('Audit evidence/counts missing')
    planning.text(value.get('evidence'),'audit evidence',3)
    expected=[job['authorized_transient']['type']] if job['authorized_transient'] else []
    coherent=(not value['permanent_changes'] and not value['unexpected_objects'] and
              value['authorized_edit_achieved'] and value['placement_legal'] and value['landmarks_visible'] and
              value['observed_added_transients']==expected)
    if value['status']=='candidate' and not coherent:
        value={**value,'status':'unknown','schema_warning':'Candidate contradicts explicit evidence or authorized count'}
    return value


def audit(args):
    cfg,jobs=load_plan(args.output_dir)
    with run.run_lock(args.output_dir):
        _,generated,_=engine()._read_state(args.output_dir,jobs,cfg)
        results=quality_records(args.output_dir,generated)
        if len(results)!=len(generated):raise ValueError('Finish verify --execute before the VLM audit')
        if not args.execute:return dict(completed_images=len(results),model_calls=0,next='audit --execute')
        campaign.ensure_stopped()
        planner_cfg=campaign.config_file(args.output_dir/'planner_config.json')
        if str(args.planner_model)!=planner_cfg['request']['model_path']:
            raise ValueError('Use the frozen planner/auditor model')
        for name,digest in planner_cfg['files_sha256'].items():
            if common.file_sha256(name)!=digest:raise ValueError('Frozen planner/auditor provider changed')
        reader=None;records=[]
        try:
            for job in jobs:
                result=results.get(job['sample_id'])
                if result is None:continue
                dest=args.output_dir/'audit'/f"{job['sample_id']}.json"
                if dest.exists():
                    entry=json.loads(dest.read_text())
                    if (entry!=run.sealed(entry) or entry['output_sha256']!=result['output_sha256'] or
                            entry['planner_fingerprint']!=planner_cfg['fingerprint']):
                        raise ValueError('Changed audit receipt')
                    records.append(entry);continue
                request=dest.with_suffix('.request.json')
                if request.exists():raise ValueError('Unresolved audit request; inspect raw response before a new bounded audit')
                common.write_json(request,dict(source_sha256=job['source_sha256'],output_sha256=result['output_sha256']))
                if reader is None:reader=planning.LocalVisionPlanner(args.planner_model)
                authorization={key:job[key] for key in
                    ('scene_summary','appearance','authorized_transient','protected_regions','with_transient')}
                # The optional object in the style proposal is not authorized
                # in its appearance-only control.
                authorization['appearance']={k:v for k,v in job['appearance'].items() if k!='transient'}
                raw=reader.audit(job['source_path'],result['output_path'],authorization)
                common.write_json(dest.with_suffix('.raw.json'),dict(response=raw))
                try:value=validate_audit(planning.parse_json(raw),job)
                except ValueError as error:
                    value=dict(status='unknown',audit_schema_error=str(error))
                entry=run.sealed(dict(sample_id=job['sample_id'],source_sha256=job['source_sha256'],
                    output_sha256=result['output_sha256'],planner_fingerprint=planner_cfg['fingerprint'],
                    verdict=value,base_quality_passed=result['base_quality_passed'],
                    candidate_passed=value['status']=='candidate' and result['base_quality_passed'],
                    eligible_for_training=False,planned_region_is_verified_mask=False))
                common.write_json(dest,entry);records.append(entry)
        finally:
            if reader is not None:reader.close()
        candidates=[]
        for entry in records:
            if entry['candidate_passed']:
                candidates.append(run.sealed({**results[entry['sample_id']],
                    'scene_audit':entry,'eligible_for_training':False,'content_status':'vlm_candidate_exploratory'}))
        common.write_jsonl(args.output_dir/'candidate_manifest.jsonl',candidates)
        summary=dict(status='audited',audited=len(records),candidates=len(candidates),
            counts=dict(Counter(e['verdict']['status'] for e in records)),training_eligible=0,
            interpretation='Automated VLM and geometry evidence; same planner family, not independent semantic ground truth')
        common.write_json(args.output_dir/'audit_summary.json',summary)
        return summary


def report(directory):
    cfg,jobs=load_plan(directory)
    with run.run_lock(directory):
        _,generated,attempts=engine()._read_state(directory,jobs,cfg)
        results={**generated,**quality_records(directory,generated)}
    out=directory.parent/'review';out.mkdir(exist_ok=True)
    groups={}
    for job in jobs:groups.setdefault(job['pair_id'],[]).append(job)
    pairs=list(groups.values());sheets=[]
    for start in range(0,len(pairs),4):
        chunk=pairs[start:start+4]
        sheet=Image.new('RGB',(1200,len(chunk)*330),'white');draw=ImageDraw.Draw(sheet)
        for i,group in enumerate(chunk):
            first=group[0];y=i*330
            draw.text((6,y+2),f"{first['city']}/{first['place_id']} | {first['style_name']} | source / appearance / + transient",fill='black')
            paths=[first['source_path']]+[results[j['sample_id']]['output_path'] if j['sample_id'] in results else None for j in group]
            for col,path in enumerate(paths):
                if path:
                    with Image.open(path) as opened:sheet.paste(ImageOps.contain(opened.convert('RGB'),(396,297)),(col*400,y+26))
            if len(paths)==2:draw.text((806,y+100),'No feasible transient was planned',fill='black')
        name=f'pairs_{start//4+1:02d}.jpg';sheet.save(out/name,quality=93);sheets.append(name)
    audit_path=directory/'audit_summary.json'
    summary=dict(planned=len(jobs),saved=len(results),reserved_calls=len(attempts),
        distinct_sources=len({j['source_sha256'] for j in jobs}),
        distinct_styles=len({j['style_name'] for j in jobs}),
        with_transient=sum(j['with_transient'] for j in jobs),
        weather=dict(Counter(j['appearance']['weather'] for j in jobs)),
        audit=json.loads(audit_path.read_text()) if audit_path.exists() else {'status':'pending'},
        eligible_for_training=0,plan_fingerprint=cfg['fingerprint'],sheets=sheets)
    common.write_json(out/'summary.json',summary)
    lines=['# 逐图 VLM 夜景与临时物体对照','',
        '每组为真实源图、逐图规划的夜景、同风格同 seed 加临时物体。未规划可行物体时第三列留空。',
        '风格名称与生成指令来自实际看图的 VLM；共同约束仅保护永久地点结构。',
        '候选审核是自动 VLM/几何证据，不是检索收益或独立语义真值；尚未训练。','',
        '```json',json.dumps(summary,ensure_ascii=False,indent=2),'```','']
    for group in pairs:
        job=group[0];lines.extend([f"## {job['city']}/{job['place_id']} — {job['style_name']}",'',
            job['scene_summary'],'',f"Lighting: {job['appearance']['lighting']}",
            f"Exposure: {job['appearance']['exposure']}",'','```text',job['prompt'],'```',''])
    lines += [f'![source / appearance / optional transient]({name})' for name in sheets]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('command',choices=('prepare','run','check','verify','audit','report'))
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'state')
    parser.add_argument('--image-root',type=Path,default=ROOT)
    parser.add_argument('--source-manifest',type=Path,default=common.WORKSPACE_ROOT/'outputs/weather_v4/state/results.jsonl')
    parser.add_argument('--source-ids',nargs='*',default=[])
    parser.add_argument('--max-sources',type=int,default=2)
    parser.add_argument('--variants',type=int,default=2)
    parser.add_argument('--paired-transients',action='store_true')
    parser.add_argument('--max-jobs',type=int,default=8)
    parser.add_argument('--max-calls',type=int,default=0)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--planner-model',type=Path,default=common.WORKSPACE_ROOT/'models/Qwen3-VL-4B-Instruct')
    parser.add_argument('--exclude-places',type=Path,default=common.WORKSPACE_ROOT/'outputs/qwen_curriculum/state/finetuning/real_baseline_v1/excluded_places.json')
    parser.add_argument('--service-config',type=Path,default=common.WORKSPACE_ROOT/'outputs/qwen_curriculum/state/all_city_campaign_config.json')
    parser.add_argument('--real-data',type=Path,default=common.WORKSPACE_ROOT/'dataset/gsv-cities')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--qwen-url',default='http://127.0.0.1:8001/generate')
    parser.add_argument('--request-timeout',type=float,default=600.)
    args=parser.parse_args(argv)
    for key in ('output_dir','image_root','source_manifest','planner_model','exclude_places','service_config','real_data'):
        setattr(args,key,getattr(args,key).expanduser().resolve())
    check_roots(args.output_dir,args.image_root)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    if args.command=='prepare':result=prepare(args)
    elif args.command=='run':
        if args.execute:generate(args)
        else:print('No model calls. Use run --execute --max-calls <positive budget>.')
        result=json.loads((args.output_dir/'summary.json').read_text())
    elif args.command=='check':
        cfg,jobs=load_plan(args.output_dir)
        with run.run_lock(args.output_dir):
            generated,results,attempts=engine()._read_state(args.output_dir,jobs,cfg)
        result=dict(planned=len(jobs),saved=len(generated),evaluated=len(results),calls=len(attempts),training_eligible=0)
    elif args.command=='audit':result=audit(args)
    elif args.command=='verify':result=verify(args)
    else:result=report(args.output_dir)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
