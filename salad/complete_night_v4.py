#!/usr/bin/env python3
"""Resume the read-only v4 audit using completed caches; never generate/train.

Default freezes inputs and reports. --phase audit --execute calibrates from
machine controls, audits cached v3 pairs and extracts only the existing v4 pairs.
Completed stages are committed atomically and verified on every subsequent run.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
ADAPT = ROOT/'AdaptVPR/experiments/qwen_curriculum'
sys.path.insert(0,str(ROOT/'AdaptVPR'))
from experiments.qwen_curriculum import content_audit_v4 as legacy
from workflow.resource_limits import monitored_run


def sha(path):
    return legacy.sha256(path)


def encoded(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(encoded(value));temporary.replace(path)


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():yield json.loads(line)


def verify_cache(manifest,cache):
    receipt=json.loads(Path(str(cache)+'.receipt.json').read_text())
    if (receipt.get('input_manifest_sha256')!=sha(manifest)
            or receipt.get('signals_sha256')!=sha(cache)):
        raise ValueError(f'Signal cache/manifest differs from its receipt: {cache}')
    ids={r['sample_id'] for r in rows(manifest)}
    seen=set();count=errors=0
    for record in rows(cache):
        count+=1;errors+=int('error' in record);seen.add(record['sample_id'])
    if count!=len(ids) or seen!=ids:
        raise ValueError(f'Signal cache does not cover this exact manifest: {cache}')
    return dict(rows=count,errors=errors,
                receipt_sha256=sha(Path(str(cache)+'.receipt.json')))


def prepare(args):
    out=args.output_dir
    if (out/'plan.json').exists() and args.v4_signals is None:
        prior=json.loads((out/'plan.json').read_text())['config'].get('v4_signals')
        if prior:args.v4_signals=Path(prior)
    config=dict(cache_root=str(args.cache_root),v4_manifest=str(args.v4_manifest),
                v4_signals=str(args.v4_signals) if args.v4_signals else None,
                checkpoint=str(args.checkpoint),backbone_repo=str(args.backbone_repo),
                real_root=str(args.real_root),exclude_places=str(args.exclude_places),
                profile=str(args.profile),device=args.device,alpha=args.false_alarm_rate)
    metadata_inventory=[str(p) for p in sorted((args.real_root/'Dataframes').glob('*.csv'))]
    if (out/'plan.json').exists():
        plan=json.loads((out/'plan.json').read_text())
        if plan['config']!=config:raise ValueError('Frozen continuation config changed; use a new output directory')
        if plan.get('metadata_inventory')!=metadata_inventory:
            raise ValueError('Frozen real-place metadata inventory changed')
        for name,checksum in plan['input_hashes'].items():
            if sha(name)!=checksum:raise ValueError('Frozen input changed: '+name)
        for name,checksum in plan['source_hashes'].items():
            if sha(name)!=checksum:raise ValueError('Frozen implementation changed: '+name)
        return plan
    cache=args.cache_root
    v3_report=json.loads((cache/'audit/report.json').read_text())
    v3_manifest=Path(v3_report['input_manifest'])
    original=dict(controls=cache/'controls/controls.jsonl',v3=v3_manifest,v4=args.v4_manifest)
    checks=dict(controls=verify_cache(original['controls'],cache/'control_signals.jsonl'),
                v3=verify_cache(v3_manifest,cache/'pair_signals.jsonl'))
    if args.v4_signals:checks['v4']=verify_cache(args.v4_manifest,args.v4_signals)
    source=[Path(__file__),ADAPT/'content_audit_v4.py',ADAPT/'audit_content_v4.py',
            ADAPT/'content_audit_v42.py',ADAPT/'audit_content_v42.py',
            ROOT/'salad/train_am.py',ROOT/'salad/train_salad.py',
            *sorted((ROOT/'salad/workflow').rglob('*.py')),
            *sorted((ROOT/'salad/models').rglob('*.py')),
            args.backbone_repo/'hubconf.py',*sorted((args.backbone_repo/'dinov2').rglob('*.py'))]
    out.mkdir(parents=True,exist_ok=True)
    snapshot={}
    for name,path in original.items():
        dest=out/'inputs'/f'{name}.jsonl';dest.parent.mkdir(exist_ok=True)
        if dest.exists():raise FileExistsError('Unsealed input snapshot exists; use a new output directory')
        dest.write_bytes(path.read_bytes());snapshot[name]=str(dest)
    inputs=[*snapshot.values(),cache/'control_signals.jsonl',cache/'pair_signals.jsonl',
            args.checkpoint,args.profile,args.exclude_places,
            *sorted((args.real_root/'Dataframes').glob('*.csv'))]
    if args.v4_signals:inputs.append(args.v4_signals)
    plan=dict(version=1,config=config,snapshots=snapshot,cache_checks=checks,
              metadata_inventory=metadata_inventory,
              input_hashes={str(p):sha(p) for p in inputs},source_hashes={str(p):sha(p) for p in source},
              original_manifests={k:dict(path=str(v),sha256=sha(v)) for k,v in original.items()},
              constraints=['no Qwen calls','no optimizer or training','no test-set calibration',
                           'old images and ledgers immutable','unknown excluded','camera remains exploratory'])
    save(out/'plan.json',plan)
    return plan


def committed(out,name):
    path=out/name
    if not path.exists():return None
    receipt=json.loads((path/'completed.json').read_text())
    for filename,checksum in receipt['files'].items():
        if sha(path/filename)!=checksum:raise ValueError('Changed completed stage: '+str(path/filename))
    return path


def stage(out,name,command):
    if path:=committed(out,name):return path
    attempts=out/'attempts';attempts.mkdir(exist_ok=True)
    index=1
    while (temporary:=attempts/f'{name}_{index:03d}').exists():index+=1
    temporary.mkdir()
    argv=command(temporary)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                                     PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
                                     CCR_CUDA_BUDGET_GIB='8',CCR_CUDA_RESERVE_GIB='4')
    limits=dict(cuda_budget_gib=8,cuda_reserve_gib=4,host_rss_limit_gib=12,host_available_reserve_gib=3)
    print(name+': '+shlex.join(argv),flush=True)
    with (temporary/'execution.log').open('w') as log:
        log.write(shlex.join(argv)+'\n');log.flush()
        monitored_run(argv,handle=log,env=env,limits=limits)
    files={p.relative_to(temporary).as_posix():sha(p) for p in temporary.rglob('*') if p.is_file()}
    save(temporary/'completed.json',dict(stage=name,files=files))
    temporary.replace(out/name)
    return out/name


def combine(out,calibration,exclude_places):
    if path:=committed(out,'combined'):return path
    selected={};counts=Counter()
    excluded={(city,int(place)) for city,place in json.loads(Path(exclude_places).read_text())}
    # v4 takes precedence per place only when it actually passed. No synthetic
    # image can overwrite a good older image merely because its policy is newer.
    for version in ('v4_audit','v3_audit'):
        for row in rows(out/version/'export/training_manifest.jsonl'):
            if row.get('condition') not in ('night','rainy_night'):continue
            key=(row['city'],int(row['place_id']))
            if key in excluded:counts['excluded_places_skipped']+=1;continue
            if key in selected:counts['duplicate_place_skipped']+=1;continue
            selected[key]=row;counts[version]+=1
    attempts=out/'attempts';attempts.mkdir(exist_ok=True)
    index=1
    while (temporary:=attempts/f'combined_{index:03d}').exists():index+=1
    temporary.mkdir()
    manifest=temporary/'training_manifest.jsonl'
    manifest.write_text(''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n'
                                 for _,row in sorted(selected.items())))
    save(temporary/'report.json',dict(counts={'pass':len(selected)},selection=dict(counts),
        policy='night/rainy_night only; at most one passed image per place; v4 preferred; no row mutation',
        training_manifest_sha256=sha(manifest),calibration_sha256=sha(calibration),
        source_manifests={name:sha(out/name/'export/training_manifest.jsonl') for name in ('v3_audit','v4_audit')}))
    save(temporary/'completed.json',dict(stage='combined',files={p.name:sha(p) for p in temporary.iterdir() if p.is_file()}))
    temporary.replace(out/'combined')
    return out/'combined'


def run_audits(plan,out):
    c=plan['config'];inputs=plan['snapshots'];cache=Path(c['cache_root'])
    python=[sys.executable,'-B'];cli=str(ADAPT/'audit_content_v42.py')
    calibration=stage(out,'calibration',lambda tmp:python+[cli,'calibrate','--manifest',inputs['controls'],
        '--signals',str(cache/'control_signals.jsonl'),'--output',str(tmp/'calibration.json'),
        '--false-alarm-rate',str(c['alpha'])])/'calibration.json'
    if json.loads(calibration.read_text()).get('usable') is not True:
        return 'blocked_control_detector'
    stage(out,'v3_audit',lambda tmp:python+[cli,'audit','--manifest',inputs['v3'],'--signals',
        str(cache/'pair_signals.jsonl'),'--calibration',str(calibration),'--output-dir',str(tmp/'export')])
    if c.get('v4_signals'):
        copy_code=('import shutil,sys;shutil.copyfile(sys.argv[1],sys.argv[2]);'
                   'shutil.copyfile(sys.argv[1]+".receipt.json",sys.argv[2]+".receipt.json")')
        signals=stage(out,'v4_signals',lambda tmp:python+['-c',copy_code,c['v4_signals'],
                                                       str(tmp/'signals.jsonl')])/'signals.jsonl'
    else:
        signals=stage(out,'v4_signals',lambda tmp:python+[str(ADAPT/'audit_content_v4.py'),'signals',
            '--manifest',inputs['v4'],'--output',str(tmp/'signals.jsonl'),'--dino-repo',c['backbone_repo'],
            '--dino-checkpoint',c['checkpoint'],'--device',c['device'],'--execute'])/'signals.jsonl'
    stage(out,'v4_audit',lambda tmp:python+[cli,'audit','--manifest',inputs['v4'],'--signals',str(signals),
        '--calibration',str(calibration),'--output-dir',str(tmp/'export')])
    combine(out,calibration,c['exclude_places'])
    return 'audits_complete'


def report(plan,out,status=None):
    c=plan['config'];data=dict(status=status or 'prepared',stages={},constraints=plan['constraints'])
    for name in ('calibration','v3_audit','v4_signals','v4_audit','combined'):
        directory=committed(out,name)
        if not directory:continue
        data['stages'][name]='complete'
        if name=='calibration':
            cal=json.loads((directory/'calibration.json').read_text())
            data['calibration']={k:v for k,v in cal.items() if k not in ('control_provenance','pipeline','provider')}
        if name in ('v3_audit','v4_audit'):
            data[name]=json.loads((directory/'export/report.json').read_text())
    if data.get('calibration',{}).get('usable') is False:data['status']='blocked_control_detector'
    if 'combined'in data['stages']:
        from workflow.night_training_preflight import build_training_preflight
        preflight=build_training_preflight(manifest=out/'combined/training_manifest.jsonl',
            audit_calibration=out/'calibration/calibration.json',audit_report=out/'combined/report.json',
            exclude_places=Path(c['exclude_places']),real_root=Path(c['real_root']),
            checkpoint=Path(c['checkpoint']),backbone_repo=Path(c['backbone_repo']),profile=Path(c['profile']),
            output_root=out/'smoke',python=sys.executable)
        data['training_preflight']=preflight
        data['status']='audits_complete_'+preflight['status']
        save(out/'training_preflight.json',preflight)
        commands=['# Generated commands only; no training was launched.']
        for entry in preflight.get('data_check_commands',[]):commands.append(shlex.join(entry['argv']))
        for entry in preflight.get('commands',[]):commands.append(shlex.join(entry['argv']))
        (out/'commands.sh').write_text('\n'.join(commands)+'\n')
        checked=[]
        check_errors={}
        for method in (() if preflight['errors'] else ('A','M','M_camera')):
            if directory:=committed(out,'data_check_'+method):
                entry=next(item for item in preflight['data_check_commands'] if item['method']==method)
                try:
                    validate_data_check(directory/'execution.log',entry['argv'])
                    checked.append(method)
                except (ValueError,KeyError) as error:check_errors[method]=str(error)
        data['data_checked_methods']=checked
        data['data_check_errors']=check_errors
        if check_errors:data['status']='blocked_filtered_sampler_pools'
        if len(checked)==3 and not preflight['errors']:
            data['status']='audits_and_data_checks_complete_training_not_started'
    save(out/'report.json',data)
    lines=['# 夜间 v4 续接状态','',f"状态：{data['status']}",'',
           '仅复用缓存、校准及审核；未启动生图或训练。','',
           '| 阶段 | 状态 |','|---|---|']
    for name in ('calibration','v3_audit','v4_signals','v4_audit','combined'):
        lines.append(f"| {name} | {data['stages'].get(name,'未完成')} |")
    for name in ('v3_audit','v4_audit'):
        if name in data:lines += ['',name+': '+json.dumps(data[name]['counts'],ensure_ascii=False)]
    lines += ['','相机参数仍为 exploratory；没有独立真实夜景校准语料时不标记 calibrated。',
              '训练命令仅供显式启动，先运行 --check-data；commands.sh 不会被本入口执行。',
              '旧 v4.1 校准失败和旧导出记录保持原样。新校准只依据 FIT 机器控制建立规则，由留出控制否决/验证。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return data


def run_data_checks(plan,out):
    current=report(plan,out)
    checks=current.get('training_preflight',{}).get('data_check_commands',[])
    if not checks:raise ValueError('Complete passing audits and training preflight first')
    for entry in checks:
        if '--check-data' not in entry['argv']:
            raise ValueError('Refusing a command without --check-data; this entry point never trains')
        directory=stage(out,'data_check_'+entry['method'],lambda _,entry=entry:entry['argv'])
        validate_data_check(directory/'execution.log',entry['argv'])


def validate_data_check(log,argv):
    """Check the filtered sampler pools, which --check-data does not construct."""
    slots=int(argv[argv.index('--synthetic-place-slots')+1])
    text=Path(log).read_text()
    if 'Data check passed; no model loaded and no training started.' not in text:
        raise ValueError('Data check did not report a successful read-only completion')
    summary=json.JSONDecoder().raw_decode(text[text.index('{\n'):])[0]
    synthetic=summary['places_with_synthetic'];total=summary['num_places']
    if slots and (synthetic<slots or total<=synthetic):
        raise ValueError('Filtered real/synthetic place pools cannot satisfy the declared sampler')
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('prepare','audit','data-check','report'),default='prepare')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/night_v4_completion_v3')
    parser.add_argument('--cache-root',type=Path,default=ROOT/'outputs/night_content_audit_v4_run1')
    parser.add_argument('--v4-manifest',type=Path,default=ROOT/'outputs/weather_v4/state/results.jsonl')
    parser.add_argument('--v4-signals',type=Path,help='Reuse an exact manifest-bound existing v4 signal cache')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'salad/checkpoint/dino_salad.ckpt')
    parser.add_argument('--backbone-repo',type=Path,default=Path('/home/admin123/.cache/torch/hub/facebookresearch_dinov2_main'))
    parser.add_argument('--real-root',type=Path,default=ROOT/'dataset/gsv-cities')
    parser.add_argument('--exclude-places',type=Path,default=ROOT/'outputs/qwen_curriculum/state/finetuning/real_baseline_v1/excluded_places.json')
    parser.add_argument('--profile',type=Path,default=ROOT/'salad/configs/synthetic_camera_exploratory.json')
    parser.add_argument('--false-alarm-rate',type=float,default=.05)
    args=parser.parse_args(argv)
    for key in ('output_dir','cache_root','v4_manifest','checkpoint','backbone_repo','real_root','exclude_places','profile'):
        setattr(args,key,getattr(args,key).expanduser().resolve())
    if args.v4_signals:args.v4_signals=args.v4_signals.expanduser().resolve()
    plan=prepare(args)
    status=None
    try:
        if args.phase=='audit' and args.execute:status=run_audits(plan,args.output_dir)
        if args.phase=='data-check' and args.execute:run_data_checks(plan,args.output_dir)
    finally:
        result=report(plan,args.output_dir,status)
        print('Status: '+result['status']+'; report='+str(args.output_dir/'report.md'),flush=True)
    if not args.execute:print('No model forwards, generation or training launched.')


if __name__=='__main__':main()
