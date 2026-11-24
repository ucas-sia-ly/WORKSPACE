#!/usr/bin/env python3
"""Plan or execute a bounded, resumable A/M/B0/B4/B5 feasibility experiment.

Default: metadata-only preparation, no model loading or training. Use --execute
explicitly. Evaluation happens between complete epochs of one fixed schedule.
"""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
METHODS = ('A', 'M', 'B0', 'B4', 'B5')
WEATHER = ('night', 'rain', 'snow', 'fog')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(encoded(value)); temporary.replace(path)


def immutable(path, content):
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f'Frozen experiment input changed: {path}; use a new output directory')
    else:
        path.write_text(content)


def load_config(path):
    c = json.loads(path.read_text())
    required = {'output_dir','real_data','synthetic_manifest','checkpoint','teacher_checkpoint',
                'backbone_repo','pair_calibration','cities','validation_places_per_weather','steps',
                'evaluation_interval','seeds','batch_size','images_per_place','learning_rate',
                'num_trainable_blocks','precision','device','num_workers','exposure_probability',
                'global_weight','feature_weight','ot_weight','mix_fraction','partition_seed',
                'max_real_drop_pp','minimum_weather_gain_pp','external_validations'}
    optional = {'pair_geometry_cache','torch_threads','eval_batch_size','methods','augment',
                'calibration_places_per_weather','calibration_tail_probability',
                'prefetch_factor','teacher_batch_size','eval_query_chunk_size',
                'eval_reference_chunk_size','eval_reference_cache_mib','resource_limits',
                'include_proxy_evaluations'}
    optional.update({'training_place_pool','real_global_distill_weight','real_global_distill_batch_size',
                     'local_distill_eval','log_mined_pairs','isolate_auxiliary_rng'})
    if required - c.keys() or c.keys() - required - optional:
        raise ValueError(f'Config keys missing={required-c.keys()}, unknown={c.keys()-required-optional}')
    for name in ('output_dir','real_data','synthetic_manifest','checkpoint','teacher_checkpoint',
                 'backbone_repo','pair_calibration','pair_geometry_cache'):
        if c.get(name):
            c[name] = str((path.parent / Path(c[name]).expanduser()).resolve())
    for name in ('steps','evaluation_interval','batch_size','images_per_place','validation_places_per_weather','num_trainable_blocks'):
        if type(c[name]) is not int or c[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if c['batch_size'] < 2 or c['images_per_place'] < 2 or c['num_workers'] < 0:
        raise ValueError('Need P>=2, K>=2 and nonnegative workers')
    if not c['cities'] or not c['seeds'] or len(c['seeds']) != len(set(c['seeds'])):
        raise ValueError('Nonempty cities and unique seeds required')
    if any(type(seed) is not int or seed < 0 for seed in c['seeds']):
        raise ValueError('Seeds must be nonnegative integers')
    for name in ('learning_rate','global_weight','feature_weight','ot_weight'):
        if not math.isfinite(c[name]) or c[name] <= 0:
            raise ValueError(f'{name} must be finite and positive')
    for name in ('max_real_drop_pp','minimum_weather_gain_pp'):
        if not math.isfinite(c[name]) or c[name] < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    for name in ('exposure_probability','mix_fraction'):
        if not 0 < c[name] <= 1:
            raise ValueError(f'{name} must be in (0,1]')
    if c['precision'] not in ('16','bf16','32'):
        raise ValueError('Use explicit precision 16/bf16/32')
    c.setdefault('methods', list(METHODS)); c.setdefault('augment', True)
    if type(c.get('include_proxy_evaluations',True)) is not bool:
        raise ValueError('include_proxy_evaluations must be a boolean')
    if c.get('training_place_pool','synthetic') not in ('synthetic','all_real'):
        raise ValueError('training_place_pool must be synthetic or all_real')
    for name in ('local_distill_eval','log_mined_pairs','isolate_auxiliary_rng'):
        if name in c and type(c[name]) is not bool: raise ValueError(f'{name} must be boolean')
    if 'real_global_distill_weight' in c and (not math.isfinite(c['real_global_distill_weight']) or c['real_global_distill_weight']<0):
        raise ValueError('real_global_distill_weight must be finite and nonnegative')
    if 'real_global_distill_batch_size' in c and (type(c['real_global_distill_batch_size']) is not int or c['real_global_distill_batch_size']<1):
        raise ValueError('real_global_distill_batch_size must be positive')
    c.setdefault('torch_threads', 2); c.setdefault('eval_batch_size', 8)
    for name,default in [('prefetch_factor',2),('teacher_batch_size',1),
                         ('eval_query_chunk_size',32),('eval_reference_chunk_size',4096)]:
        if name in c and (type(c[name]) is not int or c[name]<1):
            raise ValueError(f'{name} must be a positive integer')
    for name in ('torch_threads','eval_batch_size'):
        if type(c[name]) is not int or c[name]<1: raise ValueError(f'{name} must be a positive integer')
    if type(c.get('eval_reference_cache_mib',0)) is not int or c.get('eval_reference_cache_mib',0)<0:
        raise ValueError('eval_reference_cache_mib must be a nonnegative integer')
    if limits:=c.get('resource_limits'):
        keys={'cuda_budget_gib','cuda_reserve_gib','host_rss_limit_gib','host_available_reserve_gib'}
        if set(limits)!=keys or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in limits.values()):
            raise ValueError(f'resource_limits needs positive finite values for exactly {sorted(keys)}')
    c.setdefault('calibration_places_per_weather', 3); c.setdefault('calibration_tail_probability', .01)
    if (type(c['calibration_places_per_weather']) is not int or c['calibration_places_per_weather'] < 1
            or not 0 < c['calibration_tail_probability'] < 1):
        raise ValueError('Calibration needs positive place count and an explicit tail probability in (0,1)')
    if not c['methods'] or len(set(c['methods'])) != len(c['methods']) or set(c['methods']) - set((*METHODS,'B1')):
        raise ValueError('methods must be unique members of A/M/B0/B1/B4/B5')
    for evaluation in c['external_validations']:
        evaluation.setdefault('cadence','final')
        if evaluation['cadence'] not in ('final','every'):
            raise ValueError('Evaluation cadence must be final or every')
        if (evaluation.get('role'),evaluation.get('split')) not in (('validation','val'),('test','test')):
            raise ValueError('External evaluations require validation/val or test/test')
        if evaluation['role']=='test' and evaluation['cadence']!='final':
            raise ValueError('Test evaluations must use final cadence; do not select intermediate checkpoints on test')
        if evaluation.get('dataset') not in ('SVOX','manifest') or evaluation.get('kind') not in ('real','natural_weather'):
            raise ValueError('External validation supports SVOX or explicit manifests, kind real/natural_weather')
        evaluation['root'] = str((path.parent / evaluation['root']).resolve())
        if evaluation.get('manifest'):
            evaluation['manifest']=str((path.parent/evaluation['manifest']).resolve())
        if evaluation['dataset']=='manifest' and not evaluation.get('manifest'):
            raise ValueError('Explicit evaluation dataset requires a manifest')
        if 'query_subdirs' in evaluation:
            folders=evaluation['query_subdirs']
            if (evaluation['dataset']!='SVOX' or not isinstance(folders,list) or not folders
                    or any(not isinstance(f,str) or not f or Path(f).is_absolute() or '..' in Path(f).parts for f in folders)):
                raise ValueError('query_subdirs requires nonempty relative SVOX folders')
        if not evaluation['name'].replace('_','').isalnum():
            raise ValueError('Evaluation names must be simple alphanumeric/underscore names')
    if not any(e['kind']=='real' and e['role']=='validation' for e in c['external_validations']):
        raise ValueError('At least one independent external real validation is required')
    return c


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values(): yield from strings(v)
    elif isinstance(value, list):
        for v in value: yield from strings(v)


def prepare(c, out):
    from workflow.training_data import MixedGSVCitiesDataset, CoveragePlaceBatchSampler, _resolve_manifest_file
    from workflow.evaluation import load_evaluation_set
    from workflow.model import read_checkpoint, checkpoint_state_and_config, load_teacher_checkpoint
    payload=read_checkpoint(c['checkpoint'])
    state,student_config=checkpoint_state_and_config(payload);del state,payload
    student_config['image_size']=[224,224]
    state,teacher_config,teacher_provenance=load_teacher_checkpoint(c['teacher_checkpoint'],student_config);del state
    if any(cfg['agg_config'].get('reliability_ot',False) for cfg in (student_config,teacher_config)):
        raise ValueError('Core feasibility ablations require checkpoints without Reliability')
    dataset = MixedGSVCitiesDataset(Path(c['real_data']), Path(c['synthetic_manifest']), cities=c['cities'],
        images_per_place=c['images_per_place'], min_images_per_place=c['images_per_place'], synthetic_fraction=0,
        image_size=(224,224), augment=c['augment'], real_sampling='coverage_pk')
    places = {str(output): place for place in dataset.places for output in place.synthetic_paths}
    canonical = []
    for line, raw in enumerate(Path(c['synthetic_manifest']).read_text().splitlines(), 1):
        if not raw.strip(): continue
        row = json.loads(raw)
        if row.get('passed') is not True or row.get('eligible_for_training') is not True: continue
        output = _resolve_manifest_file(row['output_path'], manifest=Path(c['synthetic_manifest']),
            line_number=line, field_name='output_path', real_data=Path(c['real_data']))
        if str(output) not in places: continue
        place = places[str(output)]
        canonical.append({**row, 'output_path':str(output), 'source_path':str(place.source_by_synthetic_path[output]),
                          'city':place.city, 'place_id':place.place_id,
                          'condition':place.condition_by_synthetic_path[output]})
    # Calibration source paths are excluded at PLACE level, not merely file level.
    calibration = json.loads(Path(c['pair_calibration']).read_text()) if c['pair_calibration'] else {}
    calibration_paths = {str(Path(s).resolve()) for s in strings(calibration) if '/Images/' in s}
    excluded = {(p.city,p.place_id) for p in dataset.places if any(str(x) in calibration_paths for x in p.real_paths)}
    calibration_rows = []
    if not c['pair_calibration']:
        for weather in WEATHER:
            candidates=sorted([r for r in canonical if r['condition'].lower()==weather],
                key=lambda r:hashlib.sha256(f"calibration/{c['partition_seed']}/{r['output_path']}".encode()).hexdigest())
            chosen=0
            for row in candidates:
                key=row['city'],row['place_id']
                if key not in excluded:
                    excluded.add(key);calibration_rows.append(row);chosen+=1
                    if chosen==c['calibration_places_per_weather']: break
            if chosen!=c['calibration_places_per_weather']: raise ValueError(f'Insufficient {weather} calibration places')
        immutable(out/'calibration_pairs.json',encoded(calibration_rows))
    holdout, used = [], set()
    for weather in WEATHER:
        candidates = [r for r in canonical if r['condition'].lower() == weather and (r['city'],r['place_id']) not in excluded]
        candidates.sort(key=lambda r: hashlib.sha256(f"{c['partition_seed']}/{r['output_path']}".encode()).hexdigest())
        selected = []
        for row in candidates:
            key = row['city'],row['place_id']
            if key not in used:
                used.add(key); selected.append(row)
                if len(selected) == c['validation_places_per_weather']: break
        if len(selected) != c['validation_places_per_weather']:
            raise ValueError(f'Not enough disjoint {weather} places outside calibration; lower validation_places_per_weather or provide more accepted data')
        holdout.extend(selected)
    training = [r for r in canonical if (r['city'],r['place_id']) not in used | excluded]
    train_places = {(r['city'],r['place_id']) for r in training}
    exclusion_path=None
    if c.get('training_place_pool')=='all_real':
        exclusion_path=out/'excluded_places.json'
        immutable(exclusion_path,encoded([list(key) for key in sorted(used|excluded)]))
        train_places={(p.city,p.place_id) for p in dataset.places if (p.city,p.place_id) not in used|excluded}
    if len(train_places) < 2: raise ValueError('Fewer than two training places remain')
    immutable(out/'train_manifest.jsonl', ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in training))
    refs, real_queries = [], []
    for i, row in enumerate(holdout):
        place = places[row['output_path']]
        alternatives = [str(p) for p in place.real_paths if str(p) != row['source_path']]
        if not alternatives: raise ValueError('Need non-source real references')
        # Exact generating source never appears in the gallery, including across weather domains.
        refs.append(dict(id=f'place_{i}',path=alternatives[0]))
        real_queries.append(dict(id=f'real_{i}',path=row['source_path'],positives=[f'place_{i}']))
    evaluations = []
    for domain in ('real_holdout', *WEATHER):
        queries = real_queries if domain == 'real_holdout' else [
            dict(id=f'{domain}_{i}',path=r['output_path'],positives=[f'place_{i}'])
            for i,r in enumerate(holdout) if r['condition'].lower() == domain]
        path=out/f'validation_{domain}.json'
        immutable(path, encoded(dict(name=domain,references=refs,queries=queries)))
        evaluations.append(dict(name=domain,kind='real' if domain=='real_holdout' else 'synthetic_proxy',
            dataset='manifest',root=c['real_data'],manifest=str(path),split='val',role='validation'))
    if not c.get('include_proxy_evaluations',True): evaluations=[]
    evaluations += c['external_validations']
    if len({e['name'] for e in evaluations}) != len(evaluations): raise ValueError('Duplicate evaluation names')
    evaluation_summaries = {}
    for e in evaluations:
        data = load_evaluation_set(e['dataset'], Path(e['root']), manifest=Path(e['manifest']) if e.get('manifest') else None,
                                  split=e['split'],query_subdirs=e.get('query_subdirs'))
        evaluation_summaries[e['name']] = data.summary()
        identities=[(item.id,str(item.path),item.path.stat().st_size,item.path.stat().st_mtime_ns)
                    for item in data.references+data.queries]
        evaluation_summaries[e['name']]['image_inventory_sha256']=hashlib.sha256(encoded(identities).encode()).hexdigest()
    natural_batches = len(CoveragePlaceBatchSampler(len(train_places), min(c['batch_size'],len(train_places)), 0))
    per_epoch = min(natural_batches, c['evaluation_interval'])
    epochs = math.ceil(c['steps']/per_epoch)
    inputs = [c['checkpoint'],c['teacher_checkpoint'],c['synthetic_manifest']]
    if c['pair_calibration']: inputs.append(c['pair_calibration'])
    if exclusion_path: inputs.append(str(exclusion_path))
    inputs += [e['manifest'] for e in c['external_validations'] if e.get('manifest')]
    inputs += [str(Path(c['real_data'])/'Dataframes'/f'{city}.csv') for city in c['cities']]
    if c.get('pair_geometry_cache'): inputs += [str(p) for p in sorted(Path(c['pair_geometry_cache']).glob('*.json'))]
    # Freeze code as well as data/config; silently resuming a changed experiment is unsafe.
    sources = [ROOT/'train_salad.py', ROOT/'evaluate_salad.py',Path(__file__)]
    sources += sorted((ROOT/'workflow').glob('*.py')) + sorted((ROOT/'models/aggregators').glob('*.py'))
    return dict(config=c,train_places=len(train_places),train_generated_images=len(training),
        **({'teacher_provenance':teacher_provenance} if 'training_evidence' in teacher_provenance else {}),
        **({'training_exclude_places':str(exclusion_path)} if exclusion_path else {}),
        calibration_path=c['pair_calibration'] or str(out/'calibration.json'),
        calibration_pairs=calibration_rows,
        heldout_places=[list(key) for key in sorted(used)],calibration_excluded_places=len(excluded),
        train_manifest=str(out/'train_manifest.jsonl'),evaluations=evaluations,evaluation_summaries=evaluation_summaries,
        steps_per_epoch=per_epoch,epochs=epochs,actual_update_budget=c['steps'],
        evaluation_steps=[min((i+1)*per_epoch,c['steps']) for i in range(epochs)],
        input_hashes={p:digest(p) for p in sorted(set(inputs))},
        source_hashes={str(p):digest(p) for p in sources},
        limitations=['Generated-weather validation is a proxy, not natural-weather generalization.',
                    'Place disjointness applies to this fine-tuning experiment; pretraining exposure is unknown.',
                    'M changes main-batch composition and is not a loss-only ablation.',
                    'Final fixed-budget checkpoint is the prespecified comparison; no best-test checkpoint selection.'])


def training_command(plan, method, seed, directory, stop_epoch, resume):
    c=plan['config']
    cmd=[sys.executable,str(ROOT/'train_salad.py'),'--real-data',c['real_data'],
        '--synthetic-manifest',plan['train_manifest'],'--cities',*c['cities'],
        '--output-dir',str(directory),'--backbone-repo',c['backbone_repo'],
        '--epochs',str(plan['epochs']),'--stop-after-epoch',str(stop_epoch),'--max-steps',str(c['steps']),
        '--max-batches-per-epoch',str(plan['steps_per_epoch']),'--save-every',str(plan['epochs']),
        '--batch-size',str(c['batch_size']),'--images-per-place',str(c['images_per_place']),
        '--min-images-per-place',str(c['images_per_place']),'--real-sampling','coverage_pk','--seed',str(seed),
        '--num-workers',str(c['num_workers']),'--image-size','224','224',
        '--num-trainable-blocks',str(c['num_trainable_blocks']),'--learning-rate',str(c['learning_rate']),
        '--weight-decay','0','--precision',c['precision'],'--device',c['device'],'--profile-training',
        '--resume' if resume else '--init-checkpoint',str(directory/'checkpoint.pt') if resume else c['checkpoint']]
    if not c['augment']: cmd+=['--no-augment']
    if c.get('training_place_pool','synthetic')=='synthetic': cmd+=['--synthetic-places-only']
    else: cmd+=['--exclude-places',plan['training_exclude_places']]
    for name in ('local_distill_eval','log_mined_pairs','isolate_auxiliary_rng'):
        if c.get(name): cmd+=['--'+name.replace('_','-')]
    if 'prefetch_factor' in c: cmd+=['--prefetch-factor',str(c['prefetch_factor'])]
    if method in ('A','M'): cmd+=['--synthetic-fraction',str(c['mix_fraction'] if method=='M' else 0)]
    else: cmd+=['--synthetic-aux','--synthetic-exposure-prob',str(c['exposure_probability']),
                '--synthetic-global-weight',str(c['global_weight'])]
    retention=c.get('real_global_distill_weight',0.)
    if retention>0:
        cmd+=['--real-global-distill-weight',str(retention),'--real-global-distill-batch-size',str(c.get('real_global_distill_batch_size',2))]
    if method in ('B1','B4','B5'):
        cmd+=['--teacher-checkpoint',c['teacher_checkpoint'],'--teacher-batch-size',str(c.get('teacher_batch_size',1)),
              '--pair-calibration',plan['calibration_path'],'--feature-distill-weight',str(c['feature_weight']),
              '--ot-distill-weight',str(c['ot_weight'] if method!='B1' else 0),'--score-distill-weight','0',
              '--pair-confidence-mode','verified' if method=='B5' else 'uniform']
        if c.get('pair_geometry_cache'): cmd+=['--pair-geometry-cache',c['pair_geometry_cache']]
    elif retention>0: cmd+=['--teacher-checkpoint',c['teacher_checkpoint'],'--teacher-batch-size',str(c.get('teacher_batch_size',1))]
    return cmd


def run_child(command, log, threads, limits=None):
    print(shlex.join(command), flush=True)
    env={**os.environ,'OMP_NUM_THREADS':str(threads),'MKL_NUM_THREADS':str(threads)}
    if limits:
        env.update(CCR_CUDA_BUDGET_GIB=str(limits['cuda_budget_gib']),
                   CCR_CUDA_RESERVE_GIB=str(limits['cuda_reserve_gib']),
                   OPENBLAS_NUM_THREADS=str(threads), NUMEXPR_NUM_THREADS=str(threads))
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as handle:
        handle.write('\nCOMMAND '+shlex.join(command)+'\n');handle.flush()
        if limits:
            from workflow.resource_limits import monitored_run
            monitored_run(command,handle=handle,env=env,limits=limits)
        else:
            subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,env=env,check=True)


def calibrate(plan, out):
    """Small disjoint null/copy controls; no teacher tensors are saved."""
    import torch
    from workflow.model import FrozenSALADTeacher, checkpoint_state_and_config, read_checkpoint
    from workflow.reliability_cache import load_teacher_image
    from workflow.pair_correspondence import correspondence_measurements, fit_calibration
    c=plan['config']; path=Path(plan['calibration_path']); receipt=out/'calibration.receipt.json'
    if c['pair_calibration']:
        saved=json.loads(path.read_text());saved=saved[0] if isinstance(saved,list) else saved
        if saved['provenance']['teacher']['checkpoint_sha256']!=digest(c['teacher_checkpoint']):
            raise ValueError('Calibration teacher hash mismatch; set pair_calibration to null to recalibrate')
        return
    if path.exists() and receipt.exists():
        if json.loads(receipt.read_text())['sha256']!=digest(path):
            raise ValueError('Unverified or changed calibration file; preserve it and use a new experiment directory')
        return
    torch.set_num_threads(c['torch_threads'])
    from workflow.resource_limits import configure_cuda_budget
    configure_cuda_budget(c['device'])
    payload=read_checkpoint(c['teacher_checkpoint'])
    state,config=checkpoint_state_and_config(payload);del state,payload
    teacher=FrozenSALADTeacher(Path(c['teacher_checkpoint']),config,device=c['device'],backbone_repo=Path(c['backbone_repo']))
    if config['image_size']!=[224,224]: raise ValueError('Runner currently requires a 224x224 teacher')
    nulls,copies,pairs=[],[],[]
    rows=plan['calibration_pairs']
    with torch.no_grad():
        for i,row in enumerate(rows):
            other=rows[(i+1)%len(rows)]
            source=load_teacher_image(Path(row['source_path']),(224,224))[None].to(c['device'])
            generated=load_teacher_image(Path(other['output_path']),(224,224))[None].to(c['device'])
            _,sa=teacher(source);_,ga=teacher(generated)
            nulls.append({k:v.cpu() for k,v in correspondence_measurements(sa,ga).items()})
            copies.append({k:v.cpu() for k,v in correspondence_measurements(sa,sa).items()})
            pairs.append([row['source_path'],other['output_path']])
    calibration=fit_calibration(nulls,tail_probability=c['calibration_tail_probability'],search_radius=1,
        positive_controls=copies,provenance=dict(teacher=teacher.provenance,null_pairs=pairs,
            selection='places disjoint from training and validation',seed=c['partition_seed']))
    write_json(path,calibration.to_dict());write_json(receipt,dict(sha256=digest(path)))


def evaluate_checkpoint(plan, checkpoint, directory, *, final=True):
    c=plan['config']; checksum=digest(checkpoint)
    pending=[]
    for e in plan['evaluations']:
        if e.get('cadence')=='final' and not final: continue
        dest=directory/(e['name']+'.json'); receipt=directory/(e['name']+'.receipt.json')
        expected=dict(checkpoint_sha256=checksum,protocol=e)
        saved=json.loads(receipt.read_text()) if receipt.exists() else {}
        if (dest.exists() and all(saved.get(k)==v for k,v in expected.items())
                and saved.get('result_sha256')==digest(dest)):
            # Parse cached JSON too: never silently skip corrupt output.
            json.loads(dest.read_text());continue
        pending.append(e)
    # Group SVOX categories sharing a gallery: one model/gallery extraction per
    # split, then release each category's query descriptors before the next.
    groups={}
    for e in pending:
        key=(e['dataset'],e['root'],e['split']) if e['dataset']=='SVOX' and not e.get('manifest') else (e['name'],)
        groups.setdefault(key,[]).append(e)
    for evaluations in groups.values():
        e=evaluations[0];dest=directory/(e['name']+'.json')
        cmd=[sys.executable,str(ROOT/'evaluate_salad.py'),'--checkpoint',str(checkpoint),'--dataset',e['dataset'],
             '--dataset-root',e['root'],'--split',e['split'],'--output',str(dest),'--backbone-repo',c['backbone_repo'],
             '--image-size','224','224','--batch-size',str(c['eval_batch_size']),'--num-workers',str(c['num_workers']),
             '--device',c['device'],'--save-query-results']
        cmd+=['--prefetch-factor',str(c.get('prefetch_factor',2)),
              '--query-chunk-size',str(c.get('eval_query_chunk_size',32)),
              '--reference-chunk-size',str(c.get('eval_reference_chunk_size',4096)),
              '--reference-cache-mib',str(c.get('eval_reference_cache_mib',0))]
        if e.get('manifest'): cmd+=['--eval-manifest',e['manifest']]
        if len(evaluations)>1:
            group_file=directory/f"{e['split']}_query_groups.json"
            write_json(group_file,{item['name']:item.get('query_subdirs',['queries']) for item in evaluations})
            cmd+=['--svox-query-groups',str(group_file)]
        elif e.get('query_subdirs'): cmd+=['--query-subdirs',*e['query_subdirs']]
        run_child(cmd,directory/(e['name']+'.log'),c['torch_threads'],limits=c.get('resource_limits'))
        for item in evaluations:
            dest=directory/(item['name']+'.json')
            result=json.loads(dest.read_text())
            if not result.get('query_results'): raise RuntimeError('Evaluation missing per-query results')
            write_json(directory/(item['name']+'.receipt.json'),dict(checkpoint_sha256=checksum,
                protocol=item,result_sha256=digest(dest)))


def paired_changes(left,right):
    a={r['query_id']:r['rank'] for r in left['query_results']}
    b={r['query_id']:r['rank'] for r in right['query_results']}
    if a.keys()!=b.keys() or any((a[k] is None)!=(b[k] is None) for k in a):
        raise ValueError('Cannot compare changed query sets or positive validity')
    return dict(improved_to_correct=sum(a[k]>1 and b[k]==1 for k in a if a[k] is not None),
                regressed_to_wrong=sum(a[k]==1 and b[k]>1 for k in a if a[k] is not None),
                evaluated_queries=sum(v is not None for v in a.values()))


def summarize(plan,out):
    c=plan['config']; records=[]
    # Preserve metric scope/names: per-epoch means, patch counts, and measured peaks.
    training_metrics=[]
    def flatten_metrics(value, prefix=''):
        result={}
        for key,item in value.items():
            name=f'{prefix}.{key}' if prefix else key
            if isinstance(item,dict): result.update(flatten_metrics(item,name))
            elif isinstance(item,(int,float,str,bool)) or item is None: result[name]=item
        return result
    for method in c['methods']:
        for seed in c['seeds']:
            for step in plan['evaluation_steps']:
                path=out/'runs'/method/str(seed)/f'metrics_step_{step}.json'
                if path.exists():
                    training_metrics.append(dict(method=method,seed=seed,step=step,
                        **{f'metric.{k}':v for k,v in flatten_metrics(json.loads(path.read_text())).items()}))
    metric_fields=['method','seed','step']+sorted({k for r in training_metrics for k in r}-{'method','seed','step'})
    with (out/'training_metrics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=metric_fields);writer.writeheader();writer.writerows(training_metrics)
    for method in c['methods']:
        for seed in c['seeds']:
            for step in plan['evaluation_steps']:
                for e in plan['evaluations']:
                    p=out/'runs'/method/str(seed)/'evaluations'/str(step)/(e['name']+'.json')
                    initial=out/'initial'/(e['name']+'.json')
                    if not p.exists() or not initial.exists(): continue
                    current,baseline=json.loads(p.read_text()),json.loads(initial.read_text())
                    records.append(dict(method=method,seed=seed,step=step,domain=e['name'],kind=e['kind'],
                        role=e.get('role','validation'),split=e.get('split','val'),
                        **current['recall'],delta_initial_pp=100*(current['recall']['R@1']-baseline['recall']['R@1']),
                        **paired_changes(baseline,current)))
    fields=['method','seed','step','domain','kind','role','split','R@1','R@5','R@10','delta_initial_pp',
            'improved_to_correct','regressed_to_wrong','evaluated_queries']
    with (out/'learning_curves.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(records)
    final=[r for r in records if r['step']==c['steps']]
    comparisons=[]
    lookup={(r['method'],r['seed'],r['domain']):r for r in final}
    for candidate,reference in [('M','A'),('B0','A'),('B4','B0'),('B5','B4'),('B5','A')]:
        domains={}
        for e in plan['evaluations']:
            deltas=[100*(lookup[candidate,s,e['name']]['R@1']-lookup[reference,s,e['name']]['R@1'])
                for s in c['seeds'] if (candidate,s,e['name']) in lookup and (reference,s,e['name']) in lookup]
            if deltas:
                domains[e['name']]=dict(kind=e['kind'],seeds=len(deltas),deltas_pp=deltas,
                    role=e.get('role','validation'),split=e.get('split','val'),
                    mean_pp=statistics.mean(deltas),std_pp=statistics.stdev(deltas) if len(deltas)>1 else None)
                changes=[]
                for seed in c['seeds']:
                    base=out/'runs'/reference/str(seed)/'evaluations'/str(c['steps'])/(e['name']+'.json')
                    trial=out/'runs'/candidate/str(seed)/'evaluations'/str(c['steps'])/(e['name']+'.json')
                    if base.exists() and trial.exists():
                        changes.append(dict(seed=seed,**paired_changes(json.loads(base.read_text()),json.loads(trial.read_text()))))
                domains[e['name']]['paired_query_changes']=changes
        if not domains: continue
        complete=len(domains)==len(plan['evaluations']) and all(v['seeds']==len(c['seeds']) for v in domains.values())
        validation=[v for v in domains.values() if v['role']=='validation']
        real=[v['mean_pp'] for v in validation if v['kind']=='real']
        weather=[v['mean_pp'] for v in validation if v['kind']=='synthetic_proxy']
        natural=[v['mean_pp'] for v in validation if v['kind']=='natural_weather']
        signal=(complete and bool(real) and bool(weather) and min(real)>=-c['max_real_drop_pp'] and statistics.mean(weather)>c['minimum_weather_gain_pp'])
        if natural: signal=signal and statistics.mean(natural)>c['minimum_weather_gain_pp']
        comparisons.append(dict(candidate=candidate,reference=reference,domains=domains,
            verdict=('incomplete' if not complete else 'proxy_signal_requires_multiseed_and_natural_weather_confirmation'
                     if signal else 'descriptive_only_no_proxy_validation' if not weather
                     else 'no_prespecified_proxy_gain_or_real_regression'),
            paired_seed_count=len(c['seeds']),natural_weather_domains=len(natural)))
    report=dict(final=final,comparisons=comparisons,limitations=plan['limitations'],
                test_policy='Test categories are descriptive fixed-budget reports; excluded from validation decision gates.',
                status='No natural-weather feasibility claim is made by this runner.')
    write_json(out/'report.json',report)
    lines=['# CCR 可行性实验报告','',report['status'],'',
           '| 对比 | 当前结论 |','|---|---|']
    lines += [f"| {r['candidate']} vs {r['reference']} | {r['verdict']} |" for r in comparisons]
    lines+=['','| 方法 | seed | 数据集类别 | split | R@1 | R@5 | R@10 |',
            '|---|---|---|---|---:|---:|---:|']
    lines += [f"| {r['method']} | {r['seed']} | {r['domain']} | {r['split']} | {100*r['R@1']:.2f}% | {100*r['R@5']:.2f}% | {100*r['R@10']:.2f}% |" for r in final]
    lines+=['','Test 类别仅作固定预算结果报告，不参与 validation 决策门槛。',
            '完整数据：report.json；逐点评估：learning_curves.csv；损失/曝光/耗时/显存：training_metrics.csv。',
            '真实图退化容许值和代理天气增益门槛在运行前固定；未达门槛不证明方法永远无效。',
            '单种子没有跨种子方差；多种子改善也不能替代自然天气验证。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def execute(plan,out):
    from workflow.model import read_checkpoint
    c=plan['config']
    run_child([sys.executable,str(Path(__file__).resolve()),'--calibrate-plan',str(out/'plan.json')],
              out/'calibration.log',c['torch_threads'],limits=c.get('resource_limits'))
    evaluate_checkpoint(plan,Path(c['checkpoint']),out/'initial')
    for seed in c['seeds']:
        for method in c['methods']:
            directory=out/'runs'/method/str(seed);directory.mkdir(parents=True,exist_ok=True)
            checkpoint=directory/'checkpoint.pt'
            while True:
                epoch=step=0
                if checkpoint.exists():
                    saved=read_checkpoint(checkpoint);epoch,step=saved['epoch'],saved['global_step']
                    if saved['training_config'].get('max_steps')!=c['steps']:
                        raise ValueError('Checkpoint budget disagrees with frozen plan')
                    write_json(directory/f'metrics_step_{step}.json',saved['metrics'])
                    del saved
                    evaluate_checkpoint(plan,checkpoint,directory/'evaluations'/str(step),final=step==c['steps'])
                    summarize(plan,out)
                if step==c['steps']: break
                if step>c['steps'] or epoch>=plan['epochs']: raise RuntimeError('Training schedule and update budget disagree')
                cmd=training_command(plan,method,seed,directory,epoch+1,checkpoint.exists())
                run_child(cmd,directory/'train.log',c['torch_threads'],limits=c.get('resource_limits'))
                if not checkpoint.exists(): raise RuntimeError('Training returned without a checkpoint')
                saved=read_checkpoint(checkpoint)
                if saved['epoch']!=epoch+1 or saved['global_step']!=min(step+plan['steps_per_epoch'],c['steps']):
                    raise RuntimeError('Training did not complete the expected bounded epoch')
                del saved
    summarize(plan,out)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path)
    p.add_argument('--calibrate-plan',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--execute',action='store_true',help='Explicitly launch bounded training and validation; otherwise only prepare plan')
    p.add_argument('--summarize-only',action='store_true')
    args=p.parse_args(argv)
    if args.calibrate_plan:
        calibrate(json.loads(args.calibrate_plan.read_text()),args.calibrate_plan.parent);return
    if args.config is None: p.error('--config is required')
    if args.execute and args.summarize_only: p.error('Choose execute or summarize-only')
    c=load_config(args.config.resolve());out=Path(c['output_dir']);out.mkdir(parents=True,exist_ok=True)
    with (out/'.lock').open('w') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('Another process owns this experiment directory')
        if args.summarize_only:
            summarize(json.loads((out/'plan.json').read_text()),out);return
        plan=prepare(c,out)
        immutable(out/'plan.json',encoded(plan))
        commands=[]
        for seed in c['seeds']:
            for method in c['methods']:
                commands.append(shlex.join(training_command(plan,method,seed,out/'runs'/method/str(seed),1,False)))
        (out/'first_epoch_commands.txt').write_text('\n'.join(commands)+'\n')
        print(f"Plan: {out/'plan.json'}\nTraining places={plan['train_places']}; updates/run={c['steps']}; runs={len(c['methods'])*len(c['seeds'])}",flush=True)
        if args.execute:
            try:
                write_json(out/'status.json',dict(status='running'))
                execute(plan,out)
            except BaseException as error:
                write_json(out/'status.json',dict(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                    error=str(error),recovery='Rerun the same --config --execute; the last completed epoch and evaluations are reused.'))
                raise
            write_json(out/'status.json',dict(status='completed'))
        else: print('Prepared only. No GPU/model/training was started. Add --execute to run.',flush=True)


if __name__=='__main__':
    main()
