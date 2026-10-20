#!/usr/bin/env python3
"""Stage 4 attribution and gated local ablations; never invent a Stage 3 winner."""
from __future__ import annotations
import argparse
from collections import Counter
import fcntl
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

import run_am_multiseed as am
from workflow.am_statistics import paired,holm,query_ranks
ccr=am.ccr
FORMAL=('A','B0','B-Feature','B-OT','B4','B5','B-OT-train')
CONTROLS=('A','B0-zero','B0-zero-isolated','B0-original','B0-isolated')
REFERENCES={'A':('initial',),'B0':('initial','A'),'B-Feature':('initial','A','B0'),
            'B-OT':('initial','A','B0'),'B4':('initial','A','B0','B-Feature','B-OT'),
            'B5':('initial','A','B0','B4'),'B-OT-train':('initial','A','B0','B-OT')}


def load_config(path):
    c=json.loads(path.read_text())
    for key in ('stage3_dir','output_dir','geometry_cache'):
        if c.get(key): c[key]=str((path.parent/Path(c[key]).expanduser()).resolve())
    if c['stage3_dir']==c['output_dir']: raise ValueError('Stage 4 requires a new directory')
    weights=c['global_weights']
    if (len(weights)!=len(set(weights)) or 0. not in weights or not any(w>0 for w in weights)
            or any(not math.isfinite(w) or w<0 for w in weights)): raise ValueError('Invalid declared global weights')
    for key in ('temperature','feature_weight','ot_weight'):
        if not math.isfinite(c[key]) or c[key]<=0: raise ValueError('Invalid '+key)
    if not 0<=c['exposure_probability']<=1 or not 0<c['calibration_tail_probability']<1: raise ValueError('Invalid probability')
    if c['local_mode']!='salad_dropout_off': raise ValueError('Main local ablations share SALAD-only Dropout-off mode; OT-train is separate')
    for key in ('gradient_interval','probe_pairs_per_weather','smoke_batches'):
        if type(c[key]) is not int or c[key]<1: raise ValueError('Invalid '+key)
    for key in ('max_B0_drop_pp','max_B0_seed_delta_std_pp'):
        if not math.isfinite(c[key]) or c[key]<0: raise ValueError('Invalid gate')
    return c


def prepare(c,out):
    parent_path=Path(c['stage3_dir'])/'plan.json'
    parent=json.loads(parent_path.read_text())
    am.validate_plan(parent,Path(c['stage3_dir']))
    from workflow.training_data import MixedGSVCitiesDataset
    dataset=MixedGSVCitiesDataset(Path(parent['config']['real_data']),synthetic_manifest=None,real_sampling='coverage_pk')
    split=json.loads(Path(parent['config']['reserved_places']).read_text())['groups']
    calibration={tuple(k) for k in split['calibration']};validation={tuple(k) for k in split['validation']}
    snapshot=Path(c['stage3_dir'])/'inputs/accepted_manifest.jsonl'
    rows=[json.loads(line) for line in snapshot.read_text().splitlines()]
    cal,probes={},[]
    weather=Counter()
    for row in sorted(rows,key=lambda r:r['output_path']):
        key=dataset.source_index[Path(row['source_path'])]
        if key in calibration: cal.setdefault(key,row)
        if key in validation and weather[row['condition']]<c['probe_pairs_per_weather']:
            probes.append(row);weather[row['condition']]+=1
    if len(cal)<2 or not probes: raise ValueError('Need distinct calibration places and held-out diagnostic pairs')
    source_paths=[Path(__file__),ccr.ROOT/'train_ccr4.py',ccr.ROOT/'workflow/ccr4_diagnostics.py']
    geometry={str(p):ccr.digest(p) for p in sorted(Path(c['geometry_cache']).glob('*.json'))} if c.get('geometry_cache') else {}
    plan=dict(version=1,config=c,parent=parent,parent_plan_sha256=ccr.digest(parent_path),
        calibration_pairs=list(cal.values()),probe_pairs=probes,geometry_hashes=geometry,
        source_hashes={str(p):ccr.digest(p) for p in source_paths},
        policies=dict(selection_seed=parent['config']['screen_seed'],formal_seeds=parent['config']['confirmation_seeds'],
            comparison='fixed final full-epoch budget, independent initialization, no test',
            rng='CPU/active CUDA isolated by seed/global_step/branch; original controls explicitly retain shared RNG',
            local='initial frozen teacher; common A support; no semantic cutoff additions; FP32 epsilon numerical OT guard only',
            global_selection='isolated nonzero candidate passing A-relative validation guard; maximum val R1; ties ascending weight',
            local_gate='all formal A/B0 seeds complete; A mining gates; each B0-A R1 delta >= -max_B0_drop_pp and delta std <= max_B0_seed_delta_std_pp',
            real_weather='real training GSV has no weather labels; report unannotated rather than infer from paired edits'),
        limitations=['Stage 3 formal hyperparameters must exist before any dataset training.',
            'No natural-weather validation domain is fabricated from test or synthetic weather.',
            'Gradient norms are sampled, actual aggregate clipping frequency comes from every training batch.',
            'No additive Dropout floor is subtracted from any loss.',
            'Local weights are declared hypotheses, not gradient-balanced optima.'])
    am.immutable(out/'plan.json',ccr.encoded(plan))
    return plan


def validate(plan,out):
    c=plan['config'];root=Path(c['stage3_dir'])
    if ccr.digest(root/'plan.json')!=plan['parent_plan_sha256']: raise ValueError('Stage 3 data protocol changed')
    am.validate_plan(plan['parent'],root)
    for path,sha in {**plan['source_hashes'],**plan['geometry_hashes']}.items():
        if ccr.digest(path)!=sha: raise ValueError('Stage 4 frozen code/geometry changed: '+path)
    current={str(p):ccr.digest(p) for p in sorted(Path(c['geometry_cache']).glob('*.json'))} if c.get('geometry_cache') else {}
    if current!=plan['geometry_hashes']: raise ValueError('Geometry cache inventory changed')


def freeze(plan,out):
    path=Path(plan['config']['stage3_dir'])/'formal_protocol.json'
    if not path.exists(): raise ValueError('Stage 3 formal_protocol.json missing; finish Stage 3 selection and --phase freeze first')
    formal=json.loads(path.read_text())
    prior=[]
    for stage in am.STAGES:
        record=am.selection_record(plan['parent'],Path(plan['config']['stage3_dir']),stage,prior)
        if record is None or record['selected'] is None: raise ValueError('Unverified Stage 3 selection')
        prior.append(record)
    if formal['spec']!=prior[-1]['selected'] or formal['plan_sha256']!=plan['parent_plan_sha256']:
        raise ValueError('Stage 3 formal protocol does not match verified selection')
    result=dict(stage3_formal_sha256=ccr.digest(path),spec=formal['spec'],seeds=formal['seeds'],
                epochs=formal['total_epochs'],updates=formal['optimizer_updates'],plan_sha256=ccr.digest(out/'plan.json'))
    am.immutable(out/'protocol.json',ccr.encoded(result))
    return result


def weight_name(value): return 'W'+format(value,'.6g').replace('.','p')


def specification(plan,method,global_weight=None):
    c=plan['config']
    result=dict(aux=method!='A',global_weight=.1,feature=0.,ot=0.,confidence='uniform',rng='isolated',local_mode=c['local_mode'])
    if method=='A': result['global_weight']=0.
    elif method in ('B0-zero','B0-zero-isolated'): result['global_weight']=0.
    elif method.startswith('W'): result['global_weight']=next(w for w in c['global_weights'] if weight_name(w)==method)
    elif method in FORMAL:
        if global_weight is None: raise ValueError('Freeze B0 weight before formal/local experiments')
        result['global_weight']=global_weight
    elif method not in CONTROLS: raise ValueError('Unknown method')
    if method in ('B0-original','B0-zero'): result['rng']='original'
    if method in ('B-Feature','B4','B5'): result['feature']=c['feature_weight']
    if method in ('B-OT','B4','B5','B-OT-train'): result['ot']=c['ot_weight']
    if method=='B5': result['confidence']='verified'
    if method=='B-OT-train': result['local_mode']='train'
    return result


def command(plan,protocol,method,seed,directory,epoch,resume=False,global_weight=None,smoke=False):
    parent=plan['parent'];c=parent['config'];s=specification(plan,method,global_weight)
    cmd=am.baseline.training_command(parent,protocol['spec'],seed,directory,epoch,resume)
    cmd[1]=str(ccr.ROOT/'train_ccr4.py')
    cmd+=['--stage4-rng',s['rng'],'--stage4-local-mode',s['local_mode'],
          '--stage4-gradient-interval',str(1 if smoke else plan['config']['gradient_interval'])]
    if s['aux']:
        cmd+=['--synthetic-manifest',parent['train_manifest'],'--synthetic-aux',
              '--synthetic-exposure-prob',str(plan['config']['exposure_probability']),
              '--synthetic-global-weight',str(s['global_weight']),'--synthetic-global-temperature',str(plan['config']['temperature'])]
    if s['feature'] or s['ot']:
        cmd+=['--teacher-checkpoint',c['checkpoint'],'--teacher-batch-size','2',
              '--pair-calibration',str(Path(plan['config']['output_dir'])/'calibration.json'),
              '--feature-distill-weight',str(s['feature']),'--ot-distill-weight',str(s['ot']),
              '--pair-confidence-mode',s['confidence']]
        if plan['config']['geometry_cache']: cmd+=['--pair-geometry-cache',plan['config']['geometry_cache']]
    if smoke: cmd+=['--max-batches-per-epoch',str(plan['config']['smoke_batches'])]
    return cmd


def checked(saved,plan,protocol,method,seed,global_weight,smoke):
    # Reconstruct the exact objective and stage-4 policy through the saved config
    # checks below; the trainer also requires full config equality on resume.
    c=plan['parent']['config'];s=specification(plan,method,global_weight);t=saved['training_config']
    expected_steps=am.run_plan(plan['parent'],protocol['spec'])['steps_per_epoch'][protocol['spec']['name']]
    if smoke: expected_steps=min(expected_steps,plan['config']['smoke_batches'])
    if (saved['global_step']!=saved['epoch']*expected_steps or saved['total_epochs']!=protocol['epochs']
            or t['seed']!=seed or t['init_checkpoint_sha256']!=plan['parent']['input_hashes'][c['checkpoint']]
            or t['excluded_places']['sha256']!=plan['parent']['input_hashes'][plan['parent']['excluded_places']]
            or t['learning_rate']!=protocol['spec']['learning_rate'] or t['batch_size']!=protocol['spec']['batch_size']
            or t['model_config']['backbone_config']['num_trainable_blocks']!=protocol['spec']['num_trainable_blocks']
            or any(k in t for k in ('reliability','real_descriptor_retention'))
            or t['synthetic_fraction']!=0 or t['synthetic_places_only'] or t['real_sampling']!='coverage_pk'
            or t.get('synthetic_aux',{}).get('global_weight',0)!=s['global_weight']
            or bool(t.get('synthetic_aux'))!=s['aux'] or t['stage4']['rng']!=s['rng'] or t['stage4']['local_mode']!=s['local_mode']):
        raise ValueError('Checkpoint method, seed, objective or budget differs')
    if (saved['precision']!=c['precision'] or t['num_workers']!=c['num_workers'] or t['real_data']!=c['real_data']
            or t['metadata_sha256']!=am.metadata_digest(c,plan['parent']['train_manifest'] if s['aux'] else None)
            or t['images_per_place']!=c['images_per_place'] or not t['augment']
            or t['max_batches_per_epoch']!=(plan['config']['smoke_batches'] if smoke else 0)
            or t['stage4']['gradient_interval']!=(1 if smoke else plan['config']['gradient_interval'])):
        raise ValueError('Checkpoint data, augmentation or measurement policy differs')
    if s['aux'] and (t['synthetic_aux']['exposure_prob']!=plan['config']['exposure_probability']
                     or t['synthetic_aux']['global_temperature']!=plan['config']['temperature']):
        raise ValueError('Auxiliary exposure or temperature differs')
    local=t.get('local_distillation',{})
    if (local.get('feature_weight',0)!=s['feature'] or local.get('ot_weight',0)!=s['ot'] or local.get('score_weight',0)!=0
            or local.get('confidence_mode','uniform')!=s['confidence']): raise ValueError('Local ablation differs')
    if not all(k in saved for k in ('optimizer_state_dict','scheduler_state_dict','rng_state','sampling_coverage')): raise ValueError('Incomplete resume')
    if saved['scheduler_state_dict']['T_max']!=protocol['epochs'] or saved['scheduler_state_dict']['last_epoch']!=saved['epoch']: raise ValueError('Schedule differs')


def execute_one(plan,out,protocol,method,seed,stage,target,global_weight=None):
    from workflow.model import read_checkpoint
    directory=out/stage/str(seed)/method
    directory.mkdir(parents=True,exist_ok=True)
    smoke=stage=='smoke'
    spec=specification(plan,method,global_weight)
    if spec['feature'] or spec['ot']:
        calibration=out/'calibration.json'
        if not calibration.exists(): raise ValueError('Run --phase calibrate --execute before local supervision')
        payload=json.loads(calibration.read_text())
        if payload['provenance']['teacher']['checkpoint_sha256']!=plan['parent']['input_hashes'][plan['parent']['config']['checkpoint']]:
            raise ValueError('Calibration teacher differs from the frozen initial teacher')
        am.immutable(out/'local_protocol.json',ccr.encoded(dict(calibration_sha256=ccr.digest(calibration),
            geometry_hashes=plan['geometry_hashes'],teacher_sha256=payload['provenance']['teacher']['checkpoint_sha256'],
            feature_weight=plan['config']['feature_weight'],ot_weight=plan['config']['ot_weight'],
            exposure_probability=plan['config']['exposure_probability'],support='common teacher A only')))
    contract=dict(method=method,seed=seed,stage=stage,target_epoch=target,global_weight=global_weight,protocol=protocol,
                  specification=specification(plan,method,global_weight),plan_sha256=ccr.digest(out/'plan.json'))
    am.immutable(directory/'contract.json',ccr.encoded(contract))
    checkpoint=directory/'checkpoint.pt';previous=-1
    while True:
        epoch=0
        if checkpoint.exists():
            saved=read_checkpoint(checkpoint)
            checked(saved,plan,protocol,method,seed,global_weight,smoke)
            epoch=saved['epoch']
            if epoch<=previous or epoch>target: raise ValueError('Unexpected epoch progression')
            am.immutable(directory/f'metrics_epoch_{epoch:03d}.json',ccr.encoded(saved['metrics']))
            del saved
        if epoch==target: break
        previous=epoch
        ccr.run_child(command(plan,protocol,method,seed,directory,epoch+1,checkpoint.exists(),global_weight,smoke),
                      directory/'train.log',plan['parent']['config']['torch_threads'],limits=plan['parent']['config']['resource_limits'])
        if not checkpoint.exists(): raise RuntimeError('Missing completed checkpoint')
    if not smoke:
        ccr.evaluate_checkpoint(am.ep_plan(plan['parent']),checkpoint,directory/'validation')
    ccr.run_child([sys.executable,str(Path(__file__).resolve()),'--config',str(out/'config.json'),'--worker','audit','--run-dir',str(directory)],
                  directory/'audit.log',plan['parent']['config']['torch_threads'],limits=plan['parent']['config']['resource_limits'])


def result(plan,directory):
    cp=directory/'checkpoint.pt'
    return am.read_result(plan['parent'],directory/'validation',cp) if cp.exists() else None


def select_global(plan,out):
    seed=plan['parent']['config']['screen_seed'];base=result(plan,out/'global-screen'/str(seed)/'A')
    if base is None: raise ValueError('Complete global-screen A first')
    metric=json.loads((out/'global-screen'/str(seed)/'A/metrics_epoch_001.json').read_text())
    if am.baseline.signal_failures(metric,plan['parent']): raise ValueError('A screening has insufficient signal')
    rows=[]
    for weight in sorted(plan['config']['global_weights']):
        trial=result(plan,out/'global-screen'/str(seed)/weight_name(weight))
        if trial is None: raise ValueError('Complete every prespecified global weight')
        delta=100*(trial['svox_val']['recall']['R@1']-base['svox_val']['recall']['R@1'])
        rows.append(dict(weight=weight,delta_A_pp=delta,R1=trial['svox_val']['recall']['R@1'],eligible=weight>0 and delta>=-plan['config']['max_B0_drop_pp']))
    candidates=[r for r in rows if r['eligible']]
    if not candidates: raise ValueError('No acceptable nonzero B0 candidate; stop local formal research')
    selected=max(candidates,key=lambda r:r['R1'])['weight']
    selection=dict(rows=rows,weight=selected,rule=plan['policies']['global_selection'],seed=seed)
    am.immutable(out/'global_selection.json',ccr.encoded(selection))
    return selected


def b0_gate(plan,out):
    deltas=[];failures=[];a_values=[]
    initial=am.read_result(plan['parent'],out/'initial',Path(plan['parent']['config']['checkpoint']))
    if initial is None: return dict(passed=False,reason='initial validation incomplete')
    for seed in plan['parent']['config']['confirmation_seeds']:
        a=result(plan,out/'formal'/str(seed)/'A');b=result(plan,out/'formal'/str(seed)/'B0')
        if a is None or b is None: return dict(passed=False,reason='formal A/B0 seeds incomplete')
        for method in ('A','B0'):
            audit_path=out/'formal'/str(seed)/method/'audit.json'
            if not audit_path.exists(): return dict(passed=False,reason='final-budget audit incomplete')
            audit=json.loads(audit_path.read_text())
            protocol=out/'protocol.json'
            if (not protocol.exists() or audit['actual_epochs']!=plan['parent']['config']['epochs']
                    or audit['optimizer_updates']!=json.loads(protocol.read_text())['updates']
                    or audit['checkpoint_sha256']!=ccr.digest(audit_path.parent/'checkpoint.pt')):
                return dict(passed=False,reason='wrong final budget or stale audit')
            if method=='A':
                for metrics in audit['epoch_metrics']: failures+=am.baseline.signal_failures(metrics,plan['parent'])
        deltas.append(100*(b['svox_val']['recall']['R@1']-a['svox_val']['recall']['R@1']))
        a_values.append(100*a['svox_val']['recall']['R@1'])
        for method,values in (('A',a),('B0',b)):
            if 100*(values['svox_val']['recall']['R@1']-initial['svox_val']['recall']['R@1']) < -plan['parent']['config']['acceptance']['max_val_drop_pp']:
                failures.append(f'{method}/{seed}:validation_drop_from_initial')
    std=statistics.stdev(deltas)
    if statistics.stdev(a_values)>plan['parent']['config']['acceptance']['max_seed_std_pp']: failures.append('A_seed_variance')
    return dict(passed=not failures and min(deltas)>=-plan['config']['max_B0_drop_pp'] and std<=plan['config']['max_B0_seed_delta_std_pp'],
                deltas_pp=deltas,std_pp=std,A_signal_failures=failures,interpretation='engineering acceptability, not proof of benefit')


def calibrate(plan,out):
    parent=plan['parent']['config']
    c={**parent, 'teacher_checkpoint':parent['checkpoint'],'pair_calibration':None,
       'calibration_tail_probability':plan['config']['calibration_tail_probability'],'partition_seed':parent['screen_seed']}
    ccr.calibrate(dict(config=c,calibration_path=str(out/'calibration.json'),calibration_pairs=plan['calibration_pairs']),out)


def audit(plan,directory):
    import torch
    from workflow.model import read_checkpoint,checkpoint_state_and_config,load_checkpoint_model
    from workflow.reliability_cache import load_teacher_image
    from workflow.ccr4_diagnostics import read_committed,summarize_attribution
    c=plan['parent']['config'];cp=directory/'checkpoint.pt';sha=ccr.digest(cp)
    target=directory/'audit.json'
    if target.exists() and json.loads(target.read_text()).get('checkpoint_sha256')==sha: return
    saved=read_checkpoint(cp);contract=json.loads((directory/'contract.json').read_text())
    checked(saved,plan,contract['protocol'],contract['method'],contract['seed'],contract['global_weight'],contract['stage']=='smoke')
    initial,_=checkpoint_state_and_config(read_checkpoint(c['checkpoint']))
    state_hash=hashlib.sha256()
    for name,tensor in saved['state_dict'].items():
        state_hash.update(name.encode()+tensor.detach().cpu().contiguous().numpy().tobytes())
    displacement={}
    with torch.no_grad():
        for prefix in ('backbone','aggregator'):
            delta=norm=0.
            for name,value in saved['state_dict'].items():
                if name.startswith(prefix+'.') and value.is_floating_point():
                    a,b=value.double(),initial[name].double();delta+=float((a-b).square().sum());norm+=float(b.square().sum())
            displacement[prefix]=dict(l2=math.sqrt(delta),relative_l2=math.sqrt(delta/norm) if norm else None)
    epochs=saved['epoch'];steps=saved['global_step'];summary=saved['dataset_summary'];training=saved['training_config']
    del initial,saved
    from workflow.resource_limits import configure_cuda_budget
    configure_cuda_budget(c['device'])
    # Fixed reserved real source probe, no query/test selection by model outputs.
    paths=list(dict.fromkeys(row['source_path'] for row in plan['probe_pairs']))
    descriptions=[]
    for checkpoint in (Path(c['checkpoint']),cp):
        model=load_checkpoint_model(checkpoint,c['device'],backbone_repo=Path(c['backbone_repo']))
        model.eval();values=[]
        with torch.no_grad():
            for path in paths:
                values.append(model(load_teacher_image(Path(path),(224,224))[None].to(c['device'])).float().cpu())
        descriptions.append(torch.cat(values));del model
    first,last=descriptions
    descriptor=dict(images=len(paths),paths=paths,mode='FP32 eval fixed reserved sources',
        mean_one_minus_cosine=float((1-torch.nn.functional.cosine_similarity(first,last)).mean()),
        mean_l2=float((first-last).norm(dim=1).mean()))
    rows=[json.loads((directory/f'metrics_epoch_{e:03d}.json').read_text()) for e in range(1,epochs+1)]
    resources=[json.loads(line[9:]) for line in (directory/'train.log').read_text().splitlines() if line.startswith('RESOURCE ')]
    attribution=read_committed(directory/'attribution.jsonl',steps)
    weather={}
    for row in attribution:
        for name,data in row['weather'].items():
            record=weather.setdefault(name,dict(queries=0,loss_sum=0.))
            for k in record: record[k]+=data[k]
    for record in weather.values(): record['mean_loss']=record['loss_sum']/record['queries']
    real=sum(r['real_exposure'] for r in rows);syn=sum(r.get('synthetic_aux',{}).get('triggered_places',0) for r in rows)
    # Observer counts are a second independent measure of consumed auxiliary queries.
    observed=sum(r['actual_auxiliary_queries'] for r in attribution)
    if observed!=syn: raise ValueError('Auxiliary exposure accounting differs from trainer')
    ccr.write_json(target,dict(checkpoint_sha256=sha,optimizer_updates=steps,actual_epochs=epochs,training_config=training,dataset_summary=summary,
        parameter_displacement=displacement,descriptor_displacement=descriptor,epoch_metrics=rows,
        lr_curve=[dict(epoch=r['epoch'],used=contract['protocol']['spec']['learning_rate']*(1+math.cos(math.pi*(r['epoch']-1)/contract['protocol']['epochs']))/2,
                       next=r['learning_rate_next_epoch']) for r in rows],
        real_exposures=real,real_weather={'unannotated_GSV':real},auxiliary_queries=syn,auxiliary_fraction=syn/(real+syn),
        synthetic_weather=weather,resource_attempts=resources,gradient_samples=sum(r['gradient_sampled'] for r in attribution),
        parameter_sha256=state_hash.hexdigest(),attribution_summary=summarize_attribution(attribution),
        sampled_diagnostics='attribution.jsonl; gradients sampled, weather and support accounted on every consumed step'))


def diagnose(plan,out,checkpoint):
    import torch
    from PIL import Image
    from workflow.model import FrozenSALADTeacher,SALADModel,read_checkpoint,checkpoint_state_and_config
    from workflow.reliability_cache import preprocess_teacher_image
    from workflow.diagnostic_local import local_consistency_probe
    from workflow.resource_limits import configure_cuda_budget
    from unittest.mock import patch
    from contextlib import ExitStack
    protocol_path=out/'protocol.json'
    if not protocol_path.exists(): raise ValueError('Freeze Stage 3-selected blocks before comparing diagnostic train/eval modes')
    protocol=json.loads(protocol_path.read_text())
    c=plan['parent']['config'];configure_cuda_budget(c['device'])
    if c['device']=='cpu':
        import os
        os.environ['XFORMERS_DISABLED']='1'
    state,config=checkpoint_state_and_config(read_checkpoint(checkpoint))
    config['backbone_config']['num_trainable_blocks']=protocol['spec']['num_trainable_blocks']
    student=SALADModel(config,pretrained_backbone=False,backbone_repo=Path(c['backbone_repo']))
    student.load_state_dict(state,strict=True);del state
    student=student.to(c['device']).eval()
    teacher=FrozenSALADTeacher(Path(c['checkpoint']),config,device=c['device'],backbone_repo=Path(c['backbone_repo']))
    calibration=json.loads((out/'calibration.json').read_text())
    if calibration['provenance']['teacher']['checkpoint_sha256']!=teacher.provenance['checkpoint_sha256']:
        raise ValueError('Diagnostic calibration uses a different teacher')
    geometry={}
    from workflow.local_distillation import LocalDistillationSupervisor
    for path in plan['geometry_hashes']:
        saved=json.loads(Path(path).read_text())
        geometry[LocalDistillationSupervisor._geometry_key(saved['identity'])]=saved['evidence']
    pairs=[]
    for row in plan['probe_pairs']:
        with Image.open(row['source_path']) as im: source=im.convert('RGB')
        with Image.open(row['output_path']) as im: generated=im.convert('RGB')
        masks={}
        for name in ('edit_mask','occlusion_mask','destruction_mask'):
            if row.get(name+'_path'):
                import numpy as np
                path=Path(row[name+'_path'])
                if row.get(name+'_sha256') and ccr.digest(path)!=row[name+'_sha256']: raise ValueError('Diagnostic mask changed')
                with Image.open(path) as im:
                    if im.size!=generated.size: raise ValueError('Mask does not use native generated coordinates')
                    masks[name]=torch.from_numpy(np.asarray(im.convert('L'),dtype=np.float32).copy()/255.)
        pairs.append(dict(source_image=source,generated_image=generated,source_tensor=preprocess_teacher_image(source,(224,224))[None],
            generated_tensor=preprocess_teacher_image(generated,(224,224))[None],source_path=row['source_path'],generated_path=row['output_path'],
            quality=row,condition=row['condition'],masks=masks))
    before={n:p._version for n,p in student.named_parameters()}
    teacher_before={n:p._version for n,p in teacher.named_parameters()}
    original=local_consistency_probe(student,teacher,pairs,calibration,device=c['device'],repeats=3,geometry=geometry)
    with ExitStack() as stack:
        for module in student.aggregator.modules():
            if isinstance(module,torch.nn.modules.dropout._DropoutNd): stack.enter_context(patch.object(module,'forward',lambda x:x))
        dropout_off=local_consistency_probe(student,teacher,pairs,calibration,device=c['device'],repeats=3,geometry=geometry)
    if (before!={n:p._version for n,p in student.named_parameters()}
            or teacher_before!={n:p._version for n,p in teacher.named_parameters()}
            or any(p.grad is not None or p.requires_grad for p in teacher.parameters())):
        raise RuntimeError('Read-only diagnostic mutated parameters')
    ccr.write_json(out/'diagnostics'/ (ccr.digest(checkpoint)+'.json'),dict(checkpoint=str(checkpoint),sha256=ccr.digest(checkpoint),
        original=original,salad_dropout_off=dropout_off,teacher_frozen=True,optimizer_updates=0,
        interpretation='source_self separates noise/precision; teacher_source adds drift/cross-domain; same-coordinate vs matched exposes correspondence dependence; no floor subtraction'))


def summarize(plan,out):
    from workflow.ccr4_diagnostics import read_committed
    parent=plan['parent'];c=parent['config']
    report=dict(status='awaiting_stage3_freeze' if not (Path(plan['config']['stage3_dir'])/'formal_protocol.json').exists() else 'prepared_not_trained',
        dependencies=dict(stage3_frozen=(Path(plan['config']['stage3_dir'])/'formal_protocol.json').exists(),calibration=(out/'calibration.json').exists()),
        missing_natural_weather_validation=parent['missing_natural_weather_validation'],B0_gate=b0_gate(plan,out),
        runs=[],paired_statistics=[],seed_distributions=[],support_comparison=[],limitations=plan['limitations'])
    report['pilot_runs']=[]
    for stage in ('controls','global-screen','smoke'):
        root=out/stage/str(c['screen_seed'])
        if not root.exists(): continue
        baseline_result=result(plan,root/'A')
        for directory in sorted(root.iterdir()):
            path=directory/'audit.json'
            if not path.exists(): continue
            audit=json.loads(path.read_text())
            if audit['checkpoint_sha256']!=ccr.digest(directory/'checkpoint.pt'): raise ValueError('Stale pilot audit')
            values=result(plan,directory)
            row=dict(stage=stage,method=directory.name,seed=c['screen_seed'],audit=audit,
                     validation={k:v['recall'] for k,v in values.items()} if values else None)
            if values and baseline_result:
                row['exploratory_delta_A_pp']=100*(values['svox_val']['recall']['R@1']-baseline_result['svox_val']['recall']['R@1'])
            report['pilot_runs'].append(row)
    controls={r['method']:r['audit'] for r in report['pilot_runs'] if r['stage']=='controls'}
    report['rng_controls']={method:dict(parameter_equal_A=controls[method]['parameter_sha256']==controls['A']['parameter_sha256'])
        for method in ('B0-zero','B0-zero-isolated') if method in controls and 'A' in controls}
    lookup={}
    for seed in c['confirmation_seeds']:
        for method in FORMAL:
            directory=out/'formal'/str(seed)/method
            values=result(plan,directory)
            path=directory/'audit.json'
            if values is None or not path.exists(): continue
            audit=json.loads(path.read_text())
            if audit['checkpoint_sha256']!=ccr.digest(directory/'checkpoint.pt'): raise ValueError('Stale final audit')
            lookup[seed,method]=values
            report['runs'].append(dict(seed=seed,method=method,recall={k:v['recall'] for k,v in values.items()},audit=audit))
    initial=am.read_result(parent,out/'initial',Path(c['checkpoint']))
    for (seed,method),values in lookup.items():
        refs=[(ref,initial if ref=='initial' else lookup.get((seed,ref))) for ref in REFERENCES[method]]
        refs=[(ref,values) for ref,values in refs if values is not None]
        for reference,base in refs:
            for domain,trial in values.items():
                groups=(parent.get('grouping') or {}).get('domains',{}).get(domain)
                for k in (1,5,10):
                    report['paired_statistics'].append(dict(seed=seed,method=method,reference=reference,domain=domain,recall_k=k,
                        **paired(base[domain],trial,k,groups,c['statistics']['bootstrap_samples'],c['statistics']['bootstrap_seed'])))
    family=len(c['confirmation_seeds'])*len(parent['evaluations'])*3*sum(map(len,REFERENCES.values()))
    for row,pvalue in zip(report['paired_statistics'],holm([r['p_exact'] for r in report['paired_statistics']],family)):
        row['p_holm']=pvalue
    report['holm_family_size']=family
    groups={}
    for row in report['paired_statistics']:
        key=(row['method'],row['reference'],row['domain'],row['recall_k']);groups.setdefault(key,[]).append(row)
    for key,rows in groups.items():
        values=[r['delta_pp'] for r in rows];complete=len(rows)==len(c['confirmation_seeds'])
        means=sorted(statistics.mean(sample) for sample in itertools.product(values,repeat=len(values))) if complete and len(values)<=5 else []
        report['seed_distributions'].append(dict(method=key[0],reference=key[1],domain=key[2],recall_k=key[3],deltas={str(r['seed']):r['delta_pp'] for r in rows},
            mean_pp=statistics.mean(values),std_pp=statistics.stdev(values) if len(values)>1 else None,complete=complete,
            seed_bootstrap_95_pp=[means[int(.025*(len(means)-1))],means[int(.975*(len(means)-1))]] if means else None,
            interval_scope='empirical seed bootstrap conditional on this validation set; three seeds give weak interval resolution'))
    for seed in c['confirmation_seeds']:
        tables=[]
        for method in ('B4','B5'):
            directory=out/'formal'/str(seed)/method;path=directory/'audit.json'
            if not path.exists(): break
            audit=json.loads(path.read_text());table={}
            for step in read_committed(directory/'attribution.jsonl',audit['optimizer_updates']):
                for pair in step['local_pairs']: table[step['step'],pair['source_path'],pair['generated_path']]=pair['support_sha256']
            tables.append(table)
        if len(tables)==2:
            report['support_comparison'].append(dict(seed=seed,pairs=len(tables[0]),same_keys=tables[0].keys()==tables[1].keys(),identical=tables[0]==tables[1]))
    if report['runs']: report['status']='formal_partial' if len(lookup)<len(FORMAL)*len(c['confirmation_seeds']) else 'formal_complete_descriptive'
    if any(not r['identical'] for r in report['support_comparison']): report['status']='invalid_B4_B5_support_mismatch'
    report['module_assessment']=[]
    for module,method,reference in (('Global','B0','A'),('Feature','B-Feature','B0'),('OT','B-OT','B0'),
                                     ('OT_added_to_Feature','B4','B-Feature'),('q','B5','B4'),('OT_train_minus_dropout_off','B-OT-train','B-OT')):
        rows=[r for r in report['paired_statistics'] if (r['method'],r['reference'],r['domain'],r['recall_k'])==(method,reference,'svox_val',1)]
        status='pending'
        if len(rows)==len(c['confirmation_seeds']):
            status='consistent_positive_trend' if min(r['delta_pp'] for r in rows)>0 else 'no_consistent_positive_increment'
            if module=='q' and any(not r['identical'] for r in report['support_comparison']): status='invalid_support_comparison'
        report['module_assessment'].append(dict(module=module,contrast=method+' vs '+reference,status=status,
            holm_significant_seeds=sum(r['p_holm']<c['statistics']['alpha'] for r in rows),
            scope='descriptive real validation evidence; no automatic natural-weather claim'))
    ccr.write_json(out/'report.json',report)
    lines=['# CCR stage 4 attribution','',f"Status: {report['status']}",'',f"B0 gate: {report['B0_gate']}",'',
        'Missing real weather validation: '+', '.join(report['missing_natural_weather_validation']),'',
        '| Seed | Method | Val R1 | Aux queries | Actual aux fraction |','|---|---|---:|---:|---:|']
    for row in report['runs']:
        lines.append(f"| {row['seed']} | {row['method']} | {row['recall']['svox_val']['R@1']:.4%} | {row['audit']['auxiliary_queries']} | {row['audit']['auxiliary_fraction']:.3%} |")
    lines+=['','| Module | Contrast | Evidence status |','|---|---|---|']
    lines += [f"| {r['module']} | {r['contrast']} | {r['status']} |" for r in report['module_assessment']]
    lines+=['','Inspect paired_statistics and seed_distributions; completion alone is not evidence of module benefit.',
        'B4/B5 support_comparison must be identical before interpreting confidence weighting.',
        'Gradients/layer Gram matrices, per-weather auxiliary losses, confidence bands and spatial support: each run attribution.jsonl.',
        'Unannotated GSV is not assigned synthetic weather labels. No test results are read.', '',*plan['limitations']]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ccr.ROOT/'configs/ccr_stage4.json')
    p.add_argument('--phase',choices=('prepare','freeze','calibrate','controls','global-screen','freeze-global','confirm-b0','local','smoke','diagnose','report'),default='prepare')
    p.add_argument('--method');p.add_argument('--seed',type=int);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--worker',choices=('calibrate','audit','diagnose'),help=argparse.SUPPRESS);p.add_argument('--run-dir',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args(argv);c=load_config(args.config.resolve());out=Path(c['output_dir']);out.mkdir(parents=True,exist_ok=True)
    if args.worker:
        plan=json.loads((out/'plan.json').read_text())
        if args.worker=='calibrate': calibrate(plan,out)
        elif args.worker=='audit': audit(plan,args.run_dir)
        else: diagnose(plan,out,args.checkpoint or Path(plan['parent']['config']['checkpoint']))
        return
    with (out/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'plan.json').exists():
            plan=json.loads((out/'plan.json').read_text())
            if c!=plan['config']: raise ValueError('Frozen stage 4 config changed')
            if args.phase!='report': validate(plan,out)
        else: plan=prepare(c,out)
        am.immutable(out/'config.json',ccr.encoded(c))
        if args.phase=='freeze': freeze(plan,out)
        elif args.phase=='freeze-global': select_global(plan,out)
        elif args.execute and args.phase in ('calibrate','diagnose'):
            cmd=[sys.executable,str(Path(__file__).resolve()),'--config',str(out/'config.json'),'--worker',args.phase]
            if args.checkpoint: cmd+=['--checkpoint',str(args.checkpoint.resolve())]
            ccr.run_child(cmd,out/(args.phase+'.log'),plan['parent']['config']['torch_threads'],limits=plan['parent']['config']['resource_limits'])
        elif args.execute and args.phase in ('controls','global-screen','confirm-b0','local','smoke'):
            protocol=freeze(plan,out);parent=plan['parent'];pc=parent['config']
            selection=None
            if args.phase in ('confirm-b0','local'): selection=select_global(plan,out)
            if args.phase=='local' and not b0_gate(plan,out)['passed']: raise ValueError('B0 gate failed/incomplete: local formal training forbidden')
            methods=list(CONTROLS) if args.phase=='controls' else ['A']+[weight_name(w) for w in c['global_weights']] if args.phase=='global-screen' else ['A','B0'] if args.phase=='confirm-b0' else list(FORMAL[2:]) if args.phase=='local' else ['A','B0-zero-isolated','B0-isolated','B-Feature','B-OT','B4','B5']
            if args.method:
                if args.method not in methods: raise ValueError('Method not in this phase')
                methods=[args.method]
            seeds=pc['confirmation_seeds'] if args.phase in ('confirm-b0','local') else [pc['screen_seed']]
            if args.seed is not None:
                if args.seed not in seeds: raise ValueError('Seed not in declared phase')
                seeds=[args.seed]
            if args.phase!='smoke': ccr.evaluate_checkpoint(am.ep_plan(parent),Path(pc['checkpoint']),out/'initial')
            for seed in seeds:
                for method in methods:
                    execute_one(plan,out,protocol,method,seed,'formal' if args.phase in ('confirm-b0','local') else args.phase,
                        protocol['epochs'] if args.phase in ('confirm-b0','local') else 1,
                        .1 if args.phase=='smoke' else selection)
        report=summarize(plan,out)
        print(f"Status: {report['status']}; report={out/'report.md'}")
        if not args.execute: print('No GPU or training launched.')


if __name__=='__main__': main()
