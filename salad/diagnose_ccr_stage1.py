#!/usr/bin/env python3
"""Fixed-input, zero-update gradient/mining/local consistency diagnostics."""
from __future__ import annotations
import argparse
import copy
import csv
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time

from run_ccr_feasibility import digest, run_child, write_json

ROOT=Path(__file__).resolve().parent


def state_digest(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def distribution(values):
    import torch
    if not values: return dict(n=0,mean=None,p05=None,median=None,p95=None,min=None,max=None)
    x=torch.tensor(values,dtype=torch.float64)
    if not torch.isfinite(x).all(): raise RuntimeError('Nonfinite diagnostic summary')
    return dict(n=len(values),mean=float(x.mean()),p05=float(x.quantile(.05)),median=float(x.quantile(.5)),
                p95=float(x.quantile(.95)),min=float(x.min()),max=float(x.max()))


def fixed_inputs(plan,args):
    import torch
    from PIL import Image
    from torch.utils.data import default_collate
    from workflow.training_data import MixedGSVCitiesDataset,SyntheticAuxSampler
    from workflow.reliability_cache import load_teacher_image
    c=plan['config'];rng=random.Random(args.seed)
    random.seed(args.seed)
    full=MixedGSVCitiesDataset(Path(c['real_data']),Path(plan['train_manifest']),cities=c['cities'],
        images_per_place=4,min_images_per_place=4,synthetic_fraction=0,image_size=(224,224),
        augment=args.frozen_augmentation,real_sampling='coverage_pk',seed=args.seed,return_sampling_metadata=True)
    subset=copy.copy(full);subset.places=[copy.copy(p) for p in full.places if p.synthetic_paths]
    for label,p in enumerate(subset.places): p.label=label
    sampler=SyntheticAuxSampler(subset,.5,args.seed)
    batches=[];metadata=[]
    def record(dataset,indices,panel,index):
        images,labels,_,sampling=default_collate([dataset[(0,i)] for i in indices])
        images=images.flatten(0,1);labels=labels.flatten()
        # Natural auxiliary uses the P8 prefix without forced source inclusion.
        aux=(sampler.sample({k:v[:8] for k,v in sampling.items()}) if panel=='training_subset' else None)
        paths=[str(dataset.places[int(p)].real_paths[int(i)]) for p,ii in zip(sampling['place_index'],sampling['real_indices']) for i in ii]
        name=f'{panel}_{index}'
        batch=dict(id=name,panel=panel,images=images,labels=labels,aux=aux)
        batches.append(batch)
        metadata.append(dict(id=name,panel=panel,places=len(indices),real_paths=paths,
            tensor_sha256=hashlib.sha256(images.numpy().tobytes()).hexdigest(),
            synthetic_paths=aux['synthetic_paths'] if aux is not None else [],
            source_paths=aux['source_paths'] if aux is not None else []))
    order=rng.sample(range(len(subset.places)),32*args.batches)
    for i in range(args.batches): record(subset,order[i*32:(i+1)*32],'training_subset',i)
    excluded={tuple(k) for k in plan['heldout_places']}
    excluded.update((row['city'],row['place_id']) for row in plan.get('calibration_pairs',[]))
    candidates=[i for i,p in enumerate(full.places) if (p.city,p.place_id) not in excluded]
    order=rng.sample(candidates,32*args.full_batches)
    for i in range(args.full_batches): record(full,order[i*32:(i+1)*32],'full_real',i)
    rows=[json.loads(line) for line in Path(plan['train_manifest']).read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r:hashlib.sha256(f"{args.seed}/{r['output_path']}".encode()).hexdigest())
    selected=[];used=set()
    for weather in ('night','rain','snow','fog'):
        chosen=0
        for row in rows:
            key=row['city'],row['place_id']
            if row['condition'].lower()==weather and key not in used:
                selected.append(row);used.add(key);chosen+=1
                if chosen==2*args.controlled_batches: break
        if chosen!=2*args.controlled_batches: raise ValueError('Insufficient controlled weather places')
    by_weather={w:[r for r in selected if r['condition'].lower()==w] for w in ('night','rain','snow','fog')}
    place_map={(p.city,p.place_id):p for p in subset.places};pairs=[]
    for batch_index in range(args.controlled_batches):
        chosen=[by_weather[w][batch_index*2+j] for w in by_weather for j in range(2)]
        images=[];labels=[];source_indices=[]
        for i,row in enumerate(chosen):
            source=Path(row['source_path']);p=place_map[row['city'],row['place_id']]
            alternatives=rng.sample([path for path in p.real_paths if path!=source],3)
            images.extend(load_teacher_image(path,(224,224)) for path in [source,*alternatives])
            labels.extend([i]*4);source_indices.append(i*4)
            with Image.open(source) as im: source_image=im.convert('RGB')
            with Image.open(row['output_path']) as im: generated_image=im.convert('RGB')
            pairs.append(dict(source_tensor=images[-4][None],
                generated_tensor=load_teacher_image(Path(row['output_path']),(224,224))[None],
                source_image=source_image,generated_image=generated_image,source_path=str(source),
                generated_path=row['output_path'],condition=row['condition'],quality=row))
        aux=dict(synthetic_images=torch.cat([p['generated_tensor'] for p in pairs[-8:]]),
            synthetic_labels=torch.arange(8),source_indices=torch.tensor(source_indices),
            synthetic_paths=[r['output_path'] for r in chosen],source_paths=[r['source_path'] for r in chosen],
            condition=[r['condition'] for r in chosen],pair_valid=torch.ones(8,dtype=torch.bool))
        images=torch.stack(images);name=f'controlled_source_included_{batch_index}'
        batches.append(dict(id=name,panel='controlled_source_included',images=images,labels=torch.tensor(labels),aux=aux))
        metadata.append(dict(id=name,panel='controlled_source_included',places=8,
            source_included=True,trigger_probability=1.,synthetic_paths=aux['synthetic_paths'],source_paths=aux['source_paths'],
            tensor_sha256=hashlib.sha256(images.numpy().tobytes()).hexdigest()))
    return subset,batches,pairs[:args.dropout_pairs],metadata


def common_anchor_statistics(result,count=32):
    rows=result['per_anchor'][:count]
    return dict(anchors=len(rows),active_anchor_ratio=sum(r['ms_active'] for r in rows)/len(rows),
        both_sides_active_anchor_ratio=sum(r['both_sides_active'] for r in rows)/len(rows),
        mined_positive_pairs=sum(r['mined_positive'] for r in rows),
        mined_negative_pairs=sum(r['mined_negative'] for r in rows))


def branch_gradients(model,batch,prepared,totals,named,device,seed,precision,local_mode):
    import torch
    from workflow.metric_loss import multi_similarity_loss
    from workflow.synthetic_losses import asymmetric_multi_positive_infonce,local_distillation_losses
    from workflow.diagnostic_gradients import collect_loss_gradients
    enabled=precision=='bf16';model.train();torch.manual_seed(seed)
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=enabled): descriptors=model(batch['images'][:32].to(device))
    real=multi_similarity_loss(descriptors,batch['labels'][:32].to(device));values=dict(real=float(real.detach()))
    references=descriptors.detach();gradients=collect_loss_gradients({'real':real},named)
    del descriptors,real
    aux=batch['aux'];n=len(named)
    if len(aux['synthetic_images']):
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=enabled): queries=model(aux['synthetic_images'].to(device))
        syn=asymmetric_multi_positive_infonce(queries,aux['synthetic_labels'].to(device),references,
            batch['labels'][:32].to(device),temperature=.1,pair_valid=aux['pair_valid'].to(device))
        gradients.update(collect_loss_gradients({'syn':syn},named));values['syn']=float(syn.detach());del queries,syn
    else: gradients['syn']=(None,)*n;values['syn']=0.
    del references
    for key in ('feature','score','ot'): gradients[key]=[None]*n;values[key]=0.
    model.train(local_mode=='train')
    for pair in prepared:
        if not any(pair['masses'][key]>0 for key in ('feature','score','ot')): continue
        target={key:value.to(device) for key,value in pair['teacher'].items()}
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=enabled): _,student=model(pair['image'].to(device),return_aux=True)
        losses=local_distillation_losses(student,target,pair['index'].to(device),pair['q'].to(device))
        scaled={key:losses[key]*(losses[key+'_mass']+1e-8)/(totals[key]+1e-8) for key in ('feature','score','ot')}
        part=collect_loss_gradients(scaled,named)
        for key in scaled:
            values[key]+=float(scaled[key].detach())
            for i,g in enumerate(part[key]):
                if g is not None: gradients[key][i]=g.clone() if gradients[key][i] is None else gradients[key][i]+g
        del target,student,losses,scaled,part
    return gradients,values


def combined_from_gram(group,weights):
    import math
    gram=group['gram'];names=list(gram)
    norm=math.sqrt(max(0.,math.fsum(weights.get(a,0)*weights.get(b,0)*gram[a][b] for a in names for b in names)))
    real=math.sqrt(max(0.,gram['real']['real']))
    dot=math.fsum(weights.get(a,0)*gram[a]['real'] for a in names)
    cosine=max(-1.,min(1.,dot/norm/real)) if norm>0 and real>0 else None
    return dict(norm=norm,would_clip=norm>1.,clip_scale=min(1.,1./(norm+1e-6)),vs_real_cosine=cosine,
        vs_real_angle_degrees=math.degrees(math.acos(cosine)) if cosine is not None else None)


def uniform_targets(prepared):
    from workflow.synthetic_losses import local_target_weights
    result=[];totals=dict(feature=0.,score=0.,ot=0.)
    for pair in prepared:
        p={**pair,'q':(pair['q']>0).float()}
        masses=local_target_weights(p['teacher'],p['index'],p['q'])
        p['masses']={k:float(v.sum()) for k,v in masses.items()}
        for k,value in p['masses'].items(): totals[k]+=value
        result.append(p)
    return result,totals


def run_worker(args,out):
    import torch
    from train_salad import parse_args as training_args
    from workflow.model import SALADModel,FrozenSALADTeacher,read_checkpoint,checkpoint_state_and_config
    from workflow.local_distillation import LocalDistillationSupervisor
    from workflow.diagnostic_local import local_consistency_probe
    from workflow.diagnostic_gradients import mined_pair_statistics,gradient_vector_summary
    from workflow.resource_limits import configure_cuda_budget
    plan=json.loads((args.experiment_dir/'plan.json').read_text());c=plan['config']
    torch.set_num_threads(4);device=torch.device('cuda:0');configure_cuda_budget(device)
    checkpoints={'initial':Path(c['checkpoint']),**{m:args.experiment_dir/'runs'/m/'42/checkpoint.pt' for m in ('A','B0','B4','B5')}}
    hashes={m:digest(path) for m,path in checkpoints.items()}
    dataset,batches,pairs,metadata=fixed_inputs(plan,args);write_json(out/'fixed_batches.json',dict(seed=args.seed,batches=metadata))
    state,config=checkpoint_state_and_config(read_checkpoint(checkpoints['initial']));del state
    config['backbone_config']['num_trainable_blocks']=1
    teacher=FrozenSALADTeacher(checkpoints['initial'],config,device=device,backbone_repo=Path(c['backbone_repo']))
    teacher_before=state_digest(teacher)
    targs=training_args(['--real-data',c['real_data'],'--synthetic-manifest',plan['train_manifest'],
        '--output-dir',str(out),'--synthetic-aux','--teacher-checkpoint',c['teacher_checkpoint'],
        '--teacher-batch-size','2','--pair-calibration',str(args.experiment_dir/'calibration.json'),
        '--pair-confidence-mode','verified','--pair-geometry-cache',c['pair_geometry_cache']])
    supervisor=LocalDistillationSupervisor(teacher,dataset,targs)
    targets={}
    for batch in batches:
        if batch['aux'] is not None:
            prepared,totals,stats=supervisor.prepare(batch['aux']);targets[batch['id']]=(prepared,totals,stats)
    for pair in pairs: pair['masks']=supervisor._masks(pair['quality'] | {'manifest_line':1},pair['generated_image'])
    results={};started=time.perf_counter()
    for label,path in checkpoints.items():
        print(f'Diagnosing {label}: loading fixed checkpoint',flush=True)
        state,_=checkpoint_state_and_config(read_checkpoint(path))
        model=SALADModel(copy.deepcopy(config),pretrained_backbone=False,backbone_repo=Path(c['backbone_repo'])).to(device)
        model.load_state_dict(state,strict=True);del state;gc.collect()
        before=state_digest(model);versions={n:p._version for n,p in model.named_parameters()}
        named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        mining=[];gradient_rows=[]
        for batch_number,batch in enumerate(batches):
            if batch['panel']!='controlled_source_included':
                for mode in ('eval_fp32','train_bf16'):
                    with torch.random.fork_rng(devices=[0]),torch.no_grad():
                        torch.manual_seed(args.seed+batch_number);model.train(mode=='train_bf16')
                        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=mode=='train_bf16'):
                            desc=torch.cat([model(chunk.to(device)).float().cpu() for chunk in batch['images'].split(16)])
                    for p in (8,16,32):
                        row=mined_pair_statistics(desc[:p*4],batch['labels'][:p*4])
                        row.update(batch=batch['id'],panel=batch['panel'],mode=mode,P=p,
                                   common_8_place_anchors=common_anchor_statistics(row))
                        mining.append(row)
            if batch['aux'] is None: continue
            prepared,totals,stats=targets[batch['id']]
            confidence_mode='verified' if label=='B5' else 'uniform'
            if confidence_mode=='uniform': prepared,totals=uniform_targets(prepared)
            # Natural panels use original BF16 training; controlled additionally
            # removes local Dropout in FP32 to isolate its effect on gradients.
            conditions=[('bf16','train')]
            if batch['panel']=='controlled_source_included': conditions += [('32','train'),('32','eval')]
            for precision,local_mode in conditions:
                with torch.random.fork_rng(devices=[0]):
                    grads,values=branch_gradients(model,batch,prepared,totals,named,device,
                        args.seed+batch_number,precision,local_mode)
                weights=dict(real=1.,syn=.1,feature=.1,score=0.,ot=.1)
                summary=gradient_vector_summary(grads,named,weights)
                alternatives=dict(real_only=dict(real=1.),global_001=dict(real=1.,syn=.01),
                    global_003=dict(real=1.,syn=.03),global_010=dict(real=1.,syn=.1),
                    feature_only=dict(real=1.,syn=.1,feature=.1),feature_ot=dict(real=1.,syn=.1,feature=.1,ot=.1))
                variants={name:combined_from_gram(summary['groups']['all'],w) for name,w in alternatives.items()}
                gradient_rows.append(dict(batch=batch['id'],panel=batch['panel'],precision=precision,local_mode=local_mode,
                    confidence_mode=confidence_mode,combined_variants=variants,
                    losses=values,synthetic_queries=len(batch['aux']['synthetic_images']),target_statistics=stats,**summary))
                del grads,summary
                if any(p.grad is not None for p in model.parameters()): raise RuntimeError('Diagnostic populated parameter .grad')
            print(f'{label}: {batch["id"]} complete',flush=True)
        local=local_consistency_probe(model,teacher,pairs,supervisor.calibration,device=device,
            repeats=args.repeats,seed=args.seed,geometry=supervisor.geometry)
        after=state_digest(model)
        if after!=before or any(p._version!=versions[n] for n,p in model.named_parameters()):
            raise RuntimeError('Diagnostic changed in-memory model state')
        results[label]=dict(mining=mining,gradients=gradient_rows,local=local,
            dropout_inventory={n:dict(type=type(m).__name__,p=getattr(m,'p',None)) for n,m in model.named_modules()
                               if 'drop' in type(m).__name__.lower()},
            state_sha256_before=before,state_sha256_after=after,parameter_versions_unchanged=True)
        write_json(out/f'{label}.json',results[label]);print(f'{label}: state unchanged, all probes complete',flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    if any(p.grad is not None for p in teacher.parameters()): raise RuntimeError('Teacher received gradients')
    if state_digest(teacher)!=teacher_before: raise RuntimeError('Teacher state changed')
    after_hashes={m:digest(path) for m,path in checkpoints.items()}
    if after_hashes!=hashes: raise RuntimeError('Checkpoint file changed during diagnosis')
    protocol=dict(optimizer_created=False,optimizer_steps=0,parameter_state_unchanged=True,
        checkpoint_sha256_before=hashes,checkpoint_sha256_after=after_hashes,trainable_backbone_blocks=1,
        batch_seed=args.seed,K=4,natural_batches=args.batches,full_real_batches=args.full_batches,
        controlled_batches=args.controlled_batches,dropout_pairs=len(pairs),dropout_repeats=args.repeats,
        input_transform=('fixed_once_original_flip_color_augmentation_after_resize_224' if args.frozen_augmentation
                         else 'fixed_full_frame_resize_224_rgb_imagenet_no_random_augmentation'),
        augmentation_applies_to='natural/full_real main and natural synthetic inputs; local/controlled remain fixed Resize',
        source_sha256={str(path):digest(path) for path in [Path(__file__),ROOT/'workflow/diagnostic_gradients.py',ROOT/'workflow/diagnostic_local.py']},
        teacher_state_unchanged=True,
        limitations=['Small fixed samples, not full-epoch gradient/clip estimates.',
            'Mined counts use directed anchor-pairs and nested fixed descriptors.',
            'Controlled panel forces exact-source inclusion and p_exp=1; not natural exposure.',
            'Gradient totals are hypothetical SGD gradients, not AdamW update attribution.',
            'No evaluation, retrieval, training updates, checkpoint or feature-cache writes.'],
        seconds=time.perf_counter()-started,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    write_json(out/'protocol.json',protocol);summarize(results,out)


def summarize(results,out):
    mining_rows=[];gradient_rows=[];local_rows=[]
    for checkpoint,result in results.items():
        for r in result['mining']:
            mining_rows.append(dict(checkpoint=checkpoint,**{k:r[k] for k in ('batch','panel','mode','P','loss',
                'positive_pairs','negative_pairs','mined_positive_pairs','mined_negative_pairs',
                'positive_retention_ratio','negative_retention_ratio','active_anchor_ratio','both_sides_active_anchor_ratio')},
                **{'common_'+k:v for k,v in r['common_8_place_anchors'].items()}))
        for r in result['gradients']:
            base=dict(checkpoint=checkpoint,**{k:r[k] for k in ('batch','panel','precision','local_mode','confidence_mode','synthetic_queries')})
            for variant,values in r['combined_variants'].items():
                for term,value in values.items():
                    gradient_rows.append(dict(base,group='all',metric='variant',term=variant+'.'+term,value=value))
            for group,g in r['groups'].items():
                for loss,value in g['losses'].items():
                    gradient_rows.append(dict(base,group=group,metric='norm',term=loss,value=value['raw_norm'],
                        weighted_value=value['weighted_norm']))
                for terms,value in g['cosines'].items():
                    gradient_rows.append(dict(base,group=group,metric='cosine',term=terms,value=value['value'],
                        angle_degrees=value['angle_degrees']))
                for term,value in g['combined'].items():
                    if isinstance(value,(int,float,bool)):
                        gradient_rows.append(dict(base,group=group,metric='combined',term=term,value=value))
        for row in result['local']['rows']: local_rows.append(dict(checkpoint=checkpoint,**row))
    for name,rows in [('mined_pairs',mining_rows),('gradient_metrics',gradient_rows),('local_consistency',local_rows)]:
        fields=sorted({k for r in rows for k in r})
        with (out/(name+'.csv')).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    aggregate={}
    for r in gradient_rows:
        key='|'.join(str(r.get(k,'')) for k in ('checkpoint','panel','precision','local_mode','confidence_mode','group','metric','term'))
        if r.get('value') is not None: aggregate.setdefault(key,[]).append(r['value'])
    write_json(out/'gradient_distributions.json',{k:distribution(v) for k,v in aggregate.items()})
    lines=['# Stage 1 零更新诊断','',
        '所有 checkpoint 与内存参数均未修改；没有创建 optimizer，没有训练或检索评估。',
        '自然/受控面板分开统计；输入变换见 protocol.json。所有统计仅代表选定 batch。','',
        '| Checkpoint | 面板 | P | 有效 anchor 均值（eval FP32） | MS Loss 均值 |',
        '|---|---|---:|---:|---:|']
    for checkpoint,result in results.items():
        for panel in ('training_subset','full_real'):
            for p in (8,16,32):
                rows=[r for r in result['mining'] if r['panel']==panel and r['P']==p and r['mode']=='eval_fp32']
                if rows: lines.append(f"| {checkpoint} | {panel} | {p} | {sum(r['active_anchor_ratio'] for r in rows)/len(rows):.4f} | {sum(r['loss'] for r in rows)/len(rows):.6f} |")
    lines+=['','梯度的分组范数/夹角及分布：gradient_metrics.csv、gradient_distributions.json。',
            'Feature/Score/OT 的同图、跨图、精度、Dropout 对照：local_consistency.csv 与各 checkpoint JSON 的 local.summaries。',
            '精确样本清单：fixed_batches.json；参数与文件未修改证明：protocol.json。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-dir',type=Path,default=ROOT/'outputs/ccr_accepted_13710_svox_categories')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--batches',type=int,default=8);p.add_argument('--full-batches',type=int,default=4)
    p.add_argument('--controlled-batches',type=int,default=2);p.add_argument('--dropout-pairs',type=int,default=8)
    p.add_argument('--repeats',type=int,default=4);p.add_argument('--seed',type=int,default=3042)
    p.add_argument('--frozen-augmentation',action='store_true',
                   help='Generate main/natural-synthetic flip/color augmentation once and reuse identical tensors across checkpoints')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args();args.experiment_dir=args.experiment_dir.resolve();out=args.output_dir.resolve()
    if min(args.batches,args.full_batches,args.controlled_batches,args.dropout_pairs,args.repeats)<1:
        p.error('All sample/repeat counts must be positive')
    if args.worker: run_worker(args,out);return
    out.mkdir(parents=True,exist_ok=False)
    plan=json.loads((args.experiment_dir/'plan.json').read_text());c=plan['config']
    run_child([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker'],out/'diagnostic.log',4,
              limits=c['resource_limits'])
    print(f'Zero-update diagnostic report: {out / "report.md"}')


if __name__=='__main__': main()
