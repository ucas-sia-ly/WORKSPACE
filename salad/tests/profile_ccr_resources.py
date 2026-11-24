"""Explicit bounded GPU probe; no full training or saved weights.

Measures FP32 inference, exact retrieval, and two disposable BF16 B4-like steps.
If SVOX val is configured, also measures loading/extracting 512 real images.
This is not evidence of retrieval quality.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_ccr_feasibility import load_config, run_child, write_json


def probe(c, output):
    import torch
    from workflow.model import SALADModel, FrozenSALADTeacher, checkpoint_state_and_config, read_checkpoint
    from workflow.resource_limits import configure_cuda_budget
    from workflow.metric_loss import multi_similarity_loss
    from workflow.synthetic_losses import asymmetric_multi_positive_infonce, local_distillation_losses
    from workflow.evaluation import exact_retrieval, EvaluationImage, extract_descriptors

    torch.set_num_threads(c['torch_threads']);torch.manual_seed(42)
    device=torch.device(c['device']); budget=configure_cuda_budget(device)
    state, config=checkpoint_state_and_config(read_checkpoint(Path(c['checkpoint'])))
    config['backbone_config']['num_trainable_blocks']=c['num_trainable_blocks']
    model=SALADModel(config,pretrained_backbone=False,backbone_repo=Path(c['backbone_repo'])).to(device)
    model.load_state_dict(state);del state;gc.collect()
    images=torch.randn(max(256,c['eval_batch_size']),3,224,224,device=device)
    records=[]

    def measure(name, operation, repetitions, images_per_call=None):
        operation();torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started=time.perf_counter()
        for _ in range(repetitions): operation()
        torch.cuda.synchronize(device)
        seconds=(time.perf_counter()-started)/repetitions
        records.append(dict(name=name,seconds_per_call=seconds,
            images_per_second=images_per_call/seconds if images_per_call else None,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3))

    model.eval()
    with torch.inference_mode():
        for size in sorted({8,64,128,256,c['eval_batch_size']}):
            measure(f'eval_fp32_batch_{size}',lambda:model(images[:size]),3,size)
        references=torch.randn(4096,8448);queries=torch.randn(512,8448)
        positives=[[i%4096] for i in range(len(queries))]
        outputs={}
        def retrieve(name, chunk, cache):
            outputs[name]=exact_retrieval(references,queries,positives,device=str(device),
                query_chunk_size=chunk,reference_chunk_size=4096,reference_cache_bytes=cache)
        measure('retrieval_uncached_q32',lambda:retrieve('old',32,0),1)
        measure('retrieval_cached_q128',lambda:retrieve('new',128,2048*1024**2),1)
        ranks_agree=all(a['rank']==b['rank'] and a['predicted_index']==b['predicted_index']
                        for a,b in zip(outputs['old'],outputs['new']))
        if not ranks_agree: raise RuntimeError('Probe exact retrieval ranks changed')
        svox=next((e for e in c['external_validations'] if e['dataset']=='SVOX'),None)
        if svox:
            files=sorted((Path(svox['root'])/'images/val/gallery').rglob('*.jpg'))[:512]
            if not files: raise ValueError('No SVOX val gallery images for bounded I/O probe')
            real_images=[EvaluationImage(str(i),path) for i,path in enumerate(files)]
            for size in (64,128,256):
                measure(f'real_image_extraction_512_batch_{size}',lambda:extract_descriptors(
                    model,real_images,(224,224),str(device),size,c['num_workers'],c.get('prefetch_factor',1)),1,len(files))
    del references,queries,outputs
    teacher=FrozenSALADTeacher(Path(c['teacher_checkpoint']),config,device=device,
                              backbone_repo=Path(c['backbone_repo']))
    for size in (1,2):
        def targets():
            for _,descriptor,aux in teacher.iter_targets(images[:2],size):
                del descriptor,aux
        measure(f'teacher_pair_chunk_{size}',targets,3,2)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=c['learning_rate'])
    labels=torch.arange(8,device=device).repeat_interleave(4)
    local_index=torch.arange(256,device=device)[None]; q=torch.ones(1,256,device=device)
    losses=[]
    def step():
        model.train();optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): real=model(images[:32])
        real_loss=multi_similarity_loss(real,labels);reference=real.detach();real_loss.backward();del real
        with torch.autocast('cuda',dtype=torch.bfloat16): syn=model(images[32:36])
        global_loss=asymmetric_multi_positive_infonce(syn,torch.arange(4,device=device),reference,labels)
        (.1*global_loss).backward();del syn,reference
        local=0.
        for i in range(4):
            teacher_aux=[]
            for _,desc,aux in teacher.iter_targets(torch.stack([images[i*4],images[32+i]]),2):
                teacher_aux.append({k:v[:1].detach() for k,v in aux.items()})
                del desc,aux
            with torch.autocast('cuda',dtype=torch.bfloat16): _,student=model(images[32+i:33+i],return_aux=True)
            loss=local_distillation_losses(student,teacher_aux[0],local_index,q)
            weighted=(.1*loss['feature']+.1*loss['ot'])/4
            weighted.backward();local+=float(weighted.detach())
            del teacher_aux,student,loss,weighted
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        losses.append(dict(real=float(real_loss.detach()),global_aux=float(global_loss.detach()),local=local))
    # Exactly two disposable optimizer steps; first is measured too, no warmup step.
    for i in range(2):
        torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device);started=time.perf_counter()
        step();torch.cuda.synchronize(device)
        records.append(dict(name=f'disposable_bf16_step_{i+1}',seconds_per_call=time.perf_counter()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3))
    write_json(output,dict(gpu=torch.cuda.get_device_name(device),torch_version=torch.__version__,
        budget=budget,measurements=records,losses=losses,exact_retrieval_ranks_agree=ranks_agree,
        limitations=['Random image/descriptor inputs; no retrieval-quality conclusion.',
                    'Only real_image_extraction measurements include loading; correspondence and checkpoint I/O excluded.',
                    'Two disposable updates; no model weights or teacher features saved.']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args();c=load_config(args.config.resolve());out=args.output_dir.resolve()
    if args.worker: probe(c,out/'measurements.json');return
    out.mkdir(parents=True,exist_ok=False)
    run_child([sys.executable,str(Path(__file__).resolve()),'--config',str(args.config.resolve()),
               '--output-dir',str(out),'--worker'],out/'probe.log',c['torch_threads'],limits=c['resource_limits'])
    print(f'Resource measurements: {out / "measurements.json"}')


if __name__=='__main__': main()
