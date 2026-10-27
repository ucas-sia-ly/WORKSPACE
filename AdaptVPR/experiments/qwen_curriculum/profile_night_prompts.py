"""Bounded inference-only batch profiling; no training and no image generation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import night_place_cohort_v4 as policy
from experiments.qwen_curriculum import run_scene_night_safe as resource


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state',type=Path,default=policy.ROOT/'state')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-sizes',type=int,nargs='+',default=[4,8,16,32])
    p.add_argument('--stress-largest',action='store_true',help='Repeat largest available real inputs only for bounded resource stress; never export these prompts')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'report.json').exists(): raise ValueError('Use a new profiling directory')
    if any(n<1 or n>64 for n in args.batch_sizes): raise ValueError('Bounded batch sizes 1..64')
    policy.configure(); e=policy.engine;cfg=e.load(args.state)
    samples=[]
    for source in e.rows(args.state/'sources.jsonl'):
        path=args.state/'prompts'/f"{source['sample_id']}.json"
        if path.exists():
            item=e.sealed_read(path)
            if item['status']=='ready': samples.append(item)
        if len(samples)==max(args.batch_sizes): break
    if len(samples)<max(args.batch_sizes): raise ValueError('Need enough existing ready sources for the bounded probe')
    if args.stress_largest:
        area=max(item['source']['source_dimensions'][0]*item['source']['source_dimensions'][1] for item in samples)
        largest=[item for item in samples if item['source']['source_dimensions'][0]*item['source']['source_dimensions'][1]==area]
        samples=[largest[i%len(largest)] for i in range(max(args.batch_sizes))]
    initial=resource.snapshot()
    if initial['gpu_used_gib']>2: raise RuntimeError('Pause the existing inference task first')
    initial_swap=initial['swap_used_gib'];stop=threading.Event();limits=[]
    def watch():
        while not stop.wait(2):
            try:
                value=resource.snapshot(os.getpgrp())
                if (value['available_gib']<4 or value['gpu_used_gib']>44 or
                    value['swap_used_gib']-initial_swap>1):
                    limits.append(value);os.kill(os.getpid(),signal.SIGTERM);return
            except Exception as error:
                limits.append({'monitor_error':str(error)});os.kill(os.getpid(),signal.SIGTERM);return
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    thread=threading.Thread(target=watch,daemon=True);thread.start()
    reader=None;records=[];error=None
    try:
        reader=policy._planner(Path(cfg['model_path']))
        torch=reader.torch
        torch.cuda.set_per_process_memory_fraction(42*resource.GIB/torch.cuda.get_device_properties(0).total_memory)
        for count in args.batch_sizes:
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            subset=samples[:count];sources=[item['source'] for item in subset]
            probe={**cfg,'fingerprint':e.common.fingerprint([cfg['fingerprint'],'batch_probe',count])}
            tick=time.monotonic();phases={}
            for phase in ('analysis','compose'):
                requests=([policy.analysis_request()]*count if phase=='analysis' else
                    [policy.compose_request(item['plan']['source_evidence'],item['job']['style_id']) for item in subset])
                torch.cuda.synchronize();start=time.monotonic()
                replies=reader.ask(args.output/f'b{count}',sources,requests,phase,0,probe)
                torch.cuda.synchronize();elapsed=time.monotonic()-start;valid=0;errors=[]
                for item,raw in zip(subset,replies,strict=True):
                    try:
                        if phase=='analysis': policy.source_evidence(e.policy.parse_json(raw))
                        else: e.parse_plan(raw,item['plan']['source_evidence'],item['job']['style_id'])
                        valid+=1
                    except (ValueError,KeyError,TypeError) as exc: errors.append(str(exc))
                phases[phase]=dict(seconds=elapsed,valid=valid,total=count,errors=errors,
                    output_tokens=sum(len(reader.processor.tokenizer.encode(s)) for s in replies))
            seconds=time.monotonic()-tick;value=resource.snapshot(os.getpgrp())
            record=dict(batch_size=count,seconds=seconds,images_per_second=count/seconds,
                valid_complete_per_second=min(v['valid'] for v in phases.values())/seconds,
                cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/resource.GIB,
                cuda_peak_reserved_gib=torch.cuda.max_memory_reserved()/resource.GIB,
                resources=value,phases=phases)
            records.append(record);e.common.write_json(args.output/'progress.json',dict(records=records))
            print(json.dumps(record,ensure_ascii=False),flush=True)
            # Do not probe larger sizes when remaining VRAM cannot provide a safe growth margin.
            if record['cuda_peak_reserved_gib']>32: break
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}';raise
    finally:
        stop.set()
        if reader is not None: reader.close()
        e.common.write_json(args.output/'report.json',dict(parent_state=str(args.state.resolve()),
            parent_fingerprint=cfg['fingerprint'],records=records,error=error,guard_events=limits,
            profiler_sha256=e.common.file_sha256(__file__),
            largest_input_stress=args.stress_largest,
            inference_only=True,quality_scope='Schema validation only, not generated-image quality.',
            sampling_note='Batch membership changes sampling RNG; existing prompts remain unchanged.'))


if __name__=='__main__': main()
