#!/usr/bin/env python3
"""Small, blinded paired-Qwen daylight-residual screen; experimental flags only."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time

SYSTEM = '''You compare two photographs for a visual place recognition dataset.
Image A is the original daytime photograph. Image B is its generated night edit.
Audit ONLY retained DAYLIGHT ILLUMINATION in B. Do not judge structural edits,
text correctness, aesthetics, or whether a scene is too dark in general.
Inspect ground, grass, walls and foliage, not just the sky. Look for source-like
sunlit surfaces, broad directional solar illumination, and dappled sun/shadow
boundaries retained at the same locations after only a dark-sky/global-darkening
edit. Distinguish physical scene boundaries from lighting boundaries.
Bright night scenes, green foliage, a black sky, sharp shadows, and no visible
lamp are NOT sufficient evidence alone. Streetlights, headlights and moonlight
can produce shadows, and their sources may be outside the frame. Compare the
actual pattern in BOTH photographs. Do not invent a light source to justify B;
also do not assume daylight merely because the source is daytime. If reasonable
night lighting cannot be distinguished, choose suspected, not clear_residual.
Do not assume any input is defective; a valid night edit is allowed.
Return JSON ONLY, with no markdown, exactly this schema:
{"assessment":"clear_residual|suspected|not_found",
 "summary":"brief specific paired-image observation",
 "regions":[{"box_0_1000":[x0,y0,x1,y1],
 "source_observation":"what the source shows at this region",
 "generated_observation":"what B shows at the corresponding region",
 "evidence":"why this is a retained sunlight pattern, not just bright geometry",
 "alternative_explanation":"possible legitimate night illumination or uncertainty"}]}
Use coordinates relative to the full ORIGINAL generated image B, x/y from 0 to
1000, not a combined two-image canvas. At most 3 regions. A clear_residual or
suspected assessment needs at least one specific evidence region. For not_found
use regions=[]; it means no clear daylight residual found, not full quality PASS.
'''


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20), b''):h.update(chunk)
    return h.hexdigest()


def parse_response(text):
    text=text.strip()
    if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
    value=json.loads(text)
    if value.get('assessment') not in {'clear_residual','suspected','not_found'}:
        raise ValueError('Invalid assessment')
    if not isinstance(value.get('summary'),str) or not value['summary'].strip():
        raise ValueError('Missing summary')
    regions=value.get('regions')
    if not isinstance(regions,list) or len(regions)>3:raise ValueError('Invalid regions')
    if (value['assessment']=='not_found') != (len(regions)==0):raise ValueError('Assessment/regions conflict')
    for r in regions:
        box=r.get('box_0_1000')
        if (not isinstance(box,list) or len(box)!=4 or
            any(type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=1000 for x in box)
            or box[0]>=box[2] or box[1]>=box[3]):raise ValueError('Invalid generated-image box')
        for k in ['source_observation','generated_observation','evidence','alternative_explanation']:
            if not isinstance(r.get(k),str) or not r[k].strip():raise ValueError('Missing '+k)
    return value



def parse_shadow_response(text):
    """Observation-first shadow gate; NO brightness-based or model-label override."""
    text=text.strip()
    if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
    value=json.loads(text)
    generated=value.get('generated_ground_shadow')
    source=value.get('source_ground_shadow')
    relation=value.get('pattern_relation')
    if generated not in {'present','absent','not_visible'} or source not in {'present','absent','not_comparable'}:
        raise ValueError('Invalid shadow visibility observations')
    if relation not in {'same','different','uncertain','not_applicable'}:
        raise ValueError('Invalid shadow pattern relation')
    comparable=generated=='present' and source=='present'
    if not comparable:
        if relation!='not_applicable':raise ValueError('Cannot compare absent/unobservable shadows')
        assessment='not_found'
    else:
        if relation=='not_applicable':raise ValueError('Comparable shadows need a relation')
        assessment={'same':'clear_residual','different':'not_found','uncertain':'suspected'}[relation]
    # A different-pattern comparison may legitimately include inspection boxes.
    # Validate their geometry/evidence without calling them defect boxes.
    check=dict(value,assessment='suspected' if value.get('regions') else 'not_found')
    parse_response(json.dumps(check,ensure_ascii=False))
    if comparable and not value.get('regions'):raise ValueError('Comparable shadows require a localized observation')
    return dict(value,assessment=assessment,
                decision_basis='ground_shadow_observation_gate_v1',
                pass_scope='this shadow-reuse check only, not complete night quality')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample',required=True);p.add_argument('--model',required=True)
    p.add_argument('--response-schema', choices=['daylight','ground_shadow'], default='daylight')
    p.add_argument('--system-file', help='Explicit alternate audit instruction; frozen into protocol')
    p.add_argument('--output-dir',required=True);p.add_argument('--execute',action='store_true')
    args=p.parse_args();out=Path(args.output_dir)
    system=Path(args.system_file).read_text() if args.system_file else SYSTEM
    rows=json.loads(Path(args.sample).read_text())
    if len(rows)>32 or len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Unique pilot pairs <=32 required')
    model=Path(args.model).resolve()
    files=sorted(x for x in model.iterdir() if x.is_file())
    protocol={'version':'daylight_residual_pilot_v1','system':system,'source_sample_sha256':sha(args.sample),
              'model_path':str(model),'model_files_sha256':{x.name:sha(x) for x in files},
              'script_sha256':sha(__file__),'response_schema':args.response_schema,'max_side':672,'max_new_tokens':768,
              'do_sample':False,'dtype':'bfloat16','attention':'sdpa','batch_size':1,
              'seed':42,'max_torch_gpu_gib':18,'threads':4,
              'training':False,'generation':False,'automatic_dataset_rejection':False,
              'blinding':'model receives two images and fixed generic request only; no IDs/prompts/styles/labels/scores'}
    if not args.execute:print(json.dumps(protocol,indent=2));return
    out.mkdir(parents=True,exist_ok=True)
    receipt=out/'protocol.json'
    if receipt.exists() and json.loads(receipt.read_text())!=protocol:raise ValueError('Frozen pilot changed')
    receipt.write_text(json.dumps(protocol,indent=2))
    previous={}
    if (out/'results.jsonl').exists():
        previous={r['sample_id']:r for r in map(json.loads,(out/'results.jsonl').read_text().splitlines())}
    for r in rows:
        for key,h in [('source_path','source_sha256'),('output_path','output_sha256')]:
            if sha(r[key])!=r[h]:raise ValueError('Image hash mismatch')
    todo=[r for r in rows if r['sample_id'] not in previous]
    if not todo:print('All pairs already audited.');return
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from PIL import Image
    torch.set_num_threads(4);torch.manual_seed(42)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; do not silently change device')
    torch.cuda.set_per_process_memory_fraction(min(18*2**30/torch.cuda.get_device_properties(0).total_memory,.9))
    torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    net=Qwen3VLForConditionalGeneration.from_pretrained(str(model),local_files_only=True,
        dtype=torch.bfloat16,device_map={'':'cuda'},low_cpu_mem_usage=True,attn_implementation='sdpa').eval().requires_grad_(False)
    processor=AutoProcessor.from_pretrained(str(model),local_files_only=True)
    load_seconds=time.monotonic()-start
    with (out/'results.jsonl').open('a') as log:
        for i,r in enumerate(todo):
            tick=time.monotonic();images=[]
            for key in ['source_path','output_path']:
                with Image.open(r[key]) as im:image=im.convert('RGB');image.thumbnail((672,672))
                images.append(image)
            messages=[{'role':'system','content':[{'type':'text','text':system}]},
                {'role':'user','content':[{'type':'text','text':'Image A: original daytime source.'},
                    {'type':'image','image':images[0]}, {'type':'text','text':'Image B: generated night candidate.'},
                    {'type':'image','image':images[1]}, {'type':'text','text':'Compare illumination in A and B. Return the specified JSON.'}]}]
            inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                return_dict=True,return_tensors='pt').to('cuda')
            with torch.inference_mode():
                output=net.generate(**inputs,max_new_tokens=768,do_sample=False,use_cache=True)
            tokens=output[:,inputs['input_ids'].shape[1]:]
            raw=processor.batch_decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
            try:parsed=(parse_shadow_response if args.response_schema=='ground_shadow' else parse_response)(raw);error=None
            except (ValueError,TypeError,KeyError,IndexError) as e:parsed=None;error=str(e)
            result={'sample_id':r['sample_id'],'review_id':r.get('review_id'),
                'source_sha256':r['source_sha256'],'output_sha256':r['output_sha256'],
                'raw_response':raw,'parsed':parsed,'parse_error':error,
                'assessment':parsed['assessment'] if parsed else 'unknown',
                'pilot_action':'isolate_candidate' if parsed and parsed['assessment'] in ['clear_residual','suspected'] else 'no_automatic_action',
                'model_boxes_are_verified_masks':False,'seconds':time.monotonic()-tick,
                'generated_tokens':int(tokens.shape[1]),'hit_token_limit':tokens.shape[1]>=768}
            log.write(json.dumps(result,ensure_ascii=False)+'\n');log.flush();os.fsync(log.fileno())
            print(i+1,len(todo),r.get('review_id'),result['assessment'],round(result['seconds'],2),flush=True)
            del inputs,output,tokens
    runtime={'device':torch.cuda.get_device_name(0),'torch_version':str(torch.__version__),
        'model_load_seconds':load_seconds,'total_seconds':time.monotonic()-start,
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
        'process_max_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        'images_unchanged':all(sha(r[k])==r[h] for r in rows for k,h in [('source_path','source_sha256'),('output_path','output_sha256')])}
    (out/'runtime.json').write_text(json.dumps(runtime,indent=2))
    del net,processor;gc.collect();torch.cuda.empty_cache()
    print(json.dumps(runtime),flush=True)
if __name__=='__main__':main()
