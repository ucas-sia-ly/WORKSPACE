"""Independent image-grounded transient planning on a frozen appearance plan.

Separates the VLM's appearance-only task from its optional object-placement
task. The inherited appearance prose is never regenerated or overwritten.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

from PIL import Image

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import campaign,common,run,scene_prompting as planning,scene_aware_night as workflow

ROOT=common.WORKSPACE_ROOT/'outputs/scene_night_diversity_demo'


def object_request(reader,source,protected):
    image=Image.open(source).convert('RGB');image.thumbnail((672,672))
    request='''Inspect ONLY the ground-plane placement opportunity in this actual photo.
Find an EMPTY visible road/parking space for one NEW extra car, or a clear
sidewalk/crosswalk for one NEW full-body pedestrian; cyclist/scooter is also
possible on a road/cycle lane. Do not reuse an existing object's position or
change a seated person to standing. Some existing traffic/buildings do NOT
imply all remaining space is occupied: inspect the visible foreground/midground.
Prefer a clearly visible, normally scaled midground object, not a tiny distant
speck. Preserve road topology and keep main landmark evidence available.
An addition may occupy <=15% image area, partially occluding <=25% of any
protected box. Keep its complete body inside the frame with natural perspective.
This is a separate object proposal, with no instruction to change weather.
Return JSON only:
{"coordinate_system":"0_1000","feasible":true|false,
 "proposal":{"type":"vehicle|person|cyclist|scooter",
 "surface":"road|parking|sidewalk|crosswalk|cycle_lane", "box":[x0,y0,x1,y1],
 "description":"specific ONE NEW extra object on an identified empty region with natural scale"},
 "evidence":"specific empty-space evidence, or a concrete obstacle if infeasible"}.
All boxes use Qwen 0-to-1000 image coordinates. If infeasible, proposal=null.
Do not mark an image infeasible merely because it contains some buildings or
traffic. Never invent an empty surface that is not visible.
Prefer a road vehicle when a legal unoccupied road/parking region exists.
Lane markings, shadows and asphalt texture are NOT occupied space or obstacles.
Temporarily covering a short lane-marking segment beneath a new car is allowed;
it does not change the underlying road topology. Use a complete, visible object
at the scale implied by that actual position; do not enlarge it to hit a quota.
Protected permanent regions (all boxes in the SAME 0-to-1000 coordinates): '''+json.dumps([
    {**r,'box':[round(x*1000) for x in r['box']]} for r in protected])
    messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':request}]}]
    inputs=reader.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
        return_dict=True,return_tensors='pt').to(reader.device)
    with reader.torch.inference_mode():
        ids=reader.model.generate(**inputs,max_new_tokens=700,do_sample=False)
    return reader.processor.batch_decode(ids[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0]


def validate_proposal(raw,protected):
    value=planning.parse_json(raw)
    if type(value.get('feasible')) is not bool:
        raise ValueError('Explicit feasibility is required')
    planning.text(value.get('evidence'),'empty-space evidence',3)
    if value['feasible'] is False:
        if value.get('proposal') is not None:raise ValueError('Infeasible proposal must be null')
        return None,value
    if value.get('coordinate_system','0_1000')!='0_1000':
        raise ValueError('Coordinates differ from the explicit 0-to-1000 request contract')
    proposal=copy.deepcopy(value['proposal'])
    b=proposal.get('box')
    import math
    if not isinstance(b,list) or len(b)!=4 or any(type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=1000 for x in b):
        raise ValueError('Invalid native-coordinate box')
    proposal['box']=[x/1000 for x in proposal['box']]
    planning.validate_transient(proposal,protected)
    b=proposal['box']
    if (b[2]-b[0])*1000<1 or (b[3]-b[1])*1000<1:
        raise ValueError('Proposed box is below native-coordinate resolution')
    value['coordinate_system']='0_1000'
    value['coordinate_contract']='request-defined; explicit contradictory response units rejected'
    return proposal,value


def prepare(args):
    parent=args.parent_state
    base,jobs=workflow.load_plan(parent)
    planner_cfg=campaign.config_file(parent/'planner_config.json')
    directory=args.output_dir
    workflow.check_roots(directory,args.image_root)
    with run.run_lock(directory):
        if (directory/'plan_config.json').exists():return workflow.load_plan(directory)[0]
        sources=common.read_jsonl(parent/'source_snapshot.jsonl')
        if not args.execute:return dict(sources=len(sources),max_jobs=len(jobs)*2,model_calls=0,
            next='prepare --execute reads each source for independent object placement')
        campaign.ensure_stopped()
        for name,digest in planner_cfg['files_sha256'].items():
            if common.file_sha256(name)!=digest:raise ValueError('Frozen inherited planner changed')
        scenes={}
        for file in parent.glob('scene_plans/*.json'):
            value=json.loads(file.read_text())
            if 'plan' in value:scenes[value['source_sha256']]=value['plan']
        reader=None;plans=[];records=[]
        try:
            for source in sources:
                plan=copy.deepcopy(scenes[source['source_sha256']])
                dest=directory/'object_plans'/f"{source['source_sha256']}.json"
                if dest.exists():
                    record=json.loads(dest.read_text())
                    if (record!=run.sealed(record) or record['parent_fingerprint']!=base['fingerprint']):
                        raise ValueError('Object placement receipt changed')
                    proposal=record['proposal']
                else:
                    reserve=dest.with_suffix('.request.json')
                    if reserve.exists():raise ValueError('Unresolved bounded object-planner request')
                    common.write_json(reserve,dict(source_sha256=source['source_sha256'],parent_fingerprint=base['fingerprint']))
                    if reader is None:reader=planning.LocalVisionPlanner(planner_cfg['request']['model_path'])
                    raw=object_request(reader,source['source_path'],plan['protected_regions'])
                    common.write_json(dest.with_suffix('.raw.json'),dict(response=raw))
                    try:proposal,evidence=validate_proposal(raw,plan['protected_regions'])
                    except (ValueError,KeyError,TypeError) as error:
                        proposal=None;evidence=dict(status='unknown',error=str(error))
                    record=run.sealed(dict(source_sha256=source['source_sha256'],parent_fingerprint=base['fingerprint'],
                        proposal=proposal,evidence=evidence))
                    common.write_json(dest,record)
                for variant in plan['variants']:variant['transient']=proposal
                plan['independent_object_plan_sha256']=record['result_sha256']
                plan['transient_candidates']=[proposal] if proposal else []
                plans.append(plan);records.append(record)
        finally:
            if reader is not None:reader.close()
        jobs=workflow.make_jobs(sources,plans,True,planner_cfg['request']['seed'])
        if len(jobs)>args.max_jobs:raise ValueError('Derived image budget exceeds --max-jobs')
        common.write_jsonl(directory/'source_snapshot.jsonl',sources)
        common.write_jsonl(directory/'plan.jsonl',jobs)
        files={**planner_cfg['files_sha256'],**base['files_sha256'],
            str(Path(__file__).resolve()):common.file_sha256(__file__),
            str(parent/'plan_config.json'):common.file_sha256(parent/'plan_config.json'),
            str(parent/'planner_config.json'):common.file_sha256(parent/'planner_config.json')}
        files.update({str(p):common.file_sha256(p) for p in directory.glob('object_plans/*.json')})
        derived=campaign.frozen(directory/'planner_config.json',dict(version=1,request=planner_cfg['request'],
            parent_fingerprint=base['fingerprint'],files_sha256=files,
            object_placement_policy=dict(request_units='0_1000',minimum_native_box_span=1,
                                         max_image_area=planning.MAX_TRANSIENT_AREA,preferred='vehicle if legal space exists'),
            workflow='frozen appearance plans plus independent image-grounded object proposals'))
        from collections import Counter
        cfg=campaign.frozen(directory/'plan_config.json',dict(version=1,stage=planning.VERSION,
            planner_fingerprint=derived['fingerprint'],num_jobs=len(jobs),request_budget=len(jobs),
            domain_counts=dict(Counter(j['condition'] for j in jobs)),image_root=str(args.image_root),
            quality=base['quality'],service=base['service'],plan_sha256=common.file_sha256(directory/'plan.jsonl'),
            files_sha256={**files,str(directory/'source_snapshot.jsonl'):common.file_sha256(directory/'source_snapshot.jsonl')}))
        workflow.engine().rebuild_manifest(directory,cfg,[])
        return workflow.publish(directory,cfg,jobs,{},{},[],'prepared','all')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','verify','audit','report','check'))
    parser.add_argument('--parent-state',type=Path,default=workflow.ROOT/'state')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'state')
    parser.add_argument('--image-root',type=Path,default=ROOT)
    parser.add_argument('--max-jobs',type=int,default=8)
    parser.add_argument('--max-calls',type=int,default=0)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    for key in ('parent_state','output_dir','image_root'):setattr(args,key,getattr(args,key).resolve())
    if args.command=='prepare':print(json.dumps(prepare(args),ensure_ascii=False,indent=2))
    else:
        command=[args.command,'--output-dir',str(args.output_dir),'--image-root',str(args.image_root),
                 '--max-calls',str(args.max_calls)]
        if args.execute:command.append('--execute')
        workflow.main(command)


if __name__=='__main__':main()
