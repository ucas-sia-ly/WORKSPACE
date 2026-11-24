"""Image-conditioned, open-ended VPR edit planning and bounded transient edits.

The VLM writes each appearance prompt. The program validates its structured
edit permissions and keeps that prose, rather than replacing it with a bank.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from PIL import Image

from experiments.qwen_curriculum import common

VERSION = 'scene_conditioned_night_v2'
SURFACES = {'vehicle': {'road', 'parking'}, 'person': {'sidewalk', 'crosswalk'},
            'cyclist': {'road', 'cycle_lane'}, 'scooter': {'road', 'cycle_lane'}}
MAX_TRANSIENT_AREA = .15  # Explicit edit budget, not an estimated quality threshold.
MAX_LANDMARK_OCCLUSION = .25  # Partial transient occlusion, never replacement.
SYSTEM = '''You plan image-conditioned augmentation for visual place recognition.
Inspect the actual input photograph. Write varied, plausible NIGHT photographs
of exactly this place. Diversity is useful: do not always choose the same warm
street-lamp look or uniformly dark look. Invent scene-appropriate combinations
of illumination, night phase, weather, exposure and transient street activity.
These are open-ended descriptions, not a fixed list of prompt templates.

Permanent place identity is protected: camera, street topology, buildings,
windows/doors, signs/lettering, fences and landmark layout must remain. Changing
their illumination is allowed. Existing windows may light up. Do not invent
shops, signs, architecture, lamp fixtures or readable text. New vehicle lights
are allowed ONLY when the corresponding transient vehicle is explicitly added.
Keep natural photographic exposure and useful place evidence; neither making
everything black nor a black sky above daytime lighting is a successful edit.

Independently inspect ground-plane space for optional temporary vehicles,
pedestrians, cyclists or scooters. List feasible proposals in transient_candidates;
use [] with a concrete transient_skip_reason only when none is feasible.
Every candidate is one NEW ADDITIONAL object in visibly EMPTY space. Do not
reposition an existing person, change a seated person to standing, or reuse an
already occupied vehicle location. Describe the empty placement relative to
actual scene landmarks. Preserve all existing people and vehicles.
Overall editability describes appearance editing, and medium does NOT prohibit
objects. An empty visible lane can support a vehicle; a clear sidewalk can
support a person. Each proposal is a separate paired edit, not part of the
appearance-only prompt. Use natural perspective, avoid borders, and keep the
main landmark recognizable. Limited partial occlusion is allowed: an object
may cover at most 25% of any protected box and 15% of the whole image. Do not
replace the hidden structure. No quota forces objects into unsuitable photos.

Return JSON only, schema_version=1, coordinate_system="0_1000".
All boxes use Qwen's 0-to-1000 image coordinates [x0,y0,x1,y1].
Use exactly these keys:
{
 "schema_version":1,
 "coordinate_system":"0_1000",
 "scene_summary":"specific visible scene",
 "editability":"low|medium|high",
 "protected_regions":[{"description":"important landmark/sign", "box":[...]}],
 "transient_candidates":[{"type":"vehicle|person|cyclist|scooter", "surface":"road|parking|sidewalk|crosswalk|cycle_lane",
   "box":[...], "description":"exactly ONE object, source-grounded placement, natural scale and matched lighting"}],
 "transient_skip_reason":"empty if feasible proposals exist, otherwise explain the actual obstacle",
 "variants":[{
   "style_name":"a specific, original style name",
   "weather":"dry|wet|rain|snow|fog",
   "lighting":"specific lighting mechanism anchored to visible scene",
   "exposure":"specific photographic exposure choice",
   "appearance_prompt":"Complete English editing prompt, 60-180 words, tied to this exact image. No new objects in this prompt."
}]
}
IMPORTANT COMPONENT SEPARATION: appearance_prompt describes light, weather,
exposure and scene anchors. It may preserve existing people/traffic and change
their illumination, but must not request adding, removing or repositioning
objects. Put additional object-specific prose in transient_candidates.
Light spill, distant headlights, reflections and atmosphere can still vary.
Always consider transient candidates separately from weather. Do not omit the
transient_candidates field. Give at most two feasible alternatives.
Protected boxes should identify actual key structures/signs, not the whole frame;
use at most three. Protect permanent landmarks, not existing people or traffic.
Preserve signs, traffic lights, lamp fixtures and street
furniture; do not invent them for atmosphere or color contrast.
The optional object's whole body fits inside its box; total box area <=15% of
the frame and it does not intersect protected boxes. Return variants that are
visibly distinct in more than their names. No markdown, confidence scores or
claims that a plan is verified merely because you proposed it.'''

CONTRACT = ('Preserve the exact viewpoint, permanent architecture, windows, signs and lettering, '
            'and road topology of the source. Change illumination, atmosphere and explicitly authorized '
            'transient objects only. Keep the place recognizable; do not create new shops, signs, '
            'buildings, traffic lights, lamp fixtures or street furniture.')


def definition():
    return dict(version=VERSION, system=SYSTEM, contract=CONTRACT,
                surfaces={k: sorted(v) for k,v in SURFACES.items()}, max_transient_area=MAX_TRANSIENT_AREA,
                max_landmark_occlusion=MAX_LANDMARK_OCCLUSION,
                prompt_policy='retain VLM-written per-image prose; no bank overwrite',
                bbox_interpretation='VLM planned placement, not verified segmentation or negative supervision')


def parse_json(text):
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n',1)[1].rsplit('```',1)[0].strip()
    value = json.loads(text)
    if not isinstance(value,dict):
        raise ValueError('Planner response must be an object')
    return value


def normalize_plan(value):
    """Convert only an explicitly declared coordinate system; never guess units."""
    import copy
    value=copy.deepcopy(value)
    if value.get('coordinate_system')!='0_1000':
        raise ValueError('Declare coordinate_system="0_1000" explicitly')
    if not isinstance(value.get('transient_candidates'),list) or len(value['transient_candidates'])>2:
        raise ValueError('Include transient_candidates as a list of at most two feasible proposals')
    regions=list(value.get('protected_regions',[]))
    regions += list(value.get('transient_candidates',[]))
    regions += [v['transient'] for v in value.get('variants',[]) if v.get('transient') is not None]
    for region in regions:
        b=region.get('box')
        if not isinstance(b,list) or len(b)!=4 or any(type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=1000 for x in b):
            raise ValueError('Boxes must use explicit finite 0-to-1000 coordinates')
        region['box']=[x/1000 for x in b]
    value['coordinate_system']='normalized_0_1'
    value['source_coordinate_system']='0_1000'
    candidates=[];rejected=[]
    for proposal in value.get('transient_candidates',[]):
        try:
            validate_transient(proposal,value.get('protected_regions',[]))
            candidates.append(proposal)
        except ValueError as error:
            rejected.append(dict(proposal=proposal,reason=str(error)))
    value['transient_candidates']=candidates
    value['rejected_transient_candidates']=rejected
    for index,variant in enumerate(value.get('variants',[])):
        if candidates:variant['transient']=candidates[index%len(candidates)]
        else:variant.setdefault('transient',None)
    return value


def validate_transient(transient,protected):
    if not isinstance(transient,dict) or transient.get('type') not in SURFACES or transient.get('surface') not in SURFACES[transient['type']]:
        raise ValueError('Object is not on an authorized ground-plane surface')
    b=box(transient.get('box'))
    if (b[0]<=0 or b[1]<=0 or b[2]>=1 or b[3]>=1 or
            (b[2]-b[0])*(b[3]-b[1])>MAX_TRANSIENT_AREA or
            any(overlap(b,r['box'])>((r['box'][2]-r['box'][0])*(r['box'][3]-r['box'][1]))*MAX_LANDMARK_OCCLUSION
                for r in protected)):
        raise ValueError('Transient exceeds edit area, touches a border or covers too much protected evidence')
    text(transient.get('description'),'transient description',4)


def box(value):
    if (not isinstance(value,list) or len(value)!=4 or
            any(type(v) not in (int,float) or not math.isfinite(v) for v in value)):
        raise ValueError('Box must contain four finite normalized coordinates')
    x0,y0,x1,y1=value
    if not (0<=x0<x1<=1 and 0<=y0<y1<=1):
        raise ValueError('Invalid normalized box')
    return value


def overlap(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))


def text(value,name,minimum=1):
    if not isinstance(value,str) or len(value.strip().split())<minimum:
        raise ValueError('Missing or too short '+name)
    return value.strip()


def validate_plan(plan, variants):
    if type(plan.get('schema_version')) is not int or plan['schema_version']!=1:
        raise ValueError('Unsupported scene planning schema')
    text(plan.get('scene_summary'),'scene_summary',3)
    if plan.get('editability') not in ('low','medium','high'):
        raise ValueError('Invalid scene editability')
    protected=plan.get('protected_regions')
    if not isinstance(protected,list) or not protected:
        raise ValueError('Planner must identify protected place evidence')
    for region in protected:
        text(region.get('description'),'protected region');box(region.get('box'))
    values=plan.get('variants')
    if not isinstance(values,list) or len(values)!=variants:
        raise ValueError('Planner returned the wrong variant budget')
    names,prompts=set(),set()
    for v in values:
        name=text(v.get('style_name'),'style_name')
        prompt=text(v.get('appearance_prompt'),'appearance_prompt',30)
        if has_uncontrolled_object_directive(prompt):
            raise ValueError('appearance_prompt mixes transient additions/removals with appearance; move them to transient_candidates')
        if name.casefold() in names or prompt.casefold() in prompts:
            raise ValueError('Duplicate styles or appearance prompts')
        names.add(name.casefold());prompts.add(prompt.casefold())
        if v.get('weather') not in ('dry','wet','rain','snow','fog'):
            raise ValueError('Invalid weather metadata')
        text(v.get('lighting'),'lighting');text(v.get('exposure'),'exposure')
        transient=v.get('transient')
        if transient is not None:
            validate_transient(transient,protected)
    return plan


def has_uncontrolled_object_directive(prompt):
    """Catch explicit authorization conflicts; this is not a semantic verifier.

    Preservation phrases and existing headlights remain valid. Actual counts,
    location and permanent structure are checked from images after generation.
    """
    actor=r'(?:person|people|pedestrian(?:s)?|car(?:s)?|vehicle(?:s)?|cyclist(?:s)?|scooter(?:s)?|truck(?:s)?|van(?:s)?|bus(?:es)?)'
    for sentence in re.split(r'[.!?;]\s*',prompt.lower()):
        if not re.search(r'\b'+actor+r'\b',sentence):continue
        if re.search(r'\bno\b.*\b'+actor+r'\b.*\b(?:present|visible|in frame)\b',sentence):return True
        if re.search(r'\b(?:no|not|never|without|existing|original)\b',sentence):continue
        if re.search(r'\b(?:add|insert|introduce|place|put|remove|erase)\b.*\b'+actor+r'\b',sentence):return True
        if re.search(r'\b(?:a|one|single|new|another|lone)\b.{0,35}\b'+actor+r'\b.{0,35}\b(?:appears|walks|stands|passes|sits|moves|rides)\b',sentence):return True
    return False


def render_prompt(variant, with_transient=False):
    # Retain this exact source-conditioned prose, including when the optional
    # paired object is disabled. Only the short authorization contract changes.
    appearance=text(variant['appearance_prompt'],'appearance_prompt')
    transient=variant.get('transient') if with_transient else None
    if with_transient and transient is None:
        raise ValueError('No feasible transient was planned')
    authorized=('Do not add or remove vehicles, people or other objects.' if transient is None else
        'For this paired version, the following single transient addition is explicitly authorized '
        'even if the appearance-only description above says no new objects. '
        f"Add exactly one {transient['type']}: {transient['description']}. "
        f"Place its full body on the {transient['surface']} within normalized image box {transient['box']}; "
        'match perspective, night lighting and contact shadows. Preserve all existing objects. '
        'Add no other objects and keep protected landmarks visible.')
    return appearance+'\n\n'+CONTRACT+'\n'+authorized


class LocalVisionPlanner:
    """Read real images with the installed Qwen3-VL; no mock fallback/download."""
    def __init__(self,model_path,device='cuda',max_tokens=2200):
        import torch
        from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
        if not torch.cuda.is_available() or device!='cuda':
            raise RuntimeError('Local planning requires the available CUDA device')
        self.torch=torch;self.device=device;self.max_tokens=max_tokens
        self.model=Qwen3VLForConditionalGeneration.from_pretrained(str(model_path),local_files_only=True,
            dtype=torch.bfloat16,device_map={'':device},low_cpu_mem_usage=True,attn_implementation='sdpa').eval()
        self.processor=AutoProcessor.from_pretrained(str(model_path),local_files_only=True)

    def read(self,image_path,variants,seed,previous_styles=(),feedback=''):
        image=Image.open(image_path).convert('RGB')
        image.thumbnail((672,672))
        request=(f'Inspect this source and provide exactly {variants} distinct night variants. '
                 f'Diversity draw {seed}. Recent style names to avoid repeating: {list(previous_styles)[-20:]}. '
                 'Use free, scene-specific ideas, not a uniform template. Describe safe optional street activity only when feasible. '
                 'Use coordinate_system="0_1000" and include transient (object or null) for every variant. '+feedback)
        messages=[{'role':'system','content':[{'type':'text','text':SYSTEM}]},{'role':'user','content':[
            {'type':'image','image':image},{'type':'text','text':request}]}]
        inputs=self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
            return_dict=True,return_tensors='pt').to(self.device)
        with self.torch.inference_mode(),self.torch.random.fork_rng(devices=[self.torch.cuda.current_device()]):
            self.torch.manual_seed(seed)
            generated=self.model.generate(**inputs,max_new_tokens=self.max_tokens,do_sample=True,
                temperature=.8,top_p=.9,top_k=30)
        return self.processor.batch_decode(generated[:,inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]

    def close(self):
        del self.model,self.processor
        import gc
        gc.collect();self.torch.cuda.empty_cache()

    def audit(self,source_path,candidate_path,authorization):
        request='''Image 1 is the real source; Image 2 is the edited candidate.
Audit the supplied edit authorization against BOTH images. Ignore authorized
weather, illumination and temporary occlusion when comparing permanent structure.
Do not infer correctness merely from a plausible prompt. If uncertain return unknown.
Return JSON only: {"status":"candidate|reject|unknown", "permanent_changes":[],
"authorized_edit_achieved":true|false, "unexpected_objects":[],
"placement_legal":true|false, "landmarks_visible":true|false,
"observed_added_transients":["vehicle|person|cyclist|scooter"],
"evidence":"specific comparison including any retained daylight or wrong objects"}.
A candidate requires no permanent changes/unexpected objects, successful requested
appearance/object edit, legal ground-plane placement and useful landmark visibility.
Changing a facade/window/sign or inventing a shop is a permanent change.
Count newly ADDED objects by comparing both images, not existing source objects.
A rider and bicycle together count as one cyclist. Do not copy requested counts.
Authorization: '''+json.dumps(authorization,ensure_ascii=False)
        images=[]
        for path in (source_path,candidate_path):
            with Image.open(path) as opened:
                image=opened.convert('RGB');image.thumbnail((672,672));images.append(image)
        messages=[{'role':'user','content':[{'type':'image','image':im} for im in images]+[{'type':'text','text':request}]}]
        inputs=self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
            return_dict=True,return_tensors='pt').to(self.device)
        with self.torch.inference_mode():
            ids=self.model.generate(**inputs,max_new_tokens=700,do_sample=False)
        return self.processor.batch_decode(ids[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0]
