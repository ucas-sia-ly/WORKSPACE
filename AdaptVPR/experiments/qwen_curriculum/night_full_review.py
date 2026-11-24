"""Read-only pairwise Qwen visual audit and bounded prompt edits. Never generates images."""
from __future__ import annotations
import argparse,copy,difflib,json,os,signal,time
from pathlib import Path
from experiments.qwen_curriculum import common,campaign
from experiments.qwen_curriculum.night_full_filter import connect,dumps,atomic,sha,status,check_resources,common_lock
MODEL=Path('/home/admin123/github/checkpoints/Qwen3-VL-4B-Instruct')
STOP=False
CATEGORIES={'daylight','structure','lighting','unusable'}
AUDIT='''You inspect paired real-source and generated NIGHT photographs for visual place recognition. Image 1 is the exact real source; image 2 is its generated night version. Inspect pixels, not whether the prompt sounds good. Be permissive: keep usable images unless a clear, consequential defect exists.
Check: (1) strong retained DAY sunlight patches or unmistakable solar tree shadows despite a dark sky, (2) significant permanent building/window/door/sign/road geometry changed or invented, (3) implausible emission from walls, sign backs, tree bases or fabricated fixtures, (4) widespread corruption/blackout that hides place identity. A bright night can be valid. Dark regions, clean rendering, color style differences, plausible vehicle/person changes and ground wetness changes are not defects on their own. A reasonable off-frame light source is allowed. A small window-glow or weak shadow alone is not proof of an error. Do not equate low image similarity with structural damage. Treat ambiguous/occluded evidence as uncertain, not a fabricated fault.
Return concise JSON: {"verdict":"keep|issue|unknown","issues":[{"category":"daylight|structure|lighting|unusable","certainty":"clear|uncertain","location":"specific image area","evidence":"what is visibly wrong relative to source","persistent_solar_artifact":true|false}],"evidence":"brief pixel-based summary"}. persistent_solar_artifact=true only for strong original solar patches or complex sharp sun/tree shadows that remain in image 2; such sources will be replaced, not repeatedly prompt-edited. If no clear defect, keep. Maximum 3 issues. Do not rewrite the prompt in this response.'''
EDIT='''You plan image-conditioned augmentation for visual place recognition. These are open-ended descriptions, not fixed templates. Make ONLY small targeted changes to the supplied original NIGHT prompt to address the provided clear visual defect. Keep the existing style and all correct content; do not rewrite the whole prompt. Preserve permanent geometry, not original sunlight. Do not invent lamps, windows, signs or shops. Plausible transient people/vehicles and dry/wet changes are permitted. No visible lamp does not prevent weak natural ambient or off-frame illumination. Return JSON {"edits":[{"old":"exact existing substring copied verbatim","new":"short replacement"}],"reason":"why each change addresses observed defect"}. At most 3 edits; old strings must each occur exactly once in original appearance text. Do not alter the fixed final preservation paragraph. Replace at most 30% of original words (minimum allowance 15 words); each replacement at most 50 words. Keep the rest exactly unchanged. No code fences or whole rewritten prompt.'''
SELECT='''You select a DIFFERENT real source photo of the SAME city/place for a later NIGHT edit. Inspect each candidate independently. Prefer uniform/overcast light, few strong solar patches or crisp tree shadows, useful permanent structure and little severe privacy blur/occlusion. A different view is allowed but must use its own exact source pairing. Return JSON {"selected_index":integer|null,"reason":"specific evidence from selected image","source_summary":"actual visible structures and lighting"}. Use the supplied zero-based candidate indices. If every candidate is clearly unsuitable or unreadable return null rather than inventing a good option. Do not generate or edit an image.'''
COMPOSE='''You plan image-conditioned augmentation for visual place recognition. These are open-ended descriptions, not fixed prompt templates. Read the actual NEW source photo and write ONE source-specific English NIGHT edit prompt (50-160 words). Preserve its viewpoint, permanent buildings, windows/doors/signs/road layout and useful landmarks. Do not reuse the previous source's semantic evidence or patch positions. Change daylight lighting and shadows; sky must be night including gaps between branches. Keep texture readable without pure black. Natural weak ambient light or off-frame light is permitted without visible fixtures; never refuse solely because no lamp is visible. Do not invent fixed lamps, shops, signs, windows or luminous tree bases. Plausible transient vehicle/person and wet/dry changes are allowed, not required. Retain the original lighting style if suitable; otherwise select an appropriate supplied non-mercury style. Return JSON {"style_id":"catalog id","appearance_prompt":"complete English editing prompt","reason":"source-grounded lighting choice"}.'''

def derive_review(v):
 if not isinstance(v,dict) or v.get('verdict') not in {'keep','issue','unknown'} or not isinstance(v.get('issues'),list) or len(v['issues'])>3 or not isinstance(v.get('evidence'),str) or len(v['evidence'].strip())<10:raise ValueError('Invalid visual review schema')
 for issue in v['issues']:
  if not isinstance(issue,dict) or issue.get('category') not in CATEGORIES or issue.get('certainty') not in {'clear','uncertain'} or type(issue.get('persistent_solar_artifact')) is not bool or any(not isinstance(issue.get(k),str) or len(issue[k].strip())<5 for k in ['location','evidence']):raise ValueError('Visual issue requires explicit localized evidence')
  if issue['persistent_solar_artifact'] and issue['category']!='daylight':raise ValueError('Solar flag only applies to daylight evidence')
 clear=[i for i in v['issues'] if i['certainty']=='clear']
 if any(i['persistent_solar_artifact'] for i in clear):return 'replace_source'
 if clear:return 'minor_edit'
 return 'keep' if v['verdict']=='keep' and not v['issues'] else 'unknown'

def minor_edit(original,edits):
 if not isinstance(edits,list) or not 1<=len(edits)<=3:raise ValueError('Require 1..3 small literal edits')
 marker='\n\nKeep this exact viewpoint';body,sep,tail=original.partition(marker);new=body;old_words=0
 spans=[]
 for e in edits:
  if not isinstance(e,dict) or not isinstance(e.get('old'),str) or not e['old'] or not isinstance(e.get('new'),str) or not e['new'].strip():raise ValueError('Nonempty literal replacement required')
  if body.count(e['old'])!=1 or new.count(e['old'])!=1:raise ValueError('Old substring must occur exactly once')
  a=body.index(e['old']);b=a+len(e['old'])
  if any(a<d and b>c for c,d in spans):raise ValueError('Edits overlap')
  spans.append((a,b));old_words+=len(e['old'].split())
  if len(e['new'].split())>50:raise ValueError('Replacement too long')
  new=new.replace(e['old'],e['new'],1)
 if old_words>max(15,int(len(body.split())*.30)):raise ValueError('Too much original text replaced')
 if difflib.SequenceMatcher(None,body,new,autojunk=False).ratio()<.65:raise ValueError('Rewrite too extensive')
 if new==body:raise ValueError('No actual change')
 return new+(sep+tail if sep else '')

class VLM:
 def __init__(self,root,cfg):self.root=Path(root);self.cfg=cfg;self.backend=None
 def load(self):
  if self.backend is None:
   import torch
   from experiments.qwen_curriculum.scene_prompting import LocalVisionPlanner
   torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.cuda.set_per_process_memory_fraction(min(32*1024**3/torch.cuda.get_device_properties(0).total_memory,.85))
   for path,digest in self.cfg['model_sha256'].items():
    if sha(path)!=digest:raise ValueError('Frozen VLM checkpoint changed')
   self.backend=LocalVisionPlanner(MODEL)
  return self.backend
 def ask(self,tasks,phase,max_tokens=480):
  # tasks: one self-contained conversation per pair/image group, never a contact-sheet verdict for multiple pairs.
  identity=dict(protocol=self.cfg['fingerprint'],phase=phase,max_tokens=max_tokens,tasks=tasks);key=common.fingerprint(identity);path=self.root/'inference'/phase/(key+'.json')
  if path.exists():
   cached=json.loads(path.read_text())
   if cached['identity']!=identity or cached['result_sha256']!=common.fingerprint({k:v for k,v in cached.items() if k!='result_sha256'}):raise ValueError('Inference cache changed')
   return cached['responses']
  atomic(path.with_suffix('.request.json'),dict(identity=identity,key=key,recovery='Text-only inference can be recomputed; no image generation'))
  backend=self.load();from PIL import Image
  messages=[]
  for task in tasks:
   content=[]
   for f in task['images']:
    if sha(f['path'])!=f['sha256']:raise ValueError('Inference image hash changed')
    with Image.open(f['path']) as im:
     img=im.convert('RGB');img.thumbnail((672,672))
    content.append(dict(type='image',image=img))
   content.append(dict(type='text',text=task['text']))
   messages.append([dict(role='system',content=[dict(type='text',text=task['system'])]),dict(role='user',content=content)])
  backend.processor.tokenizer.padding_side='left';inputs=backend.processor.apply_chat_template(messages,tokenize=True,padding=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to(backend.device)
  torch=backend.torch;started=time.monotonic()
  with torch.inference_mode(),torch.random.fork_rng(devices=[torch.cuda.current_device()]):
   torch.manual_seed(int(key[:8],16));ids=backend.model.generate(**inputs,max_new_tokens=max_tokens,do_sample=False,use_cache=True)
  responses=backend.processor.batch_decode(ids[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)
  if len(responses)!=len(tasks):raise RuntimeError('Lost per-pair response alignment')
  value=dict(identity=identity,responses=responses,elapsed_seconds=time.monotonic()-started,key=key);value['result_sha256']=common.fingerprint(value);atomic(path,value)
  return responses
 def close(self):
  if self.backend:self.backend.close();self.backend=None

def parse(raw):
 from experiments.qwen_curriculum.scene_prompting import parse_json
 return parse_json(raw)
def image(path,digest=None):return dict(path=str(path),sha256=digest or sha(path))
def prompt_task(row,system,text):return dict(system=system,text=text,images=[image(row['source_path'],row['source_sha256'])])
def config(root):
 from experiments.qwen_curriculum import night_diversity_policy as policy
 root=Path(root);filter_cfg=campaign.config_file(root/'filter_protocol.json')
 deps={str(Path(__file__).resolve()):sha(__file__)}
 from experiments.qwen_curriculum import scene_prompting
 deps[str(Path(scene_prompting.__file__).resolve())]=sha(scene_prompting.__file__)
 modelsha={str(f):sha(f) for f in sorted(MODEL.glob('*')) if f.is_file() and (f.suffix=='.safetensors' or f.name in ['config.json','preprocessor_config.json','tokenizer_config.json','chat_template.json'])}
 return campaign.frozen(root/'review_protocol.json',dict(version=1,filter_fingerprint=filter_cfg['fingerprint'],model_path=str(MODEL),model_sha256=modelsha,files_sha256=deps,
  audit_system=AUDIT,edit_system=EDIT,source_selection_system=SELECT,compose_system=COMPOSE,
  batch_size=4,max_tokens=480,decode='greedy',thumbnail=[672,672],max_text_attempts=2,max_source_candidates=8,
  edits=dict(max_edits=3,max_old_word_fraction=.30,min_old_word_allowance=15,max_replacement_words=50,min_text_similarity=.65),
  styles=[k for k in policy.STYLES if k!='mercury_vapor'],generation_enabled=False,canonical_unchanged=True,
  unknown_policy='Record uncertainty; no forced pass, no invented errors, no automatic image generation'))

def index_city(c,root,source):
 city=source['city'];directory=Path(source['source_path']).parent
 c.executescript('CREATE TABLE IF NOT EXISTS source_pool(city TEXT,place_id INTEGER,path TEXT,PRIMARY KEY(city,path)); CREATE INDEX IF NOT EXISTS source_pool_place ON source_pool(city,place_id); CREATE TABLE IF NOT EXISTS indexed_cities(city TEXT PRIMARY KEY,directory TEXT);')
 if c.execute('select 1 from indexed_cities where city=?',(city,)).fetchone():return
 with c:
  for item in os.scandir(directory):
   if not item.is_file() or not item.name.lower().endswith(('.jpg','.jpeg','.png')):continue
   fields=item.name.split('_')
   if len(fields)>2 and fields[0]==city and fields[1].isdigit():c.execute('insert or ignore into source_pool values(?,?,?)',(city,int(fields[1]),str(Path(item.path).resolve())))
  c.execute('insert into indexed_cities values(?,?)',(city,str(directory)))

def repair(c,root,vlm,limit=0):
 from experiments.qwen_curriculum import night_diversity_policy as policy
 n=0
 for item in c.execute("select ordinal,payload,review,review_status from pairs where repair_status='pending' order by ordinal"):
  if STOP or (Path(root)/'pause_requested').exists() or (limit and n>=limit):break
  check_resources();r=json.loads(item['payload']);evidence=json.loads(item['review']);result=None
  try:
   if item['review_status']=='minor_edit':
    task=prompt_task(r,EDIT,'Original full prompt: '+r['prompt']+'\nClear visually observed issues: '+dumps(evidence['issues']))
    error=None
    for attempt in range(2):
     t=copy.deepcopy(task)
     if error:t['text']+='\nPrevious output was invalid: '+error+'. Return a SMALL exact literal edit, not a rewrite.'
     raw=vlm.ask([t],'minor_edit_'+str(attempt),480)[0]
     try:
      v=parse(raw);modified=minor_edit(r['prompt'],v['edits']);result=dict(kind='minor_prompt_edit',original_prompt=r['prompt'],revised_prompt=modified,edits=v['edits'],reason=v.get('reason'),raw_response=raw,source_path=r['source_path'],source_sha256=r['source_sha256']);break
     except (KeyError,ValueError,TypeError) as e:error=str(e)
    if result is None:raise ValueError('Qwen small edit invalid after bounded retry: '+str(error))
   else:
    index_city(c,root,r);pool=[x[0] for x in c.execute('select path from source_pool where city=? and place_id=? order by path',(r['city'],int(r['place_id']))) if x[0]!=str(Path(r['source_path']).resolve())]
    if not pool:raise ValueError('No alternate real source for same place')
    if len(pool)>8:pool=[pool[round(i*(len(pool)-1)/7)] for i in range(8)]
    imgs=[image(f) for f in pool];imgs=[f for f in imgs if f['sha256']!=r['source_sha256']]
    if not imgs:raise ValueError('No distinct alternative source bytes')
    task=dict(system=SELECT,images=imgs,text='Each image is an independent source candidate. Indices in image order are '+str(list(range(len(imgs))))+'. Original rejected-source issue: '+dumps(evidence['issues']))
    raw=vlm.ask([task],'select_source',400)[0];selection=parse(raw);idx=selection.get('selected_index')
    if idx is None:raise ValueError('No suitable alternate source: '+str(selection.get('reason')))
    if type(idx) is not int or not 0<=idx<len(imgs):raise ValueError('Invalid selected index')
    selected=imgs[idx];filename=Path(selected['path']).name
    if not filename.startswith(f"{r['city']}_{int(r['place_id']):07d}_"):raise ValueError('Alternate source must be exact same city/place')
    task=dict(system=COMPOSE,images=[selected],text='Original style: '+r['style_id']+'. Allowed styles: '+', '.join(vlm.cfg['styles'])+'. This is the NEW real source. Source-selection evidence: '+str(selection.get('source_summary')))
    v=parse(vlm.ask([task],'compose_new_source',550)[0]);text=v.get('appearance_prompt')
    if not isinstance(text,str) or not 30<=len(text.split())<=230 or v.get('style_id') not in vlm.cfg['styles']:raise ValueError('Invalid new-source prompt')
    if any(x in text.lower() for x in ('cannot be augmented','no lighting mechanism can','remain as-is')):raise ValueError('New-source prompt refuses to edit')
    result=dict(kind='alternate_source_prompt',old_source_path=r['source_path'],old_source_sha256=r['source_sha256'],source_path=selected['path'],source_sha256=selected['sha256'],
     source_candidates=imgs,selected_index=idx,selection=selection,revised_prompt=text,original_prompt=r['prompt'],style_id=v['style_id'],reason=v.get('reason'),old_patch_correspondence_reused=False)
   state='prepared'
  except (ValueError,KeyError,TypeError,OSError) as e:result=dict(error=str(e),action='needs_followup_no_image_generation');state='unresolved'
  result.update(sample_id=r['sample_id'],city=r['city'],place_id=r['place_id'],author='Qwen3-VL-4B-Instruct',generation_enabled=False,eligible_for_training=False)
  c.execute('update pairs set repair_status=?,repair=? where ordinal=?',(state,dumps(result),item['ordinal']));c.commit();n+=1
 return n

def run(root,limit=0):
 global STOP
 root=Path(root);cfg=campaign.config_file(root/'review_protocol.json')
 for p,digest in cfg['files_sha256'].items():
  if sha(p)!=digest:raise ValueError('Frozen review code changed')
 if campaign.config_file(root/'filter_protocol.json')['fingerprint']!=cfg['filter_fingerprint']:raise ValueError('Filter/review protocol mismatch')
 c=connect(root)
 if c.execute('select count(*) from pairs where filter_status is null').fetchone()[0]:raise ValueError('Finish ALL first-pass rules before pairwise review')
 for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:globals().__setitem__('STOP',True))
 vlm=VLM(root,cfg);done=0;last=0
 with common_lock(root/'review.lock'):
  try:
   repair(c,root,vlm)
   while not STOP and not (root/'pause_requested').exists() and (not limit or done<limit):
    check_resources();batch=c.execute("select ordinal,payload from pairs where filter_status='pass' and review_status is null order by ordinal limit ?",(min(4,limit-done) if limit else 4,)).fetchall()
    if not batch:break
    tasks=[]
    for row in batch:
     r=json.loads(row['payload']);tasks.append(dict(system=AUDIT,images=[image(r['source_path'],r['source_sha256']),image(r['output_path'],r['output_sha256'])],text='Inspect this single source/generated pair. Prompt context (not evidence that an edit succeeded): '+r['prompt']))
    raw_list=vlm.ask(tasks,'pair_review',480);values=[]
    for row,task,raw in zip(batch,tasks,raw_list):
     error=None;value=None
     for attempt in range(2):
      try:value=parse(raw);state=derive_review(value);break
      except (ValueError,KeyError,TypeError) as e:
       error=str(e)
       if not attempt:
        retry=copy.deepcopy(task);retry['text']+='\nInvalid previous schema: '+error+'. Return requested JSON only.';raw=vlm.ask([retry],'review_schema_retry',480)[0]
     if value is None or error and attempt==1:
      # Successful second parse clears the error only via a valid state below.
      try:state=derive_review(value)
      except (ValueError,TypeError,KeyError):value=dict(verdict='unknown',issues=[],evidence='Invalid VLM response after bounded text retry',error=error);state='unknown'
     value={**value,'raw_response':raw,'reviewer':'Qwen3-VL-4B-Instruct','generation_enabled':False}
     values.append((state,dumps(value),'pending' if state in {'minor_edit','replace_source'} else 'not_needed' if state=='keep' else 'unknown',row['ordinal']))
    with c:c.executemany('update pairs set review_status=?,review=?,repair_status=? where ordinal=?',values)
    done+=len(batch);repair(c,root,vlm)
    if time.monotonic()-last>=20:print(dumps(status(root,'pair_review_and_prompt_only')),flush=True);last=time.monotonic()
   status(root,'pair_review_and_prompt_only')
  finally:vlm.close();c.close()

def export(root):
 root=Path(root);c=connect(root)
 for name,condition in [('visual_reviews.jsonl',"review_status is not null"),('prompt_changes.jsonl',"repair_status='prepared' and review_status='minor_edit'"),('source_replacements.jsonl',"repair_status='prepared' and review_status='replace_source'"),('needs_followup.jsonl',"review_status='unknown' or repair_status='unresolved'")]:
  tmp=root/(name+'.tmp')
  with tmp.open('w') as f:
   for row in c.execute('select sample_id,payload,review_status,review,repair_status,repair from pairs where '+condition+' order by ordinal'):
    r=json.loads(row['payload']);f.write(dumps(dict(sample_id=row['sample_id'],city=r['city'],place_id=r['place_id'],source_path=r['source_path'],output_path=r['output_path'],review_status=row['review_status'],review=json.loads(row['review']) if row['review'] else None,repair_status=row['repair_status'],repair=json.loads(row['repair']) if row['repair'] else None,generation_enabled=False,eligible_for_training=False))+'\n')
  tmp.replace(root/name)
 c.close();status(root)

def main():
 os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','review','export']);p.add_argument('--root',type=Path,required=True);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
 if a.phase=='prepare':print(dumps(config(a.root)))
 elif a.phase=='export':export(a.root)
 else:run(a.root,a.limit)
if __name__=='__main__':main()
