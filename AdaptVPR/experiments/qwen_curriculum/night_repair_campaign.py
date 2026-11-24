"""Bounded repair of rule-rejected night pairs: Qwen prose, image edit, both rules.

Old results are immutable. One generation on the original source, then at most
TWO distinct same-place alternate sources. Unknown input pairs are not selected.
Each GPU phase runs in a separate process; uncertain HTTP reservations fail closed.
"""
from __future__ import annotations
import argparse,collections,copy,json,os,signal,sqlite3,subprocess,sys,threading,time,shlex,tempfile
from pathlib import Path
from types import SimpleNamespace
from experiments.qwen_curriculum import common,campaign,run,balanced_night
from experiments.qwen_curriculum import night_cluster_worker as worker,night_cluster_resources as resources
from experiments.qwen_curriculum import night_full_review as review,night_diversity_policy as styles
from experiments.qwen_curriculum import night_content_audit as content,content_audit_v4 as legacy
from experiments.qwen_curriculum.night_full_filter import atomic,common_lock
ROOT=common.WORKSPACE_ROOT
DEFAULT=ROOT/'outputs/night_repair_v1'
NODE=ROOT/'outputs/night_original_no_mercury_v1/nodes'/os.environ.get('CCR_REPAIR_NODE','local')
CAL=ROOT/'outputs/maintenance/night_rules_87_20261113/dino_content_trial_v1/remote_results/calibration.json'
MODEL=review.MODEL
MAX_ROUND=2
SYSTEM='''You plan image-conditioned augmentation for visual place recognition. These are open-ended descriptions, not a fixed list of prompt templates. Read the actual photograph and improve an English NIGHT editing prompt. Preserve correct scene-specific content and style when feasible, but correct contradictions, daylight/dusk tendencies, implausible illumination and missing source prerequisites. Preserve permanent geometry, NOT original sunlight or crisp solar/tree-shadow patterns. Night must apply to the sky (including gaps between branches), unlit facades, foliage and ground, not just a black sky over a daylight scene. Keep useful detail, not pure black.
Distinguish building windows, vehicle windows, shop glazing, opaque sign backs and actual fixtures. No visible lamp is NOT a reason to refuse: weak natural sky light or plausible off-frame light may illuminate existing surfaces without adding lamps. Do not invent a shop, sign, window, building, luminous wall, glowing curb or tree-base fixture to fit a style. Plausible people/vehicle changes and wet/dry changes are permitted. Do not make a dry/wet or light-source contradiction. Do not demand preserved sharp solar shadows, mimic dusk, or refuse to edit.
Use only actual rule evidence. DINO outlier scores and peak windows are safety signals, NOT semantic proof of hallucination. Geometry-match failures can reflect darkness, weather or texture. Do not assert an invented defect as certain. No style category is automatically rejected, including shopfronts. Use shop lighting only where real shop glazing exists; changing a category name alone is not a semantic repair.
Return JSON with source_summary, prompt_issues (array of objects with quote, problem, correction), audit_response (string), style_id (allowed id), appearance_prompt (complete English NIGHT instruction, about 50-170 words). Make targeted repairs instead of gratuitous stylistic rewrites. For a NEW source, compose source-specific prose and never reuse old-source positions/structures. Your appearance_prompt is the actual generation instruction. Do not emit the fixed preservation appendix; the caller attaches it unchanged.'''
SELECT=review.SELECT+' You must select unused source bytes; do not choose a previously tried source. Prefer readable permanent landmarks and soft, uniform illumination.'


def dumps(v):return json.dumps(v,ensure_ascii=False,allow_nan=False)
def rows(p):return common.read_jsonl(Path(p))
def db(root):
 c=sqlite3.connect(Path(root)/'state.sqlite',timeout=60);c.row_factory=sqlite3.Row
 c.executescript('''PRAGMA journal_mode=WAL;PRAGMA synchronous=FULL;
 CREATE TABLE IF NOT EXISTS places(id TEXT PRIMARY KEY,ordinal INTEGER UNIQUE,city TEXT,place_id INTEGER,original TEXT,defect TEXT,status TEXT DEFAULT 'pending',round INTEGER DEFAULT 0);
 CREATE TABLE IF NOT EXISTS attempts(id TEXT,round INTEGER,source TEXT,planning TEXT,job TEXT,batch TEXT,result TEXT,PRIMARY KEY(id,round));
 CREATE TABLE IF NOT EXISTS batches(id TEXT PRIMARY KEY,phase TEXT DEFAULT 'planning',created REAL,node TEXT DEFAULT 'local');
 CREATE TABLE IF NOT EXISTS source_pool(city TEXT,place_id INTEGER,path TEXT UNIQUE);
 CREATE INDEX IF NOT EXISTS source_lookup ON source_pool(city,place_id);
 CREATE TABLE IF NOT EXISTS indexed(city TEXT PRIMARY KEY);
 CREATE TABLE IF NOT EXISTS policy_rechecks(id TEXT PRIMARY KEY,result TEXT);
 ''');return c

def active_backend():return worker.service_pid('ccr-night-local-backend.service')!=0

def validate_prompt(v,original_body,*,new_source=False):
 if not isinstance(v,dict):raise ValueError('JSON object required')
 for k in ('source_summary','audit_response','appearance_prompt'):
  if not isinstance(v.get(k),str) or len(v[k].strip())<8:raise ValueError('Missing '+k)
 if not isinstance(v.get('prompt_issues'),list):raise ValueError('Explicit prompt_issues required')
 for issue in v['prompt_issues']:
  if not isinstance(issue,dict) or any(not isinstance(issue.get(k),str) for k in ('quote','problem','correction')):raise ValueError('Invalid prompt issue')
 if v.get('style_id') not in styles.STYLES or v['style_id']=='mercury_vapor':raise ValueError('Select a source-supported non-mercury style')
 text=v['appearance_prompt'].strip()
 if not 30<=len(text.split())<=240:raise ValueError('Require 30..240 English prompt words')
 if '\n\nKeep this exact viewpoint' in text:raise ValueError('Do not duplicate fixed appendix')
 if any(s in text.lower() for s in ('cannot be augmented','remain as-is','no lighting mechanism can','mimicking dusk')):raise ValueError('Must actively edit into NIGHT')
 if not new_source and text==original_body.strip():raise ValueError('Rejected image needs a substantive prompt correction')
 return {k:v[k] for k in ('source_summary','prompt_issues','audit_response','style_id','appearance_prompt')}

def remaining_sources(original,candidates,tried):
 used_paths={str(Path(r['source_path']).resolve()) for r in tried};used_hashes={r['source_sha256'] for r in tried}
 prefix=f"{original['city']}_{int(original['place_id']):07d}_"
 out=[]
 for path in sorted(candidates):
  p=Path(path).resolve()
  if p.name.startswith(prefix) and str(p) not in used_paths:
   digest=common.file_sha256(p)
   if digest not in used_hashes:out.append(dict(source_path=str(p),source_sha256=digest));used_hashes.add(digest)
 return out

def next_status(round_number,passed):
 if type(round_number) is not int or not 0<=round_number<=MAX_ROUND:raise ValueError('Bounded source round 0..2 required')
 return ('passed',round_number) if passed else ('pending',round_number+1) if round_number<MAX_ROUND else ('exhausted',round_number)

def prepare(root,chunk):
 root=Path(root);root.mkdir(parents=True,exist_ok=True)
 if (root/'protocol.json').exists():return campaign.config_file(root/'protocol.json')
 if chunk<1 or chunk>256:raise ValueError('Batch 1..256 required')
 calibration=json.loads(CAL.read_text());content.validate_calibration(calibration)
 node=worker.load_node(NODE);resources.load_policy(NODE/'resource_policy.json',node)
 c=db(root);assert c.execute('select count(*) from places').fetchone()[0]==0
 source=sqlite3.connect('file:'+str(ROOT/'outputs/night_audit_v1/audit.sqlite')+'?mode=ro',uri=True);source.row_factory=sqlite3.Row;source.execute('BEGIN')
 total=source.execute('select count(*) from pairs').fetchone()[0];assert total==61984
 reject_ids=set();count=0
 second={r['sample_id']:r for r in rows(ROOT/'outputs/night_dino_audit_v1/audited_manifest.jsonl') if r['content_audit']['decision']=='reject'}
 assert len(second)==5242
 with c,(root/'inputs.jsonl').open('x') as f:
  for item in source.execute('select ordinal,sample_id,payload,filter_status,quality from pairs order by ordinal'):
   if item['filter_status']!='reject' and item['sample_id'] not in second:continue
   r=json.loads(item['payload']);sid=r['sample_id'];assert sid not in reject_ids;reject_ids.add(sid)
   defect=dict(first_pass=json.loads(item['quality']),content=second.get(sid),origin='first_pass_reject' if item['filter_status']=='reject' else 'second_pass_reject')
   assert not (item['filter_status']=='reject' and sid in second)
   state='policy_recheck' if sid in second and second[sid]['decision_source']=='prompt_policy' else 'pending'
   c.execute('insert into places(id,ordinal,city,place_id,original,defect,status) values(?,?,?,?,?,?,?)',(sid,item['ordinal'],r['city'],int(r['place_id']),dumps(r),dumps(defect),state))
   f.write(dumps(dict(original=r,defect=defect))+'\n');count+=1
 assert count==13788;source.close();c.close()
 from experiments.qwen_curriculum import scene_prompting,content_audit_v42,quality
 from experiments.qwen_curriculum import night_cluster
 modules=[night_cluster,sys.modules[__name__],common,run,campaign,balanced_night,worker,resources,review,scene_prompting,content,legacy,content_audit_v42,quality]
 deps={str(Path(m.__file__).resolve()):common.file_sha256(m.__file__) for m in modules};deps.update(node['files_sha256'])
 modelsha={str(p):common.file_sha256(p) for p in sorted(MODEL.glob('*')) if p.is_file() and (p.suffix=='.safetensors' or p.name in ('config.json','preprocessor_config.json','tokenizer_config.json','chat_template.json'))}
 return campaign.frozen(root/'protocol.json',dict(version=1,expected=count,first_rejects=8546,second_rejects=5242,input_sha256=common.file_sha256(root/'inputs.jsonl'),
  excluded='All original DINO Unknown and both-rules Pass; no automatic canonical replacement',max_alternate_sources=2,max_images_per_place=3,
  batch_size=chunk,vlm_batch_size=4,model_path=str(MODEL),model_sha256=modelsha,files_sha256=deps,
  calibration=str(CAL),calibration_file_sha256=common.file_sha256(CAL),service=node['service_identity'],quality=balanced_night.runtime().quality_definition(clip_device='cpu'),
  prompt_system=SYSTEM,select_system=SELECT,quarantined_variants=[],execution_nodes=['local','gpu2'],seed=42,generation_authorized=True,training=False,originals_immutable=True))

def config(root):
 cfg=campaign.config_file(Path(root)/'protocol.json')
 for p,h in cfg['files_sha256'].items():
  if common.file_sha256(p)!=h:raise ValueError('Frozen dependency changed: '+p)
 if common.file_sha256(cfg['calibration'])!=cfg['calibration_file_sha256']:raise ValueError('Calibration changed')
 return cfg

def progress(root,phase=None):
 c=db(root);counts=dict(c.execute('select status,count(*) from places group by status').fetchall());attempts=c.execute('select count(*) from attempts').fetchone()[0]
 images=c.execute('select count(*) from attempts where result is not null').fetchone()[0];c.close()
 value=dict(total=sum(counts.values()),places=counts,attempts_planned=attempts,images_audited=images,phase=phase,updated_unix=time.time(),canonical_replaced=False,training=False)
 atomic(Path(root)/'progress.json',value);print(dumps(value),flush=True);return value

def check_resources(root,phase):
 sample=worker.resource_sample(0,[Path(root)])
 # Resource waits do not consume source retries or image reservations.
 if sample['disk_free_gib']<20:return 'output disk reserve below 20 GiB'
 if active_backend():return 'shared image backend is busy'
 if sample['available_gib']<(8 if phase=='images' else 5):return 'waiting for available host RAM'
 total=float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True).strip())/1024
 if total-sample['gpu_used_gib']<(42 if phase=='images' else 24 if phase=='plan' else 3):return 'waiting for free GPU memory'
 return None

def new_batch(root,cfg,size,node='local'):
 c=db(root)
 unfinished=c.execute("select id from batches where phase!='done' and node=? order by created limit 1",(node,)).fetchone()
 if unfinished:c.close();return unfinished[0]
 items=c.execute("select id,round from places where status='pending' order by round,ordinal limit ?",(size,)).fetchall()
 if not items:c.close();return None
 bid='b_'+common.fingerprint([(r['id'],r['round']) for r in items])[:20]
 with c:
  c.execute('insert into batches(id,created,node) values(?,?,?)',(bid,time.time(),node))
  for r in items:
   c.execute('insert into attempts(id,round,batch) values(?,?,?)',(r['id'],r['round'],bid));c.execute("update places set status='working' where id=?",(r['id'],))
 c.close();(Path(root)/'batches'/bid).mkdir(parents=True,exist_ok=True);return bid

def source_pool(c,r):
 city=r['city']
 if not c.execute('select 1 from indexed where city=?',(city,)).fetchone():
  with c:
   for p in sorted(Path(r['source_path']).parent.iterdir()):
    bits=p.name.split('_')
    if p.is_file() and len(bits)>2 and bits[0]==city and bits[1].isdigit() and p.suffix.lower() in {'.jpg','.jpeg','.png'}:
     c.execute('insert or ignore into source_pool values(?,?,?)',(city,int(bits[1]),str(p.resolve())))
   c.execute('insert into indexed values(?)',(city,))
 return [r[0] for r in c.execute('select path from source_pool where city=? and place_id=? order by path',(city,int(r['place_id'])))]

def select_source(c,vlm,place,attempt):
 original=json.loads(place['original']);roundno=attempt['round']
 if roundno==0:return {k:copy.deepcopy(original[k]) for k in ('source_path','source_sha256','source_dimensions','source_sample_id')},dict(kind='original_source')
 tried=[dict(source_path=original['source_path'],source_sha256=original['source_sha256'])]
 for a in c.execute('select source from attempts where id=? and source is not null',(place['id'],)):tried.append(json.loads(a[0]))
 candidates=remaining_sources(original,source_pool(c,original),tried)
 if not candidates:return None,dict(error='no_distinct_same_place_source_remaining')
 if len(candidates)>8:candidates=[candidates[round(i*(len(candidates)-1)/7)] for i in range(8)]
 task=dict(system=SELECT,images=[review.image(r['source_path'],r['source_sha256']) for r in candidates],
  text='Candidate indices in image order: '+str(list(range(len(candidates))))+'. Same city/place: '+original['city']+'/'+str(original['place_id'])+'. Prior rule evidence (not semantic truth): '+place['defect'])
 for retry in range(2):
  response=review.parse(vlm.ask([task],'source_selection_'+str(retry),500)[0]);idx=response.get('selected_index')
  if idx is None:return None,dict(error='vlm_no_suitable_distinct_source',selection=response)
  if type(idx) is int and 0<=idx<len(candidates):break
  task['text']+=' Invalid index; use one supplied integer or null.'
 else:raise ValueError('Invalid source selection schema')
 chosen=candidates[idx]
 from PIL import Image
 with Image.open(chosen['source_path']) as im:chosen['source_dimensions']=list(im.size)
 chosen['source_sample_id']='qnsource_repair_'+chosen['source_sha256'][:24]
 return chosen,dict(kind='alternate_source',selection=response,candidates=candidates,old_correspondence_reused=False)

def make_job(original,source,plan,roundno,cfg):
 key=common.fingerprint(dict(parent=original['sample_id'],source=source['source_sha256'],round=roundno,plan=plan,protocol=cfg['fingerprint']))
 seed=original['seed'] if roundno==0 else common.candidate_seed(cfg['seed'],key,roundno)
 body=plan['appearance_prompt'].strip();text=body+'\n\n'+styles.CONTRACT
 j=dict(city=original['city'],place_id=int(original['place_id']),**source,sample_id='qnightrepair_'+key[:24],parent_sample_id=original['sample_id'],
  repair_source_round=roundno,condition='night',style_id=plan['style_id'],prompt_policy='qwen_rule_repair_v1',prompt=text,negative_prompt='',prompt_variant=plan['style_id'],
  style_name='Qwen repaired: '+plan['style_id'],seed=seed,pair_id=key,with_transient=False,transient_policy='integrated plausible changes permitted',
  scene_plan_sha256=common.fingerprint(plan),appearance=dict(style_id=plan['style_id'],weather='night',appearance_prompt=body),scene_summary=plan['source_summary'],
  protected_regions=[dict(description='Preserve permanent geometry of exact current source; policy only, not a semantic mask.',box=[0.,0.,1.,1.],usable_as_correspondence_or_semantic_evidence=False)],
  authorized_transient=None,planned_region_is_verified_mask=False,old_source_correspondence_reused=False,prompt_author='Qwen3-VL-4B-Instruct')
 j['record_sha256']=common.fingerprint(j);return j

def plan_batch(root,bid):
 cfg=config(root);directory=Path(root)/'batches'/bid;c=db(root)
 vlm=review.VLM(directory,dict(fingerprint=cfg['fingerprint'],model_sha256=cfg['model_sha256']))
 try:
  items=c.execute('select * from attempts where batch=? and job is null and result is null order by id',(bid,)).fetchall()
  tasks=[];pending=[]
  def flush():
   if not tasks:return
   paths=[directory/('prompt_raw_'+common.fingerprint([x[0]['id'],x[1]['round']])+'.json') for x in pending]
   rawlist=[json.loads(p.read_text())['response'] if p.exists() else None for p in paths]
   missing=[i for i,v in enumerate(rawlist) if v is None]
   if missing:
    generated=vlm.ask([tasks[i] for i in missing],'prompt_revision',950)
    for i,raw in zip(missing,generated,strict=True):
     rawlist[i]=raw;atomic(paths[i],dict(response=raw,task_sha256=common.fingerprint(tasks[i])))
   for i,p in enumerate(paths):
    if json.loads(p.read_text())['task_sha256']!=common.fingerprint(tasks[i]):raise ValueError('Saved prompt response differs from task')
   for item,task,raw in zip(pending,tasks,rawlist,strict=True):
    place,attempt,source,selection=item;original=json.loads(place['original']);body=original['prompt'].partition('\n\nKeep this exact viewpoint')[0]
    for retry in range(2):
     try:plan=validate_prompt(review.parse(raw),body,new_source=attempt['round']>0);break
     except (ValueError,TypeError,KeyError) as e:
      if retry:raise ValueError('Qwen prompt invalid after bounded retry: '+str(e))
      t=copy.deepcopy(task);t['text']+='\nOutput validation error: '+str(e)+'. Correct JSON only.';raw=vlm.ask([t],'prompt_schema_retry',950)[0]
    plan.update(raw_response=raw,source_selection=selection,original_prompt=original['prompt'],
     prompt_body_author='Qwen3-VL-4B-Instruct',fixed_appendix=styles.CONTRACT)
    job=make_job(original,source,plan,attempt['round'],cfg)
    with c:c.execute('update attempts set planning=?,job=? where id=? and round=?',(dumps(plan),dumps(job),place['id'],attempt['round']))
   tasks.clear();pending.clear()
  for attempt in items:
   if (Path(root)/'pause_requested').exists():break
   place=c.execute('select * from places where id=?',(attempt['id'],)).fetchone();original=json.loads(place['original'])
   if attempt['source']:
    source=json.loads(attempt['source']);selection=json.loads(attempt['planning'])['source_selection']
   else:
    source,selection=select_source(c,vlm,place,attempt)
    if source is None:
     with c:
      c.execute('update attempts set result=? where id=? and round=?',(dumps(dict(decision='no_source',evidence=selection)),place['id'],attempt['round']))
      c.execute("update places set status='no_source' where id=?",(place['id'],))
     continue
    with c:c.execute('update attempts set source=?,planning=? where id=? and round=?',(dumps(source),dumps(dict(source_selection=selection)),place['id'],attempt['round']))
   for f,digest in [(source['source_path'],source['source_sha256']),(original['output_path'],original['output_sha256'])]:
    if common.file_sha256(f)!=digest:raise ValueError('Source or original generated bytes changed')
   history=[json.loads(r[0]) for r in c.execute('select result from attempts where id=? and result is not null order by round',(place['id'],))]
   text='Original prompt (context, not proof): '+original['prompt']+'\nOriginal rule rejection: '+place['defect']+'\nPrior retry rule outcomes: '+dumps(history)+'\nAllowed styles: '+', '.join(k for k in styles.STYLES if k!='mercury_vapor')
   images=[review.image(source['source_path'],source['source_sha256'])]
   if attempt['round']==0:
    images.append(review.image(original['output_path'],original['output_sha256']));text+='\nImage 1 is real source, image 2 is the ORIGINAL failed edit. Correct prompt based on both image evidence and text logic.'
   else:text+='\nThis is a NEW real source of the same place. Compose for its own structures and viewpoint, not the previous photo. Re-check contradictions, daylight tendencies and common-sense lighting just as for the first source.'
   tasks.append(dict(system=SYSTEM,images=images,text=text));pending.append((place,attempt,source,selection))
   if len(tasks)==cfg['vlm_batch_size']:flush()
  flush()
 finally:vlm.close();c.close()
 if (Path(root)/'pause_requested').exists():return
 freeze_batch(root,bid,cfg)

def freeze_batch(root,bid,cfg):
 d=Path(root)/'batches'/bid;c=db(root);jobs=[json.loads(r[0]) for r in c.execute('select job from attempts where batch=? and job is not null order by id',(bid,))]
 if not jobs:
  with c:c.execute("update batches set phase='done' where id=?",(bid,))
  c.close();return
 jp=d/'plan.jsonl'
 if jp.exists():assert rows(jp)==jobs
 else:common.write_jsonl(jp,jobs)
 campaign.frozen(d/'plan_config.json',dict(version=1,stage='qwen_rule_repair_v1',num_jobs=len(jobs),request_budget=len(jobs),domain_counts={'night':len(jobs)},
  image_root=str(d/'images'),service=cfg['service'],quality=cfg['quality'],plan_sha256=common.file_sha256(jp),files_sha256=cfg['files_sha256'],campaign_fingerprint=cfg['fingerprint']))
 with c:c.execute("update batches set phase='images' where id=?",(bid,))
 c.close()

def load_image_plan(directory):
 directory=Path(directory);name='remote_config.json' if os.environ.get('CCR_REPAIR_NODE')=='gpu2' else 'plan_config.json';cfg=campaign.config_file(directory/name);jobs=rows(directory/'plan.jsonl')
 if common.file_sha256(directory/'plan.jsonl')!=cfg['plan_sha256'] or len(jobs)!=cfg['num_jobs']:raise ValueError('Frozen image plan changed')
 if cfg['quality']!=balanced_night.runtime().quality_definition(clip_device='cpu'):raise ValueError('Quality verifier differs from frozen plan')
 for p,h in cfg['files_sha256'].items():
  if common.file_sha256(p)!=h:raise ValueError('Frozen dependency changed: '+p)
 for job in jobs:
  if job['record_sha256']!=common.fingerprint({k:v for k,v in job.items() if k!='record_sha256'}):raise ValueError('Job identity changed')
  run.validate_image(job,'source_path','source_sha256')
 return cfg,jobs

def image_batch(root,bid):
 d=Path(root)/'batches'/bid;cfg,jobs=load_image_plan(d);engine=balanced_night.runtime().engine();engine.load_plan=load_image_plan;resources.protect_reservations(engine)
 _,finished,_=engine._read_state(d,jobs,cfg)
 if len(finished)==len(jobs):return
 node=worker.load_node(NODE);policy=resources.load_policy(NODE/'resource_policy.json',node)
 total=float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True).strip())/1024;resources.install(policy,total)
 if worker.service_pid(node['backend_unit'],node.get('backend_scope','user')):raise resources.ResourceWait('shared image backend is busy')
 backend=worker.Backend(d,node);pause=threading.Event();guard=worker.ResourceGuard(d,node,backend,pause)
 for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:pause.set())
 def observe():
  while not pause.wait(2):
   if (Path(root)/'pause_requested').exists():pause.set();return
 watcher=threading.Thread(target=observe,daemon=True);watcher.start()
 with run.run_lock(d):
  try:
   backend.start(guard.sample());guard.thread.start();backend.health(pause)
   if not (Path(root)/'pause_requested').exists():(d/'pause_requested').unlink(missing_ok=True)
   with worker.commit_pause(engine,pause,d/'pause_requested'):
    result=engine.generate(SimpleNamespace(output_dir=d,domain='night',max_calls=0,qwen_url='http://127.0.0.1:8001/generate',request_timeout=1800,device='cpu'),locked=True)
   if guard.error:raise RuntimeError(guard.error)
   common.write_json(d/'image_runtime.json',dict(saved=result['saved'],peaks=guard.peaks))
  finally:
   pause.set()
   if guard.thread.ident:guard.close()
   backend.stop()


def audit_batch(root,bid):
 import torch,numpy as np,cv2
 from PIL import Image
 cfg=config(root);d=Path(root)/'batches'/bid;c=db(root);torch.set_num_threads(4);torch.set_num_interop_threads(2)
 if active_backend():raise RuntimeError('Release image backend before rules audit')
 results={r['sample_id']:r for r in rows(d/'results.jsonl')};verifier=None;provider=None;cal=json.loads(Path(cfg['calibration']).read_text())
 content.validate_calibration(cal)
 for item in c.execute('select * from attempts where batch=? and job is not null and result is null order by id',(bid,)).fetchall():
  if (Path(root)/'pause_requested').exists():break
  job=json.loads(item['job']);r=results.get(job['sample_id'])
  if r is None:raise ValueError('Generation not finished for '+job['sample_id'])
  for field,sha in [('source_path','source_sha256'),('output_path','output_sha256')]:
   if common.file_sha256(r[field])!=r[sha]:raise ValueError('Exact audit pair bytes changed')
  if verifier is None:verifier=balanced_night.runtime().CandidateVerifier(device='cuda',clip_device='cpu')
  seed=int(common.fingerprint(r['sample_id'])[:8],16)%2147483647;torch.manual_seed(seed);np.random.seed(seed);cv2.setRNGSeed(seed)
  with Image.open(r['source_path']) as im:a=im.convert('RGB')
  with Image.open(r['output_path']) as im:b=im.convert('RGB')
  with torch.inference_mode():base=verifier.evaluate(a,b,'night')
  if base['passed']:
   if provider is None:provider=content.load_provider(SimpleNamespace(dino_repo='/home/admin123/.cache/torch/hub/facebookresearch_dinov2_main',dino_checkpoint='/home/admin123/github/checkpoints/torch-hub/dinov2_vitb14_pretrain.pth',threads=4,device='cuda'),cal)
   verdict=evaluate_content(r,dict(sample_id=r['sample_id'],source_sha256=r['source_sha256'],output_sha256=r['output_sha256'],current_rule=base),cal,provider)
  else:verdict=dict(content_audit=dict(decision='not_run_base_reject',reasons=['base_geometry_weather_not_passed']),safety_screen_passed=False)
  passed=base['passed'] and verdict['safety_screen_passed'];status,nextround=next_status(item['round'],passed)
  evidence=dict(decision='pass' if passed else 'reject' if not base['passed'] else verdict['content_audit']['decision'],base=base,content=verdict,generated=r,training_approval=False)
  with c:
   c.execute('update attempts set result=? where id=? and round=?',(dumps(evidence),item['id'],item['round']))
   c.execute('update places set status=?,round=? where id=?',(status,nextround,item['id']))
  print(dumps(dict(place=item['id'],round=item['round'],decision=evidence['decision'],next=status)),flush=True)
 if not c.execute('select 1 from attempts where batch=? and job is not null and result is null',(bid,)).fetchone():
  with c:c.execute("update batches set phase='done' where id=?",(bid,))
 c.close()

def export(root):
 c=db(root)
 with (Path(root)/'accepted_replacements.jsonl').open('w') as f,(Path(root)/'unresolved_places.jsonl').open('w') as bad:
  for p in c.execute('select * from places order by ordinal'):
   if p['status']=='passed':
    a=c.execute('select result from attempts where id=? and round=?',(p['id'],p['round'])).fetchone();e=json.loads(a[0]);f.write(dumps({**e['generated'],'original_sample_id':p['id'],'source_round':p['round'],'repair_audit':{k:v for k,v in e.items() if k!='generated'},'eligible_for_training':False,'canonical_replaced':False})+'\n')
   elif p['status'] in {'no_source','exhausted','unknown_recheck'}:bad.write(dumps(dict(original_sample_id=p['id'],city=p['city'],place_id=p['place_id'],status=p['status'],source_round=p['round']))+'\n')
 c.close();return progress(root,'export')

def evaluate_content(row,evidence,calibration,provider):
 """Same calibrated classifier, explicitly without category quarantine."""
 from experiments.qwen_curriculum import content_audit_v42 as classifier
 content.validate_calibration(calibration)
 for path,key in [('source_path','source_sha256'),('output_path','output_sha256')]:
  if legacy.sha256(row[path])!=row[key]:raise ValueError('Content pair integrity failed')
 bound=content.attach_base(row,evidence)
 signals=legacy.pair_signals(bound,provider,calibration['feature_pipeline']['side'],calibration['feature_pipeline']['search_radius'])
 result=classifier.audit_pair_v42(bound,signals,calibration,quarantine_variants=())
 return dict(sample_id=row['sample_id'],content_audit=result,safety_screen_passed=result['decision']=='pass',decision_source='local_content',
  peak_region=content.peak_region(signals,calibration),peak_region_is_semantic_defect=False,eligible_for_training=False,quarantined_variants=[])


def recheck_policy(root):
 """A deleted category ban must not force a new image for an already good pair."""
 cfg=config(root);c=db(root);cal=json.loads(Path(cfg['calibration']).read_text())
 args=SimpleNamespace(dino_repo='/home/admin123/.cache/torch/hub/facebookresearch_dinov2_main',dino_checkpoint='/home/admin123/github/checkpoints/torch-hub/dinov2_vitb14_pretrain.pth',threads=4,device='cuda')
 provider=content.load_provider(args,cal)
 for p in c.execute("select * from places where status='policy_recheck' order by ordinal limit 256").fetchall():
  if (Path(root)/'pause_requested').exists():break
  original=json.loads(p['original']);defect=json.loads(p['defect']);q=defect['first_pass']
  evidence=dict(sample_id=p['id'],source_sha256=original['source_sha256'],output_sha256=original['output_sha256'],current_rule=q)
  result=evaluate_content(original,evidence,cal,provider);decision=result['content_audit']['decision']
  status={'pass':'passed_original','reject':'pending','unknown':'unknown_recheck'}[decision]
  defect['no_quarantine_recheck']=result
  with c:
   c.execute('insert into policy_rechecks values(?,?)',(p['id'],dumps(result)))
   c.execute('update places set status=?,defect=? where id=?',(status,dumps(defect),p['id']))
  print(dumps(dict(policy_recheck=p['id'],decision=decision,next=status)),flush=True)
 c.close()


def remote_command(args,**kwargs):
 return subprocess.run(['ssh','-o','ConnectTimeout=10','ccr-gpu2',shlex.join(list(map(str,args)))],text=True,capture_output=True,check=True,**kwargs).stdout.strip()


def remote_unit(bid):return 'ccr-night-repair-'+bid+'.service'


def deploy_remote(root,bid):
 """Explicit node execution profile, identical frozen jobs, exact source copies."""
 from experiments.qwen_curriculum import night_cluster as cluster
 root=Path(root);d=root/'batches'/bid;cfg=campaign.config_file(d/'plan_config.json')
 node=worker.read_sealed(ROOT/'outputs/night_original_no_mercury_v1/nodes/gpu2/node.json')
 remote_cfg={k:v for k,v in cfg.items() if k!='fingerprint'}
 remote_cfg.update(service=node['service_identity'],execution_node='gpu2',original_local_config_sha256=common.file_sha256(d/'plan_config.json'))
 campaign.frozen(d/'remote_config.json',remote_cfg)
 files=[d/'plan.jsonl',d/'plan_config.json',d/'remote_config.json',ROOT/'outputs/night_original_no_mercury_v1/nodes/gpu2/node.json',ROOT/'outputs/night_original_no_mercury_v1/nodes/gpu2/resource_policy.json']
 files+=list(map(Path,cfg['files_sha256']));files += [Path(j['source_path']) for j in rows(d/'plan.jsonl')]
 (root/'deployment').mkdir(exist_ok=True);cluster.rsync_files(root,files)
 command=['systemd-run','--user','--unit='+remote_unit(bid),'--description=CCR-repair-gpu2-'+bid,'--property=Type=exec',
  '--property=MemoryMax=2G','--property=MemorySwapMax=0','--property=CPUQuota=200%','--property=TimeoutStopSec=1900','--property=KillMode=mixed',
  '--property=StandardOutput=append:'+str(d/'remote.log'),'--property=StandardError=append:'+str(d/'remote.log'),
  '--working-directory='+str(ROOT),'--setenv=CCR_REPAIR_NODE=gpu2','--setenv=PYTHONPATH='+str(ROOT/'AdaptVPR'),
  '--setenv=PYTHONDONTWRITEBYTECODE=1','--setenv=PYTHONUNBUFFERED=1','--setenv=OMP_NUM_THREADS=2','--setenv=OPENBLAS_NUM_THREADS=2',
  '/home/epic/anaconda3/envs/ccr-night/bin/python','-B','-m',__package__+'.night_repair_campaign','worker','--root',str(root),'--batch',bid,'--step','images','--execute']
 info=remote_command(['systemctl','--user','show',remote_unit(bid),'--property=LoadState','--property=ActiveState','--property=ExecMainStatus'])
 if 'ActiveState=active' in info:return
 if 'LoadState=not-found' not in info:
  # Only recognized resource waits may restart; uncertain HTTP never auto-retries.
  if 'ExecMainStatus=75' not in info:raise RuntimeError('Existing remote unit needs explicit recovery audit: '+info)
  remote_command(['systemctl','--user','restart',remote_unit(bid)]);return
 remote_command(command)


def poll_remote(root,bid):
 root=Path(root);d=root/'batches'/bid
 info=remote_command(['systemctl','--user','show',remote_unit(bid),'--property=ActiveState','--property=ExecMainStatus','--property=LoadState'])
 if 'LoadState=not-found' in info:deploy_remote(root,bid);return False
 if 'ActiveState=active' in info:return False
 if 'ExecMainStatus=75' in info:
  remote_command(['systemctl','--user','restart',remote_unit(bid)]);return False
 if 'ExecMainStatus=0' not in info:raise RuntimeError('Remote repair worker stopped; investigate without duplicating POST: '+info)
 subprocess.run(['rsync','-a','--partial','ccr-gpu2:'+str(d)+'/',str(d)+'/'],check=True)
 cfg,jobs=load_image_plan(d);summary=json.loads((d/'summary.json').read_text())
 if summary['state']!='domain_complete' or summary['saved']!=len(jobs):raise ValueError('Remote did not finish frozen budget')
 results=rows(d/'results.jsonl');by_id={j['sample_id']:j for j in jobs}
 if len(results)!=len(jobs):raise ValueError('Remote result cardinality mismatch')
 for r in results:
  j=by_id[r['sample_id']]
  for k in ('source_path','source_sha256','prompt','seed'):
   if r[k]!=j[k]:raise ValueError('Remote pair/prose/seed mismatch')
  for k in ('output','raw_output'):
   if common.file_sha256(r[k+'_path'])!=r[k+'_sha256']:raise ValueError('Remote output copy hash mismatch')
 atomic(d/'primary_received.json',dict(node='gpu2',images=len(results),all_output_hashes_verified=True))
 c=db(root)
 with c:c.execute("update batches set phase='audit' where id=?",(bid,))
 c.close();return True


def child_step(root,bid,step):
 progress(root,step)
 cmd=[sys.executable,'-B','-m',__package__+'.night_repair_campaign','worker','--root',str(root),'--batch',bid,'--step',step,'--execute']
 result=subprocess.run(cmd,env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','ADAPTVPR_CLIP_LOCAL_FILES_ONLY':'1'})
 if (Path(root)/'pause_requested').exists():return False
 if result.returncode:
  waitfile=Path(root)/'batches'/bid/'resource_wait.json'
  if waitfile.exists():waitfile.unlink();return False
  raise RuntimeError('Repair worker failed; committed progress retained. No automatic duplicate POST.')
 return True


def orchestrate(root,once=False):
 root=Path(root);cfg=config(root)
 with common_lock(root/'controller.lock'):
  while not (root/'pause_requested').exists():
   # The remote GPU generates while the primary plans, audits or generates its own batch.
   c=db(root);seen_remote=c.execute("select count(*) from batches where node='gpu2'").fetchone()[0];c.close()
   remote_id=new_batch(root,cfg,4 if seen_remote==0 else cfg['batch_size'],'gpu2')
   if remote_id:
    c=db(root);phase=c.execute('select phase from batches where id=?',(remote_id,)).fetchone()[0];c.close()
    if phase=='planning':
     reason=check_resources(root,'plan')
     if not reason:
      if not child_step(root,remote_id,'plan'):continue
      c=db(root);phase=c.execute('select phase from batches where id=?',(remote_id,)).fetchone()[0];c.close()
      if phase=='images':deploy_remote(root,remote_id)
    elif phase=='images':poll_remote(root,remote_id)
    elif phase=='audit':
     if child_step(root,remote_id,'audit'):export(root)
     continue
   c=db(root);policy_count=c.execute("select count(*) from places where status='policy_recheck'").fetchone()[0];c.close()
   if policy_count and not check_resources(root,'audit'):
    if child_step(root,'policy','policy'):export(root)
    continue
   c=db(root);seen_local=c.execute("select count(*) from batches where node='local'").fetchone()[0];c.close()
   local_id=new_batch(root,cfg,4 if seen_local==0 else cfg['batch_size'],'local')
   if local_id:
    c=db(root);phase=c.execute('select phase from batches where id=?',(local_id,)).fetchone()[0];c.close()
    step={'planning':'plan','images':'images','audit':'audit'}.get(phase)
    if step:
     reason=check_resources(root,step)
     if reason:
      atomic(root/'waiting.json',dict(reason=reason,phase=step,batch=local_id,retry_seconds=60));time.sleep(30);continue
     if not child_step(root,local_id,step):continue
     if step=='images':
      summary=json.loads((root/'batches'/local_id/'summary.json').read_text())
      if summary['saved']!=summary['planned']:raise RuntimeError('Incomplete local images; resume same reservations')
      c=db(root)
      with c:c.execute("update batches set phase='audit' where id=?",(local_id,))
      c.close()
     if step=='audit':
      export(root)
      if once:return
     continue
   if not remote_id and not policy_count:
    export(root);atomic(root/'complete.json',dict(complete=True,updated_unix=time.time(),canonical_replaced=False));return
   progress(root,'remote_generation');time.sleep(20)
  # A pause completes active image, then releases both backends, preserving reservations.
  try:remote_command(['/home/epic/anaconda3/envs/ccr-night/bin/python','-c','from pathlib import Path; Path('+repr(str(root/'pause_requested'))+').touch()'])
  except subprocess.SubprocessError:pass
  progress(root,'paused')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','run','worker','status','pause','resume','export']);p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--batch');p.add_argument('--step',choices=['plan','images','audit','policy']);p.add_argument('--execute',action='store_true');p.add_argument('--once',action='store_true');a=p.parse_args();root=a.root.resolve()
 if a.phase=='prepare':print(dumps(prepare(root,a.batch_size)));return
 if a.phase=='status':progress(root);return
 if a.phase=='pause':
  atomic(root/'pause_requested',dict(requested_unix=time.time()))
  remote_command(['/home/epic/anaconda3/envs/ccr-night/bin/python','-c','from pathlib import Path; Path('+repr(str(root/'pause_requested'))+').touch()']);return
 if a.phase=='export':export(root);return
 if not a.execute:raise ValueError('Explicit --execute required')
 if a.phase=='resume':
  (root/'pause_requested').unlink(missing_ok=True)
  remote_command(['/home/epic/anaconda3/envs/ccr-night/bin/python','-c','from pathlib import Path; Path('+repr(str(root/'pause_requested'))+').unlink(missing_ok=True)'])
 if a.phase in {'run','resume'}:orchestrate(root,a.once);return
 if not a.batch or not a.step:raise ValueError('Worker needs frozen batch and step')
 try:{'plan':plan_batch,'images':image_batch,'audit':audit_batch,'policy':lambda root,bid:recheck_policy(root)}[a.step](root,a.batch)
 except worker.PauseRequested:
  if (root/'pause_requested').exists():return
  atomic(root/'batches'/a.batch/'resource_wait.json',dict(error='resource pause after committed image'));raise SystemExit(75)
 except Exception as exc:
  if resources.recoverable(exc):atomic(root/'batches'/a.batch/'resource_wait.json',dict(error=str(exc)));raise SystemExit(75)
  raise
if __name__=='__main__':main()
