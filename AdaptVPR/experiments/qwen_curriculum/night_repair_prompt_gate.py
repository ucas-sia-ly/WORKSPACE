"""Versioned source-grounded final Prompt pass before any new image request.

The original campaign and its protocol remain immutable. This adapter records a
separate amendment, preserves drafts, and freezes the amendment code with each
new image plan. Already dispatched plans and all image retry budgets are intact.
"""
import copy,json,re,subprocess,sys
from pathlib import Path
from experiments.qwen_curriculum import night_repair_campaign as repair
from experiments.qwen_curriculum import campaign,common,night_full_review as review
from experiments.qwen_curriculum.night_full_filter import atomic

SYSTEM='''You plan image-conditioned augmentation for visual place recognition. These are open-ended descriptions, not a fixed list of prompt templates.
The image is a REAL DAYTIME SOURCE, NOT a generated night result. Write a NIGHT edit of this real source. The draft may be wrong. Keep its valid style and details; correct mistakes with small changes where possible.
Night must replace daylight on every surface and visible sky, including gaps between branches. Keep permanent geometry and readable landmarks, replace solar shadows. Do not request daylight or twilight. Opaque doors, walls, sign backs, curbs and tree trunks reflect light; they never emit light. Light may transmit through real glazing or come from existing lamps or OFF-FRAME lamps. Daytime lamps being off does not prevent switching them on. No visible lamps does not prevent off-frame illumination. Do not invent shops, signs, windows or fixtures. Plausible people, vehicle and pavement wetness changes are allowed, without contradictions.
Return ONLY valid JSON with exactly these three string fields:
{"source_summary":"visible permanent objects in the REAL source, one sentence", "style_id":"one supplied non-mercury id", "appearance_prompt":"complete English NIGHT editing instruction, 60-140 words"}
Do not explain, criticize the photo, or add a list. Do not include the fixed preservation appendix. Use positive instructions for the night result.'''

# These checks reject explicit wrong objectives, not brightness, colour names,
# or semantically ambiguous photographs. Qwen performs the source-based review.
BAD_OBJECTIVES=(
 r'\bpreserve (?:the )?original daylight scene\b',
 r'\bscene (?:remains?|stays?) daylight\b',
 r'\bno night (?:illumination|lighting|edit)\b',
 r'\bsky (?:remains?|stays?|should (?:remain|stay)|must (?:remain|stay)) (?:a )?bright blue\b',
 r'\btransform (?:the )?scene into (?:a )?twilight\b',
 r'\bnot full night darkness\b',
 r'\b(?:metal|opaque|solid) door (?:itself )?emit(?:s|ting)?\b',
 r'\b(?:interior|inside) of (?:the )?(?:metal|opaque|solid) door emit(?:s|ting)?\b',
 r'\b(?:render|set) (?:the )?scene in (?:soft )?twilight\b',
 r'\bno lighting mechanism can\b',
 r'\bcannot be (?:augmented|edited|transformed)\b',
)

def validate(v):
 if not isinstance(v,dict):raise ValueError('JSON object required')
 result=repair.validate_prompt({**v,'prompt_issues':[],
  'audit_response':'Final NIGHT composition from the real source only; earlier paired diagnosis retained separately.'},'',new_source=True)
 text=result['appearance_prompt'].lower()
 for pattern in BAD_OBJECTIVES:
  if re.search(pattern,text):raise ValueError('Explicit non-night or opaque-emission objective: '+pattern)
 if not re.search(r'\b(?:night|nighttime|night-time)\b',text):raise ValueError('Explicit NIGHT objective required')
 return result

def amendment(root):
 cfg=repair.config(root)
 value=campaign.frozen(Path(root)/'prompt_gate_protocol_v2.json',dict(
  version=2,parent_protocol=cfg['fingerprint'],code_path=str(Path(__file__).resolve()),
  code_sha256=common.file_sha256(__file__),system=SYSTEM,forbidden_objectives=list(BAD_OBJECTIVES),
  source_only_final_pass=True,paired_diagnosis_retained=True,prior_dispatched_plans_unchanged=True,
  text_attempts=2,max_new_tokens=450,vlm_batch_size=4,max_images_per_place=3,
  max_alternate_sources=2,quarantined_variants=[]))
 if value['code_sha256']!=common.file_sha256(__file__):raise ValueError('Frozen Prompt amendment code changed')
 return value

def prepare_task(place,job,plan):
 defect=json.loads(place['defect'])
 # Scores are context, never a claim of a semantic defect. Avoid repeating huge
 # geometric matrices and generated-image metadata in the composition prompt.
 first=defect.get('first_pass',{})
 content=defect.get('content') or {}
 evidence=dict(base_reasons=first.get('rejection_reasons',[]),content_reasons=content.get('content_audit',{}).get('reasons',[]))
 return dict(system=SYSTEM,images=[review.image(job['source_path'],job['source_sha256'])],text=
  'This is the REAL SOURCE, not a proposed night result. Draft (untrusted): '+plan['appearance_prompt']+
  '\nRule signals (not semantic truth): '+repair.dumps(evidence)+
  '\nAllowed non-mercury styles: '+', '.join(k for k in repair.styles.STYLES if k!='mercury_vapor'))

def finalize(root,bid,cfg):
 root=Path(root);directory=root/'batches'/bid
 if (directory/'plan_config.json').exists():raise ValueError('Never revise a dispatched or frozen image plan')
 gate=amendment(root);c=repair.db(root)
 items=c.execute('select a.*,p.original,p.defect from attempts a join places p on a.id=p.id where a.batch=? and a.job is not null and a.result is null order by a.id',(bid,)).fetchall()
 items=[r for r in items if json.loads(r['planning']).get('prompt_gate',{}).get('fingerprint')!=gate['fingerprint']]
 vlm=prepare_amended_vlm(root,cfg,directory)
 amended=copy.deepcopy(cfg);amended['fingerprint']=gate['fingerprint'];amended['files_sha256'][gate['code_path']]=gate['code_sha256']
 try:
  for start in range(0,len(items),gate['vlm_batch_size']):
   if (root/'pause_requested').exists():return False
   chunk=items[start:start+gate['vlm_batch_size']]
   tasks=[prepare_task(r,json.loads(r['job']),json.loads(r['planning'])) for r in chunk]
   responses=vlm.ask(tasks,'source_grounded_final',gate['max_new_tokens'])
   for item,task,raw in zip(chunk,tasks,responses,strict=True):
    error=None;final=None
    for retry in range(gate['text_attempts']):
     try:final=validate(review.parse(raw));break
     except (ValueError,KeyError,TypeError) as exc:
      error=str(exc)
      if not retry:
       corrected=copy.deepcopy(task);corrected['text']+='\nInvalid previous response: '+raw+'\nValidation error: '+error+'. Correct the NIGHT prompt and JSON.'
       raw=vlm.ask([corrected],'source_grounded_retry',gate['max_new_tokens'])[0]
    archive=directory/'pre_gate_drafts'/('r'+str(item['round'])+'_'+item['id']+'.json')
    before=dict(planning=json.loads(item['planning']),job=json.loads(item['job']))
    if archive.exists():
     if json.loads(archive.read_text())!=before:raise ValueError('Prompt draft archive differs')
    else:atomic(archive,before)
    if final is None:
     state,nxt=repair.next_status(item['round'],False)
     evidence=dict(decision='prompt_invalid_no_generation',error=error,raw_response=raw,prompt_gate=gate['fingerprint'],draft_archive=str(archive),no_image_POST=True)
     with c:
      c.execute('update attempts set result=?,job=null where id=? and round=?',(repair.dumps(evidence),item['id'],item['round']))
      c.execute('update places set status=?,round=? where id=?',(state,nxt,item['id']))
     continue
    plan={**before['planning'],**final,'raw_response':raw,'prompt_gate':dict(fingerprint=gate['fingerprint'],draft_archive=str(archive),source_only_final_pass=True)}
    original=json.loads(item['original']);source=json.loads(item['source'])
    job=repair.make_job(original,source,plan,item['round'],amended)
    with c:c.execute('update attempts set planning=?,job=? where id=? and round=?',(repair.dumps(plan),repair.dumps(job),item['id'],item['round']))
    print(repair.dumps(dict(prompt_finalized=item['id'],round=item['round'],style=plan['style_id'],prompt_gate=gate['fingerprint'])),flush=True)
 finally:vlm.close();c.close()
 return not (root/'pause_requested').exists()

ORIGINAL_FREEZE=repair.freeze_batch
def freeze_batch(root,bid,cfg):
 if not finalize(root,bid,cfg):return
 amended=copy.deepcopy(cfg);gate=amendment(root)
 amended['fingerprint']=gate['fingerprint'];amended['files_sha256'][gate['code_path']]=gate['code_sha256']
 ORIGINAL_FREEZE(root,bid,amended)

def prepare_amended_vlm(root,cfg,directory):
 """Reuse CPU thread settings after the paired-draft VLM in the same process."""
 class FinalVLM(review.VLM):
  def load(self):
   if self.backend is None:
    import torch
    from experiments.qwen_curriculum.scene_prompting import LocalVisionPlanner
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(min(32*1024**3/torch.cuda.get_device_properties(0).total_memory,.85))
    for path,digest in self.cfg['model_sha256'].items():
     if common.file_sha256(path)!=digest:raise ValueError('Frozen VLM checkpoint changed')
    self.backend=LocalVisionPlanner(review.MODEL)
   return self.backend
 return FinalVLM(directory,dict(fingerprint=amendment(root)['fingerprint'],model_sha256=cfg['model_sha256']))

def child_step(root,bid,step):
 repair.progress(root,step)
 cmd=[sys.executable,'-B','-m',__name__ if __name__!='__main__' else 'experiments.qwen_curriculum.night_repair_prompt_gate','worker','--root',str(root),'--batch',bid,'--step',step,'--execute']
 import os
 result=subprocess.run(cmd,env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','ADAPTVPR_CLIP_LOCAL_FILES_ONLY':'1'})
 if (Path(root)/'pause_requested').exists():return False
 if result.returncode:
  waitfile=Path(root)/'batches'/bid/'resource_wait.json'
  if waitfile.exists():waitfile.unlink();return False
  raise RuntimeError('Amended repair worker failed; committed progress retained. No automatic duplicate POST.')
 return True

def main():
 import argparse
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--root',type=Path,default=repair.DEFAULT);args,_=p.parse_known_args()
 amendment(args.root.resolve());repair.freeze_batch=freeze_batch;repair.child_step=child_step;repair.main()

if __name__=='__main__':main()
