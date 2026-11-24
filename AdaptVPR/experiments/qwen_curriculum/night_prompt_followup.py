"""Exact reviewed-Prompt follow-up. Independent queue; wait for parent completion.

No VLM planning, no automatic alternative-source retry, no parent-state mutation.
Existing frozen image workers retain durable HTTP journals and resource guards.
"""
from __future__ import annotations
import argparse,copy,json,sqlite3,threading,time
from pathlib import Path
from experiments.qwen_curriculum import common,campaign,night_repair_campaign as repair
from experiments.qwen_curriculum.night_full_filter import atomic,common_lock
ROOT=common.WORKSPACE_ROOT
PARENT=ROOT/'outputs/night_repair_v1'
BUNDLE=PARENT/'prompt_revisions/held1661_visual_finish_20261118_v3'
DEFAULT=ROOT/'outputs/night_prompt_followup_324_v1'

def select_changes(scope,records):
 ids={r['sample_id'] for r in scope}
 selected=[r for r in records if r['sample_id'] in ids and r['proposed_prompt']!=r['latest_rejection']['generated']['prompt']]
 if len({r['sample_id'] for r in selected})!=len(selected):raise ValueError('Duplicate review identity')
 return sorted(selected,key=lambda r:r['review_index'])

def make_job(row,cfg):
 previous=row['latest_rejection']['generated']
 src=row.get('source_replacement') or row['original_source']
 source={k:src[k] for k in ('source_path','source_sha256')}
 from PIL import Image
 with Image.open(source['source_path']) as im:source['source_dimensions']=list(im.size)
 source['source_sample_id']=src.get('source_sample_id','qnsource_'+source['source_sha256'][:24])
 p=Path(source['source_path']);prefix=f"{row['city']}_{int(row['place_id']):07d}_"
 if not p.name.startswith(prefix) or common.file_sha256(p)!=source['source_sha256']:raise ValueError('Exact source/place integrity failed')
 prompt=src.get('new_source_prompt',row['proposed_prompt']) if row.get('source_replacement') else row['proposed_prompt']
 style=src.get('style_id',row['style_id']) if row.get('source_replacement') else row['style_id']
 body,sep,appendix=prompt.partition('\n\n'+repair.styles.CONTRACT)
 if not sep or appendix or not body.strip():raise ValueError('Reviewed prompt must retain the fixed contract exactly once')
 plan=dict(source_summary=row.get('visual_observation','Reviewed source-conditioned night Prompt'),appearance_prompt=body,style_id=style,prompt_body_author='Codex',reviewed_prompt_sha256=common.fingerprint(prompt))
 job=repair.make_job(previous,source,plan,0,cfg)
 job.update(prompt=prompt,prompt_author='Codex',prompt_policy='codex_visual_review_followup_v1',style_name='Codex reviewed: '+style,
  reviewed_prompt_field='source_replacement.new_source_prompt' if src.get('new_source_prompt') else 'proposed_prompt',
  review_sample_id=row['sample_id'],review_index=row['review_index'],replaces_generated_sample_id=previous['sample_id'],
  reviewed_source_replaced=bool(row.get('source_replacement')),source_change_count_from_review=int(bool(row.get('source_replacement'))),
  automatic_source_retries=False)
 # make_job's identity includes new protocol, exact source and reviewed plan; retain previous seed for a same-source controlled edit.
 job['record_sha256']=common.fingerprint({k:v for k,v in job.items() if k!='record_sha256'})
 return job

def prepare(root):
 root=Path(root);root.mkdir(parents=True,exist_ok=True)
 with common_lock(root/'prepare.lock'):
  if (root/'prepared.json').exists():
   cfg=verify(root);return cfg
  if (root/'state.sqlite').exists():raise ValueError('Incomplete preparation exists; inspect rather than overwrite')
  scope_path=PARENT/'prompt_revisions/visual_verification_20261118_v2/reviewed_records.jsonl'
  manifest=BUNDLE/'consolidated_1967_handoff.jsonl'
  selected=select_changes(repair.rows(scope_path),repair.rows(manifest))
  if len(selected)!=324:raise ValueError('Expected exactly the authorized 324 changed Prompts')
  common.write_jsonl(root/'reviewed_changes.jsonl',selected)
  cfg=copy.deepcopy(repair.config(PARENT));cfg.pop('fingerprint',None)
  cfg.update(expected=324,first_rejects=0,second_rejects=0,followup=True,parent_root=str(PARENT),parent_protocol_fingerprint=repair.config(PARENT)['fingerprint'],
   reviewed_manifest=str(manifest),reviewed_manifest_sha256=common.file_sha256(manifest),review_scope_sha256=common.file_sha256(scope_path),
   input_sha256=common.file_sha256(root/'reviewed_changes.jsonl'),prompt_body_author='Codex',vlm_planning=False,
   max_alternate_sources=0,max_images_per_place=1,execution_nodes=['local','gpu2'],
   followup_code_path=str(Path(__file__).resolve()),followup_code_sha256=common.file_sha256(__file__),
   start_barrier='parent round_one_finished.json and every parent round-0 attempt resolved; no unfinished parent batch',
   accepted_image_verdicts=['pass','unknown'],automatic_retries=False,originals_immutable=True)
  cfg=campaign.frozen(root/'protocol.json',cfg)
  c=repair.db(root)
  try:
   with c:
    for batchno,start in enumerate(range(0,len(selected),64)):
     group=selected[start:start+64];node='local' if batchno%2==0 else 'gpu2'
     bid='f_'+common.fingerprint([cfg['fingerprint'],batchno])[:20]
     (root/'batches'/bid).mkdir(parents=True)
     c.execute('insert into batches(id,created,phase,node) values(?,?,?,?)',(bid,time.time()+batchno/1000,'ready',node))
     jobs=[]
     for row in group:
      job=make_job(row,cfg);sid=row['sample_id'];jobs.append(job)
      c.execute('insert into places(id,ordinal,city,place_id,original,defect,status,round) values(?,?,?,?,?,?,?,0)',
       (sid,row['review_index'],row['city'],int(row['place_id']),repair.dumps(row['latest_rejection']['generated']),repair.dumps(row['latest_rejection']),'working'))
      c.execute('insert into attempts(id,round,batch,source,planning,job) values(?,0,?,?,?,?)',
       (sid,bid,repair.dumps({k:job[k] for k in ('source_path','source_sha256','source_dimensions','source_sample_id')}),repair.dumps(dict(author='Codex',no_vlm_rewrite=True,review_index=row['review_index'])),repair.dumps(job)))
     jobs.sort(key=lambda j:j['review_sample_id']);d=root/'batches'/bid;common.write_jsonl(d/'plan.jsonl',jobs)
     campaign.frozen(d/'plan_config.json',dict(version=1,stage='codex_visual_review_followup_v1',num_jobs=len(jobs),request_budget=len(jobs),domain_counts={'night':len(jobs)},
      image_root=str(d/'images'),service=cfg['service'],quality=cfg['quality'],plan_sha256=common.file_sha256(d/'plan.jsonl'),files_sha256=cfg['files_sha256'],campaign_fingerprint=cfg['fingerprint']))
  finally:c.close()
  atomic(root/'staged_state.json',dict(phase='waiting_parent',source_round=0))
  atomic(root/'prepared.json',dict(total=324,same_source=297,alternate_source=27,batches=6,state='queued_waiting_parent',parent=str(PARENT),
   protocol_fingerprint=cfg['fingerprint'],parent_queue_modified=False,models_loaded=False))
  (root/'README.md').write_text('''# 324 条已审核 Prompt 后续队列

297 条同源、27 条换同地点源图。仅选择最终 Prompt 与第二次生图实际 Prompt 不同的记录；未修改的 1,634 条不在队列。

等待当前 `night_repair_v1` 第一轮所有生成及审核结束后启动，复用双机原有后端和资源保护。全部 Prompt 已准备，不启动 VLM 重写；已保存结果按原持久请求记录恢复，禁止不明 POST 自动重发。

生成后沿用原 LightGlue/CLIP/DINO 校准审核。Pass/Unknown 接受，Reject 单列，不自动换图或重试；不覆盖旧图片，不自动替换训练数据。

暂停：`touch outputs/night_prompt_followup_324_v1/pause_requested`（完成当前图片后释放模型；远端控制器传播暂停）。恢复：删除该标记及远端同路径标记后，重启 `ccr-night-prompt-followup-324-v1.service`。查看 `queue_status.json` / `run.log`。
''')
 return verify(root)

def verify(root):
 root=Path(root);cfg=repair.config(root)
 if common.file_sha256(__file__)!=cfg['followup_code_sha256']:raise ValueError('Follow-up scheduler changed')
 if common.file_sha256(root/'reviewed_changes.jsonl')!=cfg['input_sha256']:raise ValueError('Reviewed input changed')
 c=repair.db(root)
 try:
  jobs=[json.loads(r[0]) for r in c.execute('select job from attempts order by id')]
  if len(jobs)!=324 or len({j['sample_id'] for j in jobs})!=324:raise ValueError('Queue identity/budget changed')
  for b in c.execute('select id from batches'):
   _,frozen=repair.load_image_plan(root/'batches'/b['id'])
   for j in frozen:
    if j['record_sha256']!=common.fingerprint({k:v for k,v in j.items() if k!='record_sha256'}):raise ValueError('Frozen job changed')
 finally:c.close()
 return cfg

def parent_finished(parent):
 parent=Path(parent)
 marker=parent/'round_one_finished.json'
 if not marker.exists():return False
 receipt=json.loads(marker.read_text())
 if not receipt.get('stopped') or receipt.get('round')!=0:return False
 c=sqlite3.connect('file:'+str(parent/'state.sqlite')+'?mode=ro',uri=True)
 try:
  c.execute('BEGIN')
  unresolved=c.execute('select count(*) from attempts where round=0 and result is null').fetchone()[0]
  unfinished=c.execute("select count(*) from batches where phase!='done'").fetchone()[0]
  return unresolved==0 and unfinished==0
 finally:c.close()

def status(root,phase):
 c=repair.db(root)
 try:
  bs=[dict(r) for r in c.execute('select id,node,phase from batches order by created')]
  decisions={}
  for r in c.execute('select result from attempts where result is not null'):
   decision=json.loads(r[0])['decision'];decisions[decision]=decisions.get(decision,0)+1
  atomic(Path(root)/'queue_status.json',dict(phase=phase,total=324,outcomes=decisions,batches=bs,updated_unix=time.time(),automatic_retries=False))
 finally:c.close()

def export_results(root):
 c=repair.db(root);accepted=[];rejected=[]
 try:
  with c:
   for a in c.execute('select id,result from attempts order by id').fetchall():
    e=json.loads(a['result']);accepted_flag=e['decision'] in ('pass','unknown')
    c.execute('update places set status=? where id=?',('passed' if accepted_flag else 'exhausted',a['id']))
    row=dict(review_sample_id=a['id'],raw_audit=e,accepted_under_user_policy=accepted_flag,eligible_for_training=False,canonical_replaced=False)
    (accepted if accepted_flag else rejected).append(row)
  common.write_jsonl(Path(root)/'accepted_followup.jsonl',accepted);common.write_jsonl(Path(root)/'rejected_followup.jsonl',rejected)
 finally:c.close()
 atomic(Path(root)/'complete.json',dict(total=324,accepted=len(accepted),reject=len(rejected),automatic_retries=False,updated_unix=time.time()))

def run(root):
 from experiments.qwen_curriculum import night_repair_control as control,night_repair_staged as staged
 root=Path(root);cfg=verify(root)
 with common_lock(root/'controller.lock'):
  if (root/'complete.json').exists():return
  status(root,'waiting_parent')
  while not parent_finished(Path(cfg['parent_root'])):
   if (root/'pause_requested').exists():status(root,'paused_waiting_parent');return
   time.sleep(20)
  # Parent has fully released both inference and audit processes at its round barrier.
  atomic(root/'parent_completion_seen.json',dict(parent=cfg['parent_root'],marker_sha256=common.file_sha256(Path(cfg['parent_root'])/'round_one_finished.json'),seen_unix=time.time()))
  atomic(root/'staged_state.json',dict(phase='images',source_round=0))
  stop=threading.Event();errors=[]
  def remote():
   while not stop.is_set():
    try:
     control.receive_finished_remote(root)
     if (root/'pause_requested').exists():
      repair.remote_command(['/home/epic/anaconda3/envs/ccr-night/bin/python','-c','from pathlib import Path; Path('+repr(str(root/'pause_requested'))+').touch()']);return
     staged.dispatch(root)
    except Exception as e:errors.append(e);return
    stop.wait(15)
  thread=threading.Thread(target=remote,daemon=True);thread.start()
  try:
   while True:
    if errors:raise RuntimeError('Remote follow-up needs inspection; no ambiguous request replay') from errors[0]
    if (root/'pause_requested').exists():status(root,'paused');return
    pending=[b for b in staged.work(root) if b['phase'] in ('ready','images')]
    status(root,'images')
    if not pending:break
    local=next((b for b in pending if b['node']=='local'),None)
    if not local or repair.check_resources(root,'images'):time.sleep(15);continue
    c=repair.db(root)
    with c:c.execute("update batches set phase='images' where id=?",(local['id'],))
    c.close()
    if repair.child_step(root,local['id'],'images'):
     d=root/'batches'/local['id'];s=json.loads((d/'summary.json').read_text())
     if s.get('saved')!=s.get('planned') or s.get('state')!='domain_complete':raise ValueError('Local follow-up incomplete')
     c=repair.db(root)
     with c:c.execute("update batches set phase='audit' where id=?",(local['id'],))
     c.close()
  finally:stop.set();thread.join(timeout=45)
  atomic(root/'staged_state.json',dict(phase='audit',source_round=0))
  while True:
   if (root/'pause_requested').exists():status(root,'paused');return
   work=staged.work(root)
   if not work:break
   batch=next((b for b in work if b['phase']=='audit'),None)
   if not batch:raise ValueError('Unexpected follow-up phase')
   if repair.check_resources(root,'audit'):time.sleep(15);continue
   status(root,'audit');repair.child_step(root,batch['id'],'audit')
  export_results(root);status(root,'complete')

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','status']);p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--execute',action='store_true');a=p.parse_args()
 if a.action=='prepare':print(json.dumps(prepare(a.root),ensure_ascii=False))
 elif a.action=='status':print((a.root/'queue_status.json').read_text() if (a.root/'queue_status.json').exists() else (a.root/'prepared.json').read_text())
 else:
  if not a.execute:raise ValueError('--execute required')
  run(a.root)
if __name__=='__main__':main()
