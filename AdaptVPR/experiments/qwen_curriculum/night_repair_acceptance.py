"""User-authorized Unknown acceptance; preserve frozen classifiers and verdicts."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from experiments.qwen_curriculum import night_repair_campaign as repair,campaign,common
from experiments.qwen_curriculum import night_repair_prompt_gate as gate
from experiments.qwen_curriculum import night_repair_pipeline as pipeline
from experiments.qwen_curriculum.night_full_filter import atomic,common_lock

def protocol(root):
 parent=pipeline.protocol(root)
 return campaign.frozen(Path(root)/'acceptance_protocol_v1.json',dict(version=1,
  parent_pipeline=parent['fingerprint'],code_sha256=common.file_sha256(__file__),
  accepted_image_verdicts=['pass','unknown'],retry_image_verdicts=['reject'],
  original_unknowns_accepted=True,raw_verdicts_immutable=True,
  classifier_and_thresholds_unchanged=True,max_alternate_sources=2,
  training_eligibility_unchanged=True,authorization='User: Unknown also passes; only Reject triggers same-place regeneration.'))

def reconcile(root):
 """No prior Unknown may remain pending for image-based source retries."""
 root=Path(root);policy=protocol(root);c=repair.db(root);changes=[]
 try:
  c.execute('BEGIN IMMEDIATE')
  items=c.execute("select a.id,a.round,a.result,p.status,p.round as place_round from attempts a join places p on a.id=p.id where json_extract(a.result,'$.decision')='unknown' and json_extract(a.result,'$.generated') is not null order by a.round").fetchall()
  for a in items:
   if a['status']=='passed':continue
   # Do not silently cancel already dispatched later work. Migration occurs at
   # the current safe boundary; future audit reconciliation precedes reservation.
   later=c.execute('select 1 from attempts where id=? and round>?',(a['id'],a['round'])).fetchone()
   if later:raise ValueError('Later source already reserved for Unknown; explicit reconciliation needed: '+a['id'])
   changes.append(dict(id=a['id'],from_status=a['status'],from_round=a['place_round'],to_status='passed',accepted_round=a['round'],raw_decision='unknown'))
   c.execute("update places set status='passed',round=? where id=?",(a['round'],a['id']))
  for a in c.execute("select id,round from places where status='unknown_recheck'").fetchall():
   changes.append(dict(id=a['id'],from_status='unknown_recheck',from_round=a['round'],to_status='passed_original_unknown',raw_decision='unknown'))
   c.execute("update places set status='passed_original_unknown' where id=?",(a['id'],))
  c.execute('create table if not exists acceptance_changes(id text, policy text, record text, primary key(id,policy))')
  for change in changes:c.execute('insert into acceptance_changes values(?,?,?)',(change['id'],policy['fingerprint'],repair.dumps(change)))
  c.commit()
 finally:c.close()
 return changes

BASE_EXPORT=repair.export
def export(root):
 root=Path(root);reconcile(root);BASE_EXPORT(root);policy=protocol(root)
 path=root/'accepted_replacements.jsonl';records=repair.rows(path)
 temp=path.with_suffix('.policy.tmp')
 with temp.open('w') as f:
  for r in records:
   r.update(acceptance_policy=policy['fingerprint'],accepted_under_user_policy=True,
    raw_audit_decision=r['repair_audit']['decision'],accepted_unknown=r['repair_audit']['decision']=='unknown')
   f.write(repair.dumps(r)+'\n')
 temp.replace(path)
 # Original Unknowns were never in the repair queue. Export acceptance without
 # adding a generation job or modifying any historical audit manifest.
 original=root/'accepted_original_unknowns.jsonl'
 if not original.exists():
  c=repair.db(root);tmp=original.with_suffix('.tmp');seen=set()
  try:
   with tmp.open('w') as f:
    for r in repair.rows(repair.ROOT/'outputs/night_dino_audit_v1/unknown_manifest.jsonl'):
     sid=r['sample_id'];seen.add(sid)
     f.write(repair.dumps({**r,'raw_audit_decision':'unknown','accepted_under_user_policy':True,'acceptance_policy':policy['fingerprint'],'eligible_for_training':False})+'\n')
    for a in c.execute("select p.original,q.result from places p join policy_rechecks q on p.id=q.id where p.status='passed_original_unknown'"):
     r=json.loads(a['original']);assert r['sample_id'] not in seen
     f.write(repair.dumps({**r,'content_recheck_no_quarantine':json.loads(a['result']),'raw_audit_decision':'unknown','accepted_under_user_policy':True,'acceptance_policy':policy['fingerprint'],'eligible_for_training':False})+'\n')
   tmp.replace(original)
  finally:c.close()

BASE_CHILD=gate.child_step
def child_step(root,bid,step):
 if step!='audit':return BASE_CHILD(root,bid,step)
 repair.progress(root,step)
 cmd=[sys.executable,'-B','-m','experiments.qwen_curriculum.night_repair_acceptance','audit','--root',str(root),'--batch',bid,'--execute']
 result=subprocess.run(cmd,env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','ADAPTVPR_CLIP_LOCAL_FILES_ONLY':'1'})
 if (Path(root)/'pause_requested').exists():return False
 if result.returncode:raise RuntimeError('Acceptance-aware audit failed; preserve committed results and stop.')
 return True

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['run','audit','migrate']);p.add_argument('--root',type=Path,default=repair.DEFAULT);p.add_argument('--batch');p.add_argument('--execute',action='store_true');a=p.parse_args()
 if not a.execute:raise ValueError('--execute required')
 root=a.root.resolve();protocol(root)
 if a.action=='migrate':
  with common_lock(root/'controller.lock'):
   changes=reconcile(root);export(root);atomic(root/'acceptance_migration_v1.json',dict(changes=len(changes),by_previous_status={s:sum(v['from_status']==s for v in changes) for s in {v['from_status'] for v in changes}},updated_unix=time.time(),raw_verdicts_unchanged=True))
 elif a.action=='audit':
  if not a.batch:raise ValueError('--batch required')
  repair.audit_batch(root,a.batch);reconcile(root)
 else:
  reconcile(root);gate.child_step=child_step;repair.export=export;pipeline.orchestrate(root)

if __name__=='__main__':main()
