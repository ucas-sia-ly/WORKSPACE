"""Operational controls/monitor for the dual repair campaign; no inference changes."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from collections import Counter
from experiments.qwen_curriculum import night_repair_campaign as repair,common
from experiments.qwen_curriculum.night_full_filter import atomic
UNIT='ccr-night-repair-v6.service'

def unit_info():
 return subprocess.run(['systemctl','--user','show',UNIT,'--property=LoadState','--property=ActiveState','--property=SubState','--property=ExecMainStatus'],text=True,capture_output=True).stdout

def status(root,export=False):
 c=repair.db(root);counts=dict(c.execute('select status,count(*) from places group by status').fetchall())
 batches=[dict(r) for r in c.execute('select id,node,phase from batches order by created')]
 saved=0
 for b in batches:
  d=root/'batches'/b['id'];summary={}
  for name in ('summary.json','remote_progress.json'):
   path=d/name
   if path.exists():summary=json.loads(path.read_text());break
  b['images_saved']=int(summary.get('saved',0));saved+=b['images_saved']
  if b['phase']=='planning':
   stats=c.execute("select count(*),sum(job is not null),sum(json_extract(planning,'$.prompt_gate.fingerprint') is not null) from attempts where batch=? and result is null",(b['id'],)).fetchone()
   b.update(attempts=int(stats[0]),drafts_ready=int(stats[1] or 0),final_prompts_ready=int(stats[2] or 0))
 outcomes=Counter();images=0
 for r in c.execute('select result from attempts where result is not null'):
  v=json.loads(r[0]);outcomes[v['decision']]+=1;images+=int('generated' in v)
 result=dict(total_original_rejects=sum(counts.values()),places=counts,images_generated_and_audited=images,outcomes=dict(outcomes),
  images_saved=saved,batches=batches,controller=unit_info().splitlines(),pause_requested=(root/'pause_requested').exists(),quarantined_variants=[],max_alternate_sources=2,updated_unix=time.time(),canonical_replaced=False)
 if (root/'acceptance_protocol_v1.json').exists():
  result.update(accepted_image_verdicts=['pass','unknown'],retry_image_verdicts=['reject'],
   accepted_generated_unknown=c.execute("select count(*) from places p join attempts a on p.id=a.id and p.round=a.round where p.status='passed' and json_extract(a.result,'$.decision')='unknown'").fetchone()[0],
   accepted_original_recheck_unknown=counts.get('passed_original_unknown',0))
 if (root/'staged_state.json').exists():result['schedule']=json.loads((root/'staged_state.json').read_text())
 if (root/'round_one_stop_protocol_v1.json').exists():
  result.update(last_automatic_source_round=0,automatic_alternate_sources=False,
   round_one_finished=(root/'round_one_finished.json').exists())
 atomic(root/'live_status.json',result)
 if export and not counts.get('policy_recheck',0):
  for state,name in [('passed_original','restored_original_pass.jsonl'),('unknown_recheck','policy_recheck_unknown.jsonl')]:
   target=root/name
   if not target.exists():
    with target.open('x') as f:
     for r in c.execute('select p.original,q.result from places p join policy_rechecks q on p.id=q.id where p.status=? order by p.ordinal',(state,)):
      f.write(repair.dumps({**json.loads(r[0]),'content_recheck_no_quarantine':json.loads(r[1]),'eligible_for_training':False})+'\n')
 c.close();return result

def recover_prompt_only(root):
 """Only skip a proven invalid TEXT result before any image plan/POST exists."""
 tail=(root/'run.log').read_text(errors='replace')[-20000:]
 progress=json.loads((root/'progress.json').read_text()) if (root/'progress.json').exists() else {}
 if progress.get('phase')!='plan':return False
 latest=tail.rsplit('"phase": "plan"',1)[-1]
 if 'Qwen prompt invalid after bounded retry' not in latest:return False
 c=repair.db(root)
 for r in c.execute("select a.*,p.original from attempts a join places p on p.id=a.id join batches b on a.batch=b.id where b.phase='planning' and a.job is null and a.result is null order by p.ordinal"):
  path=root/'batches'/r['batch']/('prompt_raw_'+common.fingerprint([r['id'],r['round']])+'.json')
  if not path.exists():continue
  original=json.loads(r['original']);raw=json.loads(path.read_text())['response']
  try:repair.validate_prompt(repair.review.parse(raw),original['prompt'].partition('\n\nKeep this exact viewpoint')[0],new_source=r['round']>0)
  except (ValueError,KeyError,TypeError) as exc:
   # No image request can exist: no frozen job was ever constructed for this attempt.
   state,nxt=repair.next_status(r['round'],False)
   with c:
    c.execute('update attempts set result=? where id=? and round=?',(repair.dumps(dict(decision='prompt_invalid_no_generation',error=str(exc),saved_raw_response=str(path))),r['id'],r['round']))
    c.execute('update places set status=?,round=? where id=?',(state,nxt,r['id']))
   atomic(root/'last_text_recovery.json',dict(id=r['id'],round=r['round'],error=str(exc),no_image_POST=True));c.close();return True
 c.close();return False

def receive_finished_remote(root):
 try:
  with repair.common_lock(Path(root)/'remote_receive.lock'):
   return _receive_finished_remote(root)
 except BlockingIOError:
  # The other monitor/controller owns this transfer; retry on its next tick.
  return

def _receive_finished_remote(root):
 """A successful transient unit may disappear before polling; trust verified artifacts."""
 c=repair.db(root);batches=c.execute("select id from batches where node='gpu2' and phase='images'").fetchall();c.close()
 for item in batches:
  bid=item['id'];d=root/'batches'/bid
  code='import json; from pathlib import Path; p=Path('+repr(str(d/'summary.json'))+'); print(p.read_text() if p.exists() else "{}")'
  try:summary=json.loads(repair.remote_command(['/home/epic/anaconda3/envs/ccr-night/bin/python','-c',code],timeout=20))
  except (subprocess.SubprocessError,ValueError):continue
  if summary:atomic(d/'remote_progress.json',summary)
  if summary.get('state')!='domain_complete' or summary.get('saved')!=summary.get('planned'):continue
  subprocess.run(['rsync','-a','--partial','ccr-gpu2:'+str(d)+'/',str(d)+'/'],check=True)
  cfg,jobs=repair.load_image_plan(d);by_id={j['sample_id']:j for j in jobs};results=repair.rows(d/'results.jsonl')
  if len(results)!=len(jobs) or len({r['sample_id'] for r in results})!=len(jobs):raise ValueError('Remote complete result cardinality mismatch')
  for r in results:
   j=by_id[r['sample_id']]
   if any(r[k]!=j[k] for k in ('source_path','source_sha256','prompt','seed')):raise ValueError('Remote complete exact pairing mismatch')
   for k in ('output','raw_output'):
    p=Path(r[k+'_path']).resolve()
    if not p.is_relative_to((d/'images').resolve()) or common.file_sha256(p)!=r[k+'_sha256']:raise ValueError('Remote artifact location or hash invalid')
  atomic(d/'primary_received.json',dict(node='gpu2',images=len(results),all_output_hashes_verified=True,completion_source='verified_artifacts_even_if_transient_unit_unloaded'))
  c=repair.db(root)
  with c:c.execute("update batches set phase='audit' where id=? and phase='images'",(bid,))
  c.close()


def launch(root):
 info=unit_info()
 if 'ActiveState=active' in info:return
 if 'LoadState=not-found' not in info:
  subprocess.run(['systemctl','--user','start',UNIT],check=True);return
 cmd=['systemd-run','--user','--unit='+UNIT,'--description=双机Qwen拒绝夜景修复','--property=Type=exec','--property=MemoryHigh=10G','--property=MemoryMax=14G','--property=MemorySwapMax=0','--property=CPUQuota=500%','--property=TimeoutStopSec=1900','--property=KillMode=mixed',
  '--property=StandardOutput=append:'+str(root/'run.log'),'--property=StandardError=append:'+str(root/'run.log'),'--working-directory='+str(repair.ROOT),
  '--setenv=PYTHONPATH='+str(repair.ROOT/'AdaptVPR'),'--setenv=PYTHONDONTWRITEBYTECODE=1','--setenv=PYTHONUNBUFFERED=1','--setenv=HF_HUB_OFFLINE=1','--setenv=TRANSFORMERS_OFFLINE=1','--setenv=ADAPTVPR_CLIP_LOCAL_FILES_ONLY=1','--setenv=OMP_NUM_THREADS=4','--setenv=OPENBLAS_NUM_THREADS=4',
  sys.executable,'-B','-m','experiments.qwen_curriculum.night_repair_round_one','run','--root',str(root),'--execute']
 subprocess.run(cmd,check=True)

def resume(root):
 if (root/'round_one_finished.json').exists():
  print('First round is finished; automatic retries are disabled. Review exports in round_one_review/.');return
 if 'ActiveState=active' in unit_info():
  print('Controller is running or finishing a pause; wait for it to exit before resume.');return
 repair.config(root);paused=(root/'pause_requested').exists()
 (root/'pause_requested').unlink(missing_ok=True)
 repair.remote_command(['/home/epic/anaconda3/envs/ccr-night/bin/python','-c','from pathlib import Path; Path('+repr(str(root/'pause_requested'))+').unlink(missing_ok=True)'])
 if paused and not (root/'staged_protocol_v1.json').exists():
  c=repair.db(root)
  for r in c.execute("select id from batches where node='gpu2' and phase='images'"):
   # Explicit user-resume of a known pause; same jobs and reservation journal.
   repair.remote_command(['systemctl','--user','restart',repair.remote_unit(r['id'])])
  c.close()
 launch(root)

def monitor(root):
 while True:
  receive_finished_remote(root)
  if (root/'pipeline_protocol_v1.json').exists() and 'ActiveState=active' in unit_info():
   if (root/'staged_protocol_v1.json').exists():
    from experiments.qwen_curriculum.night_repair_staged import dispatch
   else:
    from experiments.qwen_curriculum.night_repair_pipeline import dispatch
   dispatch(root)
  v=status(root,True)
  if (root/'complete.json').exists() or (root/'round_one_finished.json').exists():print(repair.dumps(v),flush=True);return
  if 'ActiveState=failed' in unit_info() and not (root/'pause_requested').exists():
   if recover_prompt_only(root):launch(root)
   else:
    atomic(root/'needs_attention.json',dict(reason='Controller failed; integrity, transport or generation ambiguity must not be automatically retried.',updated_unix=time.time()))
    return
  time.sleep(20)

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['status','pause','resume','monitor']);p.add_argument('--root',type=Path,default=repair.DEFAULT);p.add_argument('--execute',action='store_true');a=p.parse_args();root=a.root.resolve()
 if a.action=='status':print(json.dumps(status(root,True),ensure_ascii=False,indent=2))
 elif a.action=='pause':
  atomic(root/'pause_requested',dict(requested_unix=time.time()))
  repair.remote_command(['/home/epic/anaconda3/envs/ccr-night/bin/python','-c','from pathlib import Path; Path('+repr(str(root/'pause_requested'))+').touch()']);print('Pause requested on both machines; finish current images and release models.')
 elif not a.execute:raise ValueError('Explicit --execute required')
 elif a.action=='resume':resume(root)
 else:monitor(root)
if __name__=='__main__':main()
