"""Round barriers: all Prompts, dual-node images, then audits. No model ping-pong."""
import argparse,json,time
from pathlib import Path
from experiments.qwen_curriculum import night_repair_campaign as repair,campaign,common
from experiments.qwen_curriculum import night_repair_acceptance as accept
from experiments.qwen_curriculum.night_full_filter import atomic,common_lock

def protocol(root):
 parent=accept.protocol(root)
 return campaign.frozen(Path(root)/'staged_protocol_v1.json',dict(version=1,
  parent_acceptance=parent['fingerprint'],code_sha256=common.file_sha256(__file__),
  stages=['prompts','images','audit'],barrier='all places in the current source round',
  next_round='only after current-round image audit; Reject retries, Unknown accepted',
  frozen_jobs_and_completed_outputs_reused=True,max_alternate_sources=2))

def work(root):
 c=repair.db(root)
 try:return [dict(r) for r in c.execute("select b.*,min(a.round) as source_round from batches b join attempts a on a.batch=b.id where b.phase!='done' group by b.id order by b.created")]
 finally:c.close()

def next_round(root):
 c=repair.db(root)
 try:
  rounds=[r[0] for r in c.execute("select round from places where status='pending' union select a.round from attempts a join batches b on a.batch=b.id where b.phase!='done'")]
  return min(rounds) if rounds else None
 finally:c.close()

def state(root,phase,roundno):
 atomic(Path(root)/'staged_state.json',dict(phase=phase,source_round=roundno,updated_unix=time.time()))

def reserve(root,roundno,size):
 c=repair.db(root)
 try:
  c.execute('BEGIN IMMEDIATE')
  existing=c.execute("select b.id from batches b join attempts a on a.batch=b.id where b.phase='planning' and a.round=? order by b.created limit 1",(roundno,)).fetchone()
  if existing:c.rollback();return existing['id']
  places=c.execute("select id,round from places where status='pending' and round=? order by ordinal limit ?",(roundno,size)).fetchall()
  if not places:c.rollback();return None
  loads=dict(c.execute("select node,count(*) from batches where phase!='done' group by node"))
  node='local' if loads.get('local',0)<=loads.get('gpu2',0) else 'gpu2'
  bid='b_'+common.fingerprint([(r['id'],r['round']) for r in places])[:20]
  c.execute('insert into batches(id,created,node) values(?,?,?)',(bid,time.time(),node))
  for r in places:
   c.execute('insert into attempts(id,round,batch) values(?,?,?)',(r['id'],r['round'],bid))
   c.execute("update places set status='working' where id=?",(r['id'],))
  c.commit();(Path(root)/'batches'/bid).mkdir(parents=True,exist_ok=True);return bid
 finally:c.close()

def dispatch(root):
 try:return _dispatch(root)
 except BlockingIOError:return  # Another controller/monitor tick owns dispatch.

def _dispatch(root):
 root=Path(root)
 if (root/'pause_requested').exists():return
 p=root/'staged_state.json'
 if not p.exists():return
 s=json.loads(p.read_text())
 if s['phase']!='images':return
 with common_lock(root/'remote_dispatch.lock'):
  rows=[r for r in work(root) if r['node']=='gpu2' and r['source_round']==s['source_round']]
  active=[r for r in rows if r['phase']=='images']
  if len(active)>1:raise ValueError('Competing remote image batches')
  if active:repair.deploy_remote(root,active[0]['id']);return
  ready=next((r for r in rows if r['phase']=='ready'),None)
  if ready:
   c=repair.db(root)
   with c:c.execute("update batches set phase='images' where id=? and phase='ready'",(ready['id'],))
   c.close();repair.deploy_remote(root,ready['id'])

def run(root):
 from experiments.qwen_curriculum import night_repair_control as control
 root=Path(root);cfg=repair.config(root);protocol(root)
 with common_lock(root/'controller.lock'):
  accept.reconcile(root)
  while not (root/'pause_requested').exists():
   roundno=next_round(root)
   if roundno is None:
    accept.export(root);state(root,'complete',None);atomic(root/'complete.json',dict(complete=True,updated_unix=time.time(),canonical_replaced=False));return
   current=root/'staged_state.json'
   s=json.loads(current.read_text()) if current.exists() else {}
   phase=s.get('phase') if s.get('source_round')==roundno else 'prompts'
   if phase not in ('prompts','images','audit'):phase='prompts'
   state(root,phase,roundno)
   if phase=='prompts':
    # Do not dispatch ready plans or resume paused image workers in this stage.
    bid=reserve(root,roundno,cfg['batch_size'])
    if bid:
     if repair.check_resources(root,'plan'):time.sleep(10);continue
     if accept.BASE_CHILD(root,bid,'plan'):
      c=repair.db(root)
      with c:c.execute("update batches set phase='ready' where id=? and phase='images'",(bid,))
      c.close()
     continue
    state(root,'images',roundno);continue
   if phase=='images':
    control.receive_finished_remote(root);dispatch(root)
    rows=[r for r in work(root) if r['source_round']==roundno]
    local=next((r for r in rows if r['node']=='local' and r['phase'] in ('ready','images')),None)
    if local:
     if repair.check_resources(root,'images'):time.sleep(10);continue
     c=repair.db(root)
     with c:c.execute("update batches set phase='images' where id=?",(local['id'],))
     c.close()
     if accept.BASE_CHILD(root,local['id'],'images'):
      summary=json.loads((root/'batches'/local['id']/'summary.json').read_text())
      if summary['saved']!=summary['planned']:raise RuntimeError('Incomplete local images; keep original reservations')
      c=repair.db(root)
      with c:c.execute("update batches set phase='audit' where id=?",(local['id'],))
      c.close()
     continue
    if any(r['phase'] in ('ready','images') for r in rows):time.sleep(10);continue
    if any(r['phase']=='planning' for r in rows):raise ValueError('Image stage reached with unfinished Prompts')
    state(root,'audit',roundno);continue
   rows=[r for r in work(root) if r['source_round']==roundno]
   audit=next((r for r in rows if r['phase']=='audit'),None)
   if audit:
    if repair.check_resources(root,'audit'):time.sleep(10);continue
    if accept.child_step(root,audit['id'],'audit'):accept.export(root)
    continue
   if rows:raise ValueError('Audit stage contains unexpected unfinished work')
   accept.export(root)
   # next_round() now chooses the next rejection-source round, if any.
  repair.progress(root,'paused')

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['run']);p.add_argument('--root',type=Path,default=repair.DEFAULT);p.add_argument('--execute',action='store_true');a=p.parse_args()
 if not a.execute:raise ValueError('--execute required')
 run(a.root)

if __name__=='__main__':main()
