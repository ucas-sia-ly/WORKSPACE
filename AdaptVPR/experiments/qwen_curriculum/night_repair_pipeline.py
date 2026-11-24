"""Bounded remote prefetch, independent of the primary's audit queue.

Scheduling only: all Prompt, image and rule workers remain frozen and unchanged.
"""
import argparse,json,time
from pathlib import Path
from experiments.qwen_curriculum import night_repair_campaign as repair,campaign,common
from experiments.qwen_curriculum import night_repair_prompt_gate as gate
from experiments.qwen_curriculum.night_full_filter import atomic,common_lock


def protocol(root):
 cfg=repair.config(root);parent=gate.amendment(root)
 return campaign.frozen(Path(root)/'pipeline_protocol_v1.json',dict(version=1,
  parent_protocol=cfg['fingerprint'],prompt_protocol=parent['fingerprint'],
  code_sha256=common.file_sha256(__file__),remote_prefetch_batches=1,
  bootstrap_places=16,steady_places=cfg['batch_size'],max_waiting_audit_batches=4,
  unchanged='Prompt rules, image settings, exact pairing, request ledger, source retry cap and audit calibration'))


def batches(root):
 c=repair.db(root)
 try:return [dict(r) for r in c.execute("select * from batches where phase!='done' order by created")]
 finally:c.close()


def reserve(root,size,node):
 """Audit/ready batches do not prevent reserving one future Prompt batch."""
 c=repair.db(root)
 try:
  c.execute('BEGIN IMMEDIATE')
  existing=c.execute("select id from batches where node=? and phase='planning' order by created limit 1",(node,)).fetchone()
  if existing:c.rollback();return existing['id']
  items=c.execute("select id,round from places where status='pending' order by round,ordinal limit ?",(size,)).fetchall()
  if not items:c.rollback();return None
  bid='b_'+common.fingerprint([(r['id'],r['round']) for r in items])[:20]
  c.execute('insert into batches(id,created,node) values(?,?,?)',(bid,time.time(),node))
  for r in items:
   c.execute('insert into attempts(id,round,batch) values(?,?,?)',(r['id'],r['round'],bid))
   c.execute("update places set status='working' where id=?",(r['id'],))
  c.commit();(Path(root)/'batches'/bid).mkdir(parents=True,exist_ok=True);return bid
 finally:c.close()


def dispatch(root):
 """Called by both controller and monitor, serialized across processes."""
 root=Path(root)
 if (root/'pause_requested').exists():return
 with common_lock(root/'remote_dispatch.lock'):
  work=batches(root)
  active=[b for b in work if b['node']=='gpu2' and b['phase']=='images']
  if len(active)>1:raise ValueError('Multiple remote image batches; refuse competing backend requests')
  if active:
   repair.deploy_remote(root,active[0]['id']);return
  if sum(b['phase']=='audit' for b in work)>=4:return
  ready=next((b for b in work if b['node']=='gpu2' and b['phase']=='ready'),None)
  if ready:
   # Mark before deployment; a transport failure retains exact jobs and requires
   # existing durable request/unit checks, never an independent duplicate POST.
   c=repair.db(root)
   with c:c.execute("update batches set phase='images' where id=? and phase='ready'",(ready['id'],))
   c.close();repair.deploy_remote(root,ready['id'])


def prepare(root,bid):
 if repair.check_resources(root,'plan'):return False
 if not gate.child_step(root,bid,'plan'):return False
 c=repair.db(root)
 with c:c.execute("update batches set phase='ready' where id=? and node='gpu2' and phase='images'",(bid,))
 c.close();return True


def orchestrate(root,once=False):
 from experiments.qwen_curriculum import night_repair_control as control
 root=Path(root);cfg=repair.config(root);policy=protocol(root)
 with common_lock(root/'controller.lock'):
  while not (root/'pause_requested').exists():
   control.receive_finished_remote(root);dispatch(root);work=batches(root)
   remote=[b for b in work if b['node']=='gpu2']
   audit=[b for b in work if b['phase']=='audit']
   # Backpressure bounds audit debt; no unbounded generation disk queue.
   if len(audit)>=policy['max_waiting_audit_batches']:
    if not repair.check_resources(root,'audit'):
     if gate.child_step(root,audit[0]['id'],'audit'):repair.export(root)
    else:time.sleep(10)
    continue
   active=any(b['phase']=='images' for b in remote)
   ready=any(b['phase']=='ready' for b in remote)
   pending_plan=next((b['id'] for b in remote if b['phase']=='planning'),None)
   if not ready:
    bid=pending_plan or reserve(root,policy['steady_places'] if active else policy['bootstrap_places'],'gpu2')
    if bid:
     if prepare(root,bid):dispatch(root)
     else:time.sleep(10)
     continue
   # Audit while remote runs, then use the local GPU for its own image batch.
   if audit and not repair.check_resources(root,'audit'):
    if gate.child_step(root,audit[0]['id'],'audit'):
     repair.export(root)
     if once:return
    continue
   local=next((b for b in work if b['node']=='local' and b['phase'] in ('planning','images')),None)
   if local is None:
    bid=reserve(root,policy['steady_places'],'local')
    if bid:local=dict(id=bid,phase='planning')
   if local:
    step='plan' if local['phase']=='planning' else 'images'
    reason=repair.check_resources(root,step)
    if reason:
     atomic(root/'waiting.json',dict(reason=reason,phase=step,batch=local['id']));time.sleep(10);continue
    if gate.child_step(root,local['id'],step) and step=='images':
     summary=json.loads((root/'batches'/local['id']/'summary.json').read_text())
     if summary['saved']!=summary['planned']:raise RuntimeError('Incomplete local image batch; retain reservations')
     c=repair.db(root)
     with c:c.execute("update batches set phase='audit' where id=?",(local['id'],))
     c.close()
    continue
   if audit:continue
   c=repair.db(root);pending=c.execute("select count(*) from places where status='pending'").fetchone()[0];c.close()
   if not work and not pending:
    repair.export(root);atomic(root/'complete.json',dict(complete=True,updated_unix=time.time(),canonical_replaced=False));return
   repair.progress(root,'remote_generation');time.sleep(10)
  repair.progress(root,'paused')


def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['run']);p.add_argument('--root',type=Path,default=repair.DEFAULT);p.add_argument('--execute',action='store_true');a=p.parse_args()
 if not a.execute:raise ValueError('--execute required')
 protocol(a.root);orchestrate(a.root)

if __name__=='__main__':main()
