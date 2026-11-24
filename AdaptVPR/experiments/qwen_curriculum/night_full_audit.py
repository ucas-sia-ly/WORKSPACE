"""Strict two-stage audit orchestrator. Image generation has no entry point here."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from experiments.qwen_curriculum import common,campaign
from experiments.qwen_curriculum.night_full_filter import atomic,connect,dumps,status,common_lock
DEFAULT=common.WORKSPACE_ROOT/'outputs/night_audit_v1'

def child(module,phase,root):
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',ADAPTVPR_CLIP_LOCAL_FILES_ONLY='1')
 subprocess.run([sys.executable,'-B','-m','experiments.qwen_curriculum.'+module,phase,'--root',str(root)],check=True,env=env)

def run(root):
 root=Path(root)
 with common_lock(root/'pipeline.lock'):
  atomic(root/'pipeline_state.json',dict(state='filter',generation_enabled=False,started_unix=time.time()))
  child('night_full_filter','filter',root)
  c=connect(root);pending=c.execute('select count(*) from pairs where filter_status is null').fetchone()[0];c.close()
  if pending or (root/'pause_requested').exists():
   atomic(root/'pipeline_state.json',dict(state='paused',phase='filter',remaining=pending,generation_enabled=False));return
  child('night_full_filter','export',root)
  gate=json.loads((root/'review_gate.json').read_text()) if (root/'review_gate.json').exists() else {}
  if not gate.get('ready'):
   atomic(root/'pipeline_state.json',dict(state='waiting_review_validation',generation_enabled=False));print('Filter complete; pairwise review waits for validated review gate',flush=True);return
  if gate.get('review_fingerprint')!=campaign.config_file(root/'review_protocol.json')['fingerprint']:raise ValueError('Review gate differs from frozen protocol')
  atomic(root/'pipeline_state.json',dict(state='pairwise_review_and_prompt_edits',generation_enabled=False,updated_unix=time.time()))
  child('night_full_review','review',root)
  child('night_full_review','export',root)
  c=connect(root)
  pending=c.execute("select count(*) from pairs where filter_status='pass' and review_status is null").fetchone()[0]
  repairs=c.execute("select count(*) from pairs where repair_status='pending'").fetchone()[0];c.close()
  final=dict(state='complete' if not pending and not repairs else 'paused',pending_passed_pair_reviews=pending,pending_prompt_actions=repairs,generation_enabled=False,updated_unix=time.time())
  atomic(root/'pipeline_state.json',final);print(dumps(final),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['run','status','pause','resume']);p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--execute',action='store_true');a=p.parse_args();root=a.root.resolve()
 if a.phase=='status':
  print(dumps(status(root)));print((root/'pipeline_state.json').read_text() if (root/'pipeline_state.json').exists() else 'not_started');return
 if a.phase=='pause':
  atomic(root/'pause_requested',dict(requested_unix=time.time(),mode='finish current item/batch, close model, exit'));print('Pause requested; model released after current item or small batch completes');return
 if not a.execute:raise ValueError('Explicit --execute required')
 if a.phase=='resume':
  (root/'pause_requested').unlink(missing_ok=True)
  unit='ccr-night-audit-v1.service'
  probe=subprocess.run(['systemctl','--user','show',unit,'--property=LoadState','--property=SubState'],capture_output=True,text=True)
  if 'SubState=running' in probe.stdout:print('Audit already running; pause request cleared, no duplicate task');return
  if 'LoadState=not-found' not in probe.stdout and probe.returncode==0:
   subprocess.run(['systemctl','--user','restart',unit],check=True);print('Audit resumed from committed progress');return
  cmd=['systemd-run','--user','--unit='+unit,'--description=夜景61984对审核与Prompt小改（不生图）',
   '--property=Type=exec','--property=RemainAfterExit=yes','--property=MemoryHigh=12G','--property=MemoryMax=16G','--property=MemorySwapMax=0',
   '--property=CPUQuota=500%','--property=TimeoutStopSec=1800',
   '--working-directory='+str(common.WORKSPACE_ROOT),'--setenv=PYTHONPATH='+str(common.WORKSPACE_ROOT/'AdaptVPR'),
   '--setenv=PYTHONDONTWRITEBYTECODE=1','--setenv=PYTHONUNBUFFERED=1','--setenv=OMP_NUM_THREADS=4','--setenv=OPENBLAS_NUM_THREADS=4',
   '--property=StandardOutput=append:'+str(root/'run.log'),'--property=StandardError=append:'+str(root/'run.log'),
   sys.executable,'-B','-m','experiments.qwen_curriculum.night_full_audit','run','--root',str(root),'--execute']
  subprocess.run(cmd,check=True);print('Audit launched; no image generation');return
 if not (root/'filter_protocol.json').exists() or not (root/'review_protocol.json').exists():raise ValueError('Freeze both phase protocols before launch')
 run(root)
if __name__=='__main__':main()
