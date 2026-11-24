"""Finish the first source round, then stop for a separate Codex visual review."""
import argparse,json,time,csv
from pathlib import Path
from collections import Counter
from experiments.qwen_curriculum import night_repair_staged as staged
from experiments.qwen_curriculum import night_repair_campaign as repair,campaign,common
from experiments.qwen_curriculum import night_repair_acceptance as accept
from experiments.qwen_curriculum.night_full_filter import atomic

def protocol(root):
 parent=staged.protocol(root)
 return campaign.frozen(Path(root)/'round_one_stop_protocol_v1.json',dict(version=1,
  parent_staged=parent['fingerprint'],code_sha256=common.file_sha256(__file__),
  last_automatic_source_round=0,stop_after=['all_round_0_prompts','all_round_0_images','all_round_0_audits'],
  automatic_alternate_sources=False,next_phase='Separate user-directed Codex image review, not automatic Qwen retries',
  accepted_verdicts=['pass','unknown'],audit_evidence_and_thresholds_unchanged=True))

def handoff(root):
 root=Path(root);directory=root/'round_one_review';directory.mkdir(exist_ok=True)
 c=repair.db(root);counts=Counter();paths={k:directory/(k+'.jsonl') for k in ('reject_places','planning_failures')}
 files={k:p.with_suffix(p.suffix+'.tmp').open('w') for k,p in paths.items()}
 table=directory/'reject_places.tsv';tmp=table.with_suffix('.tsv.tmp')
 try:
  with tmp.open('w',newline='') as f:
   writer=csv.writer(f,delimiter='\t');writer.writerow(['sample_id','city','place_id','source','old_generated','new_generated','rejection_reasons'])
   query='select a.*,p.city,p.place_id,p.original,p.defect,p.status from attempts a join places p on p.id=a.id where a.round=0 order by p.ordinal'
   for a in c.execute(query):
    if a['result'] is None:raise ValueError('Cannot finish round one with an unaudited/unresolved attempt: '+a['id'])
    v=json.loads(a['result']);counts[v['decision']]+=1
    if v['decision'] not in ('reject','prompt_invalid_no_generation','no_source'):continue
    original=json.loads(a['original']);generated=v.get('generated');reasons=v.get('base',{}).get('rejection_reasons',[]) or v.get('content',{}).get('content_audit',{}).get('reasons',[])
    record=dict(sample_id=a['id'],city=a['city'],place_id=a['place_id'],source_round=0,
     original_pair=original,first_rejection=json.loads(a['defect']),repair_planning=json.loads(a['planning']) if a['planning'] else None,
     repaired_pair=generated,repair_audit=v,rejection_reasons=reasons,
     review_status='awaiting_codex_visual_review',next_action=None,regeneration_authorized_by_this_export=False)
    key='reject_places' if v['decision']=='reject' else 'planning_failures'
    files[key].write(repair.dumps(record)+'\n')
    if key=='reject_places':writer.writerow([a['id'],a['city'],a['place_id'],original['source_path'],original['output_path'],generated['output_path'],','.join(reasons)])
 finally:
  c.close()
  for f in files.values():f.close()
 for p in paths.values():p.with_suffix(p.suffix+'.tmp').replace(p)
 tmp.replace(table)
 summary=dict(round=0,outcomes=dict(counts),images_audited=sum(counts[k] for k in ('pass','unknown','reject')),
  accepted=counts['pass']+counts['unknown'],reject_places=counts['reject'],
  prompt_failures=counts['prompt_invalid_no_generation'],source_failures=counts['no_source'],
  exports={p.name:common.file_sha256(p) for p in [*paths.values(),table]},updated_unix=time.time(),
  all_rejects_visually_reviewed=False,all_images_passed=False)
 atomic(directory/'summary.json',summary)
 (directory/'README.md').write_text('''# 第一轮结束后的 Codex 逐图复核交接

`reject_places.jsonl`：全部仍Reject地点，包含真实源图、旧图、新图、Prompt及审核证据。`reject_places.tsv`便于定位。

`planning_failures.jsonl`单列未生图的Prompt/源图失败，不能当作图片Reject或伪造为通过。

下一步由用户另行安排Codex查看每一张Reject：难处理的太阳光影考虑同city/place换源；其他问题先根据实际图像小改Prompt。不自动执行该阶段，不以降低阈值或篡改审核标签保证通过。复核动作和是否需要换源目前均未填写。

`summary.json`记录真实数量及导出清单哈希。原记录和队列均保留。
''')
 return summary

class RoundOneFinished(Exception):pass
BASE_NEXT=staged.next_round
BASE_RESERVE=staged.reserve
def next_round(root):
 number=BASE_NEXT(root)
 if number is None or number>0:raise RoundOneFinished()
 return number

def reserve(root,roundno,size):
 if roundno!=0:raise RoundOneFinished()
 return BASE_RESERVE(root,roundno,size)

def run(root):
 root=Path(root);protocol(root)
 if (root/'round_one_finished.json').exists():
  print('First round is finished. Automatic next-source generation is disabled.',flush=True);return
 staged.next_round=next_round;staged.reserve=reserve
 try:staged.run(root)
 except RoundOneFinished:
  # staged.run has released controller.lock. Reacquire while freezing exports.
  with repair.common_lock(root/'controller.lock'):
   accept.export(root);summary=handoff(root)
   staged.state(root,'stopped_after_round_one',0)
   atomic(root/'round_one_finished.json',dict(round=0,stopped=True,full_campaign_complete=False,
    automatic_alternate_sources=False,next_phase_requires_user_instruction=True,summary=summary,updated_unix=time.time()))
   print(repair.dumps(dict(status='stopped_after_round_one',summary=summary)),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['run']);p.add_argument('--root',type=Path,default=repair.DEFAULT);p.add_argument('--execute',action='store_true');a=p.parse_args()
 if not a.execute:raise ValueError('--execute required')
 run(a.root)

if __name__=='__main__':main()
