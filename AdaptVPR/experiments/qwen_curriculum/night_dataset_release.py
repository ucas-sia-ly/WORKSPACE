"""User policy amendment: audit round two, include its Rejects, skip round-three QA.

Frozen generation jobs, historical verdicts and the parent controller are untouched.
The only third-round check is exact source/Prompt/output integrity, not quality QA.
"""
from __future__ import annotations
import argparse,json,sqlite3,time
from collections import Counter
from pathlib import Path
from experiments.qwen_curriculum import common,campaign,night_prompt_followup as follow,night_repair_campaign as repair
from experiments.qwen_curriculum.night_full_filter import atomic,common_lock
RELEASE=common.WORKSPACE_ROOT/'outputs/night_dataset_release_v2'

def protocol(root):
 return campaign.frozen(Path(root)/'dataset_use_policy_v1.json',dict(version=1,
  authorizing_instruction='第二轮完成之后继续审核；第三轮不审核；第二轮依旧 Reject 也作为数据集使用',
  parent_quality_audit_required=True,parent_rejects_included=True,third_round_quality_audit=False,
  third_round_integrity_check=True,raw_verdicts_immutable=True,automatic_retries=False,
  code_path=str(Path(__file__).resolve()),code_sha256=common.file_sha256(__file__),
  frozen_followup_protocol=repair.config(root)['fingerprint'],release_root=str(RELEASE)))

def as_dataset(row,identity,stage,verdict,policy):
 row=dict(row)
 # New derivative flags express inclusion, never silently rewrite historical QA.
 row.update(dataset_identity=identity,dataset_version='night_dataset_release_v2',dataset_stage=stage,
  raw_quality_verdict=verdict,quality_audit_status='not_run_user_requested' if stage=='third_round' else 'completed' if stage=='second_round' else 'historical_record',
  included_under_user_policy=True,quality_gate_overridden_by_user=verdict=='reject',
  prior_eligible_for_training=row.get('eligible_for_training'),eligible_for_training=True,
  eligibility_basis='user authorized image-quality inclusion; existing train/validation place exclusions still required',
  place_split_eligibility='not_evaluated_by_this_export',dataset_policy_fingerprint=policy['fingerprint'])
 return row

def integrity_only(root,bid):
 root=Path(root);d=root/'batches'/bid;cfg,jobs=repair.load_image_plan(d)
 results=repair.rows(d/'results.jsonl');by_id={r['sample_id']:r for r in results}
 if len(results)!=len(jobs) or len(by_id)!=len(jobs):raise ValueError('Third-round result cardinality mismatch')
 checked=[]
 for j in jobs:
  r=by_id[j['sample_id']]
  if any(r[k]!=j[k] for k in ('source_path','source_sha256','prompt','seed')):raise ValueError('Third-round source/Prompt/seed mismatch')
  for key in ('source','output','raw_output'):
   path=Path(r[key+'_path']).resolve()
   if key!='source' and not path.is_relative_to((d/'images').resolve()):raise ValueError('Output escaped batch')
   if common.file_sha256(path)!=r[key+'_sha256']:raise ValueError('Third-round file integrity failed')
  checked.append((j,r))
 c=repair.db(root)
 try:
  with c:
   for j,r in checked:
    e=dict(decision='not_audited',quality_audit_performed=False,integrity_checked=True,generated=r,
     included_under_user_policy=True,no_quality_pass_claim=True)
    current=c.execute('select result from attempts where id=? and round=0',(j['review_sample_id'],)).fetchone()
    if current is None:raise ValueError('No exact review identity')
    if current[0] and json.loads(current[0])!=e:raise ValueError('Refuse to overwrite historical verdict')
    c.execute('update attempts set result=? where id=? and round=0',(repair.dumps(e),j['review_sample_id']))
    c.execute("update places set status='included_unreviewed',round=0 where id=?",(j['review_sample_id'],))
   c.execute("update batches set phase='done' where id=?",(bid,))
 finally:c.close()
 atomic(d/'integrity_only_receipt.json',dict(images=len(checked),quality_audit_performed=False,source_and_output_hashes_verified=True))
 return True

def export_followup(root):
 root=Path(root);c=repair.db(root);count=0
 tmp=root/'dataset_included_followup.jsonl.tmp'
 try:
  with tmp.open('w') as f:
   for a in c.execute('select id,result from attempts order by id'):
    if not a['result']:raise ValueError('Unresolved third-round job')
    e=json.loads(a['result'])
    if e['decision']!='not_audited' or not e['integrity_checked']:raise ValueError('Unexpected follow-up outcome')
    f.write(repair.dumps(dict(review_sample_id=a['id'],generated=e['generated'],raw_audit_decision='not_audited',included_under_user_policy=True,quality_audit_performed=False))+'\n');count+=1
 finally:c.close()
 if count!=324:raise ValueError('Third round budget mismatch')
 tmp.replace(root/'dataset_included_followup.jsonl')
 atomic(root/'complete.json',dict(total=count,dataset_included=count,quality_audited=0,quality_status='not_audited',automatic_retries=False,updated_unix=time.time()))

def export_dataset(parent,root,stage):
 """Stream 61,984 rows; override by stable original ID, no image copying."""
 parent=Path(parent);root=Path(root);policy=protocol(root)
 if not follow.parent_finished(parent):raise ValueError('Parent round not fully audited')
 destination=Path(policy['release_root']);destination.mkdir(parents=True,exist_ok=True)
 name='second_round_manifest.jsonl' if stage=='second_round' else 'manifest.jsonl'
 target=destination/name;tmp=target.with_suffix('.jsonl.tmp')
 c=sqlite3.connect('file:'+str(parent/'state.sqlite')+'?mode=ro',uri=True)
 third={}
 if stage=='third_round':
  if not (root/'complete.json').exists():raise ValueError('Third-round generation incomplete')
  third={r['review_sample_id']:r['generated'] for r in repair.rows(root/'dataset_included_followup.jsonl')}
  if len(third)!=324:raise ValueError('Third-round inclusion scope mismatch')
 counts=Counter();seen=set()
 try:
  with tmp.open('w') as f:
   for original in repair.rows(common.WORKSPACE_ROOT/'outputs/night_dataset_v1/manifest.jsonl'):
    sid=original['sample_id'];identity=(original['city'],int(original['place_id']))
    if identity in seen:raise ValueError('Duplicate place in dataset')
    seen.add(identity);previous=c.execute('select result from attempts where id=? and round=0',(sid,)).fetchone()
    if sid in third:
     row=as_dataset(third[sid],sid,'third_round','not_audited',policy)
    elif previous and previous[0] and json.loads(previous[0]).get('generated'):
     e=json.loads(previous[0]);row=as_dataset(e['generated'],sid,'second_round',e['decision'],policy)
    else:
     row=as_dataset(original,sid,'original',original.get('raw_audit_decision','historical_not_reclassified'),policy)
    if (row['city'],int(row['place_id']))!=identity:raise ValueError('Cross-place replacement')
    counts[row['dataset_stage']]+=1;counts['verdict:'+row['raw_quality_verdict']]+=1
    f.write(repair.dumps(row)+'\n')
 finally:c.close()
 if len(seen)!=61984:raise ValueError('Full dataset coverage changed')
 tmp.replace(target)
 atomic(destination/(name+'.receipt.json'),dict(total=len(seen),counts=dict(counts),manifest_sha256=common.file_sha256(target),
  baseline_unchanged=True,images_copied=False,training_launched=False,updated_unix=time.time(),policy_fingerprint=policy['fingerprint']))
 (destination/'README.md').write_text('''# 夜景数据集 v2（用户授权纳入）

`second_round_manifest.jsonl` 在第二轮全部审核后导出：每地点一条，有第二轮图片时选用第二轮图片，包括 Reject；文本失败未生图的地点沿用原图。

`manifest.jsonl` 在第三轮 324 张生成后导出：这 324 条以第三轮图片覆盖对应地点，其余沿用第二轮汇总。第三轮不做质量审核，仅核对文件哈希、精确源图、Prompt、seed 和数量；其质量状态明确为 `not_audited`，不会伪造 Pass。

原始审核标签与图片均保留。`eligible_for_training=true` 仅表示用户授权取消图像质量筛选；不表示地点划分已审核。训练仍须应用项目已有的训练/校准/验证地点隔离及排除清单。本任务不启动训练。

这是使用既有图片路径的独立 manifest，不复制六万张图，不覆盖旧 `night_dataset_v1`。新数据集的原始样本身份在 `dataset_identity`，实际生成图片身份在 `sample_id`；换源图的精确映射来自对应生成记录。
''')
 return target

def run(root):
 root=Path(root);follow.verify(root);policy=protocol(root)
 if common.file_sha256(__file__)!=policy['code_sha256']:raise ValueError('Policy adapter changed')
 with common_lock(root/'dataset_policy_controller.lock'):
  follow.status(root,'waiting_parent_audit')
  while not follow.parent_finished(follow.PARENT):
   if (root/'pause_requested').exists():return
   time.sleep(20)
  export_dataset(follow.PARENT,root,'second_round')
  base=repair.child_step
  def child(parent,bid,step):
   if step=='audit':return integrity_only(parent,bid)
   return base(parent,bid,step)
  repair.child_step=child;follow.export_results=export_followup
  follow.run(root)
  if (root/'complete.json').exists():export_dataset(follow.PARENT,root,'third_round')

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--root',type=Path,default=follow.DEFAULT);p.add_argument('--execute',action='store_true');a=p.parse_args()
 if a.action=='prepare':print(repair.dumps(protocol(a.root)))
 else:
  if not a.execute:raise ValueError('--execute required')
  run(a.root)
if __name__=='__main__':main()
