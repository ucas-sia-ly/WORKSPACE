import json,sqlite3
from pathlib import Path
import unittest
import tempfile
from contextlib import contextmanager

@contextmanager
def raises(error):
 try:
  yield
 except error:
  return
 raise AssertionError("Expected exception")
from experiments.qwen_curriculum import night_prompt_followup as f

def item(sid,prompt='new'):
 return dict(sample_id=sid,review_index=int(sid),proposed_prompt=prompt,latest_rejection={'generated':{'prompt':'old'}})

def test_select_exact_changed_scope():
 a,b,c=item('1'),item('2','old'),item('3')
 assert f.select_changes([a,b],[c,b,a])==[a]

def test_duplicates_refused():
 a=item('1')
 with raises(ValueError):f.select_changes([a],[a,a])

def test_parent_requires_completion_and_resolved_attempts(tmp_path):
 c=sqlite3.connect(tmp_path/'state.sqlite')
 c.executescript('create table attempts(round integer,result text);create table batches(phase text);')
 c.execute('insert into attempts values(0,NULL)');c.execute("insert into batches values('images')");c.commit()
 assert not f.parent_finished(tmp_path)
 (tmp_path/'round_one_finished.json').write_text(json.dumps({'stopped':True,'round':0}))
 assert not f.parent_finished(tmp_path)
 c.execute("update attempts set result='{}'");c.commit();assert not f.parent_finished(tmp_path)
 c.execute("update batches set phase='done'");c.commit();assert f.parent_finished(tmp_path);c.close()

def test_prepared_budget_and_exact_reviewed_sources():
 cfg=f.verify(f.DEFAULT)
 rows=f.repair.rows(f.DEFAULT/'reviewed_changes.jsonl');by_id={r['sample_id']:r for r in rows}
 c=f.repair.db(f.DEFAULT)
 jobs=[json.loads(r[0]) for r in c.execute('select job from attempts')]
 assert len(jobs)==324 and len({j['sample_id'] for j in jobs})==324
 assert sum(j['reviewed_source_replaced'] for j in jobs)==27
 assert dict(c.execute('select node,count(*) from batches group by node'))=={'local':3,'gpu2':3}
 for j in jobs:
  row=by_id[j['review_sample_id']];src=row.get('source_replacement') or row['original_source']
  target=src.get('new_source_prompt',row['proposed_prompt']) if row.get('source_replacement') else row['proposed_prompt']
  assert j['prompt']==target and j['source_sha256']==src['source_sha256']
  assert j['prompt_author']=='Codex' and not j['automatic_source_retries']
  assert j['seed']==row['latest_rejection']['generated']['seed']
  assert j['sample_id']!=row['latest_rejection']['generated']['sample_id']
 c.close()

def test_prepare_idempotent():
 before=(f.DEFAULT/'protocol.json').read_bytes()
 cfg=f.prepare(f.DEFAULT)
 assert (f.DEFAULT/'protocol.json').read_bytes()==before and cfg['expected']==324


class FollowupTests(unittest.TestCase):
 def test_selection(self):test_select_exact_changed_scope()
 def test_duplicates(self):test_duplicates_refused()
 def test_parent_barrier(self):
  with tempfile.TemporaryDirectory() as d:test_parent_requires_completion_and_resolved_attempts(Path(d))
 def test_exact_budget(self):test_prepared_budget_and_exact_reviewed_sources()
 def test_idempotency(self):test_prepare_idempotent()

if __name__=='__main__':unittest.main()
