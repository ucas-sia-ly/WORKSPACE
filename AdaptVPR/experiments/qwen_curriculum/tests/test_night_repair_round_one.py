import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from experiments.qwen_curriculum import night_repair_round_one as one,night_repair_campaign as repair

class RoundOneTests(unittest.TestCase):
 def test_current_round_still_runs_but_no_automatic_alternate(self):
  with patch.object(one,'BASE_NEXT',return_value=0):self.assertEqual(one.next_round(Path('/unused')),0)
  for later in (1,2,None):
   with patch.object(one,'BASE_NEXT',return_value=later),self.assertRaises(one.RoundOneFinished):one.next_round(Path('/unused'))
 def test_retry_cannot_be_reserved(self):
  with patch.object(one,'BASE_RESERVE') as reserve:
   with self.assertRaises(one.RoundOneFinished):one.reserve(Path('/unused'),1,64)
   reserve.assert_not_called()
 def test_exports_rejects_separately_from_text_failure_and_unknown(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);c=repair.db(root)
   with c:
    for ordinal,decision in enumerate(['reject','unknown','pass','prompt_invalid_no_generation']):
     source=dict(source_path='source.jpg',output_path='old.png')
     c.execute('insert into places(id,ordinal,city,place_id,original,defect,status) values(?,?,?,?,?,?,?)',(str(ordinal),ordinal,'City',ordinal,json.dumps(source),'{}','pending'))
     v=dict(decision=decision)
     if decision!='prompt_invalid_no_generation':v.update(generated=dict(output_path='new.png'),base=dict(rejection_reasons=['weather'] if decision=='reject' else []))
     c.execute('insert into attempts(id,round,result) values(?,?,?)',(str(ordinal),0,json.dumps(v)))
   c.close();summary=one.handoff(root)
   self.assertEqual(summary['reject_places'],1);self.assertEqual(summary['accepted'],2);self.assertEqual(summary['prompt_failures'],1)
   self.assertEqual(len((root/'round_one_review/reject_places.jsonl').read_text().splitlines()),1)
   item=json.loads((root/'round_one_review/reject_places.jsonl').read_text());self.assertIsNone(item['next_action']);self.assertFalse(item['regeneration_authorized_by_this_export'])
 def test_unfinished_audit_never_marked_complete(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);c=repair.db(root)
   with c:
    c.execute("insert into places(id,ordinal,original,defect) values('p',0,'{}','{}')")
    c.execute("insert into attempts(id,round) values('p',0)")
   c.close()
   with self.assertRaisesRegex(ValueError,'unaudited'):one.handoff(root)
   self.assertFalse((root/'round_one_finished.json').exists())
 def test_resume_after_boundary_does_not_launch(self):
  from experiments.qwen_curriculum import night_repair_control as control
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'round_one_finished.json').write_text('{}')
   with patch.object(control,'launch') as launch,patch.object(repair,'remote_command') as remote:
    control.resume(root);launch.assert_not_called();remote.assert_not_called()

if __name__=='__main__':unittest.main()
