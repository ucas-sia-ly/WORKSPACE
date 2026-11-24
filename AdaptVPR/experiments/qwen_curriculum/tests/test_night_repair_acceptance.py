import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from experiments.qwen_curriculum import night_repair_acceptance as policy,night_repair_campaign as repair

class AcceptanceTests(unittest.TestCase):
 def setup_db(self,root,decision='unknown',status='pending',roundno=1):
  c=repair.db(root)
  result=json.dumps(dict(decision=decision,generated=dict(sample_id='new')))
  with c:
   c.execute('insert into places(id,ordinal,city,place_id,original,defect,status,round) values(?,?,?,?,?,?,?,?)',('p',0,'City',1,'{}','{}',status,roundno))
   c.execute('insert into attempts(id,round,result) values(?,?,?)',('p',0,result))
  c.close();return result
 def test_unknown_stops_source_retry_without_changing_verdict(self):
  with tempfile.TemporaryDirectory() as d,patch.object(policy,'protocol',return_value=dict(fingerprint='test')):
   root=Path(d);before=self.setup_db(root);self.assertEqual(len(policy.reconcile(root)),1)
   c=repair.db(root);self.assertEqual(tuple(c.execute('select status,round from places').fetchone()),('passed',0));self.assertEqual(c.execute('select result from attempts').fetchone()[0],before);c.close()
   self.assertEqual(policy.reconcile(root),[])
 def test_reject_remains_pending(self):
  with tempfile.TemporaryDirectory() as d,patch.object(policy,'protocol',return_value=dict(fingerprint='test')):
   root=Path(d);self.setup_db(root,'reject');self.assertEqual(policy.reconcile(root),[])
   c=repair.db(root);self.assertEqual(tuple(c.execute('select status,round from places').fetchone()),('pending',1));c.close()
 def test_rechecked_original_unknown_accepted_without_new_job(self):
  with tempfile.TemporaryDirectory() as d,patch.object(policy,'protocol',return_value=dict(fingerprint='test')):
   root=Path(d);self.setup_db(root,'pass','unknown_recheck',0);policy.reconcile(root)
   c=repair.db(root);self.assertEqual(c.execute('select status from places').fetchone()[0],'passed_original_unknown');self.assertEqual(c.execute('select count(*) from attempts').fetchone()[0],1);c.close()
 def test_already_reserved_later_source_is_not_silently_changed(self):
  with tempfile.TemporaryDirectory() as d,patch.object(policy,'protocol',return_value=dict(fingerprint='test')):
   root=Path(d);self.setup_db(root);c=repair.db(root)
   with c:c.execute('insert into attempts(id,round) values(?,?)',('p',1))
   c.close()
   with self.assertRaisesRegex(ValueError,'already reserved'):policy.reconcile(root)
   c=repair.db(root);self.assertEqual(c.execute('select status from places').fetchone()[0],'pending');c.close()

if __name__=='__main__':unittest.main()
