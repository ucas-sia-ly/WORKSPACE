import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from experiments.qwen_curriculum import night_dataset_release as release

class DatasetPolicyTests(unittest.TestCase):
 def test_reject_included_without_false_pass(self):
  r=release.as_dataset({'passed':False,'eligible_for_training':False},'id','second_round','reject',{'fingerprint':'test'})
  self.assertTrue(r['eligible_for_training']);self.assertTrue(r['quality_gate_overridden_by_user'])
  self.assertFalse(r['passed']);self.assertEqual(r['raw_quality_verdict'],'reject')
  self.assertEqual(r['place_split_eligibility'],'not_evaluated_by_this_export')

 def test_third_unreviewed_not_pass(self):
  r=release.as_dataset({},'id','third_round','not_audited',{'fingerprint':'test'})
  self.assertEqual(r['quality_audit_status'],'not_run_user_requested');self.assertNotIn('passed',r)

 def test_integrity_only_and_resume(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);d=root/'batches'/'batch';(d/'images').mkdir(parents=True)
   result={'sample_id':'new','prompt':'new night','seed':4}
   for key in ('source','output','raw_output'):
    path=d/'images'/key;path.write_bytes(key.encode())
    result[key+'_path']=str(path);result[key+'_sha256']=release.common.file_sha256(path)
   job={**result,'review_sample_id':'old'}
   c=release.repair.db(root)
   with c:
    c.execute("insert into places(id,status) values('old','working')")
    c.execute("insert into batches(id,phase) values('batch','audit')")
    c.execute("insert into attempts(id,round,batch) values('old',0,'batch')")
   c.close()
   with patch.object(release.repair,'load_image_plan',return_value=({},[job])),patch.object(release.repair,'rows',return_value=[result]):
    self.assertTrue(release.integrity_only(root,'batch'));self.assertTrue(release.integrity_only(root,'batch'))
   c=release.repair.db(root);e=json.loads(c.execute('select result from attempts').fetchone()[0]);c.close()
   self.assertEqual(e['decision'],'not_audited');self.assertFalse(e['quality_audit_performed']);self.assertTrue(e['integrity_checked'])

 def test_parent_barrier_required(self):
  with tempfile.TemporaryDirectory() as t:
   with patch.object(release.follow,'parent_finished',return_value=False),patch.object(release,'protocol',return_value={}):
    with self.assertRaises(ValueError):release.export_dataset(Path(t),Path(t),'second_round')

if __name__=='__main__':unittest.main()
