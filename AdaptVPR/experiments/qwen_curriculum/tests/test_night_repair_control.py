import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from experiments.qwen_curriculum import night_repair_campaign as repair,night_repair_control as control,common

class ControlTests(unittest.TestCase):
 def test_old_text_error_does_not_recover_new_integrity_failure(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'run.log').write_text('Qwen prompt invalid after bounded retry\n"phase": "plan"\nSource bytes changed')
   (root/'progress.json').write_text(json.dumps(dict(phase='plan')))
   self.assertFalse(control.recover_prompt_only(root))
 def test_generation_failure_is_never_text_retried(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'run.log').write_text('Qwen prompt invalid after bounded retry')
   (root/'progress.json').write_text(json.dumps(dict(phase='images')))
   self.assertFalse(control.recover_prompt_only(root))
 def test_invalid_text_without_job_advances_bounded_source_round(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);c=repair.db(root);source=dict(prompt='Original source prompt')
   with c:
    c.execute('insert into places(id,ordinal,city,place_id,original,defect,status) values(?,?,?,?,?,?,?)',('old',0,'City',7,json.dumps(source),'{}','working'))
    c.execute('insert into batches(id,created) values(?,?)',('batch',0))
    c.execute('insert into attempts(id,round,batch) values(?,?,?)',('old',0,'batch'))
   p=root/'batches/batch';p.mkdir(parents=True)
   (p/('prompt_raw_'+common.fingerprint(['old',0])+'.json')).write_text(json.dumps(dict(response='not JSON')))
   (root/'run.log').write_text('"phase": "plan"\nQwen prompt invalid after bounded retry')
   (root/'progress.json').write_text(json.dumps(dict(phase='plan')))
   self.assertTrue(control.recover_prompt_only(root))
   self.assertEqual(tuple(c.execute('select status,round from places').fetchone()),('pending',1));self.assertEqual(c.execute('select job from attempts').fetchone()[0],None)
   self.assertFalse((p/'attempts.jsonl').exists());c.close()
 def test_running_controller_resume_does_not_clear_pause(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'pause_requested').touch()
   with patch.object(control,'unit_info',return_value='ActiveState=active'),patch.object(repair,'remote_command') as remote:
    control.resume(root);remote.assert_not_called();self.assertTrue((root/'pause_requested').exists())
class RemoteCompletionTests(unittest.TestCase):
 def test_incomplete_remote_does_not_copy_or_advance(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);c=repair.db(root)
   with c:c.execute("insert into batches(id,phase,created,node) values('b','images',0,'gpu2')")
   c.close()
   with patch.object(repair,'remote_command',return_value=json.dumps(dict(state='running',saved=1,planned=4))),patch.object(control.subprocess,'run') as sync:
    control.receive_finished_remote(root);sync.assert_not_called()
   c=repair.db(root);self.assertEqual(c.execute('select phase from batches').fetchone()[0],'images');c.close()

 def test_corrupt_remote_output_never_advances_to_audit(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);d=root/'batches/b';(d/'images').mkdir(parents=True)
   image=d/'images/new.png';image.write_bytes(b'changed output')
   job=dict(sample_id='qnightrepair_test',source_path='source.jpg',source_sha256='sourcehash',prompt='exact qwen prose',seed=42)
   result={**job,'output_path':str(image),'output_sha256':'incorrect'}
   (d/'results.jsonl').write_text(json.dumps(result)+'\n');c=repair.db(root)
   with c:c.execute("insert into batches(id,phase,created,node) values('b','images',0,'gpu2')")
   c.close()
   with patch.object(repair,'remote_command',return_value=json.dumps(dict(state='domain_complete',saved=1,planned=1))),patch.object(control.subprocess,'run'),patch.object(repair,'load_image_plan',return_value=({},[job])):
    with self.assertRaisesRegex(ValueError,'hash invalid'):control.receive_finished_remote(root)
   c=repair.db(root);self.assertEqual(c.execute('select phase from batches').fetchone()[0],'images');c.close();self.assertFalse((d/'primary_received.json').exists())

 def test_completed_artifacts_recovered_without_live_transient_unit(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);d=root/'batches/b';(d/'images').mkdir(parents=True)
   source=d/'source.jpg';source.write_bytes(b'source');image=d/'images/new.png';image.write_bytes(b'output');raw=d/'images/full.png';raw.write_bytes(b'raw')
   job=dict(sample_id='qnightrepair_test',source_path=str(source),source_sha256=common.file_sha256(source),prompt='exact qwen prose',seed=42)
   result={**job,'output_path':str(image),'output_sha256':common.file_sha256(image),'raw_output_path':str(raw),'raw_output_sha256':common.file_sha256(raw)}
   (d/'results.jsonl').write_text(json.dumps(result)+'\n');c=repair.db(root)
   with c:c.execute("insert into batches(id,phase,created,node) values('b','images',0,'gpu2')")
   c.close()
   with patch.object(repair,'remote_command',return_value=json.dumps(dict(state='domain_complete',saved=1,planned=1))),patch.object(control.subprocess,'run'),patch.object(repair,'load_image_plan',return_value=({},[job])):
    control.receive_finished_remote(root)
   c=repair.db(root);self.assertEqual(c.execute('select phase from batches').fetchone()[0],'audit');c.close();self.assertTrue((d/'primary_received.json').exists())

if __name__=='__main__':unittest.main()
