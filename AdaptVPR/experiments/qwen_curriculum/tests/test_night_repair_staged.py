import tempfile,unittest,json
from pathlib import Path
from unittest.mock import patch
from experiments.qwen_curriculum import night_repair_staged as staged,night_repair_campaign as repair

class StageTests(unittest.TestCase):
 def test_dispatch_forbidden_during_all_prompt_and_audit_stages(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   for phase in ('prompts','audit'):
    staged.state(root,phase,0)
    with patch.object(staged,'work') as work,patch.object(repair,'deploy_remote') as deploy:
     staged.dispatch(root);work.assert_not_called();deploy.assert_not_called()
 def test_dispatch_lock_contention_is_a_wait_not_a_crash(self):
  with patch.object(staged,'_dispatch',side_effect=BlockingIOError):
   self.assertIsNone(staged.dispatch(Path('/unused')))
 def test_transfer_lock_contention_does_not_duplicate_or_fail(self):
  from experiments.qwen_curriculum import night_repair_control as control
  with patch.object(repair,'common_lock',side_effect=BlockingIOError),patch.object(control,'_receive_finished_remote') as receive:
   self.assertIsNone(control.receive_finished_remote(Path('/unused')));receive.assert_not_called()
 def test_reserve_only_current_source_round_and_keeps_node_ownership(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);c=repair.db(root)
   with c:
    for i,r in enumerate([0,0,1]):c.execute('insert into places(id,ordinal,round,status) values(?,?,?,?)',(str(i),i,r,'pending'))
   c.close();first=staged.reserve(root,0,1);self.assertEqual(staged.reserve(root,0,1),first)
   c=repair.db(root)
   with c:c.execute("update batches set phase='ready' where id=?",(first,))
   c.close();second=staged.reserve(root,0,1);self.assertNotEqual(first,second)
   c=repair.db(root);self.assertEqual(c.execute('select count(distinct node) from batches').fetchone()[0],2)
   self.assertEqual(c.execute("select status from places where id='2'").fetchone()[0],'pending');c.close()
 def test_unfinished_round_prevents_starting_retry_round(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);c=repair.db(root)
   with c:
    c.execute("insert into places(id,ordinal,round,status) values('p',0,1,'pending')")
    c.execute("insert into batches(id,phase,created) values('old','audit',0)")
    c.execute("insert into attempts(id,round,batch) values('old',0,'old')")
   c.close();self.assertEqual(staged.next_round(root),0)
   c=repair.db(root)
   with c:c.execute("update batches set phase='done'")
   c.close();self.assertEqual(staged.next_round(root),1)
 def test_resume_does_not_restart_remote_images_during_prompt_stage(self):
  from experiments.qwen_curriculum import night_repair_control as control
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'pause_requested').touch();(root/'staged_protocol_v1.json').write_text('{}')
   with patch.object(control,'unit_info',return_value='LoadState=not-found'),patch.object(repair,'config'),patch.object(repair,'remote_command') as remote,patch.object(control,'launch'):
    control.resume(root)
   self.assertEqual(remote.call_count,1);self.assertNotIn('restart',str(remote.call_args))

if __name__=='__main__':unittest.main()
