import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from experiments.qwen_curriculum import night_repair_campaign as repair,night_repair_pipeline as pipe

class PipelineTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.c=repair.db(self.root)
 def tearDown(self):self.c.close();self.tmp.cleanup()
 def batch(self,id,node,phase):
  with self.c:self.c.execute('insert into batches(id,node,phase,created) values(?,?,?,0)',(id,node,phase))
 def test_audit_batch_does_not_block_future_remote_prompt_reservation(self):
  self.batch('old','gpu2','audit')
  with self.c:
   for i in range(3):self.c.execute('insert into places(id,ordinal,city,place_id,original,defect) values(?,?,?,?,?,?)',(str(i),i,'City',i,'{}','{}'))
  bid=pipe.reserve(self.root,2,'gpu2');self.assertNotEqual(bid,'old')
  self.assertEqual(self.c.execute('select count(*) from attempts where batch=?',(bid,)).fetchone()[0],2)
  self.assertEqual(pipe.reserve(self.root,2,'gpu2'),bid)
 def test_pause_never_dispatches(self):
  self.batch('next','gpu2','ready');(self.root/'pause_requested').touch()
  with patch.object(repair,'deploy_remote') as deploy:pipe.dispatch(self.root);deploy.assert_not_called()
 def test_ready_dispatches_even_while_previous_awaits_audit(self):
  self.batch('old','gpu2','audit');self.batch('next','gpu2','ready')
  with patch.object(repair,'deploy_remote') as deploy:pipe.dispatch(self.root);deploy.assert_called_once_with(self.root,'next')
  self.assertEqual(self.c.execute("select phase from batches where id='next'").fetchone()[0],'images')
 def test_active_remote_blocks_second_request_stream(self):
  self.batch('current','gpu2','images');self.batch('next','gpu2','ready')
  with patch.object(repair,'deploy_remote') as deploy:pipe.dispatch(self.root);deploy.assert_called_once_with(self.root,'current')
  self.assertEqual(self.c.execute("select phase from batches where id='next'").fetchone()[0],'ready')
 def test_transport_failure_retains_same_active_identity(self):
  self.batch('next','gpu2','ready')
  with patch.object(repair,'deploy_remote',side_effect=RuntimeError('transport')):
   with self.assertRaises(RuntimeError):pipe.dispatch(self.root)
  self.assertEqual(self.c.execute("select phase from batches where id='next'").fetchone()[0],'images')
 def test_backpressure_bounds_unaudited_backlog(self):
  for i in range(4):self.batch(str(i),'gpu2','audit')
  self.batch('next','gpu2','ready')
  with patch.object(repair,'deploy_remote') as deploy:pipe.dispatch(self.root);deploy.assert_not_called()
 def test_duplicate_remote_active_is_rejected(self):
  self.batch('a','gpu2','images');self.batch('b','gpu2','images')
  with patch.object(repair,'deploy_remote') as deploy:
   with self.assertRaises(ValueError):pipe.dispatch(self.root)
   deploy.assert_not_called()

if __name__=='__main__':unittest.main()
