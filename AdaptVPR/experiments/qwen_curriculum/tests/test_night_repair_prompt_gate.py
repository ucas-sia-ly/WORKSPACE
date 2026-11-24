import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from experiments.qwen_curriculum import night_repair_campaign as repair,night_repair_prompt_gate as gate

def response(text):
 return dict(source_summary='An existing brick facade and opaque door.',prompt_issues=[],audit_response='Use off-frame light on the opaque door.',style_id='neutral_led',appearance_prompt=text)

GOOD='Transform this scene into a realistic nighttime photograph. A plausible off-frame light softly illuminates the existing brick facade and opaque metal door without making them emit light. Replace daylight shadows with coherent nighttime shading, preserving the road, signs, window layout and all permanent geometry. The sky is dark blue, and useful place details remain readable. Do not add fixtures or structures.'

class PromptGateTests(unittest.TestCase):
 def test_valid_dark_blue_night_and_reflective_door_allowed(self):
  self.assertEqual(gate.validate(response(GOOD))['appearance_prompt'],GOOD)
 def test_observed_bad_daylight_twilight_and_emission_instructions_rejected(self):
  for text in ['Preserve the original daylight scene.','The scene remains daylight.','The sky remains bright blue.','Transform the scene into a twilight setting.','The metal door emits a warm glow.']:
   with self.subTest(text=text),self.assertRaises(ValueError):gate.validate(response(GOOD+' '+text))
 def test_final_pass_has_only_real_source_not_generated_pixels(self):
  task=gate.prepare_task(dict(defect=json.dumps(dict(first_pass=dict(rejection_reasons=['geometry'])))),dict(source_path='source.jpg',source_sha256='sourcehash'),dict(appearance_prompt=GOOD,prompt_issues=[]))
  self.assertEqual(task['images'],[dict(path='source.jpg',sha256='sourcehash')]);self.assertIn('REAL SOURCE',task['text'])
 def test_frozen_plan_is_never_rewritten(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);d=root/'batches/b';d.mkdir(parents=True);(d/'plan_config.json').write_text('{}')
   with patch.object(gate,'amendment') as amendment,self.assertRaisesRegex(ValueError,'Never revise'):
    gate.finalize(root,'b',{})
   amendment.assert_not_called()
 def test_pause_creates_no_new_image_jobs(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'batches/b').mkdir(parents=True);(root/'pause_requested').touch();c=repair.db(root)
   with c:
    c.execute('insert into places(id,ordinal,city,place_id,original,defect,status) values(?,?,?,?,?,?,?)',('old',0,'City',7,'{}','{}','working'))
    c.execute('insert into batches(id,created) values(?,?)',('b',0))
    c.execute('insert into attempts(id,round,batch,planning,job) values(?,?,?,?,?)',('old',0,'b','{}','{}'))
   c.close()
   amendment=dict(fingerprint='gate',code_path='gate.py',code_sha256='hash',vlm_batch_size=4)
   with patch.object(gate,'amendment',return_value=amendment),patch.object(gate,'prepare_amended_vlm') as vlm:
    self.assertFalse(gate.finalize(root,'b',dict(model_sha256={},files_sha256={})))
    vlm.return_value.ask.assert_not_called();vlm.return_value.close.assert_called_once()

 def test_finalization_preserves_draft_round_and_skips_committed_text_on_resume(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'batches/b').mkdir(parents=True);c=repair.db(root)
   plan=dict(appearance_prompt=GOOD,prompt_issues=[]);job=dict(source_path='source.jpg',source_sha256='sourcehash')
   with c:
    c.execute('insert into places(id,ordinal,city,place_id,original,defect,status) values(?,?,?,?,?,?,?)',('old',0,'City',7,'{}','{}','working'))
    c.execute('insert into batches(id,created) values(?,?)',('b',0))
    c.execute('insert into attempts(id,round,batch,planning,job,source) values(?,?,?,?,?,?)',('old',0,'b',json.dumps(plan),json.dumps(job),'{}'))
   c.close();amendment=dict(fingerprint='gate',code_path='gate.py',code_sha256='hash',vlm_batch_size=4,max_new_tokens=450,text_attempts=2)
   with patch.object(gate,'amendment',return_value=amendment),patch.object(gate,'prepare_amended_vlm') as vlm,patch.object(repair,'make_job',return_value=job):
    vlm.return_value.ask.return_value=[json.dumps(response(GOOD))]
    self.assertTrue(gate.finalize(root,'b',dict(model_sha256={},files_sha256={})))
    vlm.return_value.ask.assert_called_once();vlm.return_value.ask.reset_mock()
    self.assertTrue(gate.finalize(root,'b',dict(model_sha256={},files_sha256={})))
    vlm.return_value.ask.assert_not_called()
   archived=json.loads((root/'batches/b/pre_gate_drafts/r0_old.json').read_text());self.assertEqual(archived,dict(planning=plan,job=job))
   c=repair.db(root);r=c.execute('select round,result,planning from attempts').fetchone();self.assertEqual(r['round'],0);self.assertIsNone(r['result']);self.assertEqual(json.loads(r['planning'])['prompt_gate']['fingerprint'],'gate');c.close()

 def test_bad_final_text_is_not_sent_to_image_generation(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'batches/b').mkdir(parents=True);c=repair.db(root)
   with c:
    c.execute('insert into places(id,ordinal,city,place_id,original,defect,status,round) values(?,?,?,?,?,?,?,?)',('old',0,'City',7,'{}','{}','working',2))
    c.execute('insert into batches(id,created) values(?,?)',('b',0))
    c.execute('insert into attempts(id,round,batch,planning,job,source) values(?,?,?,?,?,?)',('old',2,'b',json.dumps(dict(appearance_prompt=GOOD,prompt_issues=[])),json.dumps(dict(source_path='source.jpg',source_sha256='hash')),'{}'))
   c.close();amendment=dict(fingerprint='gate',code_path='gate.py',code_sha256='hash',vlm_batch_size=4,max_new_tokens=450,text_attempts=2)
   with patch.object(gate,'amendment',return_value=amendment),patch.object(gate,'prepare_amended_vlm') as vlm,patch.object(repair,'make_job') as make:
    vlm.return_value.ask.return_value=['invalid JSON']
    self.assertTrue(gate.finalize(root,'b',dict(model_sha256={},files_sha256={})));make.assert_not_called()
   c=repair.db(root);self.assertEqual(tuple(c.execute('select status,round from places').fetchone()),('exhausted',2));self.assertIsNone(c.execute('select job from attempts').fetchone()[0]);c.close()

if __name__=='__main__':unittest.main()
