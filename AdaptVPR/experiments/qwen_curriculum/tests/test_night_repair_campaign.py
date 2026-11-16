import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from experiments.qwen_curriculum import night_repair_campaign as repair,common

class RepairTests(unittest.TestCase):
 def prompt(self):
  return dict(source_summary='Real street with existing shop glazing and trees.',prompt_issues=[dict(quote='crisp sunlight',problem='retains solar shadows',correction='diffuse night shading')],audit_response='Night exposure and contradicting shadows corrected based on evidence.',style_id='shopfronts',appearance_prompt='Make this exact street a quiet night photograph with restrained warm light from the existing shop glazing. Keep the tree positions, road layout, signs and window openings unchanged. Use faint ambient illumination on nearby surfaces, without adding fixtures or preserving direct sunlight.')
 def original(self):
  return dict(sample_id='old',city='Test',place_id=7,seed=42,prompt='old',source_path='/old.jpg',source_sha256='oldsha')
 def test_three_image_ceiling_and_stop_on_pass(self):
  self.assertEqual(repair.next_status(0,False),('pending',1));self.assertEqual(repair.next_status(1,False),('pending',2));self.assertEqual(repair.next_status(2,False),('exhausted',2))
  for r in range(3):self.assertEqual(repair.next_status(r,True),('passed',r))
  with self.assertRaises(ValueError):repair.next_status(3,False)
 def test_shopfronts_is_allowed_no_category_ban(self):
  self.assertEqual(repair.validate_prompt(self.prompt(),'old')['style_id'],'shopfronts')
 def test_mercury_stays_retired(self):
  p=self.prompt();p['style_id']='mercury_vapor'
  with self.assertRaises(ValueError):repair.validate_prompt(p,'old')
 def test_refusal_or_unchanged_prompt_not_publishable(self):
  p=self.prompt()
  with self.assertRaises(ValueError):repair.validate_prompt(p,p['appearance_prompt'])
  p['appearance_prompt']='The scene cannot be augmented. '+p['appearance_prompt']
  with self.assertRaises(ValueError):repair.validate_prompt(p,'old')
 def test_current_source_metadata_and_prose_not_old_masks(self):
  src=dict(source_path='/new.jpg',source_sha256='newsha',source_dimensions=[600,400],source_sample_id='newsrc')
  j=repair.make_job(self.original(),src,self.prompt(),1,dict(seed=42,fingerprint='config'))
  self.assertEqual(j['source_path'],'/new.jpg');self.assertEqual(j['source_dimensions'],[600,400]);self.assertFalse(j['old_source_correspondence_reused']);self.assertFalse(j['planned_region_is_verified_mask'])
  self.assertTrue(j['prompt'].startswith(self.prompt()['appearance_prompt']));self.assertEqual(j['prompt'].count(repair.styles.CONTRACT),1)
  self.assertEqual(j,repair.make_job(self.original(),src,self.prompt(),1,dict(seed=42,fingerprint='config')))
 def test_original_source_keeps_seed(self):
  src=dict(source_path='/old.jpg',source_sha256='oldsha',source_dimensions=[400,300],source_sample_id='source')
  self.assertEqual(repair.make_job(self.original(),src,self.prompt(),0,dict(seed=43,fingerprint='config'))['seed'],42)
 def test_sources_exclude_used_bytes_and_wrong_place(self):
  with tempfile.TemporaryDirectory() as tmp:
   d=Path(tmp);paths=[]
   for name,data in [('Test_0000007_a.jpg',b'a'),('Test_0000007_copy.jpg',b'a'),('Test_0000007_b.jpg',b'b'),('Test_0000008_c.jpg',b'c')]:
    p=d/name;p.write_bytes(data);paths.append(p)
   tried=[dict(source_path=str(paths[0]),source_sha256=common.file_sha256(paths[0]))]
   result=repair.remaining_sources(self.original(),paths,tried)
   self.assertEqual([Path(r['source_path']).name for r in result],['Test_0000007_b.jpg'])
 def seed_places(self,root,n=10):
  c=repair.db(root)
  with c:
   for i in range(n):c.execute('insert into places(id,ordinal,city,place_id,original,defect) values(?,?,?,?,?,?)',(str(i),i,'Test',i,'{}','{}'))
  c.close()
 def test_nodes_never_own_same_place(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);self.seed_places(root)
   a=repair.new_batch(root,{},4,'gpu2');b=repair.new_batch(root,{},4,'local');c=repair.db(root)
   aa={r[0] for r in c.execute('select id from attempts where batch=?',(a,))};bb={r[0] for r in c.execute('select id from attempts where batch=?',(b,))}
   self.assertEqual(len(aa),4);self.assertFalse(aa&bb);self.assertEqual(repair.new_batch(root,{},4,'gpu2'),a);c.close()
 def test_export_generated_eligibility_does_not_collide(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);self.seed_places(root,1);c=repair.db(root)
   e=dict(generated=dict(sample_id='new',eligible_for_training=False,source_path='newsource'),base={},content={})
   with c:
    c.execute("update places set status='passed' where id='0'");c.execute('insert into attempts(id,round,result) values(?,?,?)',('0',0,json.dumps(e)))
   c.close();repair.export(root);r=json.loads((root/'accepted_replacements.jsonl').read_text());self.assertFalse(r['eligible_for_training']);self.assertEqual(r['original_sample_id'],'0')
 def test_content_classifier_called_without_shopfronts_quarantine(self):
  from experiments.qwen_curriculum import content_audit_v42 as classifier
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'im';p.write_bytes(b'image');h=common.file_sha256(p)
   row=dict(sample_id='shop',source_path=str(p),output_path=str(p),source_sha256=h,output_sha256=h,prompt_variant='shopfronts')
   evidence={**row,'current_rule':dict(passed=True,geometry_ok=True,weather_ok=True,estimated_H=None,matcher_image_size=512)}
   cal=dict(feature_pipeline=dict(side=448,search_radius=1))
   with patch.object(repair.content,'validate_calibration'),patch.object(repair.legacy,'pair_signals',return_value={}),patch.object(repair.content,'peak_region',return_value={}),patch.object(classifier,'audit_pair_v42',return_value=dict(decision='pass')) as call:
    result=repair.evaluate_content(row,evidence,cal,object());self.assertTrue(result['safety_screen_passed']);self.assertEqual(call.call_args.kwargs['quarantine_variants'],())

if __name__=='__main__':unittest.main()
