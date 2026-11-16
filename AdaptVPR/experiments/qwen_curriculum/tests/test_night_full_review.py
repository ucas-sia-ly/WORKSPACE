import json,tempfile,unittest
from pathlib import Path
from experiments.qwen_curriculum import night_full_review as review
from experiments.qwen_curriculum.night_full_filter import connect
class ReviewTests(unittest.TestCase):
 def issue(self,solar=False,certainty='clear',category='daylight'):
  return dict(verdict='issue',evidence='Source and generated pair visibly compared',issues=[dict(category=category,certainty=certainty,location='right side of the road',evidence='Distinct original sun patches remain on pavement',persistent_solar_artifact=solar)])
 def test_clear_solar_replaces_source(self):self.assertEqual(review.derive_review(self.issue(True)),'replace_source')
 def test_small_issue_edits(self):self.assertEqual(review.derive_review(self.issue()),'minor_edit')
 def test_uncertain_not_promoted(self):self.assertEqual(review.derive_review(self.issue(certainty='uncertain')),'unknown')
 def test_no_issue_kept(self):self.assertEqual(review.derive_review(dict(verdict='keep',issues=[],evidence='Night scene and original road layout remain readable')),'keep')
 def test_invalid_solar_category(self):
  with self.assertRaises(ValueError):review.derive_review(self.issue(True,category='structure'))
 def test_unknown_not_kept(self):self.assertEqual(review.derive_review(dict(verdict='unknown',issues=[],evidence='The structure is hidden by severe darkness')),'unknown')
 def test_minor_exact_and_contract(self):
  p='Turn this street into night with crisp tree shadows. Preserve trees and road.\n\nKeep this exact viewpoint, windows and signs.'
  s=review.minor_edit(p,[dict(old='crisp tree shadows',new='soft diffuse shading')])
  self.assertEqual(s,p.replace('crisp tree shadows','soft diffuse shading'))
 def test_contract_cannot_change(self):
  with self.assertRaises(ValueError):review.minor_edit('A night road.\n\nKeep this exact viewpoint',[dict(old='Keep this exact viewpoint',new='Change viewpoint')])
 def test_ambiguous_old_rejected(self):
  with self.assertRaises(ValueError):review.minor_edit('night night scene',[dict(old='night',new='dark')])
 def test_full_rewrite_rejected(self):
  p=' '.join('word'+str(i) for i in range(100))
  with self.assertRaises(ValueError):review.minor_edit(p,[dict(old=p,new='A completely different night scene')])
 def test_same_place_unique_queue(self):
  with tempfile.TemporaryDirectory() as d:
   c=connect(d);c.execute('insert into pairs(ordinal,sample_id,city,place_id,payload) values(0,?,?,?,?)',('one','city',1,'{}'));c.commit();c.close()
   c=connect(d);self.assertEqual(c.execute('select sample_id from pairs').fetchone()[0],'one');c.close()
 def test_review_waits_for_all_rules(self):
  import inspect
  self.assertIn('Finish ALL first-pass rules',inspect.getsource(review.run))
 def test_no_image_generation_endpoint(self):
  import inspect
  s=inspect.getsource(review)
  self.assertNotIn('qwen_url',s);self.assertNotIn('Backend(',s);self.assertNotIn('http://127.0.0.1:8001',s)
if __name__=='__main__':unittest.main()
