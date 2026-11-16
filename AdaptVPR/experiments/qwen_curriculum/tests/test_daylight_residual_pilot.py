"""Schema checks only; no claim of visual detector accuracy."""
import json
import unittest
from experiments.qwen_curriculum.daylight_residual_pilot import parse_response

class ResponseTests(unittest.TestCase):
    def test_not_found_has_no_boxes(self):
        value={'assessment':'not_found','summary':'No retained cast pattern established','regions':[]}
        self.assertEqual(parse_response(json.dumps(value)),value)
    def test_accept_localized_unknown_night_explanation(self):
        value={'assessment':'suspected','summary':'Specific pavement pattern retained','regions':[{
            'box_0_1000':[100,200,400,500],'source_observation':'Dappled light on path',
            'generated_observation':'Same boundary under dark sky','evidence':'Corresponding receiving surface',
            'alternative_explanation':'Off-frame artificial source is possible'}]}
        self.assertEqual(parse_response(json.dumps(value))['assessment'],'suspected')
    def test_malformed_or_contradictory_outputs_fail(self):
        for assessment,summary,regions in [('reject','x',[]),('clear_residual','x',[]),('suspected','x',[]),('not_found','',[])]:
            with self.assertRaises(ValueError):parse_response(json.dumps(dict(assessment=assessment,summary=summary,regions=regions)))
    def test_box_units_and_evidence_required(self):
        region={'box_0_1000':[0,0,1001,500],'source_observation':'a','generated_observation':'b','evidence':'c','alternative_explanation':'d'}
        with self.assertRaises(ValueError):parse_response(json.dumps({'assessment':'suspected','summary':'x','regions':[region]}))
        region['box_0_1000']=[0,0,400,500];del region['evidence']
        with self.assertRaises(ValueError):parse_response(json.dumps({'assessment':'suspected','summary':'x','regions':[region]}))

class GroundShadowGateTests(unittest.TestCase):
    def observation(self,g='present',s='present',relation='same'):
        return {'generated_ground_shadow':g,'source_ground_shadow':s,'pattern_relation':relation,
                'summary':'Compare cast shadows on the same ground surface','regions':[{
                'box_0_1000':[100,500,600,900],'source_observation':'Old shadow boundary',
                'generated_observation':'Observed shadow boundary','evidence':'Location and shape comparison',
                'alternative_explanation':'No extra assumptions'}] if g==s=='present' else []}
    def test_no_shadow_and_no_ground_pass_this_check(self):
        from experiments.qwen_curriculum.daylight_residual_pilot import parse_shadow_response
        for g,s in [('absent','present'),('not_visible','not_comparable'),('present','absent')]:
            self.assertEqual(parse_shadow_response(json.dumps(self.observation(g,s,'not_applicable')))['assessment'],'not_found')
    def test_changed_shadow_passes_even_with_inspection_boxes(self):
        from experiments.qwen_curriculum.daylight_residual_pilot import parse_shadow_response
        result=parse_shadow_response(json.dumps(self.observation(relation='different')))
        self.assertEqual(result['assessment'],'not_found');self.assertTrue(result['regions'])
    def test_same_flag_uncertain_is_not_definite_rejection(self):
        from experiments.qwen_curriculum.daylight_residual_pilot import parse_shadow_response
        self.assertEqual(parse_shadow_response(json.dumps(self.observation()))['assessment'],'clear_residual')
        self.assertEqual(parse_shadow_response(json.dumps(self.observation(relation='uncertain')))['assessment'],'suspected')
    def test_cannot_claim_matching_shadows_without_visible_shadows(self):
        from experiments.qwen_curriculum.daylight_residual_pilot import parse_shadow_response
        with self.assertRaises(ValueError):parse_shadow_response(json.dumps(self.observation(g='absent',relation='same')))
        r=self.observation();r['regions']=[]
        with self.assertRaises(ValueError):parse_shadow_response(json.dumps(r))

if __name__=='__main__':unittest.main()
