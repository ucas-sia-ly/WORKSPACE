import copy
import unittest
from unittest.mock import patch

from experiments.qwen_curriculum import night_place_cohort_v4 as cohort


class SourceSchemaTests(unittest.TestCase):
    def evidence(self):
        return dict(coordinate_system='0_1000', scene_summary='An actual permanent building',
            protected_regions=[dict(description='The existing facade', box=[0,0,900,900])],
            features={key:dict(status='present', evidence='Visible source evidence')
                      for key in cohort.engine.policy.FEATURES})

    def test_source_evidence_does_not_mutate_inputs_or_require_predicted_boxes(self):
        value=self.evidence();value.pop('protected_regions')
        before=copy.deepcopy(value)
        cohort.source_evidence(value)
        self.assertEqual(value,before)
        self.assertIn('Do not return bounding boxes',cohort.analysis_request())

    def test_bounded_analysis_repair_keeps_source_correspondence(self):
        full=cohort
        import json
        good=json.dumps(self.evidence());calls=[]
        def ask(_self,directory,sources,requests,phase,attempt,cfg):
            calls.append((sources,attempt))
            return [good,'{}',good,'{}'] if attempt==0 else [good,good]
        planner=object.__new__(full.Planner)
        with patch.object(full._planner,'ask',ask):
            replies=planner.ask(None,['a','b','c','d'],['x']*4,'analysis',0,{})
        self.assertEqual(calls,[(['a','b','c','d'],0),(['b','d'],1)])
        self.assertEqual(replies,[good]*4)

    def test_failed_repair_is_explicit_and_not_looped(self):
        full=cohort
        planner=object.__new__(full.Planner)
        with patch.object(full._planner,'ask',return_value=['{}']) as ask:
            self.assertEqual(planner.ask(None,['a'],['x'],'analysis',0,{}),['{}'])
        self.assertEqual(ask.call_count,2)

    def test_full_frame_contract_does_not_claim_detected_landmarks(self):
        from experiments.qwen_curriculum import night_place_cohort_v4 as full
        value=self.evidence();value.pop('protected_regions');value.pop('coordinate_system')
        out=full.source_evidence(value)
        region=out['protected_regions'][0]
        self.assertEqual(out['features'],value['features'])
        self.assertEqual(region['box'],[0,0,1000,1000])
        self.assertFalse(region['usable_as_correspondence_or_semantic_evidence'])
        self.assertEqual(region['origin'],'deterministic_preservation_contract')
        self.assertNotIn('protected_regions',value)

    def test_full_frame_does_not_fill_missing_capabilities(self):
        from experiments.qwen_curriculum import night_place_cohort_v4 as full
        value=self.evidence();value['features'].pop('outdoors')
        with self.assertRaises(ValueError): full.source_evidence(value)


if __name__=='__main__': unittest.main()
