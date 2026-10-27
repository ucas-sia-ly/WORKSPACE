"""Machine-only scene-plan, authorization and paired-experiment checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from PIL import Image
from experiments.qwen_curriculum import common,run,scene_prompting as p,scene_aware_night as workflow


def plan():
    prose=('Relight the red brick corner building and the empty paved road for a quiet late night. '
           'Keep its two arched windows, existing doorway and street geometry. Use localized window '
           'light with darker road intervals and natural night exposure, preserving all visible place evidence.')
    return dict(schema_version=1,scene_summary='red brick building beside an empty road',editability='high',
        protected_regions=[dict(description='arched building windows',box=[.1,.05,.9,.4])],
        variants=[dict(style_name='Moonlit brick and tungsten windows',weather='dry',lighting='dim ambient and window light',
            exposure='moderate shadow detail',appearance_prompt=prose,
            transient=dict(type='vehicle',surface='road',box=[.4,.6,.65,.85],
                           description='a single grey car in the empty near lane'))])


class ScenePlanningTests(unittest.TestCase):
    def test_vlm_prose_is_preserved_in_both_paired_prompts(self):
        value=p.validate_plan(plan(),1)['variants'][0]
        base=p.render_prompt(value);extra=p.render_prompt(value,True)
        self.assertTrue(base.startswith(value['appearance_prompt']))
        self.assertTrue(extra.startswith(value['appearance_prompt']))
        self.assertIn('Add exactly one vehicle',extra)
        self.assertNotIn('Add exactly one vehicle',base)
        self.assertIn('Do not add or remove vehicles',base)

    def test_open_style_names_and_exposure_do_not_require_template_membership(self):
        value=plan();value['variants'][0]['style_name']='Late-night mixed mercury spill under hanging vines'
        self.assertEqual(p.validate_plan(value,1),value)

    def test_appearance_only_field_cannot_hide_extra_objects(self):
        value=plan();value['variants'][0]['appearance_prompt']+=' A scooter appears near the right sidewalk.'
        with self.assertRaisesRegex(ValueError,'mixes transient'):p.validate_plan(value,1)

    def test_object_preservation_and_existing_headlights_are_valid_appearance_text(self):
        value=plan();value['variants'][0]['appearance_prompt']+=' No people or vehicles are added. Existing vehicle headlights glow on the road.'
        p.validate_plan(value,1)
        self.assertTrue(p.has_uncontrolled_object_directive('No vehicles are present in frame.'))
        self.assertTrue(p.has_uncontrolled_object_directive('Add one car near the curb.'))

    def test_no_safe_space_allows_appearance_only_and_never_forces_objects(self):
        value=plan();value['editability']='low';value['variants'][0]['transient']=None
        p.validate_plan(value,1)
        with self.assertRaisesRegex(ValueError,'No feasible'):p.render_prompt(value['variants'][0],True)

    def test_illegal_surface_protected_occlusion_and_large_objects_are_rejected(self):
        for change in (dict(surface='sidewalk'),dict(box=[.3,.1,.6,.4]),
                       dict(box=[0,.6,.2,.8]),dict(box=[.1,.5,.9,.9])):
            value=plan();value['variants'][0]['transient'].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):p.validate_plan(value,1)
        value=plan();value['editability']='medium'
        self.assertEqual(p.validate_plan(value,1),value)

    def test_bad_coordinates_and_duplicate_proposals_fail_closed(self):
        for b in ([0,0,2,1],[True,.6,.7,.8],[.4,.6,float('nan'),.8]):
            value=plan();value['variants'][0]['transient']['box']=b
            with self.assertRaises(ValueError):p.validate_plan(value,1)
        value=plan();value['variants']*=2
        with self.assertRaisesRegex(ValueError,'Duplicate'):p.validate_plan(value,2)
        with self.assertRaises(ValueError):p.parse_json('not a JSON scene plan')

    def test_native_coordinates_require_explicit_units_and_are_converted_once(self):
        raw=plan();raw['coordinate_system']='0_1000';raw['transient_candidates']=[]
        for region in raw['protected_regions']+[raw['variants'][0]['transient']]:
            region['box']=[x*1000 for x in region['box']]
        normalized=p.normalize_plan(raw)
        self.assertEqual(normalized['variants'][0]['transient']['box'],plan()['variants'][0]['transient']['box'])
        self.assertEqual(raw['variants'][0]['transient']['box'],[400,600,650,850])
        p.validate_plan(normalized,1)
        with self.assertRaisesRegex(ValueError,'Declare'):p.normalize_plan(plan())
        with self.assertRaisesRegex(ValueError,'Declare'):p.normalize_plan(normalized)

    def test_unsafe_proposal_is_abstained_without_discarding_good_styles_or_candidates(self):
        raw=plan();raw['coordinate_system']='0_1000'
        for region in raw['protected_regions']+[raw['variants'][0]['transient']]:
            region['box']=[x*1000 for x in region['box']]
        safe=copy.deepcopy(raw['variants'][0]['transient'])
        unsafe={**copy.deepcopy(safe),'box':[0,600,250,850]}
        raw['transient_candidates']=[unsafe,safe]
        del raw['variants'][0]['transient']
        value=p.normalize_plan(raw)
        self.assertEqual(len(value['rejected_transient_candidates']),1)
        self.assertEqual(len(value['transient_candidates']),1)
        self.assertEqual(value['variants'][0]['appearance_prompt'],raw['variants'][0]['appearance_prompt'])
        self.assertEqual(value['variants'][0]['transient']['type'],'vehicle')
        p.validate_plan(value,1)

    def test_deferred_verifier_cannot_claim_geometry_or_training_pass(self):
        verdict=workflow.DeferredVerifier().evaluate(None,None,'night')
        self.assertEqual(verdict['status'],'pending_verification')
        self.assertFalse(verdict['passed']);self.assertFalse(verdict['base_quality_passed'])
        self.assertTrue(verdict['verification_pending'])

    def test_pairing_keeps_source_seed_style_and_metadata_exact(self):
        source=dict(city='City',place_id=1,source_path='/real/source.jpg',source_sha256='a'*64,
            source_dimensions=[640,480],sample_id='source-1')
        jobs=workflow.make_jobs([source],[plan()],True,42)
        self.assertEqual(len(jobs),2)
        a,b=jobs
        for key in ('source_path','source_sha256','seed','pair_id','appearance','scene_plan_sha256'):
            self.assertEqual(a[key],b[key])
        self.assertIsNone(a['authorized_transient']);self.assertFalse(a['with_transient'])
        self.assertEqual(b['authorized_transient']['type'],'vehicle');self.assertTrue(b['with_transient'])
        self.assertFalse(b['planned_region_is_verified_mask'])
        self.assertNotEqual(a['sample_id'],b['sample_id'])
        self.assertEqual(workflow.make_jobs([source],[plan()],True,42),jobs)
        with self.assertRaisesRegex(ValueError,'Every source'):workflow.make_jobs([source],[],True,42)

    def test_audit_cannot_promote_contradictory_counts_or_string_booleans(self):
        job={'authorized_transient':{'type':'vehicle'}}
        evidence=dict(status='candidate',permanent_changes=[],authorized_edit_achieved=True,
            unexpected_objects=[],placement_legal=True,landmarks_visible=True,
            observed_added_transients=['vehicle'],evidence='one car added on the empty road; building unchanged')
        self.assertEqual(workflow.validate_audit(evidence,job)['status'],'candidate')
        for field,value in [('permanent_changes',['new storefront']),('observed_added_transients',[]),
                            ('landmarks_visible',False)]:
            changed={**evidence,field:value}
            self.assertEqual(workflow.validate_audit(changed,job)['status'],'unknown')
        with self.assertRaises(ValueError):workflow.validate_audit({**evidence,'placement_legal':'true'},job)

    def test_source_isolation_and_hash_checks_before_model_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);file=root/'Images/City/City_0000001_fixture.png';file.parent.mkdir(parents=True)
            Image.new('RGB',(20,16),'grey').save(file)
            source=run.sealed(dict(city='City',place_id=1,source_path=str(file),source_sha256=common.file_sha256(file),
                                  source_dimensions=[20,16],sample_id='one'))
            args=([source],['one'],1,set(),root,42)
            self.assertEqual(workflow.choose_sources(*args),[source])
            with self.assertRaises(ValueError):workflow.choose_sources([source],['one'],1,{('City',1)},root,42)
            file.write_bytes(b'changed')
            with self.assertRaises(ValueError):workflow.choose_sources(*args)

    def test_request_budget_rejects_before_durable_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            from experiments.qwen_curriculum import campaign
            campaign.frozen(directory/'plan_config.json',dict(request_budget=1))
            private=workflow.engine()
            with patch.object(private,'_original_append') as append:
                with self.assertRaisesRegex(RuntimeError,'budget'):workflow.bounded_append(directory/'attempts.jsonl',dict(call_number=2))
                append.assert_not_called()
                workflow.bounded_append(directory/'attempts.jsonl',dict(call_number=1));append.assert_called_once()

    def test_separate_object_planner_validates_units_visibility_and_infeasibility(self):
        from experiments.qwen_curriculum.plan_scene_transients import validate_proposal
        proposal=dict(type='vehicle',surface='road',box=[400,600,650,850],
                      description='one extra grey car on the empty near lane')
        raw=dict(feasible=True,proposal=proposal,evidence='empty visible near lane between the curbs')
        p1,_=validate_proposal(json.dumps(raw),plan()['protected_regions'])
        self.assertEqual(p1['box'],[.4,.6,.65,.85])
        for b in ([.4,.6,.65,.85],[True,600,650,850]):
            with self.assertRaises(ValueError):validate_proposal(json.dumps({**raw,'proposal':{**proposal,'box':b}}),plan()['protected_regions'])
        with self.assertRaises(ValueError):validate_proposal(json.dumps({**raw,'coordinate_system':'normalized'}),plan()['protected_regions'])
        none,_=validate_proposal(json.dumps(dict(feasible=False,proposal=None,evidence='no clear legal surface visible')),[])
        self.assertIsNone(none)

    def test_mock_scene_reader_and_durable_generation_are_image_bound_and_resumable(self):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parent))
        from test_night_first_v3 import HEALTH,Session,FakeVerifier
        from experiments.qwen_curriculum import campaign
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);real=root/'gsv';source=real/'Images/City/City_0000001_fixture.png'
            source.parent.mkdir(parents=True);Image.new('RGB',(20,16),'grey').save(source)
            row=run.sealed(dict(city='City',place_id=1,source_path=str(source),source_sha256=common.file_sha256(source),
                source_dimensions=[20,16],sample_id='one'))
            manifest=root/'input.jsonl';common.write_jsonl(manifest,[row])
            exclude=root/'exclude.json';common.write_json(exclude,[])
            model=root/'model';model.mkdir();(model/'model.safetensors').write_bytes(b'test-only-no-model')
            (model/'config.json').write_text('{}')
            servicecfg=root/'service.json';campaign.frozen(servicecfg,dict(reference_execution=dict(service=run.service_identity(HEALTH))))
            args=SimpleNamespace(output_dir=root/'pilot/state',image_root=root/'pilot',source_manifest=manifest,
                source_ids=['one'],max_sources=1,variants=1,paired_transients=True,seed=42,planner_model=model,
                max_jobs=2,exclude_places=exclude,real_data=real,service_config=servicecfg,execute=False,
                device='cpu',domain='night',max_calls=2,qwen_url='http://127.0.0.1:8001/generate',request_timeout=20.)
            with patch.object(p,'LocalVisionPlanner') as reader,patch.object(campaign,'ensure_stopped'):
                self.assertEqual(workflow.prepare(args)['planner_calls'],0);reader.assert_not_called()
                raw=plan();raw['coordinate_system']='0_1000';raw['transient_candidates']=[]
                for region in raw['protected_regions']+[raw['variants'][0]['transient']]:
                    region['box']=[x*1000 for x in region['box']]
                args.execute=True;reader.return_value.read.return_value=json.dumps(raw)
                workflow.prepare(args);workflow.prepare(args)
                reader.return_value.read.assert_called_once()
                self.assertEqual(reader.return_value.read.call_args.args[0],str(source))
                reader.return_value.close.assert_called_once()
            private=workflow.engine();service=root/'service';service.mkdir();session=Session(args.output_dir,service)
            verifier=FakeVerifier();verifier.definition=workflow.quality_definition(clip_device='cpu')
            def evaluate(*_):
                return dict(passed=True,status='passed',eligible_for_training=False,base_quality_passed=True)
            verifier.evaluate=evaluate
            with patch.object(private.requests,'Session',return_value=session), \
                    patch.object(private.run,'service_output_directory',return_value=service), \
                    patch.object(private,'QwenQualityVerifierV3',return_value=verifier):
                private.generate(args);private.generate(args)
            self.assertEqual(len(session.posts),2)
            self.assertEqual((args.output_dir/'training_manifest.jsonl').read_text(),'')
            self.assertFalse((args.image_root/'accepted').exists())
            cfg,jobs=workflow.load_plan(args.output_dir)
            self.assertEqual(cfg['request_budget'],2)
            self.assertEqual(jobs[0]['seed'],jobs[1]['seed'])


if __name__=='__main__':unittest.main()
