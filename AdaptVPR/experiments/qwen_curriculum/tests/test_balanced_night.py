import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
from experiments.qwen_curriculum import campaign, common, run
from experiments.qwen_curriculum import balanced_night as workflow, night_diversity_policy as policy


def evidence(present=('outdoors', 'drivable_road', 'paved_ground')):
    return dict(coordinate_system='0_1000', scene_summary='brick corner beside a paved street',
        features={key: dict(status='present' if key in present else 'absent',
                            evidence='clearly visible evidence' if key in present else 'not visible here')
                  for key in policy.FEATURES},
        protected_regions=[dict(description='brick corner facade', box=[100, 100, 700, 500])])


def plan(style='sodium_lamps'):
    source = evidence()
    value = dict(schema_version=1, coordinate_system='0_1000', scene_summary=source['scene_summary'],
        source_evidence=source, protected_regions=source['protected_regions'], requested_style_ids=[style],
        variants=[dict(style_id=style, style_name=style+': amber brick street', weather='night', surface_state='wet',
            lighting='localized amber street illumination', exposure='natural nighttime shadow detail',
            appearance_prompt='Relight the existing brick corner and the paved road as a late-night street photograph. '
                'Let warm localized illumination fall across the same facade and road, keeping the original '
                'windows, doorway, sign shapes and road topology. Add a plausible car on free road space, '
                'remove a pedestrian and make the dry pavement wet while retaining useful landmark evidence.')])
    return policy.normalize_plan(value)


class BalancedNightTests(unittest.TestCase):
    def test_catalog_extends_v3_without_becoming_full_prompt_templates(self):
        self.assertGreaterEqual(len(policy.STYLES), 25)
        self.assertEqual(len(policy.STYLES), len(policy.CATALOG['styles']))
        self.assertTrue({'sodium_lamps', 'led_lamps', 'shopfronts', 'sparse_lamps', 'headlights', 'blue_hour'} <= policy.STYLES.keys())
        for value in policy.STYLES.values():
            self.assertNotIn('prompt', value)
            self.assertTrue(set(value['references']) <= policy.CATALOG['references'].keys())
            self.assertTrue(all(set(clause) <= set(policy.FEATURES) for clause in value['requires']))

    def test_missing_shop_or_traffic_cannot_be_created_to_fit_a_style(self):
        source = evidence()
        eligible, excluded = policy.eligible_styles(source)
        self.assertIn('sodium_lamps', eligible)
        self.assertIn('shopfronts', excluded)
        self.assertIn('headlights', excluded)
        self.assertIn('passing_vehicle_lights', excluded)
        source['features']['shopfronts']['status'] = 'uncertain'
        self.assertNotIn('shopfronts', policy.eligible_styles(source)[0])
        source['features']['shopfronts']['status'] = 'present'
        self.assertIn('shopfronts', policy.eligible_styles(source)[0])
        source['features']['safe_vehicle_space']['status'] = 'present'
        self.assertIn('passing_vehicle_lights', policy.eligible_styles(source)[0])

    def test_missing_sky_does_not_exclude_night_and_no_style_is_forced(self):
        self.assertIn('moonlit_ambient', policy.eligible_styles(evidence())[0])
        self.assertNotIn('open_sky', policy.FEATURES)
        self.assertEqual(policy.eligible_styles(evidence(()))[0], [])
        with self.assertRaises(ValueError):
            policy.select_styles([], 1, {})

    def test_rare_eligible_style_is_preferred_with_reservations_and_deterministic_ties(self):
        eligible = ['sodium_lamps', 'led_lamps']
        counts = dict(sodium_lamps=dict(saved=20), led_lamps=dict(saved=2))
        self.assertEqual(policy.select_styles(eligible, 1, counts), ['led_lamps'])
        counts['led_lamps']['pending'] = 30
        self.assertEqual(policy.select_styles(eligible, 1, counts), ['sodium_lamps'])
        first = policy.select_styles(eligible, 2, {}, seed=43, source_key='one')
        self.assertEqual(first, policy.select_styles(list(reversed(eligible)), 2, {}, seed=43, source_key='one'))
        self.assertEqual(len(set(first)), 2)
        with self.assertRaises(ValueError):
            policy.select_styles(eligible, 1, dict(led_lamps=dict(saved=1, mismatch=2)))

    def test_integrated_vehicle_person_and_wetness_changes_are_allowed(self):
        value = plan(); policy.validate_plan(value, 1)
        variant = value['variants'][0]
        prompt = policy.render_prompt(variant)
        self.assertTrue(prompt.startswith(variant['appearance_prompt']))
        self.assertIn('remove a pedestrian', prompt)
        self.assertNotIn('Do not add or remove', prompt)
        value['variants'][0]['weather'] = 'rainy_night'
        with self.assertRaises(ValueError):
            policy.validate_plan(value, 1)

    def test_audit_accepts_transients_but_not_structure_change_or_daylight(self):
        value = dict(structure_preserved=True, is_night=True, landmarks_visible=True,
            style_matched=False, surface_state='wet', permanent_changes=[],
            transient_changes=['two cars removed', 'one person added'], evidence='Facade and road structure preserved')
        self.assertEqual(workflow.validate_audit(value, {})['status'], 'candidate')
        self.assertFalse(workflow.validate_audit(value, {})['style_matched'])
        for key in ('is_night', 'structure_preserved', 'landmarks_visible'):
            self.assertEqual(workflow.validate_audit({**value, key: False}, {})['status'], 'reject')
            self.assertEqual(workflow.validate_audit({**value, key: None}, {})['status'], 'unknown')
        self.assertEqual(workflow.validate_audit({**value, 'permanent_changes': ['new shop entrance']}, {})['status'], 'reject')

    def test_all_jobs_are_night_with_source_specific_prose_and_authoritative_mechanism(self):
        source = dict(city='City', place_id=1, source_path='/fixture', source_sha256='fixture',
                      source_dimensions=[20, 16], sample_id='one')
        jobs = workflow.make_jobs([source], [plan()], False, 42)
        self.assertEqual(jobs[0]['condition'], 'night')
        self.assertFalse(jobs[0]['with_transient'])
        self.assertEqual(jobs[0]['appearance']['requested_mechanism'], policy.STYLES['sodium_lamps']['mechanism'])
        self.assertEqual(jobs, workflow.make_jobs([source], [plan()], False, 42))
        self.assertNotEqual(jobs[0]['seed'], workflow.make_jobs([source], [plan()], False, 43)[0]['seed'])

    def test_private_runtime_keeps_old_policy_unchanged(self):
        from experiments.qwen_curriculum import scene_aware_night as old
        current = workflow.runtime()
        self.assertIsNot(current, old)
        self.assertEqual(current.engine().DOMAINS, ('night',))
        self.assertNotEqual(current.planning.VERSION, old.planning.VERSION)
        self.assertIn('rainy_night', old.engine().DOMAINS)

    def test_two_phase_reader_reuses_analysis_and_preserves_scheduled_ids(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(policy, 'STATE', Path(tmp)), \
                patch.object(policy, 'SNAPSHOT', policy.coverage_snapshot([])):
            image = Path(tmp)/'source.png'; Image.new('RGB', (32, 24), 'grey').save(image)
            source = evidence()
            eligible = policy.eligible_styles(source)[0]
            selected = policy.select_styles(eligible, 1, {}, seed=42, source_key=common.file_sha256(image))
            reply = dict(variants=plan(selected[0])['variants'])
            reader = object.__new__(policy.LocalVisionPlanner)
            with patch.object(reader, '_ask', side_effect=[json.dumps(source), json.dumps(reply), json.dumps(reply)]) as ask:
                first = reader.read(image, 1, 42)
                second = reader.read(image, 1, 43, feedback='format repair')
                for raw in (first, second):
                    result = policy.validate_plan(policy.normalize_plan(json.loads(raw)), 1)
                    self.assertEqual(result['requested_style_ids'], selected)
                self.assertEqual(ask.call_count, 3)  # One source analysis, two bounded compositions.
                self.assertEqual(ask.call_args_list[0].args[-1], False)

    def test_saved_exposure_style_confirmation_and_mismatch_are_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)/'state'; state.mkdir()
            image = Path(tmp)/'night.png'; Image.new('RGB', (20, 16), 'black').save(image)
            row = dict(sample_id='night', condition='night', style_id='sodium_lamps',
                       output_path=str(image), output_sha256=common.file_sha256(image))
            common.write_jsonl(state/'results.jsonl', [row])
            audit = state/'audit/night.json'
            common.write_json(audit, run.sealed(dict(output_sha256=row['output_sha256'],
                verdict=dict(status='candidate', style_matched=False))))
            counts = policy.coverage_snapshot([state])['counts']['sodium_lamps']
            self.assertEqual(counts, dict(saved=1, pending=0, confirmed=0, mismatch=1))
            row['style_id'] = 'unknown_legacy_name'; common.write_jsonl(state/'results.jsonl', [row])
            self.assertEqual(policy.coverage_snapshot([state])['unclassified_night_records'], 1)

    def test_cpu_preparation_does_not_invoke_models(self):
        from experiments.qwen_curriculum import run_scene_night_safe as safe
        with tempfile.TemporaryDirectory() as tmp, patch.object(safe.subprocess, 'call', return_value=0) as child:
            safe.main(['prepare', '--pipeline', 'balanced', '--output-dir', str(Path(tmp)/'state'),
                       '--image-root', tmp, '--variants', '3'])
            command = child.call_args.args[0]
            self.assertNotIn('--execute', command)
            self.assertIn('3', command)
            self.assertTrue(any(v.endswith('balanced_night.py') for v in command))

    def test_mock_plan_generation_and_resume_use_real_source_binding(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_night_first_v3 import HEALTH, Session
        module = workflow.runtime()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'real/Images/City/City_0000001_test.png'
            source.parent.mkdir(parents=True); Image.new('RGB', (20, 16), 'grey').save(source)
            row = run.sealed(dict(city='City', place_id=1, source_path=str(source), source_sha256=common.file_sha256(source),
                                  source_dimensions=[20, 16], sample_id='one'))
            manifest = root/'source.jsonl'; common.write_jsonl(manifest, [row])
            excluded = root/'excluded.json'; common.write_json(excluded, [])
            model = root/'model'; model.mkdir(); (model/'model.safetensors').write_bytes(b'CPU mock model fixture')
            cfg = root/'service.json'; campaign.frozen(cfg, dict(reference_execution=dict(service=run.service_identity(HEALTH))))
            directory = root/'pilot/state'
            snapshot = campaign.frozen(directory/'coverage_snapshot.json', {**policy.coverage_snapshot([]), 'requested_states': []})
            policy.SNAPSHOT = snapshot
            args = SimpleNamespace(output_dir=directory, image_root=directory.parent, source_manifest=manifest,
                source_ids=['one'], max_sources=1, variants=1, paired_transients=False, seed=42,
                planner_model=model, max_jobs=1, exclude_places=excluded, real_data=root/'real', service_config=cfg,
                execute=False, device='cpu', domain='night', max_calls=1, qwen_url='http://127.0.0.1:8001/generate', request_timeout=20)
            raw = plan(); raw['coordinate_system'] = '0_1000'
            for region in raw['protected_regions']: region['box'] = [x*1000 for x in region['box']]
            with patch.object(policy, 'LocalVisionPlanner') as reader, patch.object(campaign, 'ensure_stopped'):
                module.prepare(args); reader.assert_not_called()
                args.execute = True; reader.return_value.read.return_value = json.dumps(raw)
                module.prepare(args); module.prepare(args)
                reader.return_value.read.assert_called_once()
            engine = module.engine(); service = root/'service'; service.mkdir()
            session = Session(directory, service)
            with patch.object(engine.requests, 'Session', return_value=session), \
                 patch.object(engine.run, 'service_output_directory', return_value=service):
                engine.generate(args); engine.generate(args)
            self.assertEqual(len(session.posts), 1)
            self.assertEqual((directory/'training_manifest.jsonl').read_text(), '')
            self.assertEqual(policy.coverage_snapshot([directory])['counts']['sodium_lamps']['saved'], 1)
            verdict = dict(passed=True, eligible_for_training=False, base_quality_passed=True, status='passed')
            with patch.object(campaign, 'ensure_stopped'), \
                    patch.object(module, 'CandidateVerifier', return_value=SimpleNamespace(evaluate=lambda *_: verdict)):
                module.verify(args); module.verify(args)
            audit = dict(structure_preserved=True, is_night=True, landmarks_visible=True, style_matched=True,
                         surface_state='wet', transient_changes=['one car added'], permanent_changes=[],
                         evidence='Brick corner and road retained under night illumination')
            with patch.object(campaign, 'ensure_stopped'), patch.object(policy, 'LocalVisionPlanner') as reader:
                reader.return_value.audit.return_value = json.dumps(audit)
                module.audit(args); module.audit(args)
                reader.return_value.audit.assert_called_once()
            summary = workflow.report(directory)
            self.assertEqual(summary['by_style']['sodium_lamps']['confirmed'], 1)
            self.assertEqual(summary['training_eligible'], 0)


if __name__ == '__main__':
    unittest.main()
