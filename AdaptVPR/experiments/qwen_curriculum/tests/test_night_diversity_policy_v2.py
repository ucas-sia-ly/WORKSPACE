"""CPU regression cases for typed evidence and post-mercury prompt contracts."""
import copy
import json
from pathlib import Path
import unittest

from experiments.qwen_curriculum import common, night_diversity_policy as old
from experiments.qwen_curriculum import night_diversity_policy_v2 as policy


def source(*present, uncertain=()):
    return dict(scene_summary='Existing paved street beside the actual parked blue vehicle',
        features={key: dict(status='present' if key in present else 'uncertain' if key in uncertain else 'absent',
            evidence='Visible source evidence' if key in present else 'No confirmed visible evidence')
            for key in policy.FEATURES})


def variant(style='sodium_lamps', origins=('off_frame',), surface='dry'):
    return dict(style_id=style, style_name='Warm directional street illumination', weather='night',
        surface_state=surface, lighting='Warm local ground illumination', exposure='Natural nocturnal camera response',
        illumination=[dict(origin=origin, evidence='Outside the left crop' if origin=='off_frame' else 'Existing source object',
            surfaces='Existing road and curb', direction='From upper left toward the curb',
            falloff='Dimmer toward the far side') for origin in origins],
        appearance_prompt='Relight this exact paved street as a natural late night photograph. '
            'Warm directional light from outside the left edge illuminates the existing curb and nearby road, '
            'fading across the far side into weak ambient darkness. Keep material colours varied and useful '
            'permanent details readable with restrained highlights and modest photographic noise.')


def plan(evidence, entry=None):
    entry = variant() if entry is None else entry
    return policy.parse_plan(json.dumps(dict(variants=[entry])), evidence, entry['style_id'])


class GroundedNightV2Tests(unittest.TestCase):
    def test_exactly_29_active_styles_and_mercury_retired_without_editing_original(self):
        original_sha = common.file_sha256(old.CATALOG_PATH)
        self.assertEqual(len(policy.STYLES), 29)
        self.assertEqual(set(old.STYLES)-set(policy.STYLES), {'mercury_vapor'})
        self.assertEqual(policy.RETIRED_STYLES, {'mercury_vapor'})
        self.assertIn('mercury_vapor', old.STYLES)
        self.assertEqual(common.file_sha256(old.CATALOG_PATH), original_sha)
        self.assertNotEqual(policy.CATALOG_PATH, old.CATALOG_PATH)
        self.assertFalse(any('prompt' in style for style in policy.STYLES.values()))
        for style in policy.STYLES.values():
            self.assertTrue(all(set(clause) <= set(policy.FEATURES) for clause in style['requires']))
            self.assertTrue(set(style['allowed_origins']) <= policy.ORIGINS)
            self.assertTrue(all(set(group) <= set(style['allowed_origins']) for group in style['required_origin_groups']))

    def test_car_windows_never_enable_building_or_residential_window_mechanisms(self):
        evidence = source('outdoors', 'visible_vehicles', 'vehicle_windows', 'drivable_road')
        eligible, excluded = policy.eligible_styles(evidence)
        self.assertNotIn('residential_windows', eligible)
        self.assertNotIn('mixed_window_street', eligible)
        self.assertIn('residential_windows', excluded)
        self.assertIn('moonlit_ambient', eligible)
        with self.assertRaises(ValueError):
            policy.compose_request(evidence, 'residential_windows')

    def test_residential_windows_need_both_positive_typed_building_and_residential_evidence(self):
        evidence = source('outdoors', 'building_windows', uncertain=('residential_windows',))
        self.assertNotIn('residential_windows', policy.eligible_styles(evidence)[0])
        evidence['features']['residential_windows']['status'] = 'present'
        self.assertIn('residential_windows', policy.eligible_styles(evidence)[0])
        evidence['features']['building_windows']['status'] = 'uncertain'
        with self.assertRaisesRegex(ValueError, 'BUILDING'):
            policy.source_evidence(evidence)

    def test_glass_door_and_billboard_do_not_authorize_shop_display_or_sign_lighting(self):
        evidence = source('outdoors', 'ordinary_glass_doors', 'billboards', 'building_entrances')
        eligible, _ = policy.eligible_styles(evidence)
        self.assertIn('porch_entrance', eligible)
        self.assertNotIn('shopfronts', eligible)
        self.assertNotIn('existing_sign_glow', eligible)
        evidence['features']['shopfront_display_windows']['status'] = 'present'
        with self.assertRaisesRegex(ValueError, 'BUILDING'):
            policy.source_evidence(evidence)
        evidence['features']['building_windows']['status'] = 'present'
        self.assertIn('shopfronts', policy.eligible_styles(evidence)[0])

    def test_ordinary_entrance_does_not_enable_covered_passage(self):
        evidence = source('outdoors', 'building_entrances')
        self.assertNotIn('covered_fluorescent', policy.eligible_styles(evidence)[0])
        evidence['features']['covered_passage']['status'] = 'uncertain'
        self.assertNotIn('covered_fluorescent', policy.eligible_styles(evidence)[0])
        evidence['features']['covered_passage']['status'] = 'present'
        self.assertIn('covered_fluorescent', policy.eligible_styles(evidence)[0])

    def test_no_visible_fixture_requires_off_frame_light_with_surface_direction_and_falloff(self):
        evidence = source('outdoors', 'industrial_site')
        entry = variant('industrial_security')
        value = plan(evidence, entry)
        rendered = policy.render_prompt(value['variants'][0])
        self.assertIn('OUTSIDE THE FRAME', rendered)
        self.assertIn('direction: From upper left', rendered)
        self.assertIn('falloff: Dimmer', rendered)
        self.assertIn('do not insert an in-frame emitter', rendered)
        entry['illumination'][0]['origin'] = 'visible_fixture'
        with self.assertRaisesRegex(ValueError, 'off frame'):
            plan(evidence, entry)
        entry['illumination'][0]['origin'] = 'off_frame'
        entry['illumination'][0]['falloff'] = ''
        with self.assertRaises(ValueError):
            plan(evidence, entry)

    def test_existing_fixed_fixtures_can_be_used_but_are_not_required_on_a_street(self):
        evidence = source('outdoors', 'drivable_road')
        plan(evidence)
        entry = variant(origins=('visible_fixture',))
        with self.assertRaises(ValueError):
            plan(evidence, entry)
        evidence['features']['visible_fixed_luminaires']['status'] = 'present'
        plan(evidence, entry)

    def test_headlights_need_actual_vehicle_front_faces_and_new_vehicle_needs_free_space(self):
        evidence = source('outdoors', 'drivable_road', 'visible_vehicles', 'vehicle_windows')
        self.assertNotIn('headlights', policy.eligible_styles(evidence)[0])
        self.assertNotIn('tail_lights', policy.eligible_styles(evidence)[0])
        evidence['features']['visible_headlight_faces']['status'] = 'present'
        self.assertIn('headlights', policy.eligible_styles(evidence)[0])
        plan(evidence, variant('headlights', ('existing_vehicle',)))
        self.assertNotIn('passing_vehicle_lights', policy.eligible_styles(evidence)[0])
        evidence['features']['safe_vehicle_space']['status'] = 'present'
        plan(evidence, variant('passing_vehicle_lights', ('introduced_vehicle',)))

    def test_mixed_window_street_requires_two_distinct_supported_light_origins(self):
        evidence = source('outdoors', 'drivable_road', 'building_windows')
        with self.assertRaisesRegex(ValueError, 'required'):
            plan(evidence, variant('mixed_window_street', ('existing_opening',)))
        plan(evidence, variant('mixed_window_street', ('existing_opening', 'off_frame')))
        with self.assertRaises(ValueError):
            plan(evidence, variant('sodium_lamps', ('existing_opening',)))

    def test_contract_separates_preservation_from_allowed_dynamic_and_declared_ground_changes(self):
        evidence = source('outdoors', 'drivable_road', 'paved_ground')
        entry = plan(evidence)['variants'][0]
        rendered = policy.render_prompt(entry)
        self.assertLess(rendered.index('MUST PRESERVE'), rendered.index('MAY CHANGE'))
        self.assertIn('people and vehicles may be added, removed or changed', rendered)
        self.assertIn('existing sign outlines and lettering', rendered)
        self.assertIn('Declared ground surface state: dry.', rendered)
        self.assertIn('fixed facilities and their positions and shapes', rendered)
        with self.assertRaises(ValueError):
            plan(evidence, variant(surface='unspecified'))
        with self.assertRaises(ValueError):
            plan(evidence, variant('wet_reflections', surface='dry'))
        plan(evidence, variant('wet_reflections', surface='wet'))
        plan(evidence, variant('damp_matte', surface='mixed'))

    def test_source_analysis_does_not_mutate_input_or_claim_a_detection(self):
        evidence = source('outdoors', 'drivable_road')
        original = copy.deepcopy(evidence)
        result = policy.source_evidence(evidence)
        self.assertEqual(evidence, original)
        envelope = result['protected_regions'][0]
        self.assertEqual(envelope['box'], [0, 0, 1000, 1000])
        self.assertFalse(envelope['usable_as_correspondence_or_semantic_evidence'])
        self.assertEqual(envelope['origin'], 'deterministic_preservation_contract')
        broken = copy.deepcopy(evidence); broken['features']['windows'] = broken['features'].pop('building_windows')
        with self.assertRaises(ValueError):
            policy.source_evidence(broken)

    def test_insufficient_or_uncertain_evidence_is_not_forced_to_improve_category_balance(self):
        evidence = source('outdoors', 'vehicle_windows')
        eligible, _ = policy.eligible_styles(evidence)
        counts = {key: dict(saved=1000) for key in eligible}
        counts['residential_windows'] = dict(saved=0)
        selected = policy.select_styles(eligible, 1, counts, seed=42, source_key='actual-car-only')
        self.assertNotIn('residential_windows', selected)
        self.assertNotIn('mercury_vapor', selected)
        self.assertEqual(selected, policy.select_styles(list(reversed(eligible)), 1, counts, seed=42, source_key='actual-car-only'))
        with self.assertRaises(ValueError):
            policy.select_styles(['mercury_vapor'], 1, {})
        self.assertEqual(policy.eligible_styles(source())[0], [])

    def test_least_exposed_compatible_style_includes_pending_and_local_reservations(self):
        eligible = ['sodium_lamps', 'warm_led']
        counts = {'sodium_lamps': dict(saved=10, mismatch=8), 'warm_led': dict(saved=3)}
        self.assertEqual(policy.select_styles(eligible, 1, counts), ['sodium_lamps'])
        self.assertEqual(policy.select_styles(eligible, 1, counts, local=['sodium_lamps', 'sodium_lamps']), ['warm_led'])
        with self.assertRaises(ValueError):
            policy.select_styles(eligible, 1, {'sodium_lamps': dict(saved=0, mismatch=1)})

    def test_plan_cannot_change_requested_style_night_domain_or_preservation_envelope(self):
        value = plan(source('outdoors', 'drivable_road'))
        for field, invalid in (('weather', 'day'), ('style_id', 'mercury_vapor')):
            broken = copy.deepcopy(value); broken['variants'][0][field] = invalid
            with self.assertRaises(ValueError):
                policy.validate_plan(broken)
        broken = copy.deepcopy(value); broken['requested_style_ids'] = ['warm_led']
        with self.assertRaises(ValueError):
            policy.validate_plan(broken)
        broken = copy.deepcopy(value); broken['protected_regions'][0]['box'] = [0, 0, .9, 1]
        with self.assertRaises(ValueError):
            policy.validate_plan(broken)

    def test_requests_retain_open_prose_and_explicit_category_distinctions(self):
        analysis = policy.analysis_request()
        for word in ('VEHICLE windows', 'BUILDING windows', 'glass door', 'billboard', 'covered passage'):
            self.assertIn(word, analysis)
        self.assertIn('You plan image-conditioned augmentation for visual place recognition.', policy.SYSTEM)
        self.assertIn('These are open-ended descriptions, not a fixed list of prompt templates.', policy.SYSTEM)
        request = policy.compose_request(source('outdoors', 'industrial_site'), 'industrial_security')
        for word in ('outside the frame', 'surfaces', 'direction', 'decay', 'MUST PRESERVE', 'MAY CHANGE'):
            self.assertIn(word, request)
        with self.assertRaises(ValueError):
            policy.render_prompt(variant('mercury_vapor'))

    def test_off_frame_metadata_cannot_hide_new_or_wall_mounted_lamps_in_prose(self):
        evidence = source('outdoors', 'industrial_site')
        for claim in ('Add a row of industrial lamps along the wall.',
                      'Wall-mounted security lights illuminate the factory wall.',
                      'Install fluorescent tubes along the overhead wall.',
                      'Existing fixtures lining the industrial wall glow brightly.'):
            entry = variant('industrial_security')
            entry['appearance_prompt'] += ' '+claim
            with self.subTest(claim=claim), self.assertRaisesRegex(ValueError, 'fixture'):
                plan(evidence, entry)

    def test_off_frame_effects_and_explicit_prohibitions_do_not_get_misclassified_as_fixtures(self):
        evidence = source('outdoors', 'industrial_site', 'building_entrances')
        for prose in ('Do not add a row of wall-mounted lamps. Illuminate the existing wall with directional light.',
                      'Do not install fluorescent tubes; let off-frame lights illuminate the existing wall.',
                      'Light from existing lamps outside the frame fades across the entrance surface.',
                      'Off-frame industrial lamps cast light onto the existing wall with gradual falloff.'):
            entry = variant('industrial_security'); entry['appearance_prompt'] += ' '+prose
            plan(evidence, entry)

    def test_car_only_prose_cannot_smuggle_in_residential_window_glow(self):
        evidence = source('outdoors', 'drivable_road', 'vehicle_windows', 'visible_vehicles')
        for claim in ('Residential windows glow softly above the road.',
                      'Warm window glow spills onto the ground.',
                      'Illuminate the lit apartment windows along the facade.'):
            entry = variant(); entry['appearance_prompt'] += ' '+claim
            with self.subTest(claim=claim), self.assertRaisesRegex(ValueError, 'window'):
                plan(evidence, entry)
        entry = variant(); entry['appearance_prompt'] += ' Do not add glowing residential windows. Vehicle windows reflect weak ambient light.'
        plan(evidence, entry)

    def test_plain_building_windows_cannot_be_promoted_to_residential_light_in_prose(self):
        evidence = source('outdoors', 'drivable_road', 'building_windows')
        entry = variant(); entry['appearance_prompt'] += ' Residential windows glow with warm light.'
        with self.assertRaisesRegex(ValueError, 'residential'):
            plan(evidence, entry)

    def test_plain_door_and_entrance_cannot_be_smuggled_into_shop_or_roofed_passage_prose(self):
        evidence = source('outdoors', 'drivable_road', 'ordinary_glass_doors', 'building_entrances', 'billboards')
        for claim in ('Bright shop windows spill light over the road.',
                      'An illuminated storefront provides warm ground lighting.',
                      'A covered passage casts fluorescent light onto the road.',
                      'The fluorescent-lit corridor glows brightly behind the entrance.'):
            entry = variant(); entry['appearance_prompt'] += ' '+claim
            with self.subTest(claim=claim), self.assertRaises(ValueError):
                plan(evidence, entry)
        entry = variant(); entry['appearance_prompt'] += ' Do not invent an illuminated storefront or covered passage. Illuminate the existing entrance surface from off-frame light.'
        plan(evidence, entry)

    def test_analysis_request_includes_complete_nested_shape_not_a_flat_status_map(self):
        request = policy.analysis_request()
        shape = json.loads(request.split('REQUIRED COMPLETE RESPONSE SHAPE (replace all placeholders, retain every key and nested object):\n', 1)[1].split('\nReturn exactly', 1)[0])
        self.assertEqual(set(shape), {'scene_summary', 'features'})
        self.assertEqual(set(shape['features']), set(policy.FEATURES))
        self.assertTrue(all(set(item)=={'status', 'evidence'} for item in shape['features'].values()))
        self.assertIn('status STRING alone is forbidden', request)
        flat = source('outdoors'); flat['features'] = {key: 'present' for key in policy.FEATURES}
        with self.assertRaises(ValueError):
            policy.source_evidence(flat)
        with self.assertRaises(ValueError):
            policy.source_evidence(flat['features'])

    def test_analysis_requests_english_evidence_and_quote_safe_sign_descriptions(self):
        request = policy.analysis_request()
        self.assertIn('ALL scene_summary and evidence strings must be ENGLISH', request)
        self.assertIn('including every repair', request)
        self.assertIn('rather than transcribing sign lettering', request)
        self.assertIn('Do not put quoted OCR text inside evidence strings', request)
        self.assertIn('escape any unavoidable internal double quote', request)
        valid = source('outdoors', 'existing_signs')
        valid['features']['existing_signs']['evidence'] = 'Existing sign reads "PICK UP"'
        self.assertEqual(policy.source_evidence(policy.parse_json(json.dumps(valid)))['features'],
                         valid['features'])
        broken = json.dumps(valid).replace('\\"PICK UP\\"', '"PICK UP"')
        # Single-line objects are deliberately outside the narrow fallback.
        with self.assertRaises(ValueError):
            policy.parse_json(broken)

    def test_complete_source_quote_repair_is_lossless_and_reports_the_exact_field(self):
        evidence = source('outdoors', 'existing_signs')
        evidence['features']['existing_signs']['evidence'] = 'red sign, yellow sign, "PICK UP" text, "CUST" text'
        raw = json.dumps(evidence, indent=2).replace('\\"PICK UP\\"', '"PICK UP"').replace('\\"CUST\\"', '"CUST"')
        parsed, changes = policy.parse_json_with_diagnostics(raw)
        self.assertEqual(parsed, evidence)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]['field'], 'evidence')
        self.assertEqual(changes[0]['escaped_quote_count'], 4)
        self.assertEqual(policy.parse_json(raw), evidence)
        self.assertEqual(policy.parse_json_with_diagnostics(json.dumps(evidence, indent=2)), (evidence, []))

    def test_source_quote_repair_preserves_existing_escapes_and_summary_quotations(self):
        evidence = source('outdoors', 'existing_signs')
        evidence['scene_summary'] = 'Existing street beside sign marked "PICK UP" and parked cars'
        evidence['features']['existing_signs']['evidence'] = 'Existing sign shows "CUST" beside the curb'
        raw = json.dumps(evidence, indent=2).replace('\\"PICK UP\\"', '"PICK UP"')
        parsed, changes = policy.parse_json_with_diagnostics(raw)
        self.assertEqual(parsed, evidence)
        self.assertEqual(changes, [dict(kind='escape_unescaped_source_text_quotes',
            field='scene_summary', line=2, escaped_quote_count=2)])

    def test_source_quote_repair_refuses_missing_fields_structure_status_and_ambiguity(self):
        evidence = source('outdoors', 'existing_signs')
        evidence['features']['existing_signs']['evidence'] = 'Actual sign marked "PICK UP"'
        raw = json.dumps(evidence, indent=2).replace('\\"PICK UP\\"', '"PICK UP"')
        missing = copy.deepcopy(evidence); del missing['features']['covered_passage']
        missing_raw = json.dumps(missing, indent=2).replace('\\"PICK UP\\"', '"PICK UP"')
        invalids = (
            raw[:-1],
            raw.replace('"outdoors": {', '"outdoors" {'),
            raw.replace('"status": "present"', '"status": "pre"sent"', 1),
            raw.replace('"status": "present"', '"status": "invented"', 1),
            raw.replace('Actual sign marked "PICK UP"', 'Actual sign marked "PICK UP'),
            raw.replace('Actual sign marked "PICK UP"', 'Actual sign marked "PICK UP" {broken}'),
            raw.replace('Actual sign marked "PICK UP"', 'Actual sign marked "PICK UP", "status": "present"'),
            raw.replace('Actual sign marked "PICK UP"', 'Actual sign marked "PICK UP" \\x'),
            raw.replace('"status": "present",', '"status": "present", "status": "absent",', 1),
            missing_raw,
            raw.replace('"evidence": "Actual sign marked "PICK UP""',
                        '"evidence": "Actual sign marked "PICK UP"", "other": "value"'),
        )
        for invalid in invalids:
            with self.subTest(raw=invalid[-180:]), self.assertRaises(ValueError):
                policy.parse_json_with_diagnostics(invalid)
        with self.assertRaises(ValueError):
            policy.parse_json(json.dumps(evidence).replace('\\"PICK UP\\"', '"PICK UP"'))
        # The fallback is source-only and never repairs malformed composition.
        entry = variant(); entry['appearance_prompt'] += ' Existing sign marked "PICK UP".'
        composition = json.dumps(dict(variants=[entry]), indent=2).replace('\\"PICK UP\\"', '"PICK UP"')
        with self.assertRaises(ValueError):
            policy.parse_json(composition)

    def test_unspaced_chinese_source_text_keeps_logical_gates_and_english_composition(self):
        evidence = source('outdoors', 'drivable_road', 'visible_vehicles', 'vehicle_windows')
        evidence['scene_summary'] = '晴朗白天住宅街道旁有车辆树木草地与人行道'
        for name, item in evidence['features'].items():
            item['evidence'] = '原图可见车辆道路' if item['status'] == 'present' else '原图未见该类设施'
        parsed = policy.source_evidence(evidence)
        self.assertEqual(parsed['features'], evidence['features'])
        self.assertNotIn('residential_windows', policy.eligible_styles(parsed)[0])
        plan(evidence, variant('led_lamps'))
        evidence['scene_summary'] = '住宅街道'
        with self.assertRaisesRegex(ValueError, 'summary'):
            policy.source_evidence(evidence)
        evidence['scene_summary'] = '晴朗白天住宅街道旁有车辆树木草地与人行道'
        evidence['features']['outdoors']['evidence'] = '天空'
        with self.assertRaisesRegex(ValueError, 'outdoors'):
            policy.source_evidence(evidence)
        evidence['features']['outdoors']['evidence'] = '原图可见晴朗天空'
        entry = variant('led_lamps'); entry['appearance_prompt'] = '将原图编辑为夜间，保留所有建筑结构。'
        with self.assertRaises(ValueError):
            plan(evidence, entry)

    def test_composer_origin_shape_uses_off_frame_when_actual_fixtures_are_absent(self):
        evidence = source('outdoors', 'drivable_road', 'paved_ground')
        request = policy.compose_request(evidence, 'neutral_led')
        shape = json.loads(request.split('Return this complete JSON shape with every placeholder replaced:\n', 1)[1]
                           .split('\nJSON only', 1)[0])
        self.assertEqual([item['origin'] for item in shape['variants'][0]['illumination']], ['off_frame'])
        self.assertIn('Mandatory source-supported origin groups: [["off_frame"]]', request)
        self.assertIn('ambient NEVER replaces off_frame or visible_fixture', request)
        self.assertNotIn('use an allowed off_frame or ambient origin', request)
        with self.assertRaisesRegex(ValueError, 'required source-grounded'):
            plan(evidence, variant('neutral_led', ('ambient',)))
        plan(evidence, variant('neutral_led', ('off_frame',)))

    def test_composer_origin_shape_keeps_both_mixed_window_requirements(self):
        evidence = source('outdoors', 'drivable_road', 'building_windows')
        request = policy.compose_request(evidence, 'mixed_window_street')
        shape = json.loads(request.split('Return this complete JSON shape with every placeholder replaced:\n', 1)[1]
                           .split('\nJSON only', 1)[0])
        origins = [item['origin'] for item in shape['variants'][0]['illumination']]
        self.assertEqual(origins, ['existing_opening', 'off_frame'])
        self.assertIn('Mandatory source-supported origin groups: [["existing_opening"], ["off_frame"]]', request)
        evidence['features']['visible_fixed_luminaires']['status'] = 'present'
        request = policy.compose_request(evidence, 'mixed_window_street')
        self.assertIn('[["existing_opening"], ["visible_fixture", "off_frame"]]', request)

    def test_composer_feedback_targets_missing_origin_without_promoting_unknown_evidence(self):
        evidence = source('outdoors', 'drivable_road', uncertain=('visible_fixed_luminaires', 'building_windows'))
        original = copy.deepcopy(evidence)
        entry = variant('neutral_led', ('ambient',))
        raw = json.dumps(dict(variants=[entry]))
        feedback = policy.compose_feedback(raw, evidence, 'neutral_led', 'Missing required origin')
        diagnostics = json.loads(feedback.split('Repair diagnostics: ', 1)[1].split('\nMANDATORY CORRECTIONS', 1)[0])
        self.assertEqual(diagnostics['current_origins'], ['ambient'])
        self.assertEqual(diagnostics['missing_origin_groups'], [['off_frame']])
        self.assertEqual(diagnostics['source_supported_origins'], ['off_frame', 'ambient'])
        self.assertEqual(diagnostics['missing_fields'], [])
        self.assertIn('retaining valid semantics', feedback)
        self.assertEqual(evidence, original)
        self.assertEqual(entry['illumination'][0]['origin'], 'ambient')

    def test_composer_feedback_handles_malformed_and_incomplete_json_without_repeating_prior(self):
        evidence = source('outdoors', 'drivable_road')
        incomplete = json.dumps(dict(variants=[dict(style_id='neutral_led', illumination=[])]))
        feedback = policy.compose_feedback(incomplete, evidence, 'neutral_led', 'Missing fields')
        diagnostics = json.loads(feedback.split('Repair diagnostics: ', 1)[1].split('\nMANDATORY CORRECTIONS', 1)[0])
        self.assertIn('appearance_prompt', diagnostics['missing_fields'])
        self.assertNotIn('style_id', diagnostics['missing_fields'])
        feedback = policy.compose_feedback('broken response '+('z'*10000), evidence, 'neutral_led', 'Bad JSON')
        diagnostics = json.loads(feedback.split('Repair diagnostics: ', 1)[1].split('\nMANDATORY CORRECTIONS', 1)[0])
        self.assertIn('json_error', diagnostics)
        self.assertEqual(len(diagnostics['previous_response_sha256']), 64)
        self.assertNotIn('z'*100, feedback)
        with self.assertRaises(ValueError):
            policy.compose_feedback(incomplete, evidence, 'residential_windows', 'Wrong style')

    def test_composer_feedback_identifies_repeated_single_word_illumination_evidence(self):
        evidence = source('outdoors', 'drivable_road')
        entry = variant('overcast_skyglow', ('off_frame',))
        entry['illumination'][0]['evidence'] = 'off_frame'
        raw = json.dumps(dict(variants=[entry]))
        with self.assertRaisesRegex(ValueError, 'illumination evidence'):
            plan(evidence, entry)
        feedback = policy.compose_feedback(raw, evidence, 'overcast_skyglow', 'Missing or too short illumination evidence')
        diagnostics = json.loads(feedback.split('Repair diagnostics: ', 1)[1].split('\nMANDATORY CORRECTIONS', 1)[0])
        self.assertEqual(len(diagnostics['invalid_fields']), 1)
        invalid = diagnostics['invalid_fields'][0]
        self.assertEqual(invalid['field'], 'illumination[0].evidence')
        self.assertEqual(invalid['word_count'], 1)
        self.assertEqual(invalid['value'], 'off_frame')
        self.assertEqual(invalid['min_words'], 2)
        self.assertIn('Rewrite illumination[0].evidence', feedback)
        self.assertIn('actual descriptive source-location evidence, not an origin enum', feedback)
        self.assertIn('Do not repeat the rejected response', feedback)
        self.assertNotIn(entry['appearance_prompt'], feedback)
        self.assertEqual(entry['illumination'][0]['evidence'], 'off_frame')
        request = policy.compose_request(evidence, 'overcast_skyglow')
        self.assertIn('EVERY illumination evidence, surfaces, direction and falloff', request)
        self.assertIn('at least TWO whitespace-separated English words', request)

    def test_composer_feedback_reports_all_prose_field_limits_without_changing_validators(self):
        evidence = source('outdoors', 'drivable_road')
        entry = variant('neutral_led')
        entry.update(style_name='Night', lighting='dim light', exposure='dark', appearance_prompt='very short prose')
        for name in ('evidence', 'surfaces', 'direction', 'falloff'):
            entry['illumination'][0][name] = 'ambient'
        feedback = policy.compose_feedback(json.dumps(dict(variants=[entry])), evidence, 'neutral_led', 'Short prose')
        diagnostics = json.loads(feedback.split('Repair diagnostics: ', 1)[1].split('\nMANDATORY CORRECTIONS', 1)[0])
        invalids = {item['field']: item for item in diagnostics['invalid_fields']}
        self.assertEqual(invalids['style_name']['min_words'], 2)
        self.assertEqual(invalids['lighting']['min_words'], 3)
        self.assertEqual(invalids['exposure']['min_words'], 2)
        self.assertEqual(invalids['appearance_prompt']['min_words'], 40)
        self.assertEqual(invalids['appearance_prompt']['max_words'], 200)
        self.assertEqual(len(invalids), 8)
        entry = variant('neutral_led'); entry['appearance_prompt'] = ' '.join(['night']*201)
        feedback = policy.compose_feedback(json.dumps(dict(variants=[entry])), evidence, 'neutral_led', 'Long prose')
        diagnostics = json.loads(feedback.split('Repair diagnostics: ', 1)[1].split('\nMANDATORY CORRECTIONS', 1)[0])
        self.assertIn('maximum', diagnostics['invalid_fields'][0]['reason'])
        with self.assertRaises(ValueError):
            plan(evidence, entry)

    def test_declared_dry_ground_rejects_affirmative_wet_or_reflective_wet_ground(self):
        evidence = source('outdoors', 'drivable_road', 'paved_ground')
        for claim in ('Wet pavement reflects the warm light.',
                      'The road remains wet beneath the vehicle.',
                      'Watery reflections stretch across the foreground.',
                      'Puddles on the pavement mirror the light.'):
            entry = variant(); entry['appearance_prompt'] += ' '+claim
            with self.subTest(claim=claim), self.assertRaisesRegex(ValueError, 'declared dry'):
                plan(evidence, entry)

    def test_ground_permissions_prohibitions_and_consistent_transition_are_not_wet_claims(self):
        evidence = source('outdoors', 'drivable_road', 'paved_ground')
        for clause in ('Wet pavement is allowed in other variations.',
                       'Do not add wet pavement or wet reflections.',
                       'Dry pavement may become wet in a future edit.'):
            entry = variant(); entry['appearance_prompt'] += ' '+clause
            plan(evidence, entry)
        entry = variant(surface='wet')
        entry['appearance_prompt'] += ' Change the original dry road to a wet surface.'
        plan(evidence, entry)
        entry = variant(surface='wet'); entry['appearance_prompt'] += ' Keep the road entirely dry.'
        with self.assertRaisesRegex(ValueError, 'declared wet'):
            plan(evidence, entry)
        entry['surface_state'] = 'mixed'
        plan(evidence, entry)

    def test_composer_schema_placeholders_and_non_english_style_names_are_rejected(self):
        evidence = source('outdoors', 'drivable_road')
        for field, invalid in (('style_name', 'original descriptive name'),
                               ('lighting', 'source-grounded lighting mechanism'),
                               ('exposure', 'specific nighttime camera response'),
                               ('style_name', 'REPLACE: original English name'),
                               ('style_name', '短光迹 LED'),
                               ('style_name', 'LED')):
            entry = variant(); entry[field] = invalid
            with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                plan(evidence, entry)
        request = policy.compose_request(evidence, 'led_lamps')
        self.assertIn('at least TWO English words', request)
        self.assertIn('Allowed illumination.origin values for THIS style ONLY: ["visible_fixture", "off_frame", "ambient"]', request)
        evidence['features']['building_windows']['status'] = 'present'
        with self.assertRaisesRegex(ValueError, 'origin conflicts'):
            plan(evidence, variant('led_lamps', ('off_frame', 'existing_opening')))

    def test_light_trails_need_free_motion_space_and_visible_actual_lamp_faces(self):
        evidence = source('outdoors', 'drivable_road', 'visible_vehicles')
        self.assertNotIn('short_light_trails', policy.eligible_styles(evidence)[0])
        evidence['features']['safe_vehicle_space']['status'] = 'present'
        self.assertNotIn('short_light_trails', policy.eligible_styles(evidence)[0])
        evidence['features']['visible_taillight_faces']['status'] = 'present'
        self.assertIn('short_light_trails', policy.eligible_styles(evidence)[0])
        entry = variant('short_light_trails', ('existing_vehicle',))
        entry['appearance_prompt'] += ' Activate the existing rear lamps at night and show short restrained light trails.'
        plan(evidence, entry)
        entry['appearance_prompt'] += ' Do not add light trails.'
        with self.assertRaisesRegex(ValueError, 'removes the requested'):
            plan(evidence, entry)

    def test_explicit_daylight_retention_and_black_sky_only_are_rejected(self):
        evidence = source('outdoors', 'drivable_road')
        for claim in ('Keep the original daylight illumination.',
                      'The facade must remain daytime bright.',
                      'Just darken the sky.',
                      'Do not apply the requested lighting effect.',
                      'No requested effect should be visible.'):
            entry = variant(); entry['appearance_prompt'] += ' '+claim
            with self.subTest(claim=claim), self.assertRaises(ValueError):
                plan(evidence, entry)
        entry = variant(); entry['appearance_prompt'] += ' Change the original daytime lighting to night. Keep the exact daylight viewpoint. Do not preserve daylight illumination.'
        plan(evidence, entry)

    def test_empty_coverage_does_not_allocate_to_retired_category(self):
        result = policy.coverage_snapshot([])
        self.assertEqual(set(result['counts']), set(policy.STYLES))
        self.assertIn('mercury_vapor', result['retired_style_counts'])
        self.assertEqual(result['retired_style_counts']['mercury_vapor']['saved'], 0)


if __name__ == '__main__':
    unittest.main()
