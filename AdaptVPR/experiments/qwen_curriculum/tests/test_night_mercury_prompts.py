import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.qwen_curriculum import common, run, balanced_night
from experiments.qwen_curriculum import night_mercury_prompts as replacement


class ReplacementTests(unittest.TestCase):
    def inputs(self):
        evidence = dict(coordinate_system='0_1000', scene_summary='An outdoor street with old building facades',
            features={k: dict(status='present', evidence='Visible source evidence') for k in replacement.policy.FEATURES},
            protected_regions=[dict(description='An old policy envelope', box=[0, 0, 1, 1])])
        source = dict(city='city', place_id=7, source_path='/unchanged/source.jpg', source_sha256='a'*64,
            sample_id='qnsource_123', source_dimensions=[640, 480])
        task = dict(source=source, task_reason='retired_mercury_style', style_id='mercury_vapor',
            sample_id='qnight_'+'b'*24, original_record_sha256='c'*64,
            original_job=dict(prompt='old mercury prose'))
        receipt = dict(source=source, job=task['original_job'], status='ready', result_sha256='d'*64,
            plan=dict(source_evidence=evidence))
        return task, receipt

    def test_original_evidence_and_source_are_kept_without_mercury(self):
        task, receipt = self.inputs(); saved = copy.deepcopy(receipt)
        item = replacement.allocate([task], [receipt], {})[0]
        self.assertNotEqual(item['style_id'], 'mercury_vapor')
        self.assertEqual(item['source'], task['source'])
        self.assertEqual(receipt, saved)
        self.assertEqual(item['evidence']['protected_regions'][0]['box'], [0, 0, 1000, 1000])
        self.assertFalse(item['evidence']['protected_regions'][0]['usable_as_correspondence_or_semantic_evidence'])

    def test_duplicates_and_changed_source_rejected(self):
        task, receipt = self.inputs()
        with self.assertRaises(ValueError):
            replacement.allocate([task, task], [receipt, receipt], {})
        wrong = copy.deepcopy(receipt); wrong['source']['place_id'] = 8
        with self.assertRaises(ValueError):
            replacement.allocate([task], [wrong], {})

    def test_least_covered_eligible_style_is_deterministic(self):
        task, receipt = self.inputs()
        counts = {s: dict(saved=100) for s in replacement.policy.STYLES}
        counts['moonlit_ambient'] = dict(saved=0)
        first = replacement.allocate([task], [receipt], counts)
        self.assertEqual(first, replacement.allocate([task], [receipt], counts))
        self.assertEqual(first[0]['style_id'], 'moonlit_ambient')

    def test_frozen_ready_job_rejects_prompt_source_and_predecessor_changes(self):
        task, receipt = self.inputs(); item = replacement.allocate([task], [receipt], {})[0]
        variant = dict(style_id=item['style_id'], style_name='Distinct night lighting', weather='night',
            surface_state='dry', lighting='Natural dim ambient night illumination', exposure='Dark night exposure',
            appearance_prompt=' '.join(['Preserve this original street and building geometry at night.']*6))
        plan = replacement.cohort.parse_plan(json.dumps(dict(variants=[variant])), item['evidence'], item['style_id'])
        cfg = dict(fingerprint='test', seed=42)
        job = balanced_night.make_jobs([item['source']], [plan], False, 42)[0]
        record = run.sealed(dict(status='ready', config_fingerprint='test', input_sha256=item['result_sha256'],
            source=item['source'], predecessor_sample_id=item['predecessor_sample_id'], plan=plan, job=job))
        replacement.validate_receipt(record, item, cfg)
        for field, value in [('prompt', 'changed'), ('source_path', '/another.jpg'), ('style_id', 'mercury_vapor')]:
            wrong = copy.deepcopy(record); wrong['job'][field] = value; wrong = run.sealed(wrong)
            with self.assertRaises(ValueError):
                replacement.validate_receipt(wrong, item, cfg)

    def test_resume_complete_prompt_never_calls_vlm(self):
        task, receipt = self.inputs(); item = replacement.allocate([task], [receipt], {})[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); common.write_jsonl(root/'state/sources.jsonl', [item])
            record = dict(saved=True)
            common.write_json(replacement.receipt_path(root/'state', item), record)
            cfg = dict(fingerprint='test', seed=42, model_path='/not_loaded', batch_size=64)
            with patch.object(replacement, 'load', return_value=cfg), \
                    patch.object(replacement, 'validate_receipt', return_value=record), \
                    patch.object(replacement.cohort, 'sealed_read', return_value=record), \
                    patch.object(replacement.planner, 'Planner') as reader:
                replacement.plan(root)
                reader.return_value.ask.assert_not_called()
                self.assertTrue((root/'state/prompts_complete.json').exists())


if __name__ == '__main__':
    unittest.main()
