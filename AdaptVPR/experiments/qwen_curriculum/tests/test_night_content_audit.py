"""CPU tests: unchanged frozen classifier + production cohort integration."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch

from experiments.qwen_curriculum import night_content_audit as cli
from experiments.qwen_curriculum.tests.test_content_audit_v42 import controls
from experiments.qwen_curriculum import content_audit_v42 as audit
from experiments.qwen_curriculum import content_audit_v4 as legacy


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.real_sha = legacy.sha256
        self.pairs = controls()
        self.cal = audit.calibrate_controls_v42(self.pairs, .05, min_pairs=3)

    def fixture(self, index=1):
        row, signal = copy.deepcopy(self.pairs[index])
        evidence = copy.deepcopy(row)
        return row, evidence, signal

    def test_real_content_rejection_and_diagnostic_box(self):
        row, evidence, signal = self.fixture()
        with patch.object(legacy, 'sha256', side_effect=lambda p: row['source_sha256'] if p == row['source_path'] else row['output_sha256'] if p == row['output_path'] else self.real_sha(p)):
            result = cli.evaluate(row, evidence, signal, self.cal)
        self.assertEqual(result['content_audit']['decision'], 'reject')
        self.assertEqual(result['decision_source'], 'local_content')
        self.assertIsNotNone(result['peak_region'])
        self.assertFalse(result['peak_region_is_semantic_defect'])

    def test_policy_rejection_is_not_local_detection(self):
        row, evidence, signal = self.fixture(0)
        row['prompt_variant'] = 'shopfronts'
        with patch.object(legacy, 'sha256', side_effect=lambda p: row['source_sha256'] if p == row['source_path'] else row['output_sha256'] if p == row['output_path'] else self.real_sha(p)):
            result = cli.evaluate(row, evidence, signal, self.cal)
        self.assertEqual(result['decision_source'], 'prompt_policy')
        self.assertEqual(result['content_audit']['decision'], 'reject')

    def test_missing_base_bad_hash_or_geometry_cannot_approve(self):
        row, evidence, signal = self.fixture(0)
        with patch.object(legacy, 'sha256', side_effect=lambda p: row['source_sha256'] if p == row['source_path'] else row['output_sha256'] if p == row['output_path'] else self.real_sha(p)):
            for bad in [None, {**evidence, 'output_sha256': 'other'}]:
                result = cli.evaluate(row, bad, signal, self.cal)
                self.assertEqual(result['content_audit']['decision'], 'unknown')
            changed = copy.deepcopy(evidence)
            changed['estimated_H'][0][2] = .01
            self.assertEqual(cli.evaluate(row, changed, signal, self.cal)['content_audit']['decision'], 'unknown')
        with patch.object(legacy, 'sha256', return_value='changed'):
            self.assertEqual(cli.evaluate(row, evidence, signal, self.cal)['content_audit']['decision'], 'unknown')

    def test_local_pass_is_not_unconditional_training_approval(self):
        row, evidence, signal = self.fixture(0)
        with patch.object(legacy, 'sha256', side_effect=lambda p: row['source_sha256'] if p == row['source_path'] else row['output_sha256'] if p == row['output_path'] else self.real_sha(p)):
            result = cli.evaluate(row, evidence, signal, self.cal)
        self.assertTrue(result['safety_screen_passed'])
        self.assertFalse(result['eligible_for_training'])

    def test_calibration_tampering_or_holdout_failure(self):
        cli.validate_calibration(self.cal)
        bad = copy.deepcopy(self.cal);bad['rules']['thresholds']['reject_above'] = 0
        with self.assertRaises(ValueError): cli.validate_calibration(bad)
        bad = copy.deepcopy(self.cal);bad['usable'] = False
        with self.assertRaises(ValueError): cli.validate_calibration(bad)

    def test_online_provider_relocation_allowed_but_model_change_rejected(self):
        from types import SimpleNamespace
        expected = self.cal['feature_pipeline']['provider']
        args = Namespace(dino_repo='/new/repo', dino_checkpoint='/new/weights',
                         device='cpu', threads=1)
        provider = SimpleNamespace(definition={**expected, 'repository': '/new/repo'})
        with patch.object(legacy, 'FrozenDINOProvider', return_value=provider):
            loaded = cli.load_provider(args, self.cal)
            self.assertEqual(loaded.definition, expected)
        provider = SimpleNamespace(definition={**expected, 'checkpoint_sha256': '0' * 64})
        with patch.object(legacy, 'FrozenDINOProvider', return_value=provider):
            with self.assertRaisesRegex(ValueError, 'pipeline differs'):
                cli.load_provider(args, self.cal)

    def test_resumable_journal_and_no_default_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            row, signal = copy.deepcopy(self.pairs[0])
            for key, value in [('source', b'original'), ('output', b'generated')]:
                f = p/(key+'.png');f.write_bytes(value)
                row[key+'_path'] = str(f);row[key+'_sha256'] = legacy.sha256(f)
                signal[key+'_sha256'] = row[key+'_sha256']
            row['estimated_H'] = self.pairs[0][0]['estimated_H']
            for name, values in [('manifest', [row]), ('base', [row]), ('signals', [signal])]:
                (p/(name+'.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in values))
            (p/'cal.json').write_text(json.dumps(self.cal))
            args = Namespace(manifest=str(p/'manifest.jsonl'), base_measurements=str(p/'base.jsonl'),
                             signals=str(p/'signals.jsonl'), calibration=str(p/'cal.json'),
                             output_dir=str(p/'out'),execute=False,resume=False,device='cuda')
            cli.run(args);self.assertFalse((p/'out').exists())
            args.execute=True;cli.run(args)
            journal=p/'out/audited_manifest.jsonl';before=journal.read_bytes()
            with journal.open('ab') as f:f.write(b'{"partial":')
            args.resume=True;cli.run(args)
            self.assertEqual(journal.read_bytes(), before)
            self.assertEqual(json.loads((p/'out/report.json').read_text())['total'], 1)
            (p/'manifest.jsonl').write_text('')
            with self.assertRaisesRegex(ValueError, 'protocol'):cli.run(args)


if __name__ == '__main__':unittest.main()
