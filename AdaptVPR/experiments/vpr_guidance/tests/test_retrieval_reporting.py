"""Paired recovery, path exclusion, and full JSON/table reporting on CPU fixtures."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from AdaptVPR.experiments.vpr_guidance import evaluate_retrieval as evaluator
from AdaptVPR.experiments.vpr_guidance.tests.test_experiment import A, B, C


class RetrievalReportingTests(unittest.TestCase):
    def test_source_capture_and_exact_query_path_are_excluded(self):
        database = torch.tensor([[1., 0.], [.6, .8], [.8, .6]])
        # Exact source outranks all others but is excluded, even from a different directory.
        details = evaluator.retrieval_results(torch.tensor([[1., 0.]]), database, [A],
                                              [f'/db/{A}', f'/db/{B}', f'/db/{C}'], [f'/query/{A}'])
        self.assertEqual(details[0]['positive_database_ids'], [Path(B).stem])
        self.assertEqual(details[0]['positive_rank'], 2)
        # Even a path aliased under a different source_id can never be its own positive.
        details = evaluator.retrieval_results(torch.tensor([[1., 0.]]), database, [A],
                                              [f'/db/{A}', f'/db/{B}', f'/db/{C}'], [f'/db/../db/{B}'])
        self.assertEqual(details[0]['positive_database_ids'], [])
        self.assertIsNone(details[0]['positive_rank'])

    def test_recovery_ratio_example_and_undefined_cases(self):
        m = lambda value: {'recall_at_1': value}
        report = evaluator.recovery_metrics(m(.82), m(.61), m(.73))
        self.assertAlmostEqual(report['recovery_ratio'], .5714285714)
        self.assertAlmostEqual(report['recovery_ratio_percent'], 57.14285714)
        self.assertAlmostEqual(report['r1_generation_loss'], .21)
        self.assertAlmostEqual(report['r1_guidance_gain'], .12)
        for source, baseline in [(.61, .61), (.5, .61)]:
            report = evaluator.recovery_metrics(m(source), m(baseline), m(.73))
            self.assertIsNone(report['recovery_ratio'])
            self.assertIsNone(report['recovery_ratio_percent'])
            self.assertIn('did not reduce', report['recovery_ratio_reason'])
        self.assertGreater(evaluator.recovery_metrics(m(.8), m(.6), m(.9))['recovery_ratio'], 1)
        self.assertLess(evaluator.recovery_metrics(m(.8), m(.6), m(.5))['recovery_ratio'], 0)

    def test_paired_rank_and_cosine_with_ties_and_skips(self):
        def item(index, rank, cosine):
            return {'source_id': f'query_{index}', 'positive_rank': rank,
                    'positive_cosine': cosine, 'positive_database_ids': ['positive'] if rank else []}
        baseline = [item(1, 3, .7), item(2, 1, .8), item(3, 2, .9), item(4, None, None)]
        guided = [item(1, 1, .8), item(2, 1, .8), item(3, 4, .8), item(4, None, None)]
        report = evaluator.paired_statistics(baseline, guided)
        self.assertEqual(report['num_pairs'], 3)
        self.assertEqual([report[k] for k in ('guided_rank_better', 'guided_rank_equal', 'guided_rank_worse')],
                         [1, 1, 1])
        self.assertAlmostEqual(report['guided_win_rate'], 1 / 3)
        self.assertAlmostEqual(report['guided_non_worse_rate'], 2 / 3)
        self.assertAlmostEqual(report['guided_cosine_better_fraction'], 1 / 3)
        with self.assertRaises(ValueError):
            evaluator.paired_statistics(baseline, guided[:-1])
        with self.assertRaises(ValueError):
            evaluator.paired_statistics(baseline, [dict(guided[0], source_id='wrong'), *guided[1:]])

    def test_full_evaluator_three_sets_two_models_conditions_and_jsonl(self):
        class ColorDescriptor:
            def from_pil(self, image):
                pixels = torch.from_numpy(np.asarray(image).copy()).float().mean((0, 1))
                return torch.nn.functional.normalize(pixels[[0, 2]][None], dim=-1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = root / 'db'
            database.mkdir()
            for name, color in [(A, (0, 0, 255)), (B, (153, 0, 204)), (C, (204, 0, 153))]:
                Image.new('RGB', (32, 32), color).save(database / name)
            rows = []
            for index, (condition, baseline_color, guided_color) in enumerate([
                    ('snow', (255, 0, 0), (0, 0, 255)),
                    ('night', (0, 0, 255), (255, 0, 0))]):
                paths = {}
                for variant, color in [('source', (0, 0, 255)), ('baseline', baseline_color),
                                       ('guided', guided_color)]:
                    path = root / f'{index}_{variant}.png'
                    Image.new('RGB', (32, 32), color).save(path)
                    paths[variant] = str(path)
                rows.append({'sample_id': str(index), 'source_id': A, 'route': 'global',
                             'condition': condition, 'seed': 42, 'guidance_scale': .01,
                             'guidance_every': 5, 'guidance_last_n': 10, 'output_paths': paths})
            # Missing captures are skipped without opening their intentionally nonexistent images.
            rows.append(dict(rows[0], sample_id='missing', condition='fog',
                             source_id=A.replace('0000002', '0000099'),
                             output_paths={v: str(root / f'missing_{v}.png') for v in ('source', 'baseline', 'guided')}))
            without_condition = dict(rows[0])
            without_condition.pop('condition')
            self.assertIsNone(evaluator.query_metadata(without_condition)['condition'])
            records = root / 'records.jsonl'
            records.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            summary_path = root / 'summary.json'
            console = io.StringIO()
            with patch('sys.argv', ['evaluate_retrieval', '--records', str(records),
                                    '--database-root', str(database), '--output', str(summary_path),
                                    '--device', 'cpu', '--with-boq']), \
                    patch.object(evaluator, 'load_salad', return_value=ColorDescriptor()), \
                    patch.object(evaluator, 'load_boq', return_value=ColorDescriptor()), \
                    contextlib.redirect_stdout(console):
                evaluator.main()
            summary = json.loads(summary_path.read_text())
            for model in ('salad', 'boq'):
                report = summary['groups'][0][model]
                self.assertEqual(report['source']['recall_at_1'], 1)
                self.assertEqual(report['baseline']['recall_at_1'], .5)
                self.assertEqual(report['guided']['recall_at_1'], .5)
                self.assertEqual(report['recovery_ratio'], 0)
                self.assertEqual(report['source']['num_skipped_no_other_capture'], 1)
                self.assertEqual(set(report['per_condition']), {'snow', 'night', 'fog'})
                self.assertEqual(report['per_condition']['snow']['recovery_ratio'], 1)
                self.assertIsNone(report['per_condition']['night']['recovery_ratio'])
                self.assertEqual(report['per_condition']['fog']['source']['num_queries'], 0)
                self.assertEqual(report['paired_rank']['guided_rank_better'], 1)
                self.assertEqual(report['paired_rank']['guided_rank_worse'], 1)
            details = [json.loads(line) for line in Path(summary['per_query_output']).read_text().splitlines()]
            self.assertEqual(len(details), 3)
            self.assertEqual(details[0]['positive_database_ids'], [Path(B).stem])
            self.assertEqual(details[0]['salad_source_rank'], 1)
            self.assertEqual(details[0]['salad_baseline_rank'], 2)
            self.assertEqual(details[0]['salad_guided_rank'], 1)
            self.assertEqual(details[0]['boq_guided_rank'], 1)
            self.assertNotIn(Path(A).stem, details[0]['positive_database_ids'])
            self.assertIsNone(details[2]['salad_source_rank'])
            self.assertIn('Model: SALAD', console.getvalue())
            self.assertIn('Model: BOQ', console.getvalue())
            self.assertIn('Query type | R@1 | R@5', console.getvalue())
            json.dumps(summary, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
