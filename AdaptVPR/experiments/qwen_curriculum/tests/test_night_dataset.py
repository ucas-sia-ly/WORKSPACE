import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.qwen_curriculum import night_dataset as dataset


class NightDatasetTests(unittest.TestCase):
    def fixture(self, root):
        paths = [root/'real.jpg', root/'night.png', root/'raw.png', root/'raw.json']
        for index, path in enumerate(paths):
            path.write_bytes(b'original exact bytes '+str(index).encode())
        row = dict(sample_id='qnight_'+'a'*24, city='Chicago', place_id=5496,
            condition='night', style_id='sparse_lamps', source_path=str(paths[0]),
            output_path=str(paths[1]), raw_output_path=str(paths[2]),
            source_sha256=dataset.sha256(paths[0]), output_sha256=dataset.sha256(paths[1]),
            raw_output_sha256=dataset.sha256(paths[2]), metadata_sha256=dataset.sha256(paths[3]),
            prompt='Keep permanent structures.', status='pending_verification', eligible_for_training=False)
        row = dataset.sealed(row)
        manifest = root/'results.jsonl'; manifest.write_text(json.dumps(row)+'\n')
        return row, manifest

    def test_pair_is_exact_source_and_hardlinked_images_not_duplicate_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); row, manifest = self.fixture(root)
            saved = dataset.consolidate_one(('campaign', manifest, row), root/'dataset')
            self.assertTrue(Path(saved['source_link']).is_symlink())
            self.assertTrue(os.path.samefile(saved['output_path'], row['output_path']))
            self.assertEqual(saved['source_path'], row['source_path'])
            self.assertEqual(saved['prompt'], row['prompt'])
            self.assertFalse(saved['eligible_for_training'])
            self.assertEqual(saved['original_result_sha256'], row['result_sha256'])
            dataset.verify_one(saved)

    def test_changed_source_prevents_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); row, manifest = self.fixture(root)
            Path(row['source_path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                dataset.consolidate_one(('campaign', manifest, row), root/'dataset')

    def test_duplicate_places_or_ids_and_mercury_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); row, manifest = self.fixture(root)
            for another in [row, dataset.sealed({**row, 'sample_id': 'qnight_'+'b'*24}),
                            dataset.sealed({**row, 'style_id': 'mercury_vapor'})]:
                manifest.write_text(json.dumps(row)+'\n'+json.dumps(another)+'\n')
                with patch.object(dataset, 'input_manifests', return_value=[('campaign', manifest)]):
                    with self.assertRaises(ValueError):
                        list(dataset.collect_inputs(root))

    def test_build_resume_and_full_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); row, manifest = self.fixture(root)
            with patch.object(dataset, 'input_manifests', return_value=[('campaign', manifest)]):
                result = dataset.build(root/'dataset', root, expected=1, workers=1)
                self.assertEqual(result['count'], 1)
                self.assertEqual(dataset.build(root/'dataset', root, expected=1)['count'], 1)
                self.assertEqual(dataset.verify(root/'dataset', 1)['verified'], 1)
            Path(row['raw_output_path']).write_bytes(b'corrupted')
            with self.assertRaises(ValueError):
                dataset.verify(root/'dataset', 1)

    def test_incomplete_budget_never_creates_completion_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); _, manifest = self.fixture(root)
            with patch.object(dataset, 'input_manifests', return_value=[('campaign', manifest)]):
                with self.assertRaisesRegex(ValueError, 'Expected 2'):
                    dataset.build(root/'dataset', root, expected=2)
                self.assertFalse((root/'dataset/complete.json').exists())
                self.assertEqual(dataset.build(root/'dataset', root, expected=1)['count'], 1)


if __name__ == '__main__':
    unittest.main()
