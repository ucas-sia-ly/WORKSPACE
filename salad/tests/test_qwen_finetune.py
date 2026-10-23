"""Growing Qwen pools, frozen resumes and exact source integrity (CPU only)."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_qwen_finetune as qwen


class QwenSnapshotTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.curriculum = self.root / 'qwen_curriculum'
        for name in ('state', 'images/accepted/night', 'images/train'):
            (self.curriculum / name).mkdir(parents=True)
        self.real = self.root / 'gsv'
        (self.real / 'Images/City').mkdir(parents=True)
        self.manifest = self.curriculum / 'state/training_manifest.jsonl'
        self.config = dict(synthetic_manifest=str(self.manifest), real_data=str(self.real),
                           output_dir=str(self.root / 'run1'))
        self.rows = []
        self.add_pair('one')

    def add_pair(self, name, accepted=True, condition='night', **extra):
        original = self.real / f'Images/City/{name}.jpg'
        generated = self.curriculum / f'images/train/{name}.png'
        original.write_bytes(('real ' + name).encode())
        generated.write_bytes(('generated ' + name).encode())
        if accepted:
            domain = self.curriculum / f'images/accepted/{condition}'
            domain.mkdir(exist_ok=True)
            (domain / f'{name}.png').symlink_to(f'../../train/{name}.png')
        row = dict(sample_id=name, city='City', place_id=1, condition=condition,
                   source_path=str(original), output_path=str(generated),
                   source_sha256=qwen.ccr.digest(original), output_sha256=qwen.ccr.digest(generated),
                   passed=True, eligible_for_training=True, weather_ok=True,
                   prompt='Full audit text should not burden the training manifest.', **extra)
        self.rows.append(row)
        self.publish()
        return row

    def publish(self):
        self.manifest.write_text(''.join(json.dumps(row) + '\n' for row in self.rows))

    def prepare(self, config=None):
        config = config or self.config
        return qwen.prepare_inputs(config, Path(config['output_dir']))

    def frozen_rows(self, config=None):
        config = config or self.config
        path = Path(config['output_dir']) / 'inputs/accepted_manifest.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_compact_manifest_keeps_exact_pair_and_full_audit_is_archived(self):
        receipt = self.prepare()
        row = self.frozen_rows()[0]
        self.assertEqual(receipt['accepted_images'], 1)
        self.assertEqual(row['source_path'], self.rows[0]['source_path'])
        self.assertEqual(row['output_sha256'], self.rows[0]['output_sha256'])
        self.assertNotIn('prompt', row)
        with gzip.open(self.root / 'run1/inputs/audit_manifest.jsonl.gz', 'rt') as archive:
            self.assertEqual(json.loads(next(archive)), self.rows[0])
        self.assertEqual(receipt['source_manifest_sha256'], qwen.ccr.digest(self.manifest))

    def test_growth_only_affects_new_experiments_and_resume_does_not_read_live_manifest(self):
        self.prepare()
        self.add_pair('two')
        self.assertEqual(self.prepare()['accepted_images'], 1)
        second = {**self.config, 'output_dir': str(self.root / 'run2')}
        self.assertEqual(self.prepare(second)['accepted_images'], 2)
        self.manifest.unlink()
        (self.curriculum / 'images/accepted/night/one.png').unlink()
        self.assertEqual(self.prepare()['accepted_images'], 1)

    def test_only_intersection_of_accepted_and_quality_flags_is_included(self):
        self.add_pair('not_linked', accepted=False)
        self.add_pair('rejected')
        self.rows[-1]['passed'] = False
        self.add_pair('bad_weather')
        self.rows[-1]['weather_ok'] = False
        self.publish()
        receipt = self.prepare()
        self.assertEqual(receipt['accepted_images'], 1)
        self.assertEqual(receipt['counts']['not_in_accepted'], 1)
        self.assertEqual(receipt['counts']['not_eligible'], 2)
        self.assertEqual(receipt['aliases_without_eligible_record'], 2)

    def test_exact_duplicates_are_deduplicated_but_conflicting_sources_fail_atomically(self):
        self.rows.append(copy.deepcopy(self.rows[0]))
        self.publish()
        self.assertEqual(self.prepare()['counts']['duplicate_rows'], 1)
        self.add_pair('two')
        self.rows[1]['source_path'] = self.rows[2]['source_path']
        self.rows[1]['source_sha256'] = self.rows[2]['source_sha256']
        self.publish()
        other = {**self.config, 'output_dir': str(self.root / 'conflict')}
        with self.assertRaisesRegex(ValueError, 'Conflicting exact source'):
            self.prepare(other)
        self.assertFalse((Path(other['output_dir']) / 'inputs').exists())

    def test_changed_content_and_snapshot_tampering_are_rejected(self):
        self.prepare()
        Path(self.rows[0]['output_path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'content changed'):
            self.prepare()
        snapshot = self.root / 'run1/inputs/accepted_manifest.jsonl'
        snapshot.write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'snapshot changed'):
            self.prepare()

    def test_hash_mismatch_or_manifest_publication_race_leaves_no_inputs(self):
        self.rows[0]['source_sha256'] = '0' * 64
        self.publish()
        with self.assertRaisesRegex(ValueError, 'SHA256 differs'):
            self.prepare()
        self.assertFalse((self.root / 'run1/inputs').exists())
        self.rows[0]['source_sha256'] = qwen.ccr.digest(self.rows[0]['source_path'])
        self.publish()
        aliases = qwen.accepted_images(self.curriculum)
        with patch.object(qwen, 'accepted_images', side_effect=[aliases, {}]), \
                self.assertRaisesRegex(ValueError, 'changed during snapshot'):
            self.prepare()
        self.assertFalse((self.root / 'run1/inputs').exists())

    def test_mask_paths_and_hashes_survive_compaction(self):
        mask = self.curriculum / 'mask.png'
        mask.write_bytes(b'mask')
        self.rows[0]['occlusion_mask_path'] = '../mask.png'
        self.rows[0]['occlusion_mask_sha256'] = qwen.ccr.digest(mask)
        self.publish()
        self.prepare()
        self.assertEqual(self.frozen_rows()[0]['occlusion_mask_path'], str(mask))

    def test_configuration_changes_and_legacy_directories_do_not_overwrite_runs(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            self.prepare({**self.config, 'batch_size': 16})
        other = {**self.config, 'output_dir': str(self.root / 'legacy')}
        Path(other['output_dir']).mkdir()
        (Path(other['output_dir']) / 'plan.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'legacy experiment'):
            self.prepare(other)

    def test_alias_cannot_escape_train_pool_and_random_files_are_not_sources(self):
        (self.curriculum / 'images/accepted/night/one.png').unlink()
        (self.curriculum / 'images/accepted/night/one.png').symlink_to(self.rows[0]['source_path'])
        with self.assertRaisesRegex(ValueError, 'inside images/train'):
            self.prepare()

    def test_one_domain_level_includes_each_weather_without_recursing_unknown_folders(self):
        for condition in sorted(qwen.WEATHER_DOMAINS - {'night'}):
            self.add_pair(condition, condition=condition)
        ignored = self.curriculum / 'images/accepted/notes'
        ignored.mkdir()
        (ignored / 'external.png').symlink_to(self.rows[0]['source_path'])
        nested = self.curriculum / 'images/accepted/night/nested'
        nested.mkdir()
        (nested / 'external.png').symlink_to(self.rows[0]['source_path'])
        receipt = self.prepare()
        self.assertEqual(receipt['accepted_images'], 7)
        self.assertEqual(receipt['conditions'], {condition: 1 for condition in qwen.WEATHER_DOMAINS})
        self.assertTrue(all(Path(alias).parent.name in qwen.WEATHER_DOMAINS
                            for alias in qwen.accepted_images(self.curriculum).values()))

    def test_legacy_flat_alias_is_still_readable(self):
        (self.curriculum / 'images/accepted/night/one.png').unlink()
        flat = self.curriculum / 'images/accepted/one.png'
        flat.symlink_to('../train/one.png')
        self.assertEqual(self.prepare()['accepted_images'], 1)
        self.assertEqual(qwen.accepted_images(self.curriculum), {self.rows[0]['output_path']: str(flat)})

    def test_duplicate_flat_and_domain_aliases_are_rejected(self):
        (self.curriculum / 'images/accepted/one.png').symlink_to('../train/one.png')
        with self.assertRaisesRegex(ValueError, 'Duplicate accepted aliases'):
            self.prepare()

    def test_alias_must_target_same_filename_directly(self):
        generated = self.curriculum / 'images/train/other.png'
        generated.write_bytes(b'other image')
        alias = self.curriculum / 'images/accepted/night/one.png'
        alias.unlink()
        alias.symlink_to('../../train/other.png')
        with self.assertRaisesRegex(ValueError, 'matching filename'):
            self.prepare()

    def test_accepted_domain_and_train_directories_cannot_be_symlinks(self):
        accepted = self.curriculum / 'images/accepted'
        (accepted / 'night').rename(self.root / 'saved-night')
        (accepted / 'night').symlink_to(self.root / 'saved-night', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Accepted domain must be an owned directory'):
            self.prepare()
        (accepted / 'night').unlink()
        (self.root / 'saved-night').rename(accepted / 'night')
        train = self.curriculum / 'images/train'
        train.rename(self.root / 'saved-train')
        train.symlink_to(self.root / 'saved-train', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'owned images/train directory'):
            self.prepare()

    def test_non_alias_image_and_symlink_canonical_are_rejected(self):
        alias = self.curriculum / 'images/accepted/night/one.png'
        alias.unlink()
        alias.write_bytes(b'copied image')
        with self.assertRaisesRegex(ValueError, 'must be symbolic aliases'):
            self.prepare()
        alias.unlink()
        alias.symlink_to('../../train/one.png')
        generated = Path(self.rows[0]['output_path'])
        generated.unlink()
        generated.symlink_to(self.rows[0]['source_path'])
        with self.assertRaisesRegex(ValueError, 'regular canonical file'):
            self.prepare()

    def test_manifest_condition_must_match_domain_alias(self):
        self.rows[0]['condition'] = 'rain'
        self.publish()
        with self.assertRaisesRegex(ValueError, 'accepted domain differs'):
            self.prepare()
        self.assertFalse((self.root / 'run1/inputs').exists())

    def test_layout_change_does_not_refresh_existing_frozen_snapshot(self):
        self.prepare()
        alias = self.curriculum / 'images/accepted/night/one.png'
        alias.unlink()
        (self.curriculum / 'images/accepted/one.png').symlink_to('../train/one.png')
        with patch.object(qwen, 'accepted_images', side_effect=AssertionError('live pool must not be read')):
            self.assertEqual(self.prepare()['accepted_images'], 1)

    def test_entrypoint_checksum_guard_does_not_resign_old_snapshots(self):
        self.prepare()
        receipt_path = self.root / 'run1/inputs/receipt.json'
        receipt = json.loads(receipt_path.read_text())
        receipt['entrypoint_sha256'] = '0' * 64
        receipt_path.write_text(json.dumps(receipt))
        before = receipt_path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'entrypoint changed'):
            self.prepare()
        self.assertEqual(receipt_path.read_bytes(), before)


class QwenCommandTests(unittest.TestCase):
    def test_generic_config_has_live_source_fresh_calibration_and_svox_categories(self):
        config = qwen.ccr.load_config(qwen.ccr.ROOT / 'configs/ccr_qwen.json')
        self.assertTrue(config['synthetic_manifest'].endswith('/qwen_curriculum/state/training_manifest.jsonl'))
        self.assertIsNone(config['pair_calibration'])
        self.assertEqual(config['methods'], ['A', 'M'])
        self.assertEqual(config['batch_size'], 32)
        self.assertEqual({row['name'] for row in config['external_validations']},
                         {'svox_val', 'svox_default', 'svox_night', 'svox_rain',
                          'svox_snow', 'svox_sun', 'svox_overcast'})

    def test_default_does_not_execute_and_snapshot_only_does_not_load_models(self):
        with tempfile.TemporaryDirectory() as directory:
            config = dict(output_dir=directory)
            for arguments, expected in (([], 1), (['--snapshot-only'], 0), (['--execute'], 1)):
                with self.subTest(arguments=arguments), \
                        patch.object(qwen.ccr, 'load_config', return_value=config), \
                        patch.object(qwen, 'prepare_inputs', return_value=dict(accepted_images=1, conditions={'night': 1})), \
                        patch.object(qwen.ccr, 'main') as runner:
                    qwen.main(arguments)
                    self.assertEqual(runner.call_count, expected)
                    if expected:
                        self.assertEqual('--execute' in runner.call_args.args[0], '--execute' in arguments)


if __name__ == '__main__':
    unittest.main()
