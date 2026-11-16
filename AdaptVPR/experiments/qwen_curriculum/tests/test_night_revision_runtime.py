import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.qwen_curriculum import campaign, common, night_revision_runtime as runtime


class RuntimeRepairTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.code = self.root / 'code'
        self.code.mkdir()
        patched_directory = patch.object(runtime, 'MODULE_DIR', self.code)
        patched_directory.start()
        self.addCleanup(patched_directory.stop)
        for name in runtime.ORIGINAL_NAMES | {runtime.RUNTIME_NAME}:
            (self.code / name).write_text('original ' + name)
        self.dependencies = self.hashes(False)
        self.policy = {'version': 'scene_balanced_night_v2', 'typed_evidence': True}
        self.cfg = campaign.frozen(self.root / 'state/revision_config.json', dict(
            version='night_revision_v2', condition='night', files_sha256=self.dependencies,
            policy=self.policy, generation_barrier='text planning only; explicit image launch required'))
        self.original_config = (self.root / 'state/revision_config.json').read_bytes()

    def hashes(self, include_runtime=True):
        names = runtime.ORIGINAL_NAMES | ({runtime.RUNTIME_NAME} if include_runtime else set())
        return {str((self.code / name).resolve()): common.file_sha256(self.code / name) for name in names}

    def install(self):
        runtime.capture_original(self.root)
        (self.code / 'night_revision.py').write_text('repaired text retry implementation')
        return runtime.install(self.root, self.hashes(), self.policy)

    def test_capture_preserves_every_original_byte_and_is_idempotent(self):
        first = runtime.capture_original(self.root)
        self.assertEqual(first, runtime.capture_original(self.root))
        archive = self.root / 'state/runtime_patches/original_files'
        for source, digest in self.dependencies.items():
            self.assertEqual((archive / digest).read_bytes(), Path(source).read_bytes())

    def test_cannot_capture_modified_code_as_original(self):
        (self.code / 'night_revision.py').write_text('already modified')
        with self.assertRaisesRegex(ValueError, 'already changed'):
            runtime.capture_original(self.root)
        self.assertFalse((self.root / 'state/runtime_patches/original_files/manifest.json').exists())

    def test_unpatched_verification_enforces_exact_original_dependencies(self):
        value = runtime.verify(self.root, self.cfg, self.dependencies, self.policy)
        self.assertEqual(value['fingerprint'], self.cfg['fingerprint'])
        self.assertNotIn('execution_patch_sha256', value)
        (self.code / 'night_revision.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'dependency changed'):
            runtime.verify(self.root, self.cfg, self.dependencies, self.policy)

    def test_native_new_cohort_can_freeze_runtime_without_a_repair_supplement(self):
        config_path = self.root / 'state/revision_config.json'
        original = {k: v for k, v in self.cfg.items() if k != 'fingerprint'}
        original['files_sha256'] = self.hashes()
        config_path.unlink()
        native = campaign.frozen(config_path, original)
        value = runtime.verify(self.root, native, self.hashes(), self.policy)
        self.assertEqual(value['fingerprint'], native['fingerprint'])
        self.assertEqual(value['effective_files_sha256'], native['files_sha256'])
        self.assertEqual(value['execution_patch_sha256'], common.file_sha256(config_path))
        self.assertEqual(value['execution_identity_kind'], 'native_configuration')

    def test_old_cohort_needs_explicit_installation_when_runtime_is_added(self):
        with self.assertRaisesRegex(ValueError, 'installation required'):
            runtime.verify(self.root, self.cfg, self.hashes(), self.policy)

    def test_patch_binds_code_archive_and_parent_without_reidentifying_jobs(self):
        installed = self.install()
        value = runtime.verify(self.root, self.cfg, self.hashes(), self.policy)
        patch_path = self.root / 'state/runtime_patch.json'
        self.assertEqual(value['execution_patch_sha256'], common.file_sha256(patch_path))
        self.assertEqual(value['execution_identity_kind'], 'runtime_patch')
        self.assertEqual(value['effective_files_sha256'][str(patch_path)], common.file_sha256(patch_path))
        self.assertEqual(value['files_sha256'], self.cfg['files_sha256'])
        self.assertEqual(value['fingerprint'], self.cfg['fingerprint'])
        self.assertEqual((self.root / 'state/revision_config.json').read_bytes(), self.original_config)
        self.assertNotIn(str(patch_path), installed['effective_files_sha256'])
        self.assertFalse(installed['image_generation_authorized'])
        self.assertEqual(installed, runtime.install(self.root, self.hashes(), self.policy))

    def test_install_requires_original_archive(self):
        with self.assertRaises(FileNotFoundError):
            runtime.install(self.root, self.hashes(), self.policy)

    def test_semantic_policy_change_is_rejected_before_install_and_verify(self):
        self.install()
        changed = {**self.policy, 'typed_evidence': False}
        for operation in (
            lambda: runtime.install(self.root, self.hashes(), changed),
            lambda: runtime.verify(self.root, self.cfg, self.hashes(), changed),
        ):
            with self.assertRaisesRegex(ValueError, 'scene policy'):
                operation()

    def test_archive_corruption_is_rejected(self):
        self.install()
        digest = next(iter(self.dependencies.values()))
        (self.root / 'state/runtime_patches/original_files' / digest).write_text('corrupt')
        with self.assertRaisesRegex(ValueError, 'archive missing or changed'):
            runtime.verify(self.root, self.cfg, self.hashes(), self.policy)

    def test_changed_runtime_code_after_install_is_rejected(self):
        self.install()
        (self.code / runtime.RUNTIME_NAME).write_text('unapproved second repair')
        with self.assertRaisesRegex(ValueError, 'supplement changed'):
            runtime.verify(self.root, self.cfg, self.hashes(), self.policy)
        with self.assertRaises(FileNotFoundError):
            runtime.install(self.root, self.hashes(), self.policy)

    def test_numbered_supplement_links_archived_predecessor_without_overwriting_it(self):
        first = self.install()
        first_path = self.root / 'state/runtime_patch.json'
        first_bytes = first_path.read_bytes()
        archived = runtime.archive_current(self.root)
        self.assertEqual(archived, runtime.archive_current(self.root))
        (self.code / 'night_revision.py').write_text('second text recovery protocol')
        second = runtime.install(self.root, self.hashes(), self.policy)
        second_path = self.root / 'state/runtime_patch_000002.json'
        self.assertEqual(first_path.read_bytes(), first_bytes)
        self.assertEqual(second['previous_patch_path'], str(first_path))
        self.assertEqual(second['previous_patch_sha256'], common.file_sha256(first_path))
        self.assertEqual(second['before_files_sha256'], first['effective_files_sha256'])
        result = runtime.verify(self.root, self.cfg, self.hashes(), self.policy)
        self.assertEqual(result['execution_patch_sha256'], common.file_sha256(second_path))
        self.assertEqual(result['execution_patch_history'], [common.file_sha256(first_path), common.file_sha256(second_path)])
        self.assertEqual(result['effective_files_sha256'][str(first_path)], common.file_sha256(first_path))
        self.assertEqual(second, runtime.install(self.root, self.hashes(), self.policy))

    def test_runtime_must_be_archived_before_code_changes(self):
        self.install()
        (self.code / 'night_revision.py').write_text('changed before archive')
        with self.assertRaisesRegex(ValueError, 'dependency changed'):
            runtime.archive_current(self.root)
        with self.assertRaises(FileNotFoundError):
            runtime.install(self.root, self.hashes(), self.policy)

    def test_corrupt_historical_code_archive_invalidates_descendant(self):
        self.install()
        archived = runtime.archive_current(self.root)
        first_path = self.root / 'state/runtime_patch.json'
        archive_dir = self.root / 'state/runtime_patches' / common.file_sha256(first_path)
        (self.code / 'night_revision.py').write_text('second text recovery protocol')
        runtime.install(self.root, self.hashes(), self.policy)
        digest = next(iter(archived['files_sha256'].values()))
        (archive_dir / 'files' / digest).write_text('corrupt history')
        with self.assertRaisesRegex(ValueError, 'Historical runtime archive missing or changed'):
            runtime.verify(self.root, self.cfg, self.hashes(), self.policy)

    def test_missing_numbered_parent_is_rejected(self):
        self.install()
        runtime.archive_current(self.root)
        (self.code / 'night_revision.py').write_text('second text recovery protocol')
        runtime.install(self.root, self.hashes(), self.policy)
        (self.root / 'state/runtime_patch_000002.json').rename(self.root / 'state/runtime_patch_000003.json')
        with self.assertRaisesRegex(ValueError, 'missing parent'):
            runtime.verify(self.root, self.cfg, self.hashes(), self.policy)

    def test_three_repairs_preserve_full_execution_history_and_original_parent(self):
        self.install()
        for number in (2, 3):
            runtime.archive_current(self.root)
            (self.code / 'night_revision.py').write_text(f'text recovery protocol {number}')
            runtime.install(self.root, self.hashes(), self.policy)
        result = runtime.verify(self.root, self.cfg, self.hashes(), self.policy)
        patches = [self.root / 'state/runtime_patch.json',
                   self.root / 'state/runtime_patch_000002.json',
                   self.root / 'state/runtime_patch_000003.json']
        self.assertEqual(result['execution_patch_history'], [common.file_sha256(path) for path in patches])
        self.assertEqual(result['execution_patch_sha256'], common.file_sha256(patches[-1]))
        self.assertEqual((self.root / 'state/revision_config.json').read_bytes(), self.original_config)

    def test_resealed_child_cannot_change_its_previous_execution_link(self):
        self.install()
        runtime.archive_current(self.root)
        (self.code / 'night_revision.py').write_text('second text recovery protocol')
        runtime.install(self.root, self.hashes(), self.policy)
        path = self.root / 'state/runtime_patch_000002.json'
        changed = campaign.config_file(path)
        changed.pop('fingerprint')
        changed['previous_patch_sha256'] = '0' * 64
        path.unlink()
        campaign.frozen(path, changed)
        with self.assertRaisesRegex(ValueError, 'supplement changed'):
            runtime.verify(self.root, self.cfg, self.hashes(), self.policy)

    def test_extra_or_missing_dependency_is_rejected(self):
        self.install()
        extra = self.root / 'unrelated.py'
        extra.write_text('not part of the text runtime')
        changed = self.hashes()
        changed[str(extra)] = common.file_sha256(extra)
        for deps in (changed, self.dependencies):
            with self.assertRaisesRegex(ValueError, 'inventory changed'):
                runtime.verify(self.root, self.cfg, deps, self.policy)

    def test_resealed_parent_change_cannot_reuse_patch(self):
        self.install()
        parent = copy.deepcopy(self.cfg)
        parent.pop('fingerprint')
        parent['generation_barrier'] = 'image generation enabled'
        config_path = self.root / 'state/revision_config.json'
        config_path.unlink()
        resealed = campaign.frozen(config_path, parent)
        with self.assertRaisesRegex(ValueError, 'different revision'):
            runtime.verify(self.root, resealed, self.hashes(), self.policy)

    def test_resealed_patch_cannot_enable_generation_or_add_unknown_fields(self):
        self.install()
        patch_path = self.root / 'state/runtime_patch.json'
        changed = campaign.config_file(patch_path)
        changed.pop('fingerprint')
        changed['image_generation_authorized'] = True
        patch_path.unlink()
        campaign.frozen(patch_path, changed)
        with self.assertRaisesRegex(ValueError, 'supplement changed'):
            runtime.verify(self.root, self.cfg, self.hashes(), self.policy)

    def test_patch_checksum_corruption_is_rejected(self):
        self.install()
        patch_path = self.root / 'state/runtime_patch.json'
        patch_path.write_text(patch_path.read_text().replace('text response', 'fake response'))
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            runtime.verify(self.root, self.cfg, self.hashes(), self.policy)


if __name__ == '__main__':
    unittest.main()
