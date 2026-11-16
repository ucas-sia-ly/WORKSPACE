from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.qwen_curriculum import night_cleanup as cleanup


class CleanupTests(unittest.TestCase):
    def fixture(self, root, actions, protected=()):
        log = root/'outputs/maintenance/cleanup'; log.mkdir(parents=True)
        cleanup.dataset.atomic_json(log/'plan.json', cleanup.dataset.sealed(dict(
            workspace=str(root), dataset_manifest_sha256='verified', protected=list(map(str, protected)),
            actions=actions, policy='test')))
        stack = ExitStack()
        stack.enter_context(patch.object(cleanup, 'WORKSPACE', root))
        stack.enter_context(patch.object(cleanup, 'LOG', log))
        stack.enter_context(patch.object(cleanup.dataset, 'status', return_value=dict(manifest_sha256='verified', count=61984)))
        return stack

    def test_only_planned_unreferenced_file_is_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); old = root/'outputs/old/image.png'; old.parent.mkdir(parents=True)
            old.write_bytes(b'obsolete'); final = root/'outputs/final.pt'; final.write_bytes(b'keep')
            with self.fixture(root, [cleanup.entry(old, 'retired')], [final]):
                result = cleanup.apply()
                self.assertEqual(result['deleted_files'], 1)
                self.assertFalse(old.exists()); self.assertEqual(final.read_bytes(), b'keep')

    def test_changed_file_blocks_entire_plan_before_first_unlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); a = root/'outputs/a'; a.parent.mkdir(); a.write_bytes(b'first')
            b = root/'outputs/b'; b.write_bytes(b'original'); actions = [cleanup.entry(p, 'old') for p in (a, b)]
            b.write_bytes(b'changed length')
            with self.fixture(root, actions):
                with self.assertRaisesRegex(ValueError, 'changed'):
                    cleanup.apply()
                self.assertTrue(a.exists())

    def test_dataset_and_protected_paths_cannot_be_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative, protected in [('dataset/real.jpg', False), ('outputs/checkpoint.pt', True)]:
                file = root/relative; file.parent.mkdir(parents=True, exist_ok=True); file.write_bytes(b'keep')
                log = root/'outputs/maintenance/cleanup'
                if log.exists():
                    import shutil
                    shutil.rmtree(log)
                with self.fixture(root, [cleanup.entry(file, 'bad')], [file] if protected else []):
                    with self.assertRaisesRegex(ValueError, 'escapes'):
                        cleanup.apply()
                    self.assertTrue(file.exists())

    def test_symlink_parent_is_never_traversed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); actual = root/'dataset'; actual.mkdir(); image = actual/'image'; image.write_bytes(b'keep')
            output = root/'outputs'; output.mkdir(); (output/'alias').symlink_to(actual, target_is_directory=True)
            path = output/'alias/image'
            with self.fixture(root, [cleanup.entry(path, 'bad')]):
                with self.assertRaisesRegex(ValueError, 'symbolic link'):
                    cleanup.apply()
                self.assertTrue(image.exists())


if __name__ == '__main__':
    unittest.main()
