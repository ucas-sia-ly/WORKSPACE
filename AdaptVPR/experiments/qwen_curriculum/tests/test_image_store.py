"""CPU image-store safety, byte preservation, evidence and resume tests."""
import copy
import errno
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.qwen_curriculum import common, image_store


def sealed(body):
    body = {key: value for key, value in body.items() if key != "result_sha256"}
    return {**body, "result_sha256": common.fingerprint(body)}


class ImageStoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.scope = self.base / "outputs"
        self.scope.mkdir()
        self.root = self.scope / "images"
        self.store = image_store.ImageStore(self.root, self.scope)
        self.row = self.make_row("one", "selected", 40)
        self.accepted = sealed({**self.row, "status": "passed", "passed": True, "eligible_for_training": True})
        self.rejected = sealed({**self.row, "status": "rejected", "passed": False, "eligible_for_training": False})

    def make_row(self, sid, directory, color, *, scope=None, metadata=True):
        stage = (scope or self.scope) / directory
        stage.mkdir(parents=True)
        normalized, raw = stage / "normal.png", stage / "raw.png"
        Image.new("RGB", (20, 16), (color, 20, 30)).save(normalized)
        Image.new("RGB", (30, 24), (color, 20, 30)).save(raw)
        if metadata:
            common.write_json(raw.with_suffix(".json"), {"source_path": str(stage / "source_inputs.png"),
                "result_path": str(stage / "service-uuid.png"), "uuid": directory, "seed": 123})
        return sealed({"sample_id": sid, "source_path": str(self.base / "original.jpg"),
            "source_sha256": common.fingerprint("source"), "city": "City", "condition": "night", "place_id": 3,
            "output_path": str(normalized), "output_sha256": common.file_sha256(normalized),
            "raw_output_path": str(raw), "raw_output_sha256": common.file_sha256(raw),
            "execution_fingerprint": "original-frozen-execution", "record_sha256": "original-job-seal"})

    def test_static_layout_does_not_create_any_directory(self):
        self.assertFalse(self.root.exists())
        layout = image_store.layout_definition(self.root)
        self.assertEqual(layout["schema_version"], 2)
        self.assertEqual(layout["directories"]["train"], str(self.root / "train"))
        self.assertEqual(layout["accepted_name"], "condition / sample_id + .png")
        self.assertFalse(self.root.exists())

    def test_scoped_sync_collects_bytes_and_old_paths_atomically_become_links(self):
        before = copy.deepcopy(self.row)
        originals = {key: Path(self.row[key]).read_bytes() for key in ("output_path", "raw_output_path")}
        sidecar = Path(self.row["raw_output_path"]).with_suffix(".json")
        metadata = sidecar.read_bytes()
        report = self.store.sync([self.row], [self.accepted])
        self.assertEqual(self.row, before)
        self.assertEqual((report["train_images"], report["raw_images"], report["accepted_images"]), (1, 1, 1))
        self.assertEqual(report["files_hardlinked"], 3)
        self.assertEqual(report["bytes_copied"], 0)
        for field, folder in (("output_path", "train"), ("raw_output_path", "raw")):
            old, canonical = Path(self.row[field]), self.root / folder / "one.png"
            self.assertTrue(old.is_symlink())
            self.assertFalse(canonical.is_symlink())
            self.assertEqual(old.resolve(), canonical)
            self.assertEqual(canonical.read_bytes(), originals[field])
        self.assertEqual((self.root / "metadata/one.json").read_bytes(), metadata)
        self.assertTrue(sidecar.is_symlink())
        accepted = self.root / "accepted/night/one.png"
        self.assertTrue(accepted.is_symlink())
        self.assertEqual(os.readlink(accepted), "../../train/one.png")
        index = common.read_jsonl(self.root / "metadata.jsonl")[0]
        self.assertEqual(index["accepted_path"], str(accepted))
        self.assertEqual(index["original_result_sha256"], self.accepted["result_sha256"])
        self.assertEqual(index["original_output_path"], self.row["output_path"])

    def test_second_sync_is_idempotent_and_does_not_rehash_image_bytes(self):
        self.store.sync([self.row], [self.accepted])
        before = (self.root / "metadata.jsonl").read_bytes()
        original = common.file_sha256
        hashed_media = []
        def digest(path):
            path = Path(path)
            if path.suffix == ".png" or path.parent.name == "metadata":
                hashed_media.append(path)
            return original(path)
        with patch.object(common, "file_sha256", side_effect=digest):
            report = self.store.sync([copy.deepcopy(self.row)], [copy.deepcopy(self.accepted)])
        self.assertEqual(hashed_media, [])
        self.assertEqual((self.root / "metadata.jsonl").read_bytes(), before)
        self.assertEqual((report["files_created"], report["old_paths_relinked"]), (0, 0))

    def test_unchanged_indices_are_not_rewritten(self):
        extra = self.make_row("one", "evidence", 99)
        self.store.sync([self.row], [self.accepted], extra_evidence_rows=[extra])
        signatures = {name: (self.root / name).stat().st_mtime_ns for name in ("metadata.jsonl", "evidence.jsonl")}
        with patch.object(image_store, "_atomic_rows", side_effect=AssertionError("unchanged index must not be rewritten")):
            self.store.sync([self.row], [self.accepted], extra_evidence_rows=[extra])
        self.assertEqual(signatures, {name: (self.root / name).stat().st_mtime_ns for name in signatures})

    def test_prior_evidence_index_survives_default_sync_and_new_process(self):
        extra = self.make_row("one", "evidence", 99)
        first = self.store.sync([self.row], [self.accepted], extra_evidence_rows=[extra])
        evidence = (self.root / "evidence.jsonl").read_bytes()
        self.store.sync([self.row], [self.accepted])
        self.assertEqual((self.root / "evidence.jsonl").read_bytes(), evidence)
        other = image_store.ImageStore(self.root, self.scope)
        report = other.sync([self.row], [self.accepted])
        self.assertEqual((self.root / "evidence.jsonl").read_bytes(), evidence)
        self.assertEqual(report["evidence_records"], first["evidence_records"])

    def test_index_tampering_is_rejected_before_being_silently_replaced(self):
        self.store.sync([self.row], [self.accepted])
        index = self.root / "evidence.jsonl"
        index.write_text('{"sample_id": "unexpected"}\n')
        with self.assertRaisesRegex(ValueError, "index checksum mismatch"):
            self.store.sync([self.row], [self.accepted])

    def test_crash_after_index_commit_before_summary_resumes_using_commit_marker(self):
        original = image_store._atomic_json
        def fail_summary(path, value):
            if Path(path).name == "summary.json":
                raise OSError("summary commit interrupted")
            return original(path, value)
        with patch.object(image_store, "_atomic_json", side_effect=fail_summary):
            with self.assertRaisesRegex(OSError, "summary commit"):
                self.store.sync([self.row], [self.accepted])
        self.assertTrue((self.root / ".index_commit.json").exists())
        other = image_store.ImageStore(self.root, self.scope)
        self.assertEqual(other.sync([self.row], [self.accepted])["accepted_images"], 1)
        self.assertFalse((self.root / ".index_commit.json").exists())

    def test_quality_change_only_updates_accepted_view_and_keeps_rejected_images(self):
        self.store.sync([self.row], [self.accepted])
        report = self.store.sync([self.row], [self.rejected])
        self.assertFalse((self.root / "accepted/night/one.png").exists())
        self.assertTrue((self.root / "train/one.png").exists())
        self.assertTrue((self.root / "raw/one.png").exists())
        self.assertEqual(report["accepted_links_removed"], 1)
        self.assertEqual(report["rejected_images"], 1)
        self.assertEqual(common.read_jsonl(self.root / "metadata.jsonl")[0]["status"], "rejected")

    def legacy_flat_store(self):
        """Build an authenticated pre-domain store, without a legacy runtime."""
        self.store.sync([self.row], [self.accepted])
        grouped = self.root / "accepted/night/one.png"
        grouped.unlink()
        flat = self.root / "accepted/one.png"
        flat.symlink_to("../train/one.png")
        rows = common.read_jsonl(self.root / "metadata.jsonl")
        rows[0]["accepted_path"] = str(flat)
        common.write_jsonl(self.root / "metadata.jsonl", rows)
        summary = json.loads((self.root / "summary.json").read_text())
        summary["metadata_sha256"] = common.file_sha256(self.root / "metadata.jsonl")
        common.write_json(self.root / "summary.json", summary)
        return flat

    def test_legacy_flat_accepted_migrates_to_domain_and_updates_index_idempotently(self):
        flat = self.legacy_flat_store()
        report = self.store.sync([self.row], [self.accepted])
        grouped = self.root / "accepted/night/one.png"
        self.assertFalse(flat.is_symlink())
        self.assertEqual(os.readlink(grouped), "../../train/one.png")
        self.assertEqual(grouped.resolve(), self.root / "train/one.png")
        self.assertEqual((report["accepted_links_created"], report["accepted_links_removed"]), (1, 1))
        self.assertEqual(common.read_jsonl(self.root / "metadata.jsonl")[0]["accepted_path"], str(grouped))
        restarted = image_store.ImageStore(self.root, self.scope)
        report = restarted.sync([self.row], [self.accepted])
        self.assertEqual((report["accepted_links_created"], report["accepted_links_removed"]), (0, 0))

    def test_legacy_migration_preserves_unknown_flat_files_links_and_directories(self):
        self.legacy_flat_store()
        folder = self.root / "accepted"
        regular = folder / "user.png"
        regular.write_bytes(b"user bytes")
        directory = folder / "other.png"
        directory.mkdir()
        (directory / "notes.txt").write_text("keep")
        alien = folder / "alien.png"
        alien.symlink_to("../train/alien.png")
        foreign = folder / "foreign.png"
        foreign.symlink_to("/unknown/user/path")
        # Even a matching sid/target is unknown without an authenticated index.
        extra = self.make_row("new", "not-yet-indexed", 55)
        not_indexed = folder / "new.png"
        not_indexed.symlink_to("../train/new.png")
        self.store.sync([self.row, extra], [self.accepted])
        self.assertEqual(regular.read_bytes(), b"user bytes")
        self.assertEqual((directory / "notes.txt").read_text(), "keep")
        self.assertEqual(os.readlink(alien), "../train/alien.png")
        self.assertEqual(os.readlink(foreign), "/unknown/user/path")
        self.assertEqual(os.readlink(not_indexed), "../train/new.png")
        restarted = image_store.ImageStore(self.root, self.scope)
        restarted.sync([self.row, extra], [self.accepted])
        self.assertEqual(os.readlink(not_indexed), "../train/new.png")

    def test_legacy_known_flat_link_to_different_target_is_preserved(self):
        flat = self.legacy_flat_store()
        flat.unlink()
        flat.symlink_to("../train/foreign.png")
        report = self.store.sync([self.row], [self.accepted])
        self.assertEqual(os.readlink(flat), "../train/foreign.png")
        self.assertTrue((self.root / "accepted/night/one.png").is_file())
        self.assertEqual(report["accepted_links_removed"], 0)

    def test_interruption_between_new_domain_link_and_flat_removal_resumes(self):
        flat = self.legacy_flat_store()
        before = (self.root / "metadata.jsonl").read_bytes()
        original_unlink = Path.unlink
        def interrupted(path, *args, **kwargs):
            if path == flat:
                raise OSError("legacy alias unlink interrupted")
            return original_unlink(path, *args, **kwargs)
        with patch.object(Path, "unlink", interrupted):
            with self.assertRaisesRegex(OSError, "unlink interrupted"):
                self.store.sync([self.row], [self.accepted])
        self.assertTrue(flat.is_symlink())
        self.assertTrue((self.root / "accepted/night/one.png").is_file())
        self.assertEqual((self.root / "metadata.jsonl").read_bytes(), before)
        restarted = image_store.ImageStore(self.root, self.scope)
        restarted.sync([self.row], [self.accepted])
        self.assertFalse(flat.is_symlink())
        self.assertEqual(common.read_jsonl(self.root / "metadata.jsonl")[0]["accepted_path"],
                         str(self.root / "accepted/night/one.png"))

    def test_rejected_legacy_flat_link_removed_without_deleting_image(self):
        flat = self.legacy_flat_store()
        report = self.store.sync([self.row], [self.rejected])
        self.assertFalse(flat.is_symlink())
        self.assertEqual(report["accepted_links_removed"], 1)
        self.assertEqual(report["accepted_images"], 0)
        self.assertEqual(common.file_sha256(self.root / "train/one.png"), self.row["output_sha256"])

    def test_changed_condition_moves_only_owned_accepted_alias(self):
        self.store.sync([self.row], [self.accepted])
        rain = sealed({**self.row, "condition": "rain"})
        accepted = sealed({**rain, "status": "passed", "passed": True, "eligible_for_training": True})
        self.store.sync([rain], [accepted])
        self.assertFalse((self.root / "accepted/night/one.png").is_symlink())
        self.assertEqual(os.readlink(self.root / "accepted/rain/one.png"), "../../train/one.png")
        self.assertEqual(common.read_jsonl(self.root / "metadata.jsonl")[0]["accepted_path"],
                         str(self.root / "accepted/rain/one.png"))

    def test_accepted_domain_directory_cannot_escape_through_symlink(self):
        self.root.mkdir()
        (self.root / "accepted").mkdir()
        outside = self.base / "outside-accepted"
        outside.mkdir()
        (self.root / "accepted/night").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Accepted image directory"):
            self.store.sync([self.row], [self.accepted])
        self.assertEqual(list(outside.iterdir()), [])

    def test_unsafe_domain_cannot_create_accepted_path_outside_store(self):
        row = sealed({**self.row, "condition": "../../outside"})
        accepted = sealed({**row, "status": "passed", "passed": True, "eligible_for_training": True})
        with self.assertRaisesRegex(ValueError, "Unsafe image-store condition"):
            self.store.sync([row], [accepted])
        self.assertFalse((self.base / "outside").exists())

    def test_scope_outside_and_import_only_never_replace_source_files(self):
        outside = self.make_row("outside", "outside", 60, scope=self.base / "external")
        self.store.sync([outside], [])
        self.assertFalse(Path(outside["output_path"]).is_symlink())
        self.assertFalse(Path(outside["raw_output_path"]).is_symlink())
        import_only = image_store.ImageStore(self.scope / "separate-store")
        import_only.sync([self.row], [])
        self.assertFalse(Path(self.row["output_path"]).is_symlink())

    def test_scope_parent_symlink_cannot_cause_external_source_replacement(self):
        outside = self.make_row("outside", "stage", 60, scope=self.base / "external")
        alias = self.scope / "external-parent"
        alias.symlink_to(self.base / "external", target_is_directory=True)
        row = sealed({**outside, "output_path": str(alias / "stage/normal.png"),
                      "raw_output_path": str(alias / "stage/raw.png")})
        self.store.sync([row], [])
        self.assertFalse(Path(outside["output_path"]).is_symlink())
        self.assertFalse(Path(outside["raw_output_path"]).is_symlink())

    def test_conflicting_effective_sha_refuses_to_overwrite_canonical_bytes(self):
        self.store.sync([self.row], [])
        old = (self.root / "train/one.png").read_bytes()
        changed = self.make_row("one", "conflicting", 99)
        with self.assertRaisesRegex(ValueError, "SHA conflict"):
            self.store.sync([changed], [])
        self.assertEqual((self.root / "train/one.png").read_bytes(), old)
        self.assertFalse(Path(changed["output_path"]).is_symlink())

    def test_extra_different_image_and_sidecar_bytes_are_kept_as_hash_evidence(self):
        extra = self.make_row("one", "older", 99)
        raw_metadata = Path(extra["raw_output_path"]).with_suffix(".json").read_bytes()
        old_row = copy.deepcopy(extra)
        report = self.store.sync([self.row], [self.accepted], extra_evidence_rows=[extra])
        self.assertEqual(report["extra_evidence_images"], 2)
        self.assertEqual(report["extra_evidence_sidecars"], 1)
        self.assertEqual(extra, old_row)
        for field, sha_field in (("output_path", "output_sha256"), ("raw_output_path", "raw_output_sha256")):
            old = Path(extra[field])
            self.assertTrue(old.is_symlink())
            self.assertEqual(old.resolve(), self.root / "evidence" / f"{extra[sha_field]}.png")
        old_metadata = Path(extra["raw_output_path"]).with_suffix(".json")
        self.assertEqual(old_metadata.read_bytes(), raw_metadata)
        self.assertEqual(old_metadata.resolve().parent, self.root / "evidence")
        self.assertEqual(common.file_sha256(self.root / "train/one.png"), self.row["output_sha256"])

    def test_same_sha_extra_images_deduplicate_but_distinct_metadata_uuid_is_preserved(self):
        extra = self.make_row("one", "older-identical", 40)
        selected_metadata = Path(self.row["raw_output_path"]).with_suffix(".json").read_bytes()
        old_metadata = Path(extra["raw_output_path"]).with_suffix(".json").read_bytes()
        report = self.store.sync([self.row], [], extra_evidence_rows=[extra])
        self.assertEqual(report["extra_evidence_images"], 0)
        self.assertEqual(report["extra_evidence_sidecars"], 1)
        self.assertEqual(Path(extra["output_path"]).resolve(), self.root / "train/one.png")
        self.assertEqual(Path(extra["raw_output_path"]).resolve(), self.root / "raw/one.png")
        self.assertEqual((self.root / "metadata/one.json").read_bytes(), selected_metadata)
        self.assertEqual(Path(extra["raw_output_path"]).with_suffix(".json").read_bytes(), old_metadata)

    def test_cross_filesystem_fallback_copies_atomically_without_changing_bytes(self):
        with patch.object(image_store.os, "link", side_effect=OSError(errno.EXDEV, "different device")):
            report = self.store.sync([self.row], [])
        self.assertEqual(report["files_copied"], 3)
        self.assertGreater(report["bytes_copied"], 0)
        self.assertEqual(common.file_sha256(self.root / "train/one.png"), self.row["output_sha256"])
        self.assertTrue(Path(self.row["output_path"]).is_symlink())

    def test_crash_after_canonical_commit_keeps_source_and_can_resume(self):
        original_alias = self.store._alias
        def interrupted(*args):
            raise OSError("interrupted before old symlink replacement")
        with patch.object(self.store, "_alias", side_effect=interrupted):
            with self.assertRaisesRegex(OSError, "interrupted"):
                self.store.sync([self.row], [])
        self.assertTrue((self.root / "train/one.png").exists())
        self.assertFalse(Path(self.row["output_path"]).is_symlink())
        self.store.sync([self.row], [])
        self.assertTrue(Path(self.row["output_path"]).is_symlink())
        self.assertEqual(common.file_sha256(self.root / "train/one.png"), self.row["output_sha256"])

    def test_crash_before_index_commit_never_loses_signed_image_bytes(self):
        original = image_store._atomic_rows
        def fail_index(path, rows):
            if Path(path).name == "metadata.jsonl":
                raise OSError("index commit interrupted")
            return original(path, rows)
        with patch.object(image_store, "_atomic_rows", side_effect=fail_index):
            with self.assertRaisesRegex(OSError, "index commit"):
                self.store.sync([self.row], [self.accepted])
        for field, hash_field in (("output_path", "output_sha256"), ("raw_output_path", "raw_output_sha256")):
            self.assertEqual(common.file_sha256(Path(self.row[field])), self.row[hash_field])
        report = self.store.sync([self.row], [self.rejected])
        self.assertEqual(report["accepted_images"], 0)
        self.assertFalse((self.root / "accepted/night/one.png").exists())

    def test_new_process_can_resume_from_aliases_and_existing_canonical_bytes(self):
        self.store.sync([self.row], [self.accepted])
        other = image_store.ImageStore(self.root, self.scope)
        self.assertEqual(other.sync([self.row], [self.accepted])["files_created"], 0)
        self.assertEqual(other.sync([self.row], [self.rejected])["accepted_links_removed"], 1)

    def test_importing_an_existing_external_symlink_creates_regular_canonical_file(self):
        source = Path(self.row["output_path"])
        actual = source.with_name("physical.png")
        source.replace(actual)
        source.symlink_to(actual)
        self.store.sync([self.row], [])
        self.assertFalse((self.root / "train/one.png").is_symlink())
        self.assertEqual(common.file_sha256(self.root / "train/one.png"), self.row["output_sha256"])

    def test_target_parent_is_fsynced_before_source_alias_commit(self):
        events = []
        original_sync, original_alias = image_store._sync_directory, self.store._alias
        def sync(path):
            events.append(("fsync", Path(path)))
            return original_sync(path)
        def alias(source, target, *rest):
            self.assertIn(("fsync", Path(target).parent), events)
            events.append(("alias", Path(source)))
            return original_alias(source, target, *rest)
        with patch.object(image_store, "_sync_directory", side_effect=sync), patch.object(self.store, "_alias", side_effect=alias):
            self.store.sync([self.row], [])
        self.assertIn(("fsync", Path(self.row["output_path"]).parent), events)

    def test_unknown_files_temps_and_accepted_names_are_untouched(self):
        self.root.mkdir()
        unknown, temporary = self.root / "unknown.png", self.scope / "unfinished.tmp.png"
        unknown.write_bytes(b"unknown bytes")
        temporary.write_bytes(b"in-flight bytes")
        (self.root / "accepted").mkdir()
        alien = self.root / "accepted/alien.png"
        alien.symlink_to("/unknown/user/path")
        self.store.sync([self.row], [self.rejected])
        self.assertEqual(unknown.read_bytes(), b"unknown bytes")
        self.assertEqual(temporary.read_bytes(), b"in-flight bytes")
        self.assertEqual(os.readlink(alien), "/unknown/user/path")

    def test_store_lock_and_same_process_guard_block_concurrent_organizers(self):
        self.root.mkdir()
        with (self.root / ".organize.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "Another organizer"):
                self.store.sync([self.row], [])
        with self.store._guard:
            with self.assertRaisesRegex(RuntimeError, "already running"):
                self.store.sync([self.row], [])

    def test_tampered_rows_or_images_fail_without_creating_untrusted_aliases(self):
        row = dict(self.row)
        row["city"] = "tampered"
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.store.sync([row], [])
        Path(self.row["output_path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            self.store.sync([self.row], [])
        self.assertFalse(Path(self.row["output_path"]).is_symlink())


if __name__ == "__main__":
    unittest.main()
