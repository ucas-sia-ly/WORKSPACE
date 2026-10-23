"""Collect committed Qwen images without changing signed historical records.

``train`` and ``raw`` contain the effective images, ``metadata`` their byte-exact
sidecars, and ``accepted/<condition>`` links only currently accepted training images. Extra
historical bytes differing from the effective image are kept by SHA in
``evidence``. Files outside an explicit scope are imported without modification.
The owner of the generation controller freezes the layout and implementation;
this module never edits a generation plan, execution, journal, or row seal.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import contextmanager
import errno
import fcntl
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import uuid
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import common


DEFAULT_ROOT = common.WORKSPACE_ROOT / "outputs/qwen_curriculum/images"
KINDS = {"output": ("train", "output_path", "output_sha256"),
         "raw": ("raw", "raw_output_path", "raw_output_sha256")}


def layout_definition(root=DEFAULT_ROOT):
    """Return static, fingerprintable layout settings without creating files."""
    root = Path(root).resolve()
    return {"schema_version": 2, "root": str(root),
        "directories": {name: str(root / name) for name in ("train", "raw", "metadata", "accepted", "evidence")},
        "selected_name": "sample_id + .png; sidecar sample_id + .json",
        "accepted_name": "condition / sample_id + .png",
        "evidence_name": "content_sha256 + original media suffix",
        "source_policy": "same-filesystem hardlink or verified atomic copy; atomic scoped source symlink replacement",
        "record_policy": "committed signed image rows only; generation journals and seals unchanged"}


def _sync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path, value):
    common.write_json(path, value)
    _sync_directory(path.parent)


def _atomic_rows(path, rows):
    common.write_jsonl(path, rows)
    _sync_directory(path.parent)


def _signature(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _rows(values):
    return values.values() if isinstance(values, Mapping) else values


class ImageStore:
    def __init__(self, root=DEFAULT_ROOT, scope_root=None):
        self.root = Path(root).resolve()
        self.scope_root = Path(scope_root).resolve() if scope_root is not None else None
        self._guard = threading.Lock()
        self._hash_cache = {}
        self._validated_rows = {}
        self._index_cache = {}

    @contextmanager
    def _lock(self):
        if not self._guard.acquire(blocking=False):
            raise RuntimeError("Image store synchronization is already running in this process")
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with (self.root / ".organize.lock").open("a") as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError("Another organizer owns the image store") from exc
                try:
                    yield
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            self._guard.release()

    def _ensure_layout(self):
        for name in ("train", "raw", "metadata", "accepted", "evidence"):
            directory = self.root / name
            if directory.is_symlink():
                raise ValueError(f"Image store directory cannot be an external symlink: {directory}")
            directory.mkdir(exist_ok=True)
        _sync_directory(self.root)

    def _digest(self, path):
        path = Path(path)
        before = _signature(path)
        if before in self._hash_cache:
            return self._hash_cache[before]
        digest = common.file_sha256(path)
        if _signature(path) != before:
            raise RuntimeError(f"Image changed while hashing: {path}")
        self._hash_cache[before] = digest
        return digest

    def _verify(self, path, expected):
        if not Path(path).is_file() or self._digest(path) != expected:
            raise ValueError(f"Committed image/sidecar missing or changed: {path}")

    def _check_row(self, row):
        digest = row.get("result_sha256")
        if not isinstance(digest, str):
            raise ValueError("Only committed signed image rows can enter the image store")
        # The catalog authenticates full rows. Retain only immutable fields this
        # organizer actually uses; copying every nested matcher/mining metric
        # would consume substantial host RAM over a 79426-image run.
        identity = tuple(row.get(key) for key in ("sample_id", "output_path", "output_sha256", "raw_output_path",
            "raw_output_sha256", "source_path", "source_sha256", "city", "place_id", "condition",
            "execution_fingerprint", "status", "passed", "eligible_for_training", "record_sha256"))
        recovery = row.get("recovery", {})
        identity += (recovery.get("service_metadata_sha256") if isinstance(recovery, dict) else None,)
        previous = self._validated_rows.get(digest)
        if previous is not None:
            if previous != identity:
                raise ValueError("A signed image row changed without a new seal")
        else:
            body = {key: value for key, value in row.items() if key != "result_sha256"}
            if common.fingerprint(body) != digest:
                raise ValueError("Committed image row checksum mismatch")
            self._validated_rows[digest] = identity
        sid = row.get("sample_id")
        if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}", sid) or ".." in sid:
            raise ValueError("Unsafe image-store sample_id")
        for _, path_field, hash_field in KINDS.values():
            if (not isinstance(row.get(path_field), str) or not Path(row[path_field]).is_absolute()
                    or not isinstance(row.get(hash_field), str) or not re.fullmatch(r"[0-9a-f]{64}", row[hash_field])):
                raise ValueError(f"Incomplete committed image identity: {path_field}")
        return sid

    def _load_index(self, name, summary, intent):
        path = self.root / name
        if not path.exists():
            if summary.get("metadata_sha256" if name == "metadata.jsonl" else "evidence_index_sha256"):
                raise ValueError(f"Owned image-store index disappeared: {path}")
            return []
        signature = _signature(path)
        key = "metadata_sha256" if name == "metadata.jsonl" else "evidence_index_sha256"
        expected = {value for value in (summary.get(key), intent.get(key)) if value is not None}
        cached = self._index_cache.get(name)
        actual = cached[2] if cached is not None and cached[0] == signature else self._digest(path)
        if not expected or actual not in expected:
            raise ValueError(f"Owned image-store index checksum mismatch: {path}")
        if cached is not None and cached[0] == signature:
            return cached[1]
        rows = common.read_jsonl(path)
        for row in rows:
            sid = row.get("sample_id")
            if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}", sid) or ".." in sid:
                raise ValueError("Unsafe sample_id in the existing image-store index")
        self._index_cache[name] = (signature, rows, actual)
        return rows

    @staticmethod
    def _rows_digest(rows):
        import hashlib
        digest = hashlib.sha256()
        for row in rows:
            digest.update((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode())
        return digest.hexdigest()

    def _scoped(self, path):
        return (self.scope_root is not None and path.is_relative_to(self.scope_root)
                and path.parent.resolve().is_relative_to(self.scope_root))

    def _import(self, source, target, expected, report):
        """Commit the target before changing any original path."""
        source, target = Path(source), Path(target)
        self._verify(source, expected)
        if target.is_symlink():
            raise ValueError(f"Canonical image must be an owned regular file: {target}")
        if target.exists():
            if self._digest(target) != expected:
                raise ValueError(f"Image-store SHA conflict; existing bytes will not be overwritten: {target}")
            report["files_reused"] += 1
            return
        linked = False
        try:
            # Explicitly resolve the source leaf: some Linux/Python link()
            # combinations otherwise hardlink the symlink inode itself.
            os.link(source.resolve(strict=True), target)
            linked = True
        except FileExistsError:
            if target.is_symlink() or self._digest(target) != expected:
                raise ValueError(f"Image-store SHA conflict: {target}") from None
            report["files_reused"] += 1
            return
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
        temporary = None
        try:
            if linked:
                self._verify(target, expected)
                with target.open("rb") as handle:
                    os.fsync(handle.fileno())
                report["files_hardlinked"] += 1
            else:
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".image-store-copy-", delete=False) as handle:
                    temporary = Path(handle.name)
                    with source.open("rb") as reader:
                        for chunk in iter(lambda: reader.read(1 << 20), b""):
                            handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
                self._verify(temporary, expected)
                # The store lock prevents organizer races. Refuse an unrelated
                # file appearing rather than overwriting it with replace().
                if target.exists() or target.is_symlink():
                    raise ValueError(f"Canonical image appeared while copying: {target}")
                temporary.replace(target)
                report["files_copied"] += 1
                report["bytes_copied"] += target.stat().st_size
            _sync_directory(target.parent)
            report["files_created"] += 1
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _alias(self, original, target, expected, report):
        original, target = Path(original), Path(target)
        if original == target:
            return
        if not self._scoped(original):
            report["outside_scope_paths_preserved"] += 1
            return
        relative = os.path.relpath(target, original.parent)
        if original.is_symlink():
            stored = os.readlink(original)
            if stored in {relative, str(target)}:
                return
            if original.resolve() == target:
                return
        self._verify(original, expected)
        self._verify(target, expected)
        before_original, before_target = _signature(original), _signature(target)
        temporary = original.parent / f".image-store-link-{uuid.uuid4().hex}.tmp"
        created = False
        try:
            os.symlink(relative, temporary)
            created = True
            if _signature(original) != before_original or _signature(target) != before_target:
                raise RuntimeError("Image changed before its atomic storage alias commit")
            temporary.replace(original)
            _sync_directory(original.parent)
            # Removing a hardlink changes ctime, not image bytes. Remember the
            # new inode signature so the next progress tick does not rehash it.
            after = _signature(target)
            if after[:4] != before_target[:4]:
                raise RuntimeError("Canonical image changed while creating its storage alias")
            self._hash_cache[after] = expected
            report["old_paths_relinked"] += 1
        finally:
            if created:
                temporary.unlink(missing_ok=True)

    def _collect(self, source, target, expected, report):
        self._import(source, target, expected, report)
        self._alias(source, target, expected, report)

    def _sidecar(self, row):
        path = Path(row["raw_output_path"]).with_suffix(".json")
        if not path.exists():
            if path.is_symlink():
                raise ValueError(f"Recorded raw sidecar has a broken alias: {path}")
            return None
        if not path.is_file():
            raise ValueError(f"Raw sidecar is not a file: {path}")
        digest = self._digest(path)
        proof = row.get("recovery", {})
        recorded = proof.get("service_metadata_sha256") if isinstance(proof, dict) else None
        if recorded is not None and digest != recorded:
            raise ValueError(f"Frozen recovery sidecar changed: {path}")
        return path, digest

    def _accepted_path(self, sid, condition):
        if not isinstance(condition, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", condition):
            raise ValueError("Unsafe image-store condition")
        return self.root / "accepted" / condition / f"{sid}.png"

    def _owned_accepted_alias(self, path, sid):
        # Never follow an unknown domain directory while removing an alias.
        if path.parent.is_symlink() or not path.parent.is_dir() or not path.is_symlink():
            return False
        target = Path(os.path.abspath(path.parent / os.readlink(path)))
        return target == self.root / "train" / f"{sid}.png"

    def _accepted(self, selected, previous, report):
        accepted = {sid for sid, row in selected.items() if row["accepted"]}
        desired = {sid: self._accepted_path(sid, selected[sid]["condition"]) for sid in accepted}
        # Commit all new domain links before removing any older view. If this
        # process stops here, the authenticated old index still permits replay.
        for sid, path in sorted(desired.items()):
            directory = path.parent
            if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
                raise ValueError(f"Accepted image directory conflicts with an unknown file: {directory}")
            if not directory.exists():
                directory.mkdir()
                _sync_directory(directory.parent)
            if path.is_symlink():
                if not self._owned_accepted_alias(path, sid):
                    raise ValueError(f"Accepted image alias conflicts with an unknown file: {path}")
                continue
            if path.exists():
                raise ValueError(f"Accepted image alias conflicts with an unknown file: {path}")
            expected = os.path.relpath(self.root / "train" / f"{sid}.png", directory)
            temporary = directory / f".image-store-accepted-{uuid.uuid4().hex}.tmp"
            created = False
            try:
                os.symlink(expected, temporary)
                created = True
                temporary.replace(path)
                _sync_directory(directory)
            finally:
                if created:
                    temporary.unlink(missing_ok=True)
            report["accepted_links_created"] += 1
        candidates = set()
        for row in previous:
            sid = row["sample_id"]
            # Only an authenticated prior index establishes ownership of flat
            # legacy aliases. New/unindexed flat names must remain untouched.
            flat = self.root / "accepted" / f"{sid}.png"
            if row.get("accepted") is True and row.get("accepted_path") == str(flat):
                candidates.add((flat, sid))
            try:
                candidates.add((self._accepted_path(sid, row.get("condition")), sid))
            except ValueError:
                pass
        for sid, row in selected.items():
            # A first synchronization can stop before its index commits. Its
            # deterministic domain alias remains recoverable from signed rows.
            candidates.add((self._accepted_path(sid, row["condition"]), sid))
        for path, sid in sorted(candidates):
            if path == desired.get(sid):
                continue
            if self._owned_accepted_alias(path, sid):
                path.unlink()
                _sync_directory(path.parent)
                report["accepted_links_removed"] += 1
            # Unknown names, regular files, directories and differently targeted
            # links stay untouched. The underlying generated image is retained.
        return accepted

    def sync(self, generated_rows, result_rows, *, extra_evidence_rows=()):
        """Synchronize one complete effective catalog snapshot, crash-safely."""
        with self._lock():
            self._ensure_layout()
            report = {"schema_version": 1, "store_root": str(self.root),
                "files_created": 0, "files_reused": 0, "files_hardlinked": 0, "files_copied": 0,
                "bytes_copied": 0, "old_paths_relinked": 0, "outside_scope_paths_preserved": 0,
                "accepted_links_created": 0, "accepted_links_removed": 0}
            selected = {}
            for row in _rows(generated_rows):
                sid = self._check_row(row)
                if sid in selected and selected[sid] != row:
                    raise ValueError(f"Duplicate effective sample_id: {sid}")
                selected[sid] = row
            results = {}
            for row in _rows(result_rows):
                sid = self._check_row(row)
                if sid not in selected:
                    raise ValueError("Quality result has no committed generated image")
                if sid in results and results[sid] != row:
                    raise ValueError(f"Duplicate effective quality result: {sid}")
                if any(row[field] != selected[sid][field] for field in ("output_sha256", "raw_output_sha256")):
                    raise ValueError("Effective quality result changed its generated image hashes")
                if row.get("status") not in {"passed", "rejected"}:
                    raise ValueError("Only completed quality results can select accepted images")
                results[sid] = row
            summary_path, intent_path = self.root / "summary.json", self.root / ".index_commit.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            intent = json.loads(intent_path.read_text()) if intent_path.exists() else {}
            if intent:
                body = {key: value for key, value in intent.items() if key != "fingerprint"}
                if intent.get("fingerprint") != common.fingerprint(body):
                    raise ValueError("Image-store index commit marker checksum mismatch")
            previous = self._load_index("metadata.jsonl", summary, intent)
            previous_evidence = self._load_index("evidence.jsonl", summary, intent)
            entries = {}
            for sid, row in sorted(selected.items()):
                destinations = {}
                for kind, (folder, path_field, hash_field) in KINDS.items():
                    target = self.root / folder / f"{sid}.png"
                    self._collect(Path(row[path_field]), target, row[hash_field], report)
                    destinations[kind] = target
                metadata = self._sidecar(row)
                metadata_target = None
                if metadata is not None:
                    metadata_target = self.root / "metadata" / f"{sid}.json"
                    self._collect(metadata[0], metadata_target, metadata[1], report)
                result = results.get(sid)
                accepted = result is not None and result.get("passed") is True and result.get("eligible_for_training") is True
                original = result or row
                entries[sid] = {"sample_id": sid, "source_path": row.get("source_path"),
                    "source_sha256": row.get("source_sha256"), "city": row.get("city"),
                    "place_id": row.get("place_id"), "condition": row.get("condition"),
                    "status": result["status"] if result is not None else "unreviewed",
                    "passed": result.get("passed") if result is not None else None,
                    "eligible_for_training": result.get("eligible_for_training") if result is not None else None,
                    "accepted": accepted, "output_path": str(destinations["output"]),
                    "output_sha256": row["output_sha256"], "raw_output_path": str(destinations["raw"]),
                    "raw_output_sha256": row["raw_output_sha256"],
                    "metadata_path": str(metadata_target) if metadata_target is not None else None,
                    "metadata_sha256": metadata[1] if metadata is not None else None,
                    "accepted_path": str(self._accepted_path(sid, row["condition"])) if accepted else None,
                    "original_output_path": row["output_path"], "original_raw_output_path": row["raw_output_path"],
                    "generated_result_sha256": row["result_sha256"],
                    "original_result_sha256": original["result_sha256"],
                    "original_execution_fingerprint": original.get("execution_fingerprint")}
            evidence = {(row["original_result_sha256"], row["kind"], row["original_path"]): row
                        for row in previous_evidence}
            for row in _rows(extra_evidence_rows):
                sid = self._check_row(row)
                for kind, (_, path_field, hash_field) in KINDS.items():
                    same_selected = sid in entries and row[hash_field] == entries[sid][hash_field]
                    target = (Path(entries[sid][path_field]) if same_selected
                              else self.root / "evidence" / f"{row[hash_field]}.png")
                    self._collect(Path(row[path_field]), target, row[hash_field], report)
                    key = (row["result_sha256"], kind, row[path_field])
                    evidence[key] = {"sample_id": sid, "kind": kind, "original_path": row[path_field],
                        "canonical_path": str(target), "sha256": row[hash_field],
                        "original_result_sha256": row["result_sha256"], "deduplicated_with_effective": same_selected}
                metadata = self._sidecar(row)
                if metadata is not None:
                    same_selected = sid in entries and metadata[1] == entries[sid]["metadata_sha256"]
                    target = (Path(entries[sid]["metadata_path"]) if same_selected
                              else self.root / "evidence" / f"{metadata[1]}.json")
                    self._collect(metadata[0], target, metadata[1], report)
                    key = (row["result_sha256"], "metadata", str(metadata[0]))
                    evidence[key] = {"sample_id": sid, "kind": "metadata", "original_path": str(metadata[0]),
                        "canonical_path": str(target), "sha256": metadata[1],
                        "original_result_sha256": row["result_sha256"], "deduplicated_with_effective": same_selected}
            accepted = self._accepted(entries, previous, report)
            index_rows = [entries[sid] for sid in sorted(entries)]
            evidence_rows = [evidence[key] for key in sorted(evidence)]
            changed_metadata, changed_evidence = index_rows != previous, evidence_rows != previous_evidence
            metadata_sha = self._rows_digest(index_rows) if changed_metadata or not summary.get("metadata_sha256") else summary["metadata_sha256"]
            evidence_sha = self._rows_digest(evidence_rows) if changed_evidence or not summary.get("evidence_index_sha256") else summary["evidence_index_sha256"]
            if changed_metadata or changed_evidence or not summary:
                commit = {"schema_version": 1, "metadata_sha256": metadata_sha, "evidence_index_sha256": evidence_sha}
                _atomic_json(intent_path, {**commit, "fingerprint": common.fingerprint(commit)})
            for name, rows, changed in (("metadata.jsonl", index_rows, changed_metadata),
                                       ("evidence.jsonl", evidence_rows, changed_evidence)):
                path = self.root / name
                if changed or not path.exists():
                    _atomic_rows(path, rows)
                self._index_cache[name] = (_signature(path), rows, metadata_sha if name == "metadata.jsonl" else evidence_sha)
            evidence_paths = {row["canonical_path"] for row in evidence_rows
                              if Path(row["canonical_path"]).parent == self.root / "evidence"}
            report.update(generated_images=len(entries), train_images=len(entries), raw_images=len(entries),
                sidecars=sum(row["metadata_path"] is not None for row in index_rows), accepted_images=len(accepted),
                rejected_images=sum(row["status"] == "rejected" for row in index_rows),
                unreviewed_images=sum(row["status"] == "unreviewed" for row in index_rows),
                extra_evidence_images=sum(Path(path).suffix == ".png" for path in evidence_paths),
                extra_evidence_sidecars=sum(Path(path).suffix == ".json" for path in evidence_paths),
                evidence_records=len(evidence_rows), metadata_sha256=metadata_sha, evidence_index_sha256=evidence_sha)
            _atomic_json(self.root / "summary.json", report)
            if intent_path.exists():
                intent_path.unlink()
                _sync_directory(self.root)
            return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("sync", "layout"))
    parser.add_argument("--store-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--generation-dir", type=Path)
    parser.add_argument("--scope-root", type=Path, default=None,
                        help="Only original image paths under this root become links; omit for import only")
    parser.add_argument("--extra-evidence-manifest", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    if args.command == "layout":
        print(json.dumps(layout_definition(args.store_root), ensure_ascii=False, indent=2))
        return
    if args.generation_dir is None:
        parser.error("sync requires --generation-dir containing committed generated/results.jsonl")
    generated = common.read_jsonl(args.generation_dir / "generated.jsonl")
    result_path = args.generation_dir / "results.jsonl"
    results = common.read_jsonl(result_path) if result_path.exists() else []
    evidence = [row for path in args.extra_evidence_manifest for row in common.read_jsonl(path)]
    report = ImageStore(args.store_root, args.scope_root).sync(generated, results, extra_evidence_rows=evidence)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
