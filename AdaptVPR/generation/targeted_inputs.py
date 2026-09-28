"""Model-free reader for Bag-of-Queries Stage3 schema 1 target artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any

from PIL import Image

TARGET_ROLES = {"attention": "primary", "fused": "supplementary"}
SCHEMA_VERSION = 1
TASK_SCHEMA_VERSION = 1
REQUIRED_FIELDS = (
    "schema_version", "sample_id", "image_key", "place_key", "source_path",
    "mask_original_path", "target_type", "mask_ratio", "clean_margin",
    "mask_mode", "mask_token_count", "checkpoint_path", "checkpoint_sha256",
    "stage2_commit", "seed",
)
TASK_FIELDS = {
    "sample_id", "image_key", "place_key", "source_path", "mask_original_path",
    "target_type", "mask_ratio", "clean_margin", "source_width", "source_height", "target_role",
}


class TargetedInputError(ValueError):
    """An invalid artifact or incompatible resume; no tasks should be written."""


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise TargetedInputError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON number: {value}")


def _decode(text: str, context: str):
    try:
        return json.loads(text, object_pairs_hook=_object,
                          parse_constant=_reject_constant)
    except ValueError as exc:
        raise TargetedInputError(f"{context}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            context = f"{path}:{line_number}"
            if not line.strip():
                raise TargetedInputError(f"{context}: blank JSONL record")
            row = _decode(line, context)
            if not isinstance(row, dict):
                raise TargetedInputError(f"{context}: expected a JSON object")
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise TargetedInputError(f"{context}: sample_id must be a nonempty string")
            if sample_id in seen:
                raise TargetedInputError(f"{context}: duplicate sample_id: {sample_id}")
            seen.add(sample_id)
            rows.append(row)
    if not rows:
        raise TargetedInputError(f"{path}: empty manifest")
    return rows


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise TargetedInputError(f"{label} must be an integer >= {minimum}")


def _validate_record(row: dict) -> None:
    label = f"sample {row['sample_id']}"
    missing = [key for key in REQUIRED_FIELDS if key not in row or row[key] is None]
    if missing:
        raise TargetedInputError(f"{label}: missing required fields: {', '.join(missing)}")
    if type(row["schema_version"]) is not int or row["schema_version"] != SCHEMA_VERSION:
        raise TargetedInputError(f"{label}: unsupported manifest schema_version: {row['schema_version']!r}")
    for key in ("sample_id", "image_key", "place_key", "source_path", "mask_original_path", "mask_mode",
                "checkpoint_path", "checkpoint_sha256", "stage2_commit"):
        if not isinstance(row[key], str) or not row[key].strip():
            raise TargetedInputError(f"{label}: {key} must be a nonempty string")
    target_type = row["target_type"]
    if not isinstance(target_type, str) or target_type not in TARGET_ROLES:
        raise TargetedInputError(f"{label}: invalid target_type: {target_type!r}")
    if "target_role" in row and row["target_role"] != TARGET_ROLES[target_type]:
        raise TargetedInputError(f"{label}: target_role disagrees with target_type")
    if "source_role" in row and row["source_role"] != "SOURCE":
        raise TargetedInputError(f"{label}: source_role must be SOURCE")
    key = PurePosixPath(row["image_key"])
    if key.is_absolute() or ".." in key.parts or "\\" in row["image_key"] or key.as_posix() != row["image_key"]:
        raise TargetedInputError(f"{label}: image_key must be a normalized relative path")
    for field in ("mask_ratio", "clean_margin"):
        if type(row[field]) not in (int, float) or not math.isfinite(row[field]):
            raise TargetedInputError(f"{label}: {field} must be a finite number")
    if not 0 < row["mask_ratio"] <= 1:
        raise TargetedInputError(f"{label}: mask_ratio must be in (0, 1]")
    _integer(row["seed"], f"{label}: Stage2 seed")
    _integer(row["mask_token_count"], f"{label}: mask_token_count", 1)
    for field in ("source_width", "source_height"):
        if field in row:
            _integer(row[field], f"{label}: {field}", 1)


def _resolve_file(value: str, manifest: Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest.parent / path
    path = Path(os.path.abspath(path))
    if not path.is_file():
        raise TargetedInputError(f"{label} file does not exist: {path}")
    return path


def _verify_hash(row: dict, field: str, actual: str):
    if field in row and row[field] != actual:
        raise TargetedInputError(f"sample {row['sample_id']}: {field} mismatch")


def normalize_target(row: dict, manifest: Path, manifest_sha256: str, *, seed: int) -> dict:
    """Decode source and unmodified mask; pin their bytes, preserve Stage2 metadata.

    Schema 1 uses single-channel PNG values {0,1}. No thresholding, resizing,
    EXIF transpose, prompt construction or position inference takes place.
    """
    _validate_record(row)
    source = _resolve_file(row["source_path"], manifest, "source")
    mask = _resolve_file(row["mask_original_path"], manifest, "mask")
    image_parts = PurePosixPath(row["image_key"]).parts
    if source.parts[-len(image_parts):] != image_parts:
        raise TargetedInputError(f"sample {row['sample_id']}: source_path does not match image_key")
    try:
        with Image.open(source) as image:
            image.load()
            width, height = image.size
        with Image.open(mask) as image:
            image.load()
            if image.format != "PNG" or image.mode not in ("1", "L"):
                raise TargetedInputError("mask must be a single-channel binary PNG")
            if image.size != (width, height):
                raise TargetedInputError("mask size must equal source image size")
            histogram = image.histogram()
            # Pillow exposes true pixels in mode 1 at bin 255 (logical 1).
            selected = 255 if image.mode == "1" else 1
            if sum(histogram) != histogram[0] + histogram[selected]:
                raise TargetedInputError("mask must be binary with values {0,1}; no thresholding allowed")
            area = histogram[selected]
            if area == 0:
                raise TargetedInputError("mask must be nonempty")
    except (OSError, ValueError) as exc:
        raise TargetedInputError(f"sample {row['sample_id']}: {exc}") from exc
    for field, actual in (("source_width", width), ("source_height", height), ("actual_pixel_area_original", area)):
        if field in row and row[field] != actual:
            raise TargetedInputError(f"sample {row['sample_id']}: {field} mismatch")
    source_hash, mask_hash = _sha256(source), _sha256(mask)
    _verify_hash(row, "source_sha256", source_hash)
    _verify_hash(row, "mask_original_sha256", mask_hash)
    return dict(
        task_schema_version=TASK_SCHEMA_VERSION, manifest_schema_version=row["schema_version"],
        sample_id=row["sample_id"], image_key=row["image_key"], place_key=row["place_key"],
        source_path=str(source), source_width=width, source_height=height,
        source_sha256=source_hash, mask_original_path=str(mask), mask_original_sha256=mask_hash,
        mask_encoding="binary_0_1", actual_pixel_area_original=area,
        target_type=row["target_type"], target_role=TARGET_ROLES[row["target_type"]],
        mask_ratio=row["mask_ratio"], clean_margin=row["clean_margin"],
        stage2_metadata={key: value for key, value in row.items() if key not in TASK_FIELDS},
        input_manifest=str(manifest), input_manifest_sha256=manifest_sha256,
        mode="targeted", route="local", targeting="provided_mask", status="validated",
        dry_run=True, generated=False, seed=seed,
    )


def read_targeted_tasks(manifest: str | Path, *, limit: int = 0, seed: int = 0) -> list[dict]:
    """Validate every row's schema/unique ID, then decode the selected prefix.

    A sibling export_manifest.json, when present, must match schema, count and
    targets SHA256. Standalone targets.jsonl remains supported.
    """
    _integer(limit, "limit")
    _integer(seed, "seed")
    manifest = Path(manifest).resolve()
    if not manifest.is_file():
        raise TargetedInputError(f"targets manifest does not exist: {manifest}")
    digest = _sha256(manifest)
    rows = _read_jsonl(manifest)
    for row in rows:
        _validate_record(row)
    companion = manifest.parent / "export_manifest.json"
    if companion.exists():
        export = _decode(companion.read_text(encoding="utf-8"), str(companion))
        if not isinstance(export, dict) or type(export.get("schema_version")) is not int or export["schema_version"] != SCHEMA_VERSION:
            raise TargetedInputError("unsupported export manifest schema_version")
        if export.get("targets_sha256") != digest or export.get("record_count") != len(rows):
            raise TargetedInputError("export manifest targets hash/count mismatch")
    tasks = [normalize_target(row, manifest, digest, seed=seed) for row in (rows[:limit] if limit else rows)]
    if _sha256(manifest) != digest:
        raise TargetedInputError("targets manifest changed while reading")
    return tasks


def run_targeted(manifest: str | Path, output: str | Path, *, check_only: bool = False,
                 limit: int = 0, seed: int = 0, resume: bool = False) -> dict:
    """Validate only or atomically write dry-run tasks. Never generate images.

    Resume requires an exact, fully validated prefix of the current selection;
    it supports increasing --limit, but rejects changed inputs, seed or tasks.
    """
    tasks = read_targeted_tasks(manifest, limit=limit, seed=seed)
    output = Path(output).resolve()
    destination = output / "tasks.jsonl"
    protected = {Path(manifest).resolve(), Path(manifest).resolve().parent / "export_manifest.json"}
    protected.update(Path(t[k]).resolve() for t in tasks for k in ("source_path", "mask_original_path"))
    if destination in protected:
        raise TargetedInputError("output tasks.jsonl would overwrite an input artifact")
    existing = []
    if destination.exists() and resume:
        existing = _read_jsonl(destination)
        if len(existing) > len(tasks) or any(_json_text(a) != _json_text(b) for a, b in zip(existing, tasks)):
            raise TargetedInputError("resume tasks are not an identical prefix; input, seed or task content changed")
    elif destination.exists() and not check_only:
        raise TargetedInputError("tasks.jsonl already exists; use --resume or a different --output")
    result = dict(check_only=check_only, validated=len(tasks), resumed=len(existing),
                  written=0, route="local", generated=0, output=str(destination))
    if check_only or len(existing) == len(tasks):
        return result
    output.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output,
                                         prefix=".targeted-", suffix=".jsonl", delete=False) as stream:
            temporary = Path(stream.name)
            for task in tasks:
                stream.write(_json_text(task))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    result["written"] = len(tasks) - len(existing)
    return result
