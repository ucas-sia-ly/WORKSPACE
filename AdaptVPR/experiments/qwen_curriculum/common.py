"""Paths and atomic records for staged Qwen-only data construction."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

CURRICULUM_ROOT = Path(__file__).resolve().parent
GUIDANCE_ROOT = CURRICULUM_ROOT
ADAPTVPR_ROOT = GUIDANCE_ROOT.parents[1]
WORKSPACE_ROOT = ADAPTVPR_ROOT.parent
SALAD_ROOT = WORKSPACE_ROOT / "salad"


def _prepend(path: Path) -> None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def use_adaptvpr() -> None:
    """Make AdaptVPR packages importable and load its ``.env`` like the services do."""
    _prepend(ADAPTVPR_ROOT)
    _prepend(GUIDANCE_ROOT)
    env_file = ADAPTVPR_ROOT / ".env"
    if env_file.is_file():
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
    # verification/evaluator.py resolves VISMATCH_ROOT at import time relative to
    # the working directory; the shipped .env value is relative to AdaptVPR/.
    vismatch = Path(os.getenv("VISMATCH_ROOT", "../vismatch"))
    if not vismatch.is_absolute():
        os.environ["VISMATCH_ROOT"] = str((ADAPTVPR_ROOT / vismatch).resolve())


def use_salad() -> None:
    _prepend(SALAD_ROOT)
    _prepend(GUIDANCE_ROOT)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                rows.append(row)
    return rows


def _atomic_write(path: Path, chunks: Iterable[str]) -> None:
    """Replace a complete file, leaving the previous version intact on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for chunk in chunks:
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_write(path, (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))


def write_json(path: Path, payload: Any) -> None:
    _atomic_write(path, [json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def candidate_seed(base_seed: int, sample_id: str, index: int) -> int:
    """Deterministic per-candidate seed, in the same range AdaptVPR's agent uses."""
    material = f"{base_seed}|{sample_id}|{index}".encode("utf-8")
    return 1 + int(hashlib.sha256(material).hexdigest()[:8], 16) % 2_000_000_000
