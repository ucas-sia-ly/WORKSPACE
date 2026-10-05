"""Shared data helpers for the domain-only VPR generation pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def read_global_prompts(path: Path, conditions: Iterable[str] = (), limit: int = 0) -> list[dict]:
    wanted = {c.lower() for c in conditions}
    rows, seen = [], set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("route", "")).lower() != "global":
                continue
            condition = str(row.get("condition", "")).lower()
            if wanted and condition not in wanted:
                continue
            for key in ("sample_id", "source_id", "prompt", "condition"):
                if not isinstance(row.get(key), str) or not row[key]:
                    raise ValueError(f"{path}:{line_no}: missing non-empty string field {key}")
            if row["sample_id"] in seen:
                raise ValueError(f"duplicate sample_id: {row['sample_id']}")
            seen.add(row["sample_id"])
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise ValueError("no Global-route prompts matched the requested filters")
    return rows


def image_index(image_root: Path) -> dict[str, Path]:
    root = Path(image_root)
    if not root.exists():
        raise FileNotFoundError(root)
    out = {}
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        for path in root.rglob(pattern):
            if path.name in out and out[path.name] != path:
                raise ValueError(f"duplicate image basename under {root}: {path.name}")
            out[path.name] = path.resolve()
    if not out:
        raise ValueError(f"no images found under {root}")
    return out


def resolve_source(row: dict, images: dict[str, Path]) -> Path:
    name = Path(row["source_id"]).name
    if name not in images:
        raise FileNotFoundError(f"source_id not found under image root: {name}")
    return images[name]


def place_key_from_name(name: str) -> tuple[str, str]:
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) < 3 or not parts[1].isdigit():
        raise ValueError(f"not a GSV-Cities filename: {name}")
    return parts[0], parts[1]
