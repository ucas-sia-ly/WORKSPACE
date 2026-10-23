"""Durable weather journals and fixed domain quotas for the active campaign."""

from __future__ import annotations

import json
import os


def read_rows(path):
    """Read JSONL; a torn final line from an interrupted append is dropped."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if number != len(lines):
                raise ValueError(f"{path}:{number}: malformed JSON before the final line")
    return rows


def append_row(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def quotas(total, weights):
    raw = {k: total * w for k, w in weights.items()}
    result = {k: int(v) for k, v in raw.items()}
    for key in sorted(raw, key=lambda k: (result[k] - raw[k], k))[:total - sum(result.values())]:
        result[key] += 1
    return result
