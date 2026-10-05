"""Shared data helpers for the domain-only VPR generation pipeline."""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable


def read_global_prompts(path: Path, conditions: Iterable[str] = (), limit: int = 0) -> list[dict]:
    if limit < 0:
        raise ValueError("limit must be non-negative")
    wanted = {c.strip().lower() for c in conditions}
    rows, seen = [], set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("route", "")).strip().lower() != "global":
                continue
            condition = str(row.get("condition", "")).strip().lower()
            if wanted and condition not in wanted:
                continue
            for key in ("sample_id", "source_id", "prompt", "condition"):
                if not isinstance(row.get(key), str) or not row[key]:
                    raise ValueError(f"{path}:{line_no}: missing non-empty string field {key}")
            validate_sample_id(row["sample_id"])
            row = dict(row, route="global", condition=condition)
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
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
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
    return parts[0], canonical_place_id(parts[1])


def validate_sample_id(value: str) -> str:
    """Sample IDs are output basenames, never paths."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"unsafe sample_id: {value!r}")
    return value


def canonical_place_id(value) -> str:
    text = str(value)
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError(f"place_id must be a non-negative integer: {value!r}")
    return str(int(text)).zfill(7)


def gsv_image_name(row, place_id=None) -> str:
    """Use the GSV dataframe's fields, including the complete panorama ID."""
    pid = canonical_place_id(row["place_id"] if place_id is None else place_id)
    return (
        f"{row['city_id']}_{pid}_{str(row['year']).zfill(4)}_"
        f"{str(row['month']).zfill(2)}_{str(row['northdeg']).zfill(3)}_"
        f"{row['lat']}_{row['lon']}_{row['panoid']}.jpg"
    )


class GSVLabelIndex:
    """Resolve a source identity against GSV Dataframes before inheriting a label.

    Latitude/longitude strings can differ in the final decimal after pandas CSV
    float parsing. The stable identity includes city, place, capture date, view
    and the *whole* panorama ID; coordinates must also agree to 1e-10 degrees.
    Labels returned here come from the matching dataframe row.
    """

    def __init__(self, dataframe_dir: Path, cities=None):
        self.root = Path(dataframe_dir).resolve()
        paths = ([self.root / f"{city}.csv" for city in cities] if cities is not None
                 else sorted(self.root.glob("*.csv")))
        if not paths:
            raise FileNotFoundError(f"no GSV Dataframes found under {self.root}")
        self.cities = set()
        self.identities = {}
        required = {"city_id", "place_id", "year", "month", "northdeg", "lat", "lon", "panoid"}
        for path in paths:
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if not required.issubset(reader.fieldnames or ()):
                    raise ValueError(f"{path}: missing GSV dataframe columns")
                for row in reader:
                    city = row["city_id"]
                    if city != path.stem:
                        raise ValueError(f"{path}: city_id {city!r} disagrees with dataframe city")
                    pid = canonical_place_id(row["place_id"])
                    identity = (city, pid, int(row["year"]), int(row["month"]),
                                int(row["northdeg"]), row["panoid"])
                    coords = (float(row["lat"]), float(row["lon"]))
                    if not all(math.isfinite(x) for x in coords):
                        raise ValueError(f"{path}: non-finite source coordinates")
                    self.identities.setdefault(identity, []).append(coords)
                    self.cities.add(city)

    def key_for_source(self, path: Path) -> tuple[str, str]:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        # Match a known dataframe city rather than assuming cities contain no '_'.
        cities = [city for city in self.cities if path.stem.startswith(city + "_")]
        matches = []
        for city in cities:
            parts = path.stem[len(city) + 1:].split("_", 6)
            if len(parts) != 7:
                continue
            try:
                pid = canonical_place_id(parts[0])
                identity = (city, pid, int(parts[1]), int(parts[2]), int(parts[3]), parts[6])
                coords = (float(parts[4]), float(parts[5]))
            except ValueError:
                continue
            for known in self.identities.get(identity, ()):
                if all(math.isclose(a, b, rel_tol=0, abs_tol=1e-10) for a, b in zip(coords, known)):
                    matches.append((city, pid))
                    break
        if len(matches) != 1:
            raise ValueError(f"source image does not identify exactly one GSV dataframe row: {path}")
        city, pid = matches[0]
        if path.parent.name != city:
            raise ValueError(f"source directory {path.parent.name!r} disagrees with GSV city {city!r}")
        return city, pid


def require_empty_output(path: Path):
    """Never destroy a previous experiment implicitly."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty; use a new directory: {path}")


def validate_source_label(row: dict, source_path: Path, labels: GSVLabelIndex):
    city, pid = labels.key_for_source(source_path)
    if "city" in row and str(row["city"]) != city:
        raise ValueError(f"source city disagrees with manifest: {source_path}")
    if "place_id" in row and canonical_place_id(row["place_id"]) != pid:
        raise ValueError(f"source place_id disagrees with manifest: {source_path}")
    return city, pid
