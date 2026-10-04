"""GSV-Cities place/capture identities; city is part of the place key."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

_PATTERN = re.compile(
    r"^(?P<city>.+)_(?P<place>\d{7})_(?P<year>\d{4})_(?P<month>\d{2})_"
    r"(?P<heading>-?\d+)_(?P<lat>-?\d+(?:\.\d+)?)_"
    r"(?P<lon>-?\d+(?:\.\d+)?)_(?P<pano>.+)$"
)


@dataclass(frozen=True)
class Capture:
    city: str
    place_id: str
    year: int
    month: int
    heading: int
    latitude: float
    longitude: float
    panorama_id: str
    capture_id: str

    @property
    def place_key(self) -> tuple[str, str]:
        return self.city, self.place_id


def parse_filename(path: str | Path) -> Capture:
    stem = Path(path).stem
    match = _PATTERN.fullmatch(stem)
    if not match:
        raise ValueError(f"Not a GSV-Cities capture filename: {path}")
    d = match.groupdict()
    return Capture(d['city'], d['place'], int(d['year']), int(d['month']),
                   int(d['heading']), float(d['lat']), float(d['lon']), d['pano'], stem)


def positive_indices(source: str | Path, database: list[str | Path]) -> list[int]:
    src = parse_filename(source)
    return [i for i, path in enumerate(database)
            if (capture := parse_filename(path)).place_key == src.place_key
            and capture.capture_id != src.capture_id]


def image_index(root: Path) -> dict[str, Path]:
    """Index once, fail on ambiguous captures rather than silently choose a copy."""
    index = {}
    for path in sorted(root.rglob('*')):
        if path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
        capture = parse_filename(path)
        if capture.capture_id in index:
            raise ValueError(f"Duplicate capture: {path} and {index[capture.capture_id]}")
        index[capture.capture_id] = path.resolve()
    if not index:
        raise ValueError(f"No GSV-Cities images found in {root}")
    return index
