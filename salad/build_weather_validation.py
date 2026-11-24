#!/usr/bin/env python3
"""Freeze a natural-weather validation manifest from the official SVOX train split.

SVOX val has only daytime queries, so no earlier stage could see weather
regressions during selection. The train split has night/overcast/rain/snow/sun
queries and its own gallery. Queries closer than ``--buffer`` metres to ANY SVOX
test or val image (gallery or query) are dropped, so selecting on this set does
not select on test locations. The gallery is kept whole; its overlap with test
is only reported. SALAD's pretraining uses GSV-Cities, not SVOX.

One manifest holds every domain so a checkpoint extracts the gallery once;
query IDs keep their ``images/train/queries_<domain>/`` prefix and reports split
by that prefix. Output files are write-once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("night", "rain", "snow", "sun", "overcast")
OUTPUT = ROOT / "outputs/qwen_curriculum/state/weather_validation"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=1, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"Frozen file differs: {path}; use a new --output directory")
    else:
        path.write_text(content)


def domain_of(query_id):
    folder = query_id.split("/")[-2]
    return folder.removeprefix("queries_")


def build(svox_root, output, buffer, radius=25.0):
    import numpy as np
    from scipy.spatial import cKDTree
    from workflow.evaluation import _images_in, _utm, coordinate_positives

    svox_root = Path(svox_root).resolve()
    images = svox_root / "images"
    held_out = []
    for split in ("test", "val"):
        for folder in sorted((images / split).iterdir()):
            if folder.is_dir() and (folder.name == "gallery" or folder.name.startswith("queries")):
                held_out += _images_in(folder, svox_root)
    tree = cKDTree(np.array([_utm(image) for image in held_out]))
    test_tree = cKDTree(np.array([_utm(image) for image in held_out if "/test/" in image.id]))
    references = _images_in(images / "train/gallery", svox_root)
    kept, dropped = [], {}
    for domain in DOMAINS:
        queries = _images_in(images / "train" / f"queries_{domain}", svox_root)
        distance, _ = tree.query(np.array([_utm(q) for q in queries]))
        kept += [q for q, d in zip(queries, distance) if d >= buffer]
        dropped[domain] = int((distance < buffer).sum())
    positives = coordinate_positives(references, kept, radius)
    manifest = {
        "dataset": "svox_train_weather",
        "references": [{"id": r.id, "path": r.id} for r in references],
        "queries": [{"id": q.id, "path": q.id, "positives": [references[i].id for i in p]}
                    for q, p in zip(kept, positives)],
    }
    path = output / "svox_train_weather.json"
    write_once(path, manifest)
    gallery_distance, _ = test_tree.query(np.array([_utm(r) for r in references]))
    counts = {d: sum(domain_of(q.id) == d for q in kept) for d in DOMAINS}
    protocol = {
        "version": 1, "role": "natural_weather_validation", "source": "official SVOX train split",
        "root": str(svox_root), "manifest": str(path), "manifest_sha256": sha(path),
        "positive_rule": f"UTM distance <= {radius:g} m to train gallery",
        "exclusion": f"drop queries within {buffer:g} m of any SVOX test/val gallery or query image",
        "queries_per_domain": counts, "dropped_per_domain": dropped,
        "queries_without_positive": sum(not p for p in positives),
        "gallery_images": len(references),
        "gallery_within_buffer_of_test": int((gallery_distance < buffer).sum()),
        "domains": list(DOMAINS),
        "limitations": [
            "Same city (Oxford) and camera as SVOX test: a validation for this target, not an unseen-domain test.",
            "Domain queries are ~700 each: one query is ~0.15 pp; read paired counts, not small R@1 gaps.",
        ],
    }
    write_once(output / "protocol.json", protocol)
    return protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svox-root", type=Path, default=ROOT / "dataset/svox")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--buffer", type=float, default=50.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.buffer) or args.buffer < 25:
        parser.error("--buffer must be at least the 25 m positive radius")
    protocol = build(args.svox_root, args.output.resolve(), args.buffer)
    print(json.dumps({k: protocol[k] for k in ("queries_per_domain", "dropped_per_domain",
                                               "queries_without_positive", "manifest")}, indent=1))


if __name__ == "__main__":
    main()
