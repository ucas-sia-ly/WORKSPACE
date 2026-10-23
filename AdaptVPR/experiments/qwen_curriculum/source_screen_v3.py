"""Cheap, resumable source screening for the place-coverage weather campaign.

This checks exposure, decoding and visible texture on the CPU.  It is not a
semantic detector of buildings, lamps or weather and cannot guarantee that an
edit will pass the separate generated-image verifier.  Each eligible place is
visited once; its views have a stable random order and the first usable,
globally content-distinct view is retained.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import common


POLICY_VERSION = "place_source_cpu_v1"
THRESHOLDS = {
    "min_width": 160, "min_height": 120, "analysis_long_edge": 256,
    "min_luminance": 12.0, "max_luminance": 245.0,
    "dark_pixel": 8, "bright_pixel": 247, "max_extreme_fraction": 0.85,
    "min_p95_minus_p05": 22.0, "min_luminance_std": 10.0,
    "min_laplacian_variance": 8.0, "edge_gradient": 12.0,
    "min_edge_fraction": 0.004,
    "tile_gradient": 3.0, "min_textured_tile_fraction": 0.10,
}


def screen_image(path, *, thresholds=None):
    """Return a JSON-safe verdict, image hash/dimensions and CPU measurements."""
    path = Path(path)
    limits = THRESHOLDS if thresholds is None else thresholds
    verdict = {"source_path": str(path), "passed": False, "reasons": [], "metrics": {}}
    try:
        content = path.read_bytes()
        verdict["source_sha256"] = hashlib.sha256(content).hexdigest()
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            width, height = image.size
            verdict["source_dimensions"] = [width, height]
            image = image.convert("L")
            image.thumbnail((limits["analysis_long_edge"], limits["analysis_long_edge"]),
                            Image.Resampling.LANCZOS)
            gray = np.asarray(image, dtype=np.uint8)
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        verdict["reasons"] = ["source_missing_or_decode_failed"]
        verdict["error_type"] = type(exc).__name__
        return verdict
    if width < limits["min_width"] or height < limits["min_height"]:
        verdict["reasons"].append("source_resolution_too_small")
    if min(gray.shape) < 3:
        verdict["reasons"].append("source_canvas_too_small_for_measurements")
        return verdict

    # The central 80% limits the influence of JPEG borders and large sky strips.
    h, w = gray.shape
    central = gray[h // 10:h - h // 10, w // 10:w - w // 10]
    grad_x = cv2.Sobel(central, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    grad_y = cv2.Sobel(central, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    gradient = np.hypot(grad_x, grad_y)
    tiles = [float(tile.mean()) for row in np.array_split(gradient, 6, axis=0)
             for tile in np.array_split(row, 8, axis=1) if tile.size]
    p05, p95 = np.percentile(gray, [5, 95])
    metrics = {
        "luminance_mean": float(gray.mean()), "luminance_std": float(gray.std()),
        "p95_minus_p05": float(p95 - p05),
        "dark_fraction": float((gray <= limits["dark_pixel"]).mean()),
        "bright_fraction": float((gray >= limits["bright_pixel"]).mean()),
        "laplacian_variance": float(cv2.Laplacian(central, cv2.CV_32F).var()),
        "edge_fraction": float((gradient >= limits["edge_gradient"]).mean()),
        "textured_tile_fraction": float(sum(t >= limits["tile_gradient"] for t in tiles) / len(tiles)),
    }
    verdict["metrics"] = metrics
    reasons = verdict["reasons"]
    if (metrics["luminance_mean"] < limits["min_luminance"]
            or metrics["dark_fraction"] > limits["max_extreme_fraction"]):
        reasons.append("source_extremely_dark")
    if (metrics["luminance_mean"] > limits["max_luminance"]
            or metrics["bright_fraction"] > limits["max_extreme_fraction"]):
        reasons.append("source_extremely_bright")
    if (metrics["p95_minus_p05"] < limits["min_p95_minus_p05"]
            or metrics["luminance_std"] < limits["min_luminance_std"]):
        reasons.append("source_nearly_blank_or_flat")
    if metrics["laplacian_variance"] < limits["min_laplacian_variance"]:
        reasons.append("source_extremely_blurred_or_textureless")
    if (metrics["edge_fraction"] < limits["min_edge_fraction"]
            or metrics["textured_tile_fraction"] < limits["min_textured_tile_fraction"]):
        reasons.append("source_structural_texture_too_sparse")
    verdict["passed"] = not reasons
    return verdict


def eligible_places(real_data, min_images=4):
    """Read exact CSV identities, deduplicate views and retain missing paths to audit."""
    root = Path(real_data).expanduser().resolve()
    metadata = sorted((root / "Dataframes").glob("*.csv"))
    if not metadata:
        raise ValueError(f"No GSV city CSV files in {root / 'Dataframes'}")
    places = defaultdict(set)
    for csv_path in metadata:
        city = csv_path.stem
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            for line, row in enumerate(csv.DictReader(handle), 2):
                if row["city_id"].strip() != city:
                    raise ValueError(f"{csv_path}:{line}: city_id differs from the CSV city")
                place_id = int(row["place_id"])
                filename = (f"{city}_{place_id:07d}_{int(row['year']):04d}_{int(row['month']):02d}_"
                            f"{int(row['northdeg']):03d}_{row['lat']}_{row['lon']}_{row['panoid']}.jpg")
                if Path(filename).name != filename or place_id < 0:
                    raise ValueError(f"{csv_path}:{line}: invalid image identity")
                path = root / "Images" / city / filename
                places[city, place_id].add(path)
    return {key: sorted(paths) for key, paths in sorted(places.items()) if len(paths) >= min_images}


def ordered_views(paths, city, place_id, seed):
    return sorted(paths, key=lambda p: hashlib.sha256(
        json.dumps(["source_order_v1", seed, city, place_id, str(p)], separators=(",", ":")).encode()).digest())


def _worker_init():
    cv2.setNumThreads(1)


def _screen_place(task):
    city, place_id, paths, thresholds = task
    attempts = []
    for path in paths:
        verdict = screen_image(path, thresholds=thresholds)
        attempts.append(verdict)
        if verdict["passed"]:
            return {"city": city, "place_id": place_id, "attempts": attempts,
                    "selected": {**verdict, "city": city, "place_id": place_id}}
    return {"city": city, "place_id": place_id, "attempts": attempts, "selected": None}


@contextmanager
def _screen_lock(directory):
    with (directory / ".screen.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Source screening is already running in this output directory") from exc
        yield


def _connect(directory):
    db = sqlite3.connect(directory / "screening.sqlite3")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS places (city TEXT NOT NULL, place_id INTEGER NOT NULL, "
               "outcome TEXT NOT NULL, source_sha256 TEXT UNIQUE, PRIMARY KEY (city, place_id))")
    db.commit()
    return db


def _outcomes(db):
    return [json.loads(value) for (value,) in db.execute("SELECT outcome FROM places ORDER BY city, place_id")]


def _summary(db, total):
    rows = _outcomes(db)
    accepted = [r["selected"] for r in rows if r["selected"] is not None]
    return {
        "policy_version": POLICY_VERSION, "eligible_places": total, "screened_places": len(rows),
        "selected_places": len(accepted), "skipped_places": len(rows) - len(accepted),
        "remaining_places": total - len(rows), "complete": len(rows) == total,
        "inspected_images": sum(len(r["attempts"]) for r in rows),
        "by_city": dict(sorted(Counter(r["city"] for r in accepted).items())),
        "rejection_reasons": dict(sorted(Counter(reason for r in rows for a in r["attempts"]
                                                  for reason in a["reasons"]).items())),
    }


def screen_sources(real_data, output_dir, *, seed=42, min_images=4, workers=4):
    """Resume CPU screening and return one hashed suitable source per usable place.

    ``selected_sources.jsonl`` and ``summary.json`` are snapshots.  SQLite is the
    authoritative checkpoint; each place is committed before work proceeds.
    Changing workers is safe, while CSVs, seed, policy, thresholds or this file
    require a new output directory.  Previously selected source bytes are
    checked on every resume, including an already-complete run.
    """
    if min_images < 4 or workers < 1:
        raise ValueError("Screening requires min_images >= 4 and workers >= 1")
    real_data, directory = Path(real_data).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 1, "policy_version": POLICY_VERSION, "real_data": str(real_data),
        "seed": seed, "min_images": min_images, "thresholds": THRESHOLDS,
        "code_sha256": common.file_sha256(Path(__file__)),
        "csv_sha256": {p.name: common.file_sha256(p) for p in sorted((real_data / "Dataframes").glob("*.csv"))},
    }
    config = {**config, "fingerprint": common.fingerprint(config)}
    with _screen_lock(directory):
        config_path = directory / "screen_config.json"
        if config_path.exists():
            if json.loads(config_path.read_text()) != config:
                raise ValueError("Frozen source-screening CSVs, seed, code or thresholds changed; use a new output directory")
        else:
            common.write_json(config_path, config)
        places = eligible_places(real_data, min_images)
        db = _connect(directory)
        try:
            saved = db.execute("SELECT value FROM metadata WHERE key='config_fingerprint'").fetchone()
            if saved and saved[0] != config["fingerprint"]:
                raise ValueError("Source-screening checkpoint belongs to a different configuration")
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('config_fingerprint', ?)", (config["fingerprint"],))
            db.execute("INSERT OR REPLACE INTO metadata VALUES ('eligible_places', ?)", (str(len(places)),))
            db.commit()
            completed, used_hashes = set(), set()
            for outcome in _outcomes(db):
                key = outcome["city"], outcome["place_id"]
                if key not in places:
                    raise ValueError("Saved place is absent from the frozen metadata")
                completed.add(key)
                source = outcome["selected"]
                if source is not None:
                    if (Path(source["source_path"]) not in places[key]
                            or common.file_sha256(Path(source["source_path"])) != source["source_sha256"]):
                        raise ValueError(f"Previously selected source changed: {source['source_path']}")
                    if source["source_sha256"] in used_hashes:
                        raise ValueError("Duplicate source contents in screening checkpoint")
                    used_hashes.add(source["source_sha256"])

            def tasks():
                for (city, place_id), paths in places.items():
                    if (city, place_id) not in completed:
                        yield city, place_id, ordered_views(paths, city, place_id, seed), THRESHOLDS

            def commit(outcome, task):
                # A content collision uses the next view of this place, never
                # spends a generation call and never discards the whole place.
                while outcome["selected"] is not None and outcome["selected"]["source_sha256"] in used_hashes:
                    duplicate = outcome["attempts"][-1]
                    duplicate["passed"] = False
                    duplicate["reasons"] = ["source_content_already_selected"]
                    visited = {a["source_path"] for a in outcome["attempts"]}
                    extra = _screen_place((task[0], task[1], [p for p in task[2] if str(p) not in visited], task[3]))
                    outcome["attempts"].extend(extra["attempts"])
                    outcome["selected"] = extra["selected"]
                selected = outcome["selected"]
                digest = selected["source_sha256"] if selected else None
                db.execute("INSERT INTO places VALUES (?, ?, ?, ?)",
                           (outcome["city"], outcome["place_id"], json.dumps(outcome, ensure_ascii=False, allow_nan=False), digest))
                db.commit()
                if digest:
                    used_hashes.add(digest)
                completed.add((outcome["city"], outcome["place_id"]))
                if len(completed) % 1000 == 0 or len(completed) == len(places):
                    common.write_json(directory / "summary.json", _summary(db, len(places)))
                    print(f"[source screen {len(completed)}/{len(places)}] suitable={len(used_hashes)}", flush=True)

            if workers == 1:
                _worker_init()
                for task in tasks():
                    commit(_screen_place(task), task)
            else:
                # Bound queued work and consume in the original stable order:
                # duplicate-content decisions do not depend on worker timing.
                task_iterator = iter(tasks())
                with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
                    pending = deque()
                    for _ in range(workers * 2):
                        task = next(task_iterator, None)
                        if task is not None:
                            pending.append((task, pool.submit(_screen_place, task)))
                    while pending:
                        task, future = pending.popleft()
                        commit(future.result(), task)
                        task = next(task_iterator, None)
                        if task is not None:
                            pending.append((task, pool.submit(_screen_place, task)))
            selected = [r["selected"] for r in _outcomes(db) if r["selected"] is not None]
            common.write_jsonl(directory / "selected_sources.jsonl", selected)
            common.write_json(directory / "summary.json", _summary(db, len(places)))
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            return selected
        finally:
            db.close()


def status(output_dir):
    directory = Path(output_dir).expanduser().resolve()
    database = directory / "screening.sqlite3"
    if not database.exists():
        raise FileNotFoundError(database)
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
        total = int(db.execute("SELECT value FROM metadata WHERE key='eligible_places'").fetchone()[0])
        return _summary(db, total)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "status"):
        child = sub.add_parser(name)
        child.add_argument("--output-dir", type=Path, required=True)
        if name == "run":
            child.add_argument("--real-data", type=Path, default=common.WORKSPACE_ROOT / "dataset/gsv-cities")
            child.add_argument("--seed", type=int, default=42)
            child.add_argument("--min-images", type=int, default=4)
            child.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.command == "run":
        screen_sources(args.real_data, args.output_dir, seed=args.seed, min_images=args.min_images, workers=args.workers)
    print(json.dumps(status(args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
