#!/usr/bin/env python3
"""Prepare, run, inspect and export the pinned AdaptCities release.

The release has 160k prompts, not 160k published verification annotations.
All annotations exported here are measurements from this local run.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
REVISION = "ca79c45f5d2854536e37a588a69a94b112a993fa"
PROMPT_SHA = "afdedfa54e82d9550ab7deead7a35dba26604b5461c0834af4da743be54ae119"
PROMPTS = "prompts/adaptcities_160k_prompts.jsonl"
ROUTES = {"global": 50239, "local": 45804, "dual": 63957}
GIB = 1024**3


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


@contextmanager
def lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another prepare/run/export process owns this output")
        yield


def configure(args):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    if args.env_file:
        load_dotenv(args.env_file, override=True)
    from generation.preflight import load_environment
    load_environment()
    os.environ["ADAPTVPR_DISABLE_MOCK"] = "1"
    os.environ["ADAPTVPR_FORCE_MOCK_LLM"] = "0"
    os.environ["ICLIGHT_AUTO_START"] = "0"
    os.environ["LIGHTX2V_AUTO_START"] = "0"
    os.environ["LIGHTX2V_API_RETRIES"] = "1"


def provenance(args):
    prefixes = ("ADAPTVPR_", "ICLIGHT_", "LIGHTX2V_", "VISMATCH_", "OPENAI_")
    settings = {k: v for k, v in os.environ.items()
                if k.startswith(prefixes) and not any(s in k for s in ("KEY", "TOKEN", "SECRET", "PASSWORD"))}
    code = {}
    for folder in ("generation", "verification", "prompts", "adapters"):
        for p in sorted((ROOT / folder).rglob("*.py")):
            code[str(p.relative_to(ROOT))] = digest(p)
    code["scripts/adaptcities.py"] = digest(Path(__file__))
    for name in ("start_adaptcities.py", "resource_limits.py"):
        code["scripts/" + name] = digest(ROOT / "scripts" / name)
    weights = {}
    for key in ("ICLIGHT_BASE_MODEL_PATH", "ICLIGHT_MODEL_PATH", "LIGHTX2V_MODEL_PATH",
                "LIGHTX2V_LORA_PATH", "LIGHTX2V_DISK_MODEL_PATH"):
        p = Path(os.environ[key])
        if not p.exists():
            raise FileNotFoundError(f"{key}: {p}")
        files = sorted(p.rglob("*")) if p.is_dir() else [p]
        weights[key] = [{"path": str(f), "size": f.stat().st_size,
                         "mtime_ns": f.stat().st_mtime_ns}
                        for f in files if f.is_file() and ".cache" not in f.parts]
    return {"dataset": "shunpeng/AdaptCities", "revision": REVISION,
            "prompts_sha256": PROMPT_SHA, "image_root": str(args.image_root),
            "seed": 0, "reflection": True, "max_reflections": 3,
            "settings": settings, "code_sha256": code, "weight_inventory": weights,
            "resource_environment": {k: os.getenv(k) for k in ("TORCH_COMPILE_DISABLE", "TORCHINDUCTOR_COMPILE_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "QWEN_MEMORY_LIMIT", "QWEN_MEMORY_SWAP_LIMIT", "QWEN_CPU_LIMIT", "HF_HUB_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")},
            "weight_inventory_method": "pinned revisions plus file size and mtime; not weight content hashes",
            "qwen_revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17"}


def fetch_release(args):
    files = ["README.md", "LICENSE", PROMPTS, "prompts/README.md",
             "prompts/adaptcities_160k_prompts.summary.json", "prompts/by_city/index.json",
             "metadata/README.md", "metadata/annotations.example.jsonl",
             "examples/README.md", "examples/sample_record.json"]
    hashes = {}
    for name in files:
        target = args.data_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            cached = Path.home() / ".cache/huggingface/hub/datasets--shunpeng--AdaptCities/snapshots" / REVISION / name
            tmp = target.with_name(target.name + ".part")
            if cached.is_file():
                shutil.copyfile(cached, tmp)
            else:
                url = f"https://huggingface.co/datasets/shunpeng/AdaptCities/resolve/{REVISION}/{name}"
                with urllib.request.urlopen(url, timeout=90) as src, tmp.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            tmp.replace(target)
        hashes[name] = digest(target)
    if hashes[PROMPTS] != PROMPT_SHA:
        raise ValueError("Official prompt checksum mismatch")
    atomic_json(args.data_dir / "provenance.json", {
        "revision": REVISION, "files_sha256": hashes,
        "example_only_files": ["metadata/annotations.example.jsonl", "examples/sample_record.json"],
        "note": "Official example bytes are preserved. These files are example_only and are never joined to generated annotations."})


def prepare(args):
    from generation.inputs import normalize_frozen_prompt_entry
    fetch_release(args)
    rows = list(read_jsonl(args.data_dir / PROMPTS))
    seen, sources, pilot_keys = set(), set(), set()
    pilot, remaining = [], []
    routes, cities = Counter(), Counter()
    for row in rows:
        sid = row["sample_id"]
        if sid in seen or not sid.startswith("adapt_") or not sid[6:].isdigit():
            raise ValueError(f"Invalid or duplicate sample_id: {sid}")
        seen.add(sid)
        source = args.image_root / row["city"] / row["source_id"]
        if not source.is_file() or not source.resolve().is_relative_to(args.image_root):
            raise FileNotFoundError(source)
        normalize_frozen_prompt_entry(row, source)
        sources.add(str(source))
        routes[row["route"]] += 1
        cities[row["city"]] += 1
        key = (row["city"], row["route"])
        (remaining if key in pilot_keys else pilot).append(row)
        pilot_keys.add(key)
    if len(rows) != 160000 or routes != ROUTES or len(cities) != 23:
        raise ValueError(f"Release counts differ: {len(rows)}, {routes}, {cities}")
    prov = provenance(args)
    config_hash = fingerprint(prov)
    existing = args.output / "experiment.json"
    if existing.exists():
        previous = json.loads(existing.read_text())
        if previous["config_fingerprint"] != config_hash:
            raise RuntimeError("Prepared configuration changed; use a separate output directory")
        load_tasks(args, previous)
        print("Already prepared and verified", flush=True)
        return
    shards = []
    groups = [pilot] + [remaining[i:i+1000] for i in range(0, len(remaining), 1000)]
    for i, group in enumerate(groups):
        path = args.output / "shards" / f"{i:04d}.jsonl"
        write_jsonl(path, group)
        shards.append({"path": str(path), "count": len(group), "sha256": digest(path), "pilot": i == 0})
    manifest = {"config_fingerprint": config_hash, "provenance": prov, "requested": len(rows),
                "unique_sources": len(sources), "route_counts": routes, "city_counts": cities,
                "pilot_count": len(pilot), "shards": shards, "created_at": time.time()}
    atomic_json(existing, manifest)
    print(json.dumps({k: manifest[k] for k in ("requested", "unique_sources", "route_counts", "pilot_count", "config_fingerprint")}), flush=True)


def load_tasks(args, manifest):
    if digest(args.data_dir / PROMPTS) != PROMPT_SHA:
        raise ValueError("Prompt checksum changed")
    rows = []
    for shard in manifest["shards"]:
        if digest(shard["path"]) != shard["sha256"]:
            raise ValueError(f"Shard changed: {shard['path']}")
        group = list(read_jsonl(shard["path"]))
        if len(group) != shard["count"]:
            raise ValueError("Shard count mismatch")
        rows.extend(group)
    if len(rows) != manifest["requested"] or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError("Task coverage mismatch")
    return rows


def image_info(path):
    from PIL import Image
    with Image.open(path) as im:
        im.load()
        size = list(im.size)
    return {"sha256": digest(path), "size": size, "bytes": Path(path).stat().st_size}


def valid_record(record, row, config_hash):
    try:
        accepted_profiles = {config_hash} if isinstance(config_hash, str) else set(config_hash)
        if record["config_fingerprint"] not in accepted_profiles or record["input_fingerprint"] != fingerprint(row):
            return False
        if record["status"] not in ("passed", "failed") or record["generated"] is not True:
            return False
        if record["input_prompt"] != row["prompt"]:
            return False
        if digest(record["source_path"]) != record["source_sha256"]:
            return False
        for key in ("sample_id", "source_id", "city", "route", "condition"):
            if record[key] != row[key]:
                return False
        rounds = record["reflection_rounds"]
        if not 1 <= len(rounds) <= (1 if row["route"] == "global" else 4):
            return False
        if rounds[0]["prompt"] != row["prompt"] or record["rounds_used"] != len(rounds):
            return False
        passed = record["s_geo"] >= record["tau_geo"] and record["s_div"] >= record["tau_div"]
        if record["passed"] != passed or record["eligible_for_training"] != passed:
            return False
        if (record["status"] == "passed") != passed:
            return False
        needed = {record["output_path"]} | {a["image_path"] for a in rounds}
        if not needed.issubset(record["artifacts"]):
            return False
        for path, info in record["artifacts"].items():
            if image_info(path) != info or info["size"] != record["source_size"]:
                return False
        return True
    except (OSError, KeyError, ValueError, TypeError):
        return False


def read_completed(args, rows, config_hash):
    completed = Completed()
    for row in rows:
        path = args.output / "records" / (row["sample_id"] + ".json")
        if path.exists():
            try:
                record = json.loads(path.read_text())
                if valid_record(record, row, config_hash):
                    completed.add(record)
                    (args.output / "errors" / f"{row['sample_id']}.json").unlink(missing_ok=True)
            except (OSError, ValueError):
                pass
    return completed


class Completed(dict):
    """Keep small resume summaries, not all reflection transcripts, in RAM."""
    def __init__(self):
        super().__init__()
        self.routes = defaultdict(lambda: {"completed": 0, "passed": 0, "seconds": 0., "bytes": 0})
        self.passed = 0

    def add(self, r):
        if r["sample_id"] in self:
            raise ValueError("Duplicate completed sample")
        size = sum(a["bytes"] for a in r["artifacts"].values())
        self[r["sample_id"]] = {"route": r["route"], "elapsed_seconds": r["elapsed_seconds"],
                                 "artifact_bytes": size, "record_bytes": len(json.dumps(r).encode())}
        x = self.routes[r["route"]]
        x["completed"] += 1
        x["passed"] += int(r["passed"])
        x["seconds"] += r["elapsed_seconds"]
        x["bytes"] += size
        self.passed += int(r["passed"])


def enrich(record, row, source, config_hash, elapsed):
    from generation.inputs import parse_condition
    from verification.evaluator import ROUTE_THRESHOLDS, DUAL_RAINY_VEHICLE_RELAXED_MIN_DIV
    weather, occlusion = parse_condition(row["condition"])
    thresholds = ROUTE_THRESHOLDS[row["route"]]
    tau_div = thresholds["TAU_DIV"]
    if row["route"] == "dual" and weather in ("rain", "rainy_night") and occlusion == "vehicle":
        tau_div = DUAL_RAINY_VEHICLE_RELAXED_MIN_DIV
    record.update(config_fingerprint=config_hash, input_fingerprint=fingerprint(row),
                  tau_geo=thresholds["TAU_GEO"], tau_div=tau_div,
                  source_sha256=digest(source), source_size=image_info(source)["size"],
                  elapsed_seconds=elapsed, completed_at=time.time(), annotation_origin="local_regeneration",
                  dataset_revision=REVISION, example_only=False)
    for attempt in record["reflection_rounds"]:
        attempt["eval"].update(tau_geo=record["tau_geo"], tau_div=tau_div)
    paths = {record["output_path"]} | {a["image_path"] for a in record["reflection_rounds"]}
    record["artifacts"] = {p: image_info(p) for p in sorted(paths)}
    return record


def progress(args, manifest, completed, state, **extra):
    routes = completed.routes
    remaining_seconds = sum((ROUTES[k] - x["completed"]) * x["seconds"] / x["completed"] for k, x in routes.items())
    data = {"state": state, "complete": state == "complete", "pid": os.getpid(),
            "requested": manifest["requested"], "completed": len(completed),
            "passed": completed.passed, "route_stats": dict(routes),
            "updated_at": time.time(), "free_bytes": shutil.disk_usage(args.output).free,
            "estimated_remaining_seconds": remaining_seconds if len(routes) == 3 else None, **extra}
    atomic_json(args.output / "progress.json", data)
    return data


def pilot_budget(args, manifest, completed):
    pilot_ids = {r["sample_id"] for r in read_jsonl(manifest["shards"][0]["path"])}
    if not pilot_ids.issubset(completed):
        return
    by_route = defaultdict(list)
    for sid in pilot_ids:
        by_route[completed[sid]["route"]].append(completed[sid])
    expected_bytes = 0
    expected_seconds = 0
    for route, values in by_route.items():
        expected_bytes += ROUTES[route] * sum(r["artifact_bytes"] + r["record_bytes"] * 4 for r in values) / len(values)
        expected_seconds += ROUTES[route] * sum(r["elapsed_seconds"] for r in values) / len(values)
    used = sum(x["bytes"] for x in completed.routes.values())
    required = max(0, expected_bytes * 1.5 - used) + args.reserve_gib * GIB
    report = {"pilot_count": len(pilot_ids), "estimated_total_bytes": expected_bytes,
              "storage_safety_factor": 1.5, "estimated_total_seconds": expected_seconds,
              "required_free_bytes": required, "free_bytes": shutil.disk_usage(args.output).free,
              "passed": shutil.disk_usage(args.output).free >= required}
    atomic_json(args.output / "pilot_report.json", report)
    if not report["passed"]:
        raise RuntimeError("Pilot storage estimate exceeds available disk; see pilot_report.json")


def service_snapshot(args):
    import requests
    result = {}
    with requests.Session() as session:
        session.trust_env = False
        for name in ("ICLIGHT", "LIGHTX2V"):
            url = os.environ[name + "_API_URL"].rsplit("/", 1)[0] + "/health"
            response = session.get(url, timeout=10)
            response.raise_for_status()
            result[name] = response.json()
    atomic_json(args.output / "service_runtime.json", result)
    return result


def rotate_logs(args):
    # All these logs are opened with O_APPEND by our service/supervisor launchers.
    # copy/truncate retains their file descriptors and bounds verbose model logs.
    for path in args.output.parent.joinpath("logs").glob("*.log"):
        if path.stat().st_size > 50 * 1024**2:
            previous = path.with_name(path.name + ".1")
            if previous.exists():
                previous.replace(path.with_name(path.name + ".2"))
            shutil.copyfile(path, previous)
            with path.open("r+") as stream:
                stream.truncate(0)


def export_data(args, manifest, rows, completed):
    ordered = sorted(completed)
    def records():
        for sid in ordered:
            yield json.loads((args.output / "records" / f"{sid}.json").read_text())
    write_jsonl(args.output / "metadata.jsonl", records())
    fields = ("sample_id", "source_id", "city", "route", "condition", "s_geo", "s_div", "tau_geo", "tau_div",
              "passed", "eligible_for_training", "status", "rounds_used", "annotation_origin", "example_only")
    write_jsonl(args.output / "annotations.jsonl", ({k: r[k] for k in fields} for r in records()))
    train_fields = ("sample_id", "source_id", "city", "route", "condition", "source_path", "output_path", "s_geo", "s_div", "tau_geo", "tau_div", "passed", "eligible_for_training")
    write_jsonl(args.output / "train_manifest.jsonl", ({k: r[k] for k in train_fields} for r in records() if r["passed"] and r["eligible_for_training"]))
    groups = {key: Counter() for key in ("city", "route", "condition", "rounds_used")}
    for r in records():
        for key in groups:
            groups[key][str(r[key])] += 1
    errors = len(list((args.output / "errors").glob("*.json")))
    atomic_json(args.output / "summary.json", {
        "requested": manifest["requested"], "completed": len(ordered), "passed": completed.passed,
        "rejected": len(ordered) - completed.passed, "infrastructure_errors": errors,
        "complete": len(ordered) == manifest["requested"] and errors == 0, "distributions": groups,
        "generation_seconds": sum(x["seconds"] for x in completed.routes.values()), "exported_at": time.time(),
        "provenance": "experiment.json", "annotation_origin": "local_regeneration"})


def run(args, manifest, rows):
    from generation.preflight import check_environment
    from resource_limits import require_limits
    require_limits()
    config_hash = manifest["config_fingerprint"]
    if fingerprint(provenance(args)) != config_hash:
        raise RuntimeError("Code, model inventory or settings changed since prepare; refusing mixed run")
    profiles = {config_hash, *manifest.get("compatible_profile_fingerprints", [])}
    completed = read_completed(args, rows, profiles)
    agent = None
    def stop(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        failures = check_environment(planner=True)
        if failures:
            raise RuntimeError("Preflight failed: " + "; ".join(failures))
        service_snapshot(args)
        from generation.agent import SceneAugmentAgent
        agent = SceneAugmentAgent(mock=False, max_generations=4, planning_enabled=False, reflection_enabled=True, base_seed=0)
        agent.evaluator._load_matcher()
        if agent.evaluator.matcher_name != "superpoint-lightglue":
            raise RuntimeError("SuperPoint/LightGlue is required; refusing fallback verifier")
        pilot_budget(args, manifest, completed)
        for row in rows:
            sid = row["sample_id"]
            if sid in completed:
                continue
            if shutil.disk_usage(args.output).free < args.reserve_gib * GIB:
                raise RuntimeError("Disk reserve reached; generation stopped safely")
            source = args.image_root / row["city"] / row["source_id"]
            progress(args, manifest, completed, "running", current_sample=sid, phase="pilot" if len(completed) < manifest["pilot_count"] else "full")
            for retry in range(4):
                started = time.monotonic()
                try:
                    r = agent.run_path(source, args.output / "images", entry=row, frozen_prompt=True, sample_id=sid, collect_bad=False)
                    r["verification_models"] = {"geometry": agent.evaluator.matcher_name, "appearance": "openai/clip-vit-base-patch32"}
                    r = enrich(r, row, source, config_hash, time.monotonic() - started)
                    if not valid_record(r, row, config_hash):
                        raise RuntimeError("Generated record failed integrity validation")
                    atomic_json(args.output / "records" / f"{sid}.json", r)
                    completed.add(r)
                    (args.output / "errors" / f"{sid}.json").unlink(missing_ok=True)
                    # Only this task's dedicated service directory is cleaned, after commit.
                    service_root = args.output / "service_tmp" / "service_outputs"
                    for p in service_root.glob("*/*.png"):
                        try:
                            p.unlink()
                        except OSError as exc:
                            print(f"Temporary output cleanup warning: {exc}", flush=True)
                    print(f"[{len(completed)}/{len(rows)}] {sid} {r['route']} {r['status']} rounds={r['rounds_used']} seconds={r['elapsed_seconds']:.1f}", flush=True)
                    break
                except Exception as exc:
                    atomic_json(args.output / "errors" / f"{sid}.json", {"sample_id": sid, "status": "error", "error": str(exc), "retry": retry, "at": time.time()})
                    if retry == 3:
                        raise
                    progress(args, manifest, completed, "retrying", current_sample=sid, retry=retry+1, error=str(exc))
                    print(f"Retry {retry+1}/3 for {sid}: {exc}", flush=True)
                    time.sleep(30)
            if len(completed) == manifest["pilot_count"]:
                pilot_budget(args, manifest, completed)
                export_data(args, manifest, rows, completed)
            progress(args, manifest, completed, "running")
            if len(completed) % 100 == 0:
                rotate_logs(args)
        export_data(args, manifest, rows, completed)
        progress(args, manifest, completed, "complete")
    except BaseException as exc:
        progress(args, manifest, completed, "stopped_incomplete", error=f"{type(exc).__name__}: {exc}")
        export_data(args, manifest, rows, completed)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "status", "export"))
    parser.add_argument("--data-dir", type=Path, default=ROOT.parent / "AdaptCities/release")
    parser.add_argument("--output", type=Path, default=ROOT.parent / "AdaptCities/run")
    parser.add_argument("--image-root", type=Path, default=Path("/home/admin123/视频/Bag-of-Queries/data/train/gsv-cities/Images"))
    parser.add_argument("--env-file", type=Path, default=ROOT.parent / "AdaptCities/runtime.env")
    parser.add_argument("--reserve-gib", type=float, default=30.)
    args = parser.parse_args()
    for key in ("data_dir", "output", "image_root", "env_file"):
        setattr(args, key, getattr(args, key).resolve())
    if args.reserve_gib < 30:
        parser.error("--reserve-gib must be at least 30")
    if args.action == "status":
        path = args.output / "progress.json"
        startup = args.output.parent / "startup.json"
        value = json.loads(path.read_text()) if path.exists() else json.loads(startup.read_text()) if startup.exists() else {"state": "not_started"}
        try:
            os.kill(value.get("pid", -999999), 0)
            value["process_alive"] = True
        except (OSError, OverflowError):
            value["process_alive"] = False
        monitor = args.output.parent / "memory_watchdog.json"
        if monitor.exists():
            value["resource_protection"] = json.loads(monitor.read_text())
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return
    configure(args)
    with lock(args.output):
        if args.action == "prepare":
            prepare(args)
        else:
            manifest = json.loads((args.output / "experiment.json").read_text())
            rows = load_tasks(args, manifest)
            if args.action == "run":
                run(args, manifest, rows)
            else:
                profiles = {manifest["config_fingerprint"], *manifest.get("compatible_profile_fingerprints", [])}
                completed = read_completed(args, rows, profiles)
                export_data(args, manifest, rows, completed)


if __name__ == "__main__":
    main()
