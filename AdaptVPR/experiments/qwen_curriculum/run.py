"""Serial, bounded Qwen generation and zero-call reuse of historical images.

Each HTTP POST consumes a durable budget reservation before it is sent. Unknown
or failed calls are never silently retried. Saved images can be verified again
on resume without spending another call. SALAD is deliberately absent here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlparse

from PIL import Image
import requests

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import common
common.use_adaptvpr()
from experiments.qwen_curriculum.quality import QwenQualityVerifier, quality_definition

DOMAINS = {"night", "snow", "fog", "rain"}


@contextmanager
def run_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another worker owns {directory}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def read_rows(directory, name):
    path = directory / name
    return common.read_jsonl(path) if path.exists() else []


def sealed(row):
    row = dict(row)
    row.pop("result_sha256", None)
    return {**row, "result_sha256": common.fingerprint(row)}


def validate_image(row, field="output_path", hash_field="output_sha256"):
    path = Path(row[field])
    if not path.is_file() or common.file_sha256(path) != row[hash_field]:
        raise ValueError(f"Recorded image missing or changed: {path}")


def validate_records(rows, config, jobs=None):
    planned = None if jobs is None else {job["sample_id"]: job for job in jobs}
    ids = set()
    for row in rows:
        if row["sample_id"] in ids:
            raise ValueError("Duplicate saved sample_id")
        ids.add(row["sample_id"])
        if row.get("execution_fingerprint") != config["fingerprint"]:
            raise ValueError("Execution configuration changed")
        expected = row.get("result_sha256")
        if expected != sealed(row)["result_sha256"]:
            raise ValueError("Saved result checksum mismatch")
        if planned is not None:
            job = planned.get(row["sample_id"])
            if job is None or any(row.get(key) != value for key, value in job.items()):
                raise ValueError("Saved result differs from its exact planned job")
        if row.get("output_path"):
            validate_image(row)
        if row.get("raw_output_path"):
            validate_image(row, "raw_output_path", "raw_output_sha256")


def freeze(directory, config):
    config = dict(config)
    config["fingerprint"] = common.fingerprint(config)
    path = directory / "execution_config.json"
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError("Immutable execution settings changed; use a new run directory")
    else:
        if any((directory / name).exists() for name in ("results.jsonl", "generated.jsonl", "attempts.jsonl")):
            raise ValueError("Saved execution files exist without configuration")
        common.write_json(path, config)
    return config


def execution_definition(args):
    from experiments.qwen_curriculum import quality, weather_signal
    from experiments.generation_diagnosis import verifier_controls
    from verification import evaluator
    paths = [Path(__file__), Path(common.__file__), Path(quality.__file__),
             Path(weather_signal.__file__), Path(verifier_controls.__file__), Path(evaluator.__file__)]
    return {"schema_version": 1, "run_dir": str(args.run_dir),
            "quality": quality_definition(clip_device="cpu"), "matcher_device": args.device,
            "cpu_threads": args.cpu_threads,
            "implementation_sha256": {str(path.resolve().relative_to(common.ADAPTVPR_ROOT)):
                                      common.file_sha256(path) for path in paths}}


def load_plan(directory):
    config = json.loads((directory / "plan_config.json").read_text())
    expected = config.pop("fingerprint")
    if common.fingerprint(config) != expected:
        raise ValueError("Plan configuration checksum mismatch")
    config["fingerprint"] = expected
    if common.file_sha256(directory / "plan.jsonl") != config["plan_sha256"]:
        raise ValueError("Plan manifest checksum mismatch")
    rows = common.read_jsonl(directory / "plan.jsonl")
    ids = set()
    for original in rows:
        row = dict(original)
        digest = row.pop("record_sha256")
        if common.fingerprint(row) != digest or row["condition"] not in DOMAINS:
            raise ValueError("Invalid sealed plan row")
        if row["sample_id"] in ids:
            raise ValueError("Duplicate planned sample")
        if not re.fullmatch(r"qwen_[0-9a-f]{24}", row["sample_id"]):
            raise ValueError("Unsafe planned sample_id")
        ids.add(row["sample_id"])
        validate_image(row, "source_path", "source_sha256")
        with Image.open(row["source_path"]) as image:
            image.load()
            if list(image.size) != row["source_dimensions"]:
                raise ValueError("Planned source dimensions changed")
    if len(rows) != config["num_images"]:
        raise ValueError("Plan length differs from configured budget")
    return config, rows


def sealed_attempt(row):
    row = dict(row)
    row.pop("attempt_sha256", None)
    return {**row, "attempt_sha256": common.fingerprint(row)}


def validate_attempts(attempts, jobs, config):
    planned = {job["sample_id"]: job for job in jobs}
    ids = set()
    for number, row in enumerate(attempts, start=1):
        job = planned.get(row.get("sample_id"))
        if (job is None or row["sample_id"] in ids or row.get("call_number") != number
                or row.get("execution_fingerprint") != config["fingerprint"]
                or row.get("record_sha256") != job["record_sha256"]
                or row.get("attempt_sha256") != sealed_attempt(row)["attempt_sha256"]):
            raise ValueError("Invalid durable call ledger")
        ids.add(row["sample_id"])
    if len(attempts) > config["max_calls"]:
        raise ValueError("Durable call ledger exceeds the frozen budget")
    return ids


def reserve(directory, attempts, job, max_calls, config):
    if len(attempts) >= max_calls:
        return False
    if any(row["sample_id"] == job["sample_id"] for row in attempts):
        raise ValueError("A source job may receive only one generation call")
    attempts.append(sealed_attempt({"call_number": len(attempts) + 1, "sample_id": job["sample_id"],
                     "execution_fingerprint": config["fingerprint"], "record_sha256": job["record_sha256"],
                     "reserved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    common.write_jsonl(directory / "attempts.jsonl", attempts)
    return True


def publish(directory, rows, planned=None, calls=0, state="running"):
    common.write_jsonl(directory / "results.jsonl", rows)
    accepted = [row for row in rows if row.get("passed") is True and row.get("eligible_for_training") is True]
    common.write_jsonl(directory / "training_manifest.jsonl", accepted)
    summary = {"state": state, "planned": planned, "completed": len(rows), "accepted": len(accepted),
               "rejected": sum(row.get("status") == "rejected" for row in rows),
               "generation_errors": sum(row.get("status") == "generation_error" for row in rows),
               "http_calls_reserved": calls,
               "conditions_completed": dict(Counter(row["condition"] for row in rows)),
               "conditions_accepted": dict(Counter(row["condition"] for row in accepted)),
               "rejection_reasons": dict(Counter(reason for row in rows for reason in row.get("rejection_reasons", []))),
               "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    common.write_json(directory / "summary.json", summary)
    return summary


def verify(row, verifier, config):
    with Image.open(row["source_path"]) as source, Image.open(row["output_path"]) as generated:
        result = verifier.evaluate(source.convert("RGB"), generated.convert("RGB"), row["condition"])
    if any(key in row and row[key] != value for key, value in result.items()):
        raise ValueError("Quality verifier changed a recorded job field")
    return sealed({**row, **result, "execution_fingerprint": config["fingerprint"]})


def service_identity(health):
    if (health.get("status") != "ok" or health.get("model_loaded") is not True
            or health.get("generator_ready") is not True or health.get("source_modified") is not False
            or health.get("error") is not None):
        raise RuntimeError("Qwen service is not ready or its upstream source was modified")
    if health.get("model_id") != "Qwen/Qwen-Image-Edit-2511" or health.get("canvas_policy") != "source_aspect_v1":
        raise ValueError("Expected the proven Qwen-2511 source-aspect service")
    if health.get("sampling", {}).get("infer_steps") != 4 or health.get("sampling", {}).get("guidance_scale") != 1.0:
        raise ValueError("Expected 4-step guidance=1 Qwen sampling")
    return {key: value for key, value in health.items() if key not in {"status", "model_loaded", "generator_ready", "error"}}


def service_output_directory():
    return (common.ADAPTVPR_ROOT / "tmp/service_outputs/lightx2v").resolve()


def generate(args):
    if type(args.max_calls) is not int or not 1 <= args.max_calls <= 1000:
        raise ValueError("Generation call budget must be between 1 and 1000")
    directory = args.run_dir
    with run_lock(directory):
        plan_config, jobs = load_plan(directory)
        session = requests.Session()
        session.trust_env = False
        health_url = args.qwen_url.rsplit("/", 1)[0] + "/health"
        response = session.get(health_url, timeout=15)
        response.raise_for_status()
        config = freeze(directory, {**execution_definition(args), "stage": "generate",
                                   "plan_fingerprint": plan_config["fingerprint"], "max_calls": args.max_calls,
                                   "qwen_url": args.qwen_url, "request_timeout": args.request_timeout,
                                   "service": service_identity(response.json()),
                                   "call_policy": "one POST per job; durable reservation; no automatic retries",
                                   "source_encoding": "lossless RGB PNG decoded from original"})
        rows, generated = read_rows(directory, "results.jsonl"), read_rows(directory, "generated.jsonl")
        validate_records(rows, config, jobs)
        validate_records(generated, config, jobs)
        attempts = read_rows(directory, "attempts.jsonl")
        attempt_ids = validate_attempts(attempts, jobs, config)
        if not {row["sample_id"] for row in rows + generated} <= attempt_ids:
            raise ValueError("Saved generation has no durable call reservation")
        done = {row["sample_id"] for row in rows}
        saved = {row["sample_id"]: row for row in generated}
        verifier = None
        publish(directory, rows, len(jobs), len(attempts))
        for job in jobs:
            sample_id = job["sample_id"]
            if sample_id in done:
                continue
            if sample_id not in saved:
                if sample_id in attempt_ids:
                    row = sealed({**job, "execution_fingerprint": config["fingerprint"],
                                  "status": "generation_error", "passed": False, "eligible_for_training": False,
                                  "error": "Interrupted reserved call; generation outcome unknown; no repeat POST"})
                    rows.append(row)
                    done.add(sample_id)
                    publish(directory, rows, len(jobs), len(attempts))
                    continue
                if not reserve(directory, attempts, job, args.max_calls, config):
                    break
                attempt_ids.add(sample_id)
                for name in ("images", "raw", "source_inputs"):
                    (directory / name).mkdir(exist_ok=True)
                input_path = directory / "source_inputs" / f"{sample_id}.png"
                started = time.monotonic()
                try:
                    with Image.open(job["source_path"]) as image:
                        image.convert("RGB").save(input_path)
                    payload = {"image_path": str(input_path), "prompt": job["prompt"],
                               "negative_prompt": "", "seed": job["seed"], "infer_steps": 4, "guidance_scale": 1.0}
                    response = session.post(args.qwen_url, json=payload, timeout=args.request_timeout)
                    response.raise_for_status()
                    sampling = response.json()
                    result = Path(sampling["result_path"]).resolve()
                    service_dir = service_output_directory()
                    if result.parent != service_dir or not result.is_file():
                        raise ValueError("Qwen returned a missing file or path outside its owned output directory")
                    if sampling.get("canvas_policy") != "source_aspect_v1" or sampling.get("source_dimensions") != job["source_dimensions"]:
                        raise ValueError("Qwen response canvas/source dimensions changed")
                    raw_path = directory / "raw" / f"{sample_id}.png"
                    with Image.open(result) as image:
                        image.load()
                        if list(image.size) != sampling["raw_dimensions"] or list(image.size) != sampling["target_shape"][::-1]:
                            raise ValueError("Qwen output dimensions differ from reported canvas")
                        output = image.convert("RGB").resize(tuple(job["source_dimensions"]), Image.Resampling.LANCZOS)
                    result.replace(raw_path)
                    metadata = result.with_suffix(".json")
                    if metadata.exists():
                        metadata.replace(raw_path.with_suffix(".json"))
                    output_path = directory / "images" / f"{sample_id}.png"
                    temporary = output_path.with_suffix(".tmp.png")
                    output.save(temporary)
                    temporary.replace(output_path)
                    row = sealed({**job, "execution_fingerprint": config["fingerprint"], "sampling": sampling,
                                  "output_path": str(output_path), "output_sha256": common.file_sha256(output_path),
                                  "raw_output_path": str(raw_path), "raw_output_sha256": common.file_sha256(raw_path),
                                  "generation_seconds": time.monotonic() - started, "origin": "new_hard_source_qwen"})
                    generated.append(row)
                    common.write_jsonl(directory / "generated.jsonl", generated)
                    saved[sample_id] = row
                except Exception as exc:
                    row = sealed({**job, "execution_fingerprint": config["fingerprint"],
                                  "status": "generation_error", "passed": False, "eligible_for_training": False,
                                  "error": f"{type(exc).__name__}: {exc}"})
                    rows.append(row)
                    publish(directory, rows, len(jobs), len(attempts), "stopped_on_generation_error")
                    raise
                finally:
                    input_path.unlink(missing_ok=True)
            try:
                if verifier is None:
                    verifier = QwenQualityVerifier(device=args.device, clip_device="cpu")
                row = verify(saved[sample_id], verifier, config)
            except Exception:
                publish(directory, rows, len(jobs), len(attempts), "stopped_on_verification_error")
                raise
            rows.append(row)
            done.add(sample_id)
            summary = publish(directory, rows, len(jobs), len(attempts))
            print(f"[{len(rows)}/{len(jobs)}] {job['city']} {job['condition']} {row['status']} "
                  f"geo={row['s_geo']:.3f} weather_shift={row['weather_shift']:.2f} accepted={summary['accepted']}", flush=True)
        summary = publish(directory, rows, len(jobs), len(attempts),
                          "complete" if len(rows) == len(jobs) else "call_budget_exhausted")
        print(json.dumps(summary, ensure_ascii=False), flush=True)


def historical_jobs(manifest, include_positive=False):
    old_config = json.loads((manifest.parent / "generation_config.json").read_text())
    original_fingerprint = old_config.pop("fingerprint")
    # Historical runner fingerprints use the default JSON separators.
    actual_fingerprint = hashlib.sha256(json.dumps(old_config, sort_keys=True, ensure_ascii=False,
                                                   allow_nan=False).encode()).hexdigest()
    if actual_fingerprint != original_fingerprint or old_config.get("mode") != "qwen":
        raise ValueError("Historical generation configuration checksum/mode mismatch")
    old_config["fingerprint"] = original_fingerprint
    hashes = {row["source_path"]: row["source_sha256"] for row in old_config["sources"]}
    prompts = {(row["cond"], row["prompt_variant"]): row["prompt"] for row in old_config["prompts"]}
    selected = {}
    for row in common.read_jsonl(manifest):
        if row.get("status") != "ok" or row.get("mode") != "qwen" or row.get("cond") not in DOMAINS:
            continue
        if row.get("prompt_variant") not in ({"released", "positive"} if include_positive else {"released"}):
            continue
        if row.get("config_fingerprint") != old_config["fingerprint"]:
            raise ValueError("Historical row belongs to another frozen configuration")
        service_identity(row["service_health"])
        if (row.get("prompt") != prompts.get((row["cond"], row["prompt_variant"]))
                or row.get("negative_prompt") != old_config["negative_prompt"]
                or row.get("seed") != common.candidate_seed(old_config["seed"], f"{row['source_path']}|{row['cond']}", 0)):
            raise ValueError("Historical prompt or seed differs from its frozen configuration")
        validate_image(row)
        validate_image(row, "raw_output_path", "raw_output_sha256")
        if common.file_sha256(Path(row["source_path"])) != hashes[row["source_path"]]:
            raise ValueError("Historical source bytes changed")
        sample_id = "reused_" + common.fingerprint({"output_sha256": row["output_sha256"],
                                                    "source_sha256": hashes[row["source_path"]],
                                                    "source_path": row["source_path"], "condition": row["cond"],
                                                    "prompt": row["prompt"], "negative_prompt": row["negative_prompt"],
                                                    "seed": row["seed"], "prompt_variant": row["prompt_variant"]})[:24]
        parts = Path(row["source_path"]).name.split("_")
        selected[sample_id] = {"sample_id": sample_id, "source_path": row["source_path"],
                               "source_sha256": hashes[row["source_path"]], "output_path": row["output_path"],
                               "output_sha256": row["output_sha256"], "raw_output_path": row["raw_output_path"],
                               "raw_output_sha256": row["raw_output_sha256"], "condition": row["cond"],
                               "city": parts[0], "place_id": int(parts[1]), "prompt": row["prompt"],
                               "negative_prompt": row["negative_prompt"], "seed": row["seed"],
                               "origin": "historical_qwen_pilot_not_new_hard_source_budget",
                               "historical_manifest": str(manifest), "historical_config_fingerprint": old_config["fingerprint"],
                               "historical_passed": row["passed"], "prompt_variant": row["prompt_variant"]}
        with Image.open(row["source_path"]) as source, Image.open(row["output_path"]) as output:
            source.load()
            output.load()
            if output.size != source.size:
                raise ValueError("Historical normalized output differs from the source dimensions")
    return list(selected.values())


def import_existing(args):
    directory = args.run_dir
    with run_lock(directory):
        jobs = historical_jobs(args.manifest, args.include_positive)
        config = freeze(directory, {**execution_definition(args), "stage": "import_existing",
                                   "manifest": str(args.manifest), "manifest_sha256": common.file_sha256(args.manifest),
                                   "include_positive": args.include_positive, "calls": 0})
        rows = read_rows(directory, "results.jsonl")
        validate_records(rows, config, jobs)
        done = {row["sample_id"] for row in rows}
        verifier = None
        for job in jobs:
            if job["sample_id"] in done:
                continue
            try:
                if verifier is None:
                    verifier = QwenQualityVerifier(device=args.device, clip_device="cpu")
                row = verify(job, verifier, config)
            except Exception:
                publish(directory, rows, len(jobs), 0, "stopped_on_verification_error")
                raise
            rows.append(row)
            summary = publish(directory, rows, len(jobs), 0)
            print(f"[reuse {len(rows)}/{len(jobs)}] {job['condition']} {row['status']} accepted={summary['accepted']}", flush=True)
        print(json.dumps(publish(directory, rows, len(jobs), 0, "complete"), ensure_ascii=False), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "import-existing", "status"):
        child = sub.add_parser(command, allow_abbrev=False)
        child.add_argument("--run-dir", type=Path, required=True)
        if command != "status":
            child.add_argument("--device", default="cuda")
            child.add_argument("--cpu-threads", type=int, default=4,
                               help="Bound CPU CLIP/PyTorch parallelism to conserve host memory")
        if command == "generate":
            child.add_argument("--max-calls", type=int, default=1000)
            child.add_argument("--qwen-url", default="http://127.0.0.1:8001/generate")
            child.add_argument("--request-timeout", type=float, default=600)
        if command == "import-existing":
            child.add_argument("--manifest", type=Path, required=True)
            child.add_argument("--include-positive", action="store_true")
    args = parser.parse_args(argv)
    args.run_dir = args.run_dir.expanduser().resolve()
    if args.command != "status" and args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")
    if args.command == "generate":
        if not 1 <= args.max_calls <= 1000 or args.request_timeout <= 0:
            parser.error("Call budget must be 1..1000 and timeout must be positive")
        url = urlparse(args.qwen_url)
        if url.hostname not in {"127.0.0.1", "localhost", "::1"} or url.scheme != "http":
            parser.error("Only the local Qwen service can read source paths")
    if args.command == "import-existing":
        args.manifest = args.manifest.expanduser().resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command != "status":
        import torch
        torch.set_num_threads(args.cpu_threads)
    if args.command == "generate":
        generate(args)
    elif args.command == "import-existing":
        import_existing(args)
    else:
        print((args.run_dir / "summary.json").read_text())


if __name__ == "__main__":
    main()
