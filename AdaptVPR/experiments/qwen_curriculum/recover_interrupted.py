"""Recover an interrupted Qwen call from its saved service image, without HTTP.

The default ``audit`` command is read-only. ``recover`` only appends a sealed
unverified generated row; normal generation resume performs its quality checks.
A sidecar proves source-input path, seed and dimensions. It does not prove the
prompt or service identity; that limitation remains explicit in provenance.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import re
import sys

from PIL import Image

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import common


def _sealed(row, field):
    body = {key: value for key, value in row.items() if key != field}
    return {**body, field: common.fingerprint(body)}


def _check_seal(row, field):
    if row.get(field) != _sealed(row, field)[field]:
        raise ValueError(f"Invalid {field} checksum")


def _rows(path):
    return common.read_jsonl(path) if path.exists() else []


def _image(path, digest=None, dimensions=None):
    path = Path(path)
    if not path.is_file() or (digest is not None and common.file_sha256(path) != digest):
        raise ValueError(f"Image missing or changed: {path}")
    with Image.open(path) as image:
        image.load()
        size = list(image.size)
    if dimensions is not None and size != dimensions:
        raise ValueError(f"Image dimensions changed: {path}")
    return size


def _load(directory):
    config = json.loads((directory / "execution_config.json").read_text())
    _check_seal(config, "fingerprint")
    if config.get("stage") != "generate" or config.get("run_dir") != str(directory):
        raise ValueError("Expected the original frozen generation directory")
    implementation = config.get("implementation_sha256", {})
    if not implementation:
        raise ValueError("Missing frozen implementation hashes")
    for relative, expected in implementation.items():
        path = (common.ADAPTVPR_ROOT / relative).resolve()
        if not path.is_relative_to(common.ADAPTVPR_ROOT.resolve()) or common.file_sha256(path) != expected:
            raise ValueError(f"Frozen implementation changed: {relative}")
    service = config.get("service", {})
    if (service.get("model_id") != "Qwen/Qwen-Image-Edit-2511"
            or service.get("source_modified") is not False
            or service.get("canvas_policy") != "source_aspect_v1"
            or service.get("sampling", {}).get("infer_steps") != 4
            or service.get("sampling", {}).get("guidance_scale") != 1.0):
        raise ValueError("Frozen service differs from the Qwen source-aspect baseline")
    plan_config = json.loads((directory / "plan_config.json").read_text())
    _check_seal(plan_config, "fingerprint")
    if (plan_config["fingerprint"] != config["plan_fingerprint"]
            or common.file_sha256(directory / "plan.jsonl") != plan_config["plan_sha256"]):
        raise ValueError("Frozen plan differs from the execution")
    plans = _rows(directory / "plan.jsonl")
    if len(plans) != plan_config["num_images"]:
        raise ValueError("Plan length differs from its frozen budget")
    jobs = {}
    for job in plans:
        _check_seal(job, "record_sha256")
        sample_id = job["sample_id"]
        if not re.fullmatch(r"qwen_[0-9a-f]{24}", sample_id) or sample_id in jobs:
            raise ValueError("Unsafe or duplicate planned sample_id")
        if job["condition"] not in {"night", "snow", "fog", "rain"}:
            raise ValueError("Unexpected planned weather condition")
        _image(job["source_path"], job["source_sha256"], job["source_dimensions"])
        jobs[sample_id] = job
    attempts = _rows(directory / "attempts.jsonl")
    attempted = set()
    for number, row in enumerate(attempts, 1):
        _check_seal(row, "attempt_sha256")
        sample_id = row["sample_id"]
        if (sample_id not in jobs or sample_id in attempted or row["call_number"] != number
                or row["execution_fingerprint"] != config["fingerprint"]
                or row["record_sha256"] != jobs[sample_id]["record_sha256"]):
            raise ValueError("Durable call ledger differs from the frozen job")
        attempted.add(sample_id)
    if len(attempts) > config["max_calls"]:
        raise ValueError("Durable call ledger exceeds the frozen budget")
    saved = {}
    for name in ("generated.jsonl", "results.jsonl"):
        rows = _rows(directory / name)
        ids = set()
        for row in rows:
            _check_seal(row, "result_sha256")
            sample_id = row["sample_id"]
            if (sample_id not in attempted or sample_id in ids
                    or row["execution_fingerprint"] != config["fingerprint"]
                    or any(row.get(key) != value for key, value in jobs[sample_id].items())):
                raise ValueError("Saved image record differs from the frozen job")
            ids.add(sample_id)
            for path_key, hash_key in (("output_path", "output_sha256"), ("raw_output_path", "raw_output_sha256")):
                if row.get(path_key):
                    _image(row[path_key], row[hash_key])
        saved[name] = rows
    generated_ids = {row["sample_id"] for row in saved["generated.jsonl"]}
    result_ids = {row["sample_id"] for row in saved["results.jsonl"]}
    return config, jobs, attempted - generated_ids - result_ids, saved


def _validate_metadata(metadata, directory, service_dir, job, config):
    sample_id = job["sample_id"]
    raw = Path(metadata.get("result_path", "")).resolve()
    sidecar = Path(metadata.get("metadata_path", "")).resolve()
    if (metadata.get("source_path") != str(directory / "source_inputs" / f"{sample_id}.png")
            or metadata.get("seed") != job["seed"]
            or raw.parent != service_dir or sidecar != raw.with_suffix(".json")):
        raise ValueError("Recovery sidecar source, seed or owned paths differ")
    canvas = config["service"].get("canvas", {})
    if (canvas.get("target_shape_order") != "height,width"
            or canvas.get("rounding") != "nearest_multiple"
            or canvas.get("unsupported_aspect_ratio") != "reject"
            or any(type(canvas.get(key)) is not int or canvas[key] <= 0
                   for key in ("target_pixels", "multiple", "min_side", "max_side"))):
        raise ValueError("Missing or invalid frozen source-aspect canvas definition")
    width, height = job["source_dimensions"]
    if max(width, height) / min(width, height) > canvas["max_side"] / canvas["min_side"]:
        raise ValueError("Frozen source aspect ratio is unsupported")
    scale = min(math.sqrt(canvas["target_pixels"] / (width * height)), canvas["max_side"] / max(width, height))
    target = [max(canvas["min_side"], min(canvas["max_side"],
                  round(side * scale / canvas["multiple"]) * canvas["multiple"])) for side in (height, width)]
    if (metadata.get("canvas_policy") != "source_aspect_v1"
            or metadata.get("source_dimensions") != job["source_dimensions"]
            or metadata.get("target_shape") != target
            or metadata.get("raw_dimensions") != target[::-1]):
        raise ValueError("Matching service sidecar dimensions or canvas differ from frozen policy")


def _candidate(directory, service_dir, job, config):
    sample_id = job["sample_id"]
    input_path = str(directory / "source_inputs" / f"{sample_id}.png")
    intent_path = directory / "recovery_intents" / f"{sample_id}.json"
    if intent_path.exists():
        intent = json.loads(intent_path.read_text())
        _check_seal(intent, "intent_sha256")
        if (intent["execution_fingerprint"] != config["fingerprint"]
                or intent["record_sha256"] != job["record_sha256"]):
            raise ValueError("Recovery intent belongs to another job")
        metadata = intent["metadata"]
        _validate_metadata(metadata, directory, service_dir, job, config)
        source_raw = Path(intent["source_raw_path"])
        source_metadata = Path(intent["source_metadata_path"])
        if str(source_raw) != metadata["result_path"] or str(source_metadata) != metadata["metadata_path"]:
            raise ValueError("Recovery intent source paths differ from its sidecar")
        target_raw = directory / "raw" / f"{sample_id}.png"
        target_metadata = target_raw.with_suffix(".json")
        raw_path = source_raw if source_raw.exists() else target_raw
        metadata_path = source_metadata if source_metadata.exists() else target_metadata
        _image(raw_path, intent["raw_sha256"], metadata["raw_dimensions"])
        if common.file_sha256(metadata_path) != intent["metadata_sha256"]:
            raise ValueError("Recovery sidecar changed after reservation")
        return intent, raw_path, metadata_path
    matches = []
    for path in service_dir.glob("*.json"):
        metadata = json.loads(path.read_text())
        if metadata.get("source_path") != input_path or metadata.get("seed") != job["seed"]:
            continue
        raw_path = Path(metadata.get("result_path", "")).resolve()
        if raw_path.parent != service_dir or raw_path != path.with_suffix(".png"):
            raise ValueError("Matching service sidecar points outside its owned image")
        if metadata.get("metadata_path") != str(path):
            raise ValueError("Matching service sidecar metadata path differs")
        _validate_metadata(metadata, directory, service_dir, job, config)
        _image(raw_path, dimensions=metadata["raw_dimensions"])
        matches.append(_sealed({"sample_id": sample_id, "record_sha256": job["record_sha256"],
            "execution_fingerprint": config["fingerprint"], "source_raw_path": str(raw_path),
            "source_metadata_path": str(path), "raw_sha256": common.file_sha256(raw_path),
            "metadata_sha256": common.file_sha256(path), "metadata": metadata}, "intent_sha256"))
    if len(matches) > 1:
        raise ValueError(f"Multiple matching service outputs for {sample_id}; refusing ambiguous recovery")
    if not matches:
        return None
    intent = matches[0]
    return intent, Path(intent["source_raw_path"]), Path(intent["source_metadata_path"])


@contextmanager
def _lock(directory):
    with (directory / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Generation worker owns this run; recover only while stopped") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def recover_interrupted(directory, service_dir, apply=False):
    directory, service_dir = Path(directory).resolve(), Path(service_dir).resolve()
    if apply:
        with _lock(directory):
            return _recover(directory, service_dir, True)
    return _recover(directory, service_dir, False)


def _recover(directory, service_dir, apply):
    config, jobs, pending, saved = _load(directory)
    generated = saved["generated.jsonl"]
    report = {"mode": "recover" if apply else "audit", "run_dir": str(directory),
              "http_calls": 0, "quality_checks": 0, "results_changed": False,
              "reserved_without_saved_record": sorted(pending), "recoverable": [], "recovered": []}
    # Discover and validate every candidate before making any file changes.
    candidates = [(jobs[sid], _candidate(directory, service_dir, jobs[sid], config)) for sid in sorted(pending)]
    for job, candidate in candidates:
        if candidate is None:
            continue
        sample_id = job["sample_id"]
        intent, raw, metadata_path = candidate
        report["recoverable"].append(sample_id)
        if not apply:
            continue
        common.write_json(directory / "recovery_intents" / f"{sample_id}.json", intent)
        target_raw = directory / "raw" / f"{sample_id}.png"
        target_raw.parent.mkdir(exist_ok=True)
        target_metadata = target_raw.with_suffix(".json")
        for source, destination in ((raw, target_raw), (metadata_path, target_metadata)):
            if source != destination:
                if destination.exists():
                    raise ValueError(f"Recovery would overwrite an existing file: {destination}")
                source.replace(destination)
        output_path = directory / "images" / f"{sample_id}.png"
        output_path.parent.mkdir(exist_ok=True)
        temporary = output_path.with_suffix(".recovery.tmp.png")
        with Image.open(target_raw) as image:
            image.convert("RGB").resize(tuple(job["source_dimensions"]), Image.Resampling.LANCZOS).save(temporary)
        temporary.replace(output_path)
        row = _sealed({**job, "execution_fingerprint": config["fingerprint"],
            "output_path": str(output_path), "output_sha256": common.file_sha256(output_path),
            "raw_output_path": str(target_raw), "raw_output_sha256": intent["raw_sha256"],
            "origin": "new_hard_source_qwen_recovered_from_interrupted_call",
            "recovery": {"version": 1, "recovered_utc": datetime.now(timezone.utc).isoformat(),
                "proof": "exact unique service sidecar source-input path and seed; frozen job/ledger; image dimensions and hashes",
                "quality_status": "unverified; normal generation resume must evaluate this image",
                "sidecar_does_not_prove": ["prompt", "service_identity"],
                "payload_context": "original frozen worker defines the request; original frozen execution records service identity",
                "service_metadata_as_saved": intent["metadata"],
                "service_metadata_sha256": intent["metadata_sha256"],
                "intent_sha256": intent["intent_sha256"]}}, "result_sha256")
        generated.append(row)
        common.write_jsonl(directory / "generated.jsonl", generated)
        report["recovered"].append(sample_id)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("audit", "recover"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--service-dir", type=Path,
                        default=common.ADAPTVPR_ROOT / "tmp/service_outputs/lightx2v")
    args = parser.parse_args(argv)
    print(json.dumps(recover_interrupted(args.run_dir, args.service_dir, apply=args.command == "recover"),
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
