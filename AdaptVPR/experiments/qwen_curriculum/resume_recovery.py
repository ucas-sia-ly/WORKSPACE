"""Copy a saved interrupted Qwen image into an independent zero-call recovery.

The original plan, execution, attempt ledger, failed results and image bytes are
read-only.  A recovered image still needs the original quality verifier.  The
service sidecar proves the source-input path, seed and canvas, but does not prove
the prompt or service identity; that limitation is retained in provenance.
"""
from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import warnings

from PIL import Image

from experiments.qwen_curriculum import common, run


def _sealed(body, field="fingerprint"):
    body = {key: value for key, value in body.items() if key != field}
    return {**body, field: common.fingerprint(body)}


def _check_seal(value, field="fingerprint"):
    if value.get(field) != _sealed(value, field)[field]:
        raise ValueError(f"Invalid recovery {field} checksum")


def _sync_directory(directory):
    handle = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def _freeze_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Immutable recovery configuration changed: {path}")
    else:
        common.write_json(path, value)
        _sync_directory(path.parent)


def _freeze_rows(path, rows):
    if path.exists():
        if common.read_jsonl(path) != rows:
            raise ValueError(f"Immutable recovery plan changed: {path}")
    else:
        common.write_jsonl(path, rows)
        _sync_directory(path.parent)


def _png(path, digest=None, dimensions=None):
    path = Path(path)
    if not path.is_file() or (digest is not None and common.file_sha256(path) != digest):
        raise ValueError(f"Recovery image missing or changed: {path}")
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG":
            raise ValueError(f"Recovery raw is not a PNG: {path}")
        size = list(image.size)
    if dimensions is not None and size != dimensions:
        raise ValueError(f"Recovery image dimensions differ: {path}")
    return size


def _parent(directory, job, execution, attempt):
    saved_execution = json.loads((directory / "execution_config.json").read_text())
    _check_seal(saved_execution)
    if saved_execution != execution or execution.get("run_dir") != str(directory):
        raise ValueError("Recovery parent execution changed")
    if execution.get("stage") != "generate":
        raise ValueError("Recover only an original generation execution")
    service = execution.get("service", {})
    if (service.get("model_id") != "Qwen/Qwen-Image-Edit-2511"
            or service.get("source_modified") is not False
            or service.get("canvas_policy") != "source_aspect_v1"
            or service.get("sampling", {}).get("infer_steps") != 4
            or service.get("sampling", {}).get("guidance_scale") != 1.0):
        raise ValueError("Recovery parent service differs from the original Qwen configuration")
    implementation = execution.get("implementation_sha256", {})
    if not implementation:
        raise ValueError("Missing original frozen generation implementation hashes")
    for relative, digest in implementation.items():
        path = (common.ADAPTVPR_ROOT / relative).resolve()
        if not path.is_relative_to(common.ADAPTVPR_ROOT.resolve()) or common.file_sha256(path) != digest:
            raise ValueError(f"Frozen generation implementation changed: {relative}")
    plan_config, jobs = run.load_plan(directory)
    if plan_config["fingerprint"] != execution["plan_fingerprint"]:
        raise ValueError("Recovery parent plan differs from its execution")
    planned = {row["sample_id"]: row for row in jobs}
    if planned.get(job.get("sample_id")) != job:
        raise ValueError("Recovery job differs from its original sealed plan")
    attempts = run.read_rows(directory, "attempts.jsonl")
    run.validate_attempts(attempts, jobs, execution)
    if attempt not in attempts or attempt.get("sample_id") != job["sample_id"]:
        raise ValueError("Recovery has no exact original durable attempt")
    results = run.read_rows(directory, "results.jsonl")
    generated = run.read_rows(directory, "generated.jsonl")
    run.validate_records(results, execution, jobs)
    run.validate_records(generated, execution, jobs)
    if any(row["sample_id"] == job["sample_id"] for row in generated):
        raise ValueError("Original generated row already exists; verify it without recovery")
    error = next((row for row in results if row["sample_id"] == job["sample_id"]), None)
    if error is not None and error.get("status") != "generation_error":
        raise ValueError("Do not recover or regenerate an already evaluated image")
    return plan_config, error


def _warning(notes, path, reason):
    note = {"path": str(path), "reason": reason}
    notes.append(note)
    warnings.warn(f"Interrupted Qwen recovery skipped {path}: {reason}", RuntimeWarning, stacklevel=3)


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


def _discover(directory, job, execution, attempt, service_dir):
    notes = []
    local_raw = directory / "raw" / f"{job['sample_id']}.png"
    local_metadata = local_raw.with_suffix(".json")
    sidecars = list(sorted(service_dir.glob("*.json")))
    if local_metadata.exists():
        sidecars.append(local_metadata)
    matches = {}
    for path in sidecars:
        try:
            metadata = json.loads(path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            _warning(notes, path, f"incomplete/unreadable sidecar: {type(exc).__name__}")
            continue
        if not isinstance(metadata, dict):
            _warning(notes, path, "sidecar is not a JSON object")
            continue
        if (metadata.get("source_path") != str(directory / "source_inputs" / f"{job['sample_id']}.png")
                or metadata.get("seed") != job["seed"]):
            continue
        _validate_metadata(metadata, directory, service_dir, job, execution)
        original_metadata = Path(metadata["metadata_path"])
        if path not in {original_metadata, local_metadata}:
            raise ValueError("Matching recovery sidecar is outside its owned metadata path")
        key = metadata["metadata_path"]
        if key in matches and matches[key][1] != metadata:
            raise ValueError("Conflicting copies of a matching recovery sidecar")
        # A sidecar may have moved after raw or may still be in the service
        # directory while raw is already in its stage.  Identical sidecar copies
        # for one original UUID are one proof, distinct UUIDs are ambiguous.
        matches[key] = (path, metadata)
    if len(matches) > 1:
        raise ValueError(f"Multiple matching service outputs for {job['sample_id']}; refusing ambiguous recovery")
    if not matches:
        if local_raw.exists():
            _warning(notes, local_raw, "raw has no matching source/seed/canvas sidecar")
        return None, notes
    metadata_path, metadata = next(iter(matches.values()))
    service_raw = Path(metadata["result_path"])
    raw_candidates = [path for path in (service_raw, local_raw) if path.exists()]
    if len(raw_candidates) > 1:
        raise ValueError("Multiple raw paths exist for one interrupted generation")
    if not raw_candidates:
        _warning(notes, metadata_path, "matching sidecar has no saved raw PNG")
        return None, notes
    raw = raw_candidates[0]
    try:
        _png(raw, dimensions=metadata["raw_dimensions"])
    except (OSError, ValueError) as exc:
        _warning(notes, raw, f"incomplete/invalid raw PNG: {exc}")
        return None, notes
    body = {"schema_version": 1, "sample_id": job["sample_id"], "record_sha256": job["record_sha256"],
        "parent_run_dir": str(directory), "parent_execution_fingerprint": execution["fingerprint"],
        "parent_attempt_sha256": attempt["attempt_sha256"], "service_dir": str(service_dir),
        "source_raw_path": str(raw), "source_metadata_path": str(metadata_path),
        "raw_sha256": common.file_sha256(raw), "metadata_sha256": common.file_sha256(metadata_path),
        "metadata": metadata, "warnings": notes, "http_calls": 0}
    return _sealed(body, "intent_sha256"), notes


def _validate_intent(intent, directory, job, execution, attempt, service_dir):
    _check_seal(intent, "intent_sha256")
    expected = {"sample_id": job["sample_id"], "record_sha256": job["record_sha256"],
        "parent_run_dir": str(directory), "parent_execution_fingerprint": execution["fingerprint"],
        "parent_attempt_sha256": attempt["attempt_sha256"], "service_dir": str(service_dir), "http_calls": 0}
    if any(intent.get(key) != value for key, value in expected.items()):
        raise ValueError("Recovery intent belongs to a different original attempt")
    _validate_metadata(intent["metadata"], directory, service_dir, job, execution)
    raw_paths = {Path(intent["metadata"]["result_path"]), directory / "raw" / f"{job['sample_id']}.png"}
    metadata_paths = {Path(intent["metadata"]["metadata_path"]),
                      directory / "raw" / f"{job['sample_id']}.json"}
    if Path(intent["source_raw_path"]) not in raw_paths or Path(intent["source_metadata_path"]) not in metadata_paths:
        raise ValueError("Recovery intent copies an unowned image or metadata path")


def _copy_atomic(source, destination, digest):
    if destination.exists():
        if common.file_sha256(destination) != digest:
            raise ValueError(f"Copied recovery file changed: {destination}")
        return
    if not source.is_file() or common.file_sha256(source) != digest:
        raise ValueError(f"Recovery source missing or changed after intent: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".recovery.", delete=False) as handle:
            temporary = Path(handle.name)
            with source.open("rb") as reader:
                for chunk in iter(lambda: reader.read(1 << 20), b""):
                    handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if common.file_sha256(temporary) != digest:
            raise ValueError("Recovery source changed while copying")
        temporary.replace(destination)
        _sync_directory(destination.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _prepare(output_dir, directory, job, execution, attempt, parent_plan, error, intent):
    if (intent.get("original_error_result_sha256") is not None
            and (error is None or error["result_sha256"] != intent["original_error_result_sha256"])):
        raise ValueError("Original recorded generation failure changed after recovery intent")
    _freeze_rows(output_dir / "plan.jsonl", [job])
    _freeze_rows(output_dir / "sources.jsonl", [job])
    plan_body = {key: value for key, value in parent_plan.items() if key != "fingerprint"}
    plan_body.update(output_dir=str(output_dir), sources=str(output_dir / "sources.jsonl"),
        sources_sha256=common.file_sha256(output_dir / "sources.jsonl"), source_manifest_rows=1,
        num_images=1, plan_sha256=common.file_sha256(output_dir / "plan.jsonl"),
        domain_quotas={job["condition"]: 1}, parent_plan_fingerprint=parent_plan["fingerprint"],
        allocation="exact_original_job_saved_raw_recovery; no new weather allocation")
    plan_config = _sealed(plan_body)
    _freeze_json(output_dir / "plan_config.json", plan_config)
    provenance = {"parent_run_dir": str(directory), "parent_execution_fingerprint": execution["fingerprint"],
        "parent_attempt_sha256": attempt["attempt_sha256"], "intent_sha256": intent["intent_sha256"],
        "http_calls": 0}
    # An unknown original attempt may gain a failure row when the old worker
    # resumes.  That must not change this already frozen zero-call recovery.
    if intent.get("original_error_result_sha256") is not None:
        provenance["original_error_result_sha256"] = intent["original_error_result_sha256"]
    body = {key: value for key, value in execution.items() if key != "fingerprint"}
    body.update(run_dir=str(output_dir), plan_fingerprint=plan_config["fingerprint"], max_calls=1,
        stage="saved_raw_recovery", call_policy="zero POST; copied saved raw from an original durable reservation",
        recovery_provenance=provenance)
    recovery_execution = _sealed(body)
    _freeze_json(output_dir / "execution_config.json", recovery_execution)
    frozen_files = [output_dir / "intent.json", output_dir / "plan.jsonl", output_dir / "plan_config.json",
        output_dir / "sources.jsonl", directory / "execution_config.json", directory / "plan_config.json", directory / "plan.jsonl"]
    config = _sealed({"schema_version": 1, "stage": "zero_call_saved_raw_recovery", **provenance,
        "sample_id": job["sample_id"], "record_sha256": job["record_sha256"],
        "execution_fingerprint": recovery_execution["fingerprint"], "plan_fingerprint": plan_config["fingerprint"],
        "files_sha256": {str(path): common.file_sha256(path) for path in frozen_files},
        "recovery_implementation_sha256": common.file_sha256(Path(__file__))})
    _freeze_json(output_dir / "recovery_config.json", config)
    return recovery_execution


def validate_recovery(output_dir, job=None, parent_execution=None, parent_attempt=None):
    """Validate one independent recovery and return its generated row/execution."""
    output_dir = Path(output_dir).resolve()
    config = json.loads((output_dir / "recovery_config.json").read_text())
    _check_seal(config)
    if config.get("stage") != "zero_call_saved_raw_recovery" or config.get("http_calls") != 0:
        raise ValueError("Expected a zero-call saved-raw recovery")
    if config["recovery_implementation_sha256"] != common.file_sha256(Path(__file__)):
        raise ValueError("Frozen recovery implementation changed")
    for path, digest in config["files_sha256"].items():
        if common.file_sha256(Path(path)) != digest:
            raise ValueError(f"Frozen recovery input changed: {path}")
    plan_config, jobs = run.load_plan(output_dir)
    if len(jobs) != 1 or (job is not None and jobs[0] != job):
        raise ValueError("Recovery plan differs from its single original job")
    job = jobs[0]
    directory = Path(config["parent_run_dir"])
    actual_execution = json.loads((directory / "execution_config.json").read_text())
    actual_attempt = next((row for row in run.read_rows(directory, "attempts.jsonl")
                           if row.get("attempt_sha256") == config["parent_attempt_sha256"]), None)
    if (actual_attempt is None or (parent_execution is not None and parent_execution != actual_execution)
            or (parent_attempt is not None and parent_attempt != actual_attempt)):
        raise ValueError("Original recovery execution or attempt changed")
    parent_plan, error = _parent(directory, job, actual_execution, actual_attempt)
    intent = json.loads((output_dir / "intent.json").read_text())
    _validate_intent(intent, directory, job, actual_execution, actual_attempt, Path(intent["service_dir"]))
    execution = _prepare(output_dir, directory, job, actual_execution, actual_attempt, parent_plan, error, intent)
    if (execution["fingerprint"] != config["execution_fingerprint"]
            or plan_config["fingerprint"] != config["plan_fingerprint"]
            or config["sample_id"] != job["sample_id"] or config["record_sha256"] != job["record_sha256"]):
        raise ValueError("Recovery execution or plan differs from its configuration")
    if run.read_rows(output_dir, "attempts.jsonl"):
        raise ValueError("Zero-call recovery cannot have a fabricated POST reservation")
    rows = run.read_rows(output_dir, "generated.jsonl")
    if len(rows) != 1:
        raise ValueError("Recovery has not saved its single generated image")
    run.validate_records(rows, execution, jobs)
    row = rows[0]
    if row.get("raw_output_sha256") != intent["raw_sha256"]:
        raise ValueError("Recovered raw differs from the frozen recovery intent")
    if row.get("raw_output_path") != str(output_dir / "raw" / f"{job['sample_id']}.png"):
        raise ValueError("Recovered raw is outside its independent recovery directory")
    if row.get("output_path") != str(output_dir / "images" / f"{job['sample_id']}.png"):
        raise ValueError("Recovered normalized image is outside its recovery directory")
    _png(row["raw_output_path"], intent["raw_sha256"], intent["metadata"]["raw_dimensions"])
    _png(row["output_path"], row["output_sha256"], job["source_dimensions"])
    metadata_copy = output_dir / "raw" / f"{job['sample_id']}.json"
    if common.file_sha256(metadata_copy) != intent["metadata_sha256"]:
        raise ValueError("Copied recovery sidecar changed")
    results = run.read_rows(output_dir, "results.jsonl")
    run.validate_records(results, execution, jobs)
    for result in results:
        if (any(result.get(key) != value for key, value in row.items() if key != "result_sha256")
                or result.get("status") not in {"passed", "rejected"}):
            raise ValueError("Recovery quality result differs from its saved generated image")
    return row, execution


def recover(directory, job, execution, attempt, service_dir, output_dir, *, parent_locked=False):
    """Recover a reserved job without POST, including a recorded failure.

    The caller must stop Qwen before invoking this function so an in-flight
    request cannot create another candidate after discovery.  Both source and
    recovery workers are locked while validating/copying.  Missing or partial
    output returns ``None`` with an audit record, allowing an independent retry.
    ``parent_locked=True`` is reserved for a controller already holding this
    exact source directory's exclusive worker lock for the entire operation.
    """
    directory, service_dir, output_dir = [Path(path).resolve() for path in (directory, service_dir, output_dir)]
    if output_dir == directory or output_dir == service_dir:
        raise ValueError("Use an independent recovery output directory")
    parent_lock = nullcontext() if parent_locked else run.run_lock(directory)
    with parent_lock, run.run_lock(output_dir):
        parent_plan, error = _parent(directory, job, execution, attempt)
        if (output_dir / "generated.jsonl").exists():
            return validate_recovery(output_dir, job, execution, attempt)[0]
        intent_path = output_dir / "intent.json"
        if intent_path.exists():
            intent = json.loads(intent_path.read_text())
            _validate_intent(intent, directory, job, execution, attempt, service_dir)
        else:
            intent, notes = _discover(directory, job, execution, attempt, service_dir)
            common.write_json(output_dir / "audit.json", _sealed({"sample_id": job["sample_id"],
                "parent_execution_fingerprint": execution["fingerprint"], "http_calls": 0,
                "recoverable": intent is not None, "warnings": notes,
                "audited_utc": datetime.now(timezone.utc).isoformat()}))
            if intent is None:
                return None
            if error is not None:
                intent = _sealed({**intent, "original_error_result_sha256": error["result_sha256"]}, "intent_sha256")
            _freeze_json(intent_path, intent)
        recovery_execution = _prepare(output_dir, directory, job, execution, attempt, parent_plan, error, intent)
        raw = output_dir / "raw" / f"{job['sample_id']}.png"
        metadata = raw.with_suffix(".json")
        _copy_atomic(Path(intent["source_raw_path"]), raw, intent["raw_sha256"])
        _copy_atomic(Path(intent["source_metadata_path"]), metadata, intent["metadata_sha256"])
        _png(raw, intent["raw_sha256"], intent["metadata"]["raw_dimensions"])
        output = output_dir / "images" / f"{job['sample_id']}.png"
        output.parent.mkdir(exist_ok=True)
        temporary = output.with_suffix(".tmp.png")
        try:
            with Image.open(raw) as image:
                image.convert("RGB").resize(tuple(job["source_dimensions"]), Image.Resampling.LANCZOS).save(temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            temporary.replace(output)
            _sync_directory(output.parent)
        finally:
            temporary.unlink(missing_ok=True)
        row = run.sealed({**job, "execution_fingerprint": recovery_execution["fingerprint"],
            "output_path": str(output), "output_sha256": common.file_sha256(output),
            "raw_output_path": str(raw), "raw_output_sha256": intent["raw_sha256"],
            "origin": "saved_interrupted_qwen_raw_independent_zero_call_recovery",
            "recovery": {"version": 2, "http_calls": 0,
                "parent_execution_fingerprint": execution["fingerprint"],
                "parent_attempt_sha256": attempt["attempt_sha256"], "intent_sha256": intent["intent_sha256"],
                "proof": "exact unique sidecar source-input path, seed and frozen canvas; original sealed job/attempt; full PNG and hashes",
                "quality_status": "unverified; original frozen quality verifier must evaluate this image",
                "sidecar_does_not_prove": ["prompt", "service_identity"],
                "payload_context": "original frozen worker defines prompt; original execution records service identity",
                "service_metadata_as_saved": intent["metadata"],
                "service_metadata_sha256": intent["metadata_sha256"], "warnings": intent["warnings"]}})
        _freeze_rows(output_dir / "generated.jsonl", [row])
        return validate_recovery(output_dir, job, execution, attempt)[0]
