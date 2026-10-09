"""Resume the 2000-image cohort under an explicitly frozen 48 GiB profile.

Historical ledgers are retained byte for byte. Saved outputs previously marked
as unknown calls are recovered in a separate zero-call stage. New generation
uses a fresh execution directory, rather than rewriting historical settings.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time

import requests
from PIL import Image

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import common, run, recover_interrupted
from experiments.qwen_curriculum.campaign import config_file, unit_state, ACTIVE


def frozen(path, value):
    value = {**value, "fingerprint": common.fingerprint(value)}
    if path.exists() and config_file(path) != value:
        raise ValueError(f"Frozen fast campaign settings changed: {path}")
    if not path.exists():
        common.write_json(path, value)
    return value


def read(directory, name):
    return run.read_rows(Path(directory), name)


def prepare(directory):
    directory = Path(directory).resolve()
    path = directory / "fast_campaign_config.json"
    if path.exists():
        config = config_file(path)
        check_inputs(config)
        return config
    legacy = config_file(directory / "campaign_config.json")
    parent = Path(legacy["parent_run_dir"])
    parent_plan, jobs = run.load_plan(parent)
    execution = config_file(parent / "execution_config.json")
    saved = read(parent, "results.jsonl")
    run.validate_records(saved, execution, jobs)
    run.validate_records(read(parent, "generated.jsonl"), execution, jobs)
    attempted = run.validate_attempts(read(parent, "attempts.jsonl"), jobs, execution)
    if len(saved) != len(jobs) or len(attempted) != len(jobs):
        raise ValueError("Original stage must have finished before changing its runtime profile")
    missing = [row for row in saved if row.get("status") == "generation_error"]
    recovery_inputs = []
    service_dir = common.ADAPTVPR_ROOT / "tmp/service_outputs/lightx2v"
    planned = {row["sample_id"]: row for row in jobs}
    for row in missing:
        candidate = recover_interrupted._candidate(parent, service_dir, planned[row["sample_id"]], execution)
        if candidate is None:
            raise ValueError(f"Missing saved bytes for {row['sample_id']}; no repeat POST is allowed")
        intent, _, _ = candidate
        recovery_inputs.append({"job": planned[row["sample_id"]], "intent": intent,
                                "original_error_result_sha256": row["result_sha256"]})
    recovery = directory / "recovered_saved"
    recovery.mkdir(exist_ok=True)
    common.write_jsonl(recovery / "recovery_inputs.jsonl", recovery_inputs)
    recovery_execution = frozen(recovery / "execution_config.json", {
        **run.execution_definition(argparse.Namespace(run_dir=recovery, device="cuda", cpu_threads=4)),
        "stage": "recover_saved_unknown_calls", "plan_fingerprint": parent_plan["fingerprint"],
        "original_execution_fingerprint": execution["fingerprint"], "max_calls": 0,
        "inputs_sha256": common.file_sha256(recovery / "recovery_inputs.jsonl"),
        "policy": "saved raw bytes only; original error journal remains unchanged; verify without HTTP"})
    original_new = Path(legacy["additional_run_dir"])
    additional = directory / "additional_1000_fast"
    if read(original_new, "attempts.jsonl"):
        raise ValueError("Additional legacy stage already made calls; audit it before changing profile")
    additional.mkdir(exist_ok=True)
    original_config, new_jobs = run.load_plan(original_new)
    common._atomic_write(additional / "plan.jsonl", [(original_new / "plan.jsonl").read_text()])
    new_plan = {key: value for key, value in original_config.items() if key != "fingerprint"}
    new_plan["output_dir"] = str(additional)
    new_plan = frozen(additional / "plan_config.json", new_plan)
    if len(jobs) + len(new_jobs) != legacy["total_images"]:
        raise ValueError("Fast stages differ from the cumulative 2000-image plan")
    config = frozen(path, {
        "schema_version": 1, "stage": "qwen_curriculum_fast_campaign",
        "total_images": legacy["total_images"], "parent_run_dir": str(parent),
        "recovery_run_dir": str(recovery), "generation_run_dir": str(additional),
        "combined_plan": legacy["combined_plan"], "combined_plan_sha256": legacy["combined_plan_sha256"],
        "parent_execution_fingerprint": execution["fingerprint"],
        "new_plan_fingerprint": new_plan["fingerprint"],
        "recovery_execution_fingerprint": recovery_execution["fingerprint"],
        "recovery_ids": [row["sample_id"] for row in missing],
        "parent_files_sha256": {name: common.file_sha256(parent / name) for name in
            ("plan_config.json", "plan.jsonl", "execution_config.json", "results.jsonl", "generated.jsonl", "attempts.jsonl")},
        "recovery_inputs_sha256": recovery_execution["inputs_sha256"],
        "runtime_profile": "resident_bf16_48g", "resident_dit_blocks": 32,
        "domain_quotas": legacy["domain_quotas"], "quality_policy": "unchanged structure and weather checks",
        "images_policy": "reference original image paths; do not duplicate existing images",
        "implementation_sha256": {str(p.resolve()): common.file_sha256(p) for p in
            (Path(__file__), Path(run.__file__), Path(common.__file__), Path(recover_interrupted.__file__),
             common.ADAPTVPR_ROOT / "generation/qwen_resident.py", common.ADAPTVPR_ROOT / "adapters/lightx2v_qwen_image_edit.py")}})
    check_inputs(config)
    publish(directory, config, "prepared")
    return config


def check_inputs(config):
    parent = Path(config["parent_run_dir"])
    for name, digest in config["parent_files_sha256"].items():
        if common.file_sha256(parent / name) != digest:
            raise ValueError(f"Original stage changed: {name}")
    if common.file_sha256(Path(config["combined_plan"])) != config["combined_plan_sha256"]:
        raise ValueError("Combined cohort plan changed")
    if common.file_sha256(Path(config["recovery_run_dir"]) / "recovery_inputs.jsonl") != config["recovery_inputs_sha256"]:
        raise ValueError("Recovery proof changed")
    if config_file(Path(config["generation_run_dir"]) / "plan_config.json")["fingerprint"] != config["new_plan_fingerprint"]:
        raise ValueError("Additional stage plan changed")
    if config_file(Path(config["recovery_run_dir"]) / "execution_config.json")["fingerprint"] != config["recovery_execution_fingerprint"]:
        raise ValueError("Recovery execution changed")
    for path, digest in config["implementation_sha256"].items():
        if common.file_sha256(Path(path)) != digest:
            raise ValueError(f"Fast runtime implementation changed: {path}")


def recover_saved(config):
    directory = Path(config["recovery_run_dir"])
    execution = config_file(directory / "execution_config.json")
    inputs = common.read_jsonl(directory / "recovery_inputs.jsonl")
    generated, results = read(directory, "generated.jsonl"), read(directory, "results.jsonl")
    jobs = [value["job"] for value in inputs]
    run.validate_records(generated, execution, jobs)
    run.validate_records(results, execution, jobs)
    generated_by_id = {row["sample_id"]: row for row in generated}
    done = {row["sample_id"] for row in results}
    verifier = None
    for value in inputs:
        job, intent = value["job"], value["intent"]
        sid = job["sample_id"]
        if sid not in generated_by_id:
            (directory / "raw").mkdir(exist_ok=True)
            raw = directory / "raw" / f"{sid}.png"
            side = raw.with_suffix(".json")
            candidate = raw if raw.exists() else Path(intent["source_raw_path"])
            metadata = side if side.exists() else Path(intent["source_metadata_path"])
            if common.file_sha256(candidate) != intent["raw_sha256"] or common.file_sha256(metadata) != intent["metadata_sha256"]:
                raise ValueError("Saved unknown-call bytes differ from their frozen proof")
            if candidate != raw:
                candidate.replace(raw)
            if metadata != side:
                metadata.replace(side)
            (directory / "images").mkdir(exist_ok=True)
            output = directory / "images" / f"{sid}.png"
            temporary = output.with_suffix(".tmp.png")
            with Image.open(raw) as image:
                image.convert("RGB").resize(tuple(job["source_dimensions"]), Image.Resampling.LANCZOS).save(temporary)
            temporary.replace(output)
            row = run.sealed({**job, "execution_fingerprint": execution["fingerprint"],
                "raw_output_path": str(raw), "raw_output_sha256": intent["raw_sha256"],
                "output_path": str(output), "output_sha256": common.file_sha256(output),
                "origin": "saved_unknown_call_recovered_without_new_generation",
                "recovery": {"original_execution_fingerprint": intent["execution_fingerprint"],
                    "original_error_result_sha256": value["original_error_result_sha256"],
                    "intent_sha256": intent["intent_sha256"], "http_calls": 0,
                    "proof": "unique saved sidecar source path, seed, dimensions and raw bytes",
                    "sidecar_does_not_prove": ["prompt", "service_identity"]}})
            generated.append(row)
            common.write_jsonl(directory / "generated.jsonl", generated)
            generated_by_id[sid] = row
        if sid not in done:
            if verifier is None:
                from experiments.qwen_curriculum.quality import QwenQualityVerifier
                verifier = QwenQualityVerifier(device="cuda", clip_device="cpu")
            results.append(run.verify(generated_by_id[sid], verifier, execution))
            common.write_jsonl(directory / "results.jsonl", results)
            print(f"Recovered and evaluated {sid}; no HTTP call", flush=True)
    return results


def collected(config, name):
    parent = [row for row in read(config["parent_run_dir"], name)
              if row["sample_id"] not in set(config["recovery_ids"])]
    by_id = {row["sample_id"]: row for row in parent}
    for stage in ("recovery_run_dir", "generation_run_dir"):
        for row in read(config[stage], name):
            if row["sample_id"] in by_id:
                raise ValueError("Duplicate successful image in the cumulative cohort")
            by_id[row["sample_id"]] = row
    jobs = common.read_jsonl(Path(config["combined_plan"]))
    if set(by_id) - {row["sample_id"] for row in jobs}:
        raise ValueError("Unplanned saved image in the cumulative cohort")
    output = [by_id[row["sample_id"]] for row in jobs if row["sample_id"] in by_id]
    if len({row["source_sha256"] for row in output}) != len(output):
        raise ValueError("Duplicate source content in the cumulative cohort")
    return output


def publish(directory, config, state, error=None):
    generated, results = collected(config, "generated.jsonl"), collected(config, "results.jsonl")
    accepted = [r for r in results if r.get("passed") and r.get("eligible_for_training")]
    calls = len(read(config["generation_run_dir"], "attempts.jsonl"))
    summary = {"state": state, "target_images": config["total_images"], "generated_images": len(generated),
        "quality_evaluated": len(results), "accepted": len(accepted),
        "rejected": sum(r.get("status") == "rejected" for r in results),
        "generation_errors": sum(r.get("status") == "generation_error" for r in results),
        "new_calls_reserved": calls, "new_generation_calls_remaining": 1000 - calls,
        "recovery_pending": len(config["recovery_ids"]) - len(read(config["recovery_run_dir"], "generated.jsonl")),
        "conditions_generated": dict(Counter(r["condition"] for r in generated)),
        "runtime_profile": config["runtime_profile"], "campaign_fingerprint": config["fingerprint"],
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if error:
        summary["error"] = error
    common.write_jsonl(directory / "generated.jsonl", generated)
    common.write_jsonl(directory / "results.jsonl", results)
    common.write_jsonl(directory / "training_manifest.jsonl", accepted)
    common.write_json(directory / "summary.json", summary)
    return summary


def load_collection(directory, count, *, require_complete=True):
    """Load genuine stage-signed results for the fixed-ratio trainer."""
    directory = Path(directory)
    config = config_file(directory / "fast_campaign_config.json")
    check_inputs(config)
    summary = json.loads((directory / "summary.json").read_text())
    if ((require_complete and summary.get("state") != "complete")
            or summary.get("campaign_fingerprint") != config["fingerprint"]):
        raise ValueError("Cumulative generation is incomplete; finish generation before training")
    parent_plan, parent_jobs = run.load_plan(Path(config["parent_run_dir"]))
    new_plan, new_jobs = run.load_plan(Path(config["generation_run_dir"]))
    for stage, jobs in (("parent_run_dir", parent_jobs), ("recovery_run_dir", parent_jobs), ("generation_run_dir", new_jobs)):
        stage_dir = Path(config[stage])
        execution = config_file(stage_dir / "execution_config.json")
        run.validate_records(read(stage_dir, "results.jsonl"), execution, jobs)
        run.validate_records(read(stage_dir, "generated.jsonl"), execution, jobs)
        if stage != "recovery_run_dir":
            attempted = run.validate_attempts(read(stage_dir, "attempts.jsonl"), jobs, execution)
            if not {r["sample_id"] for r in read(stage_dir, "generated.jsonl")} <= attempted:
                raise ValueError("Saved generation has no original call reservation")
    results = collected(config, "results.jsonl")
    expected = {r["sample_id"] for r in parent_jobs + new_jobs}
    if (len(results) != config["total_images"] or {r["sample_id"] for r in results} != expected
            or any(not r.get("output_path") or r.get("status") not in {"passed", "rejected"} for r in results)):
        raise ValueError("The cumulative cohort lacks an evaluated image for every source")
    if results != read(directory, "results.jsonl"):
        raise ValueError("Combined results differ from their original stage journals")
    if count != config["total_images"]:
        raise ValueError("Use the full fixed 2000-image cohort for this experiment")
    return results, config["fingerprint"], config["combined_plan_sha256"]


def execute(directory, config):
    summary_path = directory / "summary.json"
    if summary_path.exists() and json.loads(summary_path.read_text()).get("state") == "complete":
        load_collection(directory, config["total_images"])
        print("Cumulative cohort already complete; no generation service started", flush=True)
        return
    if unit_state("qwen-ratio-700-vpr-scratch.service") in ACTIVE:
        raise RuntimeError("Original training is still active; generation stays stopped")
    import torch
    torch.set_num_threads(4)
    publish(directory, config, "recovering_saved_images")
    recover_saved(config)
    additional = Path(config["generation_run_dir"])
    if (additional / "execution_config.json").exists():
        report = recover_interrupted.recover_interrupted(
            additional, common.ADAPTVPR_ROOT / "tmp/service_outputs/lightx2v", apply=True)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    subprocess.run(["systemctl", "--user", "start", "adaptvpr-lightx2v.service"], check=True)
    session = requests.Session()
    session.trust_env = False
    for _ in range(180):
        try:
            health = session.get("http://127.0.0.1:8001/health", timeout=5).json()
            if health.get("error"):
                raise RuntimeError(health["error"])
            if health.get("generator_ready"):
                run.service_identity(health)
                if health.get("performance", {}).get("resident_dit_blocks") != config["resident_dit_blocks"]:
                    raise ValueError("Qwen is not running the frozen 48 GiB performance profile")
                break
        except requests.ConnectionError:
            pass
        time.sleep(2)
    else:
        raise RuntimeError("Qwen startup timed out")
    command = [sys.executable, "-u", str(common.CURRICULUM_ROOT / "run.py"), "generate",
        "--run-dir", config["generation_run_dir"], "--max-calls", "1000", "--request-timeout", "600",
        "--device", "cuda", "--cpu-threads", "4"]
    with (directory / "fast_generation.log").open("a") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                publish(directory, config, "generating")
                time.sleep(10)
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise
        if child.returncode:
            raise RuntimeError("Generation stopped; see fast_generation.log; no unknown call is retried")
    summary = publish(directory, config, "validating")
    if (summary["generated_images"] != config["total_images"] or summary["quality_evaluated"] != config["total_images"]
            or summary["generation_errors"] or summary["conditions_generated"] != config["domain_quotas"]):
        raise ValueError("Campaign did not reach its exact image and domain targets")
    # Validation reads every image and original stage seal before completion.
    load_collection(directory, config["total_images"], require_complete=False)
    subprocess.run(["systemctl", "--user", "stop", "adaptvpr-lightx2v.service"], check=True)
    summary = publish(directory, config, "complete")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("prepare", "run", "status"))
    parser.add_argument("--output-dir", type=Path, default=common.WORKSPACE_ROOT / "outputs/qwen_curriculum/generation_2000")
    args = parser.parse_args(argv)
    directory = args.output_dir.expanduser().resolve()
    if args.command == "status":
        print((directory / "summary.json").read_text())
        return
    with run.run_lock(directory):
        config = prepare(directory)
        if args.command == "run":
            try:
                execute(directory, config)
            except Exception as exc:
                publish(directory, config, "stopped", f"{type(exc).__name__}: {exc}")
                raise


if __name__ == "__main__":
    main()
