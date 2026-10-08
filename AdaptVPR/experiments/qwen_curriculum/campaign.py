"""Continue frozen Qwen stages to a cumulative image target after training.

Existing generation ledgers and images remain in their original directories.
This coordinator keeps a combined view, waits for the entire fixed training
experiment, and then invokes the unchanged bounded generator serially.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

import requests

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import common

ACTIVE = {"active", "activating", "deactivating", "reloading"}


def config_file(path):
    value = json.loads(path.read_text())
    expected = value.pop("fingerprint")
    if common.fingerprint(value) != expected:
        raise ValueError(f"Configuration checksum mismatch: {path}")
    return {**value, "fingerprint": expected}


def unit_state(unit):
    response = subprocess.run(["systemctl", "--user", "show", unit,
                               "--property=ActiveState", "--value"],
                              check=True, capture_output=True, text=True)
    return response.stdout.strip()


def unit_exit_code(unit):
    response = subprocess.run(["systemctl", "--user", "show", unit,
                               "--property=ExecMainStatus", "--value"],
                              check=True, capture_output=True, text=True)
    return int(response.stdout.strip())


def training_finished(directory, expected_fingerprint):
    path = directory / "comparison.json"
    if not path.exists():
        return False
    report = config_file(path)
    if report.get("state") != "complete" or report.get("experiment_fingerprint") != expected_fingerprint:
        raise ValueError("Training completion report belongs to another experiment")
    progress = json.loads((directory / "progress.json").read_text())
    if progress.get("stage") != "complete":
        return False
    expected_arms = {"generated_8to1", "true_8to1", "generated_4to1", "true_4to1"}
    if set(report.get("checkpoint_sha256", {})) != expected_arms:
        raise ValueError("Training report does not cover all four training arms")
    for arm, digest in report["checkpoint_sha256"].items():
        if common.file_sha256(directory / arm / "checkpoint.pt") != digest:
            raise ValueError(f"Final training checkpoint changed: {arm}")
    return True


def rows(directory, name):
    path = directory / name
    return common.read_jsonl(path) if path.exists() else []


def refresh(directory, config, phase, error=None):
    results, generated, attempts = [], [], []
    provenance = []
    result_ids, generated_ids, source_hashes = set(), set(), set()
    for stage in config["stages"]:
        run_dir = Path(stage["run_dir"])
        current_results = rows(run_dir, "results.jsonl")
        current_generated = rows(run_dir, "generated.jsonl")
        current_attempts = rows(run_dir, "attempts.jsonl")
        for row in current_results:
            if row["sample_id"] in result_ids:
                raise ValueError("Duplicate result across campaign stages")
            result_ids.add(row["sample_id"])
        for row in current_generated:
            if row["sample_id"] in generated_ids or row["source_sha256"] in source_hashes:
                raise ValueError("Duplicate generated source across campaign stages")
            generated_ids.add(row["sample_id"])
            source_hashes.add(row["source_sha256"])
        provenance.append({"run_dir": str(run_dir), "results_offset": len(results),
                           "results_count": len(current_results), "generated_count": len(current_generated),
                           "calls_reserved": len(current_attempts), "plan_fingerprint": stage["plan_fingerprint"]})
        results.extend(current_results)
        generated.extend(current_generated)
        attempts.extend(current_attempts)
    accepted = [row for row in results if row.get("passed") is True and row.get("eligible_for_training") is True]
    summary = {"state": phase, "target_images": config["total_images"],
               "generated_images": len(generated), "quality_evaluated": len(results),
               "accepted": len(accepted), "rejected": sum(row.get("status") == "rejected" for row in results),
               "generation_errors": sum(row.get("status") == "generation_error" for row in results),
               "calls_reserved": len(attempts), "new_images_remaining": max(0, config["total_images"] - len(generated)),
               "conditions_generated": dict(Counter(row["condition"] for row in generated)),
               "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "images_policy": "original stage image paths; no duplicate image copies"}
    if error:
        summary["error"] = error
    common.write_jsonl(directory / "results.jsonl", results)
    common.write_jsonl(directory / "training_manifest.jsonl", accepted)
    common.write_json(directory / "runtime_stage_provenance.json", provenance)
    common.write_json(directory / "summary.json", summary)
    return summary


def freeze_coordinator(args, config):
    training = config_file(args.training_dir / "experiment_config.json")
    from experiments.qwen_curriculum import recover_interrupted
    definition = {"campaign_fingerprint": config["fingerprint"],
                  "training_unit": args.training_unit, "training_dir": str(args.training_dir),
                  "training_fingerprint": training["fingerprint"], "qwen_unit": "adaptvpr-lightx2v.service",
                  "policy": "wait for all four arms and evaluations, recover saved bytes, finish original then additional stage",
                  "implementation_sha256": {str(path.resolve()): common.file_sha256(path) for path in
                       (Path(__file__), Path(common.__file__), Path(recover_interrupted.__file__))}}
    payload = {**definition, "fingerprint": common.fingerprint(definition)}
    path = args.output_dir / "coordinator_config.json"
    if path.exists() and config_file(path) != payload:
        raise ValueError("Coordinator settings changed; use a new campaign directory")
    if not path.exists():
        common.write_json(path, payload)
    return payload


def check_frozen_artifacts(args, config, coordinator):
    if config_file(args.output_dir / "campaign_config.json") != config:
        raise ValueError("Campaign configuration changed while queued")
    for path, digest in coordinator["implementation_sha256"].items():
        if common.file_sha256(Path(path)) != digest:
            raise ValueError("Coordinator implementation changed")
    for relative, digest in config["implementation_sha256"].items():
        if common.file_sha256(common.ADAPTVPR_ROOT / relative) != digest:
            raise ValueError("Extension selection implementation changed")
    if common.file_sha256(Path(config["combined_plan"])) != config["combined_plan_sha256"]:
        raise ValueError("Combined frozen plan changed")
    if common.file_sha256(Path(config["parent_run_dir"]) / "execution_config.json") != config["parent_execution_config_sha256"]:
        raise ValueError("Original generation settings changed")
    for stage in config["stages"]:
        directory = Path(stage["run_dir"])
        plan = config_file(directory / "plan_config.json")
        if (plan["fingerprint"] != stage["plan_fingerprint"] or plan["plan_sha256"] != stage["plan_sha256"]
                or common.file_sha256(directory / "plan.jsonl") != stage["plan_sha256"]):
            raise ValueError("Frozen stage plan changed")
        for relative, digest in plan["implementation_sha256"].items():
            if common.file_sha256(common.ADAPTVPR_ROOT / relative) != digest:
                raise ValueError("Stage planning implementation changed")
    execution = config_file(Path(config["parent_run_dir"]) / "execution_config.json")
    for relative, digest in execution["implementation_sha256"].items():
        if common.file_sha256(common.ADAPTVPR_ROOT / relative) != digest:
            raise ValueError("Original Qwen generation implementation changed")


def wait_for_training(args, config, coordinator):
    while True:
        training = config_file(args.training_dir / "experiment_config.json")
        if training["fingerprint"] != coordinator["training_fingerprint"]:
            raise ValueError("The training experiment changed while generation was queued")
        state = unit_state(args.training_unit)
        if state not in ACTIVE:
            if unit_exit_code(args.training_unit) != 0:
                raise RuntimeError("Training exited unsuccessfully; generation remains stopped")
            if training_finished(args.training_dir, coordinator["training_fingerprint"]):
                return
            raise RuntimeError(f"Training unit is {state or 'missing'} before full completion; generation remains stopped")
        summary = refresh(args.output_dir, config, "waiting_for_training")
        print(f"Waiting for {args.training_unit}; {summary['generated_images']}/{config['total_images']} images retained", flush=True)
        time.sleep(60)


def start_qwen(parent_execution):
    unit = "adaptvpr-lightx2v.service"
    command = subprocess.run(["systemctl", "--user", "show", unit, "--property=ExecStart", "--value"],
                             check=True, capture_output=True, text=True).stdout
    if str(common.ADAPTVPR_ROOT / "adapters/lightx2v_qwen_image_edit.py") not in command:
        raise RuntimeError("Configured Qwen service does not belong to this workspace")
    subprocess.run(["systemctl", "--user", "start", unit], check=True)
    from experiments.qwen_curriculum.run import service_identity
    url = parent_execution["qwen_url"].rsplit("/", 1)[0] + "/health"
    session = requests.Session()
    session.trust_env = False
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException:
            if unit_state(unit) not in ACTIVE:
                raise RuntimeError("Qwen service failed during startup")
            time.sleep(2)
            continue
        health = response.json()
        if health.get("error"):
            raise RuntimeError(f"Qwen startup error: {health['error']}")
        if health.get("generator_ready") is True:
            if service_identity(health) != parent_execution["service"]:
                raise ValueError("Qwen model/sampling/canvas identity changed since the previous generation")
            return
        time.sleep(2)
    raise TimeoutError("Qwen did not become ready within 900 seconds")


def validate_completed_campaign(config):
    from experiments.qwen_curriculum import run
    for stage in config["stages"]:
        directory = Path(stage["run_dir"])
        plan, jobs = run.load_plan(directory)
        execution = config_file(directory / "execution_config.json")
        if (plan["fingerprint"] != stage["plan_fingerprint"] or len(jobs) != stage["num_images"]
                or execution["plan_fingerprint"] != plan["fingerprint"]
                or execution["max_calls"] != stage["max_calls"]):
            raise ValueError("Final stage provenance differs from its frozen plan")
        expected_ids = {row["sample_id"] for row in jobs}
        for name in ("results.jsonl", "generated.jsonl"):
            saved = rows(directory, name)
            run.validate_records(saved, execution, jobs)
            if {row["sample_id"] for row in saved} != expected_ids:
                raise ValueError("Not every planned source has a saved and evaluated image")
        attempts = run.validate_attempts(rows(directory, "attempts.jsonl"), jobs, execution)
        if attempts != expected_ids:
            raise ValueError("Completed stage call ledger differs from its planned jobs")


def execute(args, config):
    coordinator = freeze_coordinator(args, config)
    check_frozen_artifacts(args, config, coordinator)
    wait_for_training(args, config, coordinator)
    check_frozen_artifacts(args, config, coordinator)
    from experiments.qwen_curriculum import recover_interrupted
    # Recovery performs no generation or model loading. The frozen stage runner
    # subsequently verifies these saved images without another HTTP POST.
    for stage in config["stages"]:
        directory = Path(stage["run_dir"])
        if (directory / "execution_config.json").exists():
            recovered = recover_interrupted.recover_interrupted(
                directory, common.ADAPTVPR_ROOT / "tmp/service_outputs/lightx2v", apply=True)
            print(json.dumps(recovered, ensure_ascii=False), flush=True)
    refresh(args.output_dir, config, "starting_qwen")
    parent_execution = config_file(Path(config["parent_run_dir"]) / "execution_config.json")
    start_qwen(parent_execution)
    for index, stage in enumerate(config["stages"], 1):
        command = [sys.executable, "-u", str(common.CURRICULUM_ROOT / "run.py"), "generate",
                   "--run-dir", stage["run_dir"], "--max-calls", str(stage["max_calls"]),
                   "--qwen-url", parent_execution["qwen_url"], "--request-timeout", str(parent_execution["request_timeout"]),
                   "--device", parent_execution["matcher_device"], "--cpu-threads", str(parent_execution["cpu_threads"])]
        phase = f"generating_stage_{index}"
        refresh(args.output_dir, config, phase)
        log_path = args.output_dir / f"stage_{index}.log"
        with log_path.open("a", encoding="utf-8") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                while child.poll() is None:
                    refresh(args.output_dir, config, phase)
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
                raise RuntimeError(f"Generation stage {index} stopped; see {log_path}")
        stage_summary = json.loads((Path(stage["run_dir"]) / "summary.json").read_text())
        if stage_summary.get("state") != "complete" or stage_summary.get("generation_errors"):
            raise RuntimeError("Stage did not produce all requested images; no automatic retries are allowed")
    check_frozen_artifacts(args, config, coordinator)
    validate_completed_campaign(config)
    summary = refresh(args.output_dir, config, "validating_completion")
    if summary["generated_images"] != config["total_images"] or summary["quality_evaluated"] != config["total_images"]:
        raise ValueError("Campaign output count differs from its cumulative target")
    subprocess.run(["systemctl", "--user", "stop", "adaptvpr-lightx2v.service"], check=True)
    summary = refresh(args.output_dir, config, "complete")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("run", "status"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-unit", default="qwen-ratio-700-vpr-scratch.service")
    parser.add_argument("--training-dir", type=Path,
                        default=common.WORKSPACE_ROOT / "outputs/qwen_curriculum/ratio_700_vpr_scratch_8to1_4to1")
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.training_dir = args.training_dir.expanduser().resolve()
    if not args.training_unit.endswith(".service") or "/" in args.training_unit:
        parser.error("A systemd service name is required")
    return args


def main(argv=None):
    args = parse_args(argv)
    config = config_file(args.output_dir / "campaign_config.json")
    if args.command == "status":
        print((args.output_dir / "summary.json").read_text())
        return
    with (args.output_dir / ".campaign.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another worker owns this generation campaign") from exc
        try:
            execute(args, config)
        except Exception as exc:
            refresh(args.output_dir, config, "stopped", f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
