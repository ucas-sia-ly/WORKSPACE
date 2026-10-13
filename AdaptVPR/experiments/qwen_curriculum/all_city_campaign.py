"""Generate the unlimited, fully selected 23-city cohort with unchanged Qwen.

Global weather quotas are frozen before partitioning into <=1000-call stages.
Existing images are referenced, never regenerated or re-signed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import requests

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import common, run, plan, scale_campaign, mine_all_cities, fast_campaign, recover_interrupted, extend_plan
from experiments.qwen_curriculum.campaign import config_file

DEFAULT_OUTPUT = common.WORKSPACE_ROOT / "outputs/qwen_curriculum/generation_23city"


def prepare(directory, mining_dir):
    path = directory / "all_city_campaign_config.json"
    if path.exists():
        cfg = config_file(path)
        if cfg["mining_dir"] != str(mining_dir):
            raise ValueError("Frozen mining selection changed")
        check_inputs(cfg)
        return cfg
    sources, selection, request = mine_all_cities.load_selection(mining_dir)
    old, parent = scale_campaign.load_collection(Path(request["parent"]))
    target = Counter(plan.allocate_domains(len(old) + len(sources), plan.DEFAULT_DOMAIN_WEIGHTS, 42))
    additional_quotas = target - Counter(r["condition"] for r in old)
    directory.mkdir(parents=True, exist_ok=True)
    master_dir = directory / "additional_plan"
    master_cfg, new_jobs = plan.build_plan(SimpleNamespace(sources=Path(selection["additional_sources"]),
        output_dir=master_dir, num_images=len(sources), seed=42, domain_weights=dict(additional_quotas), domain_stats=None))
    master_dir.mkdir(exist_ok=True)
    extend_plan._freeze_bytes(master_dir / "plan_config.json", extend_plan._json_bytes(master_cfg))
    extend_plan._freeze_bytes(master_dir / "plan.jsonl", plan._plan_bytes(new_jobs))
    by_path = {r["source_path"]: r for r in sources}
    stages = []
    for start in range(0, len(new_jobs), 1000):
        jobs = new_jobs[start:start + 1000]
        stage_dir = directory / f"additional_{len(stages) + 1:03d}_{len(jobs)}"
        stage_dir.mkdir(exist_ok=True)
        source_manifest = stage_dir / "sources.jsonl"
        extend_plan._freeze_bytes(source_manifest, plan._plan_bytes([by_path[r["source_path"]] for r in jobs]))
        payload = plan._plan_bytes(jobs)
        extend_plan._freeze_bytes(stage_dir / "plan.jsonl", payload)
        body = {k: v for k, v in master_cfg.items() if k != "fingerprint"}
        body.update(output_dir=str(stage_dir), sources=str(source_manifest),
            sources_sha256=common.file_sha256(source_manifest), source_manifest_rows=len(jobs),
            num_images=len(jobs), domain_quotas=dict(Counter(r["condition"] for r in jobs)),
            allocation="global_frozen_weather_quotas_partitioned_without_changing_job_seals",
            master_plan_fingerprint=master_cfg["fingerprint"], plan_sha256=hashlib.sha256(payload).hexdigest())
        stage_cfg = fast_campaign.frozen(stage_dir / "plan_config.json", body)
        stages.append({"run_dir": str(stage_dir), "num_images": len(jobs), "max_calls": len(jobs),
                       "plan_fingerprint": stage_cfg["fingerprint"]})
    old_jobs = common.read_jsonl(Path(parent["combined_plan"]))
    all_jobs = old_jobs + new_jobs
    scale_campaign.validate_unique(all_jobs)
    if Counter(r["condition"] for r in all_jobs) != target or {r["city"] for r in all_jobs} != set(request["cities"]):
        raise ValueError("Combined plan lacks a city or differs from the exact global weather quota")
    combined = directory / "combined_plan.jsonl"
    extend_plan._freeze_bytes(combined, plan._plan_bytes(all_jobs))
    files = [combined, master_dir / "plan.jsonl", master_dir / "plan_config.json",
        mining_dir / "all_city_selection.json", mining_dir / "request_config.json", Path(__file__),
        Path(plan.__file__), Path(mine_all_cities.__file__), Path(scale_campaign.__file__)]
    files += [Path(s["run_dir"]) / name for s in stages for name in ("plan_config.json", "plan.jsonl", "sources.jsonl")]
    cfg = fast_campaign.frozen(path, {"schema_version": 1, "stage": "all_eligible_23city_qwen",
        "mining_dir": str(mining_dir), "parent_run_dir": request["parent"],
        "parent_fingerprint": parent["fingerprint"], "total_images": len(all_jobs),
        "additional_images": len(new_jobs), "stages": stages, "cities": request["cities"],
        "domain_quotas": dict(target), "city_quotas": dict(Counter(r["city"] for r in all_jobs)),
        "combined_plan": str(combined), "reference_execution": parent["reference_execution"],
        "runtime_profile": parent["runtime_profile"], "estimated_bytes_per_new_image": 3_000_000,
        "files_sha256": {str(p.resolve()): common.file_sha256(p) for p in files}})
    publish(directory, cfg, "prepared")
    return cfg


def check_inputs(config):
    mine_all_cities.check_files(config)
    mine_all_cities.load_selection(Path(config["mining_dir"]))


def publish(directory, config, state, error=None):
    report = scale_campaign.publish(directory, config, state, error)
    rows = scale_campaign.collected(config, "results.jsonl")
    report["cities_generated"] = dict(Counter(r["city"] for r in scale_campaign.collected(config, "generated.jsonl")))
    report["cities_accepted"] = dict(Counter(r["city"] for r in rows if r.get("passed") and r.get("eligible_for_training")))
    remaining = config["total_images"] - report["generated_images"]
    report.update(required_free_bytes_estimate=remaining * config["estimated_bytes_per_new_image"] + 5_000_000_000,
                  available_free_bytes=shutil.disk_usage(directory).free)
    common.write_json(directory / "summary.json", report)
    return report


def load_collection(directory, *, require_complete=True):
    directory = Path(directory).resolve()
    cfg = config_file(directory / "all_city_campaign_config.json")
    check_inputs(cfg)
    summary = json.loads((directory / "summary.json").read_text())
    if summary.get("campaign_fingerprint") != cfg["fingerprint"] or (require_complete and summary["state"] != "complete"):
        raise ValueError("Finish all-city generation before training")
    parent, _ = scale_campaign.load_collection(Path(cfg["parent_run_dir"]))
    rows = list(parent)
    for stage in cfg["stages"]:
        rows += scale_campaign.check_stage(stage, cfg["reference_execution"], complete=True)
    scale_campaign.validate_unique(rows)
    if (len(rows) != cfg["total_images"] or rows != run.read_rows(directory, "results.jsonl")
            or Counter(r["city"] for r in rows) != cfg["city_quotas"]
            or Counter(r["condition"] for r in rows) != cfg["domain_quotas"]
            or scale_campaign.collected(cfg, "generated.jsonl") != run.read_rows(directory, "generated.jsonl")):
        raise ValueError("All-city aggregation differs from original stage journals")
    return rows, cfg


def execute(directory, config):
    if json.loads((directory / "summary.json").read_text())["state"] == "complete":
        load_collection(directory)
        return
    report = publish(directory, config, "preflight")
    if report["available_free_bytes"] < report["required_free_bytes_estimate"]:
        raise RuntimeError(f"Insufficient disk: need approximately {report['required_free_bytes_estimate']/1e9:.1f} GB "
            f"free for remaining raw/training PNGs, available {report['available_free_bytes']/1e9:.1f} GB. "
            "Prepare a new --output-dir on a larger disk; no image or historical ledger is deleted.")
    for stage in config["stages"]:
        scale_campaign.check_stage(stage, config["reference_execution"])
    subprocess.run(["systemctl", "--user", "start", "adaptvpr-lightx2v.service"], check=True)
    session = requests.Session()
    session.trust_env = False
    for _ in range(180):
        try:
            response = session.get("http://127.0.0.1:8001/health", timeout=5)
            response.raise_for_status()
            health = response.json()
            if health.get("error"):
                raise RuntimeError(health["error"])
            if health.get("generator_ready"):
                if run.service_identity(health) != config["reference_execution"]["service"]:
                    raise ValueError("Qwen configuration differs from the previous generation")
                break
        except requests.ConnectionError:
            pass
        time.sleep(2)
    else:
        raise RuntimeError("Qwen startup timed out")
    for stage in config["stages"]:
        stage_dir = Path(stage["run_dir"])
        if (stage_dir / "execution_config.json").exists():
            recover_interrupted.recover_interrupted(stage_dir, run.service_output_directory(), apply=True)
        results = scale_campaign.check_stage(stage, config["reference_execution"])
        if any(r.get("status") == "generation_error" for r in results):
            raise RuntimeError("Reserved request has unresolved outcome; no repeat POST")
        if len(results) == stage["num_images"]:
            continue
        command = [sys.executable, "-u", str(common.CURRICULUM_ROOT / "run.py"), "generate", "--run-dir", str(stage_dir),
            "--max-calls", str(stage["max_calls"]), "--request-timeout", "600", "--device", "cuda", "--cpu-threads", "4"]
        with (directory / "generation.log").open("a") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                while child.poll() is None:
                    publish(directory, config, "generating")
                    time.sleep(30)
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
            if child.returncode:
                raise RuntimeError("Generation stopped; inspect generation.log")
        scale_campaign.check_stage(stage, config["reference_execution"], complete=True)
    publish(directory, config, "validating")
    load_collection(directory, require_complete=False)
    subprocess.run(["systemctl", "--user", "stop", "adaptvpr-lightx2v.service"], check=True)
    print(json.dumps(publish(directory, config, "complete"), ensure_ascii=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("prepare", "run", "status", "validate"))
    parser.add_argument("--mining-dir", type=Path, default=mine_all_cities.DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    directory = args.output_dir.resolve()
    if args.command == "status":
        print((directory / "summary.json").read_text())
        return
    with run.run_lock(directory):
        config = prepare(directory, args.mining_dir.resolve())
        if args.command == "validate":
            rows, _ = load_collection(directory)
            print(f"Validated {len(rows)} evaluated 23-city images")
        elif args.command == "run":
            try:
                execute(directory, config)
            except BaseException as exc:
                publish(directory, config, "stopped", f"{type(exc).__name__}: {exc}")
                raise


if __name__ == "__main__":
    main()
