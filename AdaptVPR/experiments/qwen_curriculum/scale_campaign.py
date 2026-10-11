"""Extend the sealed 2000-image cohort using disjoint cached hard sources.

Each independent generation stage retains the existing <=1000 POST budget and
unchanged generator/verifier. Historical journals and image bytes are read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
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
from experiments.qwen_curriculum import common, run, plan, fast_campaign, extend_plan, recover_interrupted
from experiments.qwen_curriculum.campaign import config_file

DEFAULT_OUTPUT = common.WORKSPACE_ROOT / "outputs/qwen_curriculum/generation_7277"


def stage_budgets(additional):
    if additional <= 0:
        raise ValueError("Extension must add images")
    return [1000] * (additional // 1000) + ([additional % 1000] if additional % 1000 else [])


def check_inputs(config):
    for path, digest in config["files_sha256"].items():
        if common.file_sha256(Path(path)) != digest:
            raise ValueError(f"Frozen campaign input changed: {path}")
    for stage in config["stages"]:
        cfg = config_file(Path(stage["run_dir"]) / "plan_config.json")
        if cfg["fingerprint"] != stage["plan_fingerprint"]:
            raise ValueError("Stage plan configuration changed")


def prepare(directory, parent, total):
    directory, parent = Path(directory).resolve(), Path(parent).resolve()
    path = directory / "scale_campaign_config.json"
    if path.exists():
        cfg = config_file(path)
        if cfg["total_images"] != total or cfg["parent_run_dir"] != str(parent):
            raise ValueError("Use the original frozen target and parent")
        check_inputs(cfg)
        return cfg
    if directory == parent or directory.is_relative_to(parent) or parent.is_relative_to(directory):
        raise ValueError("Parent and extension must be separate directories")
    old, parent_fp, _ = fast_campaign.load_collection(parent, 2000)
    legacy = config_file(parent / "fast_campaign_config.json")
    parent_plan, _ = run.load_plan(Path(legacy["parent_run_dir"]))
    reference = config_file(Path(legacy["generation_run_dir"]) / "execution_config.json")
    sources, mining = extend_plan._cached_selection(parent_plan, total)
    additional = extend_plan._additional_sources(old, sources, mining["selection"]["max_sources_per_place"])
    additional = plan._select_sources(additional, len(additional))
    target_domains = Counter(plan.allocate_domains(total, parent_plan["domain_weights"], parent_plan["seed"]))
    remaining = target_domains - Counter(r["condition"] for r in old)
    stages, all_jobs = [], common.read_jsonl(Path(legacy["combined_plan"]))
    cursor = 0
    directory.mkdir(parents=True, exist_ok=True)
    for index, budget in enumerate(stage_budgets(len(additional)), 1):
        stage_dir = directory / f"additional_{index:02d}_{budget}"
        stage_dir.mkdir(exist_ok=True)
        source_path = stage_dir / "sources.jsonl"
        source_rows = additional[cursor:cursor + budget]
        extend_plan._freeze_bytes(source_path, plan._plan_bytes(source_rows))
        weights = parent_plan["domain_weights"] if budget == 1000 else dict(remaining)
        cfg, jobs = plan.build_plan(SimpleNamespace(sources=source_path, output_dir=stage_dir,
            num_images=budget, seed=parent_plan["seed"], domain_weights=weights, domain_stats=None))
        remaining.subtract(Counter(r["condition"] for r in jobs))
        if any(n < 0 for n in remaining.values()):
            raise ValueError("Stage allocation exceeds the cumulative weather quota")
        extend_plan._freeze_bytes(stage_dir / "plan.jsonl", plan._plan_bytes(jobs))
        fast_campaign.frozen(stage_dir / "plan_config.json", {k: v for k, v in cfg.items() if k != "fingerprint"})
        stages.append({"run_dir": str(stage_dir), "num_images": budget,
                       "max_calls": budget, "plan_fingerprint": cfg["fingerprint"]})
        all_jobs.extend(jobs)
        cursor += budget
    if any(remaining.values()) or len(all_jobs) != total:
        raise ValueError("Final weather allocation differs from target")
    validate_unique(all_jobs)
    combined = directory / "combined_plan.jsonl"
    extend_plan._freeze_bytes(combined, plan._plan_bytes(all_jobs))
    parent_paths = [parent / name for name in
        ("fast_campaign_config.json", "summary.json", "generated.jsonl", "results.jsonl")]
    parent_paths += [Path(mining["mining_config"]), Path(mining["descriptor_cache"]) / "source_scores.npz"]
    stage_paths = [Path(s["run_dir"]) / name for s in stages for name in
                   ("plan.jsonl", "plan_config.json", "sources.jsonl")]
    code_paths = [Path(__file__), Path(extend_plan.__file__), Path(plan.__file__)]
    code_paths += [common.ADAPTVPR_ROOT / p for p in reference["implementation_sha256"]]
    code_paths += [Path(p) for p in legacy["implementation_sha256"]]
    cfg = fast_campaign.frozen(path, {
        "schema_version": 1, "stage": "qwen_disjoint_hard_source_extension",
        "total_images": total, "original_images": len(old), "additional_images": len(additional),
        "parent_run_dir": str(parent), "parent_fingerprint": parent_fp, "stages": stages,
        "combined_plan": str(combined), "domain_quotas": dict(target_domains),
        "city_quotas": dict(Counter(r["city"] for r in all_jobs)),
        "source_policy": "one weather edit per unique real hard source; cumulative cap 2 sources/place",
        "mining_provenance": mining, "reference_execution": reference,
        "runtime_profile": reference["service"]["performance"]["profile"],
        "files_sha256": {str(p.resolve()): common.file_sha256(p) for p in
                         parent_paths + stage_paths + code_paths + [combined]},
    })
    publish(directory, cfg, "prepared")
    return cfg


def validate_unique(rows):
    for key in ("sample_id", "source_path", "source_sha256"):
        if len({r[key] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate cumulative {key}")


def collected(config, name):
    rows = run.read_rows(Path(config["parent_run_dir"]), name)
    for stage in config["stages"]:
        rows += run.read_rows(Path(stage["run_dir"]), name)
    validate_unique(rows)
    return rows


def publish(directory, config, state, error=None):
    generated, results = collected(config, "generated.jsonl"), collected(config, "results.jsonl")
    accepted = [r for r in results if r.get("passed") and r.get("eligible_for_training")]
    calls = sum(len(run.read_rows(Path(s["run_dir"]), "attempts.jsonl")) for s in config["stages"])
    summary = {"state": state, "target_images": config["total_images"],
        "generated_images": len(generated), "quality_evaluated": len(results),
        "accepted": len(accepted), "rejected": sum(r.get("status") == "rejected" for r in results),
        "generation_errors": sum(r.get("status") == "generation_error" for r in results),
        "new_calls_reserved": calls, "new_generation_calls_remaining": config["additional_images"] - calls,
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


def check_stage(stage, reference, complete=False):
    directory = Path(stage["run_dir"])
    cfg, jobs = run.load_plan(directory)
    results = run.read_rows(directory, "results.jsonl")
    generated = run.read_rows(directory, "generated.jsonl")
    attempts = run.read_rows(directory, "attempts.jsonl")
    execution_path = directory / "execution_config.json"
    if execution_path.exists():
        execution = config_file(execution_path)
        expected = {**reference, **run.execution_definition(SimpleNamespace(
            run_dir=directory, device="cuda", cpu_threads=4)), "max_calls": stage["max_calls"],
            "plan_fingerprint": cfg["fingerprint"]}
        expected.pop("fingerprint", None)
        if execution != {**expected, "fingerprint": common.fingerprint(expected)}:
            raise ValueError("Stage generator or quality settings differ from the original fast run")
        run.validate_records(results, execution, jobs)
        run.validate_records(generated, execution, jobs)
        reserved = run.validate_attempts(attempts, jobs, execution)
        if not {r["sample_id"] for r in results + generated} <= reserved:
            raise ValueError("Saved image lacks a durable call reservation")
    elif results or generated or attempts:
        raise ValueError("Stage journals exist without an execution configuration")
    if complete and (len(results) != len(jobs) or len(generated) != len(jobs)
                     or any(r.get("status") not in {"passed", "rejected"} for r in results)):
        raise ValueError("Stage has not produced and evaluated all planned images")
    return results


def load_collection(directory, *, require_complete=True):
    directory = Path(directory)
    cfg = config_file(directory / "scale_campaign_config.json")
    check_inputs(cfg)
    summary = json.loads((directory / "summary.json").read_text())
    if summary.get("campaign_fingerprint") != cfg["fingerprint"] or (
            require_complete and summary.get("state") != "complete"):
        raise ValueError("Finish the complete generation campaign before training")
    parent, _, _ = fast_campaign.load_collection(Path(cfg["parent_run_dir"]), cfg["original_images"])
    rows = list(parent)
    for stage in cfg["stages"]:
        rows += check_stage(stage, cfg["reference_execution"], complete=True)
    validate_unique(rows)
    if (len(rows) != cfg["total_images"] or rows != run.read_rows(directory, "results.jsonl")
            or Counter(r["condition"] for r in rows) != cfg["domain_quotas"]
            or collected(cfg, "generated.jsonl") != run.read_rows(directory, "generated.jsonl")):
        raise ValueError("Aggregate differs from the original stage journals or weather target")
    return rows, cfg


def execute(directory, config):
    fast_campaign.load_collection(Path(config["parent_run_dir"]), config["original_images"])
    if json.loads((directory / "summary.json").read_text()).get("state") == "complete":
        load_collection(directory)
        return
    if shutil.disk_usage(directory).free < config["additional_images"] * 3_000_000 + 5_000_000_000:
        raise RuntimeError("Insufficient disk space for retained raw and normalized images")
    for stage in config["stages"]:
        check_stage(stage, config["reference_execution"])
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
                    raise ValueError("Qwen service identity differs from the original 2000-image fast run")
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
        results = check_stage(stage, config["reference_execution"])
        if any(r.get("status") == "generation_error" for r in results):
            raise RuntimeError("Unresolved reserved call; stopped without repeating POST")
        if len(results) == stage["num_images"]:
            check_stage(stage, config["reference_execution"], complete=True)
            continue
        command = [sys.executable, "-u", str(common.CURRICULUM_ROOT / "run.py"), "generate",
            "--run-dir", str(stage_dir), "--max-calls", str(stage["max_calls"]),
            "--request-timeout", "600", "--device", "cuda", "--cpu-threads", "4"]
        with (directory / "generation.log").open("a") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                while child.poll() is None:
                    publish(directory, config, "generating")
                    time.sleep(15)
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
            if child.returncode:
                raise RuntimeError("Generation stopped; inspect generation.log; no automatic request retry")
        check_stage(stage, config["reference_execution"], complete=True)
    publish(directory, config, "validating")
    load_collection(directory, require_complete=False)
    subprocess.run(["systemctl", "--user", "stop", "adaptvpr-lightx2v.service"], check=True)
    print(json.dumps(publish(directory, config, "complete"), ensure_ascii=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("prepare", "run", "status", "validate"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parent", type=Path, default=common.WORKSPACE_ROOT / "outputs/qwen_curriculum/generation_2000")
    parser.add_argument("--total-images", type=int, default=7277)
    args = parser.parse_args(argv)
    directory = args.output_dir.resolve()
    if args.command == "status":
        print((directory / "summary.json").read_text())
        return
    with run.run_lock(directory):
        config = prepare(directory, args.parent, args.total_images)
        if args.command == "validate":
            rows, _ = load_collection(directory)
            print(f"Validated {len(rows)} successful evaluated images")
        elif args.command == "run":
            try:
                execute(directory, config)
            except BaseException as exc:
                publish(directory, config, "stopped", f"{type(exc).__name__}: {exc}")
                raise


if __name__ == "__main__":
    main()
