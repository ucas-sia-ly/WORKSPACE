"""Extend a frozen Qwen cohort without changing its jobs or saved images.

The historical, training-only difficulty cache supplies additional sources.
No descriptor extraction, quality evaluation or generation runs during prepare.
The original and additional stages retain separate immutable call ledgers.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import common, mine_sources, plan, run


def _checked_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(config, dict) or config.get("fingerprint") != common.fingerprint(
            {key: value for key, value in config.items() if key != "fingerprint"})):
        raise ValueError(f"Invalid frozen configuration checksum: {path}")
    return config


def _checked_file(path, expected, description):
    if not Path(path).is_file() or common.file_sha256(Path(path)) != expected:
        raise ValueError(f"{description} is missing or changed: {path}")


def _check_parent(parent):
    config, jobs = run.load_plan(parent)
    for relative, expected in config["implementation_sha256"].items():
        _checked_file(common.ADAPTVPR_ROOT / relative, expected, "Parent planning implementation")
    _checked_file(config["sources"], config["sources_sha256"], "Parent mined sources")
    execution = _checked_config(parent / "execution_config.json")
    if (execution.get("stage") != "generate" or execution.get("plan_fingerprint") != config["fingerprint"]
            or execution.get("max_calls") != len(jobs)):
        raise ValueError("Parent must be an unchanged bounded Qwen generation run")
    run.service_identity({**execution["service"], "status": "ok", "model_loaded": True,
                          "generator_ready": True, "error": None})
    for relative, expected in execution["implementation_sha256"].items():
        _checked_file(common.ADAPTVPR_ROOT / relative, expected, "Parent generation implementation")
    results = run.read_rows(parent, "results.jsonl")
    generated = run.read_rows(parent, "generated.jsonl")
    run.validate_records(results, execution, jobs)
    run.validate_records(generated, execution, jobs)
    attempted = run.validate_attempts(run.read_rows(parent, "attempts.jsonl"), jobs, execution)
    if not {row["sample_id"] for row in results + generated} <= attempted:
        raise ValueError("Parent saved images lack their durable call reservations")
    return config, jobs, execution


def _cached_selection(parent_config, total):
    """Use archived score provenance; current training code need not re-extract."""
    mining_dir = Path(parent_config["sources"]).parent
    mining = _checked_config(mining_dir / "mining_config.json")
    summary = json.loads((mining_dir / "summary.json").read_text())
    _checked_file(parent_config["sources"], summary["sources_sha256"], "Mined source manifest")
    if summary.get("config_fingerprint") != mining["fingerprint"] or mining.get("benchmark_data_used") is not False:
        raise ValueError("Invalid training-only mining provenance")
    request = mining["descriptor_request"]
    if request.get("fingerprint") != common.fingerprint({k: v for k, v in request.items() if k != "fingerprint"}):
        raise ValueError("Descriptor request checksum mismatch")
    real_data = Path(request["real_data"])
    for city, expected in request["metadata_sha256"].items():
        _checked_file(real_data / "Dataframes" / f"{city}.csv", expected, "GSV training metadata")
    _checked_file(request["checkpoint"], request["checkpoint_sha256"], "Cached mining teacher checkpoint")
    records, cities = mine_sources.load_training_sources(real_data, request["cities"], request["min_images_per_place"])
    if cities != request["cities"] or mine_sources.sequence_fingerprint(records) != request["source_order_sha256"]:
        raise ValueError("Cached source order differs from current training metadata")
    if mine_sources.image_inventory(records) != request["image_inventory"]:
        raise ValueError("Training image inventory differs from the cached mining cohort")
    cache = Path(summary["descriptor_cache"])
    if json.loads((cache / "descriptor_config.json").read_text()) != request:
        raise ValueError("Descriptor cache configuration changed")
    complete = json.loads((cache / "descriptor_complete.json").read_text())
    stat = (cache / "descriptors.npy").stat()
    if (complete.get("config_fingerprint") != request["fingerprint"]
            or complete.get("shape", [None])[0] != len(records)
            or complete.get("file_stat") != {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}):
        raise ValueError("Descriptor cache is incomplete or changed")
    scoring = mining["scoring"]
    if scoring.get("fingerprint") != common.fingerprint({k: v for k, v in scoring.items() if k != "fingerprint"}):
        raise ValueError("Difficulty scoring configuration checksum mismatch")
    if scoring.get("descriptor_fingerprint") != request["fingerprint"]:
        raise ValueError("Difficulty scores refer to another descriptor cohort")
    score_marker = json.loads((cache / "scores_complete.json").read_text())
    if score_marker.get("config_fingerprint") != scoring["fingerprint"]:
        raise ValueError("Difficulty score cache configuration changed")
    _checked_file(cache / "source_scores.npz", score_marker["sha256"], "Difficulty score cache")
    with np.load(cache / "source_scores.npz", allow_pickle=False) as saved:
        scores = {key: saved[key] for key in saved.files}
    selection = mining["selection"]
    bands = {(row["min_quantile"], row["max_quantile"]) for row in selection["cities"].values()}
    if len(bands) != 1:
        raise ValueError("Per-city custom hardness bands need an explicit extension policy")
    low, high = bands.pop()
    selected, audit = mine_sources.select_hard_sources(records, scores, total,
        seed=selection["seed"], min_quantile=low, max_quantile=high,
        max_sources_per_place=selection["max_sources_per_place"], city_quotas=selection["quota_mode"])
    provenance = {"mining_config": str(mining_dir / "mining_config.json"),
                  "mining_config_sha256": common.file_sha256(mining_dir / "mining_config.json"),
                  "mining_fingerprint": mining["fingerprint"], "descriptor_cache": str(cache),
                  "scores_sha256": score_marker["sha256"], "selection": audit,
                  "descriptor_extraction_performed": False, "benchmark_data_used": False,
                  "training_sources_scored": len(records)}
    provenance["extension_selection_fingerprint"] = common.fingerprint(provenance)
    for row in selected:
        row["source_sha256"] = common.file_sha256(Path(row["source_path"]))
        row["historical_mining_fingerprint"] = mining["fingerprint"]
        row["mining_fingerprint"] = provenance["extension_selection_fingerprint"]
    return selected, provenance


def _additional_sources(old_jobs, selected, cap):
    old_paths = {row["source_path"] for row in old_jobs}
    old_hashes = {row["source_sha256"] for row in old_jobs}
    selected_paths = {row["source_path"] for row in selected}
    if len(old_paths) != len(old_jobs) or len(old_hashes) != len(old_jobs):
        raise ValueError("Parent source paths or image content are duplicated")
    if not old_paths <= selected_paths:
        raise ValueError("Expanded hardness cohort does not preserve every original source")
    by_path = {row["source_path"]: row for row in selected}
    for old in old_jobs:
        chosen = by_path[old["source_path"]]
        for key in ("source_sha256", "city", "place_id", "source_index", "hardness"):
            if old.get(key) != chosen.get(key):
                raise ValueError(f"Expanded source identity or mining score differs: {old['source_path']}")
    counts = Counter((row["city"], row["place_id"]) for row in selected)
    if max(counts.values(), default=0) > cap:
        raise ValueError("Expanded cohort exceeds the cumulative per-place source cap")
    additional = [row for row in selected if row["source_path"] not in old_paths]
    if any(row["source_sha256"] in old_hashes for row in additional):
        raise ValueError("Additional source content duplicates an original source")
    if len({row["source_sha256"] for row in additional}) != len(additional):
        raise ValueError("Additional sources contain duplicate image content")
    return additional


@contextmanager
def _prepare_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".prepare.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _freeze_bytes(path, payload):
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Frozen extension artifact changed: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        common._atomic_write(path, [payload.decode("utf-8")])


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")


def prepare_extension(parent_run_dir, output_dir, total_images=2000):
    """Freeze a staged extension; return its immutable campaign configuration."""
    parent, output = Path(parent_run_dir).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    if parent == output or parent.is_relative_to(output) or output.is_relative_to(parent):
        raise ValueError("Extension and parent directories must be separate")
    if type(total_images) is not int:
        raise ValueError("Total image budget must be an integer")
    saved_path = output / "campaign_config.json"
    if saved_path.exists():
        saved = _checked_config(saved_path)
        if (saved.get("parent_run_dir") != str(parent) or saved.get("output_dir") != str(output)
                or saved.get("total_images") != total_images):
            raise ValueError("Frozen extension campaign configuration changed; use a new output directory")
        for relative, expected in saved["implementation_sha256"].items():
            _checked_file(common.ADAPTVPR_ROOT / relative, expected, "Frozen extension implementation")
    parent_config, old_jobs, execution = _check_parent(parent)
    additional_count = total_images - len(old_jobs)
    if not 1 <= additional_count <= 1000:
        raise ValueError("Extension requires 1..1000 additional source jobs")
    if not 1 <= len(old_jobs) <= 1000:
        raise ValueError("Parent stage must have 1..1000 source jobs")
    source_config = _checked_config(Path(parent_config["sources"]).parent / "mining_config.json")
    cap = source_config["selection"]["max_sources_per_place"]
    selected, provenance = _cached_selection(parent_config, total_images)
    if len(selected) != total_images:
        raise ValueError("Difficulty cache did not supply the requested total cohort")
    additional = _additional_sources(old_jobs, selected, cap)
    if len(additional) != additional_count:
        raise ValueError("Additional source count differs from the extension budget")
    additional_dir = output / f"additional_{additional_count}"
    sources_path = additional_dir / "sources.jsonl"
    source_bytes = plan._plan_bytes(additional)
    with _prepare_lock(output):
        _freeze_bytes(sources_path, source_bytes)
        new_config, new_jobs = plan.build_plan(SimpleNamespace(sources=sources_path, output_dir=additional_dir,
            num_images=additional_count, seed=parent_config["seed"],
            domain_weights=parent_config["domain_weights"], domain_stats=None))
        if set(row["sample_id"] for row in old_jobs) & set(row["sample_id"] for row in new_jobs):
            raise ValueError("Additional jobs duplicate an original sample identity")
        original_bytes = (parent / "plan.jsonl").read_bytes()
        if not original_bytes.endswith(b"\n"):
            raise ValueError("Parent plan must end with a newline for an exact prefix extension")
        combined = original_bytes + plan._plan_bytes(new_jobs)
        stages = [{"run_dir": str(parent), "num_images": len(old_jobs), "max_calls": execution["max_calls"],
                   "plan_fingerprint": parent_config["fingerprint"], "plan_sha256": parent_config["plan_sha256"]},
                  {"run_dir": str(additional_dir), "num_images": additional_count, "max_calls": additional_count,
                   "plan_fingerprint": new_config["fingerprint"], "plan_sha256": new_config["plan_sha256"]}]
        stage_provenance = {"row_ranges_zero_based_end_exclusive": [
            {"stage_index": 0, "start": 0, "end": len(old_jobs), "run_dir": str(parent)},
            {"stage_index": 1, "start": len(old_jobs), "end": total_images, "run_dir": str(additional_dir)}],
            "sealed_rows_modified": False, "original_plan_prefix_sha256": parent_config["plan_sha256"]}
        config = {"schema_version": 1, "stage": "qwen_curriculum_extension_campaign",
                  "output_dir": str(output), "parent_run_dir": str(parent), "additional_run_dir": str(additional_dir),
                  "total_images": total_images, "original_planned_images": len(old_jobs),
                  "additional_planned_images": additional_count, "stages": stages,
                  "combined_plan": str(output / "combined_plan.jsonl"),
                  "combined_plan_sha256": hashlib.sha256(combined).hexdigest(),
                  "parent_execution_fingerprint": execution["fingerprint"],
                  "parent_execution_config_sha256": common.file_sha256(parent / "execution_config.json"),
                  "domain_weights": parent_config["domain_weights"],
                  "domain_quotas": dict(sorted(Counter(row["condition"] for row in old_jobs + new_jobs).items())),
                  "city_quotas": dict(sorted(Counter(row["city"] for row in old_jobs + new_jobs).items())),
                  "selected_places": len({(row["city"], row["place_id"]) for row in old_jobs + new_jobs}),
                  "max_sources_per_place": cap, "source_overlap": 0, "source_content_overlap": 0,
                  "additional_sources_sha256": hashlib.sha256(source_bytes).hexdigest(),
                  "mining_provenance": provenance,
                  "generation_policy": "resume_original_stage_then_generate_disjoint_additional_stage",
                  "old_results_policy": "retain original files, seals, images and durable call ledger",
                  "training_policy": "generation only; preserve existing training experiment configuration",
                  "implementation_sha256": {str(path.resolve().relative_to(common.ADAPTVPR_ROOT)):
                                             common.file_sha256(path) for path in
                                             (Path(__file__), Path(mine_sources.__file__))}}
        config["fingerprint"] = common.fingerprint(config)
        config_path = output / "campaign_config.json"
        if config_path.exists() and _checked_config(config_path) != config:
            raise ValueError("Frozen extension campaign configuration changed; use a new output directory")
        _freeze_bytes(additional_dir / "plan_config.json", _json_bytes(new_config))
        _freeze_bytes(additional_dir / "plan.jsonl", plan._plan_bytes(new_jobs))
        _freeze_bytes(output / "combined_plan.jsonl", combined)
        _freeze_bytes(output / "stage_provenance.json", _json_bytes(stage_provenance))
        _freeze_bytes(config_path, _json_bytes(config))
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--parent-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-images", type=int, default=2000)
    args = parser.parse_args(argv)
    config = prepare_extension(args.parent_run_dir, args.output_dir, args.total_images)
    print(json.dumps({"total_images": config["total_images"], "domain_quotas": config["domain_quotas"],
                      "additional_run_dir": config["additional_run_dir"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
