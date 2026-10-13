"""Select every eligible hard source across all 23 local GSV cities.

Reuse the sealed four-city cohort. Process each remaining city independently,
with a disk-backed descriptor cache and within-city centroid negatives. There
is no fixed source/image quota; the original reliability/band/place rules apply.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import common, run, mine_sources, extend_plan, fast_campaign, scale_campaign
from experiments.qwen_curriculum.campaign import config_file

DEFAULT_OUTPUT = common.WORKSPACE_ROOT / "outputs/qwen_curriculum/mining_23city"
DEFAULT_PARENT = scale_campaign.DEFAULT_OUTPUT


def discover_cities(real_data, expected=23):
    cities = sorted(p.stem for p in (Path(real_data) / "Dataframes").glob("*.csv"))
    if len(cities) != expected:
        raise ValueError(f"Expected {expected} GSV cities, found {len(cities)}: {cities}")
    if any(not (Path(real_data) / "Images" / city).is_dir() for city in cities):
        raise ValueError("A metadata city lacks its Images directory")
    return cities


def missing_cities(cities, parent_rows):
    existing = {row["city"] for row in parent_rows}
    if not existing <= set(cities):
        raise ValueError("Historical images refer to a city outside the current training pool")
    return sorted(set(cities) - existing)


def check_files(config):
    for path, digest in config["files_sha256"].items():
        if common.file_sha256(Path(path)) != digest:
            raise ValueError(f"Frozen all-city input changed: {path}")


def prepare(directory, parent, real_data):
    path = directory / "request_config.json"
    if path.exists():
        config = config_file(path)
        if config["parent"] != str(parent) or config["real_data"] != str(real_data):
            raise ValueError("Frozen mining parent/data changed; use a new output directory")
        check_files(config)
        return config
    rows, parent_config = scale_campaign.load_collection(parent)
    cities = discover_cities(real_data)
    new_cities = missing_cities(cities, rows)
    old_mining = config_file(Path(parent_config["mining_provenance"]["mining_config"]))
    request = old_mining["descriptor_request"]
    if Path(request["real_data"]) != real_data:
        raise ValueError("New and historical generation must use the same GSV root")
    bands = {(r["min_quantile"], r["max_quantile"]) for r in old_mining["selection"]["cities"].values()}
    if len(bands) != 1:
        raise ValueError("Expected one historical hardness band")
    low, high = bands.pop()
    checkpoint = Path(request["checkpoint"])
    paths = [checkpoint, parent / "scale_campaign_config.json", parent / "results.jsonl", parent / "summary.json",
        Path(__file__), Path(mine_sources.__file__), Path(extend_plan.__file__), Path(common.__file__)]
    paths += [real_data / "Dataframes" / f"{city}.csv" for city in cities]
    directory.mkdir(parents=True, exist_ok=True)
    config = fast_campaign.frozen(path, {
        "schema_version": 1, "parent": str(parent), "parent_fingerprint": parent_config["fingerprint"],
        "real_data": str(real_data), "cities": cities, "new_cities": new_cities,
        "checkpoint": str(checkpoint), "backbone_repo": request["backbone_repo"],
        "selection": {"min_quantile": low, "max_quantile": high,
            "min_positive_similarity": old_mining["scoring"]["min_positive_similarity"],
            "near_negative_radius_m": old_mining["scoring"]["near_negative_radius_m"],
            "max_sources_per_place": old_mining["selection"]["max_sources_per_place"],
            "seed": old_mining["selection"]["seed"], "source_quota": "all_eligible_after_place_cap",
            "negative_pool": "within_each_new_city; historical four-city scores retained"},
        "files_sha256": {str(p.resolve()): common.file_sha256(p) for p in paths}})
    common.write_json(directory / "progress.json", {"state": "prepared", "cities": cities, "new_cities": new_cities})
    return config


def mine_city(directory, city, config):
    city_dir = directory / "cities" / city
    saved = city_dir / "all_selection.json"
    if saved.exists():
        selection = config_file(saved)
        check_files(selection)
        return common.read_jsonl(city_dir / "all_sources.jsonl"), selection
    policy = config["selection"]
    # The existing miner materializes its cache with a one-source probe. The
    # full-capacity selection below is read-only and preserves that provenance.
    argv = ["--checkpoint", config["checkpoint"], "--real-data", config["real_data"],
        "--cities", city, "--output-dir", str(city_dir), "--num-sources", "1",
        "--batch-size", "64", "--num-workers", "0", "--device", "cuda",
        "--backbone-repo", config["backbone_repo"], "--seed", str(policy["seed"]),
        "--min-positive-similarity", str(policy["min_positive_similarity"]),
        "--min-hardness-quantile", str(policy["min_quantile"]),
        "--max-hardness-quantile", str(policy["max_quantile"]),
        "--max-sources-per-place", str(policy["max_sources_per_place"]),
        "--near-negative-radius-m", str(policy["near_negative_radius_m"])]
    summary = mine_sources.mine(mine_sources.parse_args(argv))
    count = summary["selection_audit"]["cities"][city]["capacity_after_place_cap"]
    rows, provenance = extend_plan._cached_selection({"sources": str(city_dir / "sources.jsonl")}, count)
    common.write_jsonl(city_dir / "all_sources.jsonl", rows)
    cache = Path(summary["descriptor_cache"])
    files = [city_dir / "all_sources.jsonl", city_dir / "mining_config.json", city_dir / "summary.json",
             cache / "source_scores.npz", cache / "scores_complete.json", cache / "descriptor_complete.json"]
    selection = fast_campaign.frozen(saved, {"city": city, "selected_sources": len(rows),
        "mining_provenance": provenance, "files_sha256": {str(p): common.file_sha256(p) for p in files}})
    return rows, selection


def load_selection(directory):
    directory = Path(directory).resolve()
    config = config_file(directory / "all_city_selection.json")
    request = config_file(directory / "request_config.json")
    check_files(request)
    check_files(config)
    rows = common.read_jsonl(Path(config["additional_sources"]))
    if len(rows) != config["additional_images"] or {r["city"] for r in rows} != set(request["new_cities"]):
        raise ValueError("Full source selection is incomplete or lacks a new city")
    for selection_path in config["city_selections"]:
        check_files(config_file(Path(selection_path)))
    return rows, config, request


def execute(directory, config):
    if (directory / "all_city_selection.json").exists():
        load_selection(directory)
        print("All eligible sources already selected; no model loaded", flush=True)
        return
    if subprocess.run(["systemctl", "--user", "is-active", "--quiet", "adaptvpr-lightx2v.service"]).returncode == 0:
        raise RuntimeError("Stop Qwen before source mining to free GPU/RAM")
    rows, _ = scale_campaign.load_collection(Path(config["parent"]))
    new, selections = [], []
    for city in config["new_cities"]:
        common.write_json(directory / "progress.json", {"state": "mining", "current_city": city,
            "completed_new_cities": len(selections), "target_new_cities": len(config["new_cities"]),
            "additional_sources_selected": len(new)})
        sources, _ = mine_city(directory, city, config)
        new.extend(sources)
        selections.append(str(directory / "cities" / city / "all_selection.json"))
        gc.collect()
    scale_campaign.validate_unique([{**r, "sample_id": r.get("sample_id", "source_" + r["source_sha256"])} for r in rows + new])
    counts = Counter((r["city"], r["place_id"]) for r in rows + new)
    if max(counts.values()) > config["selection"]["max_sources_per_place"]:
        raise ValueError("All-city sources exceed the cumulative per-place cap")
    additional = directory / "additional_sources.jsonl"
    common.write_jsonl(additional, new)
    selected = fast_campaign.frozen(directory / "all_city_selection.json", {
        "request_fingerprint": config["fingerprint"], "cities": config["cities"],
        "parent_images": len(rows), "additional_images": len(new), "total_images": len(rows) + len(new),
        "additional_sources": str(additional), "city_source_counts": dict(Counter(r["city"] for r in rows + new)),
        "city_selections": selections, "files_sha256": {str(p): common.file_sha256(p) for p in
            [additional, directory / "request_config.json", *map(Path, selections)]}})
    report = {"state": "complete", "cities": config["cities"], "city_source_counts": selected["city_source_counts"],
        "additional_images": len(new), "total_images": len(rows) + len(new),
        "estimated_new_raw_and_training_bytes": len(new) * 3_000_000,
        "free_bytes_after_mining": shutil.disk_usage(directory).free,
        "source_quota": "all eligible hard views after the original per-place cap"}
    common.write_json(directory / "progress.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("prepare", "run", "status"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--real-data", type=Path, default=common.WORKSPACE_ROOT / "dataset/gsv-cities")
    args = parser.parse_args(argv)
    directory = args.output_dir.resolve()
    if args.command == "status":
        print((directory / "progress.json").read_text())
        return
    with run.run_lock(directory):
        config = prepare(directory, args.parent.resolve(), args.real_data.resolve())
        if args.command == "run":
            execute(directory, config)


if __name__ == "__main__":
    main()
