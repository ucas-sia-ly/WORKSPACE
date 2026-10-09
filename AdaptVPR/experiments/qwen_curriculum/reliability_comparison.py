"""Freeze a historical paired baseline and compare a reliability-module run.

Historical implementation hashes are provenance, not a requirement that today's
SALAD code still match. Data, training protocol, checkpoints and saved evaluation
results remain independently verified. This module never loads a model.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import common

ARMS = {"generated_8to1": (8, True), "true_8to1": (8, False),
        "generated_4to1": (4, True), "true_4to1": (4, False)}
DOMAINS = {"day": "queries", "night": "queries_night", "rain": "queries_rain",
           "snow": "queries_snow", "sun": "queries_sun", "overcast": "queries_overcast"}
QUERY_COUNTS = {"day": 14278, "night": 823, "rain": 937,
                "snow": 870, "sun": 854, "overcast": 872}
METRICS = ("R@1", "R@5", "R@10")
FROZEN_FILES = {"groups_4to1.jsonl", "groups_8to1.jsonl", "generated_700.jsonl",
                "image_inventory.jsonl"}
PATH_OPTIONS = ("generation_run_dir", "real_data", "backbone_weights", "backbone_repo", "dataset_root")
SHARED_OPTIONS = (*PATH_OPTIONS, "num_images", "epochs", "learning_rate", "trainable_blocks",
                  "batch_size", "seed", "device", "initialization", "backbone")


def _sealed(value):
    return {**value, "fingerprint": common.fingerprint(value)}


def _read_sealed(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a sealed JSON object: {path}")
    content = {key: item for key, item in value.items() if key != "fingerprint"}
    if value.get("fingerprint") != common.fingerprint(content):
        raise ValueError(f"Invalid baseline/result seal: {path}")
    return value


def _verify_sealed(value, label):
    if value.get("fingerprint") != common.fingerprint({k: v for k, v in value.items() if k != "fingerprint"}):
        raise ValueError(f"Invalid seal: {label}")


def _verify_sha(path, expected, label):
    if common.file_sha256(Path(path)) != expected:
        raise ValueError(f"Baseline bytes changed: {label}")


def _request_from_args(args):
    request = {name: str(Path(getattr(args, name)).expanduser().resolve()) for name in PATH_OPTIONS}
    request.update({name: getattr(args, name) for name in SHARED_OPTIONS
                    if name not in PATH_OPTIONS and name not in {"initialization", "backbone"}})
    request.update(initialization="random_aggregator_pretrained_backbone", backbone="dinov2_vitb14")
    return request


def _same_request(left, right):
    for name in SHARED_OPTIONS:
        if left.get(name) != right.get(name):
            raise ValueError(f"Baseline training/data option differs: {name}")


def _svox_inventory(root):
    root = Path(root)
    if (root / "evaluation_manifest.json").exists():
        raise ValueError("Native SVOX cannot have an evaluation manifest override")
    counts = {"gallery": 17166, **{DOMAINS[d]: n for d, n in QUERY_COUNTS.items()}}
    result = {}
    for folder, expected in counts.items():
        paths = sorted(p for p in (root / "images/test" / folder).iterdir()
                       if p.is_file() and p.suffix.lower() in {".jpg", ".png", ".jpeg"})
        if len(paths) != expected:
            raise ValueError(f"Native SVOX {folder} count differs")
        result[folder] = {"count": len(paths), "ordered_filename_size_mtime_sha256": common.fingerprint(
            [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in paths])}
    return result


def _validate_input_files(config, directory):
    if set(config["files_sha256"]) != FROZEN_FILES:
        raise ValueError("Baseline must have exactly the four fixed-pool input files")
    for name, digest in config["files_sha256"].items():
        _verify_sha(directory / name, digest, name)
    inventory = common.read_jsonl(directory / "image_inventory.jsonl")
    if len({row["path"] for row in inventory}) != len(inventory):
        raise ValueError("Baseline image inventory contains duplicate paths")
    for row in inventory:
        _verify_sha(row["path"], row["sha256"], row["path"])
    request = config["request"]
    _verify_sha(request["backbone_weights"], config["backbone_weights_sha256"], "DINOv2 weights")
    for city, digest in config["metadata_sha256"].items():
        _verify_sha(Path(request["real_data"]) / "Dataframes" / f"{city}.csv", digest, f"{city} metadata")
    if _svox_inventory(request["dataset_root"]) != config["svox_inventory"]:
        raise ValueError("Baseline SVOX inventory changed")
    n = request["num_images"]
    rows = common.read_jsonl(directory / "generated_700.jsonl")
    if len(rows) != n or len({row["source_path"] for row in rows}) != n:
        raise ValueError("Baseline generated-source count differs")
    inventory_by_path = {row["path"]: row for row in inventory}
    for row in rows:
        content = {key: value for key, value in row.items() if key != "result_sha256"}
        if row.get("result_sha256") != common.fingerprint(content):
            raise ValueError("Baseline generated row seal changed")
        for path, digest in (("source_path", "source_sha256"), ("output_path", "output_sha256")):
            if inventory_by_path.get(row[path], {}).get("sha256") != row[digest]:
                raise ValueError("Generated image/source differs from baseline inventory")
    groups = {ratio: common.read_jsonl(directory / f"groups_{ratio}to1.jsonl") for ratio in (4, 8)}
    if groups[4] != groups[8][:len(groups[4])]:
        raise ValueError("Baseline ratio pools are no longer nested")
    generated_sources = {row["source_path"] for row in rows}
    generated_outputs = {row["output_path"] for row in rows}
    source_pool = {p for bag in groups[8] for p in bag["sources"]}
    if (len(generated_outputs) != n or source_pool & generated_outputs
            or {row["path"] for row in inventory if row["kind"] == "source"} != source_pool
            or {row["path"] for row in inventory if row["kind"] == "generated"} != generated_outputs
            or len(inventory) != n * 10):
        raise ValueError("Baseline source/generated inventory membership differs")
    for ratio, bags in groups.items():
        paths = [p for bag in bags for p in bag["sources"]]
        if (any(len(bag["sources"]) != 4 for bag in bags) or len(paths) != n * (ratio + 1)
                or len(set(paths)) != len(paths) or len(set(paths) & generated_sources) != n
                or not set(paths) <= set(inventory_by_path)):
            raise ValueError("Baseline fixed source slots/ratios differ")


def _recall(value):
    if set(value) != set(METRICS) or any(type(value[key]) not in (int, float)
            or not math.isfinite(value[key]) or not 0 <= value[key] <= 1 for key in METRICS):
        raise ValueError("Recall must contain finite R@1/5/10 fractions")
    return value


def _validate_report(config, report):
    if report.get("state") != "complete" or report.get("experiment_fingerprint") != config["fingerprint"]:
        raise ValueError("Baseline/report is not a completed matching experiment")
    if set(config["arms"]) != set(ARMS) or set(report["checkpoint_sha256"]) != set(ARMS):
        raise ValueError("Completed experiment must contain all four paired arms")
    if config["svox_domains"] != DOMAINS or set(report["comparisons"]) != {"8", "4"}:
        raise ValueError("Completed experiment must contain both ratios and six native SVOX domains")
    _verify_sealed(report["shared_initialization"], "shared initialization")
    for ratio in (8, 4):
        if set(report["comparisons"][str(ratio)]) != set(DOMAINS):
            raise ValueError("Incomplete SVOX domain comparison")
        for domain in DOMAINS:
            row = report["comparisons"][str(ratio)][domain]
            real, generated = _recall(row["true"]), _recall(row["generated"])
            for metric in METRICS:
                if not math.isclose(row["delta_percentage_points"][metric],
                                    100 * (generated[metric] - real[metric]), abs_tol=1e-10):
                    raise ValueError("Reported paired benefit differs from saved recalls")


def _validate_results(config, report, directory):
    n, epochs = config["request"]["num_images"], config["request"]["epochs"]
    initialization = []
    for arm, (ratio, replace) in ARMS.items():
        spec = config["arms"][arm]
        expected = {"ratio": ratio, "replace": replace, "source_slots": n * (ratio + 1),
                    "generated_per_epoch": n if replace else 0,
                    "true_per_epoch": n * (ratio if replace else ratio + 1),
                    "schedule": f"groups_{ratio}to1.jsonl"}
        if spec != expected:
            raise ValueError(f"Baseline arm specification differs: {arm}")
        _verify_sha(directory / arm / "checkpoint.pt", report["checkpoint_sha256"][arm], f"{arm} checkpoint")
        logs = common.read_jsonl(directory / arm / "training_log.jsonl")
        if ([row["epoch"] for row in logs] != list(range(1, epochs + 1)) or any(
                (row["real_exposure"], row["synthetic_exposure"]) !=
                (spec["true_per_epoch"], spec["generated_per_epoch"]) for row in logs)):
            raise ValueError(f"Baseline epochs/exposure differs: {arm}")
        initial = _read_sealed(directory / arm / "initialization.json")
        if (initial["experiment_fingerprint"] != config["fingerprint"]
                or initial["backbone_weights_sha256"] != config["backbone_weights_sha256"]
                or initial["init_policy"] != config["request"]["initialization"]):
            raise ValueError(f"Baseline initialization provenance differs: {arm}")
        initialization.append(initial)
        for domain, folder in DOMAINS.items():
            row = _read_sealed(directory / arm / "evaluation" / f"SVOX_{domain}.json")
            if (row["checkpoint_sha256"] != report["checkpoint_sha256"][arm]
                    or row["experiment_fingerprint"] != config["fingerprint"]
                    or row["protocol"] != {"ground_truth": "utm_radius", "positive_radius_meters": 25,
                                           "split": "test", "query_subdirs": [folder]}
                    or row["num_references"] != 17166 or row["num_queries"] != QUERY_COUNTS[domain]
                    or row["num_evaluated_queries"] != QUERY_COUNTS[domain]
                    or row["num_queries_without_positives"] != 0 or row["image_size"] != [224, 224]
                    or row["query_folder"] != folder
                    or _recall(row["recall"]) != report["comparisons"][str(ratio)][domain][
                        "generated" if replace else "true"]):
                raise ValueError(f"Baseline SVOX evaluation differs: {arm}/{domain}")
    for key in ("initial_backbone_state_sha256", "initial_aggregator_state_sha256"):
        if len({row[key] for row in initialization}) != 1 or any(
                row[key] != report["shared_initialization"][key] for row in initialization):
            raise ValueError(f"Baseline arms did not share initial weights: {key}")


def freeze_baseline(directory: Path, output: Path, args) -> dict:
    """Verify the old experiment and create immutable provenance snapshots."""
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if directory == output or directory in output.parents:
        raise ValueError("Baseline snapshots need a separate new experiment directory")
    config = _read_sealed(directory / "experiment_config.json")
    report = _read_sealed(directory / "comparison.json")
    _same_request(config["request"], _request_from_args(args))
    _validate_report(config, report)
    _validate_input_files(config, directory)
    _validate_results(config, report, directory)
    names = {"experiment_config.json": directory / "experiment_config.json",
             "comparison.json": directory / "comparison.json"}
    names.update({f"{arm}_initialization.json": directory / arm / "initialization.json" for arm in ARMS})
    hashes = {}
    # Check all existing files before making any new snapshot writes.
    for name, source in names.items():
        target = output / "baseline" / name
        digest = common.file_sha256(source)
        if target.exists():
            _verify_sha(target, digest, f"existing baseline snapshot {name}")
        hashes[name] = digest
    for name, source in names.items():
        target = output / "baseline" / name
        if not target.exists():
            # Preserve the actual sealed JSON and its formatting/bytes exactly.
            common._atomic_write(target, [source.read_text(encoding="utf-8")])
        _verify_sha(target, hashes[name], f"baseline snapshot {name}")
    return {"source_directory": str(directory), "experiment_fingerprint": config["fingerprint"],
            "report_fingerprint": report["fingerprint"], "snapshot_files_sha256": hashes}


def build_module_comparison(config: dict, report: dict, output: Path) -> dict:
    """Compare each arm and the change in generated-minus-real paired benefit."""
    _verify_sealed(config, "new experiment config")
    _verify_sealed(report, "new experiment report")
    _validate_report(config, report)
    descriptor = config["baseline"]
    root = Path(output) / "baseline"
    expected_names = {"experiment_config.json", "comparison.json",
                      *(f"{arm}_initialization.json" for arm in ARMS)}
    if set(descriptor["snapshot_files_sha256"]) != expected_names:
        raise ValueError("Baseline snapshot is incomplete")
    for name, digest in descriptor["snapshot_files_sha256"].items():
        _verify_sha(root / name, digest, f"baseline snapshot {name}")
    old_config, old = _read_sealed(root / "experiment_config.json"), _read_sealed(root / "comparison.json")
    _validate_report(old_config, old)
    if (descriptor["experiment_fingerprint"] != old_config["fingerprint"]
            or descriptor["report_fingerprint"] != old["fingerprint"]):
        raise ValueError("Baseline snapshot differs from descriptor")
    _same_request(old_config["request"], config["request"])
    for key in ("files_sha256", "backbone_weights_sha256", "metadata_sha256", "svox_inventory",
                "svox_domains", "evaluation_protocol", "arms"):
        if old_config[key] != config[key]:
            raise ValueError(f"New and old paired protocols differ: {key}")
    if any(config["training"].get(key) != value for key, value in old_config["training"].items()):
        raise ValueError("New and old paired protocols differ: training")
    initial = report["shared_initialization"]
    standard = initial.get("standard_aggregator_state_sha256", config.get("standard_aggregator_state_sha256"))
    if standard is None:
        standard = config.get("shared_initialization", {}).get("standard_aggregator_state_sha256")
    if (standard != old["shared_initialization"]["initial_aggregator_state_sha256"]
            or initial["initial_backbone_state_sha256"] != old["shared_initialization"]["initial_backbone_state_sha256"]):
        raise ValueError("Original SALAD/DINOv2 initial weights differ from baseline")
    comparisons = {}
    for ratio in (8, 4):
        comparisons[str(ratio)] = {}
        for domain in DOMAINS:
            before, after = old["comparisons"][str(ratio)][domain], report["comparisons"][str(ratio)][domain]
            arms = {name: {"old": before[name], "new": after[name], "delta_percentage_points": {
                        metric: 100 * (after[name][metric] - before[name][metric]) for metric in METRICS}}
                    for name in ("true", "generated")}
            previous, current = before["delta_percentage_points"], after["delta_percentage_points"]
            comparisons[str(ratio)][domain] = {"arms": arms, "paired_benefit_percentage_points": {
                "old": previous, "new": current, "benefit_change": {
                    metric: current[metric] - previous[metric] for metric in METRICS}}}
    return _sealed({"state": "complete", "experiment_fingerprint": config["fingerprint"],
                    "report_fingerprint": report["fingerprint"],
                    "baseline_experiment_fingerprint": old_config["fingerprint"],
                    "baseline_report_fingerprint": old["fingerprint"],
                    "units": "recall fractions for old/new; percentage points for deltas and paired benefits",
                    "comparisons": comparisons})
