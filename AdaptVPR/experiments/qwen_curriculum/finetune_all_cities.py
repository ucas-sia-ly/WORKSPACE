"""Original pretrained SALAD: accepted-only generation, full 23-city real pool."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
from pathlib import Path
import sys
import subprocess

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import campaign, common, finetune_pretrained as base
from experiments.qwen_curriculum.campaign import config_file, frozen, run_lock

DEFAULT_OUTPUT = common.WORKSPACE_ROOT / "outputs/qwen_curriculum/state/training"


def accepted_cities(rows, cities):
    accepted = [r for r in rows if r.get("passed") is True and r.get("eligible_for_training") is True]
    missing = set(cities) - {r["city"] for r in accepted}
    if missing:
        raise ValueError(f"No approved generated images for cities: {sorted(missing)}; do not train a partial-city cohort")
    if any(r.get("weather_ok") is not True for r in accepted):
        raise ValueError("Accepted records lack the historical weather check")
    return accepted


def load_training_inputs(campaign):
    """Verify the independent, immutable data/model snapshot for this campaign."""
    if not campaign.get("training_inputs"):
        raise ValueError("Current campaign requires a frozen training_inputs snapshot")
    path = Path(campaign["training_inputs"])
    request = config_file(path)
    if type(request.get("schema_version")) is not int or request["schema_version"] != 1:
        raise ValueError("Unsupported training input snapshot schema")
    if any(key not in request for key in ("real_data", "checkpoint", "backbone_repo", "cities", "files_sha256")):
        raise ValueError("Training input snapshot lacks data, model or city provenance")
    if any(not isinstance(request[key], str) or not Path(request[key]).is_absolute()
           for key in ("real_data", "checkpoint", "backbone_repo")):
        raise ValueError("Training input data/model paths must be absolute strings")
    cities = request["cities"]
    if (not isinstance(cities, list) or not cities or any(not isinstance(city, str) or not city for city in cities)
            or len(set(cities)) != len(cities)
            or cities != campaign["cities"]):
        raise ValueError("Training input cities differ from the complete generation campaign")
    files = request["files_sha256"]
    if not isinstance(files, dict) or not files:
        raise ValueError("Training input snapshot has no frozen files")
    if any(not isinstance(name, str) or not Path(name).is_absolute()
           or not isinstance(digest, str) or len(digest) != 64 for name, digest in files.items()):
        raise ValueError("Training input frozen files need absolute paths and SHA256 strings")
    required = [Path(request["checkpoint"])]
    required += [Path(request["real_data"]) / "Dataframes" / f"{city}.csv" for city in cities]
    if any(str(file) not in files for file in required):
        raise ValueError("Training input snapshot must freeze the original checkpoint and every city CSV")
    for name, digest in files.items():
        file = Path(name)
        if not file.is_file() or common.file_sha256(file) != digest:
            raise ValueError(f"Frozen training input missing or changed: {file}")
    if not Path(request["real_data"]).is_dir() or not Path(request["backbone_repo"]).is_dir():
        raise ValueError("Training real-data or backbone repository directory is missing")
    return request


def discover_cities(real_data):
    cities = sorted(path.stem for path in (real_data / "Dataframes").glob("*.csv"))
    if len(cities) != 23:
        raise ValueError(f"Expected 23 GSV cities, found {len(cities)}")
    if any(not (real_data / "Images" / city).is_dir() for city in cities):
        raise ValueError("A metadata city lacks its Images directory")
    return cities


def prepare(directory, generation, epochs):
    path = directory / "experiment_config.json"
    if path.exists():
        cfg = config_file(path)
        if cfg["generation_dir"] != str(generation) or cfg["epochs"] != epochs:
            raise ValueError("Use a new experiment directory for different generation/epochs")
        base.check_inputs(cfg)
        return cfg
    rows, generation_config = campaign.load_collection(generation)
    request = load_training_inputs(generation_config)
    real_data = Path(request["real_data"])
    cities = discover_cities(real_data)
    if cities != generation_config["cities"]:
        raise ValueError("All-city generation no longer matches the installed GSV cities")
    accepted = accepted_cities(rows, cities)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "accepted_manifest.jsonl"
    common.write_jsonl(manifest, accepted)
    common.use_salad()
    from workflow.training_data import MixedGSVCitiesDataset
    dataset = MixedGSVCitiesDataset(real_data, manifest, cities=cities, synthetic_mode="replace",
                                    synthetic_fraction=1.0, augment=False)
    if dataset.summary["num_synthetic_images"] != len(accepted) or {p.city for p in dataset.places} != set(cities):
        raise ValueError("The loader dropped accepted variants or lacks a real training city")
    for row in accepted:
        if dataset.source_index.get(Path(row["source_path"])) != (row["city"], row["place_id"]):
            raise ValueError("Generated image label differs from its exact real source")
    common.write_json(directory / "accepted_summary.json", {"generated_images": len(rows),
        "accepted_images": len(accepted), "excluded_images": len(rows) - len(accepted),
        "conditions": dict(Counter(r["condition"] for r in accepted)),
        "cities": dict(Counter(r["city"] for r in accepted)), "training_pool": dataset.summary})
    count = len(accepted)
    del rows, accepted, dataset
    gc.collect()
    checkpoint = Path(request["checkpoint"])
    backbone = Path(request["backbone_repo"])
    code = [Path(__file__), Path(base.__file__), Path(common.__file__), Path(campaign.__file__),
        common.SALAD_ROOT / "train_salad.py"]
    code += [common.CURRICULUM_ROOT / name for name in
        ("run.py", "resume_recovery.py", "image_store.py", "quality.py", "weather_signal.py",
         "evaluate_pretrained_finetune.py")]
    code += sorted((common.SALAD_ROOT / "workflow").glob("*.py"))
    code += sorted((common.SALAD_ROOT / "models/aggregators").glob("*.py"))
    code += sorted(backbone.rglob("*.py"))
    files = code + [checkpoint, manifest, Path(generation_config["training_inputs"]),
        generation / "all_city_campaign_config.json", generation / "runtime_config.json",
        generation / "results.jsonl", generation / "training_manifest.jsonl", generation / "summary.json",
        *[real_data / "Dataframes" / f"{city}.csv" for city in cities]]
    files += sorted((generation / "imported").glob("*.json"))
    files += sorted((generation / "imported").glob("*.jsonl"))
    for name in ("*config.json", "intent.json", "input.jsonl"):
        files += sorted((generation / "resume_state").rglob(name))
    cfg = frozen(path, {"schema_version": 1, "experiment": "original_salad_23city_accepted_qwen",
        "generation_dir": str(generation), "generation_fingerprint": generation_config["fingerprint"],
        "initial_checkpoint": str(checkpoint), "accepted_manifest": str(manifest), "accepted_images": count,
        "real_data": str(real_data), "cities": cities, "epochs": epochs,
        "training_dir": str(directory / "finetuned"), "backbone_repo": str(backbone),
        "svox_root": str(common.WORKSPACE_ROOT / "dataset/svox"),
        "evaluation": {"domains": ["day", "night", "rain", "snow", "sun", "overcast"], "image_size": [322, 322]},
        "files_sha256": {str(p.resolve()): common.file_sha256(p) for p in files}})
    common.write_json(directory / "initialization_audit.json", base.checkpoint_audit(cfg, initial=True))
    base.publish(directory, cfg, "prepared")
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("prepare", "run", "check-data", "status"))
    from experiments.qwen_curriculum.resume_control import configured_output_dir
    parser.add_argument("--generation-dir", type=Path, default=configured_output_dir())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args(argv)
    if args.epochs <= 0:
        parser.error("Epochs must be positive")
    directory = args.output_dir.resolve()
    if args.command == "status":
        print((directory / "progress.json").read_text())
        return
    with run_lock(directory):
        cfg = prepare(directory, args.generation_dir.resolve(), args.epochs)
        if args.command == "check-data":
            subprocess.run(base.training_command(cfg, check_data=True), check=True, cwd=common.WORKSPACE_ROOT)
        elif args.command == "run":
            try:
                base.execute(directory, cfg)
            except BaseException as exc:
                base.publish(directory, cfg, "stopped", error=f"{type(exc).__name__}: {exc}")
                raise


if __name__ == "__main__":
    main()
