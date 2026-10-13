"""Original pretrained SALAD: accepted-only generation, full 23-city real pool."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import sys
import subprocess

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import common, run, all_city_campaign, mine_all_cities, fast_campaign, finetune_pretrained as base
from experiments.qwen_curriculum.campaign import config_file

DEFAULT_OUTPUT = common.WORKSPACE_ROOT / "outputs/qwen_curriculum/finetune_salad_pretrained_23city"


def accepted_cities(rows, cities):
    accepted = [r for r in rows if r.get("passed") is True and r.get("eligible_for_training") is True]
    missing = set(cities) - {r["city"] for r in accepted}
    if missing:
        raise ValueError(f"No approved generated images for cities: {sorted(missing)}; do not train a partial-city cohort")
    if any(r.get("weather_ok") is not True for r in accepted):
        raise ValueError("Accepted records lack the historical weather check")
    return accepted


def prepare(directory, generation, epochs):
    path = directory / "experiment_config.json"
    if path.exists():
        cfg = config_file(path)
        if cfg["generation_dir"] != str(generation) or cfg["epochs"] != epochs:
            raise ValueError("Use a new experiment directory for different generation/epochs")
        base.check_inputs(cfg)
        return cfg
    rows, campaign = all_city_campaign.load_collection(generation)
    _, _, request = mine_all_cities.load_selection(Path(campaign["mining_dir"]))
    real_data = Path(request["real_data"])
    cities = mine_all_cities.discover_cities(real_data)
    if cities != campaign["cities"]:
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
    code = [Path(__file__), Path(base.__file__), Path(all_city_campaign.__file__), Path(mine_all_cities.__file__),
        common.SALAD_ROOT / "train_salad.py", common.CURRICULUM_ROOT / "evaluate_pretrained_finetune.py"]
    code += sorted((common.SALAD_ROOT / "workflow").glob("*.py"))
    code += sorted((common.SALAD_ROOT / "models/aggregators").glob("*.py"))
    code += sorted(backbone.rglob("*.py"))
    files = code + [checkpoint, manifest, generation / "all_city_campaign_config.json", generation / "results.jsonl",
        generation / "summary.json", *[real_data / "Dataframes" / f"{city}.csv" for city in cities]]
    cfg = fast_campaign.frozen(path, {"schema_version": 1, "experiment": "original_salad_23city_accepted_qwen",
        "generation_dir": str(generation), "generation_fingerprint": campaign["fingerprint"],
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
    parser.add_argument("--generation-dir", type=Path, default=all_city_campaign.DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args(argv)
    if args.epochs <= 0:
        parser.error("Epochs must be positive")
    directory = args.output_dir.resolve()
    if args.command == "status":
        print((directory / "progress.json").read_text())
        return
    with run.run_lock(directory):
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
