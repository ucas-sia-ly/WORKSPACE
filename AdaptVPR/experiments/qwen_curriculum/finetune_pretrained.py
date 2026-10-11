"""Fine-tune the original pretrained SALAD using only audited Qwen successes.

The accepted snapshot, full checkpoint initialization and paired six-domain
SVOX protocol are frozen before training. The final epoch is chosen in advance;
SVOX results are reporting data, never a checkpoint-selection signal.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import common, run, scale_campaign, fast_campaign

DEFAULT_OUTPUT = common.WORKSPACE_ROOT / "outputs/qwen_curriculum/finetune_salad_pretrained_accepted_5004"


def training_command(config, *, check_data=False, resume=False):
    argv = [sys.executable, "-u", str(common.SALAD_ROOT / "train_salad.py"),
        "--real-data", config["real_data"], "--cities", *config["cities"],
        "--synthetic-manifest", config["accepted_manifest"], "--synthetic-mode", "replace",
        "--synthetic-fraction", "1.0", "--output-dir", config["training_dir"],
        "--epochs", str(config["epochs"]), "--batch-size", "32", "--images-per-place", "4",
        "--min-images-per-place", "4", "--image-size", "224", "224",
        "--num-workers", "0", "--num-trainable-blocks", "4", "--learning-rate", "1e-5",
        "--weight-decay", "1e-4", "--miner-margin", "0.1", "--precision", "32",
        "--device", "cuda", "--seed", "42", "--no-augment", "--save-every", str(config["epochs"]),
        "--backbone-repo", config["backbone_repo"]]
    if resume:
        argv += ["--resume", str(Path(config["training_dir"]) / "checkpoint.pt")]
    else:
        argv += ["--init-checkpoint", config["initial_checkpoint"]]
    if check_data:
        argv.append("--check-data")
    return argv


def check_inputs(config):
    for path, digest in config["files_sha256"].items():
        if common.file_sha256(Path(path)) != digest:
            raise ValueError(f"Frozen fine-tuning input changed: {path}")
    for row in common.read_jsonl(Path(config["accepted_manifest"])):
        run.validate_image(row, "source_path", "source_sha256")
        run.validate_image(row)


def checkpoint_audit(config, *, initial=False):
    """Verify architecture/provenance; initial audit compares every loaded tensor."""
    common.use_salad()
    import torch
    from workflow.model import checkpoint_state_and_config, read_checkpoint, SALADModel, load_model_state
    path = Path(config["initial_checkpoint"] if initial else Path(config["training_dir"]) / "checkpoint.pt")
    checkpoint = read_checkpoint(path)
    state, model_config = checkpoint_state_and_config(checkpoint)
    if model_config["agg_config"].get("reliability_ot") or any("reliability" in k for k in state):
        raise ValueError("Only the original SALAD architecture is authorized")
    if model_config["backbone_arch"] != "dinov2_vitb14" or (
            model_config["agg_config"]["num_clusters"], model_config["agg_config"]["cluster_dim"],
            model_config["agg_config"]["token_dim"]) != (64, 128, 256):
        raise ValueError("Unexpected original SALAD architecture")
    digest = common.file_sha256(path)
    if initial:
        model_config["backbone_config"]["num_trainable_blocks"] = 4
        model_config["image_size"] = [224, 224]
        model = SALADModel(model_config, pretrained_backbone=False, backbone_repo=Path(config["backbone_repo"]))
        load_model_state(model, state)
        loaded = model.state_dict()
        if set(loaded) != set(state) or any(not torch.equal(loaded[k], v) for k, v in state.items()):
            raise ValueError("Pretrained SALAD initialization changed model weights")
        report = {"checkpoint_sha256": digest, "state_keys": len(state),
                  "every_loaded_tensor_equal": True, "reliability_enabled": False,
                  "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        del model, loaded
    else:
        train_cfg = checkpoint["training_config"]
        if (checkpoint["epoch"] != config["epochs"] or checkpoint["total_epochs"] != config["epochs"]
                or checkpoint["precision"] != "32" or "reliability" in train_cfg
                or train_cfg["init_checkpoint_sha256"] != config["files_sha256"][config["initial_checkpoint"]]
                or train_cfg["synthetic_manifest"] != config["accepted_manifest"]
                or checkpoint["dataset_summary"]["num_synthetic_images"] != config["accepted_images"]):
            raise ValueError("Final checkpoint does not match this pretrained accepted-only experiment")
        original, _ = checkpoint_state_and_config(read_checkpoint(Path(config["initial_checkpoint"])))
        if set(original) != set(state):
            raise ValueError("Fine-tuning changed the original model architecture")
        changed = [name for name in state if not torch.equal(state[name], original[name])]
        allowed = ("aggregator.", "backbone.model.norm.") + tuple(
            f"backbone.model.blocks.{i}." for i in range(8, 12))
        if (any(not name.startswith(allowed) for name in changed)
                or not any(name.startswith("aggregator.") for name in changed)
                or not any(name.startswith("backbone.model.blocks.") for name in changed)):
            raise ValueError("Weight updates differ from the declared original SALAD fine-tuning scope")
        report = {"checkpoint_sha256": digest, "completed_epochs": checkpoint["epoch"],
                  "state_keys": len(state), "reliability_enabled": False,
                  "initial_checkpoint_sha256": train_cfg["init_checkpoint_sha256"],
                  "changed_tensors": len(changed), "frozen_tensors_unchanged": len(state) - len(changed),
                  "aggregator_updated": True, "last_backbone_blocks_updated": True,
                  "changed_tensor_names": changed}
        del original
    del state, checkpoint
    gc.collect()
    return report


def prepare(directory, generation, checkpoint, epochs):
    path = directory / "experiment_config.json"
    if path.exists():
        cfg = fast_campaign.config_file(path)
        if (cfg["generation_dir"] != str(generation) or cfg["initial_checkpoint"] != str(checkpoint)
                or cfg["epochs"] != epochs):
            raise ValueError("Use a new experiment directory for changed fine-tuning settings")
        check_inputs(cfg)
        return cfg
    rows, campaign = scale_campaign.load_collection(generation)
    accepted = [r for r in rows if r.get("passed") is True and r.get("eligible_for_training") is True]
    if not accepted or any(r.get("weather_ok") is not True for r in accepted):
        raise ValueError("Accepted subset must pass the structure and weather checks")
    if accepted != common.read_jsonl(generation / "training_manifest.jsonl"):
        raise ValueError("Published accepted manifest differs from the sealed quality records")
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "accepted_manifest.jsonl"
    common.write_jsonl(manifest, accepted)
    mining_path = Path(campaign["mining_provenance"]["mining_config"])
    request = json.loads(mining_path.read_text())["descriptor_request"]
    real_data = Path(request["real_data"])
    metadata = [real_data / "Dataframes" / f"{city}.csv" for city in request["cities"]]
    for file, city in zip(metadata, request["cities"]):
        if common.file_sha256(file) != request["metadata_sha256"][city]:
            raise ValueError("GSV labels changed since source mining")
    common.use_salad()
    from workflow.training_data import MixedGSVCitiesDataset
    dataset = MixedGSVCitiesDataset(real_data, manifest, cities=request["cities"],
        synthetic_mode="replace", synthetic_fraction=1.0, augment=False)
    if dataset.summary["num_synthetic_images"] != len(accepted):
        raise ValueError("Accepted variants were dropped or duplicated by the training loader")
    for row in accepted:
        if dataset.source_index.get(Path(row["source_path"])) != (row["city"], row["place_id"]):
            raise ValueError("Accepted image label differs from its exact real metadata source")
    dataset.summary["replacement_explanation"] = "selected accepted hard source replaces its real view; other views stay real"
    common.write_json(directory / "accepted_summary.json", {
        "generated_images": len(rows), "accepted_images": len(accepted), "excluded_images": len(rows) - len(accepted),
        "conditions": dict(Counter(r["condition"] for r in accepted)),
        "cities": dict(Counter(r["city"] for r in accepted)),
        "selected_places": len({(r["city"], r["place_id"]) for r in accepted}),
        "training_pool": dataset.summary, "quality_flags_modified": False})
    accepted_count = len(accepted)
    del dataset, accepted, rows
    gc.collect()
    backbone_repo = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
    code = [Path(__file__), common.SALAD_ROOT / "train_salad.py",
        common.CURRICULUM_ROOT / "evaluate_pretrained_finetune.py"]
    code += sorted((common.SALAD_ROOT / "workflow").glob("*.py"))
    code += sorted((common.SALAD_ROOT / "models/aggregators").glob("*.py"))
    code += sorted(backbone_repo.rglob("*.py"))
    inputs = code + metadata + [manifest, checkpoint, generation / "results.jsonl",
        generation / "summary.json", generation / "scale_campaign_config.json", mining_path]
    cfg = fast_campaign.frozen(path, {
        "schema_version": 1, "experiment": "original_pretrained_salad_accepted_qwen_finetune",
        "generation_dir": str(generation), "generation_fingerprint": campaign["fingerprint"],
        "initial_checkpoint": str(checkpoint), "accepted_manifest": str(manifest), "accepted_images": accepted_count,
        "real_data": str(real_data), "cities": request["cities"], "epochs": epochs,
        "training_dir": str(directory / "finetuned"), "backbone_repo": str(backbone_repo),
        "svox_root": str(common.WORKSPACE_ROOT / "dataset/svox"),
        "fine_tuning": {"lr": 1e-5, "weight_decay": 1e-4, "scheduler": "cosine", "last_blocks": 4,
            "batch_places": 32, "images_per_place": 4, "image_size": [224, 224], "precision": "32",
            "workers": 0, "seed": 42, "augment": False, "full_pretrained_aggregator": True,
            "reliability_ot": False, "accepted_only": True, "synthetic_mode": "replace", "replacement_probability": 1.0},
        "evaluation": {"domains": ["day", "night", "rain", "snow", "sun", "overcast"],
            "image_size": [322, 322], "precision": "32", "positive_radius_m": 25,
            "checkpoint_selection": "predeclared final epoch; no SVOX model selection"},
        "files_sha256": {str(p.resolve()): common.file_sha256(p) for p in inputs},
    })
    common.write_json(directory / "initialization_audit.json", checkpoint_audit(cfg, initial=True))
    publish(directory, cfg, "prepared")
    return cfg


def publish(directory, config, state, **extra):
    progress = {"state": state, "experiment_fingerprint": config["fingerprint"],
        "accepted_images": config["accepted_images"], "target_epochs": config["epochs"],
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}
    logs = run.read_rows(Path(config["training_dir"]), "training_log.jsonl")
    if logs:
        progress.update(completed_epochs=logs[-1]["epoch"], last_loss=logs[-1]["loss"],
                        synthetic_fraction_observed=logs[-1]["synthetic_fraction_observed"])
    common.write_json(directory / "progress.json", progress)
    return progress


def run_training(directory, config):
    common.use_salad()
    from workflow.model import read_checkpoint
    path = Path(config["training_dir"]) / "checkpoint.pt"
    resume = False
    if path.exists():
        saved = read_checkpoint(path)
        done = saved["epoch"] == config["epochs"]
        if saved["epoch"] > config["epochs"]:
            raise ValueError("Checkpoint exceeds the requested epochs")
        # A durable epoch checkpoint may precede its log after an interruption.
        logs = run.read_rows(path.parent, "training_log.jsonl")
        logs = [r for r in logs if r["epoch"] < saved["epoch"]]
        logs.append(saved["metrics"])
        common.write_jsonl(path.parent / "training_log.jsonl", logs)
        del saved
        gc.collect()
        if done:
            checkpoint_audit(config)
            return
        resume = True
    command = training_command(config, resume=resume)
    publish(directory, config, "training", command=command)
    with (directory / "training.log").open("a") as log:
        child = subprocess.Popen(command, cwd=common.WORKSPACE_ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                publish(directory, config, "training")
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
            raise RuntimeError("Fine-tuning failed; inspect training.log")
    common.write_json(directory / "final_checkpoint_audit.json", checkpoint_audit(config))


def execute(directory, config):
    check_inputs(config)
    run_training(directory, config)
    check_inputs(config)
    from experiments.qwen_curriculum.evaluate_pretrained_finetune import evaluate
    results = {}
    for arm, checkpoint in (("pretrained", Path(config["initial_checkpoint"])),
                            ("finetuned", Path(config["training_dir"]) / "checkpoint.pt")):
        publish(directory, config, f"evaluating_{arm}")
        results[arm] = evaluate(checkpoint, Path(config["svox_root"]), directory / arm / "evaluation",
            Path(config["backbone_repo"]), image_size=(322, 322), batch_size=32, num_workers=0, device="cuda")
    comparison = {}
    for domain in config["evaluation"]["domains"]:
        before, after = results["pretrained"][domain], results["finetuned"][domain]
        if (before["protocol"] != after["protocol"]
                or before["protocol_fingerprint"] != after["protocol_fingerprint"]
                or before["num_queries"] != after["num_queries"] or before["num_references"] != after["num_references"]):
            raise ValueError("Paired SVOX protocols differ")
        comparison[domain] = {"num_queries": after["num_queries"], "pretrained": before["recall"],
            "finetuned": after["recall"], "delta_percentage_points": {
                k: 100 * (after["recall"][k] - before["recall"][k]) for k in before["recall"]}}
    common.write_json(directory / "comparison.json", {
        "state": "complete", "experiment_fingerprint": config["fingerprint"],
        "accepted_images": config["accepted_images"], "epochs": config["epochs"],
        "initial_checkpoint_sha256": common.file_sha256(Path(config["initial_checkpoint"])),
        "final_checkpoint_sha256": common.file_sha256(Path(config["training_dir"]) / "checkpoint.pt"),
        "image_size": [322, 322], "precision": "32", "domains": comparison})
    publish(directory, config, "complete")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("prepare", "run", "status", "check-data"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generation-dir", type=Path, default=scale_campaign.DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=common.SALAD_ROOT / "checkpoint/dino_salad.ckpt")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args(argv)
    if args.epochs <= 0:
        parser.error("Epoch count must be positive")
    directory = args.output_dir.resolve()
    if args.command == "status":
        print((directory / "progress.json").read_text())
        return
    with run.run_lock(directory):
        config = prepare(directory, args.generation_dir.resolve(), args.checkpoint.resolve(), args.epochs)
        if args.command == "check-data":
            subprocess.run(training_command(config, check_data=True), check=True, cwd=common.WORKSPACE_ROOT)
        elif args.command == "run":
            try:
                execute(directory, config)
            except BaseException as exc:
                publish(directory, config, "stopped", error=f"{type(exc).__name__}: {exc}")
                raise


if __name__ == "__main__":
    main()
