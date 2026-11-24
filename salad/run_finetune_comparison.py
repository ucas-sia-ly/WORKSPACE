#!/usr/bin/env python3
"""One fixed small pretrained fine-tune, with matched SVOX/day and night evaluations."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--real-data", type=Path, required=True)
    parser.add_argument("--synthetic-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbone-repo", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()

    import torch
    from workflow.comparison import evaluate_svox_pair
    from workflow.model import read_checkpoint, checkpoint_state_and_config

    torch.set_num_threads(args.torch_threads)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; run with GPU device access or explicitly select CPU")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = args.output_dir / "comparison.json"
    trained_dir = args.output_dir / "finetuned"
    if report.exists() or (trained_dir / "checkpoint.pt").exists():
        raise SystemExit("This experiment already has results/checkpoint; choose a new output directory")
    source_hash = sha256(args.checkpoint)
    manifest_hash = sha256(args.synthetic_manifest)
    training_cmd = [sys.executable, str(Path(__file__).resolve().parent / "train_salad.py"),
                    "--real-data", str(args.real_data.resolve()),
                    "--synthetic-manifest", str(args.synthetic_manifest.resolve()),
                    "--init-checkpoint", str(args.checkpoint.resolve()),
                    "--output-dir", str(trained_dir.resolve()),
                    "--cities", "Bangkok", "--synthetic-places-only",
                    "--epochs", str(args.epochs), "--batch-size", str(args.train_batch_size),
                    "--images-per-place", "4", "--min-images-per-place", "4",
                    "--synthetic-fraction", "0.5", "--num-trainable-blocks", "0",
                    "--image-size", "224", "224", "--no-augment",
                    "--learning-rate", str(args.learning_rate), "--weight-decay", "0",
                    "--device", args.device, "--precision", "32",
                    "--num-workers", str(args.num_workers), "--seed", str(args.seed),
                    "--backbone-repo", str(args.backbone_repo.resolve())]
    config = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
              "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": source_hash,
              "synthetic_manifest": str(args.synthetic_manifest.resolve()), "synthetic_manifest_sha256": manifest_hash,
              "dataset_root": str(args.dataset_root.resolve()), "device": args.device,
              "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
              "torch_version": torch.__version__, "image_size": [224, 224],
              "training_command": training_cmd,
              "training_command_shell": shlex.join(training_cmd),
              "fixed_experiment": "one run; no hyperparameter search or checkpoint selection on SVOX",
              "evaluation": {"split": "test", "positive_radius_meters": 25,
                             "query_subdirs": {"SVOX": ["queries"], "SVOX_night": ["queries_night"]},
                             "batch_size": args.eval_batch_size, "num_workers": args.num_workers}}
    (args.output_dir / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("[1/3] Re-evaluating original pretrained checkpoint", flush=True)
    before = evaluate_svox_pair(args.checkpoint, args.dataset_root, args.output_dir / "before",
                                args.backbone_repo, args.device, args.eval_batch_size, args.num_workers)
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    print("[2/3] Small fine-tune:", shlex.join(training_cmd), flush=True)
    with (args.output_dir / "training_stdout.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(training_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        if process.wait() != 0:
            raise RuntimeError(f"Training failed; see {args.output_dir / 'training_stdout.log'}")
    checkpoint = trained_dir / "checkpoint.pt"
    print("[3/3] Evaluating fine-tuned checkpoint", flush=True)
    after = evaluate_svox_pair(checkpoint, args.dataset_root, args.output_dir / "after",
                               args.backbone_repo, args.device, args.eval_batch_size, args.num_workers)

    original_state, _ = checkpoint_state_and_config(read_checkpoint(args.checkpoint))
    tuned = read_checkpoint(checkpoint)
    tuned_state = tuned["state_dict"]
    changes = {"backbone": 0, "aggregator": 0}
    for key, original in original_state.items():
        if not torch.isfinite(tuned_state[key]).all():
            raise RuntimeError(f"Nonfinite fine-tuned parameter: {key}")
        if not torch.equal(original, tuned_state[key]):
            changes[key.split(".", 1)[0]] += 1
    if changes["backbone"] or not changes["aggregator"]:
        raise RuntimeError(f"Unexpected frozen-backbone/aggregator changes: {changes}")
    if sha256(args.checkpoint) != source_hash or sha256(args.synthetic_manifest) != manifest_hash:
        raise RuntimeError("Original checkpoint or frozen training manifest changed during the experiment")
    comparison = {"experiment": config, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                  "new_checkpoint": str(checkpoint.resolve()), "new_checkpoint_sha256": sha256(checkpoint),
                  "training_metrics": tuned["metrics"], "parameter_tensors_changed": changes,
                  "datasets": {}}
    for name in before:
        if before[name]["protocol"] != after[name]["protocol"] or before[name]["num_queries"] != after[name]["num_queries"]:
            raise RuntimeError(f"Before/after protocols or query counts differ for {name}")
        comparison["datasets"][name] = {
            "num_queries": before[name]["num_queries"],
            "num_references": before[name]["num_references"],
            "before": before[name]["recall"], "after": after[name]["recall"],
            "delta_percentage_points": {key: 100 * (after[name]["recall"][key] - value)
                                        for key, value in before[name]["recall"].items()},
            "before_error_queries": before[name]["num_error_queries"],
            "after_error_queries": after[name]["num_error_queries"],
        }
    report.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(comparison["datasets"], indent=2), flush=True)
    print(f"Complete: {report}", flush=True)


if __name__ == "__main__":
    main()
