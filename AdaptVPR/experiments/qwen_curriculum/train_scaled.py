"""Validated full-pool source-replacement training for the 7277-image cohort.

The all-successful selection intentionally includes automatic quality rejects,
matching the previous experiment. Exported admission flags are loader flags;
the original sealed quality results remain unchanged and referenced by hash.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import common, scale_campaign, extend_plan


def admission_rows(rows, selection):
    return [{"sample_id": r["sample_id"], "source_path": r["source_path"],
        "output_path": r["output_path"], "source_sha256": r["source_sha256"],
        "output_sha256": r["output_sha256"], "city": r["city"], "place_id": r["place_id"],
        "condition": r["condition"], "passed": True, "eligible_for_training": True,
        "selection_policy": selection, "admission_flags_semantics": "training inclusion, not quality re-evaluation",
        "original_quality_passed": r["passed"], "original_quality_status": r["status"],
        "original_result_sha256": r["result_sha256"], "original_execution_fingerprint": r["execution_fingerprint"]}
        for r in rows if selection == "all-successful" or (r["passed"] and r["eligible_for_training"])]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--generation-run-dir", type=Path, default=scale_campaign.DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path,
                        default=common.WORKSPACE_ROOT / "outputs/qwen_curriculum/train_7277_replace")
    parser.add_argument("--selection", choices=("all-successful", "accepted-only", "real-only"), default="all-successful")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--replacement-probability", type=float, default=1.0)
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    if args.epochs < 1 or not 0 <= args.replacement_probability <= 1:
        parser.error("Positive epochs and replacement probability in [0,1] required")
    rows, campaign = scale_campaign.load_collection(args.generation_run_dir.resolve())
    mining = json.loads(Path(campaign["mining_provenance"]["mining_config"]).read_text())
    request = mining["descriptor_request"]
    real_data = Path(request["real_data"])
    for city, digest in request["metadata_sha256"].items():
        if common.file_sha256(real_data / "Dataframes" / f"{city}.csv") != digest:
            raise ValueError("GSV metadata changed since hard-source selection")
    manifest = args.generation_run_dir.resolve() / f"training_{args.selection}.jsonl"
    if args.selection != "real-only":
        extend_plan._freeze_bytes(manifest, common_payload(admission_rows(rows, args.selection)))
    command = [sys.executable, "-u", str(common.SALAD_ROOT / "train_salad.py"),
        "--real-data", str(real_data), "--cities", *request["cities"],
        "--output-dir", str(args.output_dir.resolve()), "--epochs", str(args.epochs),
        "--batch-size", "8", "--images-per-place", "4", "--min-images-per-place", "4",
        "--learning-rate", "6e-5", "--weight-decay", "0", "--num-trainable-blocks", "4",
        "--precision", "32", "--device", "cuda", "--num-workers", "0", "--no-augment",
        "--seed", "42", "--save-every", str(args.epochs), "--backbone", "dinov2_vitb14",
        "--backbone-repo", str(Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main")]
    if args.resume:
        command += ["--resume", str(args.resume.resolve())]
    else:
        command += ["--backbone-weights", str(Path.home() / ".cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth")]
    if args.selection != "real-only":
        command += ["--synthetic-manifest", str(manifest), "--synthetic-mode", "replace",
                    "--synthetic-fraction", str(args.replacement_probability)]
    if args.check_data:
        command += ["--check-data"]
    print(json.dumps({"generation_images": len(rows), "selection": args.selection,
        "selected_images": 0 if args.selection == "real-only" else len(admission_rows(rows, args.selection)),
        "replacement_probability": args.replacement_probability,
        "probability_semantics": "conditional on selected source having a generated variant; not overall synthetic ratio",
        "command": command}, ensure_ascii=False), flush=True)
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    subprocess.run(command, check=True, cwd=common.WORKSPACE_ROOT, env=environment)


def common_payload(rows):
    return "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows).encode()


if __name__ == "__main__":
    main()
