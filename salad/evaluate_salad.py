#!/usr/bin/env python3
"""Evaluate a SALAD checkpoint using explicit paths and export hard queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate SALAD Recall@1/5/10 and export exact-ranked hard queries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", required=True, choices=[
        "SVOX", "Nordland", "RobotCar", "RobotCar-Seasons", "manifest", "SPED", "MSLS",
        "pitts30k_val", "pitts30k_test", "pitts250k_test",
    ])
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Evaluation results JSON.")
    parser.add_argument("--eval-manifest", type=Path,
                        help="Explicit JSON references/queries/positive IDs; overrides dataset defaults.")
    parser.add_argument("--save-hard-cases", action="store_true",
                        help="Include all Recall@1 failures in error_queries, with exact positive rank.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--query-subdirs", nargs="+", help="SVOX query folders, e.g. queries_night queries_rain.")
    parser.add_argument("--positive-radius", type=float, default=25.0, help="SVOX positive radius in meters.")
    parser.add_argument("--frame-window", type=int, default=10, help="Local Nordland positive frame tolerance.")
    parser.add_argument("--metadata-root", type=Path,
                        help="Override bundled .npy metadata root for legacy benchmark layouts.")
    parser.add_argument("--device", default="auto", help="Descriptor device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--retrieval-device", help="Device for distance computation; defaults to --device.")
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"),
                        help="Override checkpoint image size; use multiples of the backbone patch size.")
    parser.add_argument("--backbone-repo", type=Path, help="Local DINOv2 repository for offline loading.")
    parser.add_argument("--backbone-weights", type=Path, help="Local DINOv2 weights for offline loading.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--query-chunk-size", type=int, default=32)
    parser.add_argument("--reference-chunk-size", type=int, default=4096)
    parser.add_argument("--limit-queries", type=int, help="Evaluate only the first N queries for smoke checks.")
    parser.add_argument("--check-data", action="store_true",
                        help="Validate dataset metadata and images, print summary, and exit without loading a model.")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("--batch-size must be positive and --num-workers nonnegative.")
    if args.query_chunk_size <= 0 or args.reference_chunk_size <= 0:
        parser.error("Retrieval chunk sizes must be positive.")
    if args.image_size and any(value <= 0 for value in args.image_size):
        parser.error("--image-size values must be positive.")
    return args


def validate_image_size(image_size, model_config):
    if len(image_size) != 2 or any(value <= 0 or value % 14 for value in image_size):
        raise ValueError("Image height and width must be positive multiples of the DINOv2 patch size (14).")
    clusters = model_config["agg_config"]["num_clusters"]
    patches = (image_size[0] // 14) * (image_size[1] // 14)
    if patches <= clusters:
        raise ValueError(f"Image size gives {patches} patches; SALAD requires more than {clusters} clusters.")


def evaluate(args):
    from workflow.evaluation import (build_results, exact_retrieval, extract_descriptors,
                                     load_evaluation_set)

    dataset = load_evaluation_set(
        args.dataset, args.dataset_root, manifest=args.eval_manifest, split=args.split,
        query_subdirs=args.query_subdirs, positive_radius=args.positive_radius,
        frame_window=args.frame_window, metadata_root=args.metadata_root,
        limit_queries=args.limit_queries,
    )
    print(json.dumps(dataset.summary(), ensure_ascii=False, indent=2))
    if args.check_data:
        return 0

    import torch
    from workflow.model import load_checkpoint_model

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    retrieval_device = args.retrieval_device or device
    if retrieval_device == "auto":
        retrieval_device = "cuda" if torch.cuda.is_available() else "cpu"
    for value in (device, retrieval_device):
        parsed_device = torch.device(value)
        if parsed_device.type not in {"cpu", "cuda"}:
            raise ValueError("Evaluation supports cpu and cuda devices.")
        if parsed_device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(f"Requested {value}, but CUDA is unavailable; pass --device cpu.")
    model = load_checkpoint_model(args.checkpoint, device=device, backbone_repo=args.backbone_repo,
                                  backbone_weights=args.backbone_weights)
    image_size = tuple(args.image_size or model.image_size)
    validate_image_size(image_size, model.config)
    print(f"Extracting {len(dataset.references)} reference descriptors on {device}...")
    references = extract_descriptors(model, dataset.references, image_size, device,
                                    args.batch_size, args.num_workers)
    print(f"Extracting {len(dataset.queries)} query descriptors on {device}...")
    queries = extract_descriptors(model, dataset.queries, image_size, device,
                                 args.batch_size, args.num_workers)
    print(f"Computing exact positive ranks on {retrieval_device}...")
    retrieval = exact_retrieval(references, queries, dataset.positives, device=retrieval_device,
                               query_chunk_size=args.query_chunk_size,
                               reference_chunk_size=args.reference_chunk_size)
    result = build_results(dataset, retrieval, args.save_hard_cases)
    result.update({"checkpoint": str(args.checkpoint.resolve()), "image_size": list(image_size),
                   "descriptor_dimension": references.shape[1]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(" ".join(f"{key}={value:.4f}" for key, value in result["recall"].items()))
    print(f"Saved evaluation results to {args.output}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    try:
        return evaluate(args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(f"Evaluation failed: {exc}") from None


if __name__ == "__main__":
    raise SystemExit(main())
