#!/usr/bin/env python3
"""Build static local reliability targets once with a frozen SALAD DINO teacher."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from workflow.model import atomic_save_checkpoint, load_checkpoint_model
from workflow.reliability import build_local_targets
from workflow.reliability_cache import (IMAGE_PREPROCESSING, TARGET_ALGORITHM,
                                        file_sha256, load_teacher_image)
from workflow.training_data import MixedGSVCitiesDataset


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-data", type=Path, required=True)
    parser.add_argument("--synthetic-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Fixed teacher checkpoint; the teacher is frozen and used offline only")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbone-repo", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8, help="Source/generated pairs per teacher batch")
    parser.add_argument("--image-size", nargs=2, type=int, default=[224, 224], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--cities", nargs="+")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if any(n <= 0 or n % 14 for n in args.image_size):
        parser.error("--image-size must contain two positive multiples of 14")
    return args


@torch.no_grad()
def build_cache(args):
    output_dir = args.output_dir.expanduser().resolve()
    if (output_dir / "index.json").exists():
        raise FileExistsError(f"Completed reliability cache already exists: {output_dir}")
    if output_dir.exists() and any(p.name != "targets.pt" for p in output_dir.iterdir()):
        raise FileExistsError("Cache output directory must be empty or contain only an incomplete targets.pt")
    manifest = args.synthetic_manifest.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    manifest_hash, teacher_hash = file_sha256(manifest), file_sha256(checkpoint)
    image_size = tuple(args.image_size)
    dataset = MixedGSVCitiesDataset(
        args.real_data, manifest, cities=args.cities, image_size=image_size,
        images_per_place=2, min_images_per_place=2, augment=False)
    pairs = [(output, place.source_by_synthetic_path[output])
             for place in dataset.places for output in sorted(place.synthetic_paths)]
    if not pairs:
        raise ValueError("No accepted selected-city synthetic source pairs to cache")
    entries = [{"output_path": str(output), "source_path": str(source),
                "output_sha256": file_sha256(output), "source_sha256": file_sha256(source)}
               for output, source in pairs]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    teacher = load_checkpoint_model(checkpoint, device, backbone_repo=args.backbone_repo)
    teacher.requires_grad_(False).eval()
    all_targets, all_confidence = [], []
    counts = {"positive_patches": 0, "negative_patches": 0, "unknown_patches": 0}
    for start in range(0, len(pairs), args.batch_size):
        batch = pairs[start:start + args.batch_size]
        sources = torch.stack([load_teacher_image(source, image_size) for _, source in batch])
        generated = torch.stack([load_teacher_image(output, image_size) for output, _ in batch])
        # Teacher stays FP32/eval, independent of the student's training updates.
        features, _ = teacher.backbone(torch.cat((sources, generated)).to(device))
        targets, confidence, diagnostics = build_local_targets(features[:len(batch)], features[len(batch):])
        all_targets.append(targets.cpu())
        all_confidence.append(confidence.cpu())
        for key in counts:
            counts[key] += int(diagnostics[key])
        print(json.dumps({"cached_pairs": start + len(batch), "total_pairs": len(pairs), **counts}), flush=True)
    # Do not publish a cache whose image/model/manifest bytes changed mid-build.
    if file_sha256(manifest) != manifest_hash or file_sha256(checkpoint) != teacher_hash:
        raise ValueError("Teacher checkpoint or manifest changed while building the cache")
    for entry in entries:
        for kind in ("source", "output"):
            if file_sha256(Path(entry[f"{kind}_path"])) != entry[f"{kind}_sha256"]:
                raise ValueError("Source/generated image changed while building the cache")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_save_checkpoint({"targets": torch.cat(all_targets), "confidence": torch.cat(all_confidence)},
                           output_dir / "targets.pt")
    total_patches = sum(counts.values())
    index = {"format_version": 1, "target_algorithm": TARGET_ALGORITHM,
             "image_preprocessing": IMAGE_PREPROCESSING, "patch_stride": 14,
             "image_size": list(image_size), "teacher_checkpoint": str(checkpoint),
             "teacher_checkpoint_sha256": teacher_hash, "teacher_model_config": teacher.config,
             "manifest": str(manifest), "manifest_sha256": manifest_hash,
             "cities": dataset.cities, "tensor_file": "targets.pt",
             "tensor_sha256": file_sha256(output_dir / "targets.pt"), "entries": entries,
             "summary": {"pairs": len(pairs), **counts,
                         "supervised_fraction": (counts["positive_patches"] + counts["negative_patches"]) / total_patches}}
    temporary = output_dir / f".index.{os.getpid()}.tmp"
    try:
        temporary.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(output_dir / "index.json")
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"output_dir": str(output_dir), **index["summary"]}), flush=True)
    return index


def main(argv=None):
    return build_cache(parse_args(argv))


if __name__ == "__main__":
    main()
