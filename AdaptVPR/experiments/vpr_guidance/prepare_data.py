"""Cache released IC-Light outputs and SALAD source descriptors for generator training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
from PIL import Image

from .data import (GSVLabelIndex, image_index, read_global_prompts,
                   require_empty_output, resolve_source, validate_source_label)
from .iclight import generate_released, load_iclight, released_negative_prompt, sampling_policy
from .teacher import file_sha256, load_salad


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--dataframe-dir", type=Path, default=None,
                   help="GSV Dataframes; defaults to <image-root>/../Dataframes")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--conditions", nargs="*", default=[])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--salad-repo", default="serizba/salad")
    args = p.parse_args()

    rows = read_global_prompts(args.prompts, args.conditions, args.limit)
    images = image_index(args.image_root)
    labels = GSVLabelIndex(args.dataframe_dir or args.image_root.resolve().parent / "Dataframes")
    sources = []
    for row in rows:
        src_path = resolve_source(row, images)
        city, place_id = validate_source_label(row, src_path, labels)
        sampling_policy(row["condition"])
        with Image.open(src_path) as image:
            image.verify()
        sources.append((row, src_path, city, place_id))
    out = args.output_dir.resolve()
    require_empty_output(out)
    (out / "baseline").mkdir(parents=True, exist_ok=True)
    (out / "source_salad").mkdir(exist_ok=True)
    t2i, i2i, vae = load_iclight(); teacher = load_salad(repo=args.salad_repo)
    manifest = out / "generator_train.jsonl"
    # A complete new run is required. Existing caches are never reused implicitly.
    # Publish the manifest only after every row's files and metadata are ready.
    partial_manifest = out / "generator_train.jsonl.partial"
    with partial_manifest.open("x", encoding="utf-8") as handle:
        for row, src_path, city, place_id in sources:
            with Image.open(src_path) as image:
                source = image.convert("RGB")
            negative = released_negative_prompt(row.get("negative_prompt"))
            baseline = generate_released(t2i, i2i, vae, source, row["prompt"], negative, args.seed, row["condition"])
            baseline_path = out / "baseline" / f"{row['sample_id']}.png"
            desc_path = out / "source_salad" / f"{row['sample_id']}.pt"
            baseline.save(baseline_path)
            with torch.no_grad():
                descriptor = teacher.from_pil(source)
            teacher.save_source_descriptor(desc_path, descriptor, src_path)
            rec = dict(row, source_path=str(src_path), baseline_path=str(baseline_path),
                       source_descriptor=str(desc_path), negative_prompt=negative, seed=args.seed,
                       city=city, place_id=place_id, baseline_sha256=file_sha256(baseline_path),
                       source_sha256=file_sha256(src_path), sampling_policy=sampling_policy(row["condition"]))
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"prepared {row['sample_id']} ({row['condition']})")
    partial_manifest.replace(manifest)


if __name__ == "__main__":
    main()
