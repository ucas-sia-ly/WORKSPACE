"""Prepare Global source metadata and immutable frozen-SALAD descriptors only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
from PIL import Image

from .data import (GSVLabelIndex, image_index, read_global_prompts, resolve_source,
                   validate_source_label, atomic_json)
from .iclight import released_negative_prompt, sampling_policy
from .teacher import file_sha256, load_salad, PREPROCESSING_VERSION


def prepare_sources(prompts, gsv_root, salad_root, output_dir, conditions=(), teacher=None):
    rows = read_global_prompts(Path(prompts), conditions)
    root, out = Path(gsv_root).resolve(), Path(output_dir).resolve()
    images, labels = image_index(root / "Images"), GSVLabelIndex(root / "Dataframes")
    sources = []
    for row in rows:
        source = resolve_source(row, images)
        city, pid = validate_source_label(row, source, labels)
        sampling_policy(row["condition"])
        with Image.open(source) as image:
            image.verify()
        sources.append((row, source, city, pid))
    teacher = teacher or load_salad(repo=str(Path(salad_root).resolve()))
    identity = {"prompts_sha256": file_sha256(prompts), "gsv_root": str(root),
                "conditions": sorted(conditions), "teacher_sha256": teacher.model_fingerprint,
                "preprocessing_version": PREPROCESSING_VERSION}
    config = out / "preparation_config.json"
    if config.exists() and json.loads(config.read_text()) != identity:
        raise ValueError("source cache belongs to different prompts, teacher or preprocessing")
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(config, identity)
    desc_dir = out / "source_salad"
    desc_dir.mkdir(exist_ok=True)
    manifest = out / "source_manifest.jsonl"
    partial = out / "source_manifest.jsonl.partial"
    with partial.open("w", encoding="utf-8") as handle:
        for row, source, city, pid in sources:
            desc_path = desc_dir / f"{row['sample_id']}.pt"
            if desc_path.exists():
                teacher.load_source_descriptor(desc_path, source)
            else:
                with Image.open(source) as image, torch.no_grad():
                    descriptor = teacher.from_pil(image.convert("RGB"))
                temp = desc_path.with_suffix(".partial")
                teacher.save_source_descriptor(temp, descriptor, source)
                temp.replace(desc_path)
            rec = dict(row, source_path=str(source), source_sha256=file_sha256(source),
                       city=city, place_id=pid, source_descriptor=str(desc_path),
                       teacher_sha256=teacher.model_fingerprint,
                       preprocessing_version=PREPROCESSING_VERSION,
                       negative_prompt=released_negative_prompt(row.get("negative_prompt")),
                       sampling_policy=sampling_policy(row["condition"]))
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"prepared source {row['sample_id']} ({row['condition']})", flush=True)
    partial.replace(manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--gsv-root", type=Path, required=True)
    p.add_argument("--salad-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--conditions", nargs="*", default=[])
    args = p.parse_args()
    prepare_sources(args.prompts, args.gsv_root, args.salad_root, args.output_dir, args.conditions)


if __name__ == "__main__":
    main()
