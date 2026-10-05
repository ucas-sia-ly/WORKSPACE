"""Cache released IC-Light outputs and SALAD source descriptors for generator training."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import torch
from PIL import Image

from AdaptVPR.adapters import iclight_sd15_fc as adapter
from .data import image_index, read_global_prompts, resolve_source
from .iclight import generate_released, load_iclight
from .teacher import load_salad


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--conditions", nargs="*", default=[])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--salad-repo", default="serizba/salad")
    args = p.parse_args()

    rows = read_global_prompts(args.prompts, args.conditions, args.limit)
    images = image_index(args.image_root)
    out = args.output_dir.resolve(); (out / "baseline").mkdir(parents=True, exist_ok=True); (out / "source_salad").mkdir(exist_ok=True)
    t2i, i2i, vae = load_iclight(); teacher = load_salad(repo=args.salad_repo)
    manifest = out / "generator_train.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            src_path = resolve_source(row, images)
            source = Image.open(src_path).convert("RGB")
            negative = row.get("negative_prompt") or adapter.DEFAULT_NEGATIVE_PROMPT
            baseline = generate_released(t2i, i2i, vae, source, row["prompt"], negative, args.seed, row["condition"])
            baseline_path = out / "baseline" / f"{row['sample_id']}.png"
            desc_path = out / "source_salad" / f"{row['sample_id']}.pt"
            baseline.save(baseline_path)
            with torch.no_grad():
                torch.save(teacher.from_pil(source).detach().cpu(), desc_path)
            rec = dict(row, source_path=str(src_path), baseline_path=str(baseline_path),
                       source_descriptor=str(desc_path), negative_prompt=negative, seed=args.seed)
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"prepared {row['sample_id']} ({row['condition']})")


if __name__ == "__main__":
    main()
