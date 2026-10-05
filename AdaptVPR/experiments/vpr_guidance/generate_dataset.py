"""Generate verified Global/domain-shift hard positives.

Run either the original released AdaptVPR IC-Light generator (control) or the
VPR-aware LoRA generator (ours) through the exact same Global verifier.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from AdaptVPR.adapters import iclight_sd15_fc as adapter
from AdaptVPR.verification.evaluator import DualTraitEvaluator
from .data import image_index, place_key_from_name, read_global_prompts, resolve_source
from .iclight import generate_released, load_iclight, load_lora
from .teacher import load_salad


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument(
        "--lora",
        type=Path,
        default=None,
        help="optional VPR-aware IC-Light LoRA. Omit for the original released-generator control",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--conditions", nargs="*", default=[])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--salad-repo", default="serizba/salad")
    p.add_argument("--skip-salad-audit", action="store_true")
    args = p.parse_args()

    rows = read_global_prompts(args.prompts, args.conditions, args.limit)
    images = image_index(args.image_root)
    out = args.output_dir.resolve()
    accepted_dir = out / "accepted"
    rejected_dir = out / "rejected"
    accepted_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    t2i, i2i, vae = load_iclight()
    generator_variant = "released"
    if args.lora is not None:
        load_lora(t2i.unet, args.lora)
        generator_variant = "vpr_lora"
    t2i.unet.eval()

    verifier = DualTraitEvaluator()
    teacher = None if args.skip_salad_audit else load_salad(repo=args.salad_repo)
    records_path = out / "records.jsonl"
    manifest_path = out / "synthetic_manifest.jsonl"

    with records_path.open("w", encoding="utf-8") as records, manifest_path.open("w", encoding="utf-8") as manifest:
        for row in rows:
            src_path = resolve_source(row, images)
            source = Image.open(src_path).convert("RGB")
            negative = row.get("negative_prompt") or adapter.DEFAULT_NEGATIVE_PROMPT
            generated = generate_released(
                t2i,
                i2i,
                vae,
                source,
                row["prompt"],
                negative,
                args.seed,
                row["condition"],
            )
            result = verifier.evaluate(
                source,
                generated,
                entry={"route": "global", "weather": row["condition"]},
            )
            target_dir = accepted_dir if result.passed else rejected_dir
            gen_path = target_dir / f"{row['sample_id']}.png"
            generated.save(gen_path)

            preserve = None
            if teacher is not None:
                with torch.no_grad():
                    preserve = float(
                        (teacher.from_pil(source) * teacher.from_pil(generated)).sum()
                    )

            city, place_id = place_key_from_name(row["source_id"])
            rec = {
                "sample_id": row["sample_id"],
                "source_id": row["source_id"],
                "source_path": str(src_path),
                "generated_path": str(gen_path),
                "city": city,
                "place_id": place_id,
                "condition": row["condition"],
                "prompt": row["prompt"],
                "negative_prompt": negative,
                "route": "global",
                "seed": args.seed,
                "generator_variant": generator_variant,
                "generator_checkpoint": str(args.lora.resolve()) if args.lora else None,
                "s_geo": result.s_geo,
                "s_div": result.s_div,
                "passed": bool(result.passed),
                "eligible_for_training": bool(result.passed),
                "salad_preservation_cosine": preserve,
            }
            records.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records.flush()
            if result.passed:
                manifest.write(json.dumps(rec, ensure_ascii=False) + "\n")
                manifest.flush()
            print(
                f"[{generator_variant}] {row['sample_id']} {row['condition']}: "
                f"geo={result.s_geo:.3f} div={result.s_div:.3f} passed={result.passed}"
            )


if __name__ == "__main__":
    main()
