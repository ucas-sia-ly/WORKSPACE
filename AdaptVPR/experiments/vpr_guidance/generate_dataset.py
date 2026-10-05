"""Generate verified Global/domain-shift hard positives.

Run either the original released AdaptVPR IC-Light generator (control) or the
VPR-aware LoRA generator (ours) through the exact same Global verifier.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image

from AdaptVPR.verification.evaluator import DualTraitEvaluator, ROUTE_THRESHOLDS
from .data import image_index, read_global_prompts, resolve_source
from .iclight import generate_released, load_iclight, load_lora, released_negative_prompt, sampling_policy
from .teacher import load_salad


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--dataframe-dir", type=Path, help="GSV-Cities Dataframes; default: image-root/../Dataframes")
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
    audit = p.add_mutually_exclusive_group()
    audit.add_argument("--salad-audit", action="store_true", help="optionally load SALAD for a post-generation statistic")
    audit.add_argument("--skip-salad-audit", action="store_true", help="compatibility flag; generation skips SALAD by default")
    return p


def _accepted_global_result(result):
    """Require a genuine successful finite result from the released verifier."""
    scores = (result.s_geo, result.s_div)
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError(f"invalid Global verifier scores: geo={result.s_geo}, div={result.s_div}")
    thresholds = ROUTE_THRESHOLDS["global"]
    expected = result.s_geo >= thresholds["TAU_GEO"] and result.s_div >= thresholds["TAU_DIV"]
    if result.skipped or not isinstance(result.passed, bool) or result.passed != expected:
        raise ValueError("Global verifier returned skipped or inconsistent acceptance flags")
    return result.passed is True


def main():
    args = args_parser().parse_args()
    from .data import GSVLabelIndex, require_empty_output, validate_source_label
    from .teacher import file_sha256

    rows = read_global_prompts(args.prompts, args.conditions, args.limit)
    images = image_index(args.image_root)
    labels = GSVLabelIndex(args.dataframe_dir or args.image_root.resolve().parent / "Dataframes")
    sources = []
    for row in rows:
        path = resolve_source(row, images)
        city, place_id = validate_source_label(row, path, labels)
        sampling_policy(row["condition"])
        with Image.open(path) as image:
            image.verify()
        sources.append((row, path, city, place_id))
    out = args.output_dir.resolve()
    require_empty_output(out)
    accepted_dir = out / "accepted"
    rejected_dir = out / "rejected"
    accepted_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    t2i, i2i, vae = load_iclight()
    if t2i.unet is not i2i.unet:
        raise RuntimeError("released IC-Light stages must share their UNet")
    generator_variant = "released"
    if args.lora is not None:
        load_lora(t2i.unet, args.lora)
        generator_variant = "vpr_lora"
    t2i.unet.eval()

    verifier = DualTraitEvaluator()
    # Resolve the original evaluator's matcher fallback before the first sample.
    # Record the actual matcher so B/C runs can verify the same policy was used.
    verifier._load_matcher()
    verifier_policy = {
        "route": "global",
        **ROUTE_THRESHOLDS["global"],
        "matcher_name": verifier.matcher_name,
        "img_size": verifier.img_size,
        "n_kpts": verifier.n_kpts,
        "clip_model": getattr(getattr(verifier.model, "config", None), "_name_or_path", None),
    }
    teacher = load_salad(repo=args.salad_repo) if args.salad_audit else None
    records_path = out / "records.jsonl"
    manifest_path = out / "synthetic_manifest.jsonl"

    with records_path.open("w", encoding="utf-8") as records, manifest_path.open("w", encoding="utf-8") as manifest:
        for row, src_path, city, place_id in sources:
            with Image.open(src_path) as image:
                source = image.convert("RGB")
            negative = released_negative_prompt(row.get("negative_prompt"))
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
            passed = _accepted_global_result(result)
            target_dir = accepted_dir if passed else rejected_dir
            gen_path = target_dir / f"{row['sample_id']}.png"
            generated.save(gen_path)

            preserve = None
            if teacher is not None:
                with torch.no_grad():
                    preserve = float(
                        (teacher.from_pil(source) * teacher.from_pil(generated)).sum()
                    )

            rec = {
                "sample_id": row["sample_id"],
                "source_id": row["source_id"],
                "source_path": str(src_path),
                "source_sha256": file_sha256(src_path),
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
                "sampling_policy": sampling_policy(row["condition"]),
                "verifier_policy": verifier_policy,
                "s_geo": result.s_geo,
                "s_div": result.s_div,
                "passed": passed,
                "eligible_for_training": passed,
                "salad_preservation_cosine": preserve,
            }
            records.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records.flush()
            if passed:
                manifest.write(json.dumps(rec, ensure_ascii=False) + "\n")
                manifest.flush()
            print(
                f"[{generator_variant}] {row['sample_id']} {row['condition']}: "
                f"geo={result.s_geo:.3f} div={result.s_div:.3f} passed={result.passed}"
            )


if __name__ == "__main__":
    main()
