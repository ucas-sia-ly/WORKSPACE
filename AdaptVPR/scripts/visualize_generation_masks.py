"""Sample Stage3 targets and visualize derived masks. Never import generation models."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw

from generation.targeted_inputs import read_targeted_tasks
from targeted.mask_adapter import adapt_generation_mask


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def distribution(values):
    if not values:
        return dict(count=0, min=None, p05=None, median=None, mean=None, p95=None, max=None)
    values = sorted(values)
    def percentile(p):
        index = (len(values) - 1) * p
        lo = int(index)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)
    return dict(count=len(values), min=values[0], p05=percentile(.05), median=statistics.median(values),
                mean=statistics.mean(values), p95=percentile(.95), max=values[-1])


def load_pair(task):
    images = []
    for field, digest in (("source_path", "source_sha256"), ("mask_original_path", "mask_original_sha256")):
        data = Path(task[field]).read_bytes()
        if hashlib.sha256(data).hexdigest() != task[digest]:
            raise ValueError(f"Source/mask pairing changed: {task['sample_id']} {field}")
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            images.append(image.copy())
    return images[0].convert("RGB"), images[1]


def overlay(source, mask, color):
    if mask is None:
        return source.copy()
    display_mask = mask.convert("L").point(lambda x: 255 if x else 0)
    tint = Image.blend(source, Image.new("RGB", source.size, color), .45)
    return Image.composite(tint, source, display_mask)


def comparison(source, vulnerability, generation, diagnostic):
    width, height = source.size
    canvas = Image.new("RGB", (3 * width, height + 64), "white")
    draw = ImageDraw.Draw(canvas)
    labels = ["source", f"vulnerability: {diagnostic['vulnerability_area_ratio']:.2%}",
              f"generation: {diagnostic['generation_area_ratio']:.2%} | overlap: {diagnostic['overlap_ratio']:.2%}"]
    if diagnostic["status"] == "failure":
        labels[2] = "FAIL: no generation mask"
    for i, (image, label) in enumerate(zip((source, vulnerability, generation), labels)):
        canvas.paste(image, (i * width, 64))
        draw.text((i * width + 8, 8), label, fill="black")
    if diagnostic["failure_reason"]:
        draw.text((2 * width + 8, 28), diagnostic["failure_reason"], fill="red")
    else:
        draw.text((2 * width + 8, 28), f"compactness {diagnostic['compactness_before']:.3f} -> {diagnostic['compactness_after']:.3f}", fill="black")
    return canvas


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="BoQ targets.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/stage3_targeted/generation_masks")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="Only controls sample selection")
    parser.add_argument("--target-ratio", type=float, default=.06)
    parser.add_argument("--min-overlap", type=float, default=.70)
    parser.add_argument("--area-tolerance", type=float, default=.05)
    args = parser.parse_args(argv)
    if args.count < 1 or args.seed < 0:
        parser.error("--count must be positive; --seed must be nonnegative")
    # Reuse model-free manifest/schema/hash validation. No editor import or diffusion.
    tasks = read_targeted_tasks(args.input, seed=args.seed)
    if args.count > len(tasks):
        parser.error(f"Requested {args.count} records; manifest has {len(tasks)}")
    selected = random.Random(args.seed).sample(sorted(tasks, key=lambda t: t["sample_id"]), args.count)
    output = args.output.resolve()
    if output.exists():
        parser.error("Output exists; choose a fresh directory to preserve previous evidence")
    output.mkdir(parents=True)
    summary = dict(schema_version=1, status="RUNNING", count=args.count, selection_seed=args.seed,
                   input_manifest=str(args.input.resolve()),
                   input_manifest_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                   target_ratio=args.target_ratio, min_overlap=args.min_overlap, area_tolerance=args.area_tolerance,
                   selected_sample_ids=[t["sample_id"] for t in selected], samples=[])
    write_json(output / "summary.json", summary)
    rows = []
    try:
        for index, task in enumerate(selected):
            source, vulnerability = load_pair(task)
            generation, diagnostic = adapt_generation_mask(
                vulnerability, args.target_ratio, args.min_overlap,
                image_height=source.height, image_width=source.width, area_tolerance=args.area_tolerance,
            )
            suffix = hashlib.sha256(task["sample_id"].encode()).hexdigest()[:16]
            directory = output / f"{index:02d}_{suffix}"
            directory.mkdir()
            source.save(directory / "source.png")
            vulnerability.save(directory / "vulnerability_mask.png")
            vuln_overlay = overlay(source, vulnerability, "red")
            gen_overlay = overlay(source, generation, "lime")
            vuln_overlay.save(directory / "vulnerability_overlay.png")
            if generation is not None:
                generation.save(directory / "generation_mask.png")
                gen_overlay.save(directory / "generation_overlay.png")
            panel = comparison(source, vuln_overlay, gen_overlay, diagnostic)
            panel.save(directory / "side_by_side.png")
            write_json(directory / "diagnostics.json", diagnostic)
            record = dict(sample_id=task["sample_id"], image_key=task["image_key"], place_key=task["place_key"],
                          target_type=task["target_type"], source_path=task["source_path"],
                          vulnerability_mask_path=task["mask_original_path"],
                          source_sha256=task["source_sha256"], vulnerability_mask_sha256=task["mask_original_sha256"],
                          generation_mask_path=f"{directory.name}/generation_mask.png" if generation is not None else None,
                          generation_mask_sha256=hashlib.sha256((directory / "generation_mask.png").read_bytes()).hexdigest() if generation is not None else None,
                          diagnostic=diagnostic, directory=directory.name)
            summary["samples"].append(record)
            panel.thumbnail((960, 300), Image.Resampling.LANCZOS)  # RGB visualization only; never binary masks
            rows.append(panel.copy())
            print(f"{index + 1}/{args.count}: {diagnostic['status']} area={diagnostic['generation_area_ratio']:.4%} overlap={diagnostic['overlap_ratio']:.4%}", flush=True)
        successful = [r["diagnostic"] for r in summary["samples"] if r["diagnostic"]["status"] == "success"]
        failed = [r["diagnostic"] for r in summary["samples"] if r["diagnostic"]["status"] == "failure"]
        summary.update(
            status="COMPLETE", success_count=len(successful), failure_count=len(failed),
            failure_reasons={reason: sum(d["failure_reason"] == reason for d in failed) for reason in sorted({d["failure_reason"] for d in failed})},
            generation_area_pixels_distribution=distribution([d["generation_area"] for d in successful]),
            generation_area_ratio_distribution=distribution([d["generation_area_ratio"] for d in successful]),
            overlap_ratio_distribution=distribution([d["overlap_ratio"] for d in successful]),
            compactness_gain_distribution=distribution([d["compactness_gain"] for d in successful]),
            distribution_scope="successful samples only; failures separately counted",
            no_diffusion_called=True,
            code_sha256={"targeted/mask_adapter.py": hashlib.sha256((ROOT / "targeted/mask_adapter.py").read_bytes()).hexdigest(),
                         "scripts/visualize_generation_masks.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        )
        sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), "white")
        y = 0
        for row in rows:
            sheet.paste(row, (0, y))
            y += row.height
        sheet.save(output / "contact_sheet.png")
        with (output / "generation_masks.jsonl").open("w", encoding="utf-8") as stream:
            for record in summary["samples"]:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")
    except Exception as exc:
        summary.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("success_count", "failure_count", "generation_area_ratio_distribution", "overlap_ratio_distribution")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
