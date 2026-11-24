"""Rectify existing Qwen outputs as an alignment diagnostic, not a generator fix.

Only raw generated pixels are warped. Unsupported pixels are transparent and
marked in a separate validity mask; no source blending or image synthesis is
performed. A single homography cannot repair local content edits or parallax.
The preferred future remedy is a source-aspect canvas upstream of generation.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ADAPTVPR_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ADAPTVPR_ROOT))
from experiments.qwen_curriculum.common import file_sha256, use_adaptvpr, write_json

use_adaptvpr()
from experiments.generation_diagnosis.inspect_generated import select_records
from experiments.generation_diagnosis.runner import append_checkpoint, fingerprint, read_checkpoint
from experiments.generation_diagnosis.verifier_controls import CapturingMatcher, matcher_diagnostics
from verification.evaluator import DualTraitEvaluator


def resize_matrix(from_size, to_size) -> np.ndarray:
    """Map pixel centers through an uncropped resize; sizes are (width, height)."""
    scale = np.asarray(to_size, dtype=np.float64) / np.asarray(from_size, dtype=np.float64)
    return np.array([[scale[0], 0, (scale[0] - 1) / 2],
                     [0, scale[1], (scale[1] - 1) / 2], [0, 0, 1]], dtype=np.float64)


def raw_to_source_homography(fitted_h, source_size, normalized_size, raw_size, matcher_side=512):
    """Convert source-square→normalized-square H into raw-Qwen→source pixels."""
    fitted_h = np.asarray(fitted_h, dtype=np.float64)
    if fitted_h.shape != (3, 3) or not np.isfinite(fitted_h).all() or np.linalg.cond(fitted_h) > 1e12:
        raise ValueError("The fitted homography is missing, nonfinite or numerically singular")
    source_square = resize_matrix(source_size, (matcher_side, matcher_side))
    normalized_square = resize_matrix(normalized_size, (matcher_side, matcher_side))
    raw_normalized = resize_matrix(raw_size, normalized_size)
    converted = np.linalg.inv(source_square) @ np.linalg.inv(fitted_h) @ normalized_square @ raw_normalized
    return converted / converted[2, 2]


def rectify_raw(raw_rgb: np.ndarray, raw_to_source: np.ndarray, source_size):
    """Warp raw RGB and return an exact source-grid support mask separately."""
    width, height = source_size
    raw_height, raw_width = raw_rgb.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float64)
    source_grid = np.stack([xx.ravel(), yy.ravel(), np.ones(width * height)], axis=1)
    mapped_h = source_grid @ np.linalg.inv(raw_to_source).T
    valid_denominator = np.isfinite(mapped_h).all(axis=1) & (np.abs(mapped_h[:, 2]) > 1e-10)
    coordinates = np.full((len(source_grid), 2), np.nan)
    coordinates[valid_denominator] = mapped_h[valid_denominator, :2] / mapped_h[valid_denominator, 2:]
    # Bounds require both interpolation neighbors to come from the raw canvas.
    valid = valid_denominator & (coordinates[:, 0] >= 0) & (coordinates[:, 0] <= raw_width - 1)
    valid &= (coordinates[:, 1] >= 0) & (coordinates[:, 1] <= raw_height - 1)
    valid = valid.reshape(height, width)
    warped = cv2.warpPerspective(raw_rgb, raw_to_source, (width, height),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))
    # Invalid RGB storage values have no image meaning; alpha is zero there.
    rgba = np.dstack([warped, valid.astype(np.uint8) * 255])
    return rgba, valid


def contact_sheet(source, normalized, rectified, label, output):
    width, height = source.size
    canvas = Image.new("RGB", (3 * width, height + 42), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (name, image) in enumerate((("source", source), ("normalized Qwen", normalized), ("rectified diagnostic", rectified))):
        x = index * width
        draw.text((x + 4, 3), name, fill="black")
        if index == 2:
            tile = np.indices((height, width)).sum(axis=0) // 12 % 2
            background = Image.fromarray(np.repeat((190 + 45 * tile)[..., None], 3, axis=2).astype(np.uint8))
            background.paste(image.convert("RGB"), (0, 0), image.getchannel("A"))
            canvas.paste(background, (x, 42))
        else:
            canvas.paste(image.convert("RGB").resize((width, height)), (x, 42))
    draw.text((4, 23), label, fill="black")
    canvas.save(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ADAPTVPR_ROOT.parent / "outputs/gen_diagnosis/qwen_rectification")
    parser.add_argument("--evaluate-rectified", action="store_true", help="Also run unchanged production evaluation, only for completely supported source grids")
    args = parser.parse_args(argv)
    args.run_dir, args.out = args.run_dir.resolve(), args.out.resolve()
    records, configurations, sources = select_records([args.run_dir], ["released", "positive"])
    config = configurations[0]
    if config["mode"] != "qwen" or len(sources) != 2 or len(records) != 20 or config["expected_selected_rows"] != 20:
        raise ValueError("This bounded diagnostic requires the completed two-source, 20-output Qwen run")
    for record in records:
        original = record["generator_row"]
        raw_path = Path(original["raw_output_path"])
        if not raw_path.is_file() or file_sha256(raw_path) != original["raw_output_sha256"]:
            raise ValueError(f"Missing or changed raw Qwen snapshot: {raw_path}")
        record.update(raw_output_path=str(raw_path), raw_output_sha256=original["raw_output_sha256"])
    manifest = {"schema_version": 1, "generator_config": config, "sources": sources,
                "evaluate_rectified": args.evaluate_rectified,
                "matcher": os.getenv("ADAPTVPR_MATCHER_NAME", "superpoint-lightglue"),
                "limitations": "One global H; raw-pixel warp only; complete bounds do not prove correct content/geometry. No deployment or acceptance-gate changes.",
                "implementation_sha256": file_sha256(Path(__file__)),
                "inputs": [{key: record[key] for key in ("source_png_path", "source_png_sha256", "output_path", "output_sha256", "raw_output_path", "raw_output_sha256")} for record in records]}
    manifest["fingerprint"] = fingerprint(manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Frozen rectification inputs/configuration changed; use a fresh --out directory")
    write_json(manifest_path, manifest)

    import torch
    from vismatch import get_matcher

    evaluator = DualTraitEvaluator() if args.evaluate_rectified else DualTraitEvaluator.__new__(DualTraitEvaluator)
    evaluator.device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator.img_size, evaluator.n_kpts = 512, 2048
    evaluator.matcher_name = manifest["matcher"]
    capture = CapturingMatcher(get_matcher(evaluator.matcher_name, device=evaluator.device, max_num_keypoints=evaluator.n_kpts))
    evaluator.matcher = capture
    checkpoint = args.out / "results.jsonl"
    rows = read_checkpoint(checkpoint)
    latest = {(row["src"], row["cond"], row["prompt_variant"]): row for row in rows}
    for record in records:
        original = record["generator_row"]
        key = original["src"], original["cond"], original["prompt_variant"]
        if latest.get(key, {}).get("status") == "ok":
            continue
        row = {name: original[name] for name in ("src", "cond", "prompt_variant", "method", "model", "seed", "sampling", "service_health", "config_fingerprint")}
        row.update({name: record[name] for name in ("source_png_path", "source_png_sha256", "output_path", "output_sha256", "raw_output_path", "raw_output_sha256")})
        row["original_scores"] = {name: original[name] for name in ("s_geo", "s_div", "geo_ok", "div_ok", "passed")}
        try:
            source = Image.open(record["source_png_path"]).convert("RGB")
            normalized = Image.open(record["output_path"]).convert("RGB")
            raw = Image.open(record["raw_output_path"]).convert("RGB")
            geo = evaluator._compute_s_geo(source, normalized)
            diagnostics = matcher_diagnostics(capture.last_result, evaluator.img_size)
            fitted_h = capture.last_result.get("H")
            converted = raw_to_source_homography(fitted_h, source.size, normalized.size, raw.size)
            rectified_rgba, validity = rectify_raw(np.asarray(raw), converted, source.size)
            rectified = Image.fromarray(rectified_rgba)
            stem = f"s{original['src']}_{original['cond']}_{original['prompt_variant']}"
            rectified_path, mask_path, contact_path = [args.out / f"{stem}_{suffix}.png" for suffix in ("rectified", "validity", "contact")]
            rectified.save(rectified_path)
            Image.fromarray(validity.astype(np.uint8) * 255).save(mask_path)
            coverage = float(validity.mean())
            row.update(status="ok", normalized_s_geo_recomputed=geo, normalized_alignment=diagnostics,
                       raw_to_source_H=converted.tolist(), source_to_raw_H=np.linalg.inv(converted).tolist(),
                       source_dimensions=list(source.size), normalized_dimensions=list(normalized.size), raw_dimensions=list(raw.size),
                       raw_support_fraction=coverage, raw_support_complete=bool(validity.all()),
                       unsupported_source_pixels=int((~validity).sum()), matcher_name=evaluator.matcher_name,
                       rectified_path=str(rectified_path), rectified_sha256=file_sha256(rectified_path),
                       validity_mask_path=str(mask_path), validity_mask_sha256=file_sha256(mask_path), contact_path=str(contact_path))
            row["rectified_evaluation"] = None
            if args.evaluate_rectified and validity.all():
                evaluation = evaluator.evaluate(source, rectified.convert("RGB"), entry={"route": "global", "weather": original["cond"]})
                row["rectified_evaluation"] = {name: getattr(evaluation, name) for name in ("s_geo", "s_div", "geo_ok", "div_ok", "passed")}
                row["rectified_alignment"] = matcher_diagnostics(capture.last_result, evaluator.img_size)
            elif args.evaluate_rectified:
                row["evaluation_skipped_reason"] = "Source grid contains unsupported raw pixels; evaluate would score artificial border placeholders"
            contact_sheet(source, normalized, rectified, f"s{original['src']} {original['cond']} {original['prompt_variant']} raw support {coverage:.1%}; diagnostic only", contact_path)
            print(f"{stem}: raw support {coverage:.4%}, normalized H median displacement {diagnostics['estimated_H_grid_displacement_median_px']:.2f}px", flush=True)
        except Exception as exc:
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
            print(f"ERROR {key}: {row['error']}", flush=True)
        append_checkpoint(checkpoint, row)
        latest[key] = row
    ok = [row for row in latest.values() if row["status"] == "ok"]
    summary = {"attempted": len(latest), "rectified": len(ok), "errors": len(latest) - len(ok),
               "full_raw_support": sum(row["raw_support_complete"] for row in ok),
               "median_raw_support_fraction": statistics.median(row["raw_support_fraction"] for row in ok) if ok else None,
               "rectified_evaluated": sum(row.get("rectified_evaluation") is not None for row in ok),
               "interpretation": "Alignment diagnostic only. Coverage and verifier acceptance do not establish complete geometry, content preservation or realistic weather."}
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return int(summary["errors"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
