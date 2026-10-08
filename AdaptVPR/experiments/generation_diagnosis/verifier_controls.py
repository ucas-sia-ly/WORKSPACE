"""Construction-labelled controls for the unchanged dual-trait evaluator.

This is a diagnostic, not a realistic-weather benchmark. It uses the same eight
sources as outputs/gen_diagnosis/strategy_grid.py by default. In particular,
identity is expected to fail the diversity gate, even when its geometry passes.
The script does not modify production evaluation thresholds or feedback.

Run from the workspace:
    python AdaptVPR/experiments/generation_diagnosis/verifier_controls.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

GUIDANCE_ROOT = Path(__file__).resolve().parents[1] / "vpr_guidance"
sys.path.insert(0, str(GUIDANCE_ROOT))
from common import WORKSPACE_ROOT, read_jsonl, use_adaptvpr, write_json

use_adaptvpr()
from verification.evaluator import DualTraitEvaluator


class CapturingMatcher:
    """Delegate all matching and image loading, retaining the last raw result."""

    def __init__(self, matcher):
        self.matcher = matcher
        self.last_result = None

    def __getattr__(self, name):
        return getattr(self.matcher, name)

    def __call__(self, image0, image1):
        self.last_result = self.matcher(image0, image1)
        return self.last_result


def to_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.rint(np.clip(array, 0.0, 1.0) * 255).astype(np.uint8))


def build_controls(source: Image.Image, donor: Image.Image):
    """Return images, construction labels and exact construction parameters."""
    array = np.asarray(source, dtype=np.float32) / 255.0
    width, height = source.size
    corners = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    target = np.float32([
        [0.07 * width, 0.04 * height],
        [0.91 * width, 0.10 * height],
        [0.98 * width, 0.89 * height],
        [0.03 * width, 0.98 * height],
    ])
    homography = cv2.getPerspectiveTransform(corners, target)
    warped = cv2.warpPerspective(
        array, homography, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0.45, 0.45, 0.45),
    )
    patched = source.copy()
    box = (width // 2, 0, width, height // 2)
    patched.paste(donor.resize(source.size).crop(box), box[:2])
    return [
        ("identity", source.copy(), "preserved", {"operation": "identity"}),
        ("gamma_nearblack", to_image(array ** 2.6 * 0.04), "preserved", {"gamma": 2.6, "scale": 0.04}),
        ("uniform_fog_dense", to_image(array * 0.025 + 0.80 * 0.975), "preserved", {"transmission": 0.025, "airlight": 0.80}),
        ("projective_warp", to_image(warped), "changed", {"source_to_control_H": homography.tolist()}),
        ("warp_plus_darkening", to_image(warped ** 1.8 * 0.45), "changed", {"source_to_control_H": homography.tolist(), "gamma": 1.8, "scale": 0.45}),
        ("foreign_top_right_quarter", patched, "regional_content_replaced", {"replacement_box_xyxy": list(box), "area_fraction": 0.25}),
    ]


def coverage(points: np.ndarray, side: int, grid: int = 4) -> dict:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return {"occupied_grid_cells": 0, "grid_cells": grid * grid, "grid_coverage": 0.0, "convex_hull_area_fraction": 0.0}
    cells = np.floor(points / side * grid).astype(int).clip(0, grid - 1)
    occupied = len(set(map(tuple, cells)))
    hull_area = float(cv2.contourArea(cv2.convexHull(points.astype(np.float32)))) if len(points) >= 3 else 0.0
    return {"occupied_grid_cells": occupied, "grid_cells": grid * grid, "grid_coverage": occupied / (grid * grid), "convex_hull_area_fraction": hull_area / (side * side)}


def matcher_diagnostics(result: dict, side: int) -> dict:
    matched0 = np.asarray(result.get("matched_kpts0", []), dtype=np.float64).reshape(-1, 2)
    inlier0 = np.asarray(result.get("inlier_kpts0", []), dtype=np.float64).reshape(-1, 2)
    inlier1 = np.asarray(result.get("inlier_kpts1", []), dtype=np.float64).reshape(-1, 2)
    n_matched = len(matched0)
    n_inliers = int(result.get("num_inliers", 0))
    n_detected0 = len(result.get("all_kpts0", []))
    n_detected1 = len(result.get("all_kpts1", []))
    output = {
        "num_matched": n_matched,
        "num_inliers": n_inliers,
        "num_detected_source": n_detected0,
        "num_detected_control": n_detected1,
        "source_matched_fraction": n_matched / n_detected0 if n_detected0 else 0.0,
        "source_inlier_fraction": n_inliers / n_detected0 if n_detected0 else 0.0,
        "source_inlier_coverage": coverage(inlier0, side),
        "control_inlier_coverage": coverage(inlier1, side),
        "inlier_identity_displacement_median_px": None,
        "estimated_H": None,
        "estimated_H_grid_displacement_median_px": None,
        "estimated_H_grid_displacement_max_px": None,
    }
    if len(inlier0) and len(inlier0) == len(inlier1):
        displacement = np.linalg.norm(inlier1 - inlier0, axis=1)
        output["inlier_identity_displacement_median_px"] = float(np.median(displacement))
    homography = result.get("H")
    if homography is not None:
        homography = np.asarray(homography, dtype=np.float64)
        output["estimated_H"] = homography.tolist()
        axis = np.linspace(0, side - 1, 5)
        grid = np.array([(x, y) for y in axis for x in axis], dtype=np.float64)
        mapped_h = np.c_[grid, np.ones(len(grid))] @ homography.T
        valid = np.abs(mapped_h[:, 2]) > 1e-8
        if valid.any():
            mapped = mapped_h[valid, :2] / mapped_h[valid, 2:]
            displacement = np.linalg.norm(mapped - grid[valid], axis=1)
            output["estimated_H_grid_displacement_median_px"] = float(np.median(displacement))
            output["estimated_H_grid_displacement_max_px"] = float(np.max(displacement))
    return output


def draw_sheet(source: Image.Image, controls: list, rows: list[dict], destination: Path):
    width, image_height, label_height = 240, 180, 58
    sheet = Image.new("RGB", (width * (len(controls) + 1), image_height + label_height), "white")
    draw = ImageDraw.Draw(sheet)
    sheet.paste(source.resize((width, image_height)), (0, label_height))
    draw.text((5, 5), "source", fill="black")
    for i, ((name, control, truth, _), row) in enumerate(zip(controls, rows), start=1):
        x = width * i
        sheet.paste(control.resize((width, image_height)), (x, label_height))
        draw.text((x + 4, 3), f"{name}\n{truth}\ng {row['s_geo']:.3f} d {row['s_div']:.3f} matches {row['num_matched']}\ngeo_ok {row['geo_ok']} joint_pass {row['passed']}", fill="black")
    sheet.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=WORKSPACE_ROOT / "outputs/vpr_guidance_smoke/cand_probe/candidates.jsonl")
    parser.add_argument("--out", type=Path, default=WORKSPACE_ROOT / "outputs/gen_diagnosis/verifier_controls")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    candidates = read_jsonl(args.candidates)
    sources = sorted({r["source_path"] for r in candidates})[::5][:args.limit]
    if len(sources) < 2:
        parser.error("At least two distinct selected sources are needed for the foreign-region control")
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "manifest.json", {
        "sources": sources,
        "source_selection": "sorted(unique source_path)[::5][:limit]",
        "route": "global",
        "tau_geo": 0.78,
        "tau_div": 0.15,
        "geometry_labels": {"preserved": "No pixel coordinate displacement; appearance may be unrecognizable", "changed": "Known projective pixel coordinate displacement", "regional_content_replaced": "Top-right quarter replaced by another source, breaking source correspondence in that region"},
        "limitations": "Extreme photometric controls are not realistic weather. Identity's joint rejection is expected because s_div is zero. Coverage and H displacement are descriptive diagnostics, not acceptance rules.",
    })
    evaluator = DualTraitEvaluator()
    capture = CapturingMatcher(evaluator._load_matcher())
    evaluator.matcher = capture
    all_rows = []
    with (args.out / "results.jsonl").open("w", encoding="utf-8") as handle:
        for index, source_path in enumerate(sources):
            source = Image.open(source_path).convert("RGB")
            donor_path = sources[(index + 1) % len(sources)]
            donor = Image.open(donor_path).convert("RGB")
            controls = build_controls(source, donor)
            source_rows = []
            for name, control, truth, construction in controls:
                control_path = args.out / f"s{index}_{name}.png"
                control.save(control_path)
                capture.last_result = None
                evaluation = evaluator.evaluate(source, control, entry={"route": "global"})
                diagnostics = matcher_diagnostics(capture.last_result, evaluator.img_size)
                row = {
                    "source_index": index, "source_path": source_path, "donor_path": donor_path if truth == "regional_content_replaced" else None,
                    "control": name, "control_path": str(control_path.resolve()), "geometry_truth": truth, "construction": construction,
                    "s_geo": evaluation.s_geo, "s_div": evaluation.s_div,
                    "geo_ok": evaluation.geo_ok, "div_ok": evaluation.div_ok, "passed": evaluation.passed,
                    "matcher_name": evaluator.matcher_name, "matcher_device": evaluator.device,
                    "matcher_image_size": evaluator.img_size,
                    **diagnostics,
                }
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                source_rows.append(row)
                all_rows.append(row)
                print(f"s{index} {name:26s} g={evaluation.s_geo:.3f} d={evaluation.s_div:.3f} geo_ok={evaluation.geo_ok} passed={evaluation.passed} matches={row['num_matched']}", flush=True)
            if index in {0, 4}:
                draw_sheet(source, controls, source_rows, args.out / f"sheet_s{index}.png")
    summary = []
    for name in dict.fromkeys(row["control"] for row in all_rows):
        selected = [row for row in all_rows if row["control"] == name]
        summary.append({
            "control": name, "geometry_truth": selected[0]["geometry_truth"], "n": len(selected),
            "geo_accept_count": sum(row["geo_ok"] for row in selected),
            "joint_pass_count": sum(row["passed"] for row in selected),
            "median_s_geo": float(np.median([row["s_geo"] for row in selected])),
            "median_s_div": float(np.median([row["s_div"] for row in selected])),
            "median_num_matches": float(np.median([row["num_matched"] for row in selected])),
        })
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
