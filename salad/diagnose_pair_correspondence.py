"""Bounded Stage 2B weather/control diagnostics. No optimizer or training loss.

Writes human-reviewable PNG/HTML, scalar/patch evidence JSON and a small geometry
cache. Teacher tensors are consumed online and never serialized.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import sys
import time

import numpy as np
import cv2
from PIL import Image, ImageDraw
import torch

from workflow.model import FrozenSALADTeacher, checkpoint_state_and_config, read_checkpoint
from workflow.pair_correspondence import (
    build_pair_correspondence, correspondence_measurements, fit_calibration,
)
from workflow.reliability_cache import file_sha256
from workflow.training_data import _resolve_manifest_file


WEATHERS = ("night", "rain", "snow", "fog")


def jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def write_json(path, value):
    path.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def select_pairs(manifest, real_data, count, calibration_count, seed):
    # Hash sampling avoids selecting the manifest's initial easy/Bangkok block.
    rows = []
    with manifest.open() as handle:
        for line, text in enumerate(handle, 1):
            if not text.strip():
                continue
            row = json.loads(text)
            if (row.get("condition") not in WEATHERS or row.get("passed") is not True
                    or row.get("eligible_for_training") is not True):
                continue
            row = dict(row, manifest_line=line)
            row["selection_hash"] = hashlib.sha256(
                f"{seed}|{row['source_path']}|{row['output_path']}".encode()).hexdigest()
            rows.append(row)
    rows.sort(key=lambda r: r["selection_hash"])
    counts = dict.fromkeys(WEATHERS, 0)
    holdout_counts = dict.fromkeys(WEATHERS, 0)
    holdout_limits = {w: calibration_count // len(WEATHERS) + int(i < calibration_count % len(WEATHERS))
                      for i, w in enumerate(WEATHERS)}
    selected, holdout, places = [], [], set()
    for row in rows:
        place = (row.get("city"), row.get("place_id"))
        if None in place:
            raise ValueError("Place identity required for disjoint calibration/evaluation selection")
        if place in places:
            continue
        if counts[row["condition"]] < count:
            selected.append(row); counts[row["condition"]] += 1
        elif holdout_counts[row["condition"]] < holdout_limits[row["condition"]]:
            holdout.append(row); holdout_counts[row["condition"]] += 1
        else:
            continue
        places.add(place)
        if min(counts.values()) == count and len(holdout) == calibration_count:
            break
    if min(counts.values()) < count or len(holdout) < calibration_count:
        raise ValueError("Insufficient distinct places for weather sample and calibration")
    for row in selected + holdout:
        for key in ("source_path", "output_path"):
            path = _resolve_manifest_file(row[key], manifest=manifest, line_number=row["manifest_line"],
                                          field_name=key, real_data=real_data)
            actual = file_sha256(path)
            hash_key = "source_sha256" if key == "source_path" else "output_sha256"
            if row.get(hash_key) and row[hash_key] != actual:
                raise ValueError(f"Manifest image content hash mismatch: {path}")
            row[key], row[hash_key] = str(path), actual
    return selected, holdout


class GeometryCollector:
    """Reuse production JPEG95/vismatch matching and Qwen H validation; no CLIP."""
    def __init__(self, workspace, device, cache):
        sys.path.insert(0, str(workspace / "AdaptVPR"))
        sys.path.insert(0, str(workspace / "vismatch"))
        from experiments.qwen_curriculum.quality import QwenQualityVerifier
        from experiments.generation_diagnosis.verifier_controls import CapturingMatcher
        from vismatch import get_matcher

        class CanvasCapture(CapturingMatcher):
            def __call__(self, a, b):
                self.source_hw, self.generated_hw = list(a.shape[-2:]), list(b.shape[-2:])
                return super().__call__(a, b)

        matcher = get_matcher("superpoint-lightglue", device=device, max_num_keypoints=2048)
        self.capture = CanvasCapture(matcher)
        self.verifier = QwenQualityVerifier(device=device, matcher=self.capture, img_size=512)
        self.cache = cache
        cache.mkdir(parents=True, exist_ok=True)
        self.settings = dict(matcher="superpoint-lightglue", image_size=512, max_keypoints=2048,
                             preprocessing="production JPEG95 full-frame square resize",
                             ransac_reprojection_threshold=matcher.ransac_reproj_thresh,
                             ransac_iters=matcher.ransac_iters, ransac_conf=matcher.ransac_conf)

    def __call__(self, source, generated):
        identity = dict(settings=self.settings,
                        source_rgb_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
                        generated_rgb_sha256=hashlib.sha256(generated.tobytes()).hexdigest(),
                        source_size=list(source.size), generated_size=list(generated.size))
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = self.cache / f"{key}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["identity"] != identity:
                raise ValueError("Geometry cache identity mismatch")
            return saved["evidence"]
        start = time.perf_counter()
        self.capture.last_result = None
        try:
            self.verifier.evaluator._compute_s_geo(source, generated)
            result = self.capture.last_result
            valid = self.verifier._valid_homography(result, self.verifier.img_size)
            evidence = dict(status="ok" if valid else "homography_unavailable",
                            coordinate_convention="xy_pixel_centers_full_frame_resize",
                            source_canvas_hw=self.capture.source_hw,
                            generated_canvas_hw=self.capture.generated_hw,
                            H=result.get("H"), inlier_kpts0=result.get("inlier_kpts0", []),
                            inlier_kpts1=result.get("inlier_kpts1", []),
                            ransac_reprojection_threshold=self.settings["ransac_reprojection_threshold"],
                            num_matched=len(result.get("matched_kpts0", [])),
                            seconds=time.perf_counter() - start)
        except (RuntimeError, ValueError, AttributeError, cv2.error) as exc:
            # Explicit failure remains unknown; no SIFT fallback or negative.
            evidence = dict(status="matcher_failed", error=f"{type(exc).__name__}: {exc}",
                            seconds=time.perf_counter() - start)
        evidence = jsonable(evidence)
        write_json(path, dict(identity=identity, evidence=evidence))
        return evidence


def distribution(tensor):
    x = torch.as_tensor(tensor, dtype=torch.float32).flatten().cpu()
    return dict(n=x.numel(), mean=float(x.mean()) if x.numel() else None,
                quantiles=dict(zip(("min", "q10", "q25", "median", "q75", "q90", "max"),
                                   torch.quantile(x, torch.tensor([0., .1, .25, .5, .75, .9, 1.])).tolist()))
                if x.numel() else {})


def metrics(result):
    d = result["match_diagnostics"]
    p, n, u = result["positive_conf"], result["negative_conf"], result["unknown_mask"]
    return dict(patches=p.numel(), positive=int((p > 0).sum()), negative=int((n > 0).sum()),
                unknown=int(u.sum()), positive_a=d["positive_a_count"], positive_b=d["positive_b_count"],
                reciprocal_fraction=float(d["reciprocal"].float().mean()),
                geometry_points=d["geometry_valid_points"], geometry_status=d["geometry_status"],
                geometry_conflicts=int(d["geometry_conflict"].sum()),
                confidence=distribution(p[p > 0]),
                confidence_a=distribution(p[d["positive_a"]]),
                confidence_b=distribution(p[d["positive_b_candidate"]]), cosine=distribution(d["cosine"]),
                b_teacher_cosine=distribution(d["geometry_cosine"][d["positive_b_candidate"]]),
                appearance_aliases=int(d["appearance_alias"].sum()),
                margin=distribution(d["margin"]), displacement=distribution(d["displacement"]))


def draw_preview(source, generated, result, title, path):
    # Fixed display size only; all matching uses actual feature/canvas sizes.
    w, h, label = 320, 240, 44
    canvas = Image.new("RGB", (3 * w, 2 * (h + label)), "white")
    draw = ImageDraw.Draw(canvas)
    images = [source.resize((w, h)), generated.resize((w, h))]
    for i, (im, name) in enumerate(zip(images, ("Source", "Generated"))):
        canvas.paste(im, (i * w, label))
        draw.text((i * w + 5, 4), title[:40] + "\n" + name, fill="black")
    # Third upper cell: thumbnail pair with correspondence lines (A green/B cyan).
    pair = Image.new("RGB", (w, h), "black")
    pair.paste(source.resize((w // 2, h)), (0, 0))
    pair.paste(generated.resize((w // 2, h)), (w // 2, 0))
    lines = ImageDraw.Draw(pair)
    idx = result["match_index"].cpu()
    gh, gw = idx.shape
    sh, sw = result["match_diagnostics"]["source_grid_hw"]
    a = result["match_diagnostics"]["positive_a"].cpu()
    valid = (idx.flatten() >= 0).nonzero().flatten().tolist()
    # Deterministic evenly spaced subset keeps the image readable.
    stride = max(1, math_ceil_div(len(valid), 48))
    for q in valid[::stride]:
        s = int(idx.flatten()[q])
        lines.line(((s % sw + .5) * (w / 2) / sw, (s // sw + .5) * h / sh,
                    w / 2 + (q % gw + .5) * (w / 2) / gw, (q // gw + .5) * h / gh),
                   fill=(60, 255, 70) if a.flatten()[q] else (30, 220, 255), width=1)
    canvas.paste(pair, (2 * w, label))
    draw.text((2 * w + 5, 4), "Correspondences (up to 48)\nA green / B candidate cyan", fill="black")
    for i, (key, color) in enumerate((("positive_conf", (0, 255, 80)),
                                     ("negative_conf", (255, 30, 20)), ("unknown_mask", (140, 140, 255)))):
        values = result[key].detach().float().cpu().numpy()
        strength = Image.fromarray(np.uint8(np.clip(values, 0, 1) * 255)).resize((w, h), Image.Resampling.NEAREST)
        # Heatmap color carries intensity; zero is black, no hidden auto scaling.
        rgb = np.asarray(strength)[..., None] / 255 * np.array(color)[None, None]
        heat = Image.fromarray(rgb.astype(np.uint8))
        y = h + 2 * label
        canvas.paste(heat, (i * w, y))
        draw.text((i * w + 5, h + label + 4), f"{key} [0,1]\ncoverage {np.mean(values > 0):.1%}", fill="black")
    canvas.save(path)


def math_ceil_div(n, d):
    return (n + d - 1) // d


def image_tensor(image, size):
    image = image.resize((size[1], size[0]), Image.Resampling.BILINEAR)
    x = torch.from_numpy(np.asarray(image, dtype=np.float32).copy() / 255.).permute(2, 0, 1)
    return (x - x.new_tensor([.485, .456, .406])[:, None, None]) / x.new_tensor([.229, .224, .225])[:, None, None]


def declared_masks(row, manifest, real_data, generated):
    """Only explicit mask files, never masks inferred from weather prompts."""
    masks = {}
    for kind in ("edit_mask", "occlusion_mask", "destruction_mask"):
        key = kind + "_path"
        if row.get(key) is None:
            continue
        path = _resolve_manifest_file(row[key], manifest=manifest, line_number=row["manifest_line"],
                                       field_name=key, real_data=real_data)
        if row.get(kind + "_sha256") and file_sha256(path) != row[kind + "_sha256"]:
            raise ValueError(f"Mask content hash mismatch: {path}")
        with Image.open(path) as handle:
            if handle.size != generated.size:
                raise ValueError(f"Explicit mask must use generated native image coordinates: {path}")
            masks[kind] = torch.from_numpy(np.asarray(handle.convert("L"), dtype=np.float32).copy() / 255.)
    return masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--backbone-repo", type=Path, required=True)
    parser.add_argument("--real-data", type=Path, default=Path("dataset/gsv-cities"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--geometry-cache", type=Path, help="Optional reuse of image/settings-keyed geometry only")
    parser.add_argument("--pairs-per-weather", type=int, default=6)
    parser.add_argument("--calibration-pairs", type=int, default=12)
    parser.add_argument("--null-tail-probabilities", type=float, nargs="+", required=True,
                        help="Exploratory null quantile operating points; no calibrated error-rate claim")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--matcher-device", default="cpu")
    args = parser.parse_args()
    if args.pairs_per_weather < 2 or args.calibration_pairs < 4 or any(
            not 0 < p < 1 for p in args.null_tail_probabilities):
        parser.error("Need >=2 weather pairs, >=4 calibration pairs and tail probabilities in (0,1)")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    selected, holdout = select_pairs(args.manifest.resolve(), args.real_data.resolve(),
                                     args.pairs_per_weather, args.calibration_pairs, args.seed)
    write_json(out / "selected_pairs.json", dict(weather_pairs=selected, calibration_holdout=holdout))
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    cv2.setRNGSeed(args.seed % (2 ** 31))
    payload = read_checkpoint(args.teacher_checkpoint)
    _, config = checkpoint_state_and_config(payload)
    del payload
    teacher = FrozenSALADTeacher(args.teacher_checkpoint, config, device=args.device,
                                 backbone_repo=args.backbone_repo)
    size = config.get("image_size", [224, 224])
    workspace = Path(__file__).resolve().parents[1]
    collector = GeometryCollector(workspace, args.matcher_device, args.geometry_cache or out / "geometry")

    @torch.no_grad()
    def features(image):
        _, aux = teacher(image_tensor(image, size).unsqueeze(0).to(args.device))
        # Small per-pair tensors transient in RAM; never written as a feature cache.
        return {k: v.cpu() for k, v in aux.items()}

    nulls, null_pairs, copies = [], [], []
    for i, row in enumerate(holdout):
        other = holdout[(i + 1) % len(holdout)]
        with Image.open(row["source_path"]) as s, Image.open(other["output_path"]) as g:
            sa = features(s.convert("RGB"))
            nulls.append(correspondence_measurements(sa, features(g.convert("RGB"))))
            copies.append(correspondence_measurements(sa, sa))
        null_pairs.append([row["source_path"], other["output_path"]])
    provenance = dict(teacher=teacher.provenance, manifest=str(args.manifest.resolve()),
                      manifest_sha256=file_sha256(args.manifest), null_pairs=null_pairs,
                      selection="distinct places disjoint from weather evaluation", seed=args.seed)
    calibrations = [fit_calibration(nulls, tail_probability=p, search_radius=1, provenance=provenance,
                                   positive_controls=copies)
                    for p in args.null_tail_probabilities]
    write_json(out / "calibration.json", [c.to_dict() for c in calibrations])
    print("Calibration fitted:", [c.to_dict() | {"provenance": "see calibration.json"} for c in calibrations], flush=True)
    records, links = [], []
    all_conf = {w: {k: [] for k in ("all", "a", "b")} for w in WEATHERS}
    for i, row in enumerate(selected):
        with Image.open(row["source_path"]) as handle:
            source = handle.convert("RGB")
        with Image.open(row["output_path"]) as handle:
            generated = handle.convert("RGB")
        sa, ga = features(source), features(generated)
        geometry = collector(source, generated)
        masks = declared_masks(row, args.manifest.resolve(), args.real_data.resolve(), generated)
        trials = []
        for c in calibrations:
            result = build_pair_correspondence(source, generated, sa, ga, calibration=c,
                                                geometry=geometry, quality=row, **masks)
            trials.append(dict(null_tail_probability=c.tail_probability, **metrics(result)))
        # The first explicit operating point is the preview, never auto-selected
        # to inflate coverage. All operating points remain in the report.
        result = build_pair_correspondence(source, generated, sa, ga, calibration=calibrations[0],
                                            geometry=geometry, quality=row, **masks)
        stem = f"{row['condition']}_{i:03d}"
        write_json(out / f"{stem}.json", result)
        draw_preview(source, generated, result, stem, out / f"{stem}.png")
        links.append(stem)
        p = result["positive_conf"]
        all_conf[row["condition"]]["all"].append(p[p > 0])
        all_conf[row["condition"]]["a"].append(p[result["match_diagnostics"]["positive_a"]])
        all_conf[row["condition"]]["b"].append(p[result["match_diagnostics"]["positive_b_candidate"]])
        record = dict(condition=row["condition"], source_path=row["source_path"], output_path=row["output_path"],
                      preview=f"{stem}.png", operating_points=trials)
        records.append(record)
        print(stem, trials[0], flush=True)

    # Real-teacher construction-labelled controls and unrelated-image validation.
    controls = []
    for wi, weather in enumerate(WEATHERS):
        row = next(r for r in selected if r["condition"] == weather)
        other = next(r for r in selected if (r["city"], r["place_id"]) != (row["city"], row["place_id"]))
        with Image.open(row["source_path"]) as handle:
            source = handle.convert("RGB").resize((size[1], size[0]), Image.Resampling.BILINEAR)
        sa = features(source)
        arr = np.asarray(source)
        shifted = np.zeros_like(arr); shifted[:, 14:] = arr[:, :-14]
        mask = torch.zeros(size); mask[size[0] // 4:size[0] // 2, size[1] // 4:size[1] // 2] = 1
        blocked = arr.copy(); blocked[mask.bool().numpy()] = 0
        tile = source.crop((0, 0, 14, 14))
        wall = Image.new("RGB", source.size)
        for y in range(0, source.height, 14):
            for x in range(0, source.width, 14):
                wall.paste(tile, (x, y))
        with Image.open(other["output_path"]) as handle:
            unrelated = handle.convert("RGB")
        cases = [("identity", source, {}), ("shift_one_patch", Image.fromarray(shifted), {}),
                 ("known_occlusion", Image.fromarray(blocked), {"occlusion_mask": mask}),
                 ("dark_structure", Image.fromarray((arr.astype(float) * .15).astype(np.uint8)), {}),
                 ("repeated_wall", wall, {}), ("unrelated", unrelated, {})]
        for name, generated, extra in cases:
            control_source = wall if name == "repeated_wall" else source
            cs = features(wall) if name == "repeated_wall" else sa
            ga, geometry = features(generated), collector(control_source, generated)
            result = build_pair_correspondence(control_source, generated, cs, ga,
                                                calibration=calibrations[0], geometry=geometry, **extra)
            stem = f"control_{weather}_{name}"
            draw_preview(control_source, generated, result, stem, out / f"{stem}.png")
            write_json(out / f"{stem}.json", result)
            m = metrics(result)
            if name in ("identity", "shift_one_patch", "dark_structure", "known_occlusion", "repeated_wall"):
                expected = torch.arange(result["match_index"].numel()).reshape(result["match_index"].shape)
                if name == "shift_one_patch":
                    expected = expected - 1
                valid = result["positive_conf"] > 0
                correct = result["match_index"] == expected
                if name == "shift_one_patch":
                    correct[:, 0] = False
                m["constructed_coordinate_accuracy"] = float(correct[valid].float().mean()) if valid.any() else None
            controls.append(dict(weather_source=weather, control=name, preview=stem + ".png", **m))
            links.append(stem)
            print(stem, m, flush=True)
    summary = {}
    for weather in WEATHERS:
        items = [r["operating_points"][0] for r in records if r["condition"] == weather]
        total = sum(r["patches"] for r in items)
        summary[weather] = dict(pairs=len(items), patches=total,
                                **{k + "_fraction": sum(r[k] for r in items) / total
                                   for k in ("positive", "positive_a", "positive_b", "negative", "unknown")},
                                positive_confidence={k: distribution(torch.cat(v)) for k, v in all_conf[weather].items()},
                                mean_reciprocal_fraction=float(np.mean([r["reciprocal_fraction"] for r in items])))
    report = dict(teacher=teacher.provenance, preview_null_tail_probability=calibrations[0].tail_probability,
                  weather_summary=summary, weather_pairs=records, controls=controls,
                  geometry_settings=collector.settings,
                  run_arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  source_sha256={str(p.relative_to(workspace)): file_sha256(p) for p in (
                      Path(__file__).resolve(), workspace / "salad/workflow/pair_correspondence.py",
                      workspace / "salad/workflow/reliability.py")},
                  torch_version=str(torch.__version__),
                  limitations=["No human-labelled weather ground truth: coverage is not accuracy.",
                               "B remains an independent geometry candidate, not static-landmark truth.",
                               "No negative evidence masks exist for sampled weather pairs; zero negatives is intentional.",
                               "Null quantiles are exploratory and not a validated false-match-rate guarantee.",
                               "Full-frame resize only; crop/padding needs an explicit coordinate transform."])
    write_json(out / "report.json", report)
    (out / "index.html").write_text("<!doctype html><meta charset='utf-8'><title>CCR Stage 2B</title>"
        "<h1>CCR Stage 2B correspondence review</h1><p>Green A / cyan B candidate. "
        "Coverage is not accuracy. See report.json and calibration.json.</p>"
        "<pre>" + html.escape(json.dumps(summary, indent=2)) + "</pre>" +
        "".join(f"<h2>{html.escape(stem)}</h2><a href='{stem}.json'>Patch evidence</a><br>"
                f"<img width='960' src='{stem}.png'>" for stem in links))
    print("Report:", out / "report.json", flush=True)


if __name__ == "__main__":
    main()
