#!/usr/bin/env python3
"""Automated content-audit controls, signal extraction, calibration and export.

No generation/training is launched. Only `signals --execute` loads local DINO;
all other steps are CPU metadata/image operations. No manual label input.
Every command requires a NEW output location and never rewrites old ledgers.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.qwen_curriculum import common
from experiments.qwen_curriculum import content_audit_v4 as audit


def _new_file(path, value, jsonl=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        if jsonl:
            for row in value:
                stream.write(json.dumps(row, allow_nan=False, ensure_ascii=False) + "\n")
        else:
            json.dump(value, stream, allow_nan=False, ensure_ascii=False, indent=2)
            stream.write("\n")


def _new_dir(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def _rows(path):
    rows = common.read_jsonl(Path(path))
    identifiers = [row.get("sample_id") for row in rows]
    if any(not name for name in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Every row needs a unique, nonempty sample_id")
    return rows


def create_controls(args):
    root = Path(args.training_root).resolve()
    excluded = {f"{city}/{int(place)}" for city, place in json.loads(Path(args.exclude_places).read_text())}
    candidates = {}
    for row in _rows(args.manifest):
        city, place = row.get("city"), row.get("place_id")
        if city is None or place is None:
            raise ValueError("Training manifest must carry city and place_id")
        key = f"{city}/{int(place)}"
        if key in excluded:
            continue
        source = Path(row["source_path"]).resolve()
        if not source.is_relative_to(root):
            raise ValueError("A source lies outside the declared training root")
        dummy = {**row, "dataset": args.dataset, "split": "calibration", "origin_split": "train",
                 "control": "benign", "control_recipe": "pending", "place_key": key}
        audit.require_training_calibration(dummy)
        if key not in candidates:
            candidates[key] = {**row, "source_path": str(source), "place_key": key}
    if args.count < 2 or len(candidates) < args.count:
        raise ValueError("Requested controls need at least count>=2 distinct non-excluded training places")
    rng = np.random.default_rng(args.seed)
    ordered = [candidates[key] for key in sorted(candidates)]
    selected = [ordered[int(i)] for i in rng.choice(len(ordered), args.count, replace=False)]
    partitions = {name: [row for row in selected
                        if (int(audit.digest(row["place_key"])[:8], 16) % 4 == 0) == (name == "heldout")]
                  for name in ("fit", "heldout")}
    if any(len(group) < 2 for group in partitions.values()):
        raise ValueError("Need at least two training places in each deterministic fit/heldout partition")
    out = _new_dir(args.output_dir)
    (out / "images").mkdir()
    controls = []
    for i, row in enumerate(selected):
        with Image.open(row["source_path"]) as opened:
            source = opened.convert("RGB").resize((args.side, args.side), Image.Resampling.BICUBIC)
        source_sha = audit.sha256(row["source_path"])
        if row.get("source_sha256") and row["source_sha256"] != source_sha:
            raise ValueError("Training source hash differs from manifest")
        # These are calibration controls, not proposed physical camera ranges.
        # Expose the complete recipes so operating limits cannot be mistaken
        # for measured real-camera parameters or an automatic quality guarantee.
        gamma, gain = float(rng.uniform(.4, 2.5)), float(rng.uniform(.2, 2.0))
        # Known small reframing makes the valid-overlap envelope meaningful;
        # calibration from identity pairs alone would demand 100% overlap and
        # abstain on every real pair with even a sub-patch translation.
        tx, ty = map(int, rng.uniform(-.1, .1, size=2) * args.side)
        aligned = source.transform((args.side, args.side), Image.Transform.AFFINE,
                                   (1, 0, -tx, 0, 1, -ty), Image.Resampling.BILINEAR)
        rgb = np.asarray(aligned, np.float32) / 255
        # Smooth spatial gains alter only the existing source surfaces. This
        # covers localized relighting and clipping without rendering fixtures,
        # text, light-emitting objects or any new scene content. The ranges are
        # disclosed stress controls, not calibrated real-camera parameters.
        yy, xx = np.mgrid[0:args.side, 0:args.side].astype(np.float32) / args.side
        log_gain = np.zeros((args.side, args.side), np.float32)
        illumination = []
        for _ in range(3):
            cx, cy = map(float, rng.uniform(0, 1, size=2))
            sx, sy = map(float, rng.uniform(.1, .4, size=2))
            strength = float(rng.uniform(-1., 1.6))
            log_gain += strength * np.exp(-.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
            illumination.append({"center_xy_normalized": [cx, cy], "sigma_xy_normalized": [sx, sy],
                                 "log_gain_amplitude": strength})
        # Place a gain-only clipping stress at an already bright source pixel;
        # all-black regions stay black until the separate noise transform.
        probability = rgb.max(axis=-1).ravel().astype(np.float64) ** 4
        total = probability.sum()
        center_index = int(rng.choice(probability.size, p=probability / total if total > 0 else None))
        hy, hx = divmod(center_index, args.side)
        hradius, hstrength = float(rng.uniform(.03, .12)), float(rng.uniform(2., 8.))
        highlight = 1 + hstrength * np.exp(-.5 * (((xx - hx / args.side) / hradius) ** 2 +
                                               ((yy - hy / args.side) / hradius) ** 2))
        clipping = {"center_xy_normalized": [hx / args.side, hy / args.side],
                    "sigma_normalized": hradius, "multiplicative_gain_amplitude": hstrength,
                    "center_policy": "sample_existing_source_brightness_power4"}
        sigma = float(rng.uniform(0, .04))
        rgb = np.clip(gain * rgb ** gamma * np.exp(log_gain[..., None]) * highlight[..., None] +
                      rng.normal(0, sigma, rgb.shape), 0, 1)
        blur_radius = float(rng.uniform(0, 1.2))
        benign = Image.fromarray(np.uint8(rgb * 255)).filter(ImageFilter.GaussianBlur(blur_radius))
        quality = int(rng.integers(55, 96))
        buffer = io.BytesIO()
        benign.save(buffer, format="JPEG", quality=quality)
        benign = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
        # Insert known foreign content after the identical benign relighting;
        # donor comes from a different approved TRAINING place.
        partition = "heldout" if int(audit.digest(row["place_key"])[:8], 16) % 4 == 0 else "fit"
        group = partitions[partition]
        donor_row = group[(next(k for k, entry in enumerate(group) if entry["place_key"] == row["place_key"]) + 1) % len(group)]
        with Image.open(donor_row["source_path"]) as donor_opened:
            donor = donor_opened.convert("RGB")
            side = int(rng.uniform(.25, .55) * args.side)
            x, y = map(int, rng.integers(0, args.side - side + 1, size=2))
            negative = benign.copy()
            negative.paste(donor.resize((side, side), Image.Resampling.BICUBIC), (x, y))
        for label, image in (("benign", benign), ("controlled_negative", negative)):
            sample_id = f"ctrl4_{audit.digest(row['place_key'])[:16]}_{label}"
            path = out / "images" / (sample_id + ".png")
            image.save(path)
            recipe = {"gamma": gamma, "gain": gain, "gaussian_noise_sigma": sigma,
                      "jpeg_quality": quality, "translation_xy": [tx, ty],
                      "smooth_spatial_illumination": illumination, "localized_clipping_gain": clipping,
                      "gaussian_blur_radius": blur_radius, "control_index": i,
                      "recipe_version": "programmatic_controls_v2"}
            if label == "controlled_negative":
                recipe.update(insertion_xyxy=[x, y, x + side, y + side],
                              donor_path=donor_row["source_path"],
                              donor_sha256=audit.sha256(donor_row["source_path"]))
            controls.append({"sample_id": sample_id, "source_path": row["source_path"],
                "source_sha256": source_sha, "output_path": str(path), "output_sha256": audit.sha256(path),
                "place_key": row["place_key"], "control": label, "control_recipe": recipe,
                "dataset": args.dataset, "split": "calibration", "origin_split": "train",
                "estimated_H": [[1, 0, tx], [0, 1, ty], [0, 0, 1]], "matcher_image_size": args.side,
                "geometry_ok": True, "weather_ok": True, "passed": True,
                "geometry_provenance": "known_programmatic_translation_same_canvas_control"})
    _new_file(out / "controls.jsonl", controls, True)
    _new_file(out / "protocol.json", {"version": audit.VERSION, "seed": args.seed,
        "training_root": str(root), "input_manifest": str(Path(args.manifest).resolve()),
        "input_manifest_sha256": audit.sha256(args.manifest), "excluded_places_sha256": audit.sha256(args.exclude_places),
        "count_places": len(selected), "note": "No evaluated/test places or human labels; known machine controls only"})
    print(f"Created {len(controls)} programmatic controls: {out / 'controls.jsonl'}")


def extract_signals(args):
    if not args.execute:
        print("Prepared only. Add --execute to run frozen LOCAL DINO forwards; no training/generation.")
        return
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    rows = _rows(args.manifest)
    provider = audit.FrozenDINOProvider(args.dino_repo, args.dino_checkpoint, args.architecture, args.device)
    records = []
    for index, row in enumerate(rows):
        try:
            cache = None
            if args.ocr_cache:
                # Hash-derived names prevent arbitrary sample IDs becoming paths.
                path = Path(args.ocr_cache) / (audit.digest(row["sample_id"]) + ".json")
                if path.exists():
                    cache = json.loads(path.read_text())
            signals = audit.pair_signals(row, provider, args.side, args.search_radius, cache)
            records.append(signals)
        except (OSError, ValueError, TypeError, KeyError) as error:
            records.append({"sample_id": row["sample_id"], "error": str(error)})
        print(f"signals {index + 1}/{len(rows)} {row['sample_id']}", flush=True)
    _new_file(args.output, records, True)
    _new_file(str(args.output) + ".receipt.json", {"version": audit.VERSION,
        "input_manifest_sha256": audit.sha256(args.manifest), "pipeline": provider.definition,
        "signals_sha256": audit.sha256(args.output), "count": len(records)})


def fit_calibration(args):
    rows, signals = _rows(args.manifest), {row["sample_id"]: row for row in _rows(args.signals)}
    pairs = [(row, signals[row["sample_id"]]) for row in rows]
    artifact = audit.calibrate_controls(pairs, args.false_alarm_rate, min_pairs=args.min_pairs)
    artifact["input_manifest_sha256"] = audit.sha256(args.manifest)
    artifact["input_signals_sha256"] = audit.sha256(args.signals)
    artifact["artifact_sha256"] = audit.digest({k: v for k, v in artifact.items() if k != "artifact_sha256"})
    _new_file(args.output, artifact)
    print(f"Control detector usable={artifact['usable']}; heldout={artifact['heldout_controls']}")
    if not artifact["usable"]:
        print("No verified control separation: downstream audits remain UNKNOWN, excluded from training.")


def export_audit(args):
    rows = _rows(args.manifest)
    signals = {row["sample_id"]: row for row in _rows(args.signals)} if args.signals else {}
    calibration = json.loads(Path(args.calibration).read_text()) if args.calibration else None
    out = _new_dir(args.output_dir)
    decisions, accepted, counts, groups = [], [], {}, {}
    for row in rows:
        original = dict(row)
        try:
            # Always verify actual files on export; a valid cache cannot bless
            # images that were replaced after signal extraction.
            for field, sha_field in (("source_path", "source_sha256"), ("output_path", "output_sha256")):
                value = audit.sha256(row[field])
                if row.get(sha_field) and row[sha_field] != value:
                    raise ValueError(f"Image hash mismatch: {field}")
                row[sha_field] = value
            decision = audit.audit_pair(row, signals.get(row["sample_id"]), calibration)
        except (OSError, ValueError, KeyError) as error:
            decision = {"version": audit.VERSION, "decision": "unknown", "eligible_for_training": False,
                        "reasons": ["image_integrity_failure: " + str(error)]}
        record = {**row, "pre_content_eligible_for_training": original.get("eligible_for_training"),
                  "pre_content_passed": original.get("passed"),
                  "base_quality_passed": original.get("base_quality_passed", original.get("passed") is True and
                                           original.get("geometry_ok") is True and original.get("weather_ok") is True),
                  "passed": decision["eligible_for_training"],
                  "eligible_for_training": decision["eligible_for_training"],
                  "content_audit": decision, "content_status": decision["decision"],
                  "input_row_sha256": audit.digest(original)}
        # Generation result seals cover the historical flags, not this updated
        # independent audit row. Retain them as provenance, never as a stale
        # top-level seal that appears to authenticate the changed record.
        if "result_sha256" in record:
            record["original_result_sha256"] = record.pop("result_sha256")
        record.pop("audit_record_sha256", None)
        record["audit_record_sha256"] = audit.digest(record)
        decisions.append(record)
        if decision["eligible_for_training"]:
            accepted.append(record)
        label = decision["decision"]
        counts[label] = counts.get(label, 0) + 1
        key = f"{row.get('condition', 'unknown')}/{row.get('prompt_variant', 'unknown')}"
        group = groups.setdefault(key, {})
        group[label] = group.get(label, 0) + 1
    _new_file(out / "audited_manifest.jsonl", decisions, True)
    _new_file(out / "training_manifest.jsonl", accepted, True)
    _new_file(out / "report.json", {"version": audit.VERSION, "counts": counts, "groups": groups,
        "input_manifest": str(Path(args.manifest).resolve()), "input_manifest_sha256": audit.sha256(args.manifest),
        "signals_sha256": audit.sha256(args.signals) if args.signals else None,
        "calibration_sha256": audit.sha256(args.calibration) if args.calibration else None,
        "training_manifest_sha256": audit.sha256(out / "training_manifest.jsonl"),
        "interpretation": "Machine control-calibrated safety screen; pass is not semantic correctness or real-night equivalence",
        "unknown_policy": "excluded", "quarantined_variants": ["shopfronts"]})
    print(f"Audit: {counts}; eligible={len(accepted)}; {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    controls = sub.add_parser("controls", help="Create training-only benign relighting/known insertion controls")
    controls.add_argument("--manifest", required=True)
    controls.add_argument("--training-root", required=True)
    controls.add_argument("--exclude-places", required=True, help="Stage 2 excluded [city,place_id] list")
    controls.add_argument("--dataset", required=True, help="Declared TRAIN dataset; known evaluation corpora denied")
    controls.add_argument("--output-dir", required=True)
    controls.add_argument("--count", type=int, default=128)
    controls.add_argument("--seed", type=int, default=42)
    controls.add_argument("--side", type=int, default=448)
    controls.set_defaults(handler=create_controls)
    signals = sub.add_parser("signals", help="Frozen local DINO and geometry-aligned patch diagnostics")
    signals.add_argument("--manifest", required=True)
    signals.add_argument("--output", required=True)
    signals.add_argument("--dino-repo", required=True)
    signals.add_argument("--dino-checkpoint", required=True)
    signals.add_argument("--architecture", default="dinov2_vitb14")
    signals.add_argument("--device", default="cpu")
    signals.add_argument("--side", type=int, default=448)
    signals.add_argument("--search-radius", type=int, default=1)
    signals.add_argument("--ocr-cache", help="Optional hash-bound normalized-box JSON evidence directory")
    signals.add_argument("--execute", action="store_true")
    signals.set_defaults(handler=extract_signals)
    calibrate = sub.add_parser("calibrate", help="Fit and validate control envelope on disjoint training places")
    calibrate.add_argument("--manifest", required=True)
    calibrate.add_argument("--signals", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--false-alarm-rate", type=float, required=True, help="Explicit operating alpha, not a cosine cutoff")
    calibrate.add_argument("--min-pairs", type=int, default=20, help="Minimum controls per label and fit/heldout partition")
    calibrate.set_defaults(handler=fit_calibration)
    export = sub.add_parser("audit", help="Export separate pass/reject/unknown and eligible-only manifests")
    export.add_argument("--manifest", required=True)
    export.add_argument("--signals")
    export.add_argument("--calibration")
    export.add_argument("--output-dir", required=True)
    export.set_defaults(handler=export_audit)
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
