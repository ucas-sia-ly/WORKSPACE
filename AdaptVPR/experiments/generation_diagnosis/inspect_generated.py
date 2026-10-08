"""Expose matching/alignment diagnostics for frozen generation-diagnosis runs.

Only the production _compute_s_geo method is called. CLIP scores and the
original gate decisions are retained from the generator checkpoint, not
recomputed. Coverage and homography displacement are descriptive; this script
introduces no acceptance rules.

Example:
    python AdaptVPR/experiments/generation_diagnosis/inspect_generated.py \
      --run-dir outputs/gen_diagnosis/prompt_ablation \
      --run-dir outputs/gen_diagnosis/qwen_comparison
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ADAPTVPR_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ADAPTVPR_ROOT))
from experiments.vpr_guidance.common import file_sha256, read_jsonl, use_adaptvpr, write_json

use_adaptvpr()
from experiments.generation_diagnosis.runner import append_checkpoint, fingerprint, read_checkpoint
from experiments.generation_diagnosis.verifier_controls import CapturingMatcher, matcher_diagnostics
from verification.evaluator import DualTraitEvaluator

WORKSPACE_ROOT = ADAPTVPR_ROOT.parent
VARIANTS = ("released", "no_negations", "positive")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--prompt-variants", nargs="+", choices=[*VARIANTS, "all"], default=["positive"])
    parser.add_argument("--out", type=Path, default=WORKSPACE_ROOT / "outputs/gen_diagnosis/generated_geometry")
    parser.add_argument("--plan-only", action="store_true", help="Validate inputs and freeze selection without loading any models")
    args = parser.parse_args(argv)
    if "all" in args.prompt_variants and len(args.prompt_variants) != 1:
        parser.error("--prompt-variants all must be used on its own")
    args.prompt_variants = list(VARIANTS) if args.prompt_variants == ["all"] else args.prompt_variants
    args.run_dir = list(dict.fromkeys(path.expanduser().resolve() for path in args.run_dir))
    args.out = args.out.expanduser().resolve()
    return args


def select_records(run_dirs: list[Path], variants: list[str]) -> tuple[list[dict], list[dict], list[dict]]:
    records, configurations, expected_sources = [], [], {}
    for run_dir in run_dirs:
        config_path = run_dir / "generation_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        sources = config["sources"]
        if not 1 <= len(sources) <= 8 or [row["src"] for row in sources] != list(range(len(sources))):
            raise ValueError(f"Expected a frozen cohort prefix with at most eight sources: {config_path}")
        normalized_sources = [{key: row[key] for key in ("src", "source_path", "source_sha256")} for row in sources]
        for source in normalized_sources:
            if source["src"] in expected_sources and expected_sources[source["src"]] != source:
                raise ValueError(f"Source identities/hashes differ across runs: {config_path}")
            expected_sources[source["src"]] = source
        latest = {}
        for row in read_jsonl(run_dir / "results.jsonl"):
            key = (row["src"], row["cond"], row["strat"], row["prompt_variant"])
            latest[key] = row
        selected = [row for row in latest.values() if row["prompt_variant"] in variants]
        for row in selected:
            if row.get("config_fingerprint") != config["fingerprint"]:
                raise ValueError(f"Row config fingerprint differs from frozen config: {run_dir}, {row.get('method')}")
            if row["source_path"] != sources[row["src"]]["source_path"]:
                raise ValueError(f"Row source differs from frozen cohort: {run_dir}, src {row['src']}")
            if row["status"] == "ok":
                source_png = Path(row["source_png_path"])
                output_png = Path(row["output_path"])
                if not source_png.is_absolute():
                    source_png = run_dir / source_png
                if not output_png.is_absolute():
                    output_png = run_dir / output_png
                if not source_png.is_file() or not output_png.is_file():
                    raise ValueError(f"Missing frozen input/output PNG: {source_png}, {output_png}")
                output_hash = file_sha256(output_png)
                if output_hash != row["output_sha256"]:
                    raise ValueError(f"Generated PNG hash changed since generator evaluation: {output_png}")
                records.append({
                    "run_dir": str(run_dir), "generator_row": row,
                    "source_png_path": str(source_png.resolve()), "output_path": str(output_png.resolve()),
                    "source_png_sha256": file_sha256(source_png), "output_sha256": output_hash,
                })
        selected_variants = [variant for variant in variants if variant in config["prompt_variants"]]
        configurations.append({
            "run_dir": str(run_dir), "mode": config["mode"],
            "generator_config_fingerprint": config["fingerprint"],
            "source_count": len(sources),
            "selected_prompt_variants": selected_variants,
            "expected_selected_rows": len(sources) * len(config["conditions"]) * len(config["strategies"]) * len(selected_variants),
            "available_selected_rows": len(selected),
            "successful_selected_rows": sum(row["status"] == "ok" for row in selected),
            "failed_generation_rows": sum(row["status"] != "ok" for row in selected),
            "generator_sampling": config["sampling"],
            "generator_verification": config["verification"],
            "scope_change": json.loads((run_dir / "scope_change.json").read_text(encoding="utf-8"))
            if (run_dir / "scope_change.json").is_file() else None,
        })
    return records, configurations, [expected_sources[index] for index in sorted(expected_sources)]


def record_key(record: dict) -> tuple:
    row = record.get("generator_row", record)
    return record["run_dir"], row["src"], row["cond"], row["strat"], row["prompt_variant"]


def median_available(rows: list[dict], field: str):
    values = [row[field] for row in rows if row.get(field) is not None]
    return statistics.median(values) if values else None


def summarize(rows: list[dict]) -> dict:
    latest = {record_key(row): row for row in rows}
    groups = defaultdict(list)
    for row in latest.values():
        groups[(row["model"], row["method"], row["cond"])].append(row)
        groups[(row["model"], row["method"], "all")].append(row)
    summaries = []
    for (model, method, condition), group in sorted(groups.items()):
        ok = [row for row in group if row["status"] == "ok"]
        summary = {"model": model, "method": method, "cond": condition,
                   "attempted": len(group), "evaluated": len(ok), "errors": len(group) - len(ok),
                   "reference_geo_accept_count": sum(row["reference_geo_ok"] for row in ok),
                   "reference_joint_pass_count": sum(row["reference_passed"] for row in ok)}
        for field in ("s_geo_recomputed", "reference_s_geo", "reference_s_div", "num_matched", "num_inliers",
                      "source_matched_fraction", "source_inlier_fraction",
                      "inlier_identity_displacement_median_px", "estimated_H_grid_displacement_median_px",
                      "estimated_H_grid_displacement_max_px"):
            summary[f"median_{field}"] = median_available(ok, field)
        summary["median_source_inlier_grid_coverage"] = statistics.median(
            [row["source_inlier_coverage"]["grid_coverage"] for row in ok]) if ok else None
        summary["median_source_inlier_convex_hull_area_fraction"] = statistics.median(
            [row["source_inlier_coverage"]["convex_hull_area_fraction"] for row in ok]) if ok else None
        summaries.append(summary)
    return {"groups": summaries, "rows": len(latest),
            "successful_diagnostics": sum(row["status"] == "ok" for row in latest.values()),
            "interpretation": "Descriptive matching/alignment diagnostics only; no new acceptance gates or calibrated quality guarantees."}


def main(argv=None):
    args = parse_args(argv)
    records, configurations, sources = select_records(args.run_dir, args.prompt_variants)
    manifest = {
        "schema_version": 1, "run_configurations": configurations, "sources": sources,
        "prompt_variants": args.prompt_variants,
        "metric": "Untouched production DualTraitEvaluator._compute_s_geo; original JPEG95 then square 512x512 preprocessing",
        "clip": "Not loaded or recomputed; old scores/gates are retained as reference",
        "matcher_requested": os.getenv("ADAPTVPR_MATCHER_NAME", "superpoint-lightglue"),
        "implementation_sha256": {str(path.relative_to(ADAPTVPR_ROOT)): file_sha256(path) for path in (
            Path(__file__).resolve(), Path(__file__).with_name("verifier_controls.py"),
            ADAPTVPR_ROOT / "verification/evaluator.py")},
        "selected_images": [{**{key: record[key] for key in ("run_dir", "source_png_path", "output_path", "source_png_sha256", "output_sha256")},
                             **{key: record["generator_row"][key] for key in ("src", "cond", "strat", "prompt_variant", "model", "method")}}
                            for record in records],
    }
    manifest["fingerprint"] = fingerprint(manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != manifest["fingerprint"]:
            raise ValueError(f"Frozen diagnostic selection/configuration changed; choose a fresh --out directory: {manifest_path}")
    else:
        write_json(manifest_path, manifest)
    if args.plan_only:
        print(json.dumps({"selected_images": len(records), "runs": configurations, "output": str(args.out)}, indent=2))
        return 0
    if not records:
        raise ValueError("No successful generated images selected")

    from PIL import Image
    import torch

    from vismatch import get_matcher

    # Construct only the state consumed by the unchanged production geometry
    # method, avoiding the evaluator constructor's unrelated CLIP allocation.
    # A requested matcher initialization error is surfaced rather than silently
    # switching to another matcher for this comparability diagnostic.
    evaluator = DualTraitEvaluator.__new__(DualTraitEvaluator)
    evaluator.device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator.img_size, evaluator.n_kpts = 512, 2048
    evaluator.matcher_name = manifest["matcher_requested"]
    capture = CapturingMatcher(get_matcher(
        evaluator.matcher_name, device=evaluator.device, max_num_keypoints=evaluator.n_kpts))
    evaluator.matcher = capture
    checkpoint = args.out / "results.jsonl"
    output_rows = read_checkpoint(checkpoint)
    latest = {record_key(row): row for row in output_rows}
    for record in records:
        key = record_key(record)
        if latest.get(key, {}).get("status") == "ok":
            continue
        original = record["generator_row"]
        row = {key: record[key] for key in ("run_dir", "source_png_path", "output_path", "source_png_sha256", "output_sha256")}
        row.update({key: original[key] for key in ("src", "cond", "strat", "prompt_variant", "model", "method")})
        row.update({
            "reference_s_geo": original["s_geo"], "reference_s_div": original["s_div"],
            "reference_geo_ok": original["geo_ok"], "reference_div_ok": original["div_ok"],
            "reference_passed": original["passed"], "config_fingerprint": original["config_fingerprint"],
            "diagnostic_manifest_fingerprint": manifest["fingerprint"],
            "generator_metadata": {key: original.get(key) for key in ("mode", "seed", "sampling", "service_health", "raw_dimensions", "source_dimensions", "resize_applied", "resize_method")},
        })
        try:
            with Image.open(record["source_png_path"]) as source_file, Image.open(record["output_path"]) as output_file:
                capture.last_result = None
                geometry = evaluator._compute_s_geo(source_file.convert("RGB"), output_file.convert("RGB"))
            row.update(status="ok", s_geo_recomputed=geometry,
                       matcher_name=evaluator.matcher_name, matcher_device=evaluator.device,
                       matcher_image_size=evaluator.img_size,
                       **matcher_diagnostics(capture.last_result, evaluator.img_size))
            print(f"{row['method']} s{row['src']} {row['cond']} geo={geometry:.3f} matches={row['num_matched']} H_med_px={row['estimated_H_grid_displacement_median_px']}", flush=True)
        except Exception as exc:
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
            print(f"ERROR {row['method']} s{row['src']} {row['cond']}: {row['error']}", flush=True)
        append_checkpoint(checkpoint, row)
        output_rows.append(row)
        latest[key] = row
        write_json(args.out / "summary.json", summarize(output_rows))
    summary = summarize(output_rows)
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return int(any(row["status"] != "ok" for row in latest.values()))


if __name__ == "__main__":
    raise SystemExit(main())
