"""Audit a fixed generation-mask manifest with Qwen3-VL; never run diffusion."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw

from generation.targeted_inputs import _read_jsonl, read_targeted_tasks
from targeted.editability import REGION_TYPES
from targeted.planner import SYSTEM_PROMPT, USER_PROMPT, TargetedInput, TargetedPlanner
from targeted.prompt_family import FAMILIES


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(manifest, targets_manifest, count, seed):
    if count < 1:
        raise ValueError("count must be positive")
    records = _read_jsonl(manifest)
    if len(records) < count:
        raise ValueError(f"Requested {count} fixed targets; manifest only has {len(records)}")
    originals = {task["sample_id"]: task for task in read_targeted_tasks(targets_manifest, seed=seed)}
    # Preserve the prior adapter's deterministic order. Never resample or rederive masks.
    return [TargetedInput.from_record(row, manifest.parent, originals[row["sample_id"]]) for row in records[:count]]


def run_audit(inputs, output, planner, *, seed, provenance=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "planner_instructions.json", dict(system=SYSTEM_PROMPT, user=USER_PROMPT))
    summary = dict(schema_version=1, status="RUNNING", count=len(inputs), seed=seed,
                   selected_sample_ids=[item.identity["sample_id"] for item in inputs],
                   no_diffusion_called=True, planner_metadata=planner.client.metadata,
                   min_confidence=planner.min_confidence, provenance=provenance or {},
                   sample_selection="input manifest order; masks reused without changes")
    write_json(output / "planner_summary.json", summary)
    plans, thumbnails = [], []
    try:
        with (output / "targeted_edit_plans.jsonl").open("x", encoding="utf-8") as stream:
            for index, item in enumerate(inputs):
                suffix = hashlib.sha256(item.identity["sample_id"].encode()).hexdigest()[:16]
                directory = output / f"{index:02d}_{suffix}"
                directory.mkdir()
                # Preserve exact artifact mask bytes, even their 0/1 encoding.
                item.source.save(directory / "source.png")
                for prefix in ("vulnerability_mask", "generation_mask"):
                    original = Path(item.identity[f"{prefix}_path"])
                    if digest(original) != item.identity[f"{prefix}_sha256"]:
                        raise ValueError(f"Artifact changed before inference: {prefix}")
                    shutil.copyfile(original, directory / f"{prefix}.png")
                try:
                    plan, views, raw = planner.plan(item, seed=seed)
                except Exception as exc:
                    write_json(directory / "planner_error.json", {"error": f"{type(exc).__name__}: {exc}"})
                    raise  # No mock fallback or fabricated VLM decisions.
                row = plan.to_dict()
                row["audit_directory"] = directory.name
                views[1].save(directory / "generation_overlay.png")
                views[2].save(directory / "context_crop.png")
                (directory / "raw_response.txt").write_text(raw, encoding="utf-8")
                write_json(directory / "planner.json", row)
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                plans.append(row)
                width, height = item.source.size
                panel = Image.new("RGB", (width * 3, height + 60), "white")
                for i, image in enumerate(views):
                    panel.paste(image, (i * width, 60))
                decision = row["decision"] or {}
                label = f"{index:02d} {row['status']} | {decision.get('region_type', '?')} | {decision.get('occluder_family', '?')} | confidence={decision.get('confidence', '?')}"
                ImageDraw.Draw(panel).text((8, 8), label, fill="black")
                ImageDraw.Draw(panel).text((8, 30), "original                         fixed target (magenta)                         context crop (native pixels)", fill="black")
                panel.save(directory / "comparison.png")
                panel.thumbnail((1200, 400))  # RGB presentation only, never binary masks.
                thumbnails.append(panel)
                print(f"{index + 1}/{len(inputs)} {label}", flush=True)
        regions = Counter(p["decision"]["region_type"] for p in plans if p["decision"])
        families = Counter(p["decision"]["occluder_family"] for p in plans if p["decision"])
        proposed_families = Counter(p["proposed_decision"]["occluder_family"] for p in plans if p["proposed_decision"])
        schema_errors = sum(p["status"] == "schema_error" for p in plans)
        editable = sum(p["status"] == "editable" for p in plans)
        summary.update(
            status="COMPLETE" if not schema_errors else "COMPLETE_WITH_SCHEMA_ERRORS",
            editable_count=editable, rejected_count=len(plans) - editable,
            schema_error_count=schema_errors,
            region_type_counts={k: regions[k] for k in REGION_TYPES},
            occluder_family_counts={k: families[k] for k in FAMILIES},
            proposed_occluder_family_counts={k: proposed_families[k] for k in FAMILIES},
            low_confidence_count=sum(p["proposed_decision"]["confidence"] < planner.min_confidence for p in plans if p["proposed_decision"]),
            policy_rejection_counts=dict(Counter(reason for p in plans for reason in p["policy_rejections"])),
            counts_scope="region/family/confidence counts use schema-valid decisions only; schema errors reject separately",
            plans_sha256=digest(output / "targeted_edit_plans.jsonl"),
        )
        for item in inputs:
            for prefix in ("source", "vulnerability_mask", "generation_mask"):
                if digest(Path(item.identity[f"{prefix}_path"])) != item.identity[f"{prefix}_sha256"]:
                    raise ValueError(f"Input artifact changed: {prefix}")
        summary["input_files_unchanged"] = True
        sheet = Image.new("RGB", (max(p.width for p in thumbnails), sum(p.height for p in thumbnails)), "white")
        y = 0
        for panel in thumbnails:
            sheet.paste(panel, (0, y))
            y += panel.height
        sheet.save(output / "contact_sheet.jpg", quality=90)
    except Exception as exc:
        summary.update(status="ERROR", completed_count=len(plans), error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output / "planner_summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Prior adapter generation_masks.jsonl, preserving its fixed order")
    parser.add_argument("--targets", type=Path, required=True, help="Original BoQ targets.jsonl for independent identity checks")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/stage3_targeted/planner_audit")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=("local", "openai"), default="local")
    parser.add_argument("--model-path", help="Local Qwen3-VL path; otherwise ADAPTVPR_TARGETED_PLANNER_MODEL_PATH")
    parser.add_argument("--min-confidence", type=float, default=.70)
    parser.add_argument("--context-scale", type=float, default=2.0)
    args = parser.parse_args(argv)
    if args.seed < 0 or args.count < 1 or not 0 <= args.min_confidence <= 1 or not 1 <= args.context_scale < float("inf"):
        parser.error("Invalid count/seed/confidence/context-scale")
    if args.output.exists():
        parser.error("Output already exists; choose a fresh directory to preserve audit evidence")
    inputs = load_inputs(args.input.resolve(), args.targets.resolve(), args.count, args.seed)
    # Lazy, explicit VLM loading only after validating all selected input artifacts.
    from targeted.qwen_client import LocalQwenClient, OpenAIQwenClient
    client = LocalQwenClient(model_path=args.model_path) if args.backend == "local" else OpenAIQwenClient()
    planner = TargetedPlanner(client, min_confidence=args.min_confidence, context_scale=args.context_scale)
    provenance = dict(
        input_manifest=str(args.input.resolve()), input_manifest_sha256=digest(args.input),
        targets_manifest=str(args.targets.resolve()), targets_manifest_sha256=digest(args.targets),
        code_sha256={str(path.relative_to(ROOT)): digest(path) for path in [
            ROOT / "targeted/editability.py", ROOT / "targeted/prompt_family.py", ROOT / "targeted/planner.py",
            ROOT / "targeted/qwen_client.py", Path(__file__).resolve()]},
    )
    summary = run_audit(inputs, args.output.resolve(), planner, seed=args.seed, provenance=provenance)
    print(json.dumps({k: summary[k] for k in ("editable_count", "rejected_count", "schema_error_count", "low_confidence_count")}, indent=2))
    return 0 if summary["schema_error_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
