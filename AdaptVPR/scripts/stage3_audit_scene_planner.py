"""Audit nearby-scene family proposals on the first 20 frozen Stage3-dev images.

No generation masks, diffusion, editor, spatial optimizer, or training imports.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from targeted.scene_planner import (
    FAMILIES, IMAGE_LABELS, SCENE_SCHEMA, SYSTEM_PROMPT, USER_PROMPT,
    SceneInput, ScenePlanner, prepare_scene_views,
)

BOQ_DEV = ROOT.parents[1] / "Bag-of-Queries/outputs/stage3/dev"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def load_inputs(dev_dir, *, count=20, context_scale=2.0):
    """Use frozen dev order, never Stage2-100 fixed-mask planner artifacts."""
    dev_dir = Path(dev_dir).resolve()
    if type(count) is not int or not 1 <= count <= 50:
        raise ValueError("count must be in [1,50]; default is the first 20 dev samples")
    cohort_path, summary_path = dev_dir / "cohort.jsonl", dev_dir / "summary.json"
    vulnerability_dir = dev_dir / "vulnerability"
    records_path, report_path = vulnerability_dir / "vulnerability.jsonl", vulnerability_dir / "summary.json"
    summary, report = (json.loads(path.read_text(encoding="utf-8")) for path in (summary_path, report_path))
    cohort, records = read_jsonl(cohort_path), read_jsonl(records_path)
    if (summary["cohort"] != "dev" or summary["dev"] != 50 or len(cohort) != 50
            or report["cohort"] != "dev" or report["status"] != "COMPLETE"
            or report["record_count"] != 50 or len(records) != 50):
        raise ValueError("Require complete frozen Stage3-dev 50 and its vulnerability export")
    if (digest(cohort_path) != summary["cohort_sha256"] or digest(records_path) != report["records_sha256"]
            or report["inputs_sha256"].get(str(cohort_path)) != digest(cohort_path)
            or report["inputs_sha256"].get(str(summary_path)) != digest(summary_path)):
        raise ValueError("Dev/vulnerability manifest hash mismatch")
    places = {row["place_key"] for row in cohort}
    if (len(places) != 50 or len({row["image_key"] for row in cohort}) != 50
            or places & set(summary["stage2_audit"]["excluded_place_keys"])):
        raise ValueError("Dev is not place-unique/disjoint from Stage2-100")
    for source, row in zip(cohort, records):
        fields = ("image_key", "place_key", "source_path", "source_sha256", "source_identity")
        if any(row[key] != source[key] for key in fields) or row["source_role"] != "SOURCE":
            raise ValueError("Vulnerability export does not match dev source identities/order")
        if (row["cohort"] != "dev" or row["token_grid"] != [16, 16] or row["mask_ratio"] != .15
                or row["mask_tokens"] != 38 or row["primary_roi"] != "attention_roi_token_mask"
                or row["mask_mode"] != "connected_topk"):
            raise ValueError("Require the frozen 15% primary attention vulnerability ROI")
    hashes = {str(path): digest(path) for path in (cohort_path, summary_path, records_path, report_path)}
    inputs = []
    for row in records[:count]:
        source_path = Path(row["source_path"]).resolve()
        numerical_path = (vulnerability_dir / row["numerical_artifact"]).resolve()
        if not numerical_path.is_relative_to(vulnerability_dir):
            raise ValueError("Numerical artifact escapes vulnerability directory")
        if digest(source_path) != row["source_sha256"] or digest(numerical_path) != row["numerical_artifact_sha256"]:
            raise ValueError("SOURCE or vulnerability NPZ hash mismatch")
        hashes[str(source_path)], hashes[str(numerical_path)] = digest(source_path), digest(numerical_path)
        with Image.open(source_path) as image:
            image.load()
            source = image.convert("RGB")  # No EXIF transpose, matching BoQ's raw decoded orientation.
        if source.size != (row["source_width"], row["source_height"]):
            raise ValueError("SOURCE original dimensions disagree with vulnerability export")
        with np.load(numerical_path, allow_pickle=False) as arrays:
            if (str(arrays["image_key"]) != row["image_key"] or str(arrays["place_key"]) != row["place_key"]
                    or json.loads(str(arrays["source_identity_json"])) != row["source_identity"]
                    or int(arrays["source_width"]) != source.width or int(arrays["source_height"]) != source.height):
                raise ValueError("NPZ SOURCE identity/dimensions mismatch")
            roi = arrays["attention_roi_token_mask"].copy()
        views, view_metadata = prepare_scene_views(source, roi, context_scale=context_scale)
        identity = {key: row[key] for key in ("image_key", "place_key", "source_identity", "source_path", "source_sha256",
                                             "source_width", "source_height")}
        identity.update(numerical_artifact=str(numerical_path), numerical_artifact_sha256=row["numerical_artifact_sha256"],
                        cohort="dev", source_role="SOURCE")
        inputs.append(SceneInput(identity, *views, view_metadata))
    return inputs, dict(inputs_sha256=hashes, cohort_manifest=str(cohort_path), vulnerability_manifest=str(records_path),
                        selection=f"first {count} records in frozen dev cohort order; no resampling",
                        generation_mask_read=False, taxonomy_version="scene-families-seven-v1")


def comparison_panel(scene, row, index):
    panel = Image.new("RGB", (1152, 370), "white")
    draw = ImageDraw.Draw(panel)
    decision = row["decision"] or {}
    label = (f"{index:02d} {scene.identity['place_key']} | {row['status']} | "
             f"{decision.get('region_type', '?')} / {decision.get('support_surface', '?')} | "
             f"confidence={decision.get('confidence', '?')}\n"
             f"families: {', '.join(decision.get('feasible_families', [])) or '(empty)'}")
    draw.multiline_text((8, 5), textwrap.fill(label.split('\n')[0], 145) + "\n" + textwrap.fill(label.split('\n')[1], 145), fill="black")
    for index, (view, label) in enumerate(zip(scene.views, ("original", "vulnerability cue (not edit boundary)", "ROI context crop"))):
        shown = ImageOps.contain(view, (380, 280))
        panel.paste(shown, (index * 384 + (384-shown.width)//2, 78 + (280-shown.height)//2))
        draw.text((index * 384 + 8, 62), label, fill="black")
    return panel


def run_audit(inputs, output, planner, *, seed=0, provenance=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "planner_instructions.json", dict(system=SYSTEM_PROMPT, user=USER_PROMPT,
                                                          image_labels=IMAGE_LABELS, response_schema=SCENE_SCHEMA))
    summary = dict(schema_version=1, status="RUNNING", count=len(inputs), seed=seed,
                   selected_image_keys=[scene.identity["image_key"] for scene in inputs],
                   selected_place_keys=[scene.identity["place_key"] for scene in inputs],
                   generation_mask_used=False, spatial_placement_decided=False, diffusion_called=False,
                   taxonomy=list(FAMILIES), taxonomy_pruned=False, provenance=provenance or {},
                   planner_metadata=planner.client.metadata)
    rows, panels = [], []
    try:
        with (output / "scene_assessments.jsonl").open("x", encoding="utf-8") as stream:
            for index, scene in enumerate(inputs):
                suffix = hashlib.sha256(scene.identity["image_key"].encode()).hexdigest()[:16]
                directory = output / f"{index:02d}_{suffix}"
                directory.mkdir()
                for name, view in zip(("source.png", "vulnerability_overlay.png", "roi_context_crop.png"), scene.views):
                    view.save(directory / name)
                try:
                    row = planner.plan(scene, seed=seed)
                except Exception as exc:
                    write_json(directory / "planner_error.json", dict(error=f"{type(exc).__name__}: {exc}"))
                    raise
                row["audit_directory"] = directory.name
                row["audit_view_files_sha256"] = {name: digest(directory / name) for name in
                                                  ("source.png", "vulnerability_overlay.png", "roi_context_crop.png")}
                write_json(directory / "scene_assessment.json", row)
                for attempt in row["response_attempts"]:
                    (directory / f"raw_response_attempt{attempt['attempt']}.txt").write_text(attempt["raw_response"], encoding="utf-8")
                (directory / "raw_response.txt").write_text(row["response_attempts"][-1]["raw_response"], encoding="utf-8")
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
                panel = comparison_panel(scene, row, index)
                panel.save(directory / "comparison.png")
                panels.append(panel)
                decision = row["decision"] or {}
                print(f"{index+1}/{len(inputs)} {scene.identity['place_key']}: {row['status']} "
                      f"{decision.get('feasible_families', [])}", flush=True)
        valid = [row["decision"] for row in rows if row["decision"] is not None]
        errors = sum(row["status"] == "schema_error" for row in rows)
        families = Counter(family for decision in valid for family in decision["feasible_families"])
        summary.update(status="COMPLETE_WITH_SCHEMA_ERRORS" if errors else "COMPLETE", completed_count=len(rows),
                       editable_count=sum(d["editable"] for d in valid), rejected_count=sum(not d["editable"] for d in valid),
                       schema_error_count=errors, retried_count=sum(len(r["response_attempts"]) > 1 for r in rows),
                       feasible_family_counts={family: families[family] for family in FAMILIES},
                       region_type_counts=dict(Counter(d["region_type"] for d in valid)),
                       support_surface_counts=dict(Counter(d["support_surface"] for d in valid)),
                       confidence_values=[d["confidence"] for d in valid],
                       counts_scope="schema-valid model assessments only; not human ground truth or placement success",
                       records_sha256=digest(output / "scene_assessments.jsonl"))
        for path, expected in summary["provenance"].get("inputs_sha256", {}).items():
            if digest(path) != expected:
                raise ValueError(f"Frozen input changed during audit: {path}")
        summary["input_files_unchanged"] = True
        sheet = Image.new("RGB", (1152, sum(panel.height for panel in panels)), "white")
        for index, panel in enumerate(panels):
            sheet.paste(panel, (0, index * panel.height))
        sheet.save(output / "contact_sheet.jpg", quality=90)
        with (output / "human_review.csv").open("w", encoding="utf-8", newline="") as stream:
            fields = ["index", "image_key", "place_key", "status", *SCENE_SCHEMA["required"],
                      "human_region_correct", "human_support_correct", "human_family_set_correct", "human_notes"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index, row in enumerate(rows):
                decision = dict(row["decision"] or {})
                if "feasible_families" in decision:
                    decision["feasible_families"] = "|".join(decision["feasible_families"])
                writer.writerow(dict(index=index, image_key=row["target"]["image_key"], place_key=row["target"]["place_key"],
                                     status=row["status"], **decision))
        report = ["# Stage3-dev scene planner audit", "",
                  f"Status: {summary['status']}; samples: {len(rows)} (frozen dev order); seed: {seed}.", "",
                  f"Model editable: {summary['editable_count']}; rejected: {summary['rejected_count']}; schema errors: {errors}.", "",
                  "These are nearby-scene/family assessments, not placement decisions or human ground truth.",
                  "Seven-family taxonomy retained; no pruning from this partial 20-of-50 pilot.",
                  "No generation mask was read, no final spatial position was chosen, and no diffusion was called.", "",
                  "| Family | Model feasible count |", "| --- | ---: |",
                  *[f"| {family} | {families[family]} |" for family in FAMILIES], "",
                  "Review the three-view comparisons and raw responses; human_review.csv has intentionally blank human labels.",
                  "A feasible family still needs independent placement, support/contact and realism validation downstream.", ""]
        (output / "AUDIT.md").write_text("\n".join(report), encoding="utf-8")
    except Exception as exc:
        summary.update(status="ERROR", completed_count=len(rows), error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-dir", type=Path, default=BOQ_DEV)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/stage3_dev/scene_planner_audit")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="VLM seed only; dev order is unchanged")
    parser.add_argument("--context-scale", type=float, default=2.0)
    parser.add_argument("--model-path", help="Existing local Qwen3-VL directory or ADAPTVPR_TARGETED_PLANNER_MODEL_PATH")
    parser.add_argument("--device", help="Local VLM device; defaults to configured device or cuda")
    args = parser.parse_args(argv)
    if args.seed < 0 or not 1 <= args.count <= 50 or not 1 <= args.context_scale < float("inf"):
        parser.error("Invalid seed/count/context scale")
    if args.output.exists():
        parser.error("Output already exists; preserve audit evidence by choosing a new directory")
    inputs, provenance = load_inputs(args.dev_dir, count=args.count, context_scale=args.context_scale)
    provenance["code_sha256"] = {name: digest(ROOT / name) for name in
                                ("targeted/scene_planner.py", "targeted/qwen_client.py", "scripts/stage3_audit_scene_planner.py")}
    # Explicit local VLM load after every selected input passes validation.
    from targeted.scene_planner import LocalSceneQwenClient
    client = LocalSceneQwenClient(model_path=args.model_path, device=args.device)
    summary = run_audit(inputs, args.output, ScenePlanner(client), seed=args.seed, provenance=provenance)
    print(json.dumps({key: summary[key] for key in ("status", "editable_count", "rejected_count", "schema_error_count")}, indent=2))
    return 0 if summary["schema_error_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
