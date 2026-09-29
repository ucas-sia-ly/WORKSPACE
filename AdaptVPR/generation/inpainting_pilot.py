"""Deterministic dev-only pilot preparation and unfiltered local generation."""

from collections import defaultdict
from dataclasses import asdict
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops

from .inpainting_editor import TrainedInpaintingConfig, TrainedInpaintingEditor, inspect_checkpoint
from .inpainting_validation import validate_backend
from .targeted_editor import canonical_mask, file_sha256, pixel_sha256

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT.parents[1]/"Bag-of-Queries/outputs/stage3/dev"
FAMILY_OBJECTS = dict(parked_vehicle="a parked passenger vehicle", construction_barrier="a temporary construction barrier",
                      traffic_cones="traffic cones", temporary_sign="a freestanding temporary street sign",
                      vegetation="a compact leafy shrub", scaffolding="building facade scaffolding",
                      construction_tarp="a construction tarp attached to the building facade")
RULE = "one candidate per SOURCE in frozen dev order; max weighted coverage, max precision, min area, min centroid distance, candidate_id"


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prompt_for_family(family):
    return (f"A realistic street photograph with {FAMILY_OBJECTS[family]} occupying the masked region. "
            "Plausible physical support, perspective and scale, matching the scene lighting, natural contact shadows. "
            "Preserve the surrounding scene.")


def select_candidates(samples, rows, *, mode, threshold):
    if mode not in {"frozen", "render_semantics"}:
        raise ValueError("Unknown selection mode")
    grouped = defaultdict(list)
    for row in rows:
        gate = row["thresholds"][str(threshold)]
        eligible = gate["passes"] if mode == "frozen" else not (
            set(gate["rejection_reasons"])-{"weighted_coverage_below_threshold"})
        if eligible and row["render"]["feasible"]:
            grouped[row["image_key"]].append(row)
    selected = []
    for sample in samples:
        choices = grouped[sample["target"]["image_key"]]
        if not choices:
            continue
        row = min(choices, key=lambda r: (-r["metrics"]["vulnerability_weighted_coverage"],
                  -r["metrics"]["target_precision"], r["metrics"]["area_fraction"],
                  r["metrics"]["centroid_distance_normalized"], r["candidate_id"]))
        if not sample["scene_decision"]["editable"] or row["family"] not in sample["families"]:
            raise ValueError("Candidate family not allowed by the frozen scene decision")
        selected.append((sample, row))
    return selected


def verified_image(path, digest):
    import hashlib
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Input image hash mismatch: {path}")
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        return image.copy()


def prepare_pilot(*, candidate_dir, output, dev_dir=DEV, count=10, seed=0, selection_mode="frozen"):
    if type(count) is not int or not 1 <= count <= 20 or type(seed) is not int or not 0 <= seed < 2**63-count:
        raise ValueError("Require count 1..20 and valid base seed (default pilot count is 10)")
    candidate_dir, output, dev_dir = map(lambda p: Path(p).resolve(), (candidate_dir, output, dev_dir))
    if output.exists():
        raise ValueError("Pilot output exists; preserve all previous evidence")
    report = read_json(candidate_dir/"summary.json")
    rows = [json.loads(line) for line in (candidate_dir/"candidates.jsonl").read_text().splitlines()]
    samples = read_json(candidate_dir/"samples.json")
    cohort = [json.loads(line) for line in (dev_dir/"cohort.jsonl").read_text().splitlines()]
    frozen = read_json(dev_dir/"core_mask_eval/frozen_config.json")
    threshold = frozen["selected_weighted_coverage_threshold"]
    if (report["status"] != "COMPLETE" or len(rows) != report["candidate_count"]
            or file_sha256(candidate_dir/"candidates.jsonl") != report["candidates_sha256"]
            or frozen["status"] != "FROZEN_DEV_CONFIGURATION"
            or frozen["candidate_config_sha256"] != report["configuration_sha256"]):
        raise ValueError("Require complete unchanged candidates and matching frozen dev configuration")
    if [s["target"]["image_key"] for s in samples] != [r["image_key"] for r in cohort[:len(samples)]]:
        raise ValueError("Candidates must be the existing ordered dev subset")
    hashes = dict(report["inputs_sha256"])
    hashes.update({str(ROOT/name): value for name, value in report["code_sha256"].items()})
    for path in (candidate_dir/"summary.json", candidate_dir/"samples.json", candidate_dir/"candidates.jsonl",
                 dev_dir/"core_mask_eval/frozen_config.json", dev_dir/"cohort.jsonl"):
        if str(path) in hashes and hashes[str(path)] != file_sha256(path):
            raise ValueError("Conflicting candidate provenance")
        hashes[str(path)] = file_sha256(path)
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f"Changed dev input: {path}")
    available = select_candidates(samples, rows, mode=selection_mode, threshold=threshold)
    chosen = available[:count]
    if len({s["target"]["place_key"] for s, _ in chosen}) != len(chosen):
        raise ValueError("Pilot requires one SOURCE per place")
    output.mkdir(parents=True)
    records = []
    # Import numerical geometry only; no BoQ, VLM or generation in preparation.
    from targeted.candidate_masks import CoverageContext, assess_thresholds
    from targeted.render_mask import build_render_mask
    for index, (sample, row) in enumerate(chosen):
        target = sample["target"]
        source = verified_image(target["source_path"], target["source_sha256"]).convert("RGB")
        if source.size != (target["source_width"], target["source_height"]):
            raise ValueError("SOURCE dimensions changed")
        masks = {}
        for kind in ("core", "render"):
            path = (candidate_dir/row[f"{kind}_mask_path"]).resolve()
            if not path.is_relative_to(candidate_dir):
                raise ValueError("Candidate mask path outside manifest directory")
            masks[kind] = canonical_mask(verified_image(path, row[f"{kind}_mask_sha256"]), source.size)
            hashes[str(path)] = row[f"{kind}_mask_sha256"]
        core, render = (np.asarray(masks[k]) > 0 for k in ("core", "render"))
        if not core.any() or (core & ~render).any():
            raise ValueError("Nonempty Core must be contained in Render")
        artifact = Path(target["numerical_artifact"])
        if file_sha256(artifact) != target["numerical_artifact_sha256"]:
            raise ValueError("Continuous vulnerability changed")
        with np.load(artifact, allow_pickle=False) as data:
            ctx = CoverageContext(data["attention_roi_token_mask"], data[report["configuration"]["weight_map"]], core.shape)
        metrics, _ = ctx.measure(core)
        gates = assess_thresholds(metrics, report["configuration"]["families"][row["family"]], row["geometry"], report["configuration"])
        if gates != row["thresholds"]:
            raise ValueError("Candidate gates differ from actual Core pixels")
        expected_render, _ = build_render_mask(core, report["configuration"]["render"])
        if not np.array_equal(render, expected_render):
            raise ValueError("Render differs from configured bounded Core dilation")
        directory = output/f"{index:02d}_{row['candidate_id']}"
        directory.mkdir()
        source.save(directory/"source.png")
        for kind in masks:
            masks[kind].save(directory/f"{kind}_mask.png")
        prompt = prompt_for_family(row["family"])
        (directory/"prompt.txt").write_text(prompt+"\n", encoding="utf-8")
        record = dict(index=index, directory=directory.name, source_identity=target, candidate_id=row["candidate_id"],
                      family=row["family"], prompt=prompt, seed=seed+index, core_metrics=metrics,
                      thresholds=gates, frozen_threshold_pass=gates[str(threshold)]["passes"],
                      core_role="scientific targeting mask only; never passed to diffusion",
                      render_role="only sampling mask; bounded edge/shadow/blending allowance",
                      files_sha256={p.name: file_sha256(p) for p in directory.iterdir()})
        records.append(record)
    write_json(output/"planned_edits.json", records)
    protocol = dict(schema_version=1, requested_count=count, selected_count=len(records), available_count=len(available),
                    seed=seed, selection_mode=selection_mode, selection_rule=RULE, frozen_threshold=threshold,
                    scientific_gate_relaxed_for_render_pilot=selection_mode == "render_semantics",
                    frozen_scientific_config_modified=False, hardness_filtering=False, training=False,
                    generated_output_selection="none: one seed per query; retain every success/failure; no retries or ranking",
                    planned_edits_sha256=file_sha256(output/"planned_edits.json"), inputs_sha256=hashes,
                    code_sha256={name:file_sha256(ROOT/name) for name in (
                        "generation/targeted_editor.py", "generation/inpainting_editor.py", "generation/inpainting_pilot.py",
                        "generation/inpainting_validation.py", "scripts/stage3_pilot_inpainting.py")})
    write_json(output/"protocol.json", protocol)
    status = "PREPARED" if len(records) == count else "INSUFFICIENT_ELIGIBLE_QUERIES"
    write_json(output/"summary.json", dict(status=status, requested_count=count, selected_count=len(records),
               generated_count=0, diffusion_called=False, hardness_filtering=False, training=False))
    return protocol


def run_prepared(output):
    """Generate every preregistered sample exactly once; never select by hardness."""
    output = Path(output).resolve()
    protocol, summary = read_json(output/"protocol.json"), read_json(output/"summary.json")
    if summary["status"] not in {"PREPARED", "BLOCKED_MODEL_CONFIGURATION"} or summary["generated_count"]:
        raise ValueError("Require untouched, complete prepared pilot; no silent resume/retry")
    if protocol["selected_count"] != protocol["requested_count"]:
        raise ValueError("Not enough eligible queries; never silently relax gates")
    if file_sha256(output/"planned_edits.json") != protocol["planned_edits_sha256"]:
        raise ValueError("Prepared plan changed")
    records = read_json(output/"planned_edits.json")
    for path, digest in protocol["inputs_sha256"].items():
        if file_sha256(path) != digest:
            raise ValueError(f"Frozen pilot input changed: {path}")
    for path, digest in protocol["code_sha256"].items():
        if file_sha256(ROOT/path) != digest:
            raise ValueError(f"Pilot code changed; prepare a fresh plan: {path}")
    for record in records:
        directory = output/record["directory"]
        if (directory/"audit.json").exists() or (directory/"raw_output.png").exists():
            raise ValueError("Existing generated evidence; no overwrite")
        for name, digest in record["files_sha256"].items():
            if file_sha256(directory/name) != digest:
                raise ValueError("Prepared source/Core/Render/prompt changed")
    try:
        config = TrainedInpaintingConfig.from_env()
        checkpoint = inspect_checkpoint(config)
    except (ValueError, OSError, KeyError) as exc:
        summary.update(status="BLOCKED_MODEL_CONFIGURATION", error=f"{type(exc).__name__}: {exc}")
        write_json(output/"summary.json", summary)
        raise
    write_json(output/"sampling_config.json", dict(config=asdict(config), checkpoint=checkpoint,
               sampling_mask="render_mask", scheduler=(checkpoint["model_config"]["model_index.json"]["scheduler"][1]
                  if checkpoint["pipeline_class"]=="StableDiffusionXLInpaintPipeline" else "DDIMScheduler"),
               batch_size=1, eta=0., strength=1.,
               outside_render_exact_rgb_restoration=True))
    summary.update(status="RUNNING", samples=[], error=None)
    write_json(output/"summary.json", summary)
    try:
        editor = TrainedInpaintingEditor(config)
        first = records[0]
        with Image.open(output/first["directory"]/"source.png") as image:
            source = image.convert("RGB")
        with Image.open(output/first["directory"]/"render_mask.png") as image:
            render = image.copy()
        summary["backend_validation_attempted"] = True
        summary["diffusion_called"] = True
        validation = validate_backend(editor,source,render,first["prompt"],first["seed"],output/"backend_validation")
        summary.update(backend_validation_status=validation["status"], diffusion_called=True)
        if validation["status"] != "PASS":
            raise RuntimeError("Backend validation must pass before pilot generation")
    except Exception as exc:
        summary.update(status="BACKEND_VALIDATION_FAILED",error=f"{type(exc).__name__}: {exc}")
        write_json(output/"summary.json",summary)
        raise
    for record in records:
        directory = output/record["directory"]
        result = dict(index=record["index"], candidate_id=record["candidate_id"], family=record["family"],
                      seed=record["seed"], prompt=record["prompt"], source_identity=record["source_identity"])
        try:
            with Image.open(directory/"source.png") as image:
                source = image.convert("RGB")
            with Image.open(directory/"render_mask.png") as image:
                render = image.copy()
            summary["diffusion_called"] = True
            final = editor.edit(source, render, record["prompt"], record["seed"])
            raw = editor.last_raw_output
            raw.save(directory/"raw_output.png")
            if getattr(editor, "last_padded_raw_output", None) is not None:
                editor.last_padded_raw_output.save(directory/"raw_output_padded.png")
            final.save(directory/"final_output.png")
            outside = np.asarray(render) == 0
            exact = bool(np.array_equal(np.asarray(final)[outside], np.asarray(source)[outside]))
            ImageChops.difference(source, final).save(directory/"difference.png")
            result.update(status="GENERATED", outside_render_exact_rgb=exact, editor_audit=editor.last_audit,
                          source_pixel_sha256=pixel_sha256(source), raw_pixel_sha256=pixel_sha256(raw),
                          final_pixel_sha256=pixel_sha256(final), files_sha256={
                              p.name:file_sha256(p) for p in directory.iterdir() if p.name != "audit.json"})
            if not exact:
                raise RuntimeError("Outside-Render exact RGB restoration failed")
            summary["generated_count"] += 1
        except Exception as exc:
            result.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
        write_json(directory/"audit.json", result)
        summary["samples"].append(result)
        write_json(output/"summary.json", summary)
        print(f"Pilot {record['index']+1}/{len(records)}: {result['status']}", flush=True)
    summary.update(status="COMPLETE" if summary["generated_count"] == len(records) else "COMPLETE_WITH_ERRORS",
                   attempted_count=len(records), failed_count=len(records)-summary["generated_count"],
                   no_hardness_filtering=True, no_training=True)
    write_json(output/"summary.json", summary)
    return summary
