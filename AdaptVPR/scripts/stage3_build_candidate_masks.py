"""Enumerate all family Core/Render candidates for the existing dev scene audit.

Offline geometry only. No VLM calls, diffusion, selection of a winner or tuning
of the three predeclared weighted coverage thresholds.
"""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
from PIL import Image

# Resolve this model-free reader by file, avoiding unrelated installed `scripts`
# packages taking precedence over the repository's namespace directory.
_spec = importlib.util.spec_from_file_location("stage3_scene_audit_reader",ROOT/"scripts/stage3_audit_scene_planner.py")
_reader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reader)
BOQ_DEV, digest, load_inputs, read_jsonl, write_json = (
    _reader.BOQ_DEV, _reader.digest, _reader.load_inputs, _reader.read_jsonl, _reader.write_json,
)
from targeted.candidate_masks import generate_candidates
from targeted.family_constraints import DEFAULT_CONFIG, config_digest, load_constraints
from targeted.scene_planner import parse_scene_decision


def write_mask(path, mask):
    Image.fromarray(mask.astype(np.uint8)*255).save(path)
    with Image.open(path) as saved:
        if not np.array_equal(np.asarray(saved), mask.astype(np.uint8)*255):
            raise ValueError("Binary mask PNG round trip changed pixels")
    return digest(path)


def build_candidates(*, scene_audit, dev_dir, config_path, output, seed=0):
    scene_audit,output = Path(scene_audit).resolve(),Path(output).resolve()
    if output.exists():
        raise ValueError("Candidate output already exists; choose a fresh directory")
    if type(seed) is not int or seed<0:
        raise ValueError("seed must be nonnegative")
    scene_summary_path,scene_rows_path = scene_audit/"summary.json",scene_audit/"scene_assessments.jsonl"
    scene_summary = json.loads(scene_summary_path.read_text())
    assessments = read_jsonl(scene_rows_path)
    if (scene_summary["status"] not in ("COMPLETE","COMPLETE_WITH_SCHEMA_ERRORS")
            or digest(scene_rows_path)!=scene_summary["records_sha256"]
            or len(assessments)!=scene_summary["count"]):
        raise ValueError("Require a complete, hash-verified dev scene audit")
    scenes,provenance = load_inputs(dev_dir,count=len(assessments))
    for scene,row in zip(scenes,assessments):
        if row["target"]!=scene.identity:
            raise ValueError("Scene decisions are not bound to the same frozen dev SOURCE/NPZ identities")
        if row["decision"] is not None:
            parsed = parse_scene_decision(json.dumps(row["decision"],allow_nan=False))
            if parsed.to_dict()!=parse_scene_decision(row["response_attempts"][-1]["raw_response"]).to_dict():
                raise ValueError("Scene decision differs from the raw validated VLM response")
    config = load_constraints(config_path)
    hashes = dict(provenance["inputs_sha256"])
    hashes.update({str(path):digest(path) for path in (scene_summary_path,scene_rows_path,Path(config_path).resolve())})
    taus = [str(value) for value in config["weighted_coverage_thresholds"]]
    code_paths = ("targeted/family_constraints.py","targeted/candidate_masks.py","targeted/render_mask.py",
                  "targeted/scene_planner.py","scripts/stage3_audit_scene_planner.py","scripts/stage3_build_candidate_masks.py")
    code_hashes = {name:digest(ROOT/name) for name in code_paths}
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/"constraints.json",config)
    summary = dict(schema_version=1,status="RUNNING",sample_count=len(scenes),seed=seed,
                   configuration=config,configuration_sha256=config_digest(config),inputs_sha256=hashes,
                   code_sha256=code_hashes,selection="none; all configured hypotheses saved, including failures",
                   source_selection="existing scene audit; frozen dev order",weight_map=config["weight_map"],
                   weighted_coverage_scope="full_map",chosen_weighted_coverage_threshold=None,
                   tau_target_precision=config["tau_target_precision"],
                   metric_definitions=dict(
                       target_precision="native_pixel_area(Core intersect ROI) / native_pixel_area(Core)",
                       binary_roi_coverage="native_pixel_area(Core intersect ROI) / native_pixel_area(ROI)",
                       vulnerability_weighted_coverage="sum_t(original_weight[t] * Core_cell_fraction[t]) / sum_t(original_weight[t])",
                       roi_conditioned_weighted_coverage="sum_t(weight[t]*ROI[t]*Core_cell_fraction[t]) / sum_t(weight[t]*ROI[t]); diagnostic only",
                       token_occupancy="Core pixels in original native token cell / native pixels in that cell; no interpolation of weights",
                       connectivity="8-neighbor Core component count",
                       centroid_distance="Core-to-binary-ROI centroid distance, normalized by original image diagonal",
                       render="never included in Core metrics or coverage gates"),
                   geometry_warning="Footprint hypotheses only; scene support/contact and facade plane not verified",
                   vlm_called=False,diffusion_called=False)
    write_json(output/"summary.json",summary)
    counts,pass_counts = Counter(),{tau:Counter() for tau in taus}
    samples,all_ids = [],set()
    try:
        with (output/"candidates.jsonl").open("x",encoding="utf-8") as all_rows:
            for index,(scene,assessment) in enumerate(zip(scenes,assessments)):
                suffix = hashlib.sha256(scene.identity["image_key"].encode()).hexdigest()[:16]
                directory = output/f"{index:02d}_{suffix}"
                directory.mkdir()
                masks_dir = directory/"masks"
                masks_dir.mkdir()
                scene.original.save(directory/"source.png")
                scene.vulnerability_overlay.save(directory/"vulnerability_overlay.png")
                with np.load(scene.identity["numerical_artifact"],allow_pickle=False) as archive:
                    roi,weights = archive["attention_roi_token_mask"].copy(),archive[config["weight_map"]].copy()
                np.savez_compressed(directory/"vulnerability_token_inputs.npz",roi=roi,weights=weights,
                                    weight_map=np.array(config["weight_map"]),image_key=np.array(scene.identity["image_key"]))
                decision = assessment["decision"]
                families = decision["feasible_families"] if decision and decision["editable"] else []
                sample = dict(index=index,target=scene.identity,scene_decision=decision,families=families,
                              status="enumerated" if families else "no_candidates",
                              no_candidates_reason=None if families else ("scene_schema_error" if decision is None else "scene_rejected"),
                              candidate_count=0,by_family={},passes_by_threshold={tau:{} for tau in taus},
                              selected_candidate_id=None,chosen_weighted_coverage_threshold=None,
                              token_inputs_sha256=digest(directory/"vulnerability_token_inputs.npz"))
                local_counts,local_pass = Counter(),{tau:Counter() for tau in taus}
                ids,occupancies = [],[]
                with (directory/"candidates.jsonl").open("x",encoding="utf-8") as local_rows:
                    for candidate in generate_candidates(families=families,roi=roi,weights=weights,source=scene.original,
                                                         config=config,image_key=scene.identity["image_key"],seed=seed):
                        row = candidate.diagnostic
                        cid,family = row["candidate_id"],row["parameters"]["family"]
                        if cid in all_ids:
                            raise ValueError("Duplicate candidate identity")
                        all_ids.add(cid)
                        row.update(image_key=scene.identity["image_key"],place_key=scene.identity["place_key"],
                                   source_sha256=scene.identity["source_sha256"],family=family,sample_index=index,
                                   token_occupancy_path=(directory/"core_token_occupancy.npz").relative_to(output).as_posix(),
                                   token_occupancy_index=len(ids))
                        for kind,mask in (("core",candidate.core_mask),("render",candidate.render_mask)):
                            path = masks_dir/f"{cid}_{kind}.png"
                            row[f"{kind}_mask_sha256"] = write_mask(path,mask)
                            row[f"{kind}_mask_path"] = path.relative_to(output).as_posix()
                        text = json.dumps(row,sort_keys=True,allow_nan=False,separators=(",",":"))+"\n"
                        local_rows.write(text)
                        all_rows.write(text)
                        ids.append(cid)
                        occupancies.append(candidate.token_occupancy)
                        counts[family]+=1
                        local_counts[family]+=1
                        for tau in taus:
                            if row["thresholds"][tau]["passes"]:
                                pass_counts[tau][family]+=1
                                local_pass[tau][family]+=1
                occupancy_path = directory/"core_token_occupancy.npz"
                np.savez_compressed(occupancy_path,candidate_ids=np.array(ids,dtype="U24"),
                                    occupancy=np.stack(occupancies) if ids else np.empty((0,16,16),dtype=np.float64))
                sample.update(candidate_count=len(ids),by_family=dict(local_counts),
                              passes_by_threshold={tau:{family:local_pass[tau][family] for family in families} for tau in taus},
                              token_occupancy_sha256=digest(occupancy_path),
                              candidates_sha256=digest(directory/"candidates.jsonl"))
                write_json(directory/"summary.json",sample)
                samples.append(sample)
                all_rows.flush()
                print(f"{index+1}/{len(scenes)} {scene.identity['place_key']}: {len(ids)} candidates; "
                      f"passes="+str({tau:sum(local_pass[tau].values()) for tau in taus}),flush=True)
        for path,expected in {**hashes,**{str(ROOT/name):value for name,value in code_hashes.items()}}.items():
            if digest(path)!=expected:
                raise ValueError(f"Input/code changed during candidate generation: {path}")
        summary.update(status="COMPLETE",candidate_count=sum(counts.values()),candidate_counts_by_family=dict(counts),
                       passes_by_threshold={tau:{family:pass_counts[tau][family] for family in config["families"]} for tau in taus},
                       samples_with_any_passing_candidate={tau:sum(any(s["passes_by_threshold"][tau].values()) for s in samples) for tau in taus},
                       samples_without_families=sum(not s["families"] for s in samples),
                       candidates_sha256=digest(output/"candidates.jsonl"),input_files_unchanged=True)
        write_json(output/"samples.json",samples)
    except Exception as exc:
        summary.update(status="ERROR",error=f"{type(exc).__name__}: {exc}",completed_samples=len(samples))
        raise
    finally:
        write_json(output/"summary.json",summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-audit",type=Path,default=ROOT/"outputs/stage3_dev/scene_planner_audit")
    parser.add_argument("--dev-dir",type=Path,default=BOQ_DEV)
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/stage3_dev/candidate_masks")
    parser.add_argument("--seed",type=int,default=0)
    args = parser.parse_args(argv)
    try:
        summary = build_candidates(scene_audit=args.scene_audit,dev_dir=args.dev_dir,config_path=args.config,
                                   output=args.output,seed=args.seed)
    except (ValueError,OSError,KeyError,TypeError) as exc:
        parser.exit(2,f"Candidate generation failed: {exc}\n")
    print(json.dumps({key:summary[key] for key in ("status","candidate_count","samples_with_any_passing_candidate")},indent=2))


if __name__ == "__main__":
    main()
