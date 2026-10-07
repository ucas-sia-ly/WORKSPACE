"""Run reproducible SALAD feedback rounds, fresh final training, or an arm audit.

random: released IC-Light and random verified selection.
select: released IC-Light and current-student hardness selection.
full: hardness selection, plus conditional denoising LoRA on mined positives.

Verification failures can change the accepted groups across generators. Use
``audit --match-pools`` or ``final --match-pools`` for matched final training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

from common import GUIDANCE_ROOT, SALAD_ROOT, file_sha256, read_jsonl, write_json, write_jsonl

ARMS = ("random", "select", "full")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    parser.add_argument("stage", choices=["loop", "final", "audit"])
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--root", type=Path, required=True, help="Experiment directory shared by all arms")
    parser.add_argument("--gsv-root", type=Path)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--cities", nargs="+")
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--sources-per-round", type=int, default=500)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--student-init-epochs", type=int, default=10)
    parser.add_argument("--student-checkpoint", type=Path,
                        help="Optional shared starting student; otherwise train real-only student_0")
    parser.add_argument("--student-epochs", type=int, default=2)
    parser.add_argument("--final-epochs", type=int, default=10)
    parser.add_argument("--lora-steps", type=int, default=1000)
    parser.add_argument("--salad-args", default="--batch-size 32 --images-per-place 4 --num-workers 4")
    parser.add_argument("--score-args", default="", help="Extra scorer settings; batch/miner settings come from SALAD")
    parser.add_argument("--lora-args", default="")
    parser.add_argument("--eval-args", default="", help="Extra evaluation flags, e.g. --limit-queries 4")
    parser.add_argument("--eval", nargs="+", default=["queries", "queries_night", "queries_snow",
                                                      "queries_rain", "queries_overcast", "queries_sun"])
    parser.add_argument("--svox-root", type=Path)
    parser.add_argument("--backbone-repo", type=Path)
    parser.add_argument("--match-pools", action="store_true", help="Use the common verified prompt groups across all arms")
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="Interpreter for every child stage")
    parser.add_argument("--dry-run", action="store_true", help="Print commands; do not write outputs or load models")
    args = parser.parse_args(argv)
    if args.stage != "audit":
        for name in ("arm", "gsv_root", "cities"):
            if not getattr(args, name):
                parser.error(f"--{name.replace('_', '-')} is required for {args.stage}")
        if args.stage == "loop" and not args.prompts:
            parser.error("--prompts is required for loop")
    for name in ("rounds", "sources_per_round", "candidates", "student_init_epochs", "student_epochs",
                 "final_epochs", "lora_steps"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("root", "gsv_root", "prompts", "student_checkpoint", "svox_root", "backbone_repo", "python"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    args.cities = sorted(set(args.cities)) if args.cities else None
    return args


def _extra_flags(raw, reserved):
    flags = shlex.split(raw)
    for token in flags:
        if token.startswith("--"):
            option = token.split("=", 1)[0]
            # Child parsers allow abbreviations; disallow those for reserved flags too.
            if any(name.startswith(option) for name in reserved):
                raise ValueError(f"{option} is managed by run_loop; remove it from extra arguments")
    return flags


def _salad_module():
    spec = importlib.util.spec_from_file_location("feedback_salad_train", SALAD_ROOT / "train_salad.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _without_option(flags, option):
    result, index = [], 0
    while index < len(flags):
        token = str(flags[index])
        name = token.split("=", 1)[0]
        if name.startswith("--") and option.startswith(name) and name != "--backbone":
            index += 1 if "=" in token else 2
        else:
            result.append(flags[index])
            index += 1
    return result


def _input_signature(path):
    path = Path(path)
    if path.is_file():
        return {"method": "content_sha256", "sha256": file_sha256(path)}
    if not path.is_dir():
        raise FileNotFoundError(path)
    # Large image corpora are checked by inventory/stat; model/manifest files by content.
    digest, count = hashlib.sha256(), 0
    for child in sorted(path.rglob("*")):
        if child.is_file():
            stat = child.stat()
            digest.update(json.dumps([str(child.relative_to(path)), stat.st_size, stat.st_mtime_ns],
                                     ensure_ascii=False).encode() + b"\0")
            count += 1
    return {"method": "file_inventory_size_mtime", "sha256": digest.hexdigest(), "files": count}


def salad_flags(args, output_dir, epochs, manifest=None, init=None):
    extras = _extra_flags(args.salad_args, ["--real-data", "--output-dir", "--epochs", "--cities", "--seed",
                                           "--synthetic-manifest", "--resume", "--init-checkpoint",
                                           "--backbone-repo", "--check-data", "--synthetic-places-only"])
    if init is not None:
        # Full SALAD weights already contain the DINO backbone.
        extras = _without_option(extras, "--backbone-weights")
    flags = ["--real-data", str(args.gsv_root), "--output-dir", str(output_dir), "--epochs", str(epochs),
             "--cities", *args.cities, "--seed", str(args.seed), *extras]
    if manifest is not None:
        flags += ["--synthetic-manifest", str(manifest)]
    if init is not None:
        flags += ["--init-checkpoint", str(init)]
    if args.backbone_repo:
        flags += ["--backbone-repo", str(args.backbone_repo)]
    return flags


class Runner:
    """Skip only stages with matching command, inputs, source and complete outputs.

    Save a request before execution so a failed run cannot later resume with
    different options. A completion marker is written only after all artifacts
    exist. Child stages own their training-state recovery.
    """

    def __init__(self, dry_run=False, python=None):
        self.dry_run = dry_run
        self.python = str(python or sys.executable)

    def __call__(self, script, *arguments, complete=None, outputs=(), inputs=(), resume_arguments=None,
                 already_finished=False):
        command = [self.python, str(script), *map(str, arguments)]
        if self.dry_run:
            print("[run]", shlex.join(command), flush=True)
            return
        request = None
        if complete is not None:
            source_paths = sorted(set([Path(script), *GUIDANCE_ROOT.glob("*.py"),
                                       *(SALAD_ROOT / "workflow").glob("*.py"),
                                       *(SALAD_ROOT / "models").rglob("*.py"),
                                       *(GUIDANCE_ROOT.parents[1] / "adapters").glob("*.py"),
                                       *(GUIDANCE_ROOT.parents[1] / "prompts").glob("*.py"),
                                       *(GUIDANCE_ROOT.parents[1] / "verification").glob("*.py")]))
            request = {"schema_version": 1, "command": command,
                       "inputs": {str(Path(p).resolve()): _input_signature(p) for p in inputs},
                       "sources": {str(p): file_sha256(p) for p in source_paths}}
            request_path = Path(complete).with_name(Path(complete).stem + "_request.json")
            if request_path.exists():
                if json.loads(request_path.read_text()) != request:
                    raise ValueError(f"Stage options, inputs or code changed: {request_path}; use a new experiment root")
            else:
                write_json(request_path, request)
            if Path(complete).exists():
                saved = json.loads(Path(complete).read_text())
                if saved.get("request") != request:
                    raise ValueError(f"Completion metadata does not match this request: {complete}")
                if any(not Path(p).is_file() or file_sha256(p) != saved.get("outputs", {}).get(str(p))
                       for p in outputs):
                    raise ValueError(f"Completed stage has missing or changed artifacts: {complete}")
                print(f"[skip] {complete}", flush=True)
                return
        actual = command if resume_arguments is None else [self.python, str(script), *map(str, resume_arguments)]
        if not already_finished:
            print("[run]", shlex.join(actual), flush=True)
            subprocess.run(actual, check=True)
        if complete is not None:
            missing = [str(p) for p in outputs if not Path(p).is_file()]
            if missing:
                raise RuntimeError(f"Stage returned without required artifacts: {missing}")
            write_json(complete, {"request": request, "outputs": {str(p): file_sha256(p) for p in outputs}})


def salad_train(run, args, output_dir, epochs, manifest=None, init=None):
    original = salad_flags(args, output_dir, epochs, manifest, init)
    parsed = _salad_module().parse_args(original)
    metadata = [args.gsv_root / "Dataframes" / f"{city}.csv" for city in args.cities]
    inputs = metadata + [args.gsv_root / "Images" / city for city in args.cities]
    inputs += ([manifest] if manifest else []) + ([init] if init else [])
    if args.backbone_repo:
        inputs.append(args.backbone_repo)
    if parsed.backbone_weights:
        inputs.append(parsed.backbone_weights)
    checkpoint = output_dir / "checkpoint.pt"
    complete = output_dir / "training_complete.json"
    actual, finished = original, False
    if not run.dry_run and checkpoint.exists() and not complete.exists():
        from common import use_salad
        use_salad()
        from workflow.model import read_checkpoint
        saved = read_checkpoint(checkpoint)
        if saved.get("format_version") != 1 or "optimizer_state_dict" not in saved:
            raise ValueError(f"Interrupted SALAD requires a full training checkpoint: {checkpoint}")
        module = _salad_module()
        resume_flags = _without_option(salad_flags(args, output_dir, epochs, manifest), "--backbone-weights")
        resume_flags += ["--resume", str(checkpoint)]
        resumed = module.parse_args(resume_flags)
        config, _, provenance = module.resolve_model_initialization(resumed, saved)
        expected = module.resolved_training_config(resumed, config, provenance)
        previous = {"synthetic_places_only": False, "init_checkpoint": None,
                    "init_checkpoint_sha256": None, **saved["training_config"]}
        if previous != expected or saved["total_epochs"] != epochs:
            raise ValueError(f"Interrupted SALAD configuration does not match: {checkpoint}")
        # A crash after the final atomic checkpoint but before the marker is recoverable.
        finished = saved["epoch"] == epochs
        actual = resume_flags
    run(SALAD_ROOT / "train_salad.py", *original, complete=complete, outputs=[checkpoint], inputs=inputs,
        resume_arguments=actual, already_finished=finished)


def _pool_contract(rows):
    groups = {}
    outputs = set()
    for row in rows:
        key = row["sample_id"]
        if key in groups:
            raise ValueError(f"Duplicate selected prompt group: {key}")
        if row.get("passed") is not True or row.get("eligible_for_training") is not True:
            raise ValueError(f"Pool contains an unverified or ineligible image: {key}")
        source, output = str(Path(row["source_path"]).resolve()), str(Path(row["output_path"]).resolve())
        if output in outputs or source == output:
            raise ValueError(f"Duplicate or non-synthetic pool image: {output}")
        outputs.add(output)
        groups[key] = (source, row.get("condition"), row.get("prompt"))
    return groups


def audit_pools(root, match=False, write=True):
    pools = {arm: read_jsonl(root / arm / "final_pool.jsonl") for arm in ARMS
             if (root / arm / "final_pool.jsonl").is_file()}
    contracts = {arm: _pool_contract(rows) for arm, rows in pools.items()}
    common = set.intersection(*(set(c) for c in contracts.values())) if contracts else set()
    for key in common:
        if len({c[key] for c in contracts.values()}) != 1:
            raise ValueError(f"Arms disagree about the source/condition/prompt of {key}")
    matched = len(pools) == len(ARMS) and all(set(c) == common for c in contracts.values())
    report = {"arms_present": sorted(pools), "groups_per_arm": {a: len(r) for a, r in pools.items()},
              "composition_matched": matched, "common_groups": len(common),
              "excluded_groups": {a: sorted(set(c) - common) for a, c in contracts.items()}}
    if match:
        if len(pools) != len(ARMS) or not common:
            raise ValueError("Matched final training requires all three final pools and a nonempty common group set")
        if write:
            for arm, rows in pools.items():
                write_jsonl(root / arm / "matched_pool.jsonl", [r for r in rows if r["sample_id"] in common])
        report["matched_pool_groups"] = len(common)
    if write:
        write_json(root / "pool_comparison.json", report)
    return report


def loop(args, run):
    shared, arm_root = args.root / "shared", args.root / args.arm
    effective = _salad_module().parse_args(salad_flags(args, shared / "student_0", args.student_init_epochs))
    if effective.synthetic_fraction <= 0 or int(effective.images_per_place * effective.synthetic_fraction) < 1:
        raise ValueError("SALAD synthetic_fraction/images_per_place must allow at least one synthetic view")
    score_extras = _extra_flags(args.score_args, ["--candidates", "--checkpoint", "--real-data", "--cities",
                                                 "--output-dir", "--selection", "--seed", "--backbone-repo",
                                                 "--train-batch-size", "--images-per-place", "--min-images-per-place",
                                                 "--miner-margin"])
    lora_extras = _extra_flags(args.lora_args, ["--selected", "--output", "--steps", "--seed", "--init-lora", "--resume",
                                              "--check-data", "--help"])
    threshold_parser = argparse.ArgumentParser(add_help=False)
    threshold_parser.add_argument("--min-utility", type=float, default=0.0)
    threshold, _ = threshold_parser.parse_known_args(lora_extras)
    if not math.isfinite(threshold.min_utility):
        raise ValueError("--min-utility must be finite")
    if not run.dry_run:
        config = {"schema_version": 1, "gsv_root": str(args.gsv_root), "cities": args.cities,
                  "metadata": {city: file_sha256(args.gsv_root / "Dataframes" / f"{city}.csv") for city in args.cities},
                  "prompts": str(args.prompts), "prompts_sha256": file_sha256(args.prompts),
                  "conditions": sorted(args.conditions) if args.conditions else None,
                  "rounds": args.rounds, "sources_per_round": args.sources_per_round, "candidates": args.candidates,
                  "seed": args.seed, "student_init_epochs": args.student_init_epochs,
                  "student_epochs": args.student_epochs, "student_checkpoint": str(args.student_checkpoint) if args.student_checkpoint else None,
                  "student_checkpoint_sha256": file_sha256(args.student_checkpoint) if args.student_checkpoint else None,
                  "salad_args": shlex.split(args.salad_args), "score_args": score_extras,
                  "lora_steps": args.lora_steps, "lora_args": lora_extras,
                  "backbone_repo": str(args.backbone_repo) if args.backbone_repo else None}
        config_path = args.root / "experiment_config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError(f"Shared arm configuration changed: {config_path}; use a new experiment root")
        if not config_path.exists():
            write_json(config_path, config)
    student_ckpt = args.student_checkpoint
    if student_ckpt is None:
        student = shared / "student_0"
        salad_train(run, args, student, args.student_init_epochs)
        student_ckpt = student / "checkpoint.pt"
    elif not run.dry_run and not student_ckpt.is_file():
        raise FileNotFoundError(student_ckpt)
    lora, pool = None, []
    for r in range(args.rounds):
        round_dir = arm_root / f"round_{r}"
        candidates_dir = (shared / f"candidates_released_round_{r}" if lora is None else round_dir / "candidates")
        generation = ["--prompts", args.prompts, "--image-root", args.gsv_root / "Images",
                      "--output-dir", candidates_dir, "--cities", *args.cities,
                      "--offset", r * args.sources_per_round, "--num-sources", args.sources_per_round,
                      "--num-candidates", args.candidates, "--seed", args.seed]
        if args.conditions:
            generation += ["--conditions", *args.conditions]
        if lora is not None:
            generation += ["--lora", lora]
        run(GUIDANCE_ROOT / "generate_candidates.py", *generation)
        scoring = round_dir / "scoring"
        score = ["--candidates", candidates_dir / "candidates.jsonl", "--checkpoint", student_ckpt,
                 "--real-data", args.gsv_root, "--cities", *args.cities, "--output-dir", scoring,
                 "--selection", "random" if args.arm == "random" else "hardness", "--seed", args.seed,
                 "--train-batch-size", effective.batch_size, "--images-per-place", effective.images_per_place,
                 "--min-images-per-place", effective.min_images_per_place, "--miner-margin", effective.miner_margin,
                 *score_extras]
        if args.backbone_repo:
            score += ["--backbone-repo", args.backbone_repo]
        run(GUIDANCE_ROOT / "score_candidates.py", *score, complete=scoring / "scoring_complete.json",
            outputs=[scoring / "scored.jsonl", scoring / "selected.jsonl", scoring / "summary.json"],
            inputs=[candidates_dir / "candidates.jsonl", student_ckpt,
                    *(args.gsv_root / "Dataframes" / f"{city}.csv" for city in args.cities),
                    *(args.gsv_root / "Images" / city for city in args.cities),
                    *([args.backbone_repo] if args.backbone_repo else [])])
        pool_path = round_dir / "pool.jsonl"
        if not run.dry_run:
            pool += read_jsonl(scoring / "selected.jsonl")
            _pool_contract(pool)
            write_jsonl(pool_path, pool)
        next_student = round_dir / "student"
        salad_train(run, args, next_student, args.student_epochs, manifest=pool_path, init=student_ckpt)
        student_ckpt = next_student / "checkpoint.pt"
        if args.arm == "full" and r + 1 < args.rounds:
            lora_path = round_dir / "lora.safetensors"
            selected_files = [arm_root / f"round_{i}" / "scoring" / "selected.jsonl" for i in range(r + 1)]
            flags = ["--selected", *selected_files, "--output", lora_path, "--steps", args.lora_steps,
                     "--seed", args.seed, *lora_extras]
            if lora is not None:
                flags += ["--init-lora", lora]
            if not run.dry_run:
                # Reuse the trainer's eligibility contract without importing GPU packages in this process.
                rows = [row for p in selected_files for row in read_jsonl(p)]
                mined = [row for row in rows if row.get("passed") is True
                         and row.get("eligible_for_training") is True
                         and row.get("utility", 0) > threshold.min_utility]
                write_json(round_dir / "feedback.json", {"selected_total": len(rows), "mined_positives": len(mined),
                           "min_utility": threshold.min_utility,
                           "status": "train_lora" if mined else "skipped_no_mined_positives",
                           "previous_lora": str(lora) if lora else None})
                if not mined:
                    print(f"[feedback] round {r}: no mined positives; keep current generator", flush=True)
                    continue
            actual = flags
            recovery = lora_path.with_suffix(".training.pt")
            if not run.dry_run and recovery.exists():
                # Final-state recovery is handled by train_lora as well.
                actual = flags[:]
                if "--init-lora" in actual:
                    index = actual.index("--init-lora")
                    del actual[index:index + 2]
                actual += ["--resume", recovery]
            run(GUIDANCE_ROOT / "train_lora.py", *flags, complete=round_dir / "lora_complete.json",
                outputs=[lora_path, lora_path.with_suffix(".json")], inputs=selected_files + ([lora] if lora else []),
                resume_arguments=actual)
            lora = lora_path
    if not run.dry_run:
        write_jsonl(arm_root / "final_pool.jsonl", pool)


def final(args, run):
    arm_root = args.root / args.arm
    pool = arm_root / ("matched_pool.jsonl" if args.match_pools else "final_pool.jsonl")
    if args.match_pools and not run.dry_run:
        audit_pools(args.root, match=True)
    model_dir = arm_root / "final_salad"
    # Fresh random SALAD aggregator + pretrained DINOv2, independent of the feedback student.
    salad_train(run, args, model_dir, args.final_epochs, manifest=pool)
    if args.svox_root is None:
        print("[final] --svox-root not given; final training finished without evaluation")
        return
    extras = _extra_flags(args.eval_args, ["--checkpoint", "--dataset", "--dataset-root", "--query-subdirs",
                                          "--output", "--backbone-repo", "--check-data", "--eval-manifest"])
    for subdir in args.eval:
        if Path(subdir).name != subdir or subdir in {".", ".."}:
            raise ValueError(f"Invalid SVOX query folder: {subdir}")
        output = arm_root / "final_eval" / f"svox_{subdir}.json"
        flags = ["--checkpoint", model_dir / "checkpoint.pt", "--dataset", "SVOX",
                 "--dataset-root", args.svox_root, "--query-subdirs", subdir, "--output", output, *extras]
        if args.backbone_repo:
            flags += ["--backbone-repo", args.backbone_repo]
        run(SALAD_ROOT / "evaluate_salad.py", *flags, complete=output.with_suffix(".complete.json"),
            outputs=[output], inputs=[model_dir / "checkpoint.pt", args.svox_root,
                                     *([args.backbone_repo] if args.backbone_repo else [])])


def main(argv=None):
    args = parse_args(argv)
    run = Runner(args.dry_run, args.python)
    try:
        if args.stage == "audit":
            print(json.dumps(audit_pools(args.root, args.match_pools, write=not args.dry_run), indent=2))
        else:
            (loop if args.stage == "loop" else final)(args, run)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
