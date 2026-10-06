"""Train the online generator, freeze it, generate paired pools, train fresh SALAD A/B/C.

Evaluation is a separate explicit command. Each stage runs in a fresh process so
teacher and generator GPU memory cannot leak into downstream SALAD training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .data import atomic_json, require_empty_output
from .teacher import file_sha256

PREFIX = "AdaptVPR.experiments.vpr_guidance."


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("prompts", "gsv-root", "salad-root", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--conditions", nargs="+", default=["snow", "night", "rain", "fog"])
    for name, default in (("generation-passes", 2), ("chunk-size", 128),
                          ("generator-steps-per-chunk", 256), ("replay-rounds", 2),
                          ("lora-rank", 8), ("lora-alpha", 8), ("timestep-window", 10),
                          ("real-ratio", 8), ("synthetic-ratio", 1), ("salad-steps", 4000),
                          ("salad-workers", 8), ("salad-batch-size", 60),
                          ("salad-image-size", 224), ("seed", 42)):
        p.add_argument("--" + name, type=int, default=default)
    for name, default in (("generator-lr", 1e-4), ("lambda-diff", 1.),
                          ("lambda-vpr", .1), ("lambda-keep", .05)):
        p.add_argument("--" + name, type=float, default=default)
    p.add_argument("--cities", nargs="+", help="same GSV cities for all three fresh SALAD runs")
    p.add_argument("--tensorboard-dir", type=Path)
    p.add_argument("--resume", action="store_true", help="resume stages and the last completed generator round")
    return p


def make_shared_mix_plan(args, out):
    """Match B/C image exposure at every sampled place, including partial epochs."""
    import csv
    from collections import Counter
    from .data import canonical_place_id
    from .train_salad import DEFAULT_CITIES
    cities = args.cities or DEFAULT_CITIES
    eligible = set()
    for city in cities:
        with (args.gsv_root / "Dataframes" / f"{city}.csv").open() as handle:
            counts = Counter(canonical_place_id(r["place_id"]) for r in csv.DictReader(handle))
        eligible.update((city, pid) for pid, n in counts.items() if n >= 4)
    pools = []
    for variant in ("B", "C"):
        manifest = out / f"synthetic_{variant}/synthetic_manifest.jsonl"
        files = {}
        for line in manifest.read_text().splitlines():
            row = json.loads(line)
            if row["passed"] is not True or row["eligible_for_training"] is not True:
                raise ValueError("rejected image in downstream manifest")
            key = (row["city"], canonical_place_id(row["place_id"]))
            files.setdefault(key, set()).add(row["generated_path"])
        pools.append(files)
    capacities = [{"city": city, "place_id": pid,
                   "capacity": min(4, len(pools[0].get((city, pid), [])), len(pools[1].get((city, pid), [])))}
                  for city, pid in sorted(eligible)]
    capacities = [r for r in capacities if r["capacity"]]
    requested = round(len(eligible) * 4 * args.synthetic_ratio / (args.real_ratio + args.synthetic_ratio))
    actual = min(requested, sum(r["capacity"] for r in capacities))
    if actual == 0:
        raise ValueError("no common verified B/C places; downstream augmentation comparison would be empty")
    path = out / "shared_mix_plan.json"
    atomic_json(path, {"capacities": capacities, "total_places": len(eligible), "seed": args.seed,
                       "requested_ratio": [args.real_ratio, args.synthetic_ratio],
                       "target_synthetic_slots": requested, "matched_synthetic_slots": actual,
                       "matched_real_slots": len(eligible) * 4 - actual,
                       "coverage_limited": actual < requested})
    if actual < requested:
        print(f"B/C common coverage allows {actual}/{requested} requested synthetic slots per epoch. "
              "Actual exposure is matched and logged; add Global prompts to reach the requested ratio.", flush=True)
    return path


def main():
    args = args_parser().parse_args()
    out = args.output_dir.resolve()
    if not args.resume:
        require_empty_output(out)
    for path in (args.prompts, args.gsv_root / "Images", args.gsv_root / "Dataframes",
                 args.salad_root / "vpr_model.py"):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.real_ratio <= 0 or args.synthetic_ratio <= 0:
        raise ValueError("B/C require positive real and synthetic ratios")
    tb = (args.tensorboard_dir or out / "tensorboard").resolve()
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k != "resume"}
    config["prompts_sha256"] = file_sha256(args.prompts)
    config_path = out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("full pipeline resume settings changed")
    atomic_json(config_path, config)
    state_path = out / "pipeline_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    def run(stage, module, flags, artifacts):
        if stage in state:
            if any(not Path(path).exists() or file_sha256(path) != sha
                   for path, sha in state[stage]["artifacts"].items()):
                raise ValueError(f"completed stage {stage} artifacts were modified")
            print(f"completed: {stage}", flush=True)
            return
        command = [sys.executable, "-u", "-m", PREFIX + module, *map(str, flags)]
        print(f"running {stage}: {command}", flush=True)
        subprocess.run(command, check=True)
        state[stage] = {"artifacts": {str(path): file_sha256(path) for path in artifacts}}
        atomic_json(state_path, state)

    source_manifest = out / "generator/source_manifest.jsonl"
    shared = ["--prompts", args.prompts, "--gsv-root", args.gsv_root,
              "--salad-root", args.salad_root, "--conditions", *args.conditions]
    run("prepare", "prepare_data", [*shared, "--output-dir", out / "generator"], [source_manifest])
    generator = [*shared, "--output-dir", out / "generator", "--source-manifest", source_manifest,
                 "--tensorboard-dir", tb / "generator", "--seed", args.seed]
    mapping = {"generation_passes": "generation-passes", "chunk_size": "chunk-size",
               "generator_steps_per_chunk": "train-steps-per-chunk", "replay_rounds": "replay-rounds",
               "lora_rank": "rank", "lora_alpha": "alpha", "generator_lr": "lr",
               "lambda_diff": "lambda-diff", "lambda_vpr": "lambda-vpr", "lambda_keep": "lambda-keep",
               "timestep_window": "timestep-window"}
    for key, flag in mapping.items():
        generator += ["--" + flag, getattr(args, key)]
    final = out / "generator/final_lora.pt"
    if "generator" not in state and final.exists():
        # Finalization may have committed before the orchestrator journal flushed.
        import torch
        payload = torch.load(final, map_location="cpu", weights_only=True)
        if not payload.get("extra", {}).get("frozen"):
            raise ValueError("existing final LoRA is not a frozen online artifact")
        state["generator"] = {"artifacts": {str(final): file_sha256(final)}}
        atomic_json(state_path, state)
    if args.resume and "generator" not in state:
        latest = out / "generator/latest_checkpoint.json"
        if latest.exists():
            generator += ["--resume", json.loads(latest.read_text())["path"]]
    run("generator", "train_online_generator", generator, [final])
    frozen_sha = file_sha256(final)
    for variant in ("B", "C"):
        directory = out / f"synthetic_{variant}"
        flags = [*shared, "--output-dir", directory, "--seed", args.seed]
        if variant == "C":
            flags += ["--lora", final]
        if args.resume and directory.exists():
            flags += ["--resume"]
        run("synthetic_" + variant, "generate_dataset", flags,
            [directory / "synthetic_manifest.jsonl", directory / "records.jsonl", directory / "config.json"])
    # Inputs/policy/seed must match exactly; only the LoRA fingerprint differs.
    b = json.loads((out / "synthetic_B/config.json").read_text())
    c = json.loads((out / "synthetic_C/config.json").read_text())
    b.pop("lora_sha256"); c.pop("lora_sha256")
    if b != c:
        raise ValueError("B/C generation inputs, seeds or verifier settings differ")
    mix_plan = make_shared_mix_plan(args, out)
    for variant in ("A", "B", "C"):
        directory = out / f"salad_{variant}"
        flags = ["--salad-root", args.salad_root, "--gsv-root", args.gsv_root,
                 "--output-dir", directory, "--seed", args.seed,
                 "--tensorboard-dir", tb / f"salad_{variant}", "--real-ratio", args.real_ratio,
                 "--synthetic-ratio", 0 if variant == "A" else args.synthetic_ratio,
                 "--max-steps", args.salad_steps, "--workers", args.salad_workers,
                 "--batch-size", args.salad_batch_size, "--image-size", args.salad_image_size]
        if args.cities:
            flags += ["--cities", *args.cities]
        if variant != "A":
            flags += ["--synthetic-manifest", out / f"synthetic_{variant}/synthetic_manifest.jsonl",
                      "--shared-mix-plan", mix_plan]
        if args.resume and "salad_" + variant not in state and directory.exists():
            import torch
            checkpoint = directory / "checkpoints/last.ckpt"
            if checkpoint.exists():
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if payload["global_step"] >= args.salad_steps:
                    state["salad_" + variant] = {"artifacts": {str(checkpoint): file_sha256(checkpoint),
                                                               str(directory / "config.json"): file_sha256(directory / "config.json")}}
                    atomic_json(state_path, state)
                elif payload.get("callbacks", {}).get("SetDatasetEpoch", {}).get("epoch_complete"):
                    flags += ["--resume", checkpoint]
                else:
                    directory.rename(directory.with_name(directory.name + f".interrupted-{time.time_ns()}"))
            else:
                directory.rename(directory.with_name(directory.name + f".interrupted-{time.time_ns()}"))
        run("salad_" + variant, "train_salad", flags,
            [directory / "checkpoints/last.ckpt", directory / "config.json"])
    configs = [json.loads((out / f"salad_{v}/config.json").read_text()) for v in ("A", "B", "C")]
    if len({c["initialization_sha256"] for c in configs}) != 1:
        raise ValueError("A/B/C fresh SALAD initialization differs")
    exposure = []
    for variant in ("B", "C"):
        exposure.append([(r["epoch"], r["actual_synthetic_slots"], r["actual_real_slots"], r["global_step"])
                         for line in (out / f"salad_{variant}/mix_stats.jsonl").read_text().splitlines()
                         for r in [json.loads(line)]])
    if exposure[0] != exposure[1]:
        raise ValueError("B/C actual synthetic exposure differs")
    if file_sha256(final) != frozen_sha:
        raise ValueError("final generator changed during downstream training")
    atomic_json(out / "complete.json", {"final_lora_sha256": frozen_sha,
                                        "salad_initialization_sha256": configs[0]["initialization_sha256"]})
    print("Full training complete. Run evaluate_all separately on real datasets.", flush=True)


if __name__ == "__main__":
    main()
