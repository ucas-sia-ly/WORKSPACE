"""Formal round-based Generate -> Verify -> Train IC-Light LoRA trainer."""
from __future__ import annotations

import argparse
from collections import Counter, deque
from contextlib import ExitStack
import json
import random
import time
from pathlib import Path

import torch
from diffusers import DDIMScheduler
from PIL import Image

from AdaptVPR.verification.evaluator import DualTraitEvaluator, ROUTE_THRESHOLDS
from .data import atomic_json, balanced_chunks, sample_seed, validate_sample_id
from .generate_dataset import _accepted_global_result
from .iclight import (attach_lora, generate_released, load_iclight, load_lora,
                      sampling_policy, save_lora)
from .prepare_data import prepare_sources
from .teacher import file_sha256, load_salad, PREPROCESSING_VERSION, pil_tensor
from .train_generator import train_bilevel_step, report_trainable_parameters
from .bilevel import GSVRealIndex, construct_episode, EpisodeUnavailable
from .salad_factory import (add_meta_args, validate_meta_args, build_fresh_salad,
                            meta_identity, OBJECTIVE_VERSION)


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("prompts", "gsv-root", "salad-root", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--source-manifest", type=Path)
    p.add_argument("--conditions", nargs="+", default=["snow", "night", "rain", "fog"])
    for name, value in (("generation-passes", 2), ("chunk-size", 128),
                        ("train-steps-per-chunk", 256), ("replay-rounds", 2),
                        ("rank", 8), ("alpha", 8), ("timestep-window", 10), ("seed", 42)):
        p.add_argument("--" + name, type=int, default=value)
    for name, value in (("lr", 1e-4), ("lambda-diff", 1.),
                        ("lambda-keep", .05), ("grad-clip", 1.)):
        p.add_argument("--" + name, type=float, default=value)
    add_meta_args(p)
    p.add_argument("--tensorboard-dir", type=Path)
    p.add_argument("--disable-tensorboard", action="store_true")
    p.add_argument("--resume", type=Path, help="completed-round checkpoint only")
    return p


def read_source_manifest(path, teacher):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    seen, descriptors = set(), {}
    for row in rows:
        sid = validate_sample_id(row.get("sample_id"))
        if sid in seen or row.get("route") != "global":
            raise ValueError(f"duplicate/non-Global source: {sid}")
        seen.add(sid)
        if "baseline_path" in row:
            raise ValueError("offline baseline manifests cannot enter online training")
        if (row.get("teacher_sha256") != teacher.model_fingerprint or
                row.get("preprocessing_version") != PREPROCESSING_VERSION or
                row.get("sampling_policy") != sampling_policy(row["condition"])):
            raise ValueError(f"source manifest policy/teacher mismatch: {sid}")
        if file_sha256(row["source_path"]) != row["source_sha256"]:
            raise ValueError(f"source content changed: {sid}")
        descriptors[sid] = teacher.load_source_descriptor(
            row["source_descriptor"], row["source_path"]).cpu()
    if not rows:
        raise ValueError("source manifest is empty")
    return rows, descriptors


def sample_training_entry(current, replay, rng):
    if not current:
        raise ValueError("empty current pool: skip this round's updates")
    previous = [row for pool in replay for row in pool]
    pool = previous if previous and rng.random() < .5 else current
    entry = rng.choice(pool)
    if entry.get("passed") is not True or entry.get("eligible_for_training") is not True:
        raise ValueError("active training/replay pool contains rejected image")
    return entry


def write_step(writer, record, lr):
    for tag, value in record.items():
        if tag.startswith(("meta/", "generator/", "diagnostic/")) and isinstance(value, (int, float)):
            writer.add_scalar(tag, value, record["step"])
    writer.add_scalar("generator/lr", lr, record["step"])


def summarize_round(records, generation_pass, round_id, index):
    def metrics(pool):
        n = len(pool)
        return {"pass_rate": sum(r["passed"] for r in pool) / n,
                "accepted_count": sum(r["passed"] for r in pool),
                "rejected_count": sum(not r["passed"] for r in pool),
                "mean_s_geo": sum(r["s_geo"] for r in pool) / n,
                "mean_s_div": sum(r["s_div"] for r in pool) / n,
                "mean_salad_preservation_cosine": sum(r["salad_preservation_cosine"] for r in pool) / n}
    return {"generation_pass": generation_pass, "round": round_id, "round_index": index,
            "condition_distribution": dict(Counter(r["condition"] for r in records)),
            **metrics(records), "conditions": {
                c: metrics([r for r in records if r["condition"] == c])
                for c in sorted({r["condition"] for r in records})}}


def write_round(writer, stats):
    for key in ("pass_rate", "accepted_count", "rejected_count", "mean_s_geo",
                "mean_s_div", "mean_salad_preservation_cosine", "updates", "skipped_meta_updates"):
        writer.add_scalar("round/" + key, stats[key], stats["round_index"])
    for condition, metrics in stats["conditions"].items():
        for key in ("pass_rate", "mean_s_geo", "mean_s_div"):
            writer.add_scalar(f"round/{condition}/{key}", metrics[key], stats["round_index"])
        writer.add_scalar(f"round/{condition}/salad_cosine",
                          metrics["mean_salad_preservation_cosine"], stats["round_index"])
        writer.add_scalar(f"round/{condition}/generated_count",
                          stats["condition_distribution"][condition], stats["round_index"])


def recover_journal(path, key, boundary):
    """Keep audit tail separately, then resume from the committed checkpoint."""
    if not path.exists():
        return
    kept, tail = [], []
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
            committed = record[key] <= boundary
        except (ValueError, KeyError):
            committed = False
        (kept if committed else tail).append(line)
    if tail:
        path.with_name(path.name + f".interrupted-{time.time_ns()}").write_text("\n".join(tail) + "\n")
        path.write_text("\n".join(kept) + ("\n" if kept else ""))


def validate_bilevel_checkpoint(payload):
    version = payload.get("extra", {}).get("config", {}).get("objective_version")
    if version != OBJECTIVE_VERSION:
        raise ValueError(f"cannot resume old frozen-teacher/different objective checkpoint ({version!r}); "
                         f"expected {OBJECTIVE_VERSION}; start a new run")


def main():
    args = args_parser().parse_args()
    validate_meta_args(args)
    if args.resume:
        validate_bilevel_checkpoint(torch.load(args.resume, map_location="cpu", weights_only=True))
    for name in ("generation_passes", "chunk_size", "train_steps_per_chunk", "rank", "alpha", "timestep_window"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.replay_rounds < 0 or args.timestep_window > 25:
        raise ValueError("replay_rounds >= 0 and timestep_window <= 25 required")
    if args.lr <= 0 or args.grad_clip <= 0 or not all(
            torch.isfinite(torch.tensor(v)) and v >= 0 for v in
            (args.lr, args.grad_clip, args.lambda_diff, args.lambda_meta, args.lambda_keep)):
        raise ValueError("invalid optimizer/loss weights")
    if not any((args.lambda_diff, args.lambda_meta, args.lambda_keep)):
        raise ValueError("at least one loss weight must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full IC-Light online training")
    out = args.output_dir.resolve()
    if (out / "final_lora.pt").exists():
        raise FileExistsError("final_lora.pt is frozen; use a new run for further training")
    if not args.resume and ((out / "train.jsonl").exists() or (out / "rounds").exists()):
        raise FileExistsError("training output exists: resume a round checkpoint or use a new run")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    teacher = load_salad(repo=str(args.salad_root.resolve()))
    manifest = args.source_manifest or prepare_sources(args.prompts, args.gsv_root, args.salad_root,
                                                      out, args.conditions, teacher=teacher)
    rows, descriptors = read_source_manifest(manifest, teacher)
    if any(r["condition"] not in args.conditions for r in rows):
        raise ValueError("source manifest contains conditions outside this run")
    if Path(manifest).resolve() != out / "source_manifest.jsonl":
        out.mkdir(parents=True, exist_ok=True)
        (out / "source_manifest.jsonl").write_bytes(Path(manifest).read_bytes())
    schedule = [(p, r, chunk) for p in range(args.generation_passes)
                for r, chunk in enumerate(balanced_chunks(rows, args.chunk_size, args.seed, p))]
    real_index = GSVRealIndex(args.gsv_root, cities=sorted({r["city"] for r in rows}))
    # Validate every manifest label before any accepted pool can train.
    for row in rows:
        real_index.key_for_row(row)
    meta_salad = build_fresh_salad(args.salad_root, meta=True, seed=args.seed, device="cuda",
                                   train_backbone_blocks=args.meta_train_backbone_blocks)
    identity = meta_identity(meta_salad, args.salad_root)
    print("inner-loop trainable parameter names:\n" + "\n".join(identity["meta_trainable_names"]))
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in {"resume", "tensorboard_dir", "disable_tensorboard", "source_manifest"}}
    config.update(objective_version=OBJECTIVE_VERSION, **identity,
                  gsv_dataframe_sha256={city: file_sha256(args.gsv_root / "Dataframes" / f"{city}.csv")
                                        for city in real_index.cities},
                  source_manifest_sha256=file_sha256(out / "source_manifest.jsonl"),
                  teacher_sha256=teacher.model_fingerprint, preprocessing_version=PREPROCESSING_VERSION)
    if (out / "config.json").exists() and json.loads((out / "config.json").read_text()) != config:
        raise ValueError("resume config differs from original run")
    atomic_json(out / "config.json", config)
    t2i, i2i, vae = load_iclight()
    unet = t2i.unet
    if unet is not i2i.unet:
        raise RuntimeError("IC-Light stages must share their UNet")
    vae.requires_grad_(False).eval()
    t2i.text_encoder.requires_grad_(False).eval()
    payload = load_lora(unet, args.resume) if args.resume else None
    trainable = ([p for p in unet.parameters() if p.requires_grad] if payload
                 else attach_lora(unet, args.rank, args.alpha))
    unet.enable_gradient_checkpointing()
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    scheduler = DDIMScheduler.from_config(t2i.scheduler.config)
    scheduler.set_timesteps(25, device="cuda")
    timesteps = [int(t) for t in scheduler.timesteps[-args.timestep_window:]]
    rng = random.Random(args.seed)
    replay = deque(maxlen=args.replay_rounds)
    start, step = 0, 0
    ordering = [[row["sample_id"] for row in chunk] for _, _, chunk in schedule]
    if payload:
        validate_bilevel_checkpoint(payload)
        saved = payload.get("extra", {})
        if saved.get("boundary") != "completed_round":
            raise ValueError("mid-round/incomplete checkpoints cannot resume online training")
        if saved.get("config") != config or saved.get("sample_ordering") != ordering:
            raise ValueError("checkpoint config, teacher or ordering differs")
        if (payload["rank"], payload["alpha"]) != (args.rank, args.alpha):
            raise ValueError("checkpoint rank/alpha differs")
        start, step = saved["next_round_index"], saved["global_step"]
        replay.extend(saved["active_replay"])
        for pool in replay:
            for row in pool:
                if (not row["passed"] or not row["eligible_for_training"] or
                        file_sha256(row["generated_path"]) != row["generated_sha256"]):
                    raise ValueError("replay candidate missing, modified or rejected")
        opt.load_state_dict(saved["optimizer"])
        rng.setstate(saved["sample_rng"])
        random.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
    report_trainable_parameters(unet, trainable, {"VAE": vae, "text encoder": t2i.text_encoder,
                                                "SALAD teacher": teacher.model})
    verifier = DualTraitEvaluator()
    verifier._load_matcher()
    policy = {"route": "global", **ROUTE_THRESHOLDS["global"],
              "matcher_name": verifier.matcher_name, "img_size": verifier.img_size,
              "n_kpts": verifier.n_kpts,
              "clip_model": getattr(getattr(verifier.model, "config", None), "_name_or_path", None)}
    if payload and saved["verifier_policy"] != policy:
        raise ValueError("resume Global verifier policy changed")
    # Verifier construction can consume RNG; restore again for exact boundary continuation.
    if payload:
        random.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
    if not payload:
        initial = out / "checkpoints/initial.pt"
        save_lora(unet, initial, rank=args.rank, alpha=args.alpha, extra={
            "boundary": "completed_round", "global_step": 0, "generation_pass": -1,
            "round": -1, "next_round_index": 0, "sample_ordering": ordering,
            "active_replay": [], "optimizer": opt.state_dict(), "sample_rng": rng.getstate(),
            "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(), "config": config,
            "teacher_sha256": teacher.model_fingerprint, "verifier_policy": policy})
        atomic_json(out / "latest_checkpoint.json", {"path": str(initial)})
    recover_journal(out / "train.jsonl", "step", step)
    recover_journal(out / "round_metrics.jsonl", "round_index", start - 1)
    with ExitStack() as stack:
        log = stack.enter_context((out / "train.jsonl").open("a", encoding="utf-8"))
        round_log = stack.enter_context((out / "round_metrics.jsonl").open("a", encoding="utf-8"))
        writer = None
        if not args.disable_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(args.tensorboard_dir or out / "tensorboard"), purge_step=0, flush_secs=10)
            stack.callback(writer.close)
            for line in (out / "train.jsonl").read_text().splitlines():
                write_step(writer, json.loads(line), args.lr)
            for line in (out / "round_metrics.jsonl").read_text().splitlines():
                write_round(writer, json.loads(line))
        for index in range(start, len(schedule)):
            generation_pass, round_id, chunk = schedule[index]
            directory = out / "rounds" / f"pass_{generation_pass:02d}_round_{round_id:03d}"
            if directory.exists():
                directory.rename(directory.with_name(directory.name + f".interrupted-{time.time_ns()}"))
            for name in ("accepted", "rejected"):
                (directory / name).mkdir(parents=True, exist_ok=True)
            unet.eval()
            accepted, records = [], []
            with (directory / "records.jsonl").open("w", encoding="utf-8") as audit:
                for row in chunk:
                    seed = sample_seed(args.seed, row["sample_id"], generation_pass, round_id)
                    with Image.open(row["source_path"]) as image:
                        source = image.convert("RGB")
                    generated = generate_released(t2i, i2i, vae, source, row["prompt"],
                                                  row["negative_prompt"], seed, row["condition"])
                    result = verifier.evaluate(source, generated, route="global",
                                               entry={"weather": row["condition"]})
                    passed = _accepted_global_result(result)
                    path = directory / ("accepted" if passed else "rejected") / f"{row['sample_id']}.png"
                    generated.save(path)
                    with torch.no_grad():
                        cosine = float((teacher.from_pil(generated) * descriptors[row["sample_id"]].to(teacher.device)).sum())
                    rec = dict(row, generated_path=str(path), generated_sha256=file_sha256(path),
                               seed=seed, generation_pass=generation_pass, round=round_id,
                               s_geo=result.s_geo, s_div=result.s_div, passed=passed,
                               eligible_for_training=passed, verifier_policy=policy,
                               salad_preservation_cosine=cosine)
                    audit.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    audit.flush()
                    records.append(rec)
                    if passed:
                        accepted.append(rec)
                    if writer and len(records) <= 4:
                        source_thumb = source.resize((322, 322))
                        generated_thumb = generated.resize((322, 322))
                        panel = torch.cat([pil_tensor(source_thumb), pil_tensor(generated_thumb)], dim=3)[0]
                        tag = f"round_samples/{len(records)}"
                        writer.add_image(tag, panel, index)
                        writer.add_text(tag + "/caption", f"{row['condition']} geo={result.s_geo:.3f} "
                                        f"div={result.s_div:.3f} cosine={cosine:.3f} passed={passed}", index)
                    print(f"pass={generation_pass} round={round_id} generated {row['sample_id']} accepted={passed}", flush=True)
            stats = summarize_round(records, generation_pass, round_id, index)
            stats["updates"] = 0
            stats["skipped_meta_updates"] = 0
            stats["meta_skip_reasons"] = {}
            stats["global_step_before"] = step
            unet.train()
            for _ in range(args.train_steps_per_chunk):
                try:
                    episode = construct_episode(accepted, replay, real_index, rng,
                        places=args.meta_places, support_real_per_place=args.meta_support_real_per_place,
                        query_real_per_place=args.meta_query_real_per_place)
                except EpisodeUnavailable as exc:
                    # The eligible pool stays unchanged during this round.
                    remaining = args.train_steps_per_chunk - stats["updates"]
                    stats["skipped_meta_updates"] += remaining
                    stats["meta_skip_reasons"][str(exc)] = remaining
                    print(f"round={index} skipped {remaining} meta updates: {exc}", flush=True)
                    break
                rec = train_bilevel_step(episode, step=step + 1, args=args, t2i=t2i, vae=vae,
                    unet=unet, meta_salad=meta_salad, scheduler=scheduler, timesteps=timesteps,
                    opt=opt, trainable=trainable, rng=rng, teacher=teacher,
                    source_descriptors=descriptors)
                step += 1
                stats["updates"] += 1
                rec.update(generation_pass=generation_pass, round=round_id,
                           target_rounds=[p.synthetic_row["round"] for p in episode.places],
                           target_passes=[p.synthetic_row["generation_pass"] for p in episode.places])
                log.write(json.dumps(rec, allow_nan=False) + "\n")
                log.flush()
                if writer:
                    write_step(writer, rec, opt.param_groups[0]["lr"])
                if step % 10 == 0:
                    print(f"step={step} meta={rec['generator/loss_meta']:.4f} "
                          f"meta_grad={rec['generator/meta_only_lora_grad_norm']:.6g}", flush=True)
            replay.append(accepted)
            stats["global_step_after"] = step
            if writer:
                write_round(writer, stats)
            round_log.write(json.dumps(stats) + "\n")
            round_log.flush()
            if writer:
                writer.flush()
            extra = {"boundary": "completed_round", "global_step": step,
                     "generation_pass": generation_pass, "round": round_id,
                     "next_round_index": index + 1, "sample_ordering": ordering,
                     "active_replay": list(replay), "optimizer": opt.state_dict(),
                     "sample_rng": rng.getstate(), "python_rng": random.getstate(),
                     "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                     "config": config, "teacher_sha256": teacher.model_fingerprint,
                     "verifier_policy": policy}
            checkpoint = out / "checkpoints" / f"round_{index:06d}.pt"
            temp = checkpoint.with_suffix(".partial")
            save_lora(unet, temp, rank=args.rank, alpha=args.alpha, extra=extra)
            temp.replace(checkpoint)
            atomic_json(out / "latest_checkpoint.json", {"path": str(checkpoint)})
        if step == 0:
            raise RuntimeError("no verified samples were trained; refusing to publish an untrained final LoRA")
        temp = out / "final_lora.partial"
        save_lora(unet, temp, rank=args.rank, alpha=args.alpha,
                  extra={"frozen": True, "global_step": step, "config": config,
                         "teacher_sha256": teacher.model_fingerprint, "verifier_policy": policy})
        temp.replace(out / "final_lora.pt")


if __name__ == "__main__":
    main()
