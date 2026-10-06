"""Train IC-Light LoRA with frozen SALAD feedback on Global-route data."""
from __future__ import annotations

import argparse, json, random
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DDIMScheduler

from .data import require_empty_output, validate_sample_id
from .iclight import (attach_lora, conditioning_source, decode_latent_01, encode_image_latent,
                      encode_prompt, image_tensor_01, load_iclight, load_lora,
                      sampling_policy, save_lora)
from .teacher import file_sha256, load_salad


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=8)
    p.add_argument("--lambda-diff", type=float, default=1.0)
    p.add_argument("--lambda-vpr", type=float, default=0.1)
    p.add_argument("--lambda-keep", type=float, default=0.05)
    p.add_argument("--timestep-window", type=int, default=10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--salad-repo", default="serizba/salad")
    p.add_argument("--resume", type=Path, help="resume a generator checkpoint including optimizer/RNG state")
    p.add_argument("--tensorboard-dir", type=Path,
                   help="write live TensorBoard scalars and import existing train.jsonl history")
    return p.parse_args()


def validate_args(args):
    for name in ("max_steps", "rank", "alpha", "timestep_window", "save_every"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.timestep_window > 25:
        raise ValueError("--timestep-window cannot exceed the released 25-step schedule")
    for name in ("lr", "grad_clip"):
        value = getattr(args, name)
        if not torch.isfinite(torch.tensor(value)) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    weights = (args.lambda_diff, args.lambda_vpr, args.lambda_keep)
    if not all(torch.isfinite(torch.tensor(x)) and x >= 0 for x in weights) or not any(weights):
        raise ValueError("loss weights must be finite, nonnegative, and at least one must be positive")


def read_manifest(path):
    path = Path(path).resolve()
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    seen = set()
    for row in rows:
        sample_id = validate_sample_id(row.get("sample_id"))
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id in generator manifest: {sample_id}")
        seen.add(sample_id)
        if row.get("route") != "global":
            raise ValueError(f"generator training requires Global-route samples: {sample_id}")
        required = {"source_path", "baseline_path", "source_descriptor", "prompt", "condition",
                    "baseline_sha256", "source_sha256", "sampling_policy", "city", "place_id"}
        if not required.issubset(row):
            raise ValueError(f"unvalidated/legacy generator manifest: {sample_id}; rerun prepare_data")
        if row["sampling_policy"] != sampling_policy(row["condition"]):
            raise ValueError(f"baseline sampling policy mismatch: {sample_id}; rerun prepare_data")
        for key in ("source_path", "baseline_path", "source_descriptor"):
            value = Path(row[key]).expanduser()
            if not value.is_absolute():
                value = path.parent / value
            row[key] = str(value.resolve(strict=True))
        for key in ("source", "baseline"):
            if file_sha256(row[f"{key}_path"]) != row[f"{key}_sha256"]:
                raise ValueError(f"{key} content mismatch: {sample_id}; rerun prepare_data")
        if Path(row["baseline_path"]).stem != sample_id or Path(row["source_descriptor"]).stem != sample_id:
            raise ValueError(f"baseline/descriptor sample_id mismatch: {sample_id}")
    return rows


def predict_x0(scheduler, z_t, eps, timestep):
    if scheduler.config.prediction_type != "epsilon":
        raise ValueError("IC-Light training requires scheduler prediction_type='epsilon'")
    alpha = scheduler.alphas_cumprod[int(timestep)].to(z_t.device, z_t.dtype)
    beta = 1.0 - alpha
    return (z_t - beta.sqrt() * eps) / alpha.sqrt().clamp_min(1e-6)


def first_order_guidance_proxy(predicted_x0, grad_x0):
    """Return a scalar whose derivative is the chain-rule VPR/keep gradient.

    grad_x0 is evaluated at the same predicted x0 in Pass A. Detaching this
    cotangent prevents a second derivative; the sum (not mean) is the vector-
    Jacobian product for the already-reduced guidance loss.
    """
    if predicted_x0.shape != grad_x0.shape:
        raise ValueError("guidance gradient and predicted x0 must have identical shapes")
    return (predicted_x0.float() * grad_x0.detach()).sum()


def report_trainable_parameters(unet, trainable, frozen_models):
    names = [name for name, p in unet.named_parameters() if p.requires_grad]
    if any("lora_" not in name for name in names):
        raise RuntimeError("base UNet parameters unexpectedly require gradients")
    expected = {id(p) for p in unet.parameters() if p.requires_grad}
    if not expected or expected != {id(p) for p in trainable}:
        raise RuntimeError("optimizer parameters do not match all trainable LoRA parameters")
    for name, model in frozen_models.items():
        if any(p.requires_grad for p in model.parameters()):
            raise RuntimeError(f"{name} must be completely frozen")
    total = sum(p.numel() for p in unet.parameters())
    count = sum(p.numel() for p in trainable)
    print("trainable parameter names:\n" + "\n".join(names))
    print(f"trainable={count:,} total={total:,} ratio={count / total:.6%} dtype=float32")


def _run_config(args, teacher):
    return {
        "manifest_sha256": file_sha256(args.manifest),
        "teacher_sha256": teacher.model_fingerprint,
        **{name: getattr(args, name) for name in
           ("rank", "alpha", "lr", "lambda_diff", "lambda_vpr", "lambda_keep",
            "timestep_window", "grad_clip", "seed")},
    }


def training_state(step, run_config, optimizer, sample_rng):
    return {
        "step": step, "run_config": run_config, "optimizer": optimizer.state_dict(),
        "sample_rng": sample_rng.getstate(), "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
    }


def restore_training_state(saved, run_config, optimizer, sample_rng, max_steps):
    required = {"step", "run_config", "optimizer", "sample_rng", "python_rng", "torch_rng", "cuda_rng"}
    if not isinstance(saved, dict) or not required.issubset(saved):
        raise ValueError("checkpoint has no complete training state; it cannot resume training")
    if saved["run_config"] != run_config:
        raise ValueError("resume manifest, teacher, or training settings differ from checkpoint")
    start_step = saved["step"]
    if isinstance(start_step, bool) or not isinstance(start_step, int) or start_step < 0:
        raise ValueError("resume global step must be a nonnegative integer")
    if max_steps <= start_step:
        raise ValueError("--max-steps must exceed the resumed global step")
    optimizer.load_state_dict(saved["optimizer"])
    sample_rng.setstate(saved["sample_rng"])
    random.setstate(saved["python_rng"])
    torch.set_rng_state(saved["torch_rng"].cpu())
    torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
    return start_step


def prepare_training_log(output_dir, start_step, run_config, resume):
    """Append only when this directory ends exactly at the restored checkpoint."""
    output_dir = Path(output_dir)
    log_path, config_path = output_dir / "train.jsonl", output_dir / "run_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != run_config:
            raise ValueError("output directory belongs to a different generator run; use a new --output-dir")
    elif log_path.exists() and log_path.stat().st_size:
        raise ValueError("existing training log has no run identity; use a new --output-dir")
    if resume:
        previous = None
        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                step = json.loads(line)["step"]
                if (isinstance(step, bool) or not isinstance(step, int)
                        or step <= 0 or (previous is not None and step != previous + 1)):
                    raise ValueError("existing log steps are not consecutive; use a new --output-dir")
                previous = step
            if previous is not None and previous != start_step:
                raise ValueError("training log does not end at resumed checkpoint; use a new --output-dir")
        for checkpoint in (output_dir / "checkpoints").glob("step_*.pt"):
            step_text = checkpoint.stem.removeprefix("step_")
            if step_text.isdigit() and int(step_text) > start_step:
                raise ValueError("output has checkpoints after resumed step; use a new --output-dir")
    if not config_path.exists():
        with config_path.open("x", encoding="utf-8") as handle:
            json.dump(run_config, handle, indent=2)
    return log_path.open("a" if resume else "w", encoding="utf-8")


def write_tensorboard_record(writer, record, run_config):
    step = record["step"]
    tags = {
        "loss/diffusion": "loss_diff", "loss/vpr": "loss_vpr",
        "loss/keep": "loss_keep", "loss/total": "loss_total",
        "loss/guidance_proxy": "loss_proxy", "quality/salad_cosine": "salad_cosine",
        "grad/lora_before_clip": "grad_norm",
        "grad/guidance_x0": "guidance_x0_grad_norm", "sampling/timestep": "timestep",
    }
    for tag, key in tags.items():
        writer.add_scalar(tag, record[key], step)
    for loss in ("diff", "vpr", "keep"):
        writer.add_scalar(f"weighted_loss/{loss}",
                          run_config[f"lambda_{loss}"] * record[f"loss_{loss}"], step)
    writer.add_scalar("optimizer/learning_rate", run_config["lr"], step)
    writer.add_scalar(f"condition/{record['condition']}/salad_cosine", record["salad_cosine"], step)


def prepare_tensorboard(log_dir, log_path, start_step, run_config):
    """Rebuild persisted history, purging stale/duplicate events on each restart.

    train.jsonl is validated against the restored checkpoint before this call.
    Replaying it also recovers metrics that were not flushed before a failure.
    Use a TensorBoard directory dedicated to this training run.
    """
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=str(log_dir), purge_step=0, flush_secs=10)
    try:
        writer.add_text("run/config", json.dumps(run_config, indent=2), start_step)
        with Path(log_path).open(encoding="utf-8") as history:
            for line in history:
                if line.strip():
                    record = json.loads(line)
                    if record["step"] <= start_step:
                        write_tensorboard_record(writer, record, run_config)
        writer.flush()
    except BaseException:
        writer.close()
        raise
    return writer


def _tensor_stats(name: str, x: torch.Tensor) -> str:
    y = x.detach().float()
    finite = torch.isfinite(y)
    if not finite.any():
        return f"{name}: shape={tuple(x.shape)} dtype={x.dtype} finite=0/{y.numel()}"
    vals = y[finite]
    return (
        f"{name}: shape={tuple(x.shape)} dtype={x.dtype} "
        f"finite={int(finite.sum())}/{y.numel()} "
        f"min={float(vals.min()):.6g} max={float(vals.max()):.6g} "
        f"mean={float(vals.mean()):.6g}"
    )


def _require_finite(name: str, x: torch.Tensor, *, step: int, sample_id: str, timestep: int):
    if torch.isfinite(x).all():
        return
    raise FloatingPointError(
        f"non-finite {name} at step={step} sample_id={sample_id} timestep={timestep}\n"
        + _tensor_stats(name, x)
    )


def main():
    args = args_parser()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed); torch.manual_seed(args.seed)
    rows = read_manifest(args.manifest)
    out = args.output_dir.resolve()
    if not args.resume:
        require_empty_output(out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    t2i, _, vae = load_iclight()
    unet = t2i.unet
    vae.requires_grad_(False).eval(); t2i.text_encoder.requires_grad_(False).eval()
    resume_payload = load_lora(unet, args.resume) if args.resume else None
    if resume_payload is not None:
        if (resume_payload["rank"], resume_payload["alpha"]) != (args.rank, args.alpha):
            raise ValueError("resume LoRA rank/alpha differs from current arguments")
        trainable = [p for p in unet.parameters() if p.requires_grad]
    else:
        trainable = attach_lora(unet, args.rank, args.alpha)
    unet.enable_gradient_checkpointing(); unet.train()
    teacher = load_salad(repo=args.salad_repo)

    scheduler = DDIMScheduler.from_config(t2i.scheduler.config)
    if scheduler.config.prediction_type != "epsilon":
        raise ValueError("IC-Light training requires scheduler prediction_type='epsilon'")
    scheduler.set_timesteps(25, device="cuda")
    timesteps = [int(x) for x in scheduler.timesteps[-min(args.timestep_window, 25):]]

    if any(p.dtype != torch.float32 for p in trainable):
        raise RuntimeError("trainable LoRA parameters must be fp32 before constructing AdamW")
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    rng = random.Random(args.seed)
    report_trainable_parameters(unet, trainable, {
        "VAE": vae, "text encoder": t2i.text_encoder, "SALAD teacher": teacher.model,
    })
    run_config = _run_config(args, teacher)
    start_step = 0
    if resume_payload is not None:
        start_step = restore_training_state(resume_payload.get("extra"), run_config,
                                            opt, rng, args.max_steps)
        print(f"resumed global_step={start_step}")

    with ExitStack() as resources:
        log = resources.enter_context(prepare_training_log(out, start_step, run_config, bool(args.resume)))
        writer = None
        if args.tensorboard_dir is not None:
            writer = prepare_tensorboard(args.tensorboard_dir, out / "train.jsonl", start_step, run_config)
            resources.callback(writer.close)
            print(f"TensorBoard scalars: {args.tensorboard_dir.resolve()}")
        for step in range(start_step + 1, args.max_steps + 1):
            row = rows[rng.randrange(len(rows))]
            sample_id = str(row["sample_id"])
            source = Image.open(row["source_path"]).convert("RGB")
            baseline = Image.open(row["baseline_path"]).convert("RGB")
            width, height = baseline.width // 8 * 8, baseline.height // 8 * 8
            z0 = encode_image_latent(baseline, vae, width, height)
            from AdaptVPR.adapters import iclight_sd15_fc as adapter
            cond = adapter._concat_condition(conditioning_source(source), vae, width, height)
            text = encode_prompt(t2i, row["prompt"])
            src_desc = teacher.load_source_descriptor(row["source_descriptor"], row["source_path"])
            base_img = image_tensor_01(baseline, width, height, "cuda")

            ts = timesteps[rng.randrange(len(timesteps))]
            t = torch.tensor([ts], device="cuda", dtype=torch.long)
            noise = torch.randn_like(z0)
            zt = scheduler.add_noise(z0, noise, t)
            _require_finite("z0", z0, step=step, sample_id=sample_id, timestep=ts)
            _require_finite("zt", zt, step=step, sample_id=sample_id, timestep=ts)

            # Pass A: get d(VPR + keep)/d(x0) without keeping the UNet graph.
            with torch.no_grad():
                eps0 = unet(zt, t, encoder_hidden_states=text,
                            cross_attention_kwargs={"concat_conds": cond}, return_dict=False)[0]
                _require_finite("eps0", eps0, step=step, sample_id=sample_id, timestep=ts)
                x0 = predict_x0(scheduler, zt, eps0, ts)
                _require_finite("x0", x0, step=step, sample_id=sample_id, timestep=ts)
            x0_leaf = x0.detach().float().requires_grad_(True)
            pred_img = decode_latent_01(x0_leaf, vae)
            _require_finite("pred_img", pred_img, step=step, sample_id=sample_id, timestep=ts)
            pred_desc = teacher(pred_img)
            _require_finite("pred_desc", pred_desc, step=step, sample_id=sample_id, timestep=ts)
            loss_vpr = (1 - (pred_desc * src_desc).sum(-1)).mean()
            loss_keep = F.l1_loss(pred_img, base_img)
            guide = args.lambda_vpr * loss_vpr + args.lambda_keep * loss_keep
            _require_finite("guide", guide, step=step, sample_id=sample_id, timestep=ts)
            grad_x0, = torch.autograd.grad(guide, x0_leaf)
            _require_finite("grad_x0", grad_x0, step=step, sample_id=sample_id, timestep=ts)
            grad_x0 = grad_x0.detach()
            del x0_leaf, pred_img, pred_desc, eps0, x0, guide

            # Pass B: transfer that first-order gradient into LoRA and preserve diffusion behavior.
            opt.zero_grad(set_to_none=True)
            eps = unet(zt, t, encoder_hidden_states=text,
                       cross_attention_kwargs={"concat_conds": cond}, return_dict=False)[0]
            _require_finite("eps", eps, step=step, sample_id=sample_id, timestep=ts)
            x0_train = predict_x0(scheduler, zt, eps, ts)
            _require_finite("x0_train", x0_train, step=step, sample_id=sample_id, timestep=ts)
            loss_diff = F.mse_loss(eps.float(), noise.float())
            loss_proxy = first_order_guidance_proxy(x0_train, grad_x0)
            loss = args.lambda_diff * loss_diff + loss_proxy
            _require_finite("loss_diff", loss_diff, step=step, sample_id=sample_id, timestep=ts)
            _require_finite("loss_proxy", loss_proxy, step=step, sample_id=sample_id, timestep=ts)
            _require_finite("loss", loss, step=step, sample_id=sample_id, timestep=ts)

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"non-finite LoRA gradient norm at step={step} sample_id={sample_id} timestep={ts}: "
                    f"grad_norm={float(grad_norm)}"
                )
            opt.step()

            # Catch optimizer-state/parameter corruption immediately instead of one step later.
            for i, p in enumerate(trainable):
                if not torch.isfinite(p).all():
                    raise FloatingPointError(
                        f"AdamW produced non-finite LoRA parameter after step={step}: param_index={i} "
                        f"dtype={p.dtype} lr={args.lr:g}"
                    )

            rec = {"step": step, "sample_id": row["sample_id"], "condition": row["condition"],
                   "timestep": ts, "loss_diff": float(loss_diff.detach()),
                   "loss_vpr": float(loss_vpr.detach()), "salad_cosine": float(1-loss_vpr.detach()),
                   "loss_keep": float(loss_keep.detach()), "loss_proxy": float(loss_proxy.detach()),
                   "grad_norm": float(grad_norm),
                   "guidance_x0_grad_norm": float(grad_x0.norm()),
                   "loss_total": float(args.lambda_diff * loss_diff.detach()
                                       + args.lambda_vpr * loss_vpr.detach()
                                       + args.lambda_keep * loss_keep.detach())}
            log.write(json.dumps(rec) + "\n"); log.flush()
            if writer is not None:
                write_tensorboard_record(writer, rec, run_config)
            if step == 1 or step % 10 == 0:
                print(
                    f"step={step} diff={rec['loss_diff']:.4f} cos={rec['salad_cosine']:.4f} "
                    f"keep={rec['loss_keep']:.4f} proxy={rec['loss_proxy']:.4g} "
                    f"grad={rec['grad_norm']:.4g}"
                )
            if step % args.save_every == 0 or step == args.max_steps:
                save_lora(unet, out / "checkpoints" / f"step_{step:06d}.pt",
                          rank=args.rank, alpha=args.alpha,
                          extra=training_state(step, run_config, opt, rng))


if __name__ == "__main__":
    main()
