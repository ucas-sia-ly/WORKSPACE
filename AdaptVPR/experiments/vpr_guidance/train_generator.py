"""Legacy offline entry point; use train_online_generator or train_full.

The formal online trainer uses train_bilevel_step. Teacher-VJP utilities below
are isolated legacy regression helpers only.
"""
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
from .salad_factory import (add_meta_args, validate_meta_args, preprocess_tensor,
                            load_real_images, OBJECTIVE_VERSION, META_FIELDS)
from .bilevel import bilevel_objective


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=8)
    p.add_argument("--lambda-diff", type=float, default=1.0)
    add_meta_args(p)
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
    validate_meta_args(args)
    weights = (args.lambda_diff, args.lambda_meta, args.lambda_keep)
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
        "objective_version": OBJECTIVE_VERSION,
        "manifest_sha256": file_sha256(args.manifest),
        "teacher_sha256": teacher.model_fingerprint,
        **{name: getattr(args, name) for name in
           ("rank", "alpha", "lr", "lambda_diff", "lambda_keep",
            "timestep_window", "grad_clip", "seed", *META_FIELDS)},
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


def train_accepted_step(row, *, step, args, t2i, vae, unet, teacher, scheduler,
                        timesteps, opt, trainable, rng, source_descriptor=None):
    """LEGACY teacher-only utility; never called by formal online training."""
    if row.get("passed") is not True or row.get("eligible_for_training") is not True:
        raise ValueError("rejected candidates cannot be used by any generator loss")
    sample_id = str(row["sample_id"])
    with Image.open(row["source_path"]) as image:
        source = image.convert("RGB")
    with Image.open(row["generated_path"]) as image:
        target = image.convert("RGB")
    width, height = target.width // 8 * 8, target.height // 8 * 8
    z0 = encode_image_latent(target, vae, width, height)
    from AdaptVPR.adapters import iclight_sd15_fc as adapter
    cond = adapter._concat_condition(conditioning_source(source), vae, width, height)
    text = encode_prompt(t2i, row["prompt"])
    src_desc = source_descriptor if source_descriptor is not None else teacher.load_source_descriptor(row["source_descriptor"], row["source_path"])
    base_img = image_tensor_01(target, width, height, "cuda")

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
    return rec


def main():
    raise RuntimeError(
        "The offline fixed-baseline trainer is deprecated. Use "
        "python -m AdaptVPR.experiments.vpr_guidance.train_online_generator or train_full. "
        "Old baseline targets must not enter the online training loop.")


def assert_lora_optimizer(unet, optimizer, trainable):
    expected = {id(p) for n, p in unet.named_parameters() if p.requires_grad and "lora_" in n}
    actual = [p for group in optimizer.param_groups for p in group["params"]]
    if (not expected or expected != {id(p) for p in trainable}
            or expected != {id(p) for p in actual} or len(actual) != len(expected)
            or any(p.requires_grad and "lora_" not in n for n, p in unet.named_parameters())):
        raise RuntimeError("generator optimizer must contain ONLY all trainable IC-Light LoRA parameters")


def differentiable_accepted_prediction(row, *, step, t2i, vae, unet, scheduler, timesteps, rng):
    """One noisy accepted-target UNet prediction; RGB never leaves autograd."""
    if (row.get("passed") is not True or row.get("eligible_for_training") is not True
            or row.get("route") != "global"):
        raise ValueError("rejected/non-Global candidates cannot be used by any generator loss")
    if file_sha256(row["generated_path"]) != row["generated_sha256"]:
        raise ValueError("verified generated target content changed")
    if file_sha256(row["source_path"]) != row["source_sha256"]:
        raise ValueError("verified source content changed")
    with Image.open(row["source_path"]) as image:
        source = image.convert("RGB")
    with Image.open(row["generated_path"]) as image:
        target = image.convert("RGB")
    width, height = target.width // 8 * 8, target.height // 8 * 8
    z0 = encode_image_latent(target, vae, width, height)
    from AdaptVPR.adapters import iclight_sd15_fc as adapter
    with torch.no_grad():
        cond = adapter._concat_condition(conditioning_source(source), vae, width, height)
        text = encode_prompt(t2i, row["prompt"])
    target_tensor = image_tensor_01(target, width, height, z0.device)
    ts = timesteps[rng.randrange(len(timesteps))]
    t = torch.tensor([ts], device=z0.device, dtype=torch.long)
    noise = torch.randn_like(z0)
    zt = scheduler.add_noise(z0, noise, t)
    eps = unet(zt, t, encoder_hidden_states=text,
               cross_attention_kwargs={"concat_conds": cond}, return_dict=False)[0]
    x0 = predict_x0(scheduler, zt, eps, ts)
    image = decode_latent_01(x0, vae)
    for name, value in (("z0", z0), ("zt", zt), ("eps", eps), ("x0", x0), ("pred_img", image)):
        _require_finite(name, value, step=step, sample_id=row["sample_id"], timestep=ts)
    return image, F.mse_loss(eps.float(), noise.float()), F.l1_loss(image, target_tensor), ts


def _gradient_norm(grads, device):
    squares = [g.detach().float().square().sum() for g in grads if g is not None]
    return torch.stack(squares).sum().sqrt() if squares else torch.zeros((), device=device)


def train_bilevel_step(episode, *, step, args, t2i, vae, unet, meta_salad, scheduler,
                       timesteps, opt, trainable, rng, teacher=None, source_descriptors=None):
    """Actual LoRA -> RGB -> inner SGD -> real-query hypergradient.

    autograd.grad only requests LoRA gradients for the outer optimizer: base
    SALAD needs requires_grad for inner adaptation but never accumulates .grad.
    Teacher/cache inputs are OPTIONAL diagnostics, outside the objective graph.
    """
    episode.validate()
    assert_lora_optimizer(unet, opt, trainable)
    opt.zero_grad(set_to_none=True)
    device = trainable[0].device
    support, queries, support_labels, query_labels = [], [], [], []
    diff, keep, timesteps_used, images = [], [], [], []
    for place in episode.places:
        image, loss_diff, loss_keep, ts = differentiable_accepted_prediction(
            place.synthetic_row, step=step, t2i=t2i, vae=vae, unet=unet,
            scheduler=scheduler, timesteps=timesteps, rng=rng)
        support.extend([load_real_images(place.support_real, args.meta_image_size, device),
                        preprocess_tensor(image, args.meta_image_size)])
        support_labels.extend([place.label] * (len(place.support_real) + 1))
        queries.append(load_real_images(place.query_real, args.meta_image_size, device))
        query_labels.extend([place.label] * len(place.query_real))
        diff.append(loss_diff)
        keep.append(loss_keep)
        timesteps_used.append(ts)
        images.append(image)
    result = bilevel_objective(meta_salad, torch.cat(support),
                torch.tensor(support_labels, device=device), torch.cat(queries),
                torch.tensor(query_labels, device=device), inner_lr=args.meta_inner_lr,
                inner_steps=args.meta_inner_steps)
    loss_diff, loss_keep = torch.stack(diff).mean(), torch.stack(keep).mean()
    loss_meta = result.outer_loss
    loss = args.lambda_meta * loss_meta + args.lambda_diff * loss_diff + args.lambda_keep * loss_keep
    _require_finite("generator total loss", loss, step=step, sample_id="meta_episode", timestep=timesteps_used[0])
    # Explicit META-ONLY audit so denoising cannot hide a severed hypergradient.
    # The frozen VAE/UNet still run in fp16. Tiny second-order pixel gradients
    # underflow there unless scaled BEFORE backward; fast SALAD arithmetic stays
    # fp32. A power-of-two fixed scale preserves the exact mathematical gradient.
    scale = args.generator_grad_scale
    meta_grads = torch.autograd.grad(loss_meta * scale, trainable, retain_graph=True, allow_unused=True)
    meta_grads = [g.float() / scale if g is not None else None for g in meta_grads]
    if all(g is None for g in meta_grads):
        raise RuntimeError("meta-only LoRA hypergradient is disconnected; check second-order inner update")
    meta_norm = _gradient_norm(meta_grads, device)
    if not torch.isfinite(meta_norm):
        raise FloatingPointError("non-finite meta-only LoRA hypergradient")
    del meta_grads
    grads = torch.autograd.grad(loss * scale, trainable, allow_unused=True)
    for parameter, grad in zip(trainable, grads):
        parameter.grad = grad.float() / scale if grad is not None else None
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError(f"non-finite bilevel LoRA gradient at step={step}")
    opt.step()
    if any(not torch.isfinite(p).all() for p in trainable):
        raise FloatingPointError(f"optimizer produced non-finite LoRA at step={step}")
    rec = {"step": step, "objective_version": OBJECTIVE_VERSION,
           "sample_ids": [p.synthetic_row["sample_id"] for p in episode.places],
           "place_keys": [list(p.key) for p in episode.places],
           "timesteps": timesteps_used, **result.metrics,
           "generator/loss_diff": float(loss_diff.detach()),
           "generator/loss_keep": float(loss_keep.detach()),
           "generator/loss_meta": float(loss_meta.detach()),
           "generator/loss_total": float(loss.detach()),
           "generator/lora_grad_norm": float(grad_norm),
           "generator/meta_only_lora_grad_norm": float(meta_norm),
           "generator/grad_scale": scale,
           "generator/timestep": sum(timesteps_used) / len(timesteps_used)}
    if teacher is not None:
        with torch.no_grad():
            cosines = []
            for place, image in zip(episode.places, images):
                row = place.synthetic_row
                src = (source_descriptors[row["sample_id"]].to(teacher.device)
                       if source_descriptors is not None else
                       teacher.load_source_descriptor(row["source_descriptor"], row["source_path"]))
                cosines.append((teacher(image) * src).sum(-1).mean())
            cosine = torch.stack(cosines).mean()
            if not torch.isfinite(cosine):
                raise FloatingPointError("non-finite diagnostic teacher cosine")
            rec["diagnostic/teacher_salad_cosine"] = float(cosine)
    return rec


if __name__ == "__main__":
    main()
