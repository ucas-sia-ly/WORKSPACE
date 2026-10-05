"""Train IC-Light LoRA with frozen SALAD feedback on Global-route data."""
from __future__ import annotations

import argparse, json, random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DDIMScheduler

from .iclight import attach_lora, decode_latent_01, encode_image_latent, encode_prompt, image_tensor_01, load_iclight, save_lora
from .teacher import load_salad


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
    return p.parse_args()


def read_manifest(path):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def predict_x0(scheduler, z_t, eps, timestep):
    alpha = scheduler.alphas_cumprod[int(timestep)].to(z_t.device, z_t.dtype)
    beta = 1.0 - alpha
    return (z_t - beta.sqrt() * eps) / alpha.sqrt().clamp_min(1e-6)


def main():
    args = args_parser()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed); torch.manual_seed(args.seed)
    rows = read_manifest(args.manifest)
    out = args.output_dir.resolve(); (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    t2i, _, vae = load_iclight()
    unet = t2i.unet
    vae.requires_grad_(False).eval(); t2i.text_encoder.requires_grad_(False).eval()
    trainable = attach_lora(unet, args.rank, args.alpha)
    unet.enable_gradient_checkpointing(); unet.train()
    teacher = load_salad(repo=args.salad_repo)

    scheduler = DDIMScheduler.from_config(t2i.scheduler.config)
    scheduler.set_timesteps(25, device="cuda")
    timesteps = [int(x) for x in scheduler.timesteps[-min(args.timestep_window, 25):]]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    rng = random.Random(args.seed)

    log = (out / "train.jsonl").open("w")
    for step in range(1, args.max_steps + 1):
        row = rows[rng.randrange(len(rows))]
        source = Image.open(row["source_path"]).convert("RGB")
        baseline = Image.open(row["baseline_path"]).convert("RGB")
        width, height = baseline.width // 8 * 8, baseline.height // 8 * 8
        z0 = encode_image_latent(baseline, vae, width, height)
        from AdaptVPR.adapters import iclight_sd15_fc as adapter
        cond = adapter._concat_condition(source, vae, width, height)
        text = encode_prompt(t2i, row["prompt"])
        src_desc = F.normalize(torch.load(row["source_descriptor"], map_location="cuda").float(), dim=-1)
        base_img = image_tensor_01(baseline, width, height, "cuda")

        ts = timesteps[rng.randrange(len(timesteps))]
        t = torch.tensor([ts], device="cuda", dtype=torch.long)
        noise = torch.randn_like(z0)
        zt = scheduler.add_noise(z0, noise, t)

        # Pass A: get d(VPR + keep)/d(x0) without keeping the UNet graph.
        with torch.no_grad():
            eps0 = unet(zt, t, encoder_hidden_states=text,
                        cross_attention_kwargs={"concat_conds": cond}, return_dict=False)[0]
            x0 = predict_x0(scheduler, zt, eps0, ts)
        x0_leaf = x0.detach().float().requires_grad_(True)
        pred_img = decode_latent_01(x0_leaf, vae)
        pred_desc = teacher(pred_img)
        loss_vpr = (1 - (pred_desc * src_desc).sum(-1)).mean()
        loss_keep = F.l1_loss(pred_img, base_img)
        guide = args.lambda_vpr * loss_vpr + args.lambda_keep * loss_keep
        grad_x0, = torch.autograd.grad(guide, x0_leaf)
        grad_x0 = grad_x0.detach()
        del x0_leaf, pred_img, pred_desc, eps0, x0, guide

        # Pass B: transfer that first-order gradient into LoRA and preserve diffusion behavior.
        opt.zero_grad(set_to_none=True)
        eps = unet(zt, t, encoder_hidden_states=text,
                   cross_attention_kwargs={"concat_conds": cond}, return_dict=False)[0]
        x0_train = predict_x0(scheduler, zt, eps, ts)
        loss_diff = F.mse_loss(eps.float(), noise.float())
        loss_proxy = (x0_train.float() * grad_x0).sum()
        loss = args.lambda_diff * loss_diff + loss_proxy
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite generator loss")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip))
        opt.step()

        rec = {"step": step, "sample_id": row["sample_id"], "condition": row["condition"],
               "timestep": ts, "loss_diff": float(loss_diff.detach()),
               "loss_vpr": float(loss_vpr.detach()), "salad_cosine": float(1-loss_vpr.detach()),
               "loss_keep": float(loss_keep.detach()), "grad_norm": grad_norm}
        log.write(json.dumps(rec) + "\n"); log.flush()
        if step == 1 or step % 10 == 0:
            print(f"step={step} diff={rec['loss_diff']:.4f} cos={rec['salad_cosine']:.4f} keep={rec['loss_keep']:.4f}")
        if step % args.save_every == 0 or step == args.max_steps:
            save_lora(unet, out / "checkpoints" / f"step_{step:06d}.pt",
                      rank=args.rank, alpha=args.alpha, extra={"step": step})
    log.close()


if __name__ == "__main__":
    main()
