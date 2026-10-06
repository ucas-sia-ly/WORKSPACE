"""Opt-in real-checkpoint bilevel smoke using existing verifier-approved GSV targets.

Performs one meta-only LoRA step in memory, without publishing a trained model
or changing any source/accepted files. Requires cached DINOv2/IC-Light weights.
"""
import argparse
import json
from pathlib import Path
import random

import torch
from diffusers import DDIMScheduler

from .bilevel import GSVRealIndex, construct_episode
from .iclight import attach_lora, load_iclight
from .salad_factory import add_meta_args, build_fresh_salad, meta_identity, validate_meta_args
from .teacher import model_sha256
from .train_generator import train_bilevel_step, report_trainable_parameters


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--accepted-records', type=Path, nargs='+', required=True)
    p.add_argument('--gsv-root', type=Path, required=True)
    p.add_argument('--salad-root', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    add_meta_args(p)
    args = p.parse_args()
    validate_meta_args(args)
    args.lambda_diff = args.lambda_keep = 0.
    args.grad_clip = 1.
    if args.lambda_meta != 1.:
        raise ValueError('this smoke requires --lambda-meta 1 for the meta-only check')
    selected, seen = [], set()
    for path in args.accepted_records:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get('passed') is True and row.get('eligible_for_training') is True:
                key = (row['city'], row['place_id'])
                if key not in seen and len(selected) < args.meta_places:
                    seen.add(key)
                    selected.append(row)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    real_index = GSVRealIndex(args.gsv_root, sorted({r['city'] for r in selected}))
    episode = construct_episode(selected, [], real_index, rng, places=args.meta_places,
                 support_real_per_place=args.meta_support_real_per_place,
                 query_real_per_place=args.meta_query_real_per_place)
    meta = build_fresh_salad(args.salad_root, meta=True, seed=args.seed, device='cuda',
                             train_backbone_blocks=args.meta_train_backbone_blocks)
    identity = meta_identity(meta, args.salad_root)
    t2i, i2i, vae = load_iclight()
    unet = t2i.unet
    vae.requires_grad_(False).eval()
    t2i.text_encoder.requires_grad_(False).eval()
    trainable = attach_lora(unet)
    report_trainable_parameters(unet, trainable, {'VAE': vae, 'text encoder': t2i.text_encoder})
    unet.enable_gradient_checkpointing()
    unet.train()
    scheduler = DDIMScheduler.from_config(t2i.scheduler.config)
    scheduler.set_timesteps(25, device='cuda')
    opt = torch.optim.AdamW(trainable, lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    result = train_bilevel_step(episode, step=1, args=args, t2i=t2i, vae=vae, unet=unet,
                meta_salad=meta, scheduler=scheduler,
                timesteps=[int(t) for t in scheduler.timesteps[-10:]],
                opt=opt, trainable=trainable, rng=rng)
    assert result['generator/meta_only_lora_grad_norm'] > 0, result
    assert result['generator/lora_grad_norm'] > 0, result
    assert not any(p.grad is not None for p in meta.parameters())
    assert model_sha256(meta) == identity['salad_initialization_sha256']
    result['peak_cuda_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
    result['meta_train_backbone_blocks'] = args.meta_train_backbone_blocks
    result['salad_initialization_sha256'] = identity['salad_initialization_sha256']
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
