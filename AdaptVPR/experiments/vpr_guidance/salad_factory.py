"""Shared fresh SALAD construction and pure metric-learning utilities.

Downstream training keeps the official frozen-prefix forward. Meta training
freezes *parameters*, never pixels: the official forward's detach must be
removed even when adapting only the aggregator.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from .teacher import (_enable_image_gradients, file_sha256, model_sha256,
                      pil_tensor, salad_hub_refs)

OBJECTIVE_VERSION = "bilevel_salad_v1"
META_FIELDS = ("meta_inner_lr", "meta_inner_steps", "meta_places",
               "meta_support_real_per_place", "meta_query_real_per_place",
               "meta_image_size", "meta_train_backbone_blocks", "lambda_meta", "generator_grad_scale")
SALAD_CODE_FILES = ("vpr_model.py", "models/backbones/dinov2.py",
                    "models/aggregators/salad.py", "utils/losses.py")


def fresh_salad_config(max_steps=4000, train_backbone_blocks=4):
    return dict(
        backbone_arch="dinov2_vitb14",
        backbone_config={"num_trainable_blocks": train_backbone_blocks,
                         "return_token": True, "norm_layer": True},
        agg_arch="SALAD",
        agg_config={"num_channels": 768, "num_clusters": 64,
                    "cluster_dim": 128, "token_dim": 256},
        lr=6e-5, optimizer="adamw", weight_decay=9.5e-9,
        lr_sched="linear",
        lr_sched_args={"start_factor": 1.0, "end_factor": 0.2, "total_iters": max_steps},
        loss_name="MultiSimilarityLoss", miner_name="MultiSimilarityMiner", miner_margin=0.1,
    )


def official_vpr_class(salad_root):
    root = Path(salad_root).resolve()
    if not (root / "vpr_model.py").is_file():
        raise FileNotFoundError(f"official SALAD checkout not found: {root}")
    sys.path.insert(0, str(root))
    from vpr_model import VPRModel
    if Path(sys.modules[VPRModel.__module__].__file__).resolve().parent != root:
        raise RuntimeError("another vpr_model module shadowed the requested SALAD checkout")
    return VPRModel


def freeze_backbone_prefix(model):
    """Match downstream's last blocks plus norm, including aggregator-only meta."""
    backbone = model.backbone
    dino = backbone.model
    n = backbone.num_trainable_blocks
    if not 0 <= n <= min(4, len(dino.blocks)):
        raise ValueError("trainable DINOv2 blocks must be in 0..4")
    dino.requires_grad_(False)
    if n:
        for block in dino.blocks[-n:]:
            block.requires_grad_(True)
        if backbone.norm_layer:
            dino.norm.requires_grad_(True)


def _eager_dino_attention(self, x, attn_bias=None, is_causal=False):
    """Explicit matmul/softmax supports double backward unlike fused attention."""
    if attn_bias is not None or is_causal:
        raise ValueError("meta DINO attention requires dense noncausal image tokens")
    b, n, c = x.shape
    qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
    q, k, v = (t.transpose(1, 2) for t in qkv.unbind(2))
    weights = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
    drop = self.attn_drop
    if isinstance(drop, torch.nn.Module):
        weights = drop(weights)
    elif drop and self.training:
        weights = F.dropout(weights, p=drop)
    out = (weights @ v).transpose(1, 2).reshape(b, n, c)
    return self.proj_drop(self.proj(out))


def build_fresh_salad(salad_root=None, *, model_class=None, max_steps=4000,
                      train_backbone_blocks=4, meta=False, seed=None, device=None):
    if not 0 <= train_backbone_blocks <= 4:
        raise ValueError("trainable DINOv2 blocks must be in 0..4")
    cls = model_class or official_vpr_class(salad_root)

    def construct():
        with salad_hub_refs():
            return cls(**fresh_salad_config(max_steps, train_backbone_blocks))

    if seed is None:
        model = construct()
    else:
        # Recreate the same fixed theta0 on resume without consuming run RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model = construct()
    if meta:
        _enable_image_gradients(model)  # removes upstream no_grad AND detach
        model.aggregator.requires_grad_(True)
        for block in model.backbone.model.blocks:
            attn = getattr(block, "attn", None)
            if attn is None or not all(hasattr(attn, k) for k in
                                       ("qkv", "num_heads", "scale", "attn_drop", "proj", "proj_drop")):
                raise RuntimeError("unsupported DINO attention; cannot ensure second-order derivatives")
            attn.forward = types.MethodType(_eager_dino_attention, attn)
        # Eval disables dropout/stochastic depth; parameter adaptation still runs
        # with gradients. Deterministic before/after losses are comparable.
        model.eval()
    freeze_backbone_prefix(model)
    if device is not None:
        model = model.to(device=device, dtype=torch.float32)
    return model


def salad_code_hashes(root):
    return {name: file_sha256(Path(root) / name) for name in SALAD_CODE_FILES}


def meta_identity(model, root):
    return {"salad_initialization_sha256": model_sha256(model),
            "salad_initialization_policy": "pretrained_dinov2_random_salad_no_teacher_weights",
            "salad_code_sha256": salad_code_hashes(root),
            "meta_trainable_names": [n for n, p in model.named_parameters() if p.requires_grad],
            "meta_attention": "eager_matmul_softmax", "meta_model_mode": "eval"}


def make_metric_learning():
    """Same configuration as official utils/losses.py, usable without Lightning."""
    from pytorch_metric_learning import losses, miners
    from pytorch_metric_learning.distances import DotProductSimilarity, CosineSimilarity
    return (losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0,
                                      distance=DotProductSimilarity()),
            miners.MultiSimilarityMiner(epsilon=0.1, distance=CosineSimilarity()))


def metric_loss(descriptors, labels, loss_fn, miner):
    if descriptors.dtype != torch.float32 or not torch.isfinite(descriptors).all():
        raise FloatingPointError("SALAD descriptors must be finite fp32")
    pairs = miner(descriptors, labels)
    loss = loss_fn(descriptors, labels, pairs)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite SALAD metric loss")
    return loss, {"positive_pairs": pairs[0].numel(), "negative_pairs": pairs[2].numel()}


def validate_meta_args(args):
    if getattr(args, "lambda_vpr", None) not in (None, 0):
        raise ValueError("--lambda-vpr is retired; use --lambda-meta explicitly")
    for name, minimum in (("meta_inner_steps", 1), ("meta_places", 2),
                          ("meta_support_real_per_place", 1), ("meta_query_real_per_place", 2)):
        if getattr(args, name) < minimum:
            raise ValueError(f"--{name.replace('_', '-')} must be >= {minimum}")
    if not 0 <= args.meta_train_backbone_blocks <= 4:
        raise ValueError("--meta-train-backbone-blocks must be in 0..4")
    validate_image_size(args.meta_image_size)
    if not torch.isfinite(torch.tensor(args.meta_inner_lr)) or args.meta_inner_lr <= 0:
        raise ValueError("--meta-inner-lr must be finite and positive")
    if not torch.isfinite(torch.tensor(args.generator_grad_scale)) or args.generator_grad_scale < 1:
        raise ValueError("--generator-grad-scale must be finite and >=1")
    if not torch.isfinite(torch.tensor(args.lambda_meta)) or args.lambda_meta < 0:
        raise ValueError("--lambda-meta must be finite and nonnegative")


def add_meta_args(parser):
    parser.add_argument("--lambda-meta", type=float, default=1.)
    parser.add_argument("--lambda-vpr", type=float, default=None,
                        help="retired: nonzero values are rejected; use --lambda-meta")
    parser.add_argument("--generator-grad-scale", type=float, default=65536.,
                        help="fixed outer backward scale to prevent fp16 VAE/UNet hypergradient underflow")
    parser.add_argument("--meta-inner-lr", type=float, default=1e-3)
    for name, default in (("meta-inner-steps", 1), ("meta-places", 4),
                          ("meta-support-real-per-place", 1), ("meta-query-real-per-place", 2),
                          ("meta-image-size", 224), ("meta-train-backbone-blocks", 0)):
        parser.add_argument("--" + name, type=int, default=default)


def validate_image_size(size):
    if size < 126 or size % 14:
        raise ValueError("SALAD image size must be divisible by 14 with more than 64 patch tokens")


def preprocess_tensor(images, image_size=224):
    validate_image_size(image_size)
    images = F.interpolate(images.float(), size=(image_size, image_size),
                           mode="bilinear", align_corners=False, antialias=True)
    mean = images.new_tensor([.485, .456, .406])[None, :, None, None]
    std = images.new_tensor([.229, .224, .225])[None, :, None, None]
    return (images - mean) / std


def load_real_images(paths, image_size, device):
    result = []
    for path in paths:
        with Image.open(path) as image:
            result.append(preprocess_tensor(pil_tensor(image), image_size))
    return torch.cat(result).to(device=device, dtype=torch.float32)
