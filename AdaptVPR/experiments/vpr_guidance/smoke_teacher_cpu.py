"""Verify real pretrained VAE/SALAD input gradients on CPU, without generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from PIL import Image

from .iclight import decode_latent_01
from .teacher import load_salad, pil_tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salad-repo", default="serizba/salad")
    parser.add_argument("--vae-base-model", type=Path, required=True)
    parser.add_argument("--source-image", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    teacher = load_salad(device="cpu", repo=args.salad_repo)
    vae = AutoencoderKL.from_pretrained(args.vae_base_model, subfolder="vae",
                                       torch_dtype=torch.float32, local_files_only=True)
    vae.requires_grad_(False).eval()
    with Image.open(args.source_image) as opened:
        source = opened.convert("RGB")
    with torch.no_grad():
        source_descriptor = teacher.from_pil(source)
        backbone = teacher.model.backbone
        official_features = backbone.__class__.forward(backbone, teacher.preprocess(pil_tensor(source)))
        official_descriptor = F.normalize(teacher.model.aggregator(official_features).float(), dim=-1)
        torch.testing.assert_close(source_descriptor, official_descriptor, rtol=1e-6, atol=1e-7)
        official_forward_max_error = float((source_descriptor - official_descriptor).abs().max())

    generated = torch.rand(1, 3, 48, 80, requires_grad=True)
    descriptor = teacher(generated)
    pixel_loss = (1 - (descriptor * source_descriptor).sum(-1)).mean()
    pixel_loss.backward()
    assert generated.grad is not None and torch.isfinite(generated.grad).all()
    assert generated.grad.norm() > 0
    pixel_grad_norm = float(generated.grad.norm())
    del descriptor, pixel_loss, generated

    predicted_x0 = torch.randn(1, 4, 8, 8, requires_grad=True)
    image = decode_latent_01(predicted_x0, vae)
    descriptor = teacher(image)
    latent_loss = (1 - (descriptor * source_descriptor).sum(-1)).mean()
    latent_loss.backward()
    assert predicted_x0.grad is not None and torch.isfinite(predicted_x0.grad).all()
    assert predicted_x0.grad.norm() > 0
    assert not any(p.requires_grad or p.grad is not None for p in teacher.model.parameters())
    assert not any(p.requires_grad or p.grad is not None for p in vae.parameters())
    print(json.dumps({
        "status": "PASS", "device": "cpu", "vae_dtype": str(vae.dtype),
        "teacher_dtype": str(next(teacher.model.parameters()).dtype),
        "descriptor_shape": list(descriptor.shape),
        "descriptor_norm": float(descriptor.detach().norm()),
        "official_forward_max_descriptor_error": official_forward_max_error,
        "decoded_shape": list(image.shape), "decoded_range": [float(image.min().detach()), float(image.max().detach())],
        "pixel_gradient_norm": pixel_grad_norm,
        "predicted_x0_gradient_norm": float(predicted_x0.grad.norm()),
        "teacher_trainable_parameters": 0, "vae_trainable_parameters": 0,
        "scope": "real pretrained VAE/SALAD CPU gradient paths; full GPU generation is unverified",
    }, indent=2))


if __name__ == "__main__":
    main()
