"""Frozen SALAD teacher used only to supervise the domain generator."""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SALAD_REPO = "serizba/salad"


def freeze(model):
    model.eval()
    model.requires_grad_(False)
    return model


def _enable_image_gradients(model):
    backbone = model.backbone
    required = ("model", "norm_layer", "return_token", "num_channels")
    if not all(hasattr(backbone, name) for name in required):
        raise RuntimeError("unsupported SALAD backbone; inspect upstream implementation")

    def forward(self, image):
        batch, _, height, width = image.shape
        tokens = self.model.prepare_tokens_with_masks(image)
        for block in self.model.blocks:
            tokens = block(tokens)
        if self.norm_layer:
            tokens = self.model.norm(tokens)
        token = tokens[:, 0]
        features = tokens[:, 1:].reshape(
            batch, height // 14, width // 14, self.num_channels
        ).permute(0, 3, 1, 2)
        return (features, token) if self.return_token else features

    backbone.forward = types.MethodType(forward, backbone)
    return freeze(model)


def pil_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB")).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().unsqueeze(0) / 255.0


class SaladTeacher:
    def __init__(self, model, device: str = "cuda"):
        self.model = _enable_image_gradients(model).to(device=device, dtype=torch.float32)
        self.device = device

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(image.float(), size=(322, 322), mode="bilinear",
                              align_corners=False, antialias=True)
        mean = image.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = image.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        return (image - mean) / std

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        descriptor = self.model(self.preprocess(image.to(self.device)))
        if isinstance(descriptor, (tuple, list)):
            descriptor = descriptor[0]
        return F.normalize(descriptor.float(), dim=-1)

    def from_pil(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize((322, 322), Image.Resampling.BILINEAR)
        return self(pil_tensor(image))


def load_salad(device: str = "cuda", repo: str = SALAD_REPO) -> SaladTeacher:
    source = "local" if Path(repo).is_dir() else "github"
    model = torch.hub.load(repo, "dinov2_salad", pretrained=True, trust_repo=True, source=source)
    return SaladTeacher(model, device=device)
