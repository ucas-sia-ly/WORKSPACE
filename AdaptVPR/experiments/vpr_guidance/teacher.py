"""Frozen SALAD teacher used only to supervise the domain generator."""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SALAD_REPO = "serizba/salad"
PREPROCESSING_VERSION = "rgb01-tensor-bilinear-antialias-322-imagenet-v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_sha256(model) -> str:
    """Identify the actual weights, including local checkpoint replacements."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}\n".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


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
        self._model_fingerprint = None

    @property
    def model_fingerprint(self) -> str:
        if self._model_fingerprint is None:
            self._model_fingerprint = model_sha256(self.model)
        return self._model_fingerprint

    @property
    def descriptor_dim(self) -> int:
        aggregator = self.model.aggregator
        return aggregator.num_clusters * aggregator.cluster_dim + aggregator.token_dim

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
        return self(pil_tensor(image))

    def _descriptor_metadata(self, source_path: Path) -> dict:
        source = Path(source_path).resolve(strict=True)
        return {
            "source_path": str(source),
            "source_sha256": file_sha256(source),
            "teacher_sha256": self.model_fingerprint,
            "preprocessing_version": PREPROCESSING_VERSION,
            "descriptor_dim": self.descriptor_dim,
        }

    def _validate_descriptor(self, descriptor: torch.Tensor) -> None:
        if not isinstance(descriptor, torch.Tensor) or descriptor.shape != (1, self.descriptor_dim):
            raise ValueError(f"source descriptor must have shape (1, {self.descriptor_dim})")
        if not torch.isfinite(descriptor).all():
            raise ValueError("source descriptor contains non-finite values")
        if not torch.allclose(descriptor.float().norm(dim=-1), descriptor.new_ones(1).float(),
                              rtol=1e-4, atol=1e-4):
            raise ValueError("source descriptor must be L2-normalized")

    def save_source_descriptor(self, path: Path, descriptor: torch.Tensor, source_path: Path) -> None:
        self._validate_descriptor(descriptor)
        torch.save({"descriptor": descriptor.detach().cpu(),
                    "metadata": self._descriptor_metadata(source_path)}, Path(path))

    def load_source_descriptor(self, path: Path, source_path: Path) -> torch.Tensor:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or set(payload) != {"descriptor", "metadata"}:
            raise ValueError(f"descriptor cache has no validated metadata: {path}; rerun prepare_data")
        expected = self._descriptor_metadata(source_path)
        if payload["metadata"] != expected:
            raise ValueError(f"descriptor cache metadata mismatch: {path}; rerun prepare_data")
        descriptor = payload["descriptor"]
        self._validate_descriptor(descriptor)
        return descriptor.to(device=self.device, dtype=torch.float32)


@contextmanager
def salad_hub_refs():
    """Reuse explicit main caches without unqualified Torch Hub branch discovery."""
    hub_load = torch.hub.load

    def load_with_refs(repo_or_dir, model, *args, **kwargs):
        if kwargs.get("source", "github") == "github" and repo_or_dir in (
                SALAD_REPO, "facebookresearch/dinov2"):
            repo_or_dir += ":main"
        return hub_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = load_with_refs
    try:
        yield
    finally:
        torch.hub.load = hub_load


def load_salad(device: str = "cuda", repo: str = SALAD_REPO) -> SaladTeacher:
    """Load the frozen pretrained teacher from local SALAD or its Hub cache."""
    source = "local" if Path(repo).is_dir() else "github"
    with salad_hub_refs():
        model = torch.hub.load(repo, "dinov2_salad", pretrained=True, trust_repo=True, source=source)
    return SaladTeacher(model, device=device)
