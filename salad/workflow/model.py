"""DINOv2 + SALAD with legacy-compatible parameter names and portable checkpoints."""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path

import torch
from torch import nn

from models.aggregators.salad import SALAD


BACKBONE_CHANNELS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


def default_model_config(backbone="dinov2_vitb14", image_size=(224, 224),
                         num_trainable_blocks=4, num_clusters=64,
                         cluster_dim=128, token_dim=256):
    if backbone not in BACKBONE_CHANNELS:
        raise ValueError(f"Unsupported backbone: {backbone}")
    config = {
        "backbone_arch": backbone,
        "backbone_config": {"num_trainable_blocks": num_trainable_blocks,
                            "norm_layer": True, "return_token": True},
        "agg_arch": "SALAD",
        "agg_config": {"num_channels": BACKBONE_CHANNELS[backbone],
                       "num_clusters": num_clusters, "cluster_dim": cluster_dim,
                       "token_dim": token_dim},
        "image_size": list(image_size),
    }
    validate_model_config(config)
    return config


def validate_model_config(config):
    backbone = config["backbone_arch"]
    if backbone not in BACKBONE_CHANNELS or config.get("agg_arch", "SALAD").lower() != "salad":
        raise ValueError("These entry points support DINOv2 + SALAD only")
    height, width = config.get("image_size", (224, 224))
    clusters = config["agg_config"]["num_clusters"]
    if height <= 0 or width <= 0 or height % 14 or width % 14:
        raise ValueError("Image height and width must be positive multiples of 14")
    if clusters <= 0 or (height // 14) * (width // 14) <= clusters:
        raise ValueError("SALAD requires more image patch tokens than clusters")
    if config["agg_config"]["num_channels"] != BACKBONE_CHANNELS[backbone]:
        raise ValueError("SALAD num_channels does not match the DINOv2 backbone")
    if config["backbone_config"].get("num_trainable_blocks", 4) < 0:
        raise ValueError("num_trainable_blocks cannot be negative")
    if any(config["agg_config"][key] <= 0 for key in ("cluster_dim", "token_dim")):
        raise ValueError("SALAD feature dimensions must be positive")
    aggregator = config["agg_config"]
    if type(aggregator.get("reliability_ot", False)) is not bool:
        raise ValueError("reliability_ot must be a bool")
    if aggregator.get("reliability_ot", False):
        strength = aggregator.get("reliability_lambda", 2.0)
        hidden = aggregator.get("reliability_hidden_dim", 64)
        if type(strength) not in (int, float) or not math.isfinite(strength) or strength < 0:
            raise ValueError("reliability_lambda must be finite and nonnegative")
        if type(hidden) is not int or hidden < 1:
            raise ValueError("reliability_hidden_dim must be a positive integer")
        for name in ("reliability_context", "reliability_detach_features"):
            if type(aggregator.get(name, False)) is not bool:
                raise ValueError(f"{name} must be a bool")
        if aggregator.get("reliability_mode", "learned") not in ("learned", "fixed"):
            raise ValueError("reliability_mode must be learned or fixed")


def read_checkpoint(path: Path):
    """Load tensor/primitive-only checkpoints without executing pickled objects."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


class DINOBackbone(nn.Module):
    def __init__(self, architecture, config, pretrained=True, repo=None, weights=None):
        super().__init__()
        if repo is not None:
            repo = Path(repo).expanduser().resolve()
            if not (repo / "hubconf.py").is_file():
                raise ValueError(f"--backbone-repo must contain hubconf.py: {repo}")
        self.model = torch.hub.load(
            str(repo) if repo else "facebookresearch/dinov2", architecture,
            source="local" if repo else "github",
            pretrained=pretrained and weights is None,
        )
        if weights is not None:
            state = read_checkpoint(weights)
            state = state.get("state_dict", state.get("model", state))
            self.model.load_state_dict(state, strict=True)
        self.num_channels = BACKBONE_CHANNELS[architecture]
        self.num_trainable_blocks = config.get("num_trainable_blocks", 4)
        self.norm_layer = config.get("norm_layer", True)
        if self.num_trainable_blocks > len(self.model.blocks):
            raise ValueError("num_trainable_blocks exceeds the backbone block count")
        self.model.requires_grad_(False)
        if self.num_trainable_blocks:
            for block in self.model.blocks[-self.num_trainable_blocks:]:
                block.requires_grad_(True)
            if self.norm_layer:
                self.model.norm.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        if self.num_trainable_blocks:
            for block in self.model.blocks[-self.num_trainable_blocks:]:
                block.train(mode)
            self.model.norm.train(mode)
        return self

    def forward(self, images):
        batch, _, height, width = images.shape
        split = len(self.model.blocks) - self.num_trainable_blocks
        with torch.no_grad():
            tokens = self.model.prepare_tokens_with_masks(images)
            for block in self.model.blocks[:split]:
                tokens = block(tokens)
        for block in self.model.blocks[split:]:
            tokens = block(tokens)
        if self.norm_layer:
            tokens = self.model.norm(tokens)
        features = tokens[:, 1:].reshape(batch, height // 14, width // 14, self.num_channels)
        return features.permute(0, 3, 1, 2), tokens[:, 0]


class SALADModel(nn.Module):
    def __init__(self, config, pretrained_backbone=True, backbone_repo=None, backbone_weights=None):
        super().__init__()
        validate_model_config(config)
        self.config = config
        self.image_size = tuple(config.get("image_size", (224, 224)))
        self.backbone = DINOBackbone(config["backbone_arch"], config["backbone_config"],
                                     pretrained_backbone, backbone_repo, backbone_weights)
        # No SALAD pretrained weights are loaded: each fresh run initializes this randomly.
        self.aggregator = SALAD(**config["agg_config"])

    def forward(self, images, return_aux=False):
        features = self.backbone(images)
        if return_aux:
            return self.aggregator(features, return_aux=True)
        return self.aggregator(features)


def checkpoint_state_and_config(checkpoint):
    """Accept our checkpoints, Lightning state_dicts, and official raw SALAD weights."""
    state = checkpoint.get("state_dict", checkpoint)
    config = copy.deepcopy(checkpoint.get("model_config"))
    infer_config = config is None
    if config is None:
        hparams = checkpoint.get("hyper_parameters", {})
        channels = state["aggregator.token_features.0.weight"].shape[1]
        architecture = next((name for name, dim in BACKBONE_CHANNELS.items() if dim == channels), None)
        if architecture is None:
            raise ValueError(f"Unrecognized checkpoint backbone width: {channels}")
        config = default_model_config(
            architecture,
            num_clusters=state["aggregator.score.3.weight"].shape[0],
            cluster_dim=state["aggregator.cluster_features.3.weight"].shape[0],
            token_dim=state["aggregator.token_features.2.weight"].shape[0],
        )
        config["backbone_config"].update(hparams.get("backbone_config", {}))
        config["backbone_config"]["return_token"] = True
    reliability_weights = {key for key in state if key.startswith("aggregator.reliability_head.")}
    enabled = config["agg_config"].get("reliability_ot", False)
    if reliability_weights:
        first = state.get("aggregator.reliability_head.0.weight")
        strength = state.get("aggregator.reliability_ot_lambda")
        if (not isinstance(first, torch.Tensor) or first.ndim != 4
                or not isinstance(strength, torch.Tensor) or strength.ndim != 0
                or not bool(torch.isfinite(strength)) or float(strength) < 0):
            raise ValueError("Reliability checkpoint is missing valid head/strength metadata")
        if infer_config:
            config["agg_config"].update(reliability_ot=True, reliability_hidden_dim=first.shape[0],
                                         reliability_lambda=float(strength))
        elif (not enabled or config["agg_config"].get("reliability_hidden_dim", 64) != first.shape[0]
              or not math.isclose(config["agg_config"].get("reliability_lambda", 2.0),
                                  float(strength), rel_tol=1e-6, abs_tol=1e-7)):
            raise ValueError("Reliability weights and saved model configuration differ")
        context_keys = {key for key in state if key.startswith("aggregator.reliability_context.")}
        has_context = bool(context_keys)
        if has_context and context_keys != {"aggregator.reliability_context.weight", "aggregator.reliability_context.bias"}:
            raise ValueError("Partial reliability context weights cannot load a checkpoint")
        fixed = state.get("aggregator.reliability_fixed_mode")
        if fixed is not None and (not isinstance(fixed, torch.Tensor) or fixed.ndim != 0
                                  or fixed.dtype != torch.bool or not bool(fixed)):
            raise ValueError("reliability_fixed_mode must be a true scalar bool")
        mode = "fixed" if fixed is not None else "learned"
        if infer_config:
            if has_context:
                config["agg_config"]["reliability_context"] = True
            if mode == "fixed":
                config["agg_config"]["reliability_mode"] = mode
        elif (config["agg_config"].get("reliability_context", False) != has_context
              or config["agg_config"].get("reliability_mode", "learned") != mode):
            raise ValueError("Reliability context/mode and saved model configuration differ")
    elif enabled or "aggregator.reliability_ot_lambda" in state:
        raise ValueError("Reliability configuration/strength exists without head weights")
    validate_model_config(config)
    return state, config


def load_model_state(model, state, *, allow_new_reliability=False):
    """Strict loading, allowing only a fresh reliability head on legacy weights.

    Fresh optimizer initialization may add the new head. Resume and inference
    never hide missing weights or silently drop a learned reliability branch.
    """
    if allow_new_reliability and model.config["agg_config"].get("reliability_ot", False):
        expected = model.state_dict()
        new_keys = {key for key in expected if key.startswith("aggregator.reliability_head.")
                    or key.startswith("aggregator.reliability_context.")
                    or key in ("aggregator.reliability_ot_lambda", "aggregator.reliability_fixed_mode")}
        present = new_keys & set(state)
        if not present:
            if set(state) != set(expected) - new_keys:
                raise RuntimeError("Legacy initialization must contain every original SALAD weight exactly")
            state = {**state, **{key: expected[key] for key in new_keys}}
        elif present != new_keys:
            raise RuntimeError("Partial reliability weights cannot initialize a checkpoint")
    return model.load_state_dict(state, strict=True)


def load_checkpoint_model(checkpoint: Path, device: str, backbone_repo: Path | None = None,
                          backbone_weights: Path | None = None):
    if torch.device(device).type == "cpu":
        # DINOv2 reads this before importing its optional CUDA-only xFormers ops.
        os.environ["XFORMERS_DISABLED"] = "1"
    state, config = checkpoint_state_and_config(read_checkpoint(checkpoint))
    model = SALADModel(config, pretrained_backbone=False, backbone_repo=backbone_repo,
                       backbone_weights=backbone_weights)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def atomic_save_checkpoint(payload, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
