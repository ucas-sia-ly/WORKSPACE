"""LoRA for the IC-Light UNet attention projections.

The adapter (``adapters/iclight_sd15_fc.py``) loads checkpoints written here via
``ADAPTVPR_LORA_CHECKPOINT``: keys are ``<module name>.lora_A/.lora_B`` and the
safetensors metadata carries ``lora_rank``/``lora_alpha``.

Each LoRA layer is registered as a child (``lora_adapter``) of the Linear it
wraps, so it appears in ``unet.parameters()``/``state_dict()``. Its weights keep
their own dtype (fp32 for training) and the residual is computed in that dtype
and cast back, so an fp16 base model can train fp32 adapters.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import safetensors.torch as sf
import torch
import torch.nn as nn

DEFAULT_TARGET_MODULES = ("to_q", "to_k", "to_v", "to_out.0")
CHILD_NAME = "lora_adapter"


class LoRALayer(nn.Module):
    """Residual ``scaling * B(A(dropout(x)))`` with B initialised to zero."""

    def __init__(self, in_features: int, out_features: int, rank: int = 8,
                 alpha: float = 8.0, dropout: float = 0.0):
        super().__init__()
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(x.to(self.lora_A.dtype))
        return ((h @ self.lora_A.T) @ self.lora_B.T * self.scaling).to(x.dtype)


def _is_target(name: str, patterns: tuple[str, ...]) -> bool:
    return any(name == p or name.endswith("." + p) for p in patterns)


def inject_lora_into_unet(
    unet: nn.Module,
    rank: int = 8,
    alpha: float = 8.0,
    target_modules: list[str] | tuple[str, ...] | None = None,
    dropout: float = 0.0,
    dtype: torch.dtype | None = None,
    verbose: bool = False,
) -> dict[str, LoRALayer]:
    """Wrap matching ``nn.Linear`` modules with a LoRA residual.

    ``dtype`` defaults to the wrapped layer's dtype (inference); pass
    ``torch.float32`` to train adapters on a half-precision base model.
    """
    patterns = tuple(target_modules or DEFAULT_TARGET_MODULES)
    # Collect first: registering children while iterating named_modules is unsafe.
    targets = [(name, module) for name, module in unet.named_modules()
               if isinstance(module, nn.Linear) and _is_target(name, patterns)]
    already_wrapped = [name for name, module in targets if hasattr(module, CHILD_NAME)]
    if already_wrapped:
        raise RuntimeError(f"LoRA is already injected into {already_wrapped[0]}")
    lora_layers: dict[str, LoRALayer] = {}
    for name, module in targets:
        lora = LoRALayer(module.in_features, module.out_features, rank, alpha, dropout)
        lora.to(device=module.weight.device, dtype=dtype or module.weight.dtype)
        module.add_module(CHILD_NAME, lora)
        base_forward = module.forward

        def lora_forward(x, *args, _base=base_forward, _lora=lora, **kwargs):
            return _base(x, *args, **kwargs) + _lora(x)

        module.forward = lora_forward
        lora_layers[name] = lora
        if verbose:
            print(f"  LoRA -> {name} ({module.in_features}->{module.out_features}, r={rank})")
    if not lora_layers:
        raise ValueError(f"No nn.Linear modules matched LoRA targets {patterns}")
    return lora_layers


def get_lora_parameters(lora_layers: dict[str, LoRALayer]) -> list[nn.Parameter]:
    return [p for lora in lora_layers.values() for p in (lora.lora_A, lora.lora_B)]


def freeze_non_lora_parameters(unet: nn.Module) -> None:
    """Freeze every UNet parameter; call ``unfreeze_lora_parameters`` afterwards."""
    unet.requires_grad_(False)


def unfreeze_lora_parameters(lora_layers: dict[str, LoRALayer]) -> None:
    for parameter in get_lora_parameters(lora_layers):
        parameter.requires_grad_(True)


def save_lora_checkpoint(lora_layers: dict[str, LoRALayer], output_path: Path,
                         metadata: dict[str, Any] | None = None) -> None:
    if not lora_layers:
        raise ValueError("Cannot save an empty LoRA checkpoint")
    first = next(iter(lora_layers.values()))
    if any((layer.rank, layer.alpha) != (first.rank, first.alpha) for layer in lora_layers.values()):
        raise ValueError("All saved LoRA layers must have the same rank and alpha")
    state = {}
    for name, lora in lora_layers.items():
        state[f"{name}.lora_A"] = lora.lora_A.detach().float().cpu().contiguous()
        state[f"{name}.lora_B"] = lora.lora_B.detach().float().cpu().contiguous()
    if any(not torch.isfinite(weight).all() for weight in state.values()):
        raise ValueError("Cannot save non-finite LoRA weights")
    meta = {str(k): str(v) for k, v in (metadata or {}).items()}
    meta.update({"lora_rank": str(first.rank), "lora_alpha": str(first.alpha),
                 "num_layers": str(len(lora_layers))})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    sf.save_file(state, str(temporary), metadata=meta)
    temporary.replace(output_path)


def read_lora_metadata(checkpoint_path: Path) -> dict[str, str]:
    with sf.safe_open(str(checkpoint_path), framework="pt") as handle:
        metadata = dict(handle.metadata() or {})
    lora_config_from_metadata(metadata)
    return metadata


def lora_config_from_metadata(metadata: dict[str, Any]) -> tuple[int, float]:
    """Validate the scaling metadata before constructing or changing adapters."""
    try:
        rank, alpha = int(metadata["lora_rank"]), float(metadata["lora_alpha"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("LoRA metadata requires valid lora_rank and lora_alpha") from exc
    if str(rank) != str(metadata["lora_rank"]) or rank <= 0:
        raise ValueError("LoRA metadata rank must be a positive integer")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("LoRA metadata alpha must be finite and positive")
    return rank, alpha


def lora_state_dict(lora_layers: dict[str, LoRALayer]) -> dict[str, torch.Tensor]:
    return {f"{name}.{part}": getattr(layer, part).detach().float().cpu().clone()
            for name, layer in lora_layers.items() for part in ("lora_A", "lora_B")}


def load_lora_state_dict(lora_layers: dict[str, LoRALayer], state: dict[str, torch.Tensor],
                         metadata: dict[str, Any], strict: bool = True) -> None:
    """Validate the entire state before copying; a rejected load leaves weights intact."""
    expected = {f"{n}.{p}" for n in lora_layers for p in ("lora_A", "lora_B")}
    if strict and set(state) != expected:
        missing, unexpected = sorted(expected - set(state)), sorted(set(state) - expected)
        raise KeyError(f"LoRA keys differ: missing={missing[:5]} unexpected={unexpected[:5]}")
    rank, alpha = lora_config_from_metadata(metadata)
    if any((layer.rank, layer.alpha) != (rank, alpha) for layer in lora_layers.values()):
        raise ValueError("LoRA checkpoint rank/alpha differ from injected layers")
    if strict:
        try:
            count = int(metadata["num_layers"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("LoRA metadata requires valid num_layers") from exc
        if count != len(lora_layers):
            raise ValueError("LoRA metadata num_layers differs from injected layers")
    copies = []
    for name, layer in lora_layers.items():
        for part in ("lora_A", "lora_B"):
            key = f"{name}.{part}"
            if key not in state:
                continue
            target, value = getattr(layer, part), state[key]
            if value.shape != target.shape:
                raise ValueError(f"{key}: shape {tuple(value.shape)} != {tuple(target.shape)}")
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"{key}: invalid or non-finite LoRA weights")
            converted = value.to(device=target.device, dtype=target.dtype)
            if not torch.isfinite(converted).all():
                raise ValueError(f"{key}: weights overflow the adapter dtype {target.dtype}")
            copies.append((target, converted))
    with torch.no_grad():
        for target, value in copies:
            target.copy_(value)


def load_lora_checkpoint(lora_layers: dict[str, LoRALayer], checkpoint_path: Path,
                         strict: bool = True) -> dict[str, Any]:
    """Copy weights into injected layers; strict mode requires an exact key/shape match."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"LoRA checkpoint not found: {checkpoint_path}")
    state = sf.load_file(str(checkpoint_path))
    metadata = read_lora_metadata(checkpoint_path)
    load_lora_state_dict(lora_layers, state, metadata, strict=strict)
    return metadata


def report_trainable_parameters(unet: nn.Module, lora_layers: dict[str, LoRALayer]) -> dict[str, int]:
    """Assert that exactly the LoRA parameters are trainable and return counts."""
    lora_ids = {id(p) for p in get_lora_parameters(lora_layers)}
    trainable = [(n, p) for n, p in unet.named_parameters() if p.requires_grad]
    stray = [n for n, p in trainable if id(p) not in lora_ids]
    if stray:
        raise RuntimeError(f"Non-LoRA parameters are trainable: {stray[:5]}")
    if {id(p) for _, p in trainable} != lora_ids:
        raise RuntimeError("Some LoRA parameters are frozen or not registered in the UNet")
    counts = {
        "total": sum(p.numel() for p in unet.parameters()),
        "trainable": sum(p.numel() for _, p in trainable),
        "lora_layers": len(lora_layers),
    }
    print(f"[LoRA] trainable {counts['trainable']:,} / {counts['total']:,} "
          f"({100 * counts['trainable'] / counts['total']:.3f}%) in {counts['lora_layers']} layers")
    return counts
