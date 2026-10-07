"""LoRA utilities for IC-Light fine-tuning.

Handles LoRA injection into the UNet, checkpoint saving/loading,
and parameter management.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import safetensors.torch as sf
import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer.

    Implements W' = W + BA where B and A are low-rank matrices.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Initialize A with Kaiming uniform, B with zeros (as per LoRA paper)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA: x @ A^T @ B^T * scaling."""
        return self.dropout(x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def inject_lora_into_unet(
    unet: nn.Module,
    rank: int = 8,
    alpha: float = 8.0,
    target_modules: list[str] | None = None,
    dropout: float = 0.0,
) -> dict[str, LoRALayer]:
    """Inject LoRA layers into UNet attention modules.

    Args:
        unet: The UNet model to inject LoRA into
        rank: LoRA rank
        alpha: LoRA alpha scaling factor
        target_modules: List of module name patterns to inject LoRA into.
                       Default: ["to_q", "to_k", "to_v", "to_out.0"]
        dropout: Dropout rate for LoRA layers

    Returns:
        Dictionary mapping layer names to LoRA modules
    """
    if target_modules is None:
        target_modules = ["to_q", "to_k", "to_v", "to_out.0"]

    lora_layers = {}

    for name, module in unet.named_modules():
        # Check if this module should get LoRA
        should_inject = any(pattern in name for pattern in target_modules)
        if not should_inject:
            continue

        # Only inject into Linear layers
        if not isinstance(module, nn.Linear):
            continue

        in_features = module.in_features
        out_features = module.out_features

        # Create LoRA layer
        lora = LoRALayer(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

        # Move to same device/dtype as the original layer
        lora.to(device=module.weight.device, dtype=module.weight.dtype)

        # Inject by wrapping the forward method
        original_forward = module.forward

        def make_lora_forward(orig_forward, lora_layer):
            def lora_forward(x):
                return orig_forward(x) + lora_layer(x)
            return lora_forward

        module.forward = make_lora_forward(original_forward, lora)

        # Store reference
        lora_layers[name] = lora
        print(f"  Injected LoRA into {name} (in={in_features}, out={out_features}, rank={rank})")

    return lora_layers


def get_lora_parameters(lora_layers: dict[str, LoRALayer]) -> list[nn.Parameter]:
    """Extract all trainable parameters from LoRA layers."""
    params = []
    for lora in lora_layers.values():
        params.extend([lora.lora_A, lora.lora_B])
    return params


def freeze_non_lora_parameters(unet: nn.Module) -> None:
    """Freeze all non-LoRA parameters in the UNet."""
    for param in unet.parameters():
        param.requires_grad_(False)


def unfreeze_lora_parameters(lora_layers: dict[str, LoRALayer]) -> None:
    """Ensure all LoRA parameters are trainable."""
    for lora in lora_layers.values():
        lora.lora_A.requires_grad_(True)
        lora.lora_B.requires_grad_(True)


def save_lora_checkpoint(
    lora_layers: dict[str, LoRALayer],
    output_path: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save LoRA weights to safetensors format.

    Args:
        lora_layers: Dictionary of LoRA layers
        output_path: Path to save checkpoint
        metadata: Optional metadata to include
    """
    state_dict = {}
    for name, lora in lora_layers.items():
        state_dict[f"{name}.lora_A"] = lora.lora_A.detach().cpu()
        state_dict[f"{name}.lora_B"] = lora.lora_B.detach().cpu()

    # Add metadata
    if metadata is None:
        metadata = {}
    metadata.update({
        "lora_rank": str(lora_layers[list(lora_layers.keys())[0]].rank),
        "lora_alpha": str(lora_layers[list(lora_layers.keys())[0]].alpha),
        "num_layers": str(len(lora_layers)),
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.save_file(state_dict, str(output_path), metadata=metadata)
    print(f"Saved LoRA checkpoint to {output_path}")


def load_lora_checkpoint(
    lora_layers: dict[str, LoRALayer],
    checkpoint_path: Path,
    strict: bool = True,
) -> dict[str, Any]:
    """Load LoRA weights from checkpoint.

    Args:
        lora_layers: Dictionary of LoRA layers to load into
        checkpoint_path: Path to checkpoint
        strict: Whether to require exact key matching

    Returns:
        Metadata from checkpoint
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = sf.load_file(str(checkpoint_path))
    metadata = {}

    # Extract metadata if available
    with sf.safe_open(str(checkpoint_path), framework="pt") as f:
        metadata = f.metadata() or {}

    # Load weights
    for name, lora in lora_layers.items():
        key_A = f"{name}.lora_A"
        key_B = f"{name}.lora_B"

        if key_A not in state_dict or key_B not in state_dict:
            if strict:
                raise KeyError(f"Missing keys for {name}: {key_A}, {key_B}")
            else:
                print(f"Warning: Skipping {name} (keys not found)")
                continue

        lora.lora_A.data.copy_(state_dict[key_A])
        lora.lora_B.data.copy_(state_dict[key_B])

    print(f"Loaded LoRA checkpoint from {checkpoint_path}")
    return metadata


def report_trainable_parameters(unet: nn.Module, lora_layers: dict[str, LoRALayer]) -> None:
    """Print report of trainable vs total parameters."""
    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    lora_params = sum(p.numel() for p in get_lora_parameters(lora_layers))

    print("\n" + "="*60)
    print("Parameter Report:")
    print(f"  Total UNet parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  LoRA parameters: {lora_params:,}")
    print(f"  Percentage trainable: {100 * trainable_params / total_params:.3f}%")
    print(f"  LoRA layers: {len(lora_layers)}")
    print("="*60 + "\n")

    # Verify all trainable params are LoRA params
    lora_param_ids = {id(p) for p in get_lora_parameters(lora_layers)}
    trainable_param_ids = {id(p) for p in unet.parameters() if p.requires_grad}

    non_lora_trainable = trainable_param_ids - lora_param_ids
    if non_lora_trainable:
        print(f"WARNING: Found {len(non_lora_trainable)} trainable non-LoRA parameters!")
        for name, param in unet.named_parameters():
            if param.requires_grad and id(param) not in lora_param_ids:
                print(f"  - {name}")
    else:
        print("✓ All trainable parameters are LoRA parameters")


def save_training_state(
    lora_layers: dict[str, LoRALayer],
    optimizer: torch.optim.Optimizer,
    step: int,
    output_dir: Path,
    extra_state: dict[str, Any] | None = None,
) -> None:
    """Save full training state (LoRA weights + optimizer + step).

    Args:
        lora_layers: LoRA layers
        optimizer: Optimizer
        step: Current training step
        output_dir: Directory to save state
        extra_state: Additional state to save
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA weights
    lora_path = output_dir / f"lora_step_{step}.safetensors"
    save_lora_checkpoint(
        lora_layers,
        lora_path,
        metadata={"training_step": str(step)},
    )

    # Save optimizer state
    optimizer_path = output_dir / f"optimizer_step_{step}.pt"
    torch.save(optimizer.state_dict(), optimizer_path)

    # Save training metadata
    metadata = {
        "step": step,
        "lora_checkpoint": str(lora_path.name),
        "optimizer_checkpoint": str(optimizer_path.name),
    }
    if extra_state:
        metadata.update(extra_state)

    metadata_path = output_dir / f"training_state_{step}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"Saved training state at step {step}")


def load_training_state(
    lora_layers: dict[str, LoRALayer],
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: Path,
    step: int | None = None,
) -> dict[str, Any]:
    """Load full training state.

    Args:
        lora_layers: LoRA layers to load into
        optimizer: Optimizer to load into
        checkpoint_dir: Directory containing checkpoints
        step: Specific step to load (None = latest)

    Returns:
        Training metadata
    """
    # Find checkpoint
    if step is None:
        # Find latest checkpoint
        metadata_files = sorted(checkpoint_dir.glob("training_state_*.json"))
        if not metadata_files:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        metadata_path = metadata_files[-1]
    else:
        metadata_path = checkpoint_dir / f"training_state_{step}.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Checkpoint metadata not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text())

    # Load LoRA weights
    lora_path = checkpoint_dir / metadata["lora_checkpoint"]
    load_lora_checkpoint(lora_layers, lora_path)

    # Load optimizer state
    optimizer_path = checkpoint_dir / metadata["optimizer_checkpoint"]
    optimizer.load_state_dict(torch.load(optimizer_path))

    print(f"Loaded training state from step {metadata['step']}")
    return metadata
