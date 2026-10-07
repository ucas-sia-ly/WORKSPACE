"""VPR-guided generator fine-tuning for AdaptVPR.

This package implements hard-case-driven LoRA fine-tuning for IC-Light,
using SALAD validation errors to guide generator improvement.
"""

__version__ = "0.1.0"

from .hard_cases import HardCase, load_hard_cases, filter_cases_with_source
from .losses import DiffusionLoss, IdentityLoss, DiversityLoss
from .lora_utils import (
    LoRALayer,
    inject_lora_into_unet,
    save_lora_checkpoint,
    load_lora_checkpoint,
    get_lora_parameters,
    freeze_non_lora_parameters,
    unfreeze_lora_parameters,
)

__all__ = [
    "HardCase",
    "load_hard_cases",
    "filter_cases_with_source",
    "DiffusionLoss",
    "IdentityLoss",
    "DiversityLoss",
    "LoRALayer",
    "inject_lora_into_unet",
    "save_lora_checkpoint",
    "load_lora_checkpoint",
    "get_lora_parameters",
    "freeze_non_lora_parameters",
    "unfreeze_lora_parameters",
]
