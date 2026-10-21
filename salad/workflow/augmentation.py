"""Training augmentation: the original SALAD GSV-Cities RandAugment recipe.

``dataloaders/GSVCitiesDataloader.py`` trains with Resize -> RandAugment(num_ops=3,
bilinear) -> ToTensor -> Normalize and no flip. Images arrive here already resized.
Each call is keyed by an explicit integer seed and runs inside a forked Torch CPU
RNG, so augmentation never consumes the model's Dropout RNG and does not depend on
DataLoader worker count. Every image passed in one call receives the identical
operation/magnitude draw, which keeps paired source/generated views aligned.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

RANDAUGMENT = {"num_ops": 3, "magnitude": 9, "interpolation": "bilinear",
               "order": "after_resize_before_normalize", "flip": False,
               "rng": "sha256_keyed_forked_torch_cpu_v1"}

_TRANSFORM = None


def augmentation_seed(*parts: Any) -> int:
    payload = json.dumps(["randaugment_v1", *parts], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2 ** 63)


def _transform():
    global _TRANSFORM
    if _TRANSFORM is None:
        from torchvision import transforms as T
        _TRANSFORM = T.RandAugment(num_ops=RANDAUGMENT["num_ops"], magnitude=RANDAUGMENT["magnitude"],
                                   interpolation=T.InterpolationMode.BILINEAR)
    return _TRANSFORM


def randaugment(images, seed: int):
    """Return RandAugment views of PIL ``images``, all sharing one keyed draw."""
    import torch

    transform = _transform()
    result = []
    with torch.random.fork_rng(devices=[]):
        for image in images:
            torch.manual_seed(seed)
            result.append(transform(image))
    return result
