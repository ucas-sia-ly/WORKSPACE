"""Known geometric damage for reliability training, independent of weather.

These copies are an auxiliary task on real views. They never replace metric
images, create generated-image labels, or use an additional feature model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def make_local_corruptions(images: torch.Tensor, is_synthetic: torch.Tensor, *,
                           max_images: int = 4, patch_size: int = 14,
                           generator: torch.Generator | None = None
                           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Copy a patch-aligned rectangle from a disjoint distant region.

    Only randomly selected real views are returned, at most ``max_images``.
    Source and destination occupy opposite vertical halves; the destination
    mask covers at most a quarter of the patch grid, and only patches whose
    copied pixels actually differ are labeled. Pixel values are copied
    unchanged, so this task does not label darkening or weather as an error.
    Uses the optional device-matched generator (otherwise the caller's Torch
    RNG) and never modifies the original tensors.
    """
    if images.ndim != 4 or not images.is_floating_point():
        raise ValueError("images must be a floating-point [B,C,H,W] tensor")
    if (is_synthetic.shape != (images.shape[0],) or is_synthetic.dtype != torch.bool
            or is_synthetic.device != images.device):
        raise ValueError("is_synthetic must be a boolean [B] mask on the image device")
    if isinstance(max_images, bool) or not isinstance(max_images, int) or max_images < 0:
        raise ValueError("max_images must be a nonnegative integer")
    if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size < 1:
        raise ValueError("patch_size must be a positive integer")
    height, width = images.shape[-2:]
    if height % patch_size or width % patch_size:
        raise ValueError("image dimensions must be multiples of patch_size")
    grid_h, grid_w = height // patch_size, width // patch_size
    if grid_h < 4 or grid_w < 4:
        raise ValueError("local corruptions require at least a 4x4 patch grid")
    candidates = (~is_synthetic).nonzero(as_tuple=False).flatten()
    count = min(max_images, candidates.numel())
    if count == 0:
        return (images.new_empty((0, *images.shape[1:])),
                torch.zeros((0, 1, grid_h, grid_w), dtype=torch.bool, device=images.device),
                candidates[:0])
    selected = candidates[torch.randperm(candidates.numel(), device=images.device,
                                        generator=generator)[:count]]
    corrupted = images[selected].detach().clone()
    mask = torch.zeros((count, 1, grid_h, grid_w), dtype=torch.bool, device=images.device)

    def draw(low: int, high: int) -> int:
        return int(torch.randint(low, high + 1, (), device=images.device, generator=generator))

    for index in range(count):
        rectangle_h, rectangle_w = draw(2, grid_h // 2), draw(2, grid_w // 2)
        source_y = draw(0, grid_h // 2 - rectangle_h)
        target_y = draw(grid_h // 2, grid_h - rectangle_h)
        if draw(0, 1):
            source_y, target_y = target_y, source_y
        source_x, target_x = draw(0, grid_w - rectangle_w), draw(0, grid_w - rectangle_w)
        sy, sx = source_y * patch_size, source_x * patch_size
        ty, tx = target_y * patch_size, target_x * patch_size
        rh, rw = rectangle_h * patch_size, rectangle_w * patch_size
        # Read from the original selected view, avoiding in-place overlap and
        # keeping every unmasked pixel bitwise equal to the input.
        copied = images[selected[index], :, sy:sy + rh, sx:sx + rw].detach()
        original_destination = images[selected[index], :, ty:ty + rh, tx:tx + rw].detach()
        changed_patches = (copied != original_destination).reshape(
            images.shape[1], rectangle_h, patch_size, rectangle_w, patch_size,
        ).any(dim=(0, 2, 4))
        corrupted[index, :, ty:ty + rh, tx:tx + rw] = copied
        mask[index, 0, target_y:target_y + rectangle_h,
             target_x:target_x + rectangle_w] = changed_patches
    return corrupted, mask, selected


def corrupted_patch_loss(logits: torch.Tensor, mask: torch.Tensor, *,
                         reference_logits: torch.Tensor | None = None
                         ) -> dict[str, torch.Tensor]:
    """Soft negative BCE with optional detached-clean consistency outside.

    Without a reference, the unmasked logits have exactly zero gradient. With
    real-view clean logits, a small outside consistency term prevents lowering
    the entire image's reliability to satisfy the local negatives. It does not
    label any original generated patch or propagate into the clean reference.
    """
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("logits must have shape [B,1,H,W]")
    if mask.shape != logits.shape or mask.dtype != torch.bool or mask.device != logits.device:
        raise ValueError("mask must be boolean with the logits shape and device")
    if (reference_logits is not None and (reference_logits.shape != logits.shape
                                          or reference_logits.device != logits.device)):
        raise ValueError("reference_logits must have the logits shape and device")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits = logits.float()
        if not torch.isfinite(logits).all():
            raise ValueError("corruption logits must be finite")
        selected_logits = logits[mask]
        zero = logits.sum() * 0.0
        negative_loss = (F.binary_cross_entropy_with_logits(
            selected_logits, torch.full_like(selected_logits, 0.05)
        ) if selected_logits.numel() else zero)
        mean_reliability = (selected_logits.sigmoid().mean().detach()
                            if selected_logits.numel() else zero.detach())
        outside_logits = logits[~mask]
        outside_mean = (outside_logits.sigmoid().mean().detach()
                        if outside_logits.numel() else zero.detach())
        consistency, reference_outside_mean = zero, zero.detach()
        if reference_logits is not None:
            reference = reference_logits.detach().float()
            if not torch.isfinite(reference).all():
                raise ValueError("reference corruption logits must be finite")
            if outside_logits.numel():
                reference_outside = reference[~mask].sigmoid()
                consistency = F.smooth_l1_loss(outside_logits.sigmoid(), reference_outside)
                reference_outside_mean = reference_outside.mean().detach()
        return {"corruption_loss": negative_loss + 0.1 * consistency,
                "corruption_negative_loss": negative_loss,
                "corruption_consistency": consistency,
                "corruption_negative_patches": mask.sum().detach(),
                "corruption_mean_reliability": mean_reliability,
                "corruption_outside_mean_reliability": outside_mean,
                "corruption_reference_outside_mean_reliability": reference_outside_mean}
