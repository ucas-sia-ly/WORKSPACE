"""Conservative structural weak supervision for SALAD's reliability head.

Only training needs a same-view real companion. Relative local self-similarity
provides evidence about spatial consistency, rather than treating a weather
appearance change or low cross-image feature cosine as a generation error.
Ambiguous regions have zero supervision weight. The detached teacher shares
the existing DINO backbone; it is neither an extra model nor an inference input.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


_NEIGHBORS = tuple((dy * radius, dx * radius) for radius in (1, 2)
                   for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)


def _shift(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Read the offset neighbor, with zero outside the feature grid."""
    height, width = x.shape[-2:]
    padding = max(abs(dy), abs(dx))
    padded = F.pad(x, (padding, padding, padding, padding))
    return padded[..., padding + dy:padding + dy + height,
                  padding + dx:padding + dx + width]


def _structural_signature(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Centering makes a global feature offset, positive scale, or orthogonal
    # channel rotation leave these cosines unchanged. This is a useful limited
    # invariance, not a claim that every weather edit leaves DINO unchanged.
    centered = features - features.mean(dim=(-2, -1), keepdim=True)
    energy = centered.square().sum(dim=1, keepdim=True).sqrt()
    normalized = F.normalize(centered, dim=1, eps=1e-6)
    signature = torch.cat([(normalized * _shift(normalized, dy, dx)).sum(1, keepdim=True)
                           for dy, dx in _NEIGHBORS], dim=1)
    valid = (energy > 1e-6) & (signature.std(dim=1, keepdim=True, correction=0) > 0.02)
    # An incomplete neighborhood cannot establish structural evidence.
    valid[..., :2, :] = False
    valid[..., -2:, :] = False
    valid[..., :, :2] = False
    valid[..., :, -2:] = False
    return signature, valid


def _local_correspondence(query: torch.Tensor, reference: torch.Tensor,
                          query_valid: torch.Tensor, reference_valid: torch.Tensor,
                          radius: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, channels, height, width = query.shape
    side = 2 * radius + 1
    windows = F.unfold(reference, kernel_size=side, padding=radius)
    windows = windows.reshape(batch, channels, side * side, height * width)
    distances = (query.flatten(2).unsqueeze(2) - windows).square().mean(1).sqrt()
    valid = F.unfold(reference_valid.float(), kernel_size=side, padding=radius).bool()
    distances = distances.masked_fill(~valid | ~query_valid.flatten(2), torch.inf)
    best_two, offset = distances.topk(2, dim=1, largest=False)
    distance, margin = best_two[:, 0], best_two[:, 1] - best_two[:, 0]
    # inf - inf occurs in unknown regions; they must not leak NaNs to losses.
    margin = torch.nan_to_num(margin, nan=0.0, posinf=1.0)
    y, x = torch.meshgrid(torch.arange(height, device=query.device),
                          torch.arange(width, device=query.device), indexing="ij")
    dy = offset[:, 0] // side - radius
    dx = offset[:, 0] % side - radius
    indices = ((y.flatten()[None] + dy) * width + x.flatten()[None] + dx)
    return indices.clamp(0, height * width - 1), distance, margin


@torch.no_grad()
def build_local_targets(source_features: torch.Tensor, generated_features: torch.Tensor,
                        *, search_radius: int = 3, alignment_tolerance: int = 1
                        ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return soft reliability targets, confidence, and counts for paired grids.

    Positive anchors have unique reciprocal structural matches within the
    coordinate tolerance. Negative anchors require a farther reciprocal match,
    worse structure at the expected coordinate, and at least three neighbors
    agreeing on the displacement. Low feature cosine alone never creates a
    negative. Novel/unmatched hallucinations can remain unknown; these labels
    are evidence of consistency/contradiction, not exhaustive error detection.
    Thresholds are initial conservative heuristics and are not calibrated on a
    VPR evaluation benchmark. A one-patch registration difference is tolerated.
    """
    if source_features.shape != generated_features.shape or source_features.ndim != 4:
        raise ValueError("source and generated features must have identical [B,C,H,W] shapes")
    if source_features.device != generated_features.device:
        raise ValueError("paired feature grids must be on the same device")
    if isinstance(search_radius, bool) or not isinstance(search_radius, int) or search_radius < 1:
        raise ValueError("search_radius must be an integer >= 1")
    if not isinstance(alignment_tolerance, int) or not 0 <= alignment_tolerance < search_radius:
        raise ValueError("alignment_tolerance must be an integer in [0, search_radius)")
    if any(size < 1 for size in source_features.shape):
        raise ValueError("feature dimensions must be non-empty")
    with torch.autocast(device_type=source_features.device.type, enabled=False):
        source, generated = source_features.detach().float(), generated_features.detach().float()
        if not torch.isfinite(source).all() or not torch.isfinite(generated).all():
            raise ValueError("paired features must be finite")
        source_signature, source_valid = _structural_signature(source)
        generated_signature, generated_valid = _structural_signature(generated)
        generated_to_source, distance, margin = _local_correspondence(
            generated_signature, source_signature, generated_valid, source_valid, search_radius)
        source_to_generated, _, _ = _local_correspondence(
            source_signature, generated_signature, source_valid, generated_valid, search_radius)
        batch, _, height, width = source.shape
        identity = torch.arange(height * width, device=source.device)[None].expand(batch, -1)
        reciprocal = source_to_generated.gather(1, generated_to_source) == identity
        dy = generated_to_source // width - identity // width
        dx = generated_to_source % width - identity % width
        displacement = torch.maximum(dy.abs(), dx.abs())
        identity_distance = (source_signature - generated_signature).square().mean(1).sqrt().flatten(1)
        evidence = reciprocal & (distance < 0.08) & (margin > 0.025)
        near_identity = (displacement <= alignment_tolerance) & (identity_distance < 0.08)
        positive = evidence & near_identity
        # Neighbors must independently support the same non-trivial movement.
        dy, dx = dy.reshape(batch, 1, height, width), dx.reshape(batch, 1, height, width)
        evidence_grid = evidence.reshape(batch, 1, height, width)
        agreeing_neighbors = torch.zeros_like(dy)
        for offset_y, offset_x in _NEIGHBORS[:8]:
            agree = ((_shift(dy, offset_y, offset_x) == dy)
                     & (_shift(dx, offset_y, offset_x) == dx)
                     & _shift(evidence_grid, offset_y, offset_x))
            agreeing_neighbors += agree
        negative = (evidence & (displacement > alignment_tolerance)
                    & (identity_distance - distance > 0.12)
                    & (agreeing_neighbors.flatten(1) >= 3))
        targets = torch.full_like(distance, 0.5)
        targets[positive], targets[negative] = 0.95, 0.05
        confidence = ((1 - distance / 0.08).clamp(0, 1)
                      * (margin / 0.05).clamp(0, 1))
        confidence = torch.where(positive | negative, confidence, torch.zeros_like(confidence))
        targets = targets.reshape(batch, 1, height, width)
        confidence = confidence.reshape_as(targets)
        supervised = positive | negative
        diagnostics = {
            "positive_patches": positive.sum().detach(),
            "negative_patches": negative.sum().detach(),
            "unknown_patches": (~supervised).sum().detach(),
            "supervised_fraction": supervised.float().mean().detach(),
        }
        return targets, confidence, diagnostics


def reliability_losses(logits: torch.Tensor, features: torch.Tensor,
                       pair_features: torch.Tensor | None, paired_mask: torch.Tensor,
                       is_synthetic: torch.Tensor, *, auxiliary_weight: float = 0.1,
                       real_prior_weight: float = 0.01, coverage_weight: float = 0.1,
                       coverage_floor: float = 0.5,
                       cached_targets: torch.Tensor | None = None,
                       cached_confidence: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Masked weak BCE + real-view prior + a per-image reliability floor.

    ``pair_features`` is either compact in ``paired_mask`` order or a full
    aligned batch. Teacher targets never backpropagate into either feature
    tensor. A real prior discourages treating all ordinary real patches as
    unreliable; the mean-r floor also covers synthetic/unsupervised views. No
    negative quota is imposed: a completely sound generated image is allowed.
    Optional cached grids come from an offline frozen teacher and supersede
    live correspondence. They never require source features during training.
    """
    if (logits.ndim != 4 or logits.shape[1] != 1 or features.ndim != 4
            or features.shape[0] != logits.shape[0] or features.shape[2:] != logits.shape[2:]):
        raise ValueError("expected logits [B,1,H,W] and features [B,C,H,W] with aligned grids")
    if any(size < 1 for size in logits.shape) or features.shape[1] < 1:
        raise ValueError("feature and logit dimensions must be non-empty")
    batch = logits.shape[0]
    if paired_mask.shape != (batch,) or is_synthetic.shape != (batch,):
        raise ValueError("paired_mask and is_synthetic must be flat [B] masks")
    if paired_mask.dtype != torch.bool or is_synthetic.dtype != torch.bool:
        raise ValueError("paired_mask and is_synthetic must be boolean")
    if (logits.device != features.device or logits.device != paired_mask.device
            or logits.device != is_synthetic.device):
        raise ValueError("logits, features, and masks must share a device")
    if (paired_mask & ~is_synthetic).any():
        raise ValueError("only synthetic views may carry a source companion")
    if (cached_targets is None) != (cached_confidence is None):
        raise ValueError("cached targets and confidence must be provided together")
    if cached_targets is not None and pair_features is not None:
        raise ValueError("use either cached targets or live companion features")
    weights = (auxiliary_weight, real_prior_weight, coverage_weight)
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("loss weights must be finite and non-negative")
    if not math.isfinite(coverage_floor) or not 0 <= coverage_floor <= 1:
        raise ValueError("coverage_floor must be in [0,1]")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits = logits.float()
        if not torch.isfinite(logits).all():
            raise ValueError("reliability logits must be finite")
        zero = logits.sum() * 0.0
        supervision = zero
        diagnostics = {key: zero.detach() for key in (
            "positive_patches", "negative_patches", "unknown_patches", "supervised_fraction")}
        paired_count = int(paired_mask.sum())
        if paired_count:
            if cached_targets is not None:
                if (cached_targets.shape != cached_confidence.shape or cached_targets.ndim != 4
                        or cached_targets.shape[1:] != logits.shape[1:]
                        or cached_targets.shape[0] not in (batch, paired_count)
                        or cached_targets.device != logits.device or cached_confidence.device != logits.device):
                    raise ValueError("cached grids must be aligned full-batch or compact patch targets on the same device")
                targets, confidence = cached_targets.detach().float(), cached_confidence.detach().float()
                if targets.shape[0] == batch:
                    targets, confidence = targets[paired_mask], confidence[paired_mask]
                if (not torch.isfinite(targets).all() or not torch.isfinite(confidence).all()
                        or (targets < 0).any() or (targets > 1).any()
                        or (confidence < 0).any() or (confidence > 1).any()
                        or ((targets == 0.5) & (confidence > 0)).any()):
                    raise ValueError("cached targets/confidence must be finite probabilities with unknowns masked")
                positive = (targets > 0.5) & (confidence > 0)
                negative = (targets < 0.5) & (confidence > 0)
                diagnostics = {"positive_patches": positive.sum().detach(),
                               "negative_patches": negative.sum().detach(),
                               "unknown_patches": (~(positive | negative)).sum().detach(),
                               "supervised_fraction": (positive | negative).float().mean().detach()}
            else:
                if pair_features is None:
                    raise ValueError("paired views require source companion features")
                if pair_features.ndim != 4 or pair_features.shape[1:] != features.shape[1:]:
                    raise ValueError("source companion feature dimensions must match generated features")
                if pair_features.shape[0] == batch:
                    pair_features = pair_features[paired_mask]
                elif pair_features.shape[0] != paired_count:
                    raise ValueError("source companions must be compact in paired_mask order or full aligned batch")
                targets, confidence, diagnostics = build_local_targets(pair_features, features[paired_mask])
            pixel_loss = F.binary_cross_entropy_with_logits(logits[paired_mask], targets, reduction="none")
            class_terms, class_present = [], []
            for class_mask in (targets > 0.5, targets < 0.5):
                weight = confidence * class_mask
                mass = weight.sum()
                class_terms.append((pixel_loss * weight).sum() / mass.clamp_min(1e-6))
                class_present.append((mass > 0).float())
            supervision = torch.stack(class_terms).sum() / torch.stack(class_present).sum().clamp_min(1)
        real_logits = logits[~is_synthetic]
        real_prior = (F.binary_cross_entropy_with_logits(real_logits, torch.full_like(real_logits, 0.95))
                      if real_logits.numel() else zero)
        reliability = logits.sigmoid()
        image_means = reliability.flatten(1).mean(1)
        coverage = F.relu(coverage_floor - image_means).square().mean()
        total = (auxiliary_weight * supervision + real_prior_weight * real_prior
                 + coverage_weight * coverage)
        return {"total": total, "supervision": supervision, "real_prior": real_prior,
                "coverage": coverage, "mean_reliability": image_means.mean().detach(),
                "below_floor_fraction": (image_means < coverage_floor).float().mean().detach(),
                **diagnostics}
