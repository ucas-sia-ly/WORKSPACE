"""Exact MS saturation certificates and real/synthetic directed-pair counts.

No gradients are calculated here. An empty mining mask certifies zero gradient
for this repository's mined MS loss at these descriptors; a rounded zero loss
with nonempty masks does not certify a zero gradient.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from workflow.diagnostic_gradients import mined_pair_statistics
from workflow.metric_loss import multi_similarity_loss


@torch.no_grad()
def source_aware_mined_pair_statistics(descriptors, labels, is_synthetic, epsilon=0.1,
                                      *, include_per_anchor=True, loss_value=None):
    if is_synthetic.shape != labels.shape or is_synthetic.dtype != torch.bool:
        raise ValueError("is_synthetic must be a bool tensor with the labels' shape")
    if is_synthetic.device != labels.device:
        raise ValueError("Synthetic flags and labels must share a device")
    if include_per_anchor:
        result = mined_pair_statistics(descriptors, labels, epsilon)
    else:
        if (descriptors.ndim != 2 or not len(descriptors) or not descriptors.shape[1]
                or labels.shape != (len(descriptors),) or labels.device != descriptors.device):
            raise ValueError("Expected nonempty same-device [B,D] descriptors and [B] labels")
        if not math.isfinite(epsilon) or epsilon < 0 or not torch.isfinite(descriptors).all():
            raise ValueError("Descriptors must be finite and mining epsilon nonnegative")
        result = {"batch_size": len(labels), "places": labels.unique().numel(),
                  "epsilon": float(epsilon), "pair_counting": "directed",
                  "loss": float(multi_similarity_loss(descriptors, labels, epsilon=epsilon)
                                if loss_value is None else loss_value)}
    with torch.autocast(device_type=descriptors.device.type, enabled=False):
        normalized = F.normalize(descriptors.float(), dim=1)
        similarities = normalized @ normalized.T
        same = labels[:, None] == labels[None, :]
        positive = same & ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        negative = ~same
        hardest_negative = similarities.masked_fill(~negative, -torch.inf).max(1).values
        hardest_positive = similarities.masked_fill(~positive, torch.inf).min(1).values
        mined_positive = positive & (similarities < hardest_negative[:, None] + epsilon)
        mined_negative = negative & (similarities > hardest_positive[:, None] - epsilon)
    active = mined_positive.any(1) | mined_negative.any(1)
    # Pack scalar counts into one device transfer: no per-anchor GPU .item().
    scalars = [positive.sum(), negative.sum(), mined_positive.sum(), mined_negative.sum(),
               active.sum(), (mined_positive.any(1) & mined_negative.any(1)).sum(),
               mined_positive.any(1).sum(), mined_negative.any(1).sum()]
    names = ("positive_pairs", "negative_pairs", "mined_positive_pairs", "mined_negative_pairs",
             "active_anchors", "both_active_anchors", "positive_active_anchors", "negative_active_anchors")
    pair_keys = []
    for anchor_synthetic, anchor_name in ((False, "real"), (True, "synthetic")):
        for reference_synthetic, reference_name in ((False, "real"), (True, "synthetic")):
            domain = ((is_synthetic == anchor_synthetic)[:, None]
                      & (is_synthetic == reference_synthetic)[None, :])
            for kind, mask in (("positive", positive), ("negative", negative),
                               ("mined_positive", mined_positive), ("mined_negative", mined_negative)):
                pair_keys.append((anchor_name + "_" + reference_name, kind))
                scalars.append((mask & domain).sum())
    for flag in (False, True):
        scalars.extend(((is_synthetic == flag).sum(), (active & (is_synthetic == flag)).sum()))
    counts = torch.stack(scalars).cpu().tolist()
    totals = dict(zip(names, counts[:8]))
    if not include_per_anchor:
        result.update({key: totals[key] for key in names[:4]})
        result.update(positive_retention_ratio=totals["mined_positive_pairs"] / max(totals["positive_pairs"], 1),
            negative_retention_ratio=totals["mined_negative_pairs"] / max(totals["negative_pairs"], 1),
            active_anchor_ratio=totals["active_anchors"] / len(labels),
            both_sides_active_anchor_ratio=totals["both_active_anchors"] / len(labels),
            positive_active_anchor_ratio=totals["positive_active_anchors"] / len(labels),
            negative_active_anchor_ratio=totals["negative_active_anchors"] / len(labels))
    pairs = {source: {} for source, _ in pair_keys}
    for (source, kind), value in zip(pair_keys, counts[8:24]):
        pairs[source][kind] = value
    result["source_pairs"] = pairs
    result["source_anchors"] = {
        name: {"anchors": counts[24 + index * 2], "active_anchors": counts[25 + index * 2]}
        for index, name in enumerate(("real", "synthetic"))}
    empty = totals["mined_positive_pairs"] + totals["mined_negative_pairs"] == 0
    result["empty_mining_mask"] = empty
    result["zero_ms_gradient_certified_by_empty_mask"] = empty
    result["numerically_zero_loss_with_nonempty_mask"] = result["loss"] == 0 and not empty
    return result


def common_anchor_statistics(statistics, count):
    """Same first anchors, with a progressively larger negative candidate pool."""
    anchors = statistics["per_anchor"][:count]
    if len(anchors) != count or count < 1:
        raise ValueError("Common-anchor count must be positive and within the batch")
    return {"anchors": count, "active_anchors": sum(a["ms_active"] for a in anchors),
            "active_anchor_ratio": sum(a["ms_active"] for a in anchors) / count,
            "mined_positive_pairs": sum(a["mined_positive"] for a in anchors),
            "mined_negative_pairs": sum(a["mined_negative"] for a in anchors)}
