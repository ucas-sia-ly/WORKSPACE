"""CCR-VPR Stage 1: synthetic queries against detached real references."""

import math

import torch
import torch.nn.functional as F


def asymmetric_multi_positive_infonce(queries, query_labels, references, reference_labels,
                                     temperature=0.1, pair_valid=None):
    """Mean -log(sum positive exp / sum all real exp), with cosine logits.

    All same-place real images are positives, including the exact source.
    Invalid/unmatched queries are excluded from the mean. Empty batches return
    a differentiable zero. Detaching references isolates this loss's DIRECT
    reference gradient; the synthetic branch still updates shared student weights.
    """
    if not math.isfinite(temperature) or temperature < torch.finfo(torch.float32).tiny:
        raise ValueError("temperature must be finite and positive in FP32")
    if queries.ndim != 2 or references.ndim != 2 or queries.shape[1] != references.shape[1]:
        raise ValueError("queries and references must have matching descriptor dimensions")
    if query_labels.shape != (len(queries),) or reference_labels.shape != (len(references),):
        raise ValueError("one label is required per descriptor")
    if pair_valid is not None and (pair_valid.shape != (len(queries),) or pair_valid.dtype != torch.bool):
        raise ValueError("pair_valid must be a boolean mask over synthetic queries")
    # Disabling autocast is essential: .float() alone would not prevent AMP
    # from casting matmul back to half precision.
    with torch.autocast(device_type=queries.device.type, enabled=False):
        q, r = queries.float(), references.detach().float()
        positives = query_labels[:, None] == reference_labels[None, :]
        valid = positives.any(dim=1)
        if pair_valid is not None:
            valid = valid & pair_valid
        q, positives = q[valid], positives[valid]
        if not len(q):
            return q.sum() * 0.0
        if not torch.isfinite(q).all() or not torch.isfinite(r).all():
            raise RuntimeError("Nonfinite descriptors in synthetic global loss")
        logits = (F.normalize(q, dim=1) @ F.normalize(r, dim=1).T) / temperature
        # Centering also reduces cancellation for low temperatures.
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        positive_logits = logits.masked_fill(~positives, -torch.inf)
        return (torch.logsumexp(logits, dim=1) - torch.logsumexp(positive_logits, dim=1)).mean()
