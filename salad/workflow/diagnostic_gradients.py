"""Read-only MS mining and loss-gradient diagnostics for fixed training batches.

These helpers never call backward(), zero_grad(), clip_grad_norm_(), or an
optimizer. Gradients returned by autograd.grad are detached for CPU inspection;
the caller owns model modes, forward RNG isolation, and diagnostic inputs.
"""

from __future__ import annotations

import itertools
import math
import re

import torch
import torch.nn.functional as F

from workflow.metric_loss import multi_similarity_loss


def _distribution(values):
    values = torch.as_tensor(values, dtype=torch.float64).flatten()
    if not values.numel():
        return {"count": 0, "mean": None, "std": None, "min": None,
                "p05": None, "p25": None, "median": None, "p75": None,
                "p95": None, "max": None}
    if not torch.isfinite(values).all():
        raise ValueError("Diagnostic statistics contain nonfinite values")
    quantiles = torch.quantile(values, torch.tensor([.05, .25, .5, .75, .95], dtype=values.dtype))
    return {"count": values.numel(), "mean": values.mean().item(),
            "std": values.std(unbiased=False).item(), "min": values.min().item(),
            **dict(zip(("p05", "p25", "median", "p75", "p95"), quantiles.tolist())),
            "max": values.max().item()}


@torch.no_grad()
def mined_pair_statistics(descriptors, labels, epsilon=0.1):
    """Mirror metric_loss.py's directed cosine-mining masks exactly.

    Both per-anchor counts and their distributions are retained. Pair counts
    are directed (anchor, reference), so a symmetric positive pair counts twice.
    An MS-active anchor has at least one retained pair on either side; the
    stricter both-sides ratio is reported separately. No descriptor gradients
    are constructed by this diagnostic.
    """
    if (descriptors.ndim != 2 or not len(descriptors) or not descriptors.shape[1]
            or labels.shape != (len(descriptors),)):
        raise ValueError("Expected nonempty [B,D] descriptors and [B] labels")
    if descriptors.device != labels.device:
        raise ValueError("Descriptors and labels must share a device")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("Mining epsilon must be finite and nonnegative")
    if not torch.isfinite(descriptors).all():
        raise ValueError("Descriptor mining diagnostics require finite inputs")
    with torch.autocast(device_type=descriptors.device.type, enabled=False):
        normalized = F.normalize(descriptors.float(), dim=1)
        similarity = normalized @ normalized.T
        same = labels[:, None] == labels[None, :]
        positive = same & ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        negative = ~same
        hardest_negative = similarity.masked_fill(~negative, -torch.inf).max(1).values
        hardest_positive = similarity.masked_fill(~positive, torch.inf).min(1).values
        mined_positive = positive & (similarity < hardest_negative[:, None] + epsilon)
        mined_negative = negative & (similarity > hardest_positive[:, None] - epsilon)
        p, n = positive.sum(1), negative.sum(1)
        mp, mn = mined_positive.sum(1), mined_negative.sum(1)
        loss = multi_similarity_loss(descriptors, labels, epsilon=epsilon).item()
    active = (mp > 0) | (mn > 0)
    both = (mp > 0) & (mn > 0)
    result = {"batch_size": len(labels), "places": labels.unique().numel(),
              "epsilon": float(epsilon), "loss": loss, "pair_counting": "directed",
              "positive_pairs": p.sum().item(), "negative_pairs": n.sum().item(),
              "mined_positive_pairs": mp.sum().item(), "mined_negative_pairs": mn.sum().item(),
              "positive_retention_ratio": mp.sum().item() / max(p.sum().item(), 1),
              "negative_retention_ratio": mn.sum().item() / max(n.sum().item(), 1),
              "active_anchor_ratio": active.float().mean().item(),
              "both_sides_active_anchor_ratio": both.float().mean().item(),
              "positive_active_anchor_ratio": (mp > 0).float().mean().item(),
              "negative_active_anchor_ratio": (mn > 0).float().mean().item(),
              "per_anchor": []}
    for key, values in (("positive", p), ("negative", n),
                        ("mined_positive", mp), ("mined_negative", mn)):
        result[key + "_per_anchor"] = _distribution(values.cpu())
    for index in range(len(labels)):
        result["per_anchor"].append({
            "anchor_index": index, "label": labels[index].item(),
            "positive": p[index].item(), "negative": n[index].item(),
            "mined_positive": mp[index].item(), "mined_negative": mn[index].item(),
            "hardest_positive_similarity": (hardest_positive[index].item() if p[index] else None),
            "hardest_negative_similarity": (hardest_negative[index].item() if n[index] else None),
            "ms_active": bool(active[index]), "both_sides_active": bool(both[index])})
    return result


def _parameters(named_parameters):
    named_parameters = list(named_parameters)
    names = [name for name, _ in named_parameters]
    if not names or len(set(names)) != len(names):
        raise ValueError("Provide a nonempty ordered list of unique named trainable parameters")
    if any(not parameter.requires_grad for _, parameter in named_parameters):
        raise ValueError("Diagnostic named_parameters must contain only trainable parameters")
    if len({id(parameter) for _, parameter in named_parameters}) != len(named_parameters):
        raise ValueError("Each trainable parameter must appear once")
    return named_parameters


def collect_loss_gradients(losses, named_parameters, *, retain_graph=False, to_cpu=True):
    """Return separate unscaled gradients without changing any parameter .grad.

    Shared graphs are retained between losses automatically. A loss with no
    graph (for example a deliberately absent auxiliary batch) yields None for
    every parameter; a differentiable zero can yield zero tensors. Calling with
    retain_graph=True additionally keeps the final graph for a later consumer.
    Returned gradients preserve FP32/BF16/etc storage dtype; summary reductions
    use FP64 independently. No GradScaler should be applied to diagnostic losses.
    """
    named_parameters = _parameters(named_parameters)
    parameters = tuple(parameter for _, parameter in named_parameters)
    for name, loss in losses.items():
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise ValueError(f"Diagnostic loss {name!r} must be a scalar Tensor")
        if not torch.isfinite(loss.detach()).all():
            raise RuntimeError(f"Nonfinite diagnostic loss: {name}")
    pending = sum(loss.requires_grad for loss in losses.values())
    result = {}
    for name, loss in losses.items():
        if not loss.requires_grad:
            result[name] = (None,) * len(parameters)
            continue
        pending -= 1
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True,
                                        retain_graph=retain_graph or pending > 0)
        result[name] = tuple(None if gradient is None else
                             (gradient.detach().cpu() if to_cpu else gradient.detach())
                             for gradient in gradients)
    return result


def default_parameter_groups(named_parameters):
    """SALAD groups, including the trainable backbone blocks and final norm."""
    named_parameters = _parameters(named_parameters)
    groups = {"all": list(range(len(named_parameters)))}
    for index, (name, _) in enumerate(named_parameters):
        if name.startswith("backbone."):
            groups.setdefault("backbone", []).append(index)
            block = re.search(r"(?:^|\.)blocks\.(\d+)\.", name)
            if block:
                groups.setdefault("backbone_block_" + block.group(1), []).append(index)
            elif name.startswith("backbone.model.norm."):
                groups.setdefault("backbone_norm", []).append(index)
        else:
            group = next((group for prefix, group in (
                ("aggregator.score.", "score"),
                ("aggregator.cluster_features.", "content"),
                ("aggregator.token_features.", "token"),
                ("aggregator.dust_bin", "dustbin")) if name.startswith(prefix)), "other")
            groups.setdefault(group, []).append(index)
    return groups


def _cosine(dot, norm_a, norm_b):
    defined = norm_a > 0 and norm_b > 0
    value = max(-1.0, min(1.0, dot / norm_a / norm_b)) if defined else None
    return {"value": value, "defined": defined,
            "angle_degrees": math.degrees(math.acos(value)) if defined else None}


def gradient_vector_summary(grads_by_loss, named_parameters, weights=None, *, max_norm=1.0,
                            groups=None, real_loss_name="real"):
    """Summarize gradient Gram matrices without flattened full-model copies.

    `groups` optionally maps names to sequences of parameter indices. The
    default includes `all`, backbone, individual trainable blocks, norm, SALAD
    score/content/token/dustbin. Weighted totals use the supplied nonnegative
    weights (missing names default to 1). This is hypothetical clipping only:
    no gradient tensors or model parameters are modified.

    The Gram matrix can be re-used to inspect alternative weights by calling
    this function again on the same detached gradients. Reductions stream one
    tensor pair at a time in FP64 on CPU, bounding temporary GPU allocations.
    """
    named_parameters = _parameters(named_parameters)
    if not grads_by_loss:
        raise ValueError("At least one diagnostic loss gradient is required")
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("max_norm must be finite and positive")
    loss_names = list(grads_by_loss)
    weights = dict(weights or {})
    if set(weights) - set(loss_names):
        raise ValueError("Weights name a loss without diagnostic gradients")
    weights = {name: float(weights.get(name, 1.0)) for name in loss_names}
    if any(not math.isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError("Diagnostic weights must be finite and nonnegative")
    for name, gradients in grads_by_loss.items():
        if len(gradients) != len(named_parameters):
            raise ValueError(f"Gradient tuple size differs for {name}")
        for gradient, (_, parameter) in zip(gradients, named_parameters):
            if gradient is not None and gradient.shape != parameter.shape:
                raise ValueError(f"Gradient shape differs for {name}")
    groups = default_parameter_groups(named_parameters) if groups is None else groups
    output = {"weights": weights, "max_norm": max_norm, "groups": {}}
    for group_name, raw_indices in groups.items():
        indices = list(raw_indices)
        if len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < len(named_parameters)
                                                   for i in indices):
            raise ValueError("Gradient groups must contain unique valid parameter indices")
        gram = {a: {b: 0.0 for b in loss_names} for a in loss_names}
        present = {name: 0 for name in loss_names}
        for index in indices:
            # Full-model flattened vectors are deliberately never materialized.
            tensors = {}
            for name in loss_names:
                gradient = grads_by_loss[name][index]
                if gradient is not None:
                    value = gradient.detach().to(device="cpu", dtype=torch.float64)
                    if not torch.isfinite(value).all():
                        raise RuntimeError(f"Nonfinite gradient in loss {name}")
                    tensors[name] = value.reshape(-1)
                    present[name] += 1
            for a, b in itertools.combinations_with_replacement(tensors, 2):
                dot = torch.dot(tensors[a], tensors[b]).item()
                gram[a][b] += dot
                if a != b:
                    gram[b][a] += dot
        norms = {name: math.sqrt(max(0.0, gram[name][name])) for name in loss_names}
        combined_sq = math.fsum(weights[a] * weights[b] * gram[a][b]
                                for a in loss_names for b in loss_names)
        # Gram cancellation at machine precision can produce a tiny negative.
        combined_norm = math.sqrt(max(0.0, combined_sq))
        clip_scale = min(1.0, max_norm / (combined_norm + 1e-6))
        combined = {"norm": combined_norm, "clip_scale": clip_scale,
                    "would_clip": combined_norm > max_norm,
                    "post_clip_norm": combined_norm * clip_scale}
        if real_loss_name in loss_names:
            dot_real = math.fsum(weights[name] * gram[name][real_loss_name] for name in loss_names)
            combined["vs_real"] = _cosine(dot_real, combined_norm, norms[real_loss_name])
        output["groups"][group_name] = {
            "parameter_tensors": len(indices),
            "parameter_elements": sum(named_parameters[i][1].numel() for i in indices),
            "losses": {name: {"raw_norm": norms[name], "weighted_norm": norms[name] * weights[name],
                               "weight": weights[name], "gradient_parameter_tensors": present[name]}
                       for name in loss_names},
            "cosines": {a + "|" + b: _cosine(gram[a][b], norms[a], norms[b])
                        for a, b in itertools.combinations(loss_names, 2)},
            "gram": gram, "combined": combined}
    return output
