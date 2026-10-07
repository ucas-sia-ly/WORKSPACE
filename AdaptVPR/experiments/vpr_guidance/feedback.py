"""Real-view proxy for SALAD's mined MultiSimilarity positive loss.

Utility is averaged *after* mining each sampled batch: mining at the mean
hardest-negative similarity can erase positives mined in occasional batches.
Grouped pools contain K distinct real views per place; a draw samples B-1
other places, matching the place-grouped training sampler. Positive draws may
sample K-1 real views to model a candidate anchor in that real-only batch.
This proxy excludes other synthetic views and training image augmentation.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Sequence

import torch
import torch.nn.functional as F


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _descriptor_tensor(value: torch.Tensor, name: str, dimensions: tuple[int, ...]) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim not in dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions")
    if not value.is_floating_point() or value.shape[-1] == 0:
        raise ValueError(f"{name} must have a nonempty floating-point descriptor dimension")
    norms = value.norm(dim=-1)
    if (not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(norms).all()) or
            bool((norms == 0).any())):
        raise ValueError(f"{name} must contain finite nonzero descriptors")


def _negative_maxima(
    candidates: torch.Tensor,
    pool: torch.Tensor,
    same_place: torch.Tensor,
    negatives_per_batch: int,
    draws: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Return [C, draws] maxima, with common random sets for fair comparisons."""
    _positive_integer(negatives_per_batch, "negatives_per_batch")
    _positive_integer(draws, "draws")
    _descriptor_tensor(candidates, "candidates", (2,))
    _descriptor_tensor(pool, "pool", (2, 3))
    if candidates.shape[-1] != pool.shape[-1] or candidates.device != pool.device:
        raise ValueError("Candidates and pool must have matching descriptor dimensions and devices")
    if candidates.dtype != pool.dtype:
        raise ValueError("Candidates and pool must have the same dtype")
    if (not isinstance(same_place, torch.Tensor) or same_place.dtype != torch.bool or
            same_place.shape != (len(candidates), len(pool)) or same_place.device != candidates.device):
        raise ValueError("same_place must be a boolean [candidates, pool entries] mask on the descriptor device")
    views_per_place = pool.shape[1] if pool.ndim == 3 else 1
    if views_per_place == 0 or negatives_per_batch % views_per_place:
        raise ValueError("negatives_per_batch must be divisible by views per negative place")
    places_per_batch = negatives_per_batch // views_per_place
    if len(candidates) == 0:
        return candidates.new_empty((0, draws))
    if bool(((~same_place).sum(dim=1) < places_per_batch).any()):
        raise ValueError("Negative pool is smaller than one training batch of other places")
    if pool.ndim == 3:
        similarity = (candidates @ pool.flatten(0, 1).T).reshape(
            len(candidates), len(pool), views_per_place).max(dim=2).values
    else:
        similarity = candidates @ pool.T
    similarity = similarity.masked_fill(same_place, -torch.inf)
    maxima = []
    for _ in range(draws):
        # Shared keys give same-place candidates the same negative context.
        keys = torch.rand((1, len(pool)), generator=generator, device=similarity.device)
        keys = keys.expand(len(candidates), -1).masked_fill(same_place, 2.0)
        index = keys.topk(places_per_batch, dim=1, largest=False).indices
        maxima.append(similarity.gather(1, index).max(dim=1).values)
    return torch.stack(maxima, dim=1)


def expected_hardest_negative(
    candidates: torch.Tensor,
    pool: torch.Tensor,
    same_place: torch.Tensor,
    negatives_per_batch: int,
    draws: int = 16,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Mean maximum cosine similarity across sampled negative batches.

    Descriptors are normalized. ``pool`` is [M, D] for independent images or
    [M, K, D] for place-grouped views; ``same_place`` is always [C, M].
    This diagnostic is not the threshold used to compute expected utility.
    """
    return _negative_maxima(candidates, pool, same_place, negatives_per_batch,
                            draws, generator).mean(dim=1)


def ms_positive_utility(
    positive_similarity: torch.Tensor,
    hardest_negative: torch.Tensor,
    positive_mask: torch.Tensor | None = None,
    alpha: float = 1.0,
    base: float = 0.0,
    epsilon: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (utility [C], mined count [C]) for one batch's positives [C, P]."""
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if not math.isfinite(base) or not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("base must be finite and epsilon must be finite and nonnegative")
    if (not isinstance(positive_similarity, torch.Tensor) or positive_similarity.ndim != 2 or
            not positive_similarity.is_floating_point() or not bool(torch.isfinite(positive_similarity).all())):
        raise ValueError("positive_similarity must be a finite floating-point [C, P] tensor")
    if (not isinstance(hardest_negative, torch.Tensor) or
            hardest_negative.shape != (len(positive_similarity),) or
            not hardest_negative.is_floating_point() or not bool(torch.isfinite(hardest_negative).all()) or
            hardest_negative.device != positive_similarity.device):
        raise ValueError("hardest_negative must be finite [C] values on the similarity device")
    if positive_mask is None:
        positive_mask = torch.ones_like(positive_similarity, dtype=torch.bool)
    if (not isinstance(positive_mask, torch.Tensor) or positive_mask.dtype != torch.bool or
            positive_mask.shape != positive_similarity.shape or
            positive_mask.device != positive_similarity.device):
        raise ValueError("positive_mask must be boolean and match the similarity shape and device")
    # Keep the official MultiSimilarityMiner's strict comparison and operation order.
    mined = positive_mask & (positive_similarity - epsilon < hardest_negative[:, None])
    terms = (-alpha * (positive_similarity - base)).masked_fill(~mined, -torch.inf)
    zero = positive_similarity.new_zeros((len(positive_similarity), 1))
    utility = torch.logsumexp(torch.cat((zero, terms), dim=1), dim=1) / alpha
    return utility, mined.sum(dim=1)


@torch.no_grad()
def score_candidates(
    candidate_descriptors: torch.Tensor,
    positive_descriptors: Sequence[torch.Tensor],
    pool_descriptors: torch.Tensor,
    same_place: torch.Tensor,
    negatives_per_batch: int,
    draws: int = 16,
    alpha: float = 1.0,
    base: float = 0.0,
    epsilon: float = 0.1,
    seed: int = 0,
    positives_per_batch: int | None = None,
) -> list[dict[str, float | int]]:
    """Average mined positive utility over batch draws against a real-only pool.

    ``positives_per_batch`` samples that many real co-anchors without replacement
    in each draw. Omitting it uses every supplied positive (legacy tensor API).
    ``mined_pairs`` is an expected count and ``mining_probability`` is the share
    of draws with any mined positive. Shared random draws eliminate Monte Carlo
    differences between duplicate or same-place candidate descriptors.
    """
    _descriptor_tensor(candidate_descriptors, "candidate_descriptors", (2,))
    _descriptor_tensor(pool_descriptors, "pool_descriptors", (2, 3))
    if len(positive_descriptors) != len(candidate_descriptors):
        raise ValueError("Need one positive set per candidate")
    if positives_per_batch is not None:
        _positive_integer(positives_per_batch, "positives_per_batch")
    for positives in positive_descriptors:
        _descriptor_tensor(positives, "positive_descriptors", (2,))
        if positives.shape[-1] != candidate_descriptors.shape[-1] or positives.device != candidate_descriptors.device:
            raise ValueError("Positive descriptors must match candidate dimensions and device")
        if len(positives) < (positives_per_batch or 1):
            raise ValueError("Every candidate needs enough distinct real views for its positive batch")
    candidates = F.normalize(candidate_descriptors.float(), dim=1)
    pool = F.normalize(pool_descriptors.float(), dim=-1)
    # Separate generators make the negative distribution independent of positive padding.
    generator = torch.Generator(device=candidates.device).manual_seed(seed)
    hardest = _negative_maxima(candidates, pool, same_place, negatives_per_batch, draws, generator)
    if len(candidates) == 0:
        return []
    width = max(len(p) for p in positive_descriptors)
    positive_similarity = candidates.new_zeros((len(candidates), width))
    mask = torch.zeros_like(positive_similarity, dtype=torch.bool)
    for i, positives in enumerate(positive_descriptors):
        positive_similarity[i, :len(positives)] = candidates[i] @ F.normalize(positives.float(), dim=1).T
        mask[i, :len(positives)] = True
    positive_generator = torch.Generator(device=candidates.device).manual_seed(seed ^ 0x5DEECE66D)
    utilities, counts = [], []
    for draw in range(draws):
        draw_mask = mask
        if positives_per_batch is not None:
            keys = torch.rand((1, width), generator=positive_generator, device=candidates.device)
            keys = keys.expand(len(candidates), -1).masked_fill(~mask, 2.0)
            indices = keys.topk(positives_per_batch, dim=1, largest=False).indices
            draw_mask = torch.zeros_like(mask).scatter_(1, indices, True)
        utility, mined = ms_positive_utility(positive_similarity, hardest[:, draw], draw_mask,
                                             alpha, base, epsilon)
        utilities.append(utility)
        counts.append(mined)
    utility = torch.stack(utilities, dim=1).mean(dim=1)
    mined_counts = torch.stack(counts, dim=1)
    return [
        {
            "utility": float(utility[i]),
            "mined_pairs": float(mined_counts[i].float().mean()),
            "mining_probability": float((mined_counts[i] > 0).float().mean()),
            "positive_pairs": positives_per_batch or int(mask[i].sum()),
            "available_real_views": int(mask[i].sum()),
            "mean_positive_similarity": float(positive_similarity[i][mask[i]].mean()),
            "min_positive_similarity": float(positive_similarity[i][mask[i]].min()),
            "expected_hardest_negative": float(hardest[i].mean()),
        }
        for i in range(len(candidates))
    ]


def select_per_group(rows: Sequence[dict[str, Any]], method: str, seed: int = 0,
                     group_key: str = "sample_id") -> list[dict[str, Any]]:
    """Keep one verified candidate per source/condition/prompt group.

    Hardness maximizes expected utility, then minimizes mean positive similarity.
    Random uses the identical verified pool, groups, and deterministic ordering.
    """
    if method not in {"hardness", "random"}:
        raise ValueError(f"Unknown selection method: {method}")
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("passed") is True:
            if group_key not in row:
                raise ValueError(f"Verified candidate lacks {group_key}")
            if method == "hardness" and any(
                    not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key])
                    for key in ("utility", "mean_positive_similarity")):
                raise ValueError("Hardness selection requires finite utility and mean positive similarity")
            groups[row[group_key]].append(row)
    rng = random.Random(seed)
    selected = []
    for key in sorted(groups, key=str):
        members = sorted(groups[key], key=lambda r: r.get("candidate_index", 0))
        if method == "hardness":
            choice = max(members, key=lambda r: (r["utility"], -r["mean_positive_similarity"]))
        else:
            choice = rng.choice(members)
        selected.append({**choice, "selection": method, "group_size": len(members)})
    return selected
