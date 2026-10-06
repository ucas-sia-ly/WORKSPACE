"""GSV-only place episodes and truncated, second-order functional SGD.

The fixed base initialization is never optimized. Fast weights start afresh on
EVERY episode; gradients of the inner gradient must retain the support graph.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.func import functional_call

from .data import GSVLabelIndex, canonical_place_id, validate_source_label
from .salad_factory import metric_loss


class EpisodeUnavailable(ValueError):
    """A valid current-anchored, multi-place episode cannot be formed."""


@dataclass(frozen=True)
class MetaPlace:
    key: tuple[str, str]
    label: int
    synthetic_row: dict
    support_real: tuple[str, ...]
    query_real: tuple[str, ...]


@dataclass(frozen=True)
class MetaEpisode:
    places: tuple[MetaPlace, ...]

    def validate(self):
        if len(self.places) < 2:
            raise ValueError("meta episode requires multiple distinct places")
        if len({p.key for p in self.places}) != len(self.places):
            raise ValueError("duplicate geographical place in meta episode")
        if len({p.label for p in self.places}) != len(self.places):
            raise ValueError("distinct places must have distinct labels")
        all_support, all_query, generated = set(), set(), set()
        for place in self.places:
            row = place.synthetic_row
            if (row.get("passed") is not True or row.get("eligible_for_training") is not True
                    or row.get("route") != "global"):
                raise ValueError("rejected/non-Global candidate cannot enter meta support")
            if (str(row["city"]), canonical_place_id(row["place_id"])) != place.key:
                raise ValueError("synthetic support has wrong place label")
            support = {str(Path(p).resolve()) for p in place.support_real}
            query = {str(Path(p).resolve()) for p in place.query_real}
            if not support or len(query) < 2 or len(support) != len(place.support_real) or len(query) != len(place.query_real):
                raise ValueError("meta support/query need unique real images and >=2 query images")
            if str(Path(row["source_path"]).resolve()) not in support:
                raise ValueError("synthetic source must be real support, never held-out query")
            if (support | query) & (all_support | all_query):
                raise ValueError("real files overlap within/across meta places")
            if support & query:
                raise ValueError("support/query real files must be disjoint")
            all_support |= support
            all_query |= query
            generated.add(str(Path(row["generated_path"]).resolve()))
        if generated & (all_support | all_query):
            raise ValueError("real support/query cannot contain generated files")


class GSVRealIndex:
    """Index existing REAL GSV files with labels validated against Dataframes.

    Filename parsing is only identity resolution; GSVLabelIndex requires a
    matching official dataframe record before it returns any place label.
    """
    def __init__(self, gsv_root, cities):
        root = Path(gsv_root).resolve()
        self.cities = sorted(set(cities))
        self.labels = GSVLabelIndex(root / "Dataframes", self.cities)
        self.paths = defaultdict(list)
        for city in self.cities:
            for path in sorted((root / "Images" / city).iterdir()):
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    key = self.labels.key_for_source(path)
                    resolved = str(path.resolve())
                    self.paths[key].append(resolved)
        self.paths = {k: tuple(sorted(set(v))) for k, v in self.paths.items()}

    def key_for_row(self, row):
        key = validate_source_label(row, Path(row["source_path"]), self.labels)
        if str(Path(row["source_path"]).resolve()) not in self.paths.get(key, ()):
            raise ValueError("synthetic source is outside indexed real GSV training files")
        return key


def construct_episode(current, replay, real_index, rng, *, places=4,
                      support_real_per_place=1, query_real_per_place=2):
    if places < 2 or support_real_per_place < 1 or query_real_per_place < 2:
        raise ValueError("invalid metric-learning episode sizes")
    if not current:
        raise EpisodeUnavailable("no current accepted anchors; replay alone cannot train")
    groups, current_groups = defaultdict(list), defaultdict(list)
    for is_current, pool in [(True, current), *[(False, p) for p in replay]]:
        for row in pool:
            if (row.get("passed") is not True or row.get("eligible_for_training") is not True
                    or row.get("route") != "global"):
                raise ValueError("active meta/replay pool contains rejected/non-Global sample")
            key = real_index.key_for_row(row)
            if len(real_index.paths.get(key, ())) >= support_real_per_place + query_real_per_place:
                groups[key].append(row)
                if is_current:
                    current_groups[key].append(row)
    if not current_groups:
        raise EpisodeUnavailable("no current place has enough distinct real support/query images")
    if len(groups) < places:
        raise EpisodeUnavailable(f"need {places} eligible distinct places, found {len(groups)}")
    anchor = rng.choice(sorted(current_groups))
    selected = [anchor, *rng.sample(sorted(k for k in groups if k != anchor), places - 1)]
    result = []
    for label, key in enumerate(selected):
        # Prefer current accepted samples at each place; replay fills missing places.
        row = rng.choice(current_groups.get(key) or groups[key])
        source = str(Path(row["source_path"]).resolve())
        remaining = [p for p in real_index.paths[key] if p != source]
        chosen = rng.sample(remaining, support_real_per_place - 1 + query_real_per_place)
        split = support_real_per_place - 1
        result.append(MetaPlace(key, label, row, (source, *chosen[:split]), tuple(chosen[split:])))
    episode = MetaEpisode(tuple(result))
    episode.validate()
    return episode


@dataclass
class BilevelResult:
    outer_loss: torch.Tensor
    inner_loss_before: torch.Tensor
    inner_loss_after: torch.Tensor
    metrics: dict
    fast_params: dict


def _validate_labels(support_labels, query_labels):
    keys, counts = query_labels.unique(return_counts=True)
    if len(keys) < 2 or (counts < 2).any():
        raise ValueError("query requires >=2 distinct places with >=2 real images each")
    support_keys, support_counts = support_labels.unique(return_counts=True)
    if not torch.equal(keys, support_keys) or (support_counts < 2).any():
        raise ValueError("support needs real + synthetic for every query place")


def bilevel_objective(model, support_images, support_labels, query_images, query_labels,
                      *, inner_lr=1e-3, inner_steps=1, loss_fn=None, miner=None):
    """Pure functional SGD, no Lightning hooks, base mutation, or detached VJP."""
    if inner_steps < 1 or inner_lr <= 0:
        raise ValueError("inner_steps >=1 and inner_lr >0 required")
    _validate_labels(support_labels, query_labels)
    if query_images.requires_grad:
        raise ValueError("outer query must contain independent real images")
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("meta SALAD parameters must be fp32")
    loss_fn = loss_fn if loss_fn is not None else model.loss_fn
    miner = miner if miner is not None else model.miner
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if not names:
        raise ValueError("no trainable inner SALAD parameters")
    # Clones isolate fast parameters and any buffers without severing the graph.
    fast = {n: p.clone() for n, p in model.named_parameters()}
    buffers = {n: b.clone() for n, b in model.named_buffers()}
    before, pairs = None, None
    # No AMP in the higher-order inner/outer computation.
    with torch.autocast(device_type=support_images.device.type, enabled=False):
        for _ in range(inner_steps):
            desc = functional_call(model, (fast, buffers), (support_images.float(),), strict=True)
            inner, pair_counts = metric_loss(desc, support_labels, loss_fn, miner)
            if before is None:
                before, pairs = inner, pair_counts
            grads = torch.autograd.grad(inner, [fast[n] for n in names],
                                        create_graph=True, allow_unused=True)
            for name, grad in zip(names, grads):
                if grad is not None:
                    if not torch.isfinite(grad).all():
                        raise FloatingPointError(f"non-finite inner gradient: {name}")
                    fast[name] = fast[name] - inner_lr * grad
        # Logging only; no need to retain another support graph after adaptation.
        with torch.no_grad():
            after, _ = metric_loss(functional_call(model, (fast, buffers),
                                    (support_images.float(),), strict=True), support_labels, loss_fn, miner)
        query_desc = functional_call(model, (fast, buffers), (query_images.float(),), strict=True)
        outer, outer_pairs = metric_loss(query_desc, query_labels, loss_fn, miner)
    metrics = {"meta/inner_loss_before": float(before.detach()),
               "meta/inner_loss_after": float(after), "meta/outer_loss": float(outer.detach()),
               "meta/inner_steps": inner_steps, "meta/places": query_labels.unique().numel(),
               "meta/support_images": support_labels.numel(), "meta/query_images": query_labels.numel(),
               "meta/mined_inner_pairs": pairs["positive_pairs"] + pairs["negative_pairs"],
               "meta/mined_outer_pairs": outer_pairs["positive_pairs"] + outer_pairs["negative_pairs"],
               "meta/mined_inner_positive_pairs": pairs["positive_pairs"],
               "meta/mined_inner_negative_pairs": pairs["negative_pairs"],
               "meta/mined_outer_positive_pairs": outer_pairs["positive_pairs"],
               "meta/mined_outer_negative_pairs": outer_pairs["negative_pairs"]}
    return BilevelResult(outer, before, after, metrics, fast)
