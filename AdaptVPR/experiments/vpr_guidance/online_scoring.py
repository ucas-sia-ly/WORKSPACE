"""Persistent SALAD scoring context for adaptive, in-process generation.

Real descriptors, the grouped negative pool and the leave-one-out gate are
computed once. Each verified candidate is decoded from its final image file
with the same deterministic preprocessing as the standalone scorer. Positive
random draws use a dataset-wide width, so scores do not depend on the other
candidates in a batch or on their generation order.
"""

from __future__ import annotations

import copy
import math
import random
import statistics
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable

from common import file_sha256, use_salad

use_salad()

import torch  # noqa: E402

from feedback import identity_margin, plausibility_floor, score_candidates  # noqa: E402
from score_candidates import _distribution, sample_negative_pool  # noqa: E402


class OnlineScorer:
    """Load one student and score verified candidates against fixed real context.

    ``args`` uses the names from ``score_candidates.parse_args``; candidate
    manifests, selection and output_dir are not needed. ``source_paths`` must
    contain all sources the generation run will use. Sources outside eligible
    training metadata fail before the student is loaded.

    Optional factory injections have the same signatures as
    ``MixedGSVCitiesDataset``, ``load_checkpoint_model`` and
    ``extract_descriptors`` respectively, allowing CPU tests without DINOv2.
    The extractor returns a floating-point [images, descriptor_dim] tensor.
    """

    def __init__(
        self,
        args,
        source_paths: Iterable[str | Path],
        *,
        dataset_factory: Callable | None = None,
        model_factory: Callable | None = None,
        descriptor_extractor: Callable | None = None,
    ) -> None:
        self.args = SimpleNamespace(**copy.deepcopy(vars(args)))
        args = self.args
        for name in ("negative_pool_size", "negative_draws", "batch_size", "calibration_places"):
            value = getattr(args, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if args.train_batch_size < 2 or args.images_per_place < 2:
            raise ValueError("train_batch_size and images_per_place must be at least 2")
        if args.min_images_per_place < args.images_per_place:
            raise ValueError("min_images_per_place must be at least images_per_place")
        if not 0 <= args.plausibility_quantile < 1 or args.num_workers < 0:
            raise ValueError("plausibility_quantile must be in [0, 1) and num_workers nonnegative")
        if not math.isfinite(args.alpha) or args.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if not math.isfinite(args.base) or not math.isfinite(args.miner_margin) or args.miner_margin < 0:
            raise ValueError("base must be finite and miner_margin finite and nonnegative")
        args.real_data = Path(args.real_data).expanduser().resolve()
        args.checkpoint = Path(args.checkpoint).expanduser().resolve()

        if dataset_factory is None:
            from workflow.training_data import MixedGSVCitiesDataset

            dataset_factory = MixedGSVCitiesDataset
        self.dataset = dataset_factory(
            args.real_data, None, cities=args.cities, images_per_place=args.images_per_place,
            min_images_per_place=args.min_images_per_place, augment=False,
        )
        if len(self.dataset) < 2:
            raise ValueError("Need at least two eligible training places to define negatives")
        self.real_views = {(p.city, p.place_id): p.real_paths for p in self.dataset.places}
        self.place_of = {path: key for key, paths in self.real_views.items() for path in paths}
        self.source_paths = {Path(path).expanduser().resolve() for path in source_paths}
        if not self.source_paths:
            raise ValueError("OnlineScorer requires at least one selected source")
        for path in sorted(self.source_paths):
            if path not in self.place_of:
                raise ValueError(f"Selected source is not an eligible training metadata image: {path}")
        self._selected_places = {self.place_of[path] for path in self.source_paths}
        self.effective_batch_size = min(args.train_batch_size, len(self.dataset))
        self.negatives_per_batch = (self.effective_batch_size - 1) * args.images_per_place
        self.positive_context_width = max(len(views) for views in self.real_views.values())
        self.pool = sample_negative_pool(
            self.real_views, args.negative_pool_size, args.images_per_place, args.seed,
        )
        self.pool_keys = [key for key, _ in self.pool]
        self.calibration_keys = (
            random.Random(args.seed + 1).sample(
                sorted(self.real_views), min(args.calibration_places, len(self.real_views)),
            ) if args.plausibility_quantile > 0 else []
        )
        for place in self._selected_places | set(self.calibration_keys):
            if sum(key != place for key in self.pool_keys) < self.effective_batch_size - 1:
                raise ValueError("Negative pool lacks enough other places; increase --negative-pool-size")

        self.checkpoint_sha256 = file_sha256(args.checkpoint)
        self.score_settings = {
            "real_data": str(args.real_data), "cities": self.dataset.cities,
            "train_batch_size": args.train_batch_size, "images_per_place": args.images_per_place,
            "effective_train_batch_size": self.effective_batch_size,
            "min_images_per_place": args.min_images_per_place,
            "negative_pool_size_requested": args.negative_pool_size,
            "negative_draws": args.negative_draws, "miner_margin": args.miner_margin,
            "alpha": args.alpha, "base": args.base, "seed": args.seed,
            "utility": "expected_mined_ms_positive_term",
            "plausibility_quantile": args.plausibility_quantile,
            "calibration_places": args.calibration_places,
            "batch_context": "real_only_fixed_negative_views_per_place",
            "positive_views_per_draw": args.images_per_place - 1,
            "positive_context_width": self.positive_context_width,
            "positive_draw_context": "dataset_max_real_views",
            "training_augmentation_simulated": False,
            "partial_tail_batches_simulated": False,
        }
        if model_factory is None:
            from workflow.model import load_checkpoint_model

            model_factory = load_checkpoint_model
        if descriptor_extractor is None:
            from workflow.evaluation import extract_descriptors

            descriptor_extractor = extract_descriptors
        self._extractor = descriptor_extractor
        self._model_lock = threading.Lock()
        self.model = model_factory(args.checkpoint, args.device, backbone_repo=args.backbone_repo)
        self.model.eval()

        images: dict[Path, None] = {}
        for source in sorted(self.source_paths):
            for path in self.real_views[self.place_of[source]]:
                images.setdefault(path, None)
        for key in self.calibration_keys:
            for path in self.real_views[key]:
                images.setdefault(path, None)
        for _, views in self.pool:
            for path in views:
                images.setdefault(path, None)
        paths = list(images)
        descriptors = self._extract(paths)
        index = {path: i for i, path in enumerate(paths)}
        self._positive_descriptors = {
            key: descriptors[[index[path] for path in views]]
            for key, views in self.real_views.items()
            if key in self._selected_places or key in self.calibration_keys
        }
        self._pool_descriptors = torch.stack([
            descriptors[[index[path] for path in views]] for _, views in self.pool
        ])
        self.floor = None
        self.plausibility_calibration = None
        if self.calibration_keys:
            anchors, positive_sets, places = [], [], []
            for key in self.calibration_keys:
                views = self.real_views[key]
                for i, view in enumerate(views):
                    anchors.append(index[view])
                    positive_sets.append(descriptors[[index[p] for j, p in enumerate(views) if j != i]])
                    places.append(key)
            real_scores = self._score_descriptors(descriptors[anchors], positive_sets, places)
            self.floor = plausibility_floor(real_scores, args.plausibility_quantile)
            self.plausibility_calibration = {
                "places": len(self.calibration_keys), "real_anchors": len(real_scores),
                "quantile": args.plausibility_quantile, "margin_floor": self.floor,
                "real_mined_rate": statistics.fmean(s["mining_probability"] for s in real_scores),
                "real_margin": _distribution([identity_margin(s) for s in real_scores]),
            }
        self.calibration = self.plausibility_calibration

    def _extract(self, paths: list[Path]) -> torch.Tensor:
        args = self.args
        with self._model_lock, torch.inference_mode():
            descriptors = self._extractor(
                self.model, [SimpleNamespace(path=p) for p in paths], self.model.image_size,
                args.device, args.batch_size, args.num_workers,
            )
        if not isinstance(descriptors, torch.Tensor) or descriptors.ndim != 2 or len(descriptors) != len(paths):
            raise ValueError("Descriptor extractor must return one descriptor per image")
        return descriptors.detach().float().cpu()

    def _score_descriptors(self, anchors, positive_sets, places):
        args = self.args
        same = torch.tensor([[place == key for key in self.pool_keys] for place in places], dtype=torch.bool)
        return score_candidates(
            anchors, positive_sets, self._pool_descriptors, same, self.negatives_per_batch,
            args.negative_draws, args.alpha, args.base, args.miner_margin, args.seed,
            positives_per_batch=args.images_per_place - 1,
            positive_context_width=self.positive_context_width,
        )

    def score(self, row: dict[str, Any]) -> dict[str, Any]:
        """Return the standalone scorer's manifest fields for one verified image.

        The final file is read on each call; replacing a generated JPEG cannot
        reuse a stale descriptor. ``_place`` is internal and never emitted.
        """
        if row.get("passed") is not True:
            raise ValueError("OnlineScorer only scores verified candidates (passed must be true)")
        if not isinstance(row.get("sample_id"), str) or not row["sample_id"].strip():
            raise ValueError("Verified candidate requires a nonempty sample_id")
        candidate_index = row.get("candidate_index")
        if isinstance(candidate_index, bool) or not isinstance(candidate_index, int) or candidate_index < 0:
            raise ValueError("candidate_index must be a nonnegative integer")
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError("Verified candidate requires a nonempty prompt")
        if row.get("condition") is not None and not isinstance(row["condition"], str):
            raise ValueError("condition must be a string when provided")
        if "s_geo" in row and (isinstance(row["s_geo"], bool) or
                              not isinstance(row["s_geo"], (int, float)) or not math.isfinite(row["s_geo"])):
            raise ValueError("s_geo must be a finite number when provided")
        paths = {}
        for name in ("source_path", "output_path"):
            value = row.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty path string")
            paths[name] = Path(value).expanduser().resolve()
            if not paths[name].is_file():
                raise FileNotFoundError(f"{name} does not resolve to an existing file: {value!r}")
        source, output = paths["source_path"], paths["output_path"]
        if source not in self.source_paths:
            raise ValueError(f"Source was not selected when this scoring context was initialized: {source}")
        if output in self.dataset.source_index or output == source:
            raise ValueError("output_path must be a generated image, not a real GSV image")
        key = self.place_of[source]
        scores = self._score_descriptors(
            self._extract([output]), [self._positive_descriptors[key]], [key],
        )[0]
        margin = identity_margin(scores)
        return {
            **{k: v for k, v in row.items() if k != "_place"},
            **{name: str(path) for name, path in paths.items()}, **scores,
            "identity_margin": margin, "plausible": self.floor is None or margin >= self.floor,
            "eligible_for_training": True,
        }

    def metadata(self) -> dict[str, Any]:
        """Return serializable checkpoint, scoring-context and gate provenance."""
        return copy.deepcopy({
            "checkpoint": str(self.args.checkpoint), "checkpoint_sha256": self.checkpoint_sha256,
            "score_settings": self.score_settings,
            "plausibility_calibration": self.plausibility_calibration,
            "negatives_per_batch": self.negatives_per_batch,
            "negative_places_per_batch": self.effective_batch_size - 1,
            "negative_pool_size": len(self.pool),
            "negative_pool_images": len(self.pool) * self.args.images_per_place,
        })
