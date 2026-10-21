"""Place-grouped GSV-Cities training with accepted AdaptVPR augmentations.

No legacy dataloader imports or hard-coded dataset locations are used. Paths in
the synthetic JSONL must use AdaptVPR's ``source_path`` and ``output_path``
fields. Absolute paths are used directly. Relative paths are checked relative
to the manifest directory and its ancestors, and the invocation's working
directory: AdaptVPR can write paths relative to its project working directory.
Sources can also be relative to the GSV root or its ``Images`` directory.
Multiple distinct existing resolutions are rejected as ambiguous. Relocated
absolute paths must be fixed in the manifest rather than silently remapped.

Only entries with both acceptance flags literally ``true`` are consumed.
Optional ``plausible`` and ``weather_ok`` flags must also be literally ``true``
when present. Source replacement preserves the original distinct-view sampler:
accepted variants can replace only their exact metadata source view.
Augmentations inherit the place label of the exact source file in the metadata.
Training places must have enough distinct *real* views. Each sample retains at
least one real image and never samples the same path twice within that sample.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PlaceImages:
    city: str
    place_id: int
    label: int = -1
    real_paths: list[Path] = field(default_factory=list)
    synthetic_paths: list[Path] = field(default_factory=list)
    synthetic_by_source: dict[Path, dict[str, list[Path]]] = field(default_factory=dict)
    source_by_synthetic_path: dict[Path, Path] = field(default_factory=dict)
    condition_by_synthetic_path: dict[Path, str] = field(default_factory=dict)


def _sampling_seed(*parts: Any) -> int:
    """Process-independent seed; never use Python's salted hash()."""
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def coverage_pk_indices(n: int, k: int, *, seed: int, city: str,
                        place_id: int, epoch: int) -> list[int]:
    """Return epoch's K unique indices into a stable, complete real pool.

    Concatenate N-element permutations and read slots [epoch*K, (epoch+1)*K).
    At a boundary inside a group, move the previous tail's identities behind
    the next permutation's other identities. This prevents within-group
    duplicates WITHOUT discarding any entries of either permutation.

    Repairs depend only on the previous permutation. Every lcm(N,K) slots the
    permutation and group boundaries align, so reconstruction starts there:
    at most K/gcd(N,K) preceding cycles, independent of epoch/resume history.
    No queue, worker state, global RNG, or history of actual visits is used.
    """
    if not 1 <= k <= n or epoch < 0:
        raise ValueError("coverage_pk requires 1 <= K <= pool size and epoch >= 0")
    start, stop = epoch * k, (epoch + 1) * k
    first_cycle, last_cycle = start // n, (stop - 1) // n
    cycles_per_alignment = k // math.gcd(n, k)
    aligned_cycle = first_cycle // cycles_per_alignment * cycles_per_alignment
    previous: list[int] = []
    selected = []
    for cycle in range(aligned_cycle, last_cycle + 1):
        permutation = list(range(n))
        random.Random(_sampling_seed("coverage_pk_v1", seed, city, place_id, cycle)).shuffle(permutation)
        tail_size = cycle * n % k
        if tail_size:
            blocked = set(previous[-tail_size:])
            permutation = ([i for i in permutation if i not in blocked]
                           + [i for i in permutation if i in blocked])
        if cycle >= first_cycle:
            left, right = max(0, start - cycle * n), min(n, stop - cycle * n)
            selected.extend(permutation[left:right])
        previous = permutation
    return selected


class CoveragePlaceBatchSampler:
    """One visit/place/epoch, with epoch carried in every worker's item key.

    A final singleton borrows a place from the preceding batch. With P=2 it
    merges instead (one batch of three), retaining metric-learning negatives.
    Explicit keys also work with persistent workers; set_epoch on a parent
    dataset alone would not update their copies.
    """

    def __init__(self, num_places: int, batch_size: int, seed: int) -> None:
        if num_places < 2 or batch_size < 2:
            raise ValueError("Coverage batches require at least two places")
        self.num_places, self.batch_size, self.seed = num_places, batch_size, seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative (zero-based)")
        self.epoch = epoch

    def __iter__(self):
        epoch = self.epoch
        indices = list(range(self.num_places))
        random.Random(_sampling_seed("coverage_places_v1", self.seed, epoch)).shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        if len(batches[-1]) == 1:
            if len(batches[-2]) > 2:
                batches[-1].insert(0, batches[-2].pop())
            else:
                tail = batches.pop()
                batches[-1].extend(tail)
        for batch in batches:
            yield [(epoch, index) for index in batch]

    def __len__(self) -> int:
        count = (self.num_places + self.batch_size - 1) // self.batch_size
        return count - int(self.batch_size == 2 and self.num_places > 2 and self.num_places % 2 == 1)


class StratifiedCoverageBatchSampler:
    """Reserve ``slots`` places per batch for places with accepted variants.

    One epoch is one coverage pass over the OTHER places (P - slots per batch).
    Eligible places follow their own continuous stream of keyed permutations,
    continuing across epochs, so they are revisited about
    ``len(self) * slots / num_eligible`` times per epoch. A place is never placed
    twice in one batch; a duplicate at a permutation boundary moves to the next
    batch. Real-only controls must use the same sampler to keep identical place
    streams. Keys carry the epoch like ``CoveragePlaceBatchSampler``.
    """

    def __init__(self, eligible: list[bool], batch_size: int, slots: int, seed: int) -> None:
        self.eligible = [i for i, flag in enumerate(eligible) if flag]
        self.other = [i for i, flag in enumerate(eligible) if not flag]
        if not 1 <= slots < batch_size:
            raise ValueError("synthetic place slots must be in [1, batch_size)")
        if len(self.eligible) < slots or not self.other:
            raise ValueError("Need at least `slots` eligible places and one other place")
        self.batch_size, self.slots, self.seed = batch_size, slots, seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative (zero-based)")
        self.epoch = epoch

    def __len__(self) -> int:
        per_batch = self.batch_size - self.slots
        return (len(self.other) + per_batch - 1) // per_batch

    def _eligible_stream(self, start: int):
        n = len(self.eligible)
        cycle, offset = divmod(start, n)
        while True:
            order = list(self.eligible)
            random.Random(_sampling_seed("stratified_eligible_v1", self.seed, cycle)).shuffle(order)
            yield from order[offset:]
            cycle, offset = cycle + 1, 0

    def __iter__(self):
        epoch, per_batch = self.epoch, self.batch_size - self.slots
        other = list(self.other)
        random.Random(_sampling_seed("stratified_other_v1", self.seed, epoch)).shuffle(other)
        # Epochs start at fixed stream positions so resume is stateless; a place
        # still deferred at the end of an epoch is simply not visited then.
        stream = self._eligible_stream(epoch * len(self) * self.slots)
        deferred: list[int] = []
        for start in range(0, len(other), per_batch):
            chosen: list[int] = []
            pending, deferred = deferred, []
            while len(chosen) < self.slots:
                index = pending.pop(0) if pending else next(stream)
                (deferred if index in chosen else chosen).append(index)
            deferred = pending + deferred
            yield [(epoch, index) for index in chosen + other[start:start + per_batch]]


class SamplingCoverage:
    """Main-process accounting of consumed metric images, never prefetched data.

    Synthetic slots and auxiliary real companions are excluded. Accounting
    state is checkpointed, but never feeds back into sampling decisions.
    """

    def __init__(self, dataset, *, since_epoch: int = 0, state: dict | None = None):
        self.places = dataset.places
        self.pool_sizes = [len(place.real_paths) for place in self.places]
        self.total_real_images = sum(self.pool_sizes)
        digest = hashlib.sha256()
        for place in self.places:
            digest.update(json.dumps([place.city, place.place_id, [p.name for p in place.real_paths]],
                                     ensure_ascii=False).encode("utf-8") + b"\n")
        self.pool_digest = digest.hexdigest()
        self.seen = [bytearray(size) for size in self.pool_sizes]
        self.visits = [0] * len(self.places)
        self.exposures = self.unique = 0
        self.since_epoch = since_epoch
        if state is not None:
            if state.get("version") != 1 or state.get("pool_digest") != self.pool_digest:
                raise ValueError("Sampling coverage checkpoint does not match the real image pools")
            self.seen = [bytearray(bits) for bits in state["seen"]]
            self.visits = list(state["visits"])
            self.exposures, self.unique = state["exposures"], state["unique"]
            self.since_epoch = state["since_epoch"]
            if [len(bits) for bits in self.seen] != self.pool_sizes or len(self.visits) != len(self.places):
                raise ValueError("Invalid sampling coverage checkpoint dimensions")
        self.begin_epoch()

    def begin_epoch(self) -> None:
        self.epoch_seen: set[tuple[int, int]] = set()
        self.epoch_visits = [0] * len(self.places)
        self.epoch_exposures = self.epoch_new = 0

    def update(self, metadata) -> None:
        for place_index, indices in zip(metadata["place_index"].tolist(), metadata["real_indices"].tolist()):
            self.visits[place_index] += 1
            self.epoch_visits[place_index] += 1
            for index in indices:
                if index < 0:  # Synthetic image, not a real exposure.
                    continue
                self.exposures += 1
                self.epoch_exposures += 1
                self.epoch_seen.add((place_index, index))
                if not self.seen[place_index][index]:
                    self.seen[place_index][index] = 1
                    self.unique += 1
                    self.epoch_new += 1

    def summary(self) -> dict:
        return {
            "statistics_since_epoch": self.since_epoch + 1,
            "eligible_real_images": self.total_real_images,
            "epoch_real_exposures": self.epoch_exposures,
            "epoch_unique_real_images": len(self.epoch_seen),
            "epoch_new_real_images": self.epoch_new,
            "epoch_repeated_real_exposures": self.epoch_exposures - len(self.epoch_seen),
            "epoch_previously_seen_real_exposures": self.epoch_exposures - self.epoch_new,
            "cumulative_real_exposures": self.exposures,
            "cumulative_unique_real_images": self.unique,
            "cumulative_real_coverage": self.unique / self.total_real_images,
            "cumulative_repeated_real_exposures": self.exposures - self.unique,
            "epoch_place_visits": sum(self.epoch_visits),
            "epoch_unique_places": sum(count > 0 for count in self.epoch_visits),
        }

    def place_statistics(self) -> list[dict]:
        epoch_total, total = sum(self.epoch_visits), sum(self.visits)
        return [
            {"city": place.city, "place_id": place.place_id,
             "epoch_visits": self.epoch_visits[i], "cumulative_visits": self.visits[i],
             "epoch_sampling_frequency": self.epoch_visits[i] / epoch_total if epoch_total else 0.0,
             "cumulative_sampling_frequency": self.visits[i] / total if total else 0.0}
            for i, place in enumerate(self.places)
        ]

    def state_dict(self) -> dict:
        return {"version": 1, "pool_digest": self.pool_digest,
                "seen": [bytes(bits) for bits in self.seen], "visits": list(self.visits),
                "exposures": self.exposures, "unique": self.unique, "since_epoch": self.since_epoch}


class SyntheticAuxSampler:
    """At most one exact-source synthetic query per current main-batch place.

    Selection and image augmentation use a private, checkpointed RNG. There is
    no fallback to a source outside the K real images and no extra source decode.
    ``source_indices`` index the flattened P*K main image tensor.
    """

    def __init__(self, dataset, exposure_prob: float, seed: int):
        if not math.isfinite(exposure_prob) or not 0 <= exposure_prob <= 1:
            raise ValueError("synthetic exposure probability must be in [0, 1]")
        self.dataset = dataset
        self.exposure_prob = exposure_prob
        self.rng = random.Random(_sampling_seed("synthetic_aux_v1", seed))
        self.begin_epoch()

    def begin_epoch(self):
        self.counts = dict(main_places=0, places_with_synthetic=0, eligible_places=0,
                           main_real_images=0, available_sources=0, triggered_places=0)
        self.conditions: dict[str, int] = {}

    def state_dict(self):
        return {"version": 1, "rng": self.rng.getstate()}

    def load_state_dict(self, state):
        if state.get("version") != 1:
            raise ValueError("Unsupported synthetic auxiliary sampler checkpoint")
        self.rng.setstate(state["rng"])

    def _read_image(self, path):
        import numpy as np
        import torch
        from PIL import Image
        from workflow.augmentation import randaugment

        height, width = self.dataset.image_size
        try:
            with Image.open(path) as original:
                image = original.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Cannot decode auxiliary synthetic image {path}: {exc}") from exc
        if self.dataset.augment:
            # The key comes from this sampler's checkpointed RNG.
            image, = randaugment([image], self.rng.getrandbits(63))
        pixels = np.asarray(image, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(pixels).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (tensor - mean) / std

    def sample(self, metadata):
        import torch

        groups: dict[int, list[tuple[int, Path]]] = {}
        flat_index = 0
        for place_index, indices in zip(metadata["place_index"].tolist(), metadata["real_indices"].tolist()):
            place = self.dataset.places[place_index]
            candidates = groups.setdefault(place_index, [])
            for index in indices:
                if index < 0:
                    raise ValueError("Synthetic auxiliary training requires an entirely real main batch")
                source = place.real_paths[index]
                self.counts["main_real_images"] += 1
                if source in place.synthetic_by_source:
                    candidates.append((flat_index, source))
                    self.counts["available_sources"] += 1
                flat_index += 1
        images, labels, source_indices = [], [], []
        synthetic_paths, source_paths, conditions = [], [], []
        for place_index, candidates in groups.items():
            place = self.dataset.places[place_index]
            self.counts["main_places"] += 1
            self.counts["places_with_synthetic"] += bool(place.synthetic_paths)
            self.counts["eligible_places"] += bool(candidates)
            if not candidates or self.exposure_prob == 0:
                continue
            if self.rng.random() >= self.exposure_prob:
                continue
            source_index, source = self.rng.choice(candidates)
            domains = place.synthetic_by_source[source]
            domain = self.rng.choice(sorted(domains))
            output = self.rng.choice(domains[domain])
            # Manifest loader has validated this exact source relation.
            if place.source_by_synthetic_path[output] != source:
                raise ValueError("Auxiliary synthetic image has a conflicting exact source")
            condition = place.condition_by_synthetic_path.get(output, "unknown")
            images.append(self._read_image(output))
            labels.append(place.label)
            source_indices.append(source_index)
            synthetic_paths.append(str(output))
            source_paths.append(str(source))
            conditions.append(condition)
            self.conditions[condition] = self.conditions.get(condition, 0) + 1
            self.counts["triggered_places"] += 1
        return {
            "synthetic_images": torch.stack(images) if images else torch.empty((0, 3, *self.dataset.image_size)),
            "synthetic_labels": torch.tensor(labels, dtype=torch.long),
            "source_indices": torch.tensor(source_indices, dtype=torch.long),
            "synthetic_paths": synthetic_paths, "source_paths": source_paths,
            "condition": conditions, "pair_valid": torch.ones(len(images), dtype=torch.bool),
        }

    def summary(self):
        c = self.counts
        def ratio(n, d):
            return n / d if d else 0.0
        return {**c, "exposure_probability": self.exposure_prob,
                "actual_trigger_rate": ratio(c["triggered_places"], c["eligible_places"]),
                "trigger_rate_all_places": ratio(c["triggered_places"], c["main_places"]),
                "available_source_fraction": ratio(c["available_sources"], c["main_real_images"]),
                "eligible_place_fraction": ratio(c["eligible_places"], c["main_places"]),
                "synthetic_fraction_observed": ratio(c["triggered_places"], c["main_real_images"] + c["triggered_places"]),
                "condition_exposure": dict(self.conditions)}


def _required_path(value: Any, *, location: str, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: {field_name} must be a non-empty path string")
    return Path(value).expanduser()


def _resolve_manifest_file(
    value: Any, *, manifest: Path, line_number: int, field_name: str,
    real_data: Path,
) -> Path:
    location = f"{manifest}:{line_number}"
    path = _required_path(value, location=location, field_name=field_name)
    if path.is_absolute():
        candidates = [path]
    else:
        bases = [manifest.parent, *manifest.parent.parents, Path.cwd()]
        if field_name == "source_path":
            bases.extend([real_data, real_data / "Images"])
        candidates = [base / path for base in bases]
    existing = sorted({p.resolve() for p in candidates if p.is_file()})
    if not existing:
        raise FileNotFoundError(
            f"{location}: {field_name} does not resolve to an existing file: {value!r}. "
            "Use an absolute path or a path relative to the manifest/project directory."
        )
    if len(existing) > 1:
        choices = ", ".join(str(p) for p in existing)
        raise ValueError(f"{location}: ambiguous {field_name} {value!r}: {choices}")
    return existing[0]


def _metadata_filename(row: dict[str, str], location: str) -> tuple[str, int, str]:
    required = ("place_id", "year", "month", "northdeg", "city_id", "lat", "lon", "panoid")
    if any(not isinstance(row.get(key), str) or not row[key].strip() for key in required):
        raise ValueError(f"{location}: missing required GSV metadata fields: {required}")
    try:
        place_id = int(row["place_id"])
        year = int(row["year"])
        month = int(row["month"])
        northdeg = int(row["northdeg"])
    except ValueError as exc:
        raise ValueError(f"{location}: place_id/year/month/northdeg must be integers") from exc
    if place_id < 0 or year < 0 or not 1 <= month <= 12:
        raise ValueError(f"{location}: invalid place_id, year, or month")
    city = row["city_id"].strip()
    if Path(city).name != city or city in {".", ".."}:
        raise ValueError(f"{location}: invalid city_id {city!r}")
    # Keep coordinate strings verbatim: these strings are part of the released
    # GSV filenames and converting them to floats can change their spelling.
    name = (
        f"{city}_{place_id:07d}_{year:04d}_{month:02d}_{northdeg:03d}_"
        f"{row['lat']}_{row['lon']}_{row['panoid']}.jpg"
    )
    if Path(name).name != name:
        raise ValueError(f"{location}: metadata contains a path separator")
    return city, place_id, name


class MixedGSVCitiesDataset:
    """One item is a place containing K distinct real/synthetic training views.

    ``image_size`` is ``(height, width)``. Images are resized, optionally
    RandAugment-ed with a keyed per-image draw (``workflow.augmentation``), and
    ImageNet normalized. With ``synthetic_mode="mix"``
    (the default), ``synthetic_fraction`` controls ``floor(K * fraction)``
    synthetic slots, capped by availability and ``K-1``. With ``"replace"``, K
    distinct real sources are sampled first; each available source variant is
    independently attempted with that replacement probability. Domains are
    sampled uniformly, then variants uniformly within the selected domain. If
    every slot was replaced, one uniformly chosen source is restored to real.
    A source without an accepted variant keeps its original real view, and a
    source's real and synthetic views cannot coexist in a replacement sample.
    Missing mixed-mode synthetic slots are filled with real views. Places with fewer than
    ``min_images_per_place`` real views are excluded even when augmentations
    exist. ``summary`` describes eligible data and manifest filtering; the bool
    flags returned with each sample measure actual training exposure.

    With ``reliability_pairs=True``, two tensors are appended to each returned
    item: exact real-source companions and a bool validity mask. Only selected
    synthetic views have valid pairs. Real companions reuse the selected image
    tensor. The RandAugment draw is shared within each pair, leaving the
    generator's weather change as the observed difference.

    ``reliability_target_cache`` instead returns cached target/confidence grids
    in those two positions. This mode requires non-augmented paired training,
    validates exact image bytes, and never decodes real-source companions.

    ``random_pk`` sampling uses Python's random module, which PyTorch DataLoader
    seeds per worker. Augmentation never uses it: draws are keyed by seed, epoch,
    place and file name, independent of worker count, slot order and resume.
    Heavy image/tensor imports happen only when an item is read.

    ``coverage_pk`` selects K real source identities from the COMPLETE pool,
    using zero-based epoch, seed, city and place_id. Existing synthetic mix or
    replacement rules then apply; actual real exposure can therefore be < K.
    ``(epoch, index)`` keys carry epoch to workers; direct callers can instead
    use ``set_epoch``. Optional sampling metadata is appended AFTER all normal
    outputs and identifies only real images actually used in the metric batch.

    ``prefer_synthetic_sources`` (coverage_pk only) puts up to K-1 sources that
    have accepted variants into each visit's K sources before the coverage
    order fills the rest. Real-only controls should use the same flag so that
    their real source stream is identical to the synthetic run's.
    """

    def __init__(
        self,
        real_data: Path,
        synthetic_manifest: Path | None = None,
        cities: list[str] | None = None,
        images_per_place: int = 4,
        min_images_per_place: int = 4,
        synthetic_fraction: float = 0.5,
        image_size: tuple[int, int] = (224, 224),
        augment: bool = True,
        synthetic_mode: str = "mix",
        reliability_pairs: bool = False,
        reliability_target_cache: Path | None = None,
        real_sampling: str = "random_pk",
        seed: int = 42,
        return_sampling_metadata: bool = False,
        prefer_synthetic_sources: bool = False,
    ) -> None:
        if images_per_place < 2:
            raise ValueError("images_per_place must be >= 2 to supply positive pairs")
        if min_images_per_place < images_per_place:
            raise ValueError("min_images_per_place must be >= images_per_place")
        if not math.isfinite(synthetic_fraction) or not 0 <= synthetic_fraction <= 1:
            raise ValueError("synthetic_fraction must be between 0 and 1")
        if synthetic_mode not in {"mix", "replace"}:
            raise ValueError("synthetic_mode must be 'mix' or 'replace'")
        if real_sampling not in {"random_pk", "coverage_pk"}:
            raise ValueError("real_sampling must be 'random_pk' or 'coverage_pk'")
        if not isinstance(reliability_pairs, bool):
            raise ValueError("reliability_pairs must be a bool")
        if prefer_synthetic_sources and real_sampling != "coverage_pk":
            raise ValueError("prefer_synthetic_sources requires coverage_pk real sampling")
        if reliability_target_cache is not None and (not reliability_pairs or augment):
            raise ValueError("Static reliability target cache requires reliability_pairs=True and augment=False")
        if len(image_size) != 2 or any(not isinstance(n, int) or n <= 0 for n in image_size):
            raise ValueError("image_size must contain positive integer (height, width)")
        self.real_data = Path(real_data).expanduser().resolve()
        self.images_per_place = images_per_place
        self.min_images_per_place = min_images_per_place
        self.synthetic_fraction = synthetic_fraction
        self.synthetic_mode = synthetic_mode
        self.real_sampling = real_sampling
        self.seed = seed
        self.epoch = 0
        self.return_sampling_metadata = return_sampling_metadata
        self.image_size = tuple(image_size)
        self.augment = augment
        self.prefer_synthetic_sources = prefer_synthetic_sources
        self.reliability_pairs = reliability_pairs
        self.reliability_target_cache = None
        dataframe_dir = self.real_data / "Dataframes"
        image_dir = self.real_data / "Images"
        if not dataframe_dir.is_dir() or not image_dir.is_dir():
            raise FileNotFoundError(
                f"GSV root must contain Dataframes/ and Images/: {self.real_data}"
            )
        available_cities = {path.stem for path in dataframe_dir.glob("*.csv")}
        self.cities = sorted(set(cities)) if cities is not None else sorted(available_cities)
        if not self.cities:
            raise ValueError(f"No selected city CSV files in {dataframe_dir}")
        unknown = set(self.cities) - available_cities
        if unknown:
            raise FileNotFoundError(f"Missing GSV city CSV files: {', '.join(sorted(unknown))}")
        groups: dict[tuple[str, int], PlaceImages] = {}
        source_index: dict[Path, tuple[str, int]] = {}
        duplicate_real_rows = 0
        for city in self.cities:
            csv_path = dataframe_dir / f"{city}.csv"
            with csv_path.open(newline="", encoding="utf-8-sig") as handle:
                for line_number, row in enumerate(csv.DictReader(handle), start=2):
                    row_city, place_id, filename = _metadata_filename(row, f"{csv_path}:{line_number}")
                    if row_city != city:
                        raise ValueError(
                            f"{csv_path}:{line_number}: city_id {row_city!r} differs from CSV city {city!r}"
                        )
                    path = (image_dir / city / filename).resolve()
                    if not path.is_file():
                        raise FileNotFoundError(f"{csv_path}:{line_number}: missing real image: {path}")
                    key = (city, place_id)
                    if path in source_index:
                        if source_index[path] != key:
                            raise ValueError(f"Real image has conflicting place labels: {path}")
                        duplicate_real_rows += 1
                        continue
                    source_index[path] = key
                    groups.setdefault(key, PlaceImages(city, place_id)).real_paths.append(path)
        # Scorers must also recognize real images of places excluded below.
        self.source_index = dict(source_index)
        manifest_counts = {
            "rows": 0, "accepted_rows": 0, "ignored_rows": 0,
            "duplicate_outputs": 0, "excluded_city_rows": 0,
        }
        if synthetic_manifest is not None:
            manifest = Path(synthetic_manifest).expanduser().resolve()
            self._load_synthetic(manifest, groups, source_index, available_cities, manifest_counts)
        self.places = []
        for key in sorted(groups):
            place = groups[key]
            if len(place.real_paths) >= min_images_per_place:
                if real_sampling == "coverage_pk":
                    # Stable across CSV row order and relocation; retain all N views.
                    place.real_paths.sort(key=lambda path: path.name)
                place.label = len(self.places)
                self.places.append(place)
        if not self.places:
            raise ValueError(
                f"No eligible places have at least {min_images_per_place} distinct real images"
            )
        self.summary = {
            "real_data": str(self.real_data),
            "synthetic_manifest": str(Path(synthetic_manifest).expanduser().resolve()) if synthetic_manifest else None,
            "cities": self.cities,
            "num_places": len(self.places),
            "num_real_images": sum(len(p.real_paths) for p in self.places),
            "num_synthetic_images": sum(len(p.synthetic_paths) for p in self.places),
            "places_with_synthetic": sum(bool(p.synthetic_paths) for p in self.places),
            "excluded_places": len(groups) - len(self.places),
            "duplicate_real_rows": duplicate_real_rows,
            "manifest": manifest_counts,
            "images_per_place": images_per_place,
            "min_images_per_place": min_images_per_place,
            "synthetic_fraction": synthetic_fraction,
            "image_size": list(self.image_size),
        }
        if augment:
            from workflow.augmentation import RANDAUGMENT
            self.summary["augmentation"] = {"policy": "randaugment", **RANDAUGMENT}
        if prefer_synthetic_sources:
            self.summary.update(source_selection="synthetic_bearing_sources_first_then_coverage_v1",
                                max_preferred_sources=images_per_place - 1)
        if real_sampling == "coverage_pk":
            self.summary.update(real_sampling=real_sampling, sampling_seed=seed,
                                coverage_algorithm="permutation_boundary_repair_v1")
        if synthetic_mode == "replace":
            self.summary.update({
                "synthetic_mode": "replace",
                "synthetic_fraction_semantics": "per_selected_source_replacement_probability",
                "replacement_probability": synthetic_fraction,
                "replacement_domain_sampling": "uniform_domain_then_uniform_variant",
                "replacement_domain_field": "condition_or_output_path",
                "minimum_real_views_per_sample": 1,
                "max_synthetic_views_per_sample": images_per_place - 1,
                "replacement_real_retention": "restore_one_uniform_source_if_all_replaced",
                "num_sources_with_synthetic": sum(len(p.synthetic_by_source) for p in self.places),
                "num_source_domains": sum(len(domains) for p in self.places
                                          for domains in p.synthetic_by_source.values()),
            })
        if reliability_pairs:
            self.summary.update({
                "reliability_pairs": True,
                "reliability_pair_source": "exact_manifest_source",
                "reliability_pair_augmentation": "shared_randaugment_draw",
            })
        if reliability_target_cache is not None:
            from workflow.reliability_cache import ReliabilityTargetCache, file_sha256
            if synthetic_manifest is None:
                raise ValueError("Static reliability target cache requires a synthetic manifest")
            cache = ReliabilityTargetCache(reliability_target_cache, self.image_size)
            if file_sha256(Path(self.summary["synthetic_manifest"])) != cache.index["manifest_sha256"]:
                raise ValueError("Reliability cache manifest hash differs from the training manifest")
            # Fail before training if the selected dataset is not fully covered.
            for place in self.places:
                for output in place.synthetic_paths:
                    cache.fetch(output, place.source_by_synthetic_path[output])
            self.reliability_target_cache = cache
            self.summary.update({
                "reliability_pair_augmentation": "none_static_targets",
                "reliability_target_cache": str(cache.path),
                "reliability_target_cache_digest": cache.integrity_digest,
                "reliability_target_teacher_sha256": cache.index["teacher_checkpoint_sha256"],
            })

    def _load_synthetic(
        self, manifest: Path, groups: dict[tuple[str, int], PlaceImages],
        source_index: dict[Path, tuple[str, int]], available_cities: set[str],
        counts: dict[str, int],
    ) -> None:
        output_labels: dict[Path, tuple[str, int]] = {}
        output_sources: dict[Path, Path] = {}
        output_domains: dict[Path, str] = {}
        with manifest.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                counts["rows"] += 1
                location = f"{manifest}:{line_number}"
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{location}: malformed JSON: {exc.msg}") from exc
                if not isinstance(entry, dict):
                    raise ValueError(f"{location}: manifest entry must be an object")
                if entry.get("passed") is not True or entry.get("eligible_for_training") is not True:
                    counts["ignored_rows"] += 1
                    continue
                if any(flag in entry and entry[flag] is not True for flag in ("plausible", "weather_ok")):
                    counts["ignored_rows"] += 1
                    continue
                counts["accepted_rows"] += 1
                source = _resolve_manifest_file(
                    entry.get("source_path"), manifest=manifest, line_number=line_number,
                    field_name="source_path", real_data=self.real_data,
                )
                output = _resolve_manifest_file(
                    entry.get("output_path"), manifest=manifest, line_number=line_number,
                    field_name="output_path", real_data=self.real_data,
                )
                key = source_index.get(source)
                if key is None:
                    # A mixed-city manifest can be reused with --cities. Files
                    # from unselected city directories are ignored explicitly.
                    try:
                        relative = source.relative_to(self.real_data / "Images")
                    except ValueError:
                        relative = None
                    if relative is not None and len(relative.parts) == 2 and relative.parts[0] in available_cities - set(self.cities):
                        counts["excluded_city_rows"] += 1
                        continue
                    raise ValueError(
                        f"{location}: source_path is not a selected GSV metadata image: {source}"
                    )
                if output in source_index:
                    raise ValueError(f"{location}: output_path must be a generated image, not a real GSV image: {output}")
                condition = entry.get("condition")
                domain = condition.strip() if isinstance(condition, str) and condition.strip() else str(output)
                if output in output_labels:
                    if output_labels[output] != key:
                        raise ValueError(f"{location}: generated image has conflicting source place labels: {output}")
                    if output_sources[output] != source:
                        raise ValueError(f"{location}: generated image has conflicting exact source views: {output}")
                    if output_domains[output] != domain:
                        raise ValueError(f"{location}: generated image has conflicting condition domains: {output}")
                    counts["duplicate_outputs"] += 1
                    continue
                output_labels[output] = key
                output_sources[output] = source
                output_domains[output] = domain
                groups[key].synthetic_paths.append(output)
                groups[key].synthetic_by_source.setdefault(source, {}).setdefault(domain, []).append(output)
                groups[key].source_by_synthetic_path[output] = source
                groups[key].condition_by_synthetic_path[output] = (
                    condition.strip() if isinstance(condition, str) and condition.strip() else "unknown")

    def __len__(self) -> int:
        return len(self.places)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative (zero-based)")
        self.epoch = epoch

    def sample_paths(self, index: int, *, epoch: int | None = None) -> list[tuple[Path, bool]]:
        """Select paths without decoding images; random_pk keeps its original RNG calls."""
        epoch = self.epoch if epoch is None else epoch
        place = self.places[index]
        k = self.images_per_place
        rng = random
        sources = None
        if self.real_sampling == "coverage_pk":
            sources = [place.real_paths[i] for i in coverage_pk_indices(
                len(place.real_paths), k, seed=self.seed, city=place.city,
                place_id=place.place_id, epoch=epoch)]
            rng = random.Random(_sampling_seed("coverage_variants_v1", self.seed, place.city,
                                               place.place_id, epoch, k))
            if self.prefer_synthetic_sources and place.synthetic_by_source:
                preferred = [path for path in place.real_paths if path in place.synthetic_by_source]
                random.Random(_sampling_seed("preferred_sources_v1", self.seed, place.city,
                                             place.place_id, epoch)).shuffle(preferred)
                preferred = preferred[:k - 1]
                rest = [path for path in sources if path not in preferred]
                rest += [path for path in place.real_paths if path not in preferred and path not in rest]
                sources = preferred + rest[:k - len(preferred)]
        if self.synthetic_mode == "replace":
            sources = sources if sources is not None else rng.sample(place.real_paths, k)
            selected = []
            for source in sources:
                domains = place.synthetic_by_source.get(source)
                if domains and rng.random() < self.synthetic_fraction:
                    domain = rng.choice(sorted(domains))
                    selected.append((rng.choice(domains[domain]), True))
                else:
                    selected.append((source, False))
            if all(is_synthetic for _, is_synthetic in selected):
                retained = rng.randrange(k)
                selected[retained] = (sources[retained], False)
        else:
            num_synthetic = min(
                math.floor(k * self.synthetic_fraction), len(place.synthetic_paths), k - 1,
            )
            real = sources[:k - num_synthetic] if sources is not None else rng.sample(place.real_paths, k - num_synthetic)
            selected = [(path, False) for path in real]
            selected.extend((path, True) for path in rng.sample(place.synthetic_paths, num_synthetic))
        rng.shuffle(selected)
        return selected

    def __getitem__(self, index: int | tuple[int, int]):
        import numpy as np
        import torch
        from PIL import Image
        from workflow.augmentation import augmentation_seed, randaugment

        epoch = self.epoch
        if isinstance(index, tuple):
            epoch, index = index
        place = self.places[index]
        k = self.images_per_place
        selected = self.sample_paths(index, epoch=epoch)
        tensors = []
        kinds = []
        companions = []
        pair_valid = []
        cached_targets = []
        cached_confidence = []
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        height, width = self.image_size
        def read_image(path):
            try:
                with Image.open(path) as original:
                    return original.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"Cannot decode training image {path}: {exc}") from exc

        def normalize(image):
            pixels = np.asarray(image, dtype=np.float32).copy() / 255.0
            tensor = torch.from_numpy(pixels).permute(2, 0, 1)
            return (tensor - mean) / std

        for path, is_synthetic in selected:
            image = read_image(path)
            companion = None
            if self.reliability_pairs and self.reliability_target_cache is None and is_synthetic:
                # The manifest loader rejects ambiguous/conflicting sources.
                # Do not substitute another same-place view for weak supervision.
                companion = read_image(place.source_by_synthetic_path[path])
            if self.augment:
                # Keyed by file, not slot: an image gets the same draw whatever
                # order the mix/replace branch shuffled it into.
                key = augmentation_seed(self.seed, epoch, place.city, place.place_id, path.name)
                if companion is None:
                    image, = randaugment([image], key)
                else:
                    image, companion = randaugment([image, companion], key)
            tensor = normalize(image)
            tensors.append(tensor)
            kinds.append(is_synthetic)
            if self.reliability_pairs:
                if self.reliability_target_cache is not None:
                    if is_synthetic:
                        target, confidence = self.reliability_target_cache.fetch(
                            path, place.source_by_synthetic_path[path])
                    else:
                        target = torch.full((1, height // 14, width // 14), 0.5)
                        confidence = torch.zeros_like(target)
                    cached_targets.append(target)
                    cached_confidence.append(confidence)
                else:
                    companions.append(normalize(companion) if companion is not None else tensor)
                    pair_valid.append(companion is not None)
        result = (
            torch.stack(tensors),
            torch.full((k,), place.label, dtype=torch.long),
            torch.tensor(kinds, dtype=torch.bool),
        )
        if self.reliability_pairs:
            if self.reliability_target_cache is not None:
                result = (*result, torch.stack(cached_targets), torch.stack(cached_confidence))
            else:
                result = (*result, torch.stack(companions), torch.tensor(pair_valid, dtype=torch.bool))
        if self.return_sampling_metadata:
            real_indices = {path: i for i, path in enumerate(place.real_paths)}
            result = (*result, {"place_index": index, "real_indices": torch.tensor(
                [-1 if is_synthetic else real_indices[path] for path, is_synthetic in selected],
                dtype=torch.long)})
        return result
