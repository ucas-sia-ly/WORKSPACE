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

    ``image_size`` is ``(height, width)``. Images are resized, optionally flipped
    and color-jittered, and ImageNet normalized. With ``synthetic_mode="mix"``
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
    tensor. Spatial transforms and color-jitter factors are shared within each
    pair, leaving the generator's weather change as the observed difference.

    ``reliability_target_cache`` instead returns cached target/confidence grids
    in those two positions. This mode requires non-augmented paired training,
    validates exact image bytes, and never decodes real-source companions.

    PyTorch DataLoader seeds Python's random module in each worker. Seeding
    ``random`` in the main process also makes direct sampling reproducible.
    Heavy image/tensor imports happen only when an item is read.
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
    ) -> None:
        if images_per_place < 2:
            raise ValueError("images_per_place must be >= 2 to supply positive pairs")
        if min_images_per_place < images_per_place:
            raise ValueError("min_images_per_place must be >= images_per_place")
        if not math.isfinite(synthetic_fraction) or not 0 <= synthetic_fraction <= 1:
            raise ValueError("synthetic_fraction must be between 0 and 1")
        if synthetic_mode not in {"mix", "replace"}:
            raise ValueError("synthetic_mode must be 'mix' or 'replace'")
        if not isinstance(reliability_pairs, bool):
            raise ValueError("reliability_pairs must be a bool")
        if reliability_target_cache is not None and (not reliability_pairs or augment):
            raise ValueError("Static reliability target cache requires reliability_pairs=True and augment=False")
        if len(image_size) != 2 or any(not isinstance(n, int) or n <= 0 for n in image_size):
            raise ValueError("image_size must contain positive integer (height, width)")
        self.real_data = Path(real_data).expanduser().resolve()
        self.images_per_place = images_per_place
        self.min_images_per_place = min_images_per_place
        self.synthetic_fraction = synthetic_fraction
        self.synthetic_mode = synthetic_mode
        self.image_size = tuple(image_size)
        self.augment = augment
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
                "reliability_pair_augmentation": "shared_spatial_and_color_jitter",
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

    def __len__(self) -> int:
        return len(self.places)

    def __getitem__(self, index: int):
        import numpy as np
        import torch
        from PIL import Image, ImageEnhance, ImageOps

        place = self.places[index]
        k = self.images_per_place
        if self.synthetic_mode == "replace":
            sources = random.sample(place.real_paths, k)
            selected = []
            for source in sources:
                domains = place.synthetic_by_source.get(source)
                if domains and random.random() < self.synthetic_fraction:
                    domain = random.choice(sorted(domains))
                    selected.append((random.choice(domains[domain]), True))
                else:
                    selected.append((source, False))
            if all(is_synthetic for _, is_synthetic in selected):
                retained = random.randrange(k)
                selected[retained] = (sources[retained], False)
        else:
            num_synthetic = min(
                math.floor(k * self.synthetic_fraction), len(place.synthetic_paths), k - 1,
            )
            selected = [(path, False) for path in random.sample(place.real_paths, k - num_synthetic)]
            selected.extend((path, True) for path in random.sample(place.synthetic_paths, num_synthetic))
        random.shuffle(selected)
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
                if random.random() < 0.5:
                    image = ImageOps.mirror(image)
                    if companion is not None:
                        companion = ImageOps.mirror(companion)
                if random.random() < 0.5:
                    for enhancement in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
                        factor = random.uniform(0.8, 1.2)
                        image = enhancement(image).enhance(factor)
                        if companion is not None:
                            companion = enhancement(companion).enhance(factor)
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
                return (*result, torch.stack(cached_targets), torch.stack(cached_confidence))
            return (*result, torch.stack(companions), torch.tensor(pair_valid, dtype=torch.bool))
        return result
