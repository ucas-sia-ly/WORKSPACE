"""Mine difficult real GSV-Cities training views before spending Qwen calls.

Difficulty is the nearest other-place centroid cosine minus the source's
leave-one-out own-place centroid cosine. Nearby place centroids are excluded
from negatives. This is a training-only proxy, not benchmark Recall or a GIFT
reproduction. Descriptor and centroid arrays live in disk-backed NumPy files;
only bounded query/reference chunks are transferred to the scoring device.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import file_sha256, use_salad, write_json, write_jsonl


EARTH_RADIUS_M = 6371008.8
PREPROCESSING = {
    "image_size": "loaded_checkpoint.image_size (height,width)",
    "decode": "PIL RGB", "resize": "PIL BILINEAR",
    "scale": "float32 pixels / 255", "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225], "augmentation": False,
    "descriptor": "L2 normalized float32", "gradient": False,
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def sequence_fingerprint(records):
    """Hash canonical list JSON incrementally; avoid a full-corpus JSON copy."""
    digest = hashlib.sha256(b"[")
    for index, record in enumerate(records):
        if index:
            digest.update(b",")
        digest.update(json.dumps(record, sort_keys=True, ensure_ascii=False,
                                 allow_nan=False, separators=(",", ":")).encode())
    digest.update(b"]")
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("checkpoint", "real-data", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--cities", nargs="+")
    parser.add_argument("--num-sources", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--query-chunk-size", type=int, default=128)
    parser.add_argument("--reference-chunk-size", type=int, default=1024)
    parser.add_argument("--min-positive-similarity", type=float, default=0.15)
    parser.add_argument("--min-hardness-quantile", type=float, default=0.70)
    parser.add_argument("--max-hardness-quantile", type=float, default=0.975)
    parser.add_argument("--max-sources-per-place", type=int, default=2)
    parser.add_argument("--near-negative-radius-m", type=float, default=25.0)
    parser.add_argument("--min-images-per-place", type=int, default=4)
    parser.add_argument("--city-quotas", choices=("even", "proportional"), default="even")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone-repo", type=Path)
    parser.add_argument("--cache-dir", type=Path,
                        help="Reusable descriptor cache; default OUTPUT/cache")
    args = parser.parse_args(argv)
    for name in ("num_sources", "batch_size", "query_chunk_size", "reference_chunk_size",
                 "max_sources_per_place", "min_images_per_place"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.min_images_per_place < 2 or args.num_workers < 0:
        parser.error("At least two real views per place and nonnegative --num-workers required")
    if not math.isfinite(args.min_positive_similarity) or not -1 <= args.min_positive_similarity <= 1:
        parser.error("--min-positive-similarity must be finite and in [-1,1]")
    if not (0 <= args.min_hardness_quantile <= args.max_hardness_quantile <= 1):
        parser.error("Hardness quantiles must satisfy 0 <= min <= max <= 1")
    if not math.isfinite(args.near_negative_radius_m) or args.near_negative_radius_m < 0:
        parser.error("--near-negative-radius-m must be finite and nonnegative")
    for name in ("checkpoint", "real_data", "output_dir", "cache_dir", "backbone_repo"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    args.cities = sorted(set(args.cities)) if args.cities else None
    return args


def load_training_sources(real_data, cities=None, min_images_per_place=4):
    """Resolve exact training CSV filenames/labels; never infer place from NN."""
    use_salad()
    from workflow.training_data import _metadata_filename

    root = Path(real_data).expanduser().resolve()
    available = {path.stem for path in (root / "Dataframes").glob("*.csv")}
    cities = sorted(set(cities)) if cities else sorted(available)
    if not cities or set(cities) - available:
        raise ValueError(f"Unknown or empty GSV training cities: {sorted(set(cities) - available)}")
    if min_images_per_place < 2:
        raise ValueError("Need at least two distinct real views per place")
    grouped, identities = defaultdict(list), {}
    for city in cities:
        metadata = root / "Dataframes" / f"{city}.csv"
        with metadata.open(newline="", encoding="utf-8-sig") as handle:
            for line, row in enumerate(csv.DictReader(handle), 2):
                location = f"{metadata}:{line}"
                row_city, place_id, filename = _metadata_filename(row, location)
                if row_city != city:
                    raise ValueError(f"{location}: city_id differs from selected city")
                try:
                    latitude, longitude = float(row["lat"]), float(row["lon"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{location}: invalid coordinates") from exc
                if (not math.isfinite(latitude) or not math.isfinite(longitude)
                        or not -90 <= latitude <= 90 or not -180 <= longitude <= 180):
                    raise ValueError(f"{location}: invalid coordinates")
                path = (root / "Images" / city / filename).resolve()
                if not path.is_relative_to(root / "Images") or not path.is_file():
                    raise FileNotFoundError(f"{location}: missing/external metadata image: {path}")
                key = (city, place_id)
                if path in identities:
                    if identities[path] != key:
                        raise ValueError(f"{location}: conflicting place labels for {path}")
                    continue
                identities[path] = key
                grouped[key].append({"source_path": str(path), "source_id": f"{city}/{filename}",
                                     "city": city, "place_id": place_id,
                                     "latitude": latitude, "longitude": longitude})
    records, next_place_index = [], 0
    for key in sorted(grouped):
        views = grouped[key]
        if len(views) < min_images_per_place:
            continue
        place_index = next_place_index
        next_place_index += 1
        for view in sorted(views, key=lambda row: row["source_path"]):
            records.append({**view, "place_index": place_index, "source_index": len(records)})
    if not records:
        raise ValueError("No eligible GSV training places have enough distinct real views")
    if len({row["place_index"] for row in records}) < 2:
        raise ValueError("Need at least two eligible training places for other-place negatives")
    return records, cities


def image_inventory(records):
    """Bind every metadata image by path, byte size and nanosecond mtime."""
    digest = hashlib.sha256()
    for record in records:
        path = Path(record["source_path"])
        stat = path.stat()
        digest.update(json.dumps([str(path), stat.st_size, stat.st_mtime_ns],
                                 separators=(",", ":")).encode() + b"\0")
    return {"method": "metadata_image_path_size_mtime", "files": len(records),
            "sha256": digest.hexdigest()}


def descriptor_request(args, records, cities):
    use_salad()
    import workflow.evaluation as evaluation
    import workflow.model as model
    import workflow.training_data as training_data

    payload = {
        "schema_version": 1, "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint), "real_data": str(args.real_data),
        "cities": cities, "min_images_per_place": args.min_images_per_place,
        "metadata_sha256": {city: file_sha256(args.real_data / "Dataframes" / f"{city}.csv")
                            for city in cities},
        "image_inventory": image_inventory(records), "source_order_sha256": sequence_fingerprint(records),
        "preprocessing": PREPROCESSING, "device": args.device, "batch_size": args.batch_size,
        "backbone_repo": str(args.backbone_repo) if args.backbone_repo else None,
        "implementation_sha256": {str(path): file_sha256(path) for path in (
            Path(__file__).resolve(), Path(evaluation.__file__), Path(model.__file__), Path(training_data.__file__))},
    }
    if args.backbone_repo:
        payload["backbone_source_sha256"] = {str(path.relative_to(args.backbone_repo)): file_sha256(path)
            for path in sorted(args.backbone_repo.rglob("*.py")) if ".git" not in path.parts}
    payload["fingerprint"] = fingerprint(payload)
    return payload


def validate_descriptors(descriptors, chunk_size=2048):
    """Check disk-backed vectors in bounded chunks without reading all at once."""
    if (not isinstance(descriptors, np.ndarray) or descriptors.ndim != 2
            or not descriptors.shape[0] or not descriptors.shape[1]
            or not np.issubdtype(descriptors.dtype, np.floating)):
        raise ValueError("Descriptors must be a nonempty floating-point [images,dimension] array")
    for start in range(0, len(descriptors), chunk_size):
        batch = np.asarray(descriptors[start:start + chunk_size], dtype=np.float32)
        norms = np.linalg.norm(batch, axis=1)
        if not np.isfinite(batch).all() or not np.isfinite(norms).all() or np.any(norms <= 1e-12):
            raise ValueError("Descriptors contain nonfinite or zero vectors")


def extract_descriptor_cache(args, records, request, cache_dir, *, model_factory=None):
    """Load the descriptor model once on a cache miss; stream batches to .npy."""
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    array_path, marker_path = cache_dir / "descriptors.npy", cache_dir / "descriptor_complete.json"
    if marker_path.is_file() and array_path.is_file():
        saved = json.loads(marker_path.read_text())
        stat = array_path.stat()
        if (saved.get("config_fingerprint") == request["fingerprint"]
                and saved.get("file_stat") == {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}):
            descriptors = np.load(array_path, mmap_mode="r")
            if list(descriptors.shape) != saved.get("shape") or len(descriptors) != len(records):
                raise ValueError("Descriptor cache has an invalid shape")
            validate_descriptors(descriptors)
            return descriptors, saved
    if model_factory is None:
        use_salad()
        from workflow.model import load_checkpoint_model
        model_factory = load_checkpoint_model
    model = model_factory(args.checkpoint, args.device, backbone_repo=args.backbone_repo)
    model.eval()
    image_size = tuple(model.image_size)

    class RealImages(Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, index):
            with Image.open(records[index]["source_path"]) as image:
                image = image.convert("RGB").resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)
                array = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            mean = torch.tensor(PREPROCESSING["mean"]).view(3, 1, 1)
            std = torch.tensor(PREPROCESSING["std"]).view(3, 1, 1)
            return (tensor - mean) / std

    loader = DataLoader(RealImages(), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=torch.device(args.device).type == "cuda")
    incomplete = cache_dir / "descriptors.incomplete.npy"
    marker_path.unlink(missing_ok=True)
    descriptors, offset = None, 0
    try:
        with torch.inference_mode():
            for batch in loader:
                output = model(batch.to(args.device, non_blocking=True))
                if not isinstance(output, torch.Tensor) or output.ndim != 2 or len(output) != len(batch):
                    raise ValueError("SALAD model must produce a descriptor tensor for each source")
                if not bool(torch.isfinite(output).all()) or bool((output.float().norm(dim=1) <= 1e-12).any()):
                    raise ValueError("SALAD returned nonfinite or zero descriptors")
                output = torch.nn.functional.normalize(output.float(), dim=1).cpu().numpy()
                if descriptors is None:
                    descriptors = np.lib.format.open_memmap(incomplete, mode="w+", dtype=np.float32,
                                                             shape=(len(records), output.shape[1]))
                if output.shape[1] != descriptors.shape[1]:
                    raise ValueError("Descriptor dimensions changed between source batches")
                descriptors[offset:offset + len(output)] = output
                offset += len(output)
                if offset % (args.batch_size * 100) == 0 or offset == len(records):
                    print(f"[mine descriptors] {offset}/{len(records)}", flush=True)
        if descriptors is None or offset != len(records):
            raise ValueError("Descriptor extraction did not cover every source")
        descriptors.flush()
        shape = list(descriptors.shape)
        del descriptors
        with incomplete.open("rb") as handle:
            os.fsync(handle.fileno())
        incomplete.replace(array_path)
        stat = array_path.stat()
        saved = {"config_fingerprint": request["fingerprint"], "shape": shape,
                 "dtype": "float32", "image_size": list(image_size),
                 "file_stat": {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
                 "integrity": "file stat plus bounded finite/nonzero-vector validation"}
        write_json(cache_dir / "descriptor_config.json", request)
        write_json(marker_path, saved)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.load(array_path, mmap_mode="r"), saved


def _validated_labels(place_indices, count):
    labels = np.asarray(place_indices)
    if labels.shape != (count,) or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError("Place indices must be one nonnegative integer per descriptor")
    labels = labels.astype(np.int64, copy=False)
    unique = np.unique(labels)
    if len(unique) < 2 or not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError("Need at least two contiguous, nonempty place labels")
    counts = np.bincount(labels, minlength=len(unique))
    if np.any(counts < 2):
        raise ValueError("Every place needs at least two distinct real views")
    return labels, counts


def build_place_context(descriptors, place_indices, coordinates, cache_dir=None, chunk_size=2048):
    """Return disk-backed vector sums/centroids and spherical place coordinates."""
    validate_descriptors(descriptors, chunk_size)
    labels, counts = _validated_labels(place_indices, len(descriptors))
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if (coordinates.shape != (len(descriptors), 2) or not np.isfinite(coordinates).all()
            or np.any(np.abs(coordinates[:, 0]) > 90) or np.any(np.abs(coordinates[:, 1]) > 180)):
        raise ValueError("Coordinates must be finite [latitude,longitude] per source")
    shape = (len(counts), descriptors.shape[1])
    if cache_dir is None:
        sums = np.zeros(shape, dtype=np.float32)
        centroids = np.empty(shape, dtype=np.float32)
    else:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        sums = np.lib.format.open_memmap(cache_dir / "place_sums.npy", mode="w+", dtype=np.float32, shape=shape)
        centroids = np.lib.format.open_memmap(cache_dir / "place_centroids.npy", mode="w+", dtype=np.float32, shape=shape)
        sums[:] = 0
    position_sums = np.zeros((len(counts), 3), dtype=np.float64)
    for start in range(0, len(descriptors), chunk_size):
        stop = min(start + chunk_size, len(descriptors))
        batch = np.array(descriptors[start:stop], dtype=np.float32)
        batch /= np.linalg.norm(batch, axis=1, keepdims=True)
        np.add.at(sums, labels[start:stop], batch)
        lat, lon = np.radians(coordinates[start:stop]).T
        xyz = np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=1)
        np.add.at(position_sums, labels[start:stop], xyz)
    for start in range(0, len(counts), chunk_size):
        batch = np.array(sums[start:start + chunk_size], dtype=np.float32)
        norms = np.linalg.norm(batch, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
            raise ValueError("A place's mean descriptor is zero/nonfinite; cannot define its centroid")
        centroids[start:start + len(batch)] = batch / norms
    if np.any(np.linalg.norm(position_sums, axis=1) <= 1e-12):
        raise ValueError("A place has ambiguous spherical coordinates")
    place_coords = np.degrees(np.stack((np.arctan2(position_sums[:, 2], np.hypot(position_sums[:, 0], position_sums[:, 1])),
                                       np.arctan2(position_sums[:, 1], position_sums[:, 0])), axis=1))
    if isinstance(sums, np.memmap):
        sums.flush()
        centroids.flush()
    return sums, centroids, place_coords


def score_source_difficulty(descriptors, place_indices, coordinates, *, device="cpu",
                            query_chunk_size=128, reference_chunk_size=1024,
                            near_negative_radius_m=25.0, min_positive_similarity=0.15,
                            cache_dir=None):
    """Score all real sources exactly against centroid negatives in chunks.

    The source itself is subtracted before computing its positive centroid.
    Rank is 1 plus the number of admissible negatives strictly exceeding this
    leave-one-out positive. Invalid positives/no admissible negatives are marked
    unreliable, with finite placeholder similarities rather than fake negatives.
    """
    import torch

    if query_chunk_size <= 0 or reference_chunk_size <= 0:
        raise ValueError("Query and reference chunks must be positive")
    if not math.isfinite(near_negative_radius_m) or near_negative_radius_m < 0:
        raise ValueError("Nearby-negative radius must be finite and nonnegative")
    if not math.isfinite(min_positive_similarity) or not -1 <= min_positive_similarity <= 1:
        raise ValueError("Minimum positive similarity must be finite and in [-1,1]")
    labels, counts = _validated_labels(place_indices, len(descriptors))
    coordinates = np.asarray(coordinates, dtype=np.float64)
    sums, centroids, place_coords = build_place_context(descriptors, labels, coordinates, cache_dir)
    n, p = len(descriptors), len(counts)
    result = {"own_loo_positive_similarity": np.zeros(n, np.float32),
              "hardest_negative_similarity": np.zeros(n, np.float32),
              "hardness": np.zeros(n, np.float32),
              "centroid_positive_rank": np.zeros(n, np.int64),
              "num_negative_places": np.zeros(n, np.int64),
              "reliable": np.zeros(n, bool)}
    with torch.inference_mode():
        for start in range(0, n, query_chunk_size):
            stop = min(start + query_chunk_size, n)
            query = torch.as_tensor(np.array(descriptors[start:stop], dtype=np.float32), device=device)
            query = torch.nn.functional.normalize(query, dim=1)
            own_sums = torch.as_tensor(np.array(sums[labels[start:stop]], dtype=np.float32), device=device)
            loo = own_sums - query
            positive_valid = loo.norm(dim=1) > 1e-12
            positive = (query * torch.nn.functional.normalize(loo, dim=1)).sum(dim=1).clamp(-1, 1)
            labels_tensor = torch.as_tensor(labels[start:stop], device=device)
            query_coords = torch.as_tensor(np.radians(coordinates[start:stop]), dtype=torch.float64, device=device)
            hardest = torch.full((stop - start,), -torch.inf, device=device)
            negative_count = torch.zeros(stop - start, dtype=torch.int64, device=device)
            outranking = torch.zeros_like(negative_count)
            for ref_start in range(0, p, reference_chunk_size):
                ref_stop = min(ref_start + reference_chunk_size, p)
                reference = torch.as_tensor(np.array(centroids[ref_start:ref_stop], dtype=np.float32), device=device)
                similarity = (query @ reference.T).clamp(-1, 1)
                reference_labels = torch.arange(ref_start, ref_stop, device=device)
                valid = labels_tensor[:, None] != reference_labels[None, :]
                if near_negative_radius_m > 0:
                    reference_coords = torch.as_tensor(np.radians(place_coords[ref_start:ref_stop]),
                                                        dtype=torch.float64, device=device)
                    delta = query_coords[:, None] - reference_coords[None, :]
                    a = (torch.sin(delta[:, :, 0] / 2).square()
                         + torch.cos(query_coords[:, 0, None]) * torch.cos(reference_coords[None, :, 0])
                         * torch.sin(delta[:, :, 1] / 2).square()).clamp(0, 1)
                    distances = EARTH_RADIUS_M * 2 * torch.asin(a.sqrt())
                    valid &= distances >= near_negative_radius_m
                negative_count += valid.sum(dim=1)
                outranking += (valid & (similarity > positive[:, None])).sum(dim=1)
                hardest = torch.maximum(hardest, similarity.masked_fill(~valid, -torch.inf).max(dim=1).values)
            has_negative = negative_count > 0
            hardest = torch.where(has_negative, hardest, torch.zeros_like(hardest))
            reliable = positive_valid & has_negative & (positive >= min_positive_similarity)
            for name, value in (("own_loo_positive_similarity", positive),
                                ("hardest_negative_similarity", hardest), ("hardness", hardest - positive),
                                ("centroid_positive_rank", torch.where(has_negative & positive_valid, outranking + 1, 0)),
                                ("num_negative_places", negative_count), ("reliable", reliable)):
                result[name][start:stop] = value.cpu().numpy()
            if stop % (query_chunk_size * 32) == 0 or stop == n:
                print(f"[mine difficulty] {stop}/{n}", flush=True)
    return result


def allocate_city_quotas(capacities, count, weights=None):
    """Allocate exact capped integer quotas, redistributing scarce-city slots."""
    cities = sorted(capacities)
    if count < 1 or any(type(capacities[city]) is not int or capacities[city] < 0 for city in cities):
        raise ValueError("Need positive source count and nonnegative integer city capacities")
    if sum(capacities.values()) < count:
        raise ValueError(f"Only {sum(capacities.values())} eligible source slots remain after the per-place cap; "
                         f"requested {count}. City capacities={capacities}. Reduce --num-sources, widen hardness "
                         "quantiles, lower --min-positive-similarity, or increase --max-sources-per-place.")
    weights = {city: 1.0 for city in cities} if weights is None else weights
    if any(city not in weights or not math.isfinite(weights[city]) or weights[city] <= 0 for city in cities):
        raise ValueError("City quota weights must be finite and positive")
    quotas = {city: 0 for city in cities}
    remaining = count
    while remaining:
        active = [city for city in cities if quotas[city] < capacities[city]]
        total_weight = sum(weights[city] for city in active)
        ideals = {city: remaining * weights[city] / total_weight for city in active}
        saturated = [city for city in active if capacities[city] - quotas[city] <= ideals[city]]
        if saturated:
            for city in saturated:
                increment = capacities[city] - quotas[city]
                quotas[city] += increment
                remaining -= increment
        else:
            for city in active:
                increment = math.floor(ideals[city])
                quotas[city] += increment
                remaining -= increment
            residual_order = sorted(active, key=lambda city: (-(ideals[city] - math.floor(ideals[city])), city))
            for city in residual_order[:remaining]:
                quotas[city] += 1
                remaining -= 1
    return quotas


def select_hard_sources(records, scores, count, *, seed=42, min_quantile=0.70,
                        max_quantile=0.975, max_sources_per_place=2, city_quotas="even"):
    """Choose reproducible, diverse hard views with audited training-city quotas."""
    if not 0 <= min_quantile <= max_quantile <= 1 or max_sources_per_place < 1:
        raise ValueError("Invalid hardness quantile band or per-place cap")
    if city_quotas not in {"even", "proportional"}:
        raise ValueError("city_quotas must be even or proportional")
    required = ("hardness", "own_loo_positive_similarity", "hardest_negative_similarity",
                "centroid_positive_rank", "num_negative_places", "reliable")
    if any(name not in scores or np.asarray(scores[name]).shape != (len(records),) for name in required):
        raise ValueError("Every source requires all difficulty/reliability score fields")
    if any(not np.isfinite(np.asarray(scores[name], dtype=float)).all() for name in required):
        raise ValueError("Source difficulty scores must be finite")
    identities, paths = set(), set()
    cities = sorted({row["city"] for row in records})
    by_city, audit = {}, {}
    for i, row in enumerate(records):
        if type(row.get("place_id")) is not int or row["place_id"] < 0:
            raise ValueError("Source place_id must be a nonnegative integer")
        if row.get("source_path") in paths or (row["city"], row.get("source_id")) in identities:
            raise ValueError("Duplicate source path or identity in training cohort")
        paths.add(row.get("source_path"))
        identities.add((row["city"], row.get("source_id")))
    for city in cities:
        indexes = [i for i, row in enumerate(records) if row["city"] == city]
        reliable = [i for i in indexes if scores["reliable"][i]]
        values = np.asarray(scores["hardness"])[reliable]
        if len(values):
            low, high = map(float, np.quantile(values, [min_quantile, max_quantile]))
            band = [i for i in reliable if low <= float(scores["hardness"][i]) <= high]
        else:
            low, high, band = None, None, []
        random.Random(f"{seed}|{city}").shuffle(band)
        band.sort(key=lambda i: (-float(scores["hardness"][i]), -int(scores["centroid_positive_rank"][i])))
        place_counts, eligible = Counter(), []
        for i in band:
            key = (city, records[i]["place_id"])
            if place_counts[key] < max_sources_per_place:
                eligible.append(i)
                place_counts[key] += 1
        by_city[city] = eligible
        audit[city] = {"training_sources": len(indexes), "reliable_sources": len(reliable),
                       "band_sources": len(band), "capacity_after_place_cap": len(eligible),
                       "hardness_floor": low, "hardness_ceiling": high,
                       "min_quantile": min_quantile, "max_quantile": max_quantile}
    weights = {city: audit[city]["training_sources"] for city in cities} if city_quotas == "proportional" else None
    quotas = allocate_city_quotas({city: len(by_city[city]) for city in cities}, count, weights)
    selected = []
    for city in cities:
        audit[city]["selected_sources"] = quotas[city]
        for i in by_city[city][:quotas[city]]:
            selected.append({**records[i], **{name: np.asarray(scores[name])[i].item() for name in required}})
    return selected, {"cities": audit, "quota_mode": city_quotas, "quotas": quotas,
                      "max_sources_per_place": max_sources_per_place,
                      "hardness_band": "per-city quantiles among reliable real training sources",
                      "tail_exclusion": "configured upper quantile; not a calibrated outlier detector",
                      "seed": seed}


def mine(args):
    records, cities = load_training_sources(args.real_data, args.cities, args.min_images_per_place)
    request = descriptor_request(args, records, cities)
    cache_root = args.cache_dir or args.output_dir / "cache"
    cache_dir = cache_root / request["fingerprint"][:20]
    descriptors, descriptor_metadata = extract_descriptor_cache(args, records, request, cache_dir)
    score_config = {"descriptor_fingerprint": request["fingerprint"], "device": args.device,
                    "query_chunk_size": args.query_chunk_size, "reference_chunk_size": args.reference_chunk_size,
                    "near_negative_radius_m": args.near_negative_radius_m,
                    "min_positive_similarity": args.min_positive_similarity}
    score_config["fingerprint"] = fingerprint(score_config)
    scores_path, scores_marker = cache_dir / "source_scores.npz", cache_dir / "scores_complete.json"
    scores = None
    if scores_path.is_file() and scores_marker.is_file():
        complete = json.loads(scores_marker.read_text())
        if complete.get("config_fingerprint") == score_config["fingerprint"] and complete.get("sha256") == file_sha256(scores_path):
            with np.load(scores_path, allow_pickle=False) as saved:
                scores = {name: saved[name] for name in saved.files}
    if scores is None:
        labels = np.array([row["place_index"] for row in records], dtype=np.int64)
        coordinates = np.array([[row["latitude"], row["longitude"]] for row in records], dtype=np.float64)
        scores = score_source_difficulty(descriptors, labels, coordinates, device=args.device,
                                        query_chunk_size=args.query_chunk_size,
                                        reference_chunk_size=args.reference_chunk_size,
                                        near_negative_radius_m=args.near_negative_radius_m,
                                        min_positive_similarity=args.min_positive_similarity,
                                        cache_dir=cache_dir)
        incomplete = cache_dir / "source_scores.incomplete.npz"
        np.savez(incomplete, **scores)
        incomplete.replace(scores_path)
        write_json(scores_marker, {"config_fingerprint": score_config["fingerprint"], "sha256": file_sha256(scores_path)})
    selected, selection_audit = select_hard_sources(records, scores, args.num_sources, seed=args.seed,
        min_quantile=args.min_hardness_quantile, max_quantile=args.max_hardness_quantile,
        max_sources_per_place=args.max_sources_per_place, city_quotas=args.city_quotas)
    for row in selected:
        row["source_sha256"] = file_sha256(Path(row["source_path"]))
    config = {"descriptor_request": request, "scoring": score_config,
              "selection": {"num_sources": args.num_sources, **selection_audit},
              "benchmark_data_used": False, "paper_attribution": "our training-source difficulty proposal; not GIFT reproduction"}
    config["fingerprint"] = fingerprint(config)
    for row in selected:
        row["mining_fingerprint"] = config["fingerprint"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "mining_config.json", config)
    write_jsonl(args.output_dir / "sources.jsonl", selected)
    summary = {"status": "complete", "config_fingerprint": config["fingerprint"],
               "cohort_fingerprint": fingerprint(selected), "sources_sha256": file_sha256(args.output_dir / "sources.jsonl"),
               "training_sources_scored": len(records), "training_places": len({r["place_index"] for r in records}),
               "selected_sources": len(selected), "selected_places": len({(r["city"], r["place_id"]) for r in selected}),
               "selection_audit": selection_audit, "descriptor_cache": str(cache_dir),
               "descriptor_metadata": descriptor_metadata, "benchmark_data_used": False,
               "difficulty_formula": "nearest_other_admissible_place_centroid_cosine - own_leave_one_out_centroid_cosine",
               "near_negative_radius_m": args.near_negative_radius_m,
               "min_positive_similarity": args.min_positive_similarity,
               "centroid_positive_rank": "1 + count(admissible_negative_cosine > own_loo_positive_cosine)"}
    write_json(args.output_dir / "summary.json", summary)
    print(f"[mine complete] {len(selected)} sources from {summary['selected_places']} places; "
          f"cities={selection_audit['quotas']}", flush=True)
    return summary


def main(argv=None):
    mine(parse_args(argv))


if __name__ == "__main__":
    main()
