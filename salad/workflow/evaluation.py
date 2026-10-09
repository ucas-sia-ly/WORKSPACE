"""Evaluation metadata and exact, bounded-memory descriptor retrieval.

This module does not import the legacy dataset loaders: those loaders validate
hardcoded paths at import time. Metadata parsing uses only the standard library;
NumPy/PyTorch/Pillow are imported only by the functions that need them.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


@dataclass(frozen=True)
class EvaluationImage:
    id: str
    path: Path
    source_id: str | None = None


@dataclass
class EvaluationSet:
    name: str
    references: list[EvaluationImage]
    queries: list[EvaluationImage]
    positives: list[list[int]]
    protocol: dict[str, Any] = field(default_factory=dict)

    def validate(self, check_images: bool = True) -> None:
        if not self.references or not self.queries:
            raise ValueError("Evaluation requires nonempty references and queries.")
        if len(self.positives) != len(self.queries):
            raise ValueError("Every query must have a ground-truth entry.")
        for group, images in (("reference", self.references), ("query", self.queries)):
            ids = [image.id for image in images]
            if len(set(ids)) != len(ids):
                raise ValueError(f"Duplicate {group} IDs in evaluation metadata.")
            paths = [image.path.resolve() for image in images]
            if len(set(paths)) != len(paths):
                raise ValueError(f"Duplicate {group} image paths in evaluation metadata.")
        reference_paths = {image.path.resolve() for image in self.references}
        overlap = [image.path for image in self.queries if image.path.resolve() in reference_paths]
        if overlap:
            raise ValueError(f"Reference/query images overlap; this would permit self-matches: {overlap[0]}")
        for query, positives in zip(self.queries, self.positives):
            if any(index < 0 or index >= len(self.references) for index in positives):
                raise ValueError(f"Invalid positive reference index for {query.id}.")
        if not any(self.positives):
            raise ValueError("No queries have valid positives under the selected protocol.")
        if check_images:
            missing = [str(image.path) for image in self.references + self.queries
                       if not image.path.is_file()]
            if missing:
                preview = "\n".join(missing[:5])
                raise FileNotFoundError(f"Missing {len(missing)} evaluation images:\n{preview}")

    def summary(self) -> dict[str, Any]:
        return {
            "dataset": self.name,
            "num_references": len(self.references),
            "num_queries": len(self.queries),
            "num_evaluated_queries": sum(bool(indices) for indices in self.positives),
            "num_queries_without_positives": sum(not indices for indices in self.positives),
            "protocol": self.protocol,
        }


def _image_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def load_manifest(path: Path, root: Path, dataset_name: str = "manifest") -> EvaluationSet:
    """Read explicit reference IDs, positive IDs and optional real source IDs.

    Format: {"references": [{"id": "db1", "path": "db.jpg"}],
             "queries": [{"id": "q1", "path": "q.jpg",
                          "positives": ["db1"], "source_id": "City/source.jpg"}]}.
    Image paths are relative to root, not to the manifest directory. An explicit
    empty positives list is allowed and excluded from the Recall denominator.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Evaluation manifest must contain a JSON object.")

    def parse_images(key: str) -> list[EvaluationImage]:
        rows = data.get(key)
        if not isinstance(rows, list):
            raise ValueError(f"Manifest {key!r} must be a list.")
        result = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"{key}[{index}] must contain id and path fields.")
            image_id, image_path = row.get("id"), row.get("path")
            if not isinstance(image_id, str) or not image_id:
                raise ValueError(f"{key}[{index}] has no nonempty string id.")
            if not isinstance(image_path, str) or not image_path:
                raise ValueError(f"{key}[{index}] has no nonempty string path.")
            source_id = row.get("source_id")
            if source_id is not None:
                if not isinstance(source_id, str) or not source_id:
                    raise ValueError(f"{key}[{index}] has an invalid source_id.")
                source = PurePosixPath(source_id)
                if source.is_absolute() or ".." in source.parts:
                    raise ValueError("source_id must be a relative path within GSV Images/.")
            result.append(EvaluationImage(image_id, _image_path(root, image_path), source_id))
        return result

    references = parse_images("references")
    queries = parse_images("queries")
    reference_indices = {image.id: index for index, image in enumerate(references)}
    if len(reference_indices) != len(references):
        raise ValueError("Duplicate reference IDs in evaluation manifest.")
    positives = []
    for query, row in zip(queries, data["queries"]):
        ids = row.get("positives")
        if not isinstance(ids, list) or any(not isinstance(value, str) for value in ids):
            raise ValueError(f"Query {query.id} must have a list of positive reference IDs.")
        unknown = [value for value in ids if value not in reference_indices]
        if unknown:
            raise ValueError(f"Query {query.id} refers to unknown positives: {unknown[:5]}")
        positives.append(sorted({reference_indices[value] for value in ids}))
    name = data.get("dataset", dataset_name)
    if not isinstance(name, str) or not name:
        raise ValueError("Manifest dataset must be a nonempty string if supplied.")
    result = EvaluationSet(name, references, queries, positives, {
        "ground_truth": "explicit_manifest",
        "manifest": str(path.resolve()),
        "image_root": str(root.resolve()),
    })
    result.validate(check_images=False)
    return result


def _images_in(folder: Path, root: Path) -> list[EvaluationImage]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {folder}")
    paths = sorted(path for path in folder.rglob("*")
                   if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    return [EvaluationImage(path.relative_to(root).as_posix(), path.resolve()) for path in paths]


def _utm(image: EvaluationImage) -> tuple[float, float]:
    fields = image.path.name.split("@")
    try:
        coordinates = float(fields[1]), float(fields[2])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Image filename has no @UTM_Easting@UTM_Northing@: {image.path}") from exc
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError(f"Nonfinite UTM coordinates in {image.path}")
    return coordinates


def coordinate_positives(
    references: Sequence[EvaluationImage], queries: Sequence[EvaluationImage], radius: float,
) -> list[list[int]]:
    """Find exact planar radius neighbors using a standard-library spatial grid."""
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("positive-radius must be a finite positive number.")
    grid: dict[tuple[int, int], list[tuple[int, float, float]]] = defaultdict(list)
    for index, image in enumerate(references):
        x, y = _utm(image)
        grid[(math.floor(x / radius), math.floor(y / radius))].append((index, x, y))
    result = []
    for image in queries:
        x, y = _utm(image)
        cell_x, cell_y = math.floor(x / radius), math.floor(y / radius)
        matches = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for index, ref_x, ref_y in grid.get((cell_x + dx, cell_y + dy), ()):
                    if (ref_x - x) ** 2 + (ref_y - y) ** 2 <= radius ** 2:
                        matches.append(index)
        result.append(sorted(matches))
    return result


def _frame(image: EvaluationImage) -> int:
    # Local Patch-NetVLAD-format data stores frame IDs in the seventh @ field.
    fields = image.path.name.split("@")
    try:
        return int(fields[7])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Nordland filename has no dummy-coordinate frame ID: {image.path}") from exc


def _load_packaged(dataset: str, root: Path, metadata_root: Path) -> EvaluationSet:
    import numpy as np

    key = dataset.lower()
    if key == "nordland":
        directory, stem, ground_truth = "Nordland", "Nordland", "Nordland_gt.npy"
    elif key == "sped":
        directory, stem, ground_truth = "SPED", "SPED", "SPED_gt.npy"
    elif key.startswith("pitts"):
        directory, stem, ground_truth = "Pittsburgh", key, f"{key}_gt.npy"
    elif key == "msls":
        directory, stem, ground_truth = "msls_val", "msls_val", "msls_val_pIdx.npy"
    else:
        raise ValueError(f"No packaged metadata for {dataset}.")
    metadata = metadata_root / directory
    ref_names = np.load(metadata / f"{stem}_dbImages.npy", allow_pickle=False)
    query_names = np.load(metadata / f"{stem}_qImages.npy", allow_pickle=False)
    gt = np.load(metadata / ground_truth, allow_pickle=True)
    if key == "msls":
        query_indices = np.load(metadata / "msls_val_qIdx.npy", allow_pickle=False)
        query_names = query_names[query_indices]
    references = [EvaluationImage(str(value), _image_path(root, str(value))) for value in ref_names]
    queries = [EvaluationImage(str(value), _image_path(root, str(value))) for value in query_names]
    positives = [sorted({int(index) for index in np.asarray(row).reshape(-1)}) for row in gt]
    return EvaluationSet(dataset, references, queries, positives, {
        "ground_truth": "packaged_ground_truth", "metadata_root": str(metadata.resolve()),
    })


def load_evaluation_set(
    dataset: str, root: Path, *, manifest: Path | None = None, split: str = "test",
    query_subdirs: Sequence[str] | None = None, positive_radius: float = 25.0,
    frame_window: int = 10, metadata_root: Path | None = None,
    limit_queries: int | None = None,
) -> EvaluationSet:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    key = dataset.lower()
    default_manifest = root / "evaluation_manifest.json"
    if manifest is not None or default_manifest.is_file():
        result = load_manifest(manifest or default_manifest, root, dataset)
    elif key == "manifest":
        raise ValueError("--dataset manifest requires --eval-manifest or root/evaluation_manifest.json.")
    elif key == "svox":
        split_root = root / "images" / split
        references = _images_in(split_root / "gallery", root)
        folders = query_subdirs or ("queries",)
        queries = []
        for folder in folders:
            part = PurePosixPath(folder)
            if part.is_absolute() or ".." in part.parts:
                raise ValueError("query-subdirs must be relative folders within the dataset split.")
            queries.extend(_images_in(split_root / folder, root))
        result = EvaluationSet(dataset, references, queries,
                               coordinate_positives(references, queries, positive_radius), {
            "ground_truth": "utm_radius", "positive_radius_meters": positive_radius,
            "split": split, "query_subdirs": list(folders),
        })
    elif key == "nordland" and (root / "images" / split / "database").is_dir():
        if frame_window < 0:
            raise ValueError("frame-window must be nonnegative.")
        references = _images_in(root / "images" / split / "database", root)
        queries = _images_in(root / "images" / split / "queries", root)
        reference_frames: dict[int, list[int]] = defaultdict(list)
        for index, image in enumerate(references):
            reference_frames[_frame(image)].append(index)
        positives = []
        for image in queries:
            frame = _frame(image)
            positives.append(sorted(index for offset in range(-frame_window, frame_window + 1)
                                    for index in reference_frames.get(frame + offset, ())))
        result = EvaluationSet(dataset, references, queries, positives, {
            "ground_truth": "frame_window", "positive_frame_window": frame_window, "split": split,
        })
    elif key in {"robotcar", "robotcar-seasons"}:
        raise ValueError(
            "RobotCar-Seasons requires --eval-manifest with explicit reference/query positives. "
            "Its test image list has no ground-truth poses; per-segment COLMAP models cannot be "
            "combined into a global retrieval protocol. No positives will be guessed."
        )
    else:
        bundled = metadata_root or Path(__file__).resolve().parents[1] / "datasets"
        result = _load_packaged(dataset, root, bundled)
    if limit_queries is not None:
        if limit_queries <= 0:
            raise ValueError("limit-queries must be positive.")
        result.queries = result.queries[:limit_queries]
        result.positives = result.positives[:limit_queries]
        result.protocol["query_limit"] = limit_queries
    result.validate()
    return result


def extract_descriptors(model, images: Sequence[EvaluationImage], image_size: tuple[int, int],
                        device: str, batch_size: int = 32, num_workers: int = 4):
    """Extract float32 CPU descriptors with PIL resize and ImageNet normalization."""
    import numpy as np
    from PIL import Image
    import torch
    from torch.utils.data import DataLoader, Dataset

    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers nonnegative.")

    class ImageDataset(Dataset):
        def __len__(self):
            return len(images)

        def __getitem__(self, index):
            path = images[index].path
            with Image.open(path) as image:
                image = image.convert("RGB").resize(
                    (image_size[1], image_size[0]), Image.Resampling.BILINEAR)
                pixels = np.array(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(pixels).permute(2, 0, 1)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            return (tensor - mean) / std

    loader = DataLoader(ImageDataset(), batch_size=batch_size, num_workers=num_workers,
                        shuffle=False, pin_memory=torch.device(device).type == "cuda")
    descriptors = None
    offset = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            output = model(batch.to(device, non_blocking=True))
            if not isinstance(output, torch.Tensor) or output.ndim != 2:
                raise ValueError("Model must return a [batch, descriptor_dim] tensor.")
            if not torch.isfinite(output).all():
                raise ValueError("Model produced nonfinite descriptors.")
            output = output.detach().float().cpu()
            if descriptors is None:
                descriptors = torch.empty((len(images), output.shape[1]), dtype=torch.float32)
            descriptors[offset:offset + len(output)] = output
            offset += len(output)
    if descriptors is None or offset != len(images):
        raise ValueError("Descriptor extraction did not cover every image.")
    return descriptors


def exact_retrieval(
    references, queries, positives: Sequence[Sequence[int]], *, device: str = "cpu",
    query_chunk_size: int = 32, reference_chunk_size: int = 4096,
) -> list[dict[str, Any] | None]:
    """Retrieve nearest references and exact 1-based rank of the first positive.

    Distances are Euclidean. Two bounded-memory passes avoid truncating hard
    cases to top-k. Equal distances are resolved by original reference index.
    Queries with no positives return None and do not enter the recall denominator.
    """
    import torch

    if query_chunk_size <= 0 or reference_chunk_size <= 0:
        raise ValueError("Retrieval chunk sizes must be positive.")
    references = torch.as_tensor(references, dtype=torch.float32).detach().cpu()
    queries = torch.as_tensor(queries, dtype=torch.float32).detach().cpu()
    if references.ndim != 2 or queries.ndim != 2 or references.shape[1] != queries.shape[1]:
        raise ValueError("Reference and query descriptors must have matching dimensions.")
    if len(references) == 0 or len(queries) != len(positives):
        raise ValueError("Invalid reference count or query ground-truth count.")
    if not torch.isfinite(references).all() or not torch.isfinite(queries).all():
        raise ValueError("Descriptors must be finite.")
    for indices in positives:
        if any(index < 0 or index >= len(references) for index in indices):
            raise ValueError("Positive reference index out of range.")

    result: list[dict[str, Any] | None] = []
    for q_start in range(0, len(queries), query_chunk_size):
        q = queries[q_start:q_start + query_chunk_size].to(device)
        q_norm = (q * q).sum(dim=1, keepdim=True)
        gt = positives[q_start:q_start + len(q)]
        best_gt_distance = torch.full((len(q),), float("inf"), device=device)
        best_gt_index = torch.full((len(q),), len(references), dtype=torch.long, device=device)
        predicted_distance = torch.full((len(q),), float("inf"), device=device)
        predicted_index = torch.full((len(q),), len(references), dtype=torch.long, device=device)

        def blocks():
            for r_start in range(0, len(references), reference_chunk_size):
                r = references[r_start:r_start + reference_chunk_size].to(device)
                distances = (q_norm + (r * r).sum(dim=1)[None, :] - 2 * q @ r.T).clamp_min_(0)
                yield r_start, distances

        for r_start, distances in blocks():
            values, indices = distances.min(dim=1)
            better = values < predicted_distance
            predicted_distance = torch.where(better, values, predicted_distance)
            predicted_index = torch.where(better, indices + r_start, predicted_index)
            r_end = r_start + distances.shape[1]
            positive_rows, positive_columns = [], []
            for row, positive_ids in enumerate(gt):
                for index in positive_ids:
                    if r_start <= index < r_end:
                        positive_rows.append(row)
                        positive_columns.append(index - r_start)
            if positive_rows:
                positive_distances = torch.full_like(distances, float("inf"))
                row_ids = torch.tensor(positive_rows, device=device, dtype=torch.long)
                column_ids = torch.tensor(positive_columns, device=device, dtype=torch.long)
                positive_distances[row_ids, column_ids] = distances[row_ids, column_ids]
                values, indices = positive_distances.min(dim=1)
                better = values < best_gt_distance
                best_gt_distance = torch.where(better, values, best_gt_distance)
                best_gt_index = torch.where(better, indices + r_start, best_gt_index)

        ranks = torch.ones(len(q), dtype=torch.long, device=device)
        for r_start, distances in blocks():
            indices = torch.arange(r_start, r_start + distances.shape[1], device=device)
            precedes = ((distances < best_gt_distance[:, None]) |
                        ((distances == best_gt_distance[:, None]) &
                         (indices[None, :] < best_gt_index[:, None])))
            ranks += precedes.sum(dim=1)
        rank_values = ranks.cpu().tolist()
        gt_indices = best_gt_index.cpu().tolist()
        pred_indices = predicted_index.cpu().tolist()
        gt_distances = best_gt_distance.sqrt().cpu().tolist()
        pred_distances = predicted_distance.sqrt().cpu().tolist()
        for row, positive_ids in enumerate(gt):
            result.append({
                "rank": rank_values[row], "ground_truth_index": gt_indices[row],
                "predicted_index": pred_indices[row], "distance_gt": gt_distances[row],
                "distance_pred": pred_distances[row],
            } if positive_ids else None)
    return result


def build_results(dataset: EvaluationSet, retrieval: Sequence[dict[str, Any] | None],
                  save_hard_cases: bool = False) -> dict[str, Any]:
    if len(retrieval) != len(dataset.queries):
        raise ValueError("Retrieval result count does not match query count.")
    valid = [row for row in retrieval if row is not None]
    if not valid:
        raise ValueError("No queries with valid positives to evaluate.")
    result = dataset.summary()
    result["recall"] = {f"R@{k}": sum(row["rank"] <= k for row in valid) / len(valid)
                        for k in (1, 5, 10)}
    result["distance_metric"] = "euclidean"
    result["rank_base"] = 1
    result["tie_break"] = "reference_order"
    result["error_queries"] = []
    result["num_error_queries"] = sum(row["rank"] > 1 for row in valid)
    result["hard_cases_saved"] = save_hard_cases
    if save_hard_cases:
        for query, row, positive_indices in zip(dataset.queries, retrieval, dataset.positives):
            if row is None or row["rank"] <= 1:
                continue
            correct = dataset.references[row["ground_truth_index"]]
            predicted = dataset.references[row["predicted_index"]]
            result["error_queries"].append({
                "query_id": query.id, "query_path": str(query.path),
                "ground_truth": correct.id, "ground_truth_path": str(correct.path),
                "ground_truth_ids": [dataset.references[index].id for index in positive_indices],
                "predicted": predicted.id, "predicted_path": str(predicted.path),
                "rank": row["rank"], "distance_pred": row["distance_pred"],
                "distance_gt": row["distance_gt"], "source_id": query.source_id,
            })
    return result
