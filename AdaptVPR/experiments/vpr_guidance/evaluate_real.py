"""Evaluate a fresh SALAD checkpoint exclusively on real VPR benchmarks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from .data import atomic_json
from .real_data import load_real_split
from .teacher import pil_tensor, file_sha256, salad_hub_refs

DEFAULT_ARCHITECTURE = {
    "backbone_arch": "dinov2_vitb14",
    "backbone_config": {"num_trainable_blocks": 4, "return_token": True, "norm_layer": True},
    "agg_arch": "SALAD",
    "agg_config": {"num_channels": 768, "num_clusters": 64, "cluster_dim": 128, "token_dim": 256},
}


def load_checkpoint(checkpoint, salad_root, device):
    salad_root = Path(salad_root).resolve()
    sys.path.insert(0, str(salad_root))
    from vpr_model import VPRModel
    if Path(sys.modules[VPRModel.__module__].__file__).resolve().parent != salad_root:
        raise RuntimeError("another SALAD checkout shadows --salad-root")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    architecture = dict(DEFAULT_ARCHITECTURE)
    if "state_dict" in payload:
        state = payload["state_dict"]
        hyper = payload.get("hyper_parameters", {})
        architecture.update({k: hyper[k] for k in architecture if k in hyper})
    else:
        state = payload
    with salad_hub_refs():
        model = VPRModel(**architecture)
    model.load_state_dict(state, strict=True)
    return model.requires_grad_(False).eval().to(device), architecture


class EvaluationImages(Dataset):
    def __init__(self, paths, size):
        self.paths, self.size = paths, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            tensor = pil_tensor(image)[0]
        tensor = F.interpolate(tensor[None], size=self.size, mode="bilinear", align_corners=False,
                               antialias=True)[0]
        mean = tensor.new_tensor([.485, .456, .406])[:, None, None]
        std = tensor.new_tensor([.229, .224, .225])[:, None, None]
        return (tensor - mean) / std


def descriptors(model, paths, image_size, batch_size, workers, device):
    loader = DataLoader(EvaluationImages(paths, image_size), batch_size=batch_size,
                        shuffle=False, num_workers=workers, pin_memory=device != "cpu")
    result = None
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            descriptor = F.normalize(model(batch.to(device)).float(), dim=-1)
            if not torch.isfinite(descriptor).all() or (descriptor.norm(dim=-1) < .99).any():
                raise FloatingPointError("invalid real benchmark descriptors")
            values = descriptor.cpu().numpy()
            if result is None:
                result = np.empty((len(paths), values.shape[1]), dtype=np.float32)
            result[offset:offset + len(values)] = values
            offset += len(values)
            print(f"descriptors {offset}/{len(paths)}", flush=True)
    return result


def retrieval_ranks(references, queries, positives, search_batch=32):
    """Exact CPU FlatIP; search all references in small batches for true full ranks."""
    index = faiss.IndexFlatIP(references.shape[1])
    index.add(np.ascontiguousarray(references, dtype=np.float32))
    ranks, top1 = [], []
    for start in range(0, len(queries), search_batch):
        _, order = index.search(np.ascontiguousarray(queries[start:start + search_batch], dtype=np.float32), len(references))
        for j, row in enumerate(order):
            hits = np.flatnonzero(np.isin(row, positives[start + j]))
            if not len(hits):
                raise ValueError("positive reference absent from complete retrieval ranking")
            ranks.append(int(hits[0]) + 1)
            top1.append(int(row[0]))
    return np.asarray(ranks), np.asarray(top1)


def recall_metrics(ranks):
    return {"R@1": float(np.mean(ranks <= 1) * 100),
            "R@5": float(np.mean(ranks <= 5) * 100),
            "R@10": float(np.mean(ranks <= 10) * 100),
            "median_rank": float(np.median(ranks)), "mean_rank": float(np.mean(ranks)),
            "num_queries": len(ranks)}


def official_pose_metrics(split, top1, query_indices):
    """Apply official joint thresholds to the explicitly labeled top-1 pose transfer.

    This is a retrieval-only pose estimator. It does not run local matching/PnP.
    All poses must use the same COLMAP/NVM frame and world->camera quaternion.
    """
    if (not split.reference_poses or not split.query_poses or not all(split.reference_poses)
            or not all(split.query_poses)):
        return {"status": "unavailable", "reason": "metric reference/query poses not provided"}
    translation, rotation = [], []
    for i in query_indices:
        truth, prediction = split.query_poses[i], split.reference_poses[int(top1[i])]
        centers = np.asarray([truth["center_m"], prediction["center_m"]], dtype=float)
        quaternions = np.asarray([truth["quaternion_wxyz"], prediction["quaternion_wxyz"]], dtype=float)
        if centers.shape != (2, 3) or quaternions.shape != (2, 4) or not np.isfinite(centers).all() or not np.isfinite(quaternions).all():
            raise ValueError("invalid metric pose metadata")
        norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
        if (norms <= 0).any():
            raise ValueError("zero pose quaternion")
        quaternions /= norms
        translation.append(np.linalg.norm(centers[0] - centers[1]))
        rotation.append(np.degrees(2 * np.arccos(np.clip(abs(quaternions[0] @ quaternions[1]), 0, 1))))
    translation, rotation = np.asarray(translation), np.asarray(rotation)
    return {"status": "computed", "estimator": "top1_reference_pose_transfer",
            "source": "https://www.visuallocalization.net/benchmark/",
            "localization_accuracy_percent": {
                f"{meters}m_{degrees}deg": float(np.mean((translation <= meters) & (rotation <= degrees)) * 100)
                for meters, degrees in ((.25, 2), (.5, 5), (5., 10))}}


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "salad-root", "dataset-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--dataset", choices=["svox", "robotcar-seasons", "nordland"], required=True)
    p.add_argument("--image-size", type=int, nargs="+", default=[322, 322])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--metadata", type=Path)
    p.add_argument("--reference-dir", type=Path)
    p.add_argument("--query-dirs", type=Path, nargs="+")
    p.add_argument("--positive-radius", type=float, default=25., help="SVOX UTM retrieval radius in meters")
    p.add_argument("--frame-tolerance", type=int, default=10, help="prepared Nordland layout only")
    p.add_argument("--faiss-threads", type=int, default=8)
    return p


def main():
    args = args_parser().parse_args()
    size = args.image_size * 2 if len(args.image_size) == 1 else args.image_size
    if len(size) != 2 or any(s < 126 or s % 14 for s in size):
        raise ValueError("image-size needs one/two dimensions divisible by 14, >=126")
    if args.batch_size <= 0 or args.workers < 0 or args.faiss_threads <= 0:
        raise ValueError("invalid batch-size/workers/faiss-threads")
    split = load_real_split(args)
    model, architecture = load_checkpoint(args.checkpoint, args.salad_root, args.device)
    refs = descriptors(model, split.references, size, args.batch_size, args.workers, args.device)
    queries = descriptors(model, split.queries, size, args.batch_size, args.workers, args.device)
    faiss.omp_set_num_threads(args.faiss_threads)
    ranks, top1 = retrieval_ranks(refs, queries, split.positives)
    condition_metrics = {c: recall_metrics(ranks[np.asarray(split.conditions) == c]) for c in sorted(set(split.conditions))}
    macro = {k: float(np.mean([m[k] for m in condition_metrics.values()]))
             for k in ("R@1", "R@5", "R@10", "median_rank", "mean_rank")}
    fingerprint = hashlib.sha256(json.dumps({"references": [str(p) for p in split.references],
                                              "queries": [str(p) for p in split.queries],
                                              "positives": split.positives, "conditions": split.conditions}, sort_keys=True).encode()).hexdigest()
    result = {"dataset": args.dataset, "checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_sha256": file_sha256(args.checkpoint), "architecture": architecture,
              "protocol": split.protocol, "split_sha256": fingerprint,
              "num_references": len(refs), "num_queries": len(queries),
              "preprocessing": {"image_size": size, "rgb": True, "resize": "tensor_bilinear_antialias",
                                "normalization": "ImageNet", "descriptor_normalization": "L2"},
              "retrieval": {"index": "faiss_cpu_IndexFlatIP", "exact": True,
                            "threads": args.faiss_threads, "rank_base": 1, "recall_unit": "percent"},
              "recall_metrics": {"conditions": condition_metrics, "overall": recall_metrics(ranks), "macro_average": macro}}
    if args.dataset == "robotcar-seasons":
        result["official_metrics"] = {"overall": official_pose_metrics(split, top1, range(len(queries))),
                                      "conditions": {c: official_pose_metrics(split, top1,
                                          np.flatnonzero(np.asarray(split.conditions) == c)) for c in condition_metrics}}
    else:
        result["official_metrics"] = {"status": "not_applicable", "reason": "retrieval protocol"}
    atomic_json(args.output, result)
    atomic_json(args.output.with_suffix(".ranks.json"), {
        "queries": [{"path": str(path), "condition": condition, "first_positive_rank": int(rank),
                     "top1_reference": str(split.references[int(neighbor)])}
                    for path, condition, rank, neighbor in zip(split.queries, split.conditions, ranks, top1)]})
    print(json.dumps(result["recall_metrics"], indent=2))


if __name__ == "__main__":
    main()
