"""Run comparable SVOX daytime/night evaluations with one gallery extraction.

The two query protocols share the exact reference image order. Gallery features
are shared only within one checkpoint evaluation; a changed model always gets a
fresh gallery extraction. All descriptor extraction remains FP32 at 224 x 224.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from .evaluation import (build_results, exact_retrieval, extract_descriptors,
                         load_evaluation_set)


def _checkpoint_sha256(checkpoint: Path) -> str:
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_svox_pair(
    checkpoint: Path, dataset_root: Path, output_dir: Path, backbone_repo: Path,
    device: str = "cuda", batch_size: int = 32, num_workers: int = 4,
) -> dict[str, dict]:
    """Write SVOX_results.json and SVOX_night_results.json for one checkpoint.

    Both use the full test gallery, 25 m UTM positives, FP32 224 x 224 ImageNet
    preprocessing and the existing exact rank/distance implementation. Daytime
    uses queries/; nighttime uses queries_night/. Complete Recall@1 failures are
    exported, and checkpoint SHA256 allows before/after provenance checks.
    """
    from .model import load_checkpoint_model

    checkpoint = Path(checkpoint).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    backbone_repo = Path(backbone_repo).expanduser().resolve()
    started = time.perf_counter()

    def progress(message: str) -> None:
        print(f"[SVOX comparison +{time.perf_counter() - started:.1f}s] {message}", flush=True)

    progress(f"Loading test metadata for checkpoint {checkpoint}")
    day = load_evaluation_set("SVOX", dataset_root, split="test", positive_radius=25.0)
    night = load_evaluation_set("SVOX", dataset_root, split="test", positive_radius=25.0,
                                query_subdirs=["queries_night"])
    for dataset, folders in ((day, ["queries"]), (night, ["queries_night"])):
        expected_protocol = {"ground_truth": "utm_radius", "positive_radius_meters": 25.0,
                             "split": "test", "query_subdirs": folders}
        if dataset.protocol != expected_protocol:
            raise ValueError("Comparison requires the native full SVOX test/25 m protocol, "
                             "without an evaluation manifest or query limit.")
    reference_keys = lambda dataset: [(image.id, image.path) for image in dataset.references]
    if reference_keys(day) != reference_keys(night):
        raise ValueError("SVOX daytime/nighttime galleries differ; shared descriptors would be invalid.")
    checkpoint_hash = _checkpoint_sha256(checkpoint)
    progress(f"Loading FP32 model on {device}; checkpoint SHA256={checkpoint_hash}")
    model = load_checkpoint_model(checkpoint, device=device, backbone_repo=backbone_repo)
    image_size = (224, 224)
    if 16 * 16 <= model.config["agg_config"]["num_clusters"]:
        raise ValueError("This checkpoint has too many clusters for the fixed 224 x 224 comparison size.")

    progress(f"Extracting {len(day.references)} shared gallery descriptors once (batch={batch_size})")
    references = extract_descriptors(model, day.references, image_size, device, batch_size, num_workers)
    progress(f"Shared gallery ready, descriptor dimension={references.shape[1]}")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for key, dataset in (("SVOX", day), ("SVOX_night", night)):
        progress(f"Extracting {key}: {len(dataset.queries)} query descriptors")
        queries = extract_descriptors(model, dataset.queries, image_size, device, batch_size, num_workers)
        progress(f"Computing {key} exact positive ranks")
        retrieval = exact_retrieval(references, queries, dataset.positives, device=device)
        result = build_results(dataset, retrieval, save_hard_cases=True)
        result.update({
            "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
            "image_size": list(image_size), "descriptor_dimension": references.shape[1],
            "descriptor_dtype": "float32", "comparison_subset": key,
        })
        path = output_dir / f"{key}_results.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        results[key] = result
        recall = " ".join(f"{metric}={value:.6f}" for metric, value in result["recall"].items())
        progress(f"Saved {path}; {recall}; errors={result['num_error_queries']}")
        del queries, retrieval
    progress("Both SVOX protocols finished")
    return results
