"""Full native six-domain SVOX evaluation for original SALAD checkpoints.

Each checkpoint gets its own shared gallery extraction. Results are atomic and
resumable only with unchanged weights, input inventory, implementation and
protocol. The final fine-tune and its original pretrained baseline must use the
same protocol fingerprint; historical 224-pixel results are not 322-pixel runs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import gc
import json
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import common


DOMAINS = {
    "day": "queries", "night": "queries_night", "rain": "queries_rain",
    "snow": "queries_snow", "sun": "queries_sun", "overcast": "queries_overcast",
}
COUNTS = {
    "gallery": 17166, "queries": 14278, "queries_night": 823,
    "queries_rain": 937, "queries_snow": 870, "queries_sun": 854,
    "queries_overcast": 872,
}


def _backend():
    common.use_salad()
    from workflow import evaluation, model
    return evaluation, model


def _seal(body):
    return {**body, "record_sha256": common.fingerprint(body)}


def _read_sealed(path):
    row = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(row, dict):
        raise ValueError(f"Evaluation record is not an object: {path}")
    body = {key: value for key, value in row.items() if key != "record_sha256"}
    if row.get("record_sha256") != common.fingerprint(body):
        raise ValueError(f"Evaluation record checksum differs: {path}")
    return row


@contextmanager
def _lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".evaluation.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Evaluation is already running: {directory}") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _original_checkpoint(checkpoint, model_backend):
    state, config = model_backend.checkpoint_state_and_config(
        model_backend.read_checkpoint(checkpoint))
    if config["agg_config"].get("reliability_ot", False) or any(
            key.startswith("aggregator.") and "reliability" in key for key in state):
        raise ValueError("Evaluation requires original SALAD without reliability modules or head weights")
    return config


def _datasets_and_inventory(dataset_root, evaluation_backend):
    if (dataset_root / "evaluation_manifest.json").exists():
        raise ValueError("Native SVOX evaluation cannot use a manifest override")
    datasets = {}
    gallery_keys = None
    inventory = {}
    for domain, folder in DOMAINS.items():
        dataset = evaluation_backend.load_evaluation_set(
            "SVOX", dataset_root, split="test", positive_radius=25.0,
            query_subdirs=[folder])
        expected = {"ground_truth": "utm_radius", "positive_radius_meters": 25.0,
                    "split": "test", "query_subdirs": [folder]}
        if dataset.protocol != expected:
            raise ValueError(f"Requires full native SVOX test/25 m protocol: {domain}")
        if len(dataset.references) != COUNTS["gallery"] or len(dataset.queries) != COUNTS[folder]:
            raise ValueError(f"Native SVOX counts differ for {domain}")
        keys = [(image.id, str(image.path.resolve())) for image in dataset.references]
        if gallery_keys is None:
            gallery_keys = keys
        elif keys != gallery_keys:
            raise ValueError("SVOX domains do not share the same ordered gallery")
        datasets[domain] = dataset
        for name, images in (("gallery", dataset.references), (folder, dataset.queries)):
            records = [(image.id, str(image.path.resolve()), image.path.stat().st_size,
                        image.path.stat().st_mtime_ns) for image in images]
            row = {"count": len(images), "ordered_id_path_size_mtime_sha256": common.fingerprint(records)}
            if name in inventory and inventory[name] != row:
                raise ValueError("Shared gallery inventory changed while loading SVOX domains")
            inventory[name] = row
    return datasets, inventory


def _definition(checkpoint, dataset_root, backbone_repo, inventory, image_size, batch_size,
                num_workers, device):
    implementation = [Path(__file__), common.SALAD_ROOT / "workflow/evaluation.py",
                      common.SALAD_ROOT / "workflow/model.py",
                      common.SALAD_ROOT / "models/aggregators/salad.py", Path(common.__file__)]
    protocol = {
        "version": 1, "dataset": "SVOX", "dataset_root": str(dataset_root),
        "domains": DOMAINS, "native_inventory": inventory, "split": "test",
        "positive_radius_meters": 25.0, "image_size": list(image_size),
        "descriptor_dtype": "float32", "autocast": False, "tf32": False,
        "preprocessing": "PIL RGB BILINEAR resize; ImageNet normalization",
        "distance_metric": "exact euclidean", "tie_break": "reference_order",
        "without_positives": "excluded from Recall denominator",
        "model": "original DINOv2 + SALAD; no reliability modules",
        "batch_size": batch_size, "num_workers": num_workers, "device": device,
        "backbone_repo": str(backbone_repo),
        "backbone_hubconf_sha256": common.file_sha256(backbone_repo / "hubconf.py"),
        "implementation_sha256": {str(path.resolve()): common.file_sha256(path) for path in implementation},
    }
    return _seal({"checkpoint": str(checkpoint), "checkpoint_sha256": common.file_sha256(checkpoint),
                  "protocol": protocol, "protocol_fingerprint": common.fingerprint(protocol)})


def _validate_cached(row, definition, dataset, domain):
    expected = {
        "checkpoint_sha256": definition["checkpoint_sha256"],
        "evaluation_fingerprint": definition["record_sha256"],
        "protocol_fingerprint": definition["protocol_fingerprint"],
        "image_size": definition["protocol"]["image_size"],
        "descriptor_dtype": "float32", "query_folder": DOMAINS[domain],
        **dataset.summary(),
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Cached SVOX results belong to different weights, inventory or protocol: {domain}")
    recall = row.get("recall", {})
    if set(recall) != {"R@1", "R@5", "R@10"} or any(
            not isinstance(value, (int, float)) or not 0 <= value <= 1 for value in recall.values()):
        raise ValueError(f"Cached SVOX Recall metrics are invalid: {domain}")


def _validate_descriptors(descriptors, count, dimension):
    import torch
    if (not isinstance(descriptors, torch.Tensor) or descriptors.shape != (count, dimension)
            or descriptors.dtype != torch.float32 or not torch.isfinite(descriptors).all()):
        raise ValueError("Descriptor count, dimension, FP32 dtype or finiteness differs from ordered inputs")


def evaluate(checkpoint: Path, dataset_root: Path, output_dir: Path, backbone_repo: Path,
             *, batch_size=32, num_workers=0, device="cuda", image_size=(322, 322)) -> dict:
    """Evaluate all six full SVOX subsets, reusing only sealed matching results."""
    checkpoint, dataset_root, output_dir, backbone_repo = [
        Path(path).expanduser().resolve() for path in (checkpoint, dataset_root, output_dir, backbone_repo)]
    image_size = tuple(image_size)
    if (batch_size < 1 or num_workers < 0 or len(image_size) != 2
            or any(type(size) is not int or size < 1 or size % 14 for size in image_size)):
        raise ValueError("Batch size must be positive; workers nonnegative; image dimensions multiples of 14")
    with _lock(output_dir):
        evaluation_backend, model_backend = _backend()
        config = _original_checkpoint(checkpoint, model_backend)
        aggregator = config["agg_config"]
        if (image_size[0] // 14) * (image_size[1] // 14) <= aggregator["num_clusters"]:
            raise ValueError("SALAD evaluation requires more image patches than clusters")
        print(f"[SVOX] Loading six native full test subsets; image_size={image_size}", flush=True)
        datasets, inventory = _datasets_and_inventory(dataset_root, evaluation_backend)
        definition = _definition(checkpoint, dataset_root, backbone_repo, inventory,
                                 image_size, batch_size, num_workers, device)
        config_path = output_dir / "evaluation_config.json"
        if config_path.exists():
            if _read_sealed(config_path) != definition:
                raise ValueError("Evaluation weights, input inventory or frozen protocol changed")
        else:
            common.write_json(config_path, definition)
        results = {}
        for domain, dataset in datasets.items():
            path = output_dir / f"SVOX_{domain}.json"
            if path.exists():
                row = _read_sealed(path)
                _validate_cached(row, definition, dataset, domain)
                results[domain] = row
        if len(results) < len(DOMAINS):
            import torch
            parsed_device = torch.device(device)
            if parsed_device.type not in {"cuda", "cpu"}:
                raise ValueError("SVOX evaluation supports cpu and cuda devices")
            if parsed_device.type == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA evaluation was requested but CUDA is unavailable")
            matmul_tf32, cudnn_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            model = gallery = None
            try:
                model = model_backend.load_checkpoint_model(checkpoint, device=device, backbone_repo=backbone_repo)
                if getattr(model.aggregator, "reliability_ot", False) or any(
                        "reliability" in name for name in model.state_dict()):
                    raise ValueError("Loaded model has reliability modules instead of original SALAD")
                if any(parameter.is_floating_point() and parameter.dtype != torch.float32
                       for parameter in model.parameters()):
                    raise ValueError("Original SALAD evaluation requires FP32 model parameters")
                dimension = aggregator["num_clusters"] * aggregator["cluster_dim"] + aggregator["token_dim"]
                print(f"[SVOX] Extracting shared gallery once: {len(datasets['day'].references)} images", flush=True)
                with torch.autocast(device_type=parsed_device.type, enabled=False):
                    gallery = evaluation_backend.extract_descriptors(
                        model, datasets["day"].references, image_size, device, batch_size, num_workers)
                _validate_descriptors(gallery, len(datasets["day"].references), dimension)
                for domain, dataset in datasets.items():
                    if domain in results:
                        continue
                    print(f"[SVOX/{domain}] Extracting {len(dataset.queries)} queries", flush=True)
                    with torch.autocast(device_type=parsed_device.type, enabled=False):
                        queries = evaluation_backend.extract_descriptors(
                            model, dataset.queries, image_size, device, batch_size, num_workers)
                    _validate_descriptors(queries, len(dataset.queries), dimension)
                    retrieval = evaluation_backend.exact_retrieval(gallery, queries, dataset.positives, device=device)
                    row = _seal({
                        **evaluation_backend.build_results(dataset, retrieval, save_hard_cases=True),
                        "checkpoint": str(checkpoint), "checkpoint_sha256": definition["checkpoint_sha256"],
                        "evaluation_fingerprint": definition["record_sha256"],
                        "protocol_fingerprint": definition["protocol_fingerprint"],
                        "image_size": list(image_size), "descriptor_dtype": "float32",
                        "descriptor_dimension": dimension, "query_folder": DOMAINS[domain],
                    })
                    _validate_cached(row, definition, dataset, domain)
                    common.write_json(output_dir / f"SVOX_{domain}.json", row)
                    results[domain] = row
                    print(f"[SVOX/{domain}] {row['recall']}", flush=True)
                    del queries, retrieval
            finally:
                del model, gallery
                gc.collect()
                if parsed_device.type == "cuda":
                    torch.cuda.empty_cache()
                torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
                torch.backends.cudnn.allow_tf32 = cudnn_tf32
        # Check metadata again before declaring the complete set usable.
        _, final_inventory = _datasets_and_inventory(dataset_root, evaluation_backend)
        if final_inventory != inventory or common.file_sha256(checkpoint) != definition["checkpoint_sha256"]:
            raise ValueError("SVOX inputs or checkpoint changed during evaluation")
        common.write_json(output_dir / "summary.json", _seal({
            "state": "complete", "checkpoint_sha256": definition["checkpoint_sha256"],
            "evaluation_fingerprint": definition["record_sha256"],
            "protocol_fingerprint": definition["protocol_fingerprint"],
            "image_size": list(image_size), "domains": {domain: results[domain]["recall"] for domain in DOMAINS},
        }))
        return {domain: results[domain] for domain in DOMAINS}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=common.WORKSPACE_ROOT / "dataset/svox")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbone-repo", type=Path,
                        default=Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main")
    parser.add_argument("--image-size", type=int, nargs=2, default=[322, 322])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    evaluate(args.checkpoint, args.dataset_root, args.output_dir, args.backbone_repo,
             batch_size=args.batch_size, num_workers=args.num_workers, device=args.device,
             image_size=tuple(args.image_size))


if __name__ == "__main__":
    main()
