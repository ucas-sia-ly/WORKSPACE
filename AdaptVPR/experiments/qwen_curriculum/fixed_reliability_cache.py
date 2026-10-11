"""Frozen raw-DINO targets for every image in the fixed paired experiment.

Unlike the standard accepted-image cache builder, this entry point preserves
the experiment's complete cohort, including automatic quality rejects. Cache
construction is a separate offline stage; training loads only small CPU grids.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import common


IMAGE_SIZE = (224, 224)
PAIR_BATCH_SIZE = 8
TEACHER_MODEL_CONFIG = {
    "model_type": "DINOv2BackboneOnly",
    "backbone_arch": "dinov2_vitb14",
    "backbone_config": {"num_trainable_blocks": 0, "norm_layer": True, "return_token": True},
    "image_size": list(IMAGE_SIZE),
    "initialization": "raw_pretrained_DINOv2_no_SALAD_or_VPR_checkpoint",
}


def _sealed(value):
    return {**value, "fingerprint": common.fingerprint(value)}


def _check_seal(value, label):
    if (not isinstance(value, dict) or value.get("fingerprint") !=
            common.fingerprint({k: v for k, v in value.items() if k != "fingerprint"})):
        raise ValueError(f"Invalid {label} seal")


def _canonical_path(value, label):
    if not isinstance(value, str) or not Path(value).is_absolute() or str(Path(value).resolve()) != value:
        raise ValueError(f"{label} must be a canonical absolute path")
    return Path(value)


def _cache_path(config):
    _check_seal(config, "experiment config")
    value = config.get("reliability", {}).get("target_cache")
    return _canonical_path(value, "Reliability target_cache") if value is not None else None


def _check_directory(path):
    if path.exists() and not path.is_dir():
        raise ValueError(f"Reliability cache is not a directory: {path}")
    if path.exists():
        unexpected = {p.name for p in path.iterdir()} - {"index.json", "targets.pt"}
        if unexpected:
            raise ValueError(f"Unexpected reliability cache files: {', '.join(sorted(unexpected))}")
        if any(not p.is_file() for p in path.iterdir()):
            raise ValueError("Reliability cache artifacts must be regular files")


def _inputs(config, directory):
    manifest = Path(directory).resolve() / "generated_700.jsonl"
    manifest_hash = common.file_sha256(manifest)
    if manifest_hash != config["files_sha256"]["generated_700.jsonl"]:
        raise ValueError("Frozen reliability manifest bytes changed")
    rows = common.read_jsonl(manifest)
    count = config["request"]["num_images"]
    if type(count) is not int or count < 1 or len(rows) != count:
        raise ValueError("Reliability cache must cover every fixed generated source pair")
    sources, outputs, entries = set(), set(), []
    for row in rows:
        source = _canonical_path(row.get("source_path"), "Reliability source_path")
        output = _canonical_path(row.get("output_path"), "Reliability output_path")
        if source == output or source in sources or output in outputs:
            raise ValueError("Reliability fixed source/output pairs must be distinct and unique")
        sources.add(source)
        outputs.add(output)
        entry = {"source_path": str(source), "output_path": str(output)}
        for kind, image in (("source", source), ("output", output)):
            expected = row.get(f"{kind}_sha256")
            if not isinstance(expected, str) or common.file_sha256(image) != expected:
                raise ValueError(f"Frozen reliability {kind} image hash changed: {image}")
            entry[f"{kind}_sha256"] = expected
        entries.append(entry)
    if sources & outputs:
        raise ValueError("Reliability generated outputs overlap the fixed real source images")
    return manifest, manifest_hash, entries


def _validate_index(config, directory, path):
    common.use_salad()
    from workflow.reliability_cache import ReliabilityTargetCache

    manifest, manifest_hash, entries = _inputs(config, directory)
    cache = ReliabilityTargetCache(path, IMAGE_SIZE)
    index = cache.index
    teacher_path = _canonical_path(config["request"]["backbone_weights"], "Raw DINO teacher weights")
    if (index["teacher_checkpoint_sha256"] != config["backbone_weights_sha256"]
            or index.get("teacher_checkpoint") != str(teacher_path)
            or index.get("teacher_model_config") != TEACHER_MODEL_CONFIG):
        raise ValueError("Reliability cache teacher differs from the fixed raw pretrained DINO backbone")
    if index["manifest_sha256"] != manifest_hash or index.get("manifest") != str(manifest):
        raise ValueError("Reliability cache manifest differs from the complete fixed image cohort")
    expected = {entry["output_path"]: entry for entry in entries}
    actual = {entry["output_path"]: entry for entry in index["entries"]}
    if actual != expected:
        raise ValueError("Reliability cache source pairs/hashes differ from the complete fixed cohort")
    for entry in entries:
        cache.fetch(Path(entry["output_path"]), Path(entry["source_path"]))
    return _sealed({
        "format_version": 1,
        "experiment_fingerprint": config["fingerprint"],
        "cache_path": str(path),
        "index_sha256": common.file_sha256(path / "index.json"),
        "tensor_sha256": common.file_sha256(path / "targets.pt"),
        "integrity_digest": cache.integrity_digest,
        "source_pair_count": len(entries),
        "teacher_checkpoint_sha256": config["backbone_weights_sha256"],
        "manifest_sha256": manifest_hash,
    })


def validate_cache(config, directory, *, required=False):
    """Validate CPU grids/provenance, without loading any teacher model.

    A completed index without its experiment binding can be inspected during
    preparation; ``ensure_cache`` repairs that publication window. Training
    requires both artifacts. A dangling or altered binding always fails.
    """
    directory = Path(directory).resolve()
    path = _cache_path(config)
    binding_path = directory / "cache_binding.json"
    if path is None:
        if required or binding_path.exists():
            raise ValueError("This experiment has no configured reliability target cache")
        return None
    _check_directory(path)
    if not (path / "index.json").is_file():
        if binding_path.exists():
            raise ValueError("Reliability cache binding exists without a completed cache index")
        if required:
            raise FileNotFoundError(f"Reliability target cache is not complete: {path}")
        return None
    descriptor = _validate_index(config, directory, path)
    if binding_path.exists():
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        _check_seal(binding, "reliability cache binding")
        if binding != descriptor:
            raise ValueError("Reliability cache binding differs from this experiment/cache")
    elif required:
        raise FileNotFoundError("Completed reliability cache lacks its experiment cache_binding.json")
    return descriptor


def ensure_cache(args, config):
    """Build or reuse the complete fixed cohort with one frozen raw-DINO teacher."""
    directory = Path(args.output_dir).resolve()
    path = _cache_path(config)
    if path is None:
        raise ValueError("Reliability cache construction requires a configured target_cache")
    descriptor = validate_cache(config, directory)
    if descriptor is not None:
        if not (directory / "cache_binding.json").exists():
            common.write_json(directory / "cache_binding.json", descriptor)
        return validate_cache(config, directory, required=True)

    weights = Path(args.backbone_weights).expanduser().resolve()
    if (str(weights) != config["request"]["backbone_weights"]
            or common.file_sha256(weights) != config["backbone_weights_sha256"]):
        raise ValueError("Raw DINO teacher weights differ from the fixed student initialization")
    manifest, manifest_hash, entries = _inputs(config, directory)
    common.use_salad()
    import torch
    from workflow.model import DINOBackbone, atomic_save_checkpoint
    from workflow.reliability import build_local_targets
    from workflow.reliability_cache import IMAGE_PREPROCESSING, TARGET_ALGORITHM, load_teacher_image

    torch.set_num_threads(4)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Reliability cache construction supports only CPU or CUDA")
    teacher = None
    all_targets, all_confidence = [], []
    counts = {"positive_patches": 0, "negative_patches": 0, "unknown_patches": 0}
    try:
        teacher = DINOBackbone(
            "dinov2_vitb14", TEACHER_MODEL_CONFIG["backbone_config"],
            pretrained=True, repo=args.backbone_repo, weights=weights,
        ).to(device).requires_grad_(False).eval()
        with torch.inference_mode():
            for start in range(0, len(entries), PAIR_BATCH_SIZE):
                batch = entries[start:start + PAIR_BATCH_SIZE]
                sources = torch.stack([load_teacher_image(Path(e["source_path"]), IMAGE_SIZE) for e in batch])
                outputs = torch.stack([load_teacher_image(Path(e["output_path"]), IMAGE_SIZE) for e in batch])
                features, _ = teacher(torch.cat((sources, outputs)).to(device))
                target, confidence, diagnostics = build_local_targets(features[:len(batch)], features[len(batch):])
                all_targets.append(target.cpu())
                all_confidence.append(confidence.cpu())
                for key in counts:
                    counts[key] += int(diagnostics[key])
                print(json.dumps({"stage": "reliability_cache", "cached_pairs": start + len(batch),
                                  "total_pairs": len(entries), **counts}), flush=True)
                del sources, outputs, features, target, confidence, diagnostics
        # Publish only if the exact cohort and raw teacher remained unchanged.
        if common.file_sha256(weights) != config["backbone_weights_sha256"]:
            raise ValueError("Raw DINO teacher weights changed during cache construction")
        if _inputs(config, directory) != (manifest, manifest_hash, entries):
            raise ValueError("Frozen reliability cohort changed during cache construction")
        _check_directory(path)
        if (path / "index.json").exists():
            raise FileExistsError("Completed reliability cache must never be overwritten")
        path.mkdir(parents=True, exist_ok=True)
        atomic_save_checkpoint({"targets": torch.cat(all_targets), "confidence": torch.cat(all_confidence)},
                               path / "targets.pt")
        total = sum(counts.values())
        common.write_json(path / "index.json", {
            "format_version": 1, "target_algorithm": TARGET_ALGORITHM,
            "image_preprocessing": IMAGE_PREPROCESSING, "patch_stride": 14,
            "image_size": list(IMAGE_SIZE), "teacher_checkpoint": str(weights),
            "teacher_checkpoint_sha256": config["backbone_weights_sha256"],
            "teacher_model_config": TEACHER_MODEL_CONFIG,
            "manifest": str(manifest), "manifest_sha256": manifest_hash,
            "cities": config["request"].get("cities", ["Bangkok", "BuenosAires", "LosAngeles", "Medellin"]),
            "tensor_file": "targets.pt", "tensor_sha256": common.file_sha256(path / "targets.pt"),
            "entries": entries,
            "summary": {"pairs": len(entries), **counts,
                        "supervised_fraction": (counts["positive_patches"] + counts["negative_patches"]) / total,
                        "cohort": "all_fixed_generated_outputs_including_quality_rejects"},
        })
    finally:
        del teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()
    descriptor = _validate_index(config, directory, path)
    common.write_json(directory / "cache_binding.json", descriptor)
    return validate_cache(config, directory, required=True)
