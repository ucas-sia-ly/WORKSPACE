"""Portable, explicitly identified BF16 disk-offload deployment of Qwen.

The trusted source machine exports a sealed manifest and communicates its whole
file SHA256 separately through CCR_BUNDLE_SHA256. Original DiT shards are not
deployed: the original prepared-weight manifest remains provenance, and every
deployed file is verified in full instead of pretending those shards exist.
No inference, canvas, LoRA-merge or upstream source implementation is changed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time

ADAPTVPR_ROOT = Path(__file__).resolve().parents[1]
if str(ADAPTVPR_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTVPR_ROOT))

from generation.qwen_disk_weights import MANIFEST, block_file, read_header

KIND = "ccr-qwen-portable-bf16-v1"
IDENTITY = {
    "model_id": "Qwen/Qwen-Image-Edit-2511",
    "model_revision": "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9",
    "lora_id": "lightx2v/Qwen-Image-Edit-2511-Lightning",
    "lora_revision": "d74eba145674fd7e31b949324e148e21e7118abd",
    "lora_filename": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    "lightx2v_commit": "522609ecc121b49c20d201b3f00c3dc052821bce",
}
_VALIDATION = {"status": "not_started"}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _sha(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected a lowercase SHA256")
    return value


def file_sha256(path):
    path = Path(path)
    before = path.stat()
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("Bundle file changed during validation: " + str(path))
    return result.hexdigest()


def _path(root, relative):
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts
            or relative != str(PurePosixPath(relative)) or relative == "."):
        raise ValueError("Unsafe bundle relative path")
    path = root
    for part in PurePosixPath(relative).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Bundle symlinks are not permitted: " + relative)
    if not path.resolve().is_relative_to(root):
        raise ValueError("Bundle path escapes root")
    return path


def _inventory(directory, root):
    if not directory.is_dir():
        raise ValueError("Missing bundle directory: " + str(directory))
    names = set()
    for path in directory.rglob("*"):
        _path(root, path.relative_to(root).as_posix())
        if path.is_file():
            names.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise ValueError("Non-regular bundle entry: " + str(path))
    return names


def validate_bundle(manifest_path, expected_sha256, model_path, lora_path, prepared_path, *, bundle_root=None):
    """Validate deployed bytes and prepared provenance without torch or a GPU.

    This intentionally performs a full read at every service start. There is no
    metadata-only cache that could silently accept modified payload bytes.
    """
    started = time.monotonic()
    manifest_path = Path(manifest_path)
    if manifest_path.is_symlink() or file_sha256(manifest_path) != _sha(expected_sha256):
        raise ValueError("Portable bundle manifest SHA256 mismatch")
    bundle = json.loads(manifest_path.read_text())
    body = {k: v for k, v in bundle.items() if k != "result_sha256"}
    if bundle.get("result_sha256") != fingerprint(body):
        raise ValueError("Portable bundle seal mismatch")
    if bundle.get("kind") != KIND or type(bundle.get("format_version")) is not int or bundle["format_version"] != 1:
        raise ValueError("Unsupported portable bundle format")
    if any(bundle.get(key) != value for key, value in IDENTITY.items()):
        raise ValueError("Portable Qwen model/LoRA/upstream identity mismatch")
    configured_root = bundle_root if bundle_root is not None else os.getenv("CCR_BUNDLE_ROOT")
    if configured_root is not None:
        configured_root = Path(configured_root)
        if not configured_root.is_absolute() or not configured_root.is_dir():
            raise ValueError("CCR_BUNDLE_ROOT must be an existing absolute directory")
        root = configured_root.resolve()
    else:
        root = manifest_path.resolve().parent
    if not manifest_path.resolve().is_relative_to(root):
        raise ValueError("Portable manifest is outside its bundle root")
    paths = bundle.get("paths", {})
    if set(paths) != {"model", "lora", "prepared"}:
        raise ValueError("Portable bundle requires model/lora/prepared paths")
    model, lora, prepared = (_path(root, paths[key]) for key in ("model", "lora", "prepared"))
    if any(Path(actual).resolve() != expected for actual, expected in (
            (model_path, model), (lora_path, lora), (prepared_path, prepared))):
        raise ValueError("Runtime weight paths differ from trusted portable bundle")
    if model == prepared or model in prepared.parents or prepared in model.parents:
        raise ValueError("Model and prepared directories must be separate")
    files = bundle.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Empty portable bundle file inventory")
    required = _inventory(model, root) | _inventory(prepared, root) | {paths["lora"]}
    if not required <= set(files):
        raise ValueError("Unlisted deployed model/prepared file")
    if any((model / "transformer").rglob("*.safetensors")):
        raise ValueError("Portable deployment must not include original DiT payload shards")
    for directory in ("text_encoder", "vae", "tokenizer", "processor"):
        if not (model / directory).is_dir() or not any((model / directory).rglob("*")):
            raise ValueError("Missing model component: " + directory)
    for directory in ("text_encoder", "vae"):
        if not any((model / directory).rglob("*.safetensors")):
            raise ValueError("Missing model weights: " + directory)
    total = 0
    for relative, record in files.items():
        path = _path(root, relative)
        if path == manifest_path.resolve() or not path.is_file():
            raise ValueError("Missing/invalid bundle file: " + relative)
        if (not isinstance(record, dict) or type(record.get("size")) is not int or record["size"] < 0
                or path.stat().st_size != record["size"]
                or file_sha256(path) != _sha(record.get("sha256"))):
            raise ValueError("Bundle file SHA256/size mismatch: " + relative)
        total += record["size"]
    prepared_manifest = prepared / MANIFEST
    if (prepared_manifest.relative_to(root).as_posix() not in files
            or file_sha256(prepared_manifest) != _sha(bundle.get("prepared_manifest_sha256"))):
        raise ValueError("Original prepared manifest SHA256 mismatch")
    original = json.loads(prepared_manifest.read_text())
    identity = original.get("identity", {})
    config_path = model / "transformer/config.json"
    if config_path.relative_to(root).as_posix() not in files:
        raise ValueError("Unlisted transformer configuration")
    config = json.loads(config_path.read_text())
    depth = config.get("num_layers")
    if (original.get("complete") is not True or identity.get("format_version") != 1
            or identity.get("dtype") != "BF16" or identity.get("lora_merged") is not True
            or identity.get("lora_strength") != 1.0 or type(depth) is not int or depth <= 0
            or identity.get("num_layers") != depth
            or identity.get("config_sha256") != file_sha256(config_path)
            or not isinstance(identity.get("sources"), list) or not identity["sources"]):
        raise ValueError("Invalid original prepared BF16/LoRA provenance")
    # The original source records remain unmodified evidence, not local paths.
    for source in identity["sources"]:
        if (not isinstance(source, dict) or not isinstance(source.get("path"), str)
                or type(source.get("size")) is not int or source["size"] <= 0):
            raise ValueError("Invalid original DiT source provenance")
        _sha(source.get("header_sha256"))
    lora_header, lora_header_sha = read_header(lora)
    original_lora = identity.get("lora", {})
    if (not lora_header or original_lora.get("size") != lora.stat().st_size
            or original_lora.get("header_sha256") != lora_header_sha):
        raise ValueError("LoRA differs from the original prepared provenance")
    expected_blocks = {f"block_{index}.safetensors" for index in range(depth)} | {"non_block.safetensors"}
    if set(original.get("files", {})) != expected_blocks:
        raise ValueError("Prepared block inventory differs from transformer depth")
    if {path.name for path in prepared.glob("*.safetensors")} != expected_blocks:
        raise ValueError("Deployed prepared block inventory mismatch")
    keys = set()
    for name in sorted(expected_blocks):
        path = prepared / name
        record = original["files"][name]
        header, header_sha = read_header(path)
        deployed = files[path.relative_to(root).as_posix()]
        if (record.get("size") != deployed["size"] or record.get("sha256") != deployed["sha256"]
                or record.get("header_sha256") != header_sha or record.get("tensor_count") != len(header)
                or not header or keys.intersection(header)
                or any(spec.get("dtype") != "BF16" or block_file(key) != name for key, spec in header.items())):
            raise ValueError("Prepared block payload/header/provenance mismatch: " + name)
        keys.update(header)
    validation = dict(status="verified", manifest_sha256=expected_sha256,
                      prepared_manifest_sha256=bundle["prepared_manifest_sha256"],
                      verified_files=len(files), verified_bytes=total,
                      elapsed_seconds=time.monotonic() - started,
                      source_payload_policy="original_transformer_payload_not_deployed",
                      original_source_identity_preserved=True, lora_already_merged=True,
                      validation_policy="full_sha256_each_service_start")
    return original, validation


def validate_prepared_weights(model_path, lora_path, output):
    global _VALIDATION
    _VALIDATION = {"status": "validating"}
    try:
        manifest = os.environ["CCR_BUNDLE_MANIFEST"]
        expected = os.environ["CCR_BUNDLE_SHA256"]
        original, _VALIDATION = validate_bundle(manifest, expected, model_path, lora_path, output)
        return original
    except Exception as exc:
        _VALIDATION = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        raise


def create_app(adapter=None):
    """Reuse the original adapter's lifespan and generate handler unchanged."""
    if os.getenv("LIGHTX2V_DISK_OFFLOAD", "").lower() not in {"1", "true", "yes"}:
        raise ValueError("Portable remote deployment requires premerged BF16 disk offload")
    profile = os.getenv("LIGHTX2V_MEMORY_PROFILE")
    if profile not in {"disk24g", "disk24g_cpu64", "disk24g_ditcache24", "disk24g_phase24"}:
        raise ValueError("Portable remote deployment requires explicit disk24g, disk24g_cpu64, disk24g_ditcache24 or disk24g_phase24 profile")
    _sha(os.environ["CCR_BUNDLE_SHA256"])
    if not os.environ.get("CCR_BUNDLE_MANIFEST"):
        raise ValueError("CCR_BUNDLE_MANIFEST is required")
    if adapter is None:
        from adapters import lightx2v_qwen_image_edit as adapter
    if any(getattr(adapter, key) != value for key, value in (
            ("MODEL_ID", IDENTITY["model_id"]), ("MODEL_REVISION", IDENTITY["model_revision"]),
            ("LORA_ID", IDENTITY["lora_id"]), ("LORA_REVISION", IDENTITY["lora_revision"]),
            ("LORA_FILENAME", IDENTITY["lora_filename"]), ("LIGHTX2V_COMMIT", IDENTITY["lightx2v_commit"]),
            ("INFER_STEPS", 4), ("GUIDANCE_SCALE", 1.0), ("CANVAS_POLICY", "source_aspect_v1"))):
        raise ValueError("Original adapter inference identity changed")
    adapter.validate_prepared_weights = validate_prepared_weights
    performance = None
    if profile in {"disk24g_cpu64", "disk24g_ditcache24", "disk24g_phase24"}:
        if profile == "disk24g_phase24":
            from generation.qwen_phase_cache import install_gpu_phase_cache, performance_identity
            performance = performance_identity()
        else:
            from generation.qwen_cpu_cache import install_cpu_phase_cache, performance_identity
            performance = performance_identity(profile)
        base_loader = getattr(adapter, "_ccr_original_load_pipeline", adapter.load_pipeline)
        adapter._ccr_original_load_pipeline = base_loader

        def load_pipeline():
            pipe = base_loader()
            if _VALIDATION.get("status") != "verified":
                raise RuntimeError("CPU phase cache requires successful portable bundle validation")
            if profile == "disk24g_phase24":
                install_gpu_phase_cache(pipe, os.environ["LIGHTX2V_DISK_MODEL_PATH"])
            else:
                install_cpu_phase_cache(pipe, os.environ["LIGHTX2V_DISK_MODEL_PATH"],
                                        cache_phases=profile == "disk24g_cpu64")
            return pipe

        adapter.load_pipeline = load_pipeline
    elif hasattr(adapter, "_ccr_original_load_pipeline"):
        adapter.load_pipeline = adapter._ccr_original_load_pipeline
    from fastapi import FastAPI
    app = FastAPI(title="CCR remote Qwen BF16 " + profile, lifespan=adapter.lifespan)

    def deployment():
        # Keep health identity stable before/after validation and model loading.
        return dict(kind=KIND, manifest_sha256=os.environ["CCR_BUNDLE_SHA256"],
                    weight_dtype="BF16", validation="full-file-sha256")

    @app.get("/health")
    def health():
        result = adapter.health()
        result["deployment"] = deployment()
        if performance is not None:
            # Original health hashes qwen_resident.py for its 48 GiB cache;
            # this explicitly distinct profile has its own stable source hash.
            result["performance"] = dict(performance)
        return result

    @app.get("/deployment-validation")
    def deployment_validation():
        return dict(deployment=deployment(), validation=dict(_VALIDATION))

    @app.get("/phase-cache-diagnostics")
    def phase_cache_diagnostics():
        pipe = getattr(getattr(adapter, "state", None), "pipe", None)
        runner = getattr(pipe, "runner", None)
        value = getattr(runner, "ccr_phase_cache_diagnostics", {"status": "not_initialized"})
        return dict(profile=profile, diagnostics=dict(value))

    app.add_api_route("/generate", adapter.generate, methods=["POST"])
    return app


def main():
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=int(os.getenv("LIGHTX2V_PORT", "8001")))


if __name__ == "__main__":
    main()
