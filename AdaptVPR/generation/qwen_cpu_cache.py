"""Optional CPU phase cache for a 24 GiB GPU and a 64 GiB host.

Only lifetime and weight transport change. Fixed BF16 CPU blocks feed the
upstream pinned double buffers; the upstream CUDA buffers and DiT end_run stay
unchanged. Encoder/VAE objects remain on CPU between their original forwards.
This profile needs target-device validation; it makes no throughput promise.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
from types import MethodType, SimpleNamespace

from generation.qwen_disk_weights import read_header

PROFILE = "disk24g_cpu64"
DIT_ONLY_PROFILE = "disk24g_ditcache24"
CPU_DIT_BUDGET_BYTES = 24 * 2**30
HOST_RESERVE_BYTES = 8 * 2**30


def performance_identity(profile=PROFILE):
    if profile not in {PROFILE, DIT_ONLY_PROFILE}:
        raise ValueError("Unknown CPU cache profile")
    return dict(version=1, profile=profile, cpu_dit_budget_bytes=CPU_DIT_BUDGET_BYTES,
                cache_encoder=profile == PROFILE, cache_vae=profile == PROFILE, weight_dtype="BF16",
                skipped_denoising_steps=0,
                implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def available_memory():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("CPU cache requires MemAvailable accounting")


def load_cpu_block(path):
    """Clone one complete block, then close all mappings before returning it."""
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as reader:
        # clone() detaches the resident cache from safetensors mmap/page views.
        result = {key: reader.get_tensor(key).clone() for key in reader.keys()}
    return result


def _block_index(name):
    match = re.fullmatch(r"block_(\d+)\.safetensors", name)
    return int(match[1]) if match else None


class FixedCPUBlocks:
    """An immutable prefix, not a cyclic-scan-thrashing LRU cache."""
    def __init__(self, directory, budget_bytes=CPU_DIT_BUDGET_BYTES, *, loader=load_cpu_block,
                 memory_available=available_memory, reserve_bytes=HOST_RESERVE_BYTES):
        if type(budget_bytes) is not int or budget_bytes <= 0:
            raise ValueError("Positive integer CPU block budget required")
        self.blocks = {}
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        directory = Path(directory)
        paths = sorted((path for path in directory.glob("block_*.safetensors")
                        if _block_index(path.name) is not None), key=lambda p: _block_index(p.name))
        if not paths or [_block_index(path.name) for path in paths] != list(range(len(paths))):
            raise ValueError("CPU cache requires a contiguous prepared block inventory")
        for path in paths:
            header, _ = read_header(path)
            if any(spec["dtype"] != "BF16" for spec in header.values()):
                raise ValueError("CPU cache accepts only prepared BF16 blocks")
            # File size conservatively includes the small header overhead.
            if self.bytes + path.stat().st_size > budget_bytes:
                break
            if memory_available() - path.stat().st_size < reserve_bytes:
                raise RuntimeError("Insufficient host reserve while preparing fixed CPU cache")
            tensors = loader(str(path))
            if set(tensors) != set(header):
                raise ValueError("Loaded CPU block keys differ from verified header")
            actual_bytes = 0
            for key, tensor in tensors.items():
                if (getattr(tensor.device, "type", None) != "cpu" or str(tensor.dtype) != "torch.bfloat16"
                        or list(tensor.shape) != header[key]["shape"] or tensor.is_pinned()):
                    raise ValueError("CPU cache requires unpinned, unchanged BF16 CPU tensors")
                actual_bytes += tensor.numel() * tensor.element_size()
            if actual_bytes > path.stat().st_size or self.bytes + actual_bytes > budget_bytes:
                raise ValueError("CPU block exceeds its declared memory budget")
            self.blocks[_block_index(path.name)] = tensors
            self.bytes += actual_bytes
        if not self.blocks:
            raise ValueError("CPU block budget cannot hold the first complete block")


def _resolve(name, index):
    return re.sub(r"\.\d+", lambda match: "." + str(index), name, count=1)


def copy_to_cpu_buffer(module, tensors, block_index):
    """Fill existing pinned buffers using the upstream mappings/transpositions.

    No buffers are rebound to cached storage, no H2D stream is replaced, and no
    compute operation is changed. copy_ casts to the existing buffer dtype just
    as the upstream norm loaders do after .to(infer_dtype).
    """
    if getattr(module, "has_lora_branch", False) or getattr(module, "has_diff", False):
        raise ValueError("Prepared CPU cache forbids runtime LoRA/diff branches")
    if getattr(module, "is_post_adapter", False):
        raise ValueError("Prepared CPU cache does not support adapter block remapping")
    if hasattr(module, "base_attrs"):
        for name, attribute, transpose in module.base_attrs:
            actual = _resolve(name, block_index)
            value = tensors[actual]
            if transpose:
                value = value.t()
            target = getattr(module, "pin_" + attribute, None)
            if target is None or tuple(target.shape) != tuple(value.shape):
                raise ValueError("Unexpected upstream CPU buffer shape: " + actual)
            target.copy_(value)
            if hasattr(module, attribute + "_name"):
                setattr(module, attribute + "_name", actual)
        return
    if hasattr(module, "tensor_name"):
        actual = _resolve(module.tensor_name, block_index)
        target = getattr(module, "pin_tensor", None)
        if target is None or tuple(target.shape) != tuple(tensors[actual].shape):
            raise ValueError("Unexpected upstream CPU tensor buffer: " + actual)
        target.copy_(tensors[actual])
        module.tensor_name = actual
        return
    for collection in (getattr(module, "_parameters", {}), getattr(module, "_modules", {})):
        for child in collection.values():
            if child is not None:
                copy_to_cpu_buffer(child, tensors, block_index)


def attach_buffer_cache(manager, cache):
    if manager.offload_granularity != "block" or len(manager.cpu_buffers) != 2:
        raise ValueError("CPU cache requires the original two block CPU buffers")
    for buffer in manager.cpu_buffers:
        if getattr(buffer, "_ccr_cpu_cached", False):
            raise ValueError("CPU buffer cache already installed")
        original = buffer.load_state_dict_from_disk

        def load(this, block_index, adapter_block_index=None, *, fallback=original):
            if adapter_block_index is not None:
                raise ValueError("CPU cache forbids runtime adapter remapping")
            tensors = cache.blocks.get(block_index)
            if tensors is None:
                cache.misses += 1
                return fallback(block_index, adapter_block_index)
            cache.hits += 1
            copy_to_cpu_buffer(this, tensors, block_index)

        buffer.load_state_dict_from_disk = MethodType(load, buffer)
        buffer._ccr_cpu_cached = True


def _cpu_only(module):
    for method in ("parameters", "buffers"):
        for tensor in getattr(module, method)():
            if tensor.device.type != "cpu":
                raise RuntimeError("Cached encoder/VAE was not returned to CPU")


def _phase_method(owner, name, model_attribute):
    original = getattr(owner, name)

    def call(*args, **kwargs):
        model = getattr(owner, model_attribute)
        _cpu_only(model)
        try:
            return original(*args, **kwargs)
        finally:
            # Upstream normally already did this; also clean up failed forwards.
            model.to("cpu")
            _cpu_only(model)

    setattr(owner, name, call)


def install_cpu_phase_cache(pipe, prepared_path, *, cache_phases=True):
    if type(cache_phases) is not bool:
        raise ValueError("cache_phases must be an explicit boolean")
    profile = PROFILE if cache_phases else DIT_ONLY_PROFILE
    runner = pipe.runner
    config = runner.config
    if (not config.get("lazy_load") or not config.get("cpu_offload")
            or config.get("offload_granularity") != "block" or config.get("dit_quantized")
            or config.get("feature_caching") != "NoCaching"):
        raise ValueError("CPU phase cache requires unchanged BF16 full-step block disk offload")
    if getattr(runner, "ccr_cpu_phase_cache", None) is not None:
        raise ValueError("CPU phase cache already installed")
    model_path = Path(config["model_path"])
    phase_bytes = (sum(path.stat().st_size for name in ("text_encoder", "vae")
                       for path in (model_path / name).rglob("*.safetensors")) if cache_phases else 0)
    # Reserve room for future CPU encoder weights plus buffers/framework state;
    # the separate cgroup limit must still cap reclaimable filesystem cache.
    if available_memory() < CPU_DIT_BUDGET_BYTES + phase_bytes + HOST_RESERVE_BYTES + 4 * 2**30:
        raise RuntimeError("Insufficient RAM for " + profile + "; use disk24g instead")
    cache = FixedCPUBlocks(prepared_path)
    if cache_phases:
        runner.set_config(dict(qwen25vl_cpu_offload=True, vae_cpu_offload=True))
    saved = SimpleNamespace(blocks=cache, encoders=None, vae=None)
    original_encoders = runner.load_text_encoder
    original_vae = runner.load_vae
    original_transformer = runner.load_transformer

    def load_encoders():
        if saved.encoders is None:
            encoders = original_encoders()
            if len(encoders) != 1 or not encoders[0].cpu_offload:
                raise ValueError("Expected the original CPU-offloaded Qwen encoder")
            _cpu_only(encoders[0].text_encoder)
            _phase_method(encoders[0], "infer", "text_encoder")
            saved.encoders = tuple(encoders)
        return list(saved.encoders)  # Upstream deletes element 0 after encoding.

    def load_vae():
        if saved.vae is None:
            vae = original_vae()
            if not vae.cpu_offload:
                raise ValueError("Expected the original CPU-offloaded VAE")
            _cpu_only(vae.model)
            _phase_method(vae, "encode_vae_image", "model")
            _phase_method(vae, "decode", "model")
            saved.vae = vae
        return saved.vae

    def load_transformer():
        model = original_transformer()
        attach_buffer_cache(model.transformer_infer.offload_manager, cache)
        return model

    if cache_phases:
        runner.load_text_encoder = load_encoders
        runner.load_vae = load_vae
    runner.load_transformer = load_transformer
    runner.ccr_cpu_phase_cache = saved
    # Deliberately retain upstream end_run: no GPU DiT lifetime extension.
    return performance_identity(profile)
