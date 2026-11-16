"""Experimental GPU DiT prefix limited strictly to the denoising phase.

Reuse the frozen resident installer/hybrid infer through a temporary runner
proxy. The real encoder/VAE loaders and real DiT end_run remain untouched.
All streams finish and the prefix is removed before VAE decoding can start.
Target-device equivalence, memory and throughput must be measured separately.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

from generation.qwen_cpu_cache import CPU_DIT_BUDGET_BYTES, install_cpu_phase_cache

PROFILE = "disk24g_phase24"
GPU_PREFIX_BLOCKS = 24


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def performance_identity():
    directory = Path(__file__).resolve().parent
    return dict(version=1, profile=PROFILE, gpu_dit_prefix_blocks=GPU_PREFIX_BLOCKS,
                cpu_dit_budget_bytes=CPU_DIT_BUDGET_BYTES, cache_encoder=False, cache_vae=False,
                weight_dtype="BF16", skipped_denoising_steps=0,
                implementation_sha256=_sha(__file__),
                resident_implementation_sha256=_sha(directory / "qwen_resident.py"),
                cpu_cache_implementation_sha256=_sha(directory / "qwen_cpu_cache.py"))


def install_gpu_phase_cache(pipe, prepared_path):
    import torch
    from generation.qwen_resident import install_resident_cache

    runner = pipe.runner
    if getattr(runner, "ccr_gpu_phase_cache", None) is not None:
        raise ValueError("GPU phase cache already installed")
    install_cpu_phase_cache(pipe, prepared_path, cache_phases=False)
    original_load_transformer = runner.load_transformer
    original_run_dit = runner.run_dit
    state = SimpleNamespace(active=False, model=None, original_infer=None)
    runner.ccr_gpu_phase_cache = state
    runner.ccr_phase_cache_diagnostics = dict(status="not_run", prefix_blocks=GPU_PREFIX_BLOCKS)

    def memory():
        return dict(allocated_bytes=int(torch.cuda.memory_allocated()),
                    reserved_bytes=int(torch.cuda.memory_reserved()))

    def forbidden_loader():
        raise RuntimeError("Phase proxy must never load encoder or VAE")

    def load_transformer():
        if not state.active:
            raise RuntimeError("GPU prefix may only load inside the protected DiT phase")
        model = original_load_transformer()
        infer = model.transformer_infer
        if hasattr(infer, "resident_blocks"):
            raise ValueError("DiT already has a resident prefix")
        state.model = model
        state.original_infer = infer.infer_func
        # install_resident_cache changes this isolated proxy only. Its cached
        # encoder/VAE/end_run functions are never used by the real runner.
        proxy = SimpleNamespace(config=runner.config, load_transformer=lambda: model,
                                load_text_encoder=forbidden_loader, load_vae=forbidden_loader)
        install_resident_cache(SimpleNamespace(runner=proxy), resident_blocks=GPU_PREFIX_BLOCKS)
        try:
            installed = proxy.load_transformer()
        except BaseException as exc:
            # The frozen installer publishes resident_blocks only after every
            # block succeeds. Release partial locals held by a failed frame.
            try:
                torch.cuda.synchronize()
            finally:
                traceback.clear_frames(exc.__traceback__)
            raise
        if installed is not model or len(infer.resident_blocks) != GPU_PREFIX_BLOCKS:
            raise RuntimeError("Frozen resident installer returned an unexpected prefix")
        runner.ccr_phase_cache_diagnostics.update(prefix_installed=True,
                                                  after_install=memory())
        return model

    def run_dit(*args, **kwargs):
        if state.active:
            raise RuntimeError("Concurrent/reentrant DiT phase is not supported")
        state.active = True
        diagnostics = dict(status="running", prefix_blocks=GPU_PREFIX_BLOCKS,
                           prefix_installed=False)
        runner.ccr_phase_cache_diagnostics = diagnostics
        try:
            diagnostics["before_phase"] = memory()
            torch.cuda.reset_peak_memory_stats()
            result = original_run_dit(*args, **kwargs)
            diagnostics["status"] = "complete"
            return result
        except BaseException as exc:
            diagnostics.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            propagating_error = sys.exc_info()[0] is not None
            cleanup_error = None
            try:
                # All compute, prefetch and copy streams must finish before
                # dropping the BF16 containers, even on a failed forward.
                torch.cuda.synchronize()
                diagnostics.update(before_release=memory(),
                                   max_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                                   max_reserved_bytes=int(torch.cuda.max_memory_reserved()))
            except Exception as exc:
                cleanup_error = exc
            finally:
                model = state.model
                if model is not None:
                    infer = model.transformer_infer
                    infer.infer_func = state.original_infer
                    if hasattr(infer, "resident_blocks"):
                        del infer.resident_blocks
                    diagnostics["prefix_exists_after_release"] = hasattr(infer, "resident_blocks")
                else:
                    diagnostics["prefix_exists_after_release"] = False
                state.model = None
                state.original_infer = None
                state.active = False
                try:
                    torch.cuda.empty_cache()
                    diagnostics["after_release"] = memory()
                except Exception as exc:
                    cleanup_error = cleanup_error or exc
            if cleanup_error is not None:
                diagnostics.update(status="cleanup_failed", cleanup_error=str(cleanup_error))
                if not propagating_error:
                    raise RuntimeError("GPU phase cleanup failed") from cleanup_error

    runner.load_transformer = load_transformer
    runner.run_dit = run_dit
    return performance_identity()
