"""GPU-phase lifetime contracts using mocks; no torch/CUDA import or execution."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from generation import qwen_phase_cache as phase, qwen_resident


class CUDA:
    def __init__(self, models):
        self.models = models
        self.events = []
        self.fail_sync = False

    def synchronize(self):
        self.events.append(("sync", any(hasattr(m.transformer_infer, "resident_blocks") for m in self.models)))
        if self.fail_sync:
            raise RuntimeError("sync failed")

    def reset_peak_memory_stats(self):
        self.events.append(("reset", None))

    def memory_allocated(self):
        return 100 + sum(len(getattr(m.transformer_infer, "resident_blocks", [])) for m in self.models)

    def memory_reserved(self):
        return self.memory_allocated() + 20

    def max_memory_allocated(self):
        return 124

    def max_memory_reserved(self):
        return 144

    def empty_cache(self):
        self.events.append(("empty", any(hasattr(m.transformer_infer, "resident_blocks") for m in self.models)))


class GPUPhaseTests(unittest.TestCase):
    def setUp(self):
        self.models = []
        self.cuda = CUDA(self.models)
        self.runner = SimpleNamespace(config={}, load_text_encoder=Mock(), load_vae=Mock(), end_run=Mock())
        self.original_infer = Mock()

        def model():
            result = SimpleNamespace(transformer_infer=SimpleNamespace(infer_func=self.original_infer))
            self.models.append(result)
            return result

        self.runner.load_transformer = Mock(side_effect=model)
        self.forward_error = False

        def run_dit(value="latents"):
            self.runner.model = self.runner.load_transformer()
            self.assertEqual(len(self.runner.model.transformer_infer.resident_blocks), 24)
            self.assertEqual(self.runner.model.transformer_infer.infer_func, "hybrid")
            if self.forward_error:
                raise RuntimeError("denoising failed")
            return value, "generator"

        self.runner.run_dit = run_dit
        self.encoder, self.vae, self.end_run = self.runner.load_text_encoder, self.runner.load_vae, self.runner.end_run
        self.pipe = SimpleNamespace(runner=self.runner)
        self.cpu = patch.object(phase, "install_cpu_phase_cache")
        self.cpu_install = self.cpu.start()
        self.addCleanup(self.cpu.stop)
        self.torch = patch.dict("sys.modules", torch=SimpleNamespace(cuda=self.cuda))
        self.torch.start()
        self.addCleanup(self.torch.stop)
        self.resident = patch.object(qwen_resident, "install_resident_cache", side_effect=self.install_resident)
        self.resident_install = self.resident.start()
        self.addCleanup(self.resident.stop)

    def install_resident(self, pipe, resident_blocks):
        self.assertEqual(resident_blocks, 24)
        proxy = pipe.runner
        self.assertIsNot(proxy, self.runner)
        original = proxy.load_transformer

        def load():
            model = original()
            model.transformer_infer.resident_blocks = [object() for _ in range(resident_blocks)]
            model.transformer_infer.infer_func = "hybrid"
            return model

        proxy.load_transformer = load
        proxy.end_run = Mock()

    def install(self):
        return phase.install_gpu_phase_cache(self.pipe, "/prepared")

    def test_successive_images_release_prefix_before_decoder_and_encoder(self):
        identity = self.install()
        self.cpu_install.assert_called_once_with(self.pipe, "/prepared", cache_phases=False)
        for value in ("first", "second"):
            self.assertEqual(self.runner.run_dit(value), (value, "generator"))
            infer = self.runner.model.transformer_infer
            self.assertFalse(hasattr(infer, "resident_blocks"))
            self.assertIs(infer.infer_func, self.original_infer)
            self.assertFalse(self.runner.ccr_gpu_phase_cache.active)
            self.assertIsNone(self.runner.ccr_gpu_phase_cache.model)
            diagnostic = self.runner.ccr_phase_cache_diagnostics
            self.assertEqual(diagnostic["status"], "complete")
            self.assertFalse(diagnostic["prefix_exists_after_release"])
            self.assertEqual(diagnostic["after_release"]["allocated_bytes"], 100)
            self.assertEqual(diagnostic["max_allocated_bytes"], 124)
            self.assertEqual(diagnostic["max_reserved_bytes"], 144)
        self.assertIs(self.runner.load_text_encoder, self.encoder)
        self.assertIs(self.runner.load_vae, self.vae)
        self.assertIs(self.runner.end_run, self.end_run)
        self.assertEqual(self.cuda.events, [("reset", None), ("sync", True), ("empty", False)] * 2)
        self.assertEqual(identity["profile"], "disk24g_phase24")
        self.assertEqual(identity["gpu_dit_prefix_blocks"], 24)
        self.assertFalse(identity["cache_encoder"])
        self.assertFalse(identity["cache_vae"])

    def test_gpu_prefix_is_forbidden_outside_dit_phase(self):
        self.install()
        with self.assertRaisesRegex(RuntimeError, "protected DiT"):
            self.runner.load_transformer()
        self.resident_install.assert_not_called()

    def test_failed_denoising_still_synchronizes_restores_and_releases(self):
        self.install()
        self.forward_error = True
        with self.assertRaisesRegex(RuntimeError, "denoising failed"):
            self.runner.run_dit()
        self.assertFalse(hasattr(self.runner.model.transformer_infer, "resident_blocks"))
        self.assertIs(self.runner.model.transformer_infer.infer_func, self.original_infer)
        self.assertFalse(self.runner.ccr_gpu_phase_cache.active)
        self.assertEqual(self.runner.ccr_phase_cache_diagnostics["status"], "failed")
        self.assertEqual(self.cuda.events[-2:], [("sync", True), ("empty", False)])

    def test_cleanup_failure_is_visible_and_does_not_hide_forward_failure(self):
        self.install()
        self.cuda.fail_sync = True
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            self.runner.run_dit()
        self.assertFalse(hasattr(self.runner.model.transformer_infer, "resident_blocks"))
        self.assertEqual(self.runner.ccr_phase_cache_diagnostics["status"], "cleanup_failed")
        self.forward_error = True
        with self.assertRaisesRegex(RuntimeError, "denoising failed"):
            self.runner.run_dit()
        self.assertFalse(self.runner.ccr_gpu_phase_cache.active)

    def test_partial_install_failure_clears_frames_and_allows_clean_retry(self):
        freed = []
        class Partial:
            def __del__(self):
                freed.append(True)

        def fail_install(pipe, resident_blocks):
            def failed_load():
                partial = [Partial()]
                raise RuntimeError("less than 6 GiB GPU headroom")
            pipe.runner.load_transformer = failed_load

        self.resident_install.side_effect = fail_install
        self.install()
        with self.assertRaisesRegex(RuntimeError, "6 GiB"):
            self.runner.run_dit()
        self.assertTrue(freed)
        self.assertFalse(self.runner.ccr_gpu_phase_cache.active)
        self.assertFalse(self.runner.ccr_phase_cache_diagnostics["prefix_exists_after_release"])
        self.resident_install.side_effect = self.install_resident
        self.assertEqual(self.runner.run_dit(), ("latents", "generator"))

    def test_duplicate_install_rejected(self):
        self.install()
        with self.assertRaisesRegex(ValueError, "already installed"):
            self.install()


if __name__ == "__main__":
    unittest.main()
