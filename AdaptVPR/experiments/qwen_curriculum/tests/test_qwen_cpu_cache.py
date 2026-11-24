"""CPU phase-cache contracts with fake tensors; never imports torch/CUDA."""
from pathlib import Path
import struct
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from generation import qwen_cpu_cache as cache


class Tensor:
    def __init__(self, data, shape=(2, 2), device="cpu", pinned=False):
        self.data = list(data)
        self.shape = tuple(shape)
        self.device = SimpleNamespace(type=device)
        self.dtype = "torch.bfloat16"
        self.pinned = pinned
        self.copies = 0

    def clone(self):
        return Tensor(self.data, self.shape)

    def numel(self):
        size = 1
        for value in self.shape:
            size *= value
        return size

    def element_size(self):
        return 2

    def is_pinned(self):
        return self.pinned

    def t(self):
        if len(self.shape) == 1:
            return self
        h, w = self.shape
        return Tensor([self.data[row * w + col] for col in range(w) for row in range(h)], (w, h))

    def copy_(self, source):
        self.data[:] = source.data
        self.copies += 1
        return self


class Module:
    def __init__(self):
        self.tensor = Tensor([1], shape=(1,))
        self.moves = []

    def parameters(self):
        return [self.tensor]

    def buffers(self):
        return []

    def to(self, device):
        self.moves.append(device)
        self.tensor.device.type = device
        return self


def block(path, index):
    key = f"transformer_blocks.{index}.weight"
    header = json.dumps({key: dict(dtype="BF16", shape=[2, 2], data_offsets=[0, 8])}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 8)


class CPUCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="qwen-cpu-cache-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for i in range(3):
            block(self.root / f"block_{i}.safetensors", i)

    def loader(self, path):
        index = int(Path(path).stem.split("_")[1])
        return {f"transformer_blocks.{index}.weight": Tensor([1, 2, 3, 4])}

    def fixed(self, budget=256):
        return cache.FixedCPUBlocks(self.root, budget, loader=self.loader,
                                    memory_available=lambda: 1_000_000, reserve_bytes=0)

    def test_fixed_prefix_respects_budget_and_never_evicts_for_scan_misses(self):
        first = (self.root / "block_0.safetensors").stat().st_size
        value = self.fixed(first)
        self.assertEqual(list(value.blocks), [0])
        self.assertEqual(value.bytes, 8)
        self.assertLessEqual(value.bytes, first)
        self.assertIsNone(value.blocks.get(2))
        self.assertIsNone(value.blocks.get(1))
        self.assertEqual(list(value.blocks), [0])
        with self.assertRaisesRegex(ValueError, "first complete block"):
            self.fixed(1)

    def test_host_reserve_and_contiguous_inventory_are_required(self):
        with self.assertRaisesRegex(RuntimeError, "host reserve"):
            cache.FixedCPUBlocks(self.root, loader=self.loader, memory_available=lambda: 1)
        (self.root / "block_1.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.fixed()

    def test_pinned_or_wrong_device_cache_is_rejected(self):
        for attributes in (dict(device=SimpleNamespace(type="cuda")), dict(pinned=True)):
            def loader(path):
                result = self.loader(path)
                for tensor in result.values():
                    for name, value in attributes.items():
                        setattr(tensor, name, value)
                return result
            with self.subTest(attributes=attributes), self.assertRaisesRegex(ValueError, "unpinned"):
                cache.FixedCPUBlocks(self.root, 256, loader=loader, memory_available=lambda: 1_000_000,
                                     reserve_bytes=0)

    def test_mmap_context_is_closed_and_cached_tensor_is_an_independent_clone(self):
        original = Tensor([1, 2, 3, 4])
        closed = []

        class Reader:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                closed.append(True)

            def keys(self):
                return ["weight"]

            def get_tensor(self, key):
                return original

        with patch.dict("sys.modules", safetensors=SimpleNamespace(safe_open=lambda *args, **kwargs: Reader())):
            loaded = cache.load_cpu_block("unused")
        self.assertEqual(closed, [True])
        self.assertIsNot(loaded["weight"], original)
        original.data[0] = 99
        self.assertEqual(loaded["weight"].data, [1, 2, 3, 4])

    def buffer(self):
        leaf = SimpleNamespace(base_attrs=[("transformer_blocks.0.weight", "weight", True)],
                               weight_name="transformer_blocks.0.weight",
                               pin_weight=Tensor([0, 0, 0, 0], pinned=True))
        fallback = Mock()
        buffer = SimpleNamespace(_modules=dict(leaf=leaf), _parameters={}, load_state_dict_from_disk=fallback)
        return buffer, leaf, fallback

    def test_manager_buffers_receive_exact_transposed_values_and_disk_fallback(self):
        fixed = self.fixed((self.root / "block_0.safetensors").stat().st_size)
        first, leaf, fallback = self.buffer()
        second, _, _ = self.buffer()
        manager = SimpleNamespace(offload_granularity="block", cpu_buffers=[first, second])
        cache.attach_buffer_cache(manager, fixed)
        first.load_state_dict_from_disk(0)
        self.assertEqual(leaf.pin_weight.data, [1, 3, 2, 4])
        self.assertEqual(leaf.pin_weight.copies, 1)
        self.assertEqual(fixed.blocks[0]["transformer_blocks.0.weight"].data, [1, 2, 3, 4])
        fallback.assert_not_called()
        first.load_state_dict_from_disk(2)
        fallback.assert_called_once_with(2, None)
        self.assertEqual((fixed.hits, fixed.misses), (1, 1))
        with self.assertRaisesRegex(ValueError, "adapter remapping"):
            first.load_state_dict_from_disk(0, 1)

    def test_shape_lora_and_duplicate_install_fail_closed(self):
        first, leaf, _ = self.buffer()
        second, _, _ = self.buffer()
        manager = SimpleNamespace(offload_granularity="block", cpu_buffers=[first, second])
        fixed = self.fixed()
        cache.attach_buffer_cache(manager, fixed)
        with self.assertRaisesRegex(ValueError, "already installed"):
            cache.attach_buffer_cache(manager, fixed)
        leaf.has_lora_branch = True
        with self.assertRaisesRegex(ValueError, "LoRA"):
            first.load_state_dict_from_disk(0)
        leaf.has_lora_branch = False
        leaf.pin_weight.shape = (1, 4)
        with self.assertRaisesRegex(ValueError, "shape"):
            first.load_state_dict_from_disk(0)

    def runner(self):
        encoder_model, vae_model = Module(), Module()
        encoder = SimpleNamespace(cpu_offload=True, text_encoder=encoder_model)

        def infer(value):
            encoder_model.to("cuda")
            encoder_model.to("cpu")
            return value * 2

        encoder.infer = infer
        vae = SimpleNamespace(cpu_offload=True, model=vae_model)

        def phase(value):
            vae_model.to("cuda")
            vae_model.to("cpu")
            return value + 1

        vae.encode_vae_image = phase
        vae.decode = phase
        buffers = [self.buffer()[0], self.buffer()[0]]
        manager = SimpleNamespace(offload_granularity="block", cpu_buffers=buffers)
        config = dict(lazy_load=True, cpu_offload=True, offload_granularity="block", dit_quantized=False,
                      feature_caching="NoCaching", model_path=str(self.root))
        runner = SimpleNamespace(config=config, set_config=lambda values: config.update(values),
                                 load_text_encoder=Mock(return_value=[encoder]), load_vae=Mock(return_value=vae),
                                 load_transformer=Mock(return_value=SimpleNamespace(
                                     transformer_infer=SimpleNamespace(offload_manager=manager))), end_run=Mock())
        return runner, encoder, vae

    def test_encoder_and_vae_reuse_cpu_objects_without_extending_dit_gpu_lifetime(self):
        runner, encoder, vae = self.runner()
        original_encoder, original_vae, end_run = runner.load_text_encoder, runner.load_vae, runner.end_run
        fixed = self.fixed()
        with patch.object(cache, "available_memory", return_value=128 * 2**30), \
                patch.object(cache, "FixedCPUBlocks", return_value=fixed):
            identity = cache.install_cpu_phase_cache(SimpleNamespace(runner=runner), self.root)
        first = runner.load_text_encoder()
        self.assertEqual(first[0].infer(3), 6)
        del first[0]
        second = runner.load_text_encoder()
        self.assertIs(second[0], encoder)
        self.assertEqual(second[0].infer(4), 8)
        original_encoder.assert_called_once()
        self.assertIs(runner.load_vae(), vae)
        self.assertEqual(vae.encode_vae_image(3), 4)
        self.assertEqual(vae.decode(4), 5)
        self.assertIs(runner.load_vae(), vae)
        original_vae.assert_called_once()
        self.assertEqual(encoder.text_encoder.tensor.device.type, "cpu")
        self.assertEqual(vae.model.tensor.device.type, "cpu")
        self.assertIs(runner.end_run, end_run)
        self.assertEqual(identity["profile"], "disk24g_cpu64")
        self.assertEqual(identity["skipped_denoising_steps"], 0)
        self.assertTrue(runner.config["qwen25vl_cpu_offload"])
        self.assertTrue(runner.config["vae_cpu_offload"])

    def test_failed_forward_returns_weights_to_cpu_and_rejects_gpu_at_next_entry(self):
        model = Module()

        def failure():
            model.to("cuda")
            raise RuntimeError("forward failed")

        owner = SimpleNamespace(model=model, infer=failure)
        cache._phase_method(owner, "infer", "model")
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            owner.infer()
        self.assertEqual(model.tensor.device.type, "cpu")
        model.to("cuda")
        with self.assertRaisesRegex(RuntimeError, "not returned"):
            owner.infer()

    def test_dit_only_cache_keeps_original_encoder_vae_config_and_end_run(self):
        runner, _, _ = self.runner()
        original_encoder, original_vae, end_run = runner.load_text_encoder, runner.load_vae, runner.end_run
        initial_config = dict(runner.config)
        with patch.object(cache, "available_memory", return_value=36 * 2**30), \
                patch.object(cache, "FixedCPUBlocks", return_value=self.fixed()):
            identity = cache.install_cpu_phase_cache(SimpleNamespace(runner=runner), self.root, cache_phases=False)
        self.assertIs(runner.load_text_encoder, original_encoder)
        self.assertIs(runner.load_vae, original_vae)
        self.assertIs(runner.end_run, end_run)
        self.assertEqual(runner.config, initial_config)
        self.assertFalse(identity["cache_encoder"])
        self.assertFalse(identity["cache_vae"])
        self.assertEqual(identity["profile"], "disk24g_ditcache24")
        manager = runner.load_transformer().transformer_infer.offload_manager
        self.assertTrue(all(buffer._ccr_cpu_cached for buffer in manager.cpu_buffers))

    def test_profile_rejects_quantization_feature_cache_and_insufficient_ram(self):
        runner, _, _ = self.runner()
        runner.config["dit_quantized"] = True
        with self.assertRaisesRegex(ValueError, "unchanged BF16"):
            cache.install_cpu_phase_cache(SimpleNamespace(runner=runner), self.root)
        runner.config["dit_quantized"] = False
        with patch.object(cache, "available_memory", return_value=16 * 2**30), \
                self.assertRaisesRegex(RuntimeError, "use disk24g"):
            cache.install_cpu_phase_cache(SimpleNamespace(runner=runner), self.root)


if __name__ == "__main__":
    unittest.main()
