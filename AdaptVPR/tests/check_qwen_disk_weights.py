"""Small real-tensor tests. Run separately from tests that stub numpy/torch."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import torch
from diffusers import QwenImageTransformer2DModel
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generation.qwen_disk_weights import MANIFEST, build_plan, prepare_weights, validate_prepared_weights

UPSTREAM = ROOT.parent / "LightX2V"


class QwenDiskWeightsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.temporary = tempfile.TemporaryDirectory(prefix="qwen-bf16-test-")
        cls.root = Path(cls.temporary.name)
        cls.model = cls.root / "model"
        transformer = cls.model / "transformer"
        transformer.mkdir(parents=True)
        tiny = QwenImageTransformer2DModel(
            num_layers=3, num_attention_heads=2, attention_head_dim=8,
            joint_attention_dim=16, in_channels=16, out_channels=4,
            axes_dims_rope=(2, 2, 4),
        ).to(torch.bfloat16)
        cls.config = dict(tiny.config)
        cls.base = {key: value.clone() for key, value in tiny.state_dict().items()}
        keys = list(cls.base)
        for i in range(2):
            save_file({key: cls.base[key] for key in keys[i::2]}, str(transformer / f"shard_{i}.safetensors"))
        (transformer / "config.json").write_text(json.dumps(cls.config))
        cls.lora = {}
        targets = [f"transformer_blocks.{i}.attn.to_q" for i in range(3)] + ["img_in"]
        for stem in targets:
            shape = cls.base[stem + ".weight"].shape
            cls.lora[stem + ".lora_up.weight"] = torch.randn(shape[0], 2).to(torch.bfloat16) / 16
            cls.lora[stem + ".lora_down.weight"] = torch.randn(2, shape[1]).to(torch.bfloat16) / 16
            cls.lora[stem + ".alpha"] = torch.tensor(4.0)
        cls.lora_path = cls.root / "lightning.safetensors"
        save_file(cls.lora, str(cls.lora_path))
        cls.output = cls.root / "blocks"
        cls.original_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in transformer.glob("*.safetensors")}
        prepare_weights(cls.model, cls.lora_path, cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_bf16_merge_matches_pinned_upstream_and_preserves_sources(self):
        spec = importlib.util.spec_from_file_location("pinned_lora", UPSTREAM / "lightx2v/utils/lora_loader.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        expected = {key: tensor.clone() for key, tensor in self.base.items()}
        count = module.LoRALoader().apply_lora(expected, {k: v.to(torch.bfloat16) for k, v in self.lora.items()}, strength=1.0)
        self.assertEqual(count, 4)
        actual = {}
        for path in self.output.glob("*.safetensors"):
            actual.update(load_file(str(path)))
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            self.assertEqual(actual[key].dtype, torch.bfloat16)
            self.assertTrue(torch.equal(actual[key], expected[key]), key)
        for path, digest in self.original_hashes.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_incomplete_conversion_can_resume_and_repair_corrupt_block(self):
        output = self.root / "resume"
        prepare_weights(self.model, self.lora_path, output)
        keep = output / "block_0.safetensors"
        original_mtime = keep.stat().st_mtime_ns
        manifest = json.loads((output / MANIFEST).read_text())
        manifest["complete"] = False
        (output / MANIFEST).write_text(json.dumps(manifest))
        (output / "block_1.safetensors").write_bytes(b"interrupted")
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            validate_prepared_weights(self.model, self.lora_path, output)
        prepare_weights(self.model, self.lora_path, output)
        self.assertEqual(keep.stat().st_mtime_ns, original_mtime)
        self.assertTrue(validate_prepared_weights(self.model, self.lora_path, output)["complete"])

    def test_unmatched_lora_is_rejected(self):
        lora = dict(self.lora)
        lora["unexpected.tensor"] = torch.ones(1)
        path = self.root / "bad_lora.safetensors"
        save_file(lora, str(path))
        with self.assertRaisesRegex(ValueError, "unused LoRA"):
            build_plan(self.model, path)

    def test_source_changes_invalidate_prepared_weights(self):
        path = self.model / "transformer/shard_0.safetensors"
        stat = path.stat()
        try:
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
            with self.assertRaisesRegex(RuntimeError, "stale"):
                validate_prepared_weights(self.model, self.lora_path, self.output)
        finally:
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    def test_adapter_enables_lazy_load_without_applying_lora_twice(self):
        from adapters import lightx2v_qwen_image_edit as adapter

        pipe = Mock()
        constructor = Mock(return_value=pipe)
        env = {"LIGHTX2V_MODEL_PATH": str(self.model), "LIGHTX2V_LORA_PATH": str(self.lora_path),
               "LIGHTX2V_DISK_OFFLOAD": "1", "LIGHTX2V_DISK_MODEL_PATH": str(self.output)}
        with patch.dict(os.environ, env), patch.object(adapter, "_configure_source_checkout", return_value=adapter.LIGHTX2V_COMMIT), patch.dict(
            sys.modules, {"lightx2v": types.SimpleNamespace(LightX2VPipeline=constructor)}
        ):
            self.assertIs(adapter.load_pipeline(), pipe)
        self.assertEqual(constructor.call_args.kwargs["dit_original_ckpt"], str(self.output))
        self.assertIs(pipe.lazy_load, True)
        self.assertIs(pipe.qwen25vl_load_direct_to_device, True)
        pipe.enable_lora.assert_not_called()
        self.assertEqual(pipe.create_generator.call_args.kwargs["infer_steps"], 4)
        self.assertEqual(pipe.create_generator.call_args.kwargs["guidance_scale"], 1.0)

    def test_text_encoder_streams_bf16_shards_to_target_device(self):
        os.environ.update(PLATFORM="cuda", SKIP_PLATFORM_CHECK="True")
        sys.path.insert(0, str(UPSTREAM))
        from lightx2v.models.input_encoders.hf.qwen25 import qwen25_vlforconditionalgeneration as module

        cls = module.Qwen25_VLForConditionalGeneration_TextEncoder
        for cpu_offload in (False, True):
            encoder = cls.__new__(cls)
            encoder.config = {"model_path": str(self.model), "task": "t2i", "qwen25vl_load_direct_to_device": True}
            encoder.cpu_offload = cpu_offload
            encoder.is_layered = False
            with patch.object(module.Qwen2_5_VLForConditionalGeneration, "from_pretrained") as loader, patch.object(module.Qwen2Tokenizer, "from_pretrained"):
                encoder.load()
                self.assertEqual(loader.call_args.kwargs["torch_dtype"], torch.bfloat16)
                self.assertEqual(loader.call_args.kwargs["device_map"], {"": "cpu" if cpu_offload else module.AI_DEVICE})
                self.assertTrue(loader.call_args.kwargs["low_cpu_mem_usage"])

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for tiny upstream buffers")
    def test_real_qwen_disk_buffers_prefetch_all_blocks_and_wrap(self):
        os.environ.update(PLATFORM="cuda", SKIP_PLATFORM_CHECK="True")
        sys.path.insert(0, str(UPSTREAM))
        from lightx2v.models.networks.qwen_image.model import QwenImageTransformerModel

        config = dict(self.config, task="i2i", model_cls="qwen_image", seq_parallel=False,
                      cpu_offload=True, offload_granularity="block", lazy_load=True,
                      dit_original_ckpt=str(self.output), feature_caching="NoCaching",
                      rms_norm_type="torch", layer_norm_type="torch", attn_type="torch_sdpa",
                      rope_type="torch", num_disk_workers=2)
        model = QwenImageTransformerModel(str(self.model / "transformer"), config, torch.device("cpu"))
        weights = model.transformer_weights
        manager = model.transformer_infer.offload_manager
        self.assertEqual(len(manager.cpu_buffers), 2)
        self.assertEqual(len(manager.cuda_buffers), 2)
        for block in weights.blocks:
            self.assertFalse(hasattr(block.compute_phases[0].to_q, "weight"))
        try:
            for index in list(range(3)) * 2:
                next_index = (index + 1) % 3
                manager.start_prefetch_block(next_index)
                if index == 0:
                    manager.init_first_buffer(weights.blocks)
                manager.swap_cpu_buffers()
                manager.prefetch_weights(next_index, weights.blocks)
                current = manager.cuda_buffers[0].compute_phases[0].to_q
                expected = load_file(str(self.output / f"block_{index}.safetensors"))[
                    f"transformer_blocks.{index}.attn.to_q.weight"
                ]
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(current.weight.cpu(), expected.t()), f"block {index}")
                manager.swap_blocks()
        finally:
            manager.executor.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
