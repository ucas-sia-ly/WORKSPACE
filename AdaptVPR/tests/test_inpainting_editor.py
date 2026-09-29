"""Offline backend contract tests; random components never count as pilot edits."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from generation.inpainting_editor import TrainedInpaintingConfig, TrainedInpaintingEditor, inspect_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def fixture_checkpoint(root, channels=9):
    # Metadata/loader-spy fixture only; these are NOT trained or usable weights.
    index = dict(_class_name="StableDiffusionInpaintPipeline", unet=["diffusers", "UNet2DConditionModel"],
                 vae=["diffusers", "AutoencoderKL"], text_encoder=["transformers", "CLIPTextModel"],
                 tokenizer=["transformers", "CLIPTokenizer"], scheduler=["diffusers", "DDIMScheduler"],
                 safety_checker=[None, None], feature_extractor=[None, None])
    (root/"model_index.json").write_text(json.dumps(index))
    for name in ("unet", "vae", "text_encoder", "tokenizer", "scheduler"):
        (root/name).mkdir()
        (root/name/"config.json").write_text(json.dumps(dict(in_channels=channels, out_channels=4, latent_channels=4)))
        if name in {"unet", "vae", "text_encoder"}:
            (root/name/"test_only.safetensors").write_bytes(b"not weights; inspection fixture only")


class InpaintingEditorTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        fixture_checkpoint(self.root)
        scope = patch.dict(os.environ, {"INPAINTING_MODEL_PATH": str(self.root), "INPAINTING_DEVICE": "cpu",
                                       "INPAINTING_DTYPE": "float32"}, clear=True)
        scope.start()
        self.addCleanup(scope.stop)
        self.config = TrainedInpaintingConfig.from_env()

    def test_explicit_env_no_legacy_fallback_no_online_override(self):
        with patch.dict(os.environ, {"TARGETED_EDITOR_MODEL_PATH": str(self.root)}, clear=True):
            with self.assertRaisesRegex(ValueError, "INPAINTING_MODEL_PATH"):
                TrainedInpaintingConfig.from_env()
        with self.assertRaisesRegex(ValueError, "explicit"):
            replace(self.config, model_path="/another/model")
        with self.assertRaisesRegex(ValueError, "forbidden"):
            replace(self.config, local_files_only=False)

    def test_base_four_channel_rejected_before_loader(self):
        (self.root/"unet/config.json").write_text(json.dumps(dict(in_channels=4, out_channels=4)))
        with self.assertRaisesRegex(ValueError, "4-channel base SD forbidden"):
            TrainedInpaintingEditor(self.config)

    def test_base_pipeline_metadata_and_custom_code_rejected(self):
        path = self.root/"model_index.json"
        data = json.loads(path.read_text())
        data["_class_name"] = "StableDiffusionPipeline"
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "metadata"):
            inspect_checkpoint(self.config)

    def test_sdxl_requires_dual_encoders_and_nine_channel_inpainting(self):
        path = self.root/"model_index.json"
        data = json.loads(path.read_text())
        data["_class_name"] = "StableDiffusionXLInpaintPipeline"
        data.update(text_encoder_2=["transformers","CLIPTextModelWithProjection"], tokenizer_2=["transformers","CLIPTokenizer"])
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,"text_encoder_2"):
            inspect_checkpoint(self.config)
        for name in ("text_encoder_2","tokenizer_2"):
            (self.root/name).mkdir()
        (self.root/"text_encoder_2/test.safetensors").write_bytes(b"inspection fixture")
        unet_path = self.root/"unet/config.json"
        unet = json.loads(unet_path.read_text())
        unet.update(addition_embed_type="text_time",cross_attention_dim=2048,projection_class_embeddings_input_dim=2816)
        unet_path.write_text(json.dumps(unet))
        audit = inspect_checkpoint(self.config)
        self.assertEqual(audit["pipeline_class"],"StableDiffusionXLInpaintPipeline")
        self.assertEqual(audit["model_config"]["unet/config.json"]["in_channels"],9)
        unet["in_channels"] = 4
        unet_path.write_text(json.dumps(unet))
        with self.assertRaisesRegex(ValueError,"4-channel base SD forbidden"):
            inspect_checkpoint(self.config)
        data["_class_name"] = "StableDiffusionXLPipeline"
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,"metadata"):
            inspect_checkpoint(self.config)
        data["_class_name"] = "StableDiffusionInpaintPipeline"
        data["unet"] = ["custom_pipeline", "UNet2DConditionModel"]
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "custom code"):
            inspect_checkpoint(self.config)

    def test_missing_local_safetensors_never_falls_back(self):
        (self.root/"unet/test_only.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "no download"):
            inspect_checkpoint(self.config)

    def test_checkpoint_mutation_rejected_before_loading(self):
        editor = TrainedInpaintingEditor(self.config)
        (self.root/"unet/test_only.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            editor._load_pipeline()

    def test_loader_forces_local_safetensors_and_rechecks_actual_channels(self):
        editor = TrainedInpaintingEditor(self.config)
        base = SimpleNamespace(unet=SimpleNamespace(config=SimpleNamespace(in_channels=4)))
        with patch("diffusers.StableDiffusionInpaintPipeline.from_pretrained", return_value=base) as load:
            with self.assertRaisesRegex(ValueError, "not a 9-channel"):
                editor._load_pipeline()
        self.assertTrue(load.call_args.kwargs["local_files_only"])
        self.assertTrue(load.call_args.kwargs["use_safetensors"])
        self.assertEqual(load.call_args.args, (str(self.root),))
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")

    def test_actual_nine_channel_render_sampling(self):
        result = subprocess.run([sys.executable, str(ROOT/"tests/inpainting_sampling_check.py")], cwd=ROOT,
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        result = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertTrue(result["actual_nine_channel_sampling"])
        self.assertTrue(result["outside_render_exact"])
        self.assertTrue(result["base_model_rejected"])
        self.assertTrue(result["actual_sdxl_sampling"])
        self.assertTrue(result["sdxl_dropped_mask_rejected"])


if __name__ == "__main__":
    unittest.main()
