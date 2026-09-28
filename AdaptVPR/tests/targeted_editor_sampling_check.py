"""Offline real Diffusers sampling test with tiny RANDOM weights, not smoke evidence."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionInpaintPipeline, UNet2DConditionModel
from PIL import Image

from generation.targeted_editor import DiffusersMaskedEditor, MaskedEditorConfig, pixel_sha256

torch.set_num_threads(1)
results = []
for channels in (4, 9):
    torch.manual_seed(31)
    vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4,
                        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
                        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
                        block_out_channels=(32, 64), norm_num_groups=8, sample_size=32)
    unet = UNet2DConditionModel(sample_size=16, in_channels=channels, out_channels=4,
                               layers_per_block=1, block_out_channels=(32, 64),
                               down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                               up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
                               cross_attention_dim=32, attention_head_dim=8, norm_num_groups=8)
    pipe = StableDiffusionInpaintPipeline(
        vae=vae, text_encoder=None, tokenizer=None, unet=unet,
        scheduler=DDIMScheduler(num_train_timesteps=100),
        safety_checker=None, feature_extractor=None, requires_safety_checker=False,
    ).to("cpu")
    # Fixed embeddings replace only the text model. Real VAE/UNet/scheduler run.
    pipe.encode_prompt = lambda *args, **kwargs: (torch.full((1, 4, 32), .1), None)
    pipe.set_progress_bar_config(disable=True)
    editor = DiffusersMaskedEditor(MaskedEditorConfig(model_path="random-test-components", device="cpu", dtype="float32",
                                                      num_inference_steps=2, guidance_scale=1))
    editor._pipe = pipe
    source = Image.new("RGB", (32, 32), (90, 100, 110))
    left = Image.new("L", source.size, 0)
    left.paste(1, (0, 0, 16, 32))
    right = Image.new("L", source.size, 0)
    right.paste(1, (16, 0, 32, 32))
    a = editor.edit(source, left, "test", 123)
    raw_a = pixel_sha256(editor.last_raw_output)
    trace_a = editor.last_audit["sampling_steps"]
    assert len(trace_a) == 2 and all(s["mask_operation_verified"] for s in trace_a)
    assert a.crop((16, 0, 32, 32)).tobytes() == source.crop((16, 0, 32, 32)).tobytes()
    assert a.crop((0, 0, 16, 32)).tobytes() != source.crop((0, 0, 16, 32)).tobytes()
    repeated = editor.edit(source, left, "test", 123)
    assert repeated.tobytes() == a.tobytes() and editor.last_audit["sampling_steps"] == trace_a
    editor.edit(source, right, "test", 123)
    assert pixel_sha256(editor.last_raw_output) != raw_a, "mask must change PRE-composite result"
    assert editor.last_audit["sampling_steps"][-1]["latent_sha256"] != trace_a[-1]["latent_sha256"]
    # Same mask, different source and seed must reach real encoding/noise.
    editor.edit(Image.new("RGB", (32, 32), "red"), left, "test", 124)
    assert editor.last_audit["seed"] == 124
    assert pixel_sha256(editor.last_raw_output) != raw_a
    # Corrupt only the lower-level mask: runtime audit must fail before acceptance.
    original = pipe.prepare_mask_latents
    def drop_mask(*args, **kwargs):
        mask, masked = original(*args, **kwargs)
        return torch.zeros_like(mask), masked
    pipe.prepare_mask_latents = drop_mask
    try:
        editor.edit(source, left, "test", 123)
        raise AssertionError("dropped backend mask was accepted")
    except RuntimeError as exc:
        assert "Mask reaching sampling differs" in str(exc)
    finally:
        pipe.prepare_mask_latents = original
    results.append(dict(unet_channels=channels, actual_sampling=True, moved_mask_changes_raw=True,
                        deterministic_repeat=True, dropped_mask_rejected=True, sampling_steps=2))
print(json.dumps(results))
