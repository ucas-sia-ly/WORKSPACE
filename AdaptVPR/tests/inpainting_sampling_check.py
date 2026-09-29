"""Real tiny RANDOM Diffusers components: plumbing evidence, never trained pilot."""

import json
import os
from pathlib import Path
import runpy
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Also rerun the unchanged legacy smoke's lower-level regression checks.
prior = runpy.run_path(str(ROOT/"tests/targeted_editor_sampling_check.py"))
from generation.inpainting_editor import TrainedInpaintingConfig, TrainedInpaintingEditor
from generation.targeted_editor import pixel_sha256
fixture = runpy.run_path(str(ROOT/"tests/test_inpainting_editor.py"))["fixture_checkpoint"]
from PIL import Image
import numpy as np

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory)
    fixture(path)
    os.environ["INPAINTING_MODEL_PATH"] = directory
    config = TrainedInpaintingConfig(model_path=directory, device="cpu", dtype="float32",
                                     num_inference_steps=2, guidance_scale=1.)
    editor = TrainedInpaintingEditor(config)
    # Only this test injects random components, bypassing pretrained loading.
    editor._pipe = prior["pipe"]  # Last legacy case is the real 9-channel pipeline.
    source = Image.new("RGB", (31, 29), (50, 80, 110))
    render = Image.new("L", source.size, 0)
    render.paste(255, (8, 8, 24, 24))
    final = editor.edit(source, render, "a barrier", 4)
    outside = np.asarray(render) == 0
    assert np.array_equal(np.asarray(final)[outside], np.asarray(source)[outside])
    assert editor.last_audit["unet_in_channels"] == 9
    assert editor.last_audit["sampling_mask_role"] == "render_mask"
    assert final.size == source.size and editor.last_raw_output.size == source.size
    assert editor.last_audit["padding_right_bottom"] == [1,3]
    assert editor.last_padded_raw_output.size == (32,32)
    assert not editor.last_audit["core_mask_used_for_sampling"]
    assert all(s["mask_operation_verified"] for s in editor.last_audit["sampling_steps"])
    first = pixel_sha256(editor.last_raw_output)
    trace = editor.last_audit["sampling_steps"]
    editor.edit(source, render, "a barrier", 4)
    assert first == pixel_sha256(editor.last_raw_output) and trace == editor.last_audit["sampling_steps"]
    moved = Image.new("L", source.size, 0)
    moved.paste(255, (0, 0, 16, 16))
    editor.edit(source, moved, "a barrier", 4)
    assert first != pixel_sha256(editor.last_raw_output)
    editor._pipe.unet.register_to_config(in_channels=4)
    try:
        editor.edit(source, render, "a barrier", 4)
        raise AssertionError("4-channel model accepted")
    except ValueError as exc:
        assert "4-channel latent blending is forbidden" in str(exc)
    # Separate real SDXL sampling branch with tiny random components. This is
    # not checkpoint-loading evidence; the trained local pilot runs its own gates.
    import torch
    from diffusers import StableDiffusionXLInpaintPipeline, UNet2DConditionModel, EulerDiscreteScheduler
    unet = UNet2DConditionModel(sample_size=16,in_channels=9,out_channels=4,layers_per_block=1,
        block_out_channels=(32,64),down_block_types=("DownBlock2D","CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D","UpBlock2D"),cross_attention_dim=32,
        attention_head_dim=8,norm_num_groups=8,addition_embed_type="text_time",
        addition_time_embed_dim=8,projection_class_embeddings_input_dim=80)
    xl = StableDiffusionXLInpaintPipeline(vae=prior["vae"],text_encoder=None,text_encoder_2=None,
        tokenizer=None,tokenizer_2=None,unet=unet,scheduler=EulerDiscreteScheduler(num_train_timesteps=100),
        add_watermarker=False).to("cpu")
    xl.set_progress_bar_config(disable=True)
    xl.encode_prompt = lambda *a,**kw: (torch.full((1,4,32),.1),torch.zeros(1,4,32),torch.full((1,32),.1),torch.zeros(1,32))
    editor._pipe = xl
    editor.checkpoint_audit["pipeline_class"] = "StableDiffusionXLInpaintPipeline"
    from dataclasses import replace
    editor.config = replace(editor.config,guidance_scale=3.)
    final = editor.edit(source,render,"a barrier",4)
    first_raw = pixel_sha256(editor.last_raw_output)
    first_trace = editor.last_audit["sampling_steps"]
    assert editor.last_audit["actual_pipeline_class"] == "StableDiffusionXLInpaintPipeline"
    assert np.array_equal(np.asarray(final)[outside],np.asarray(source)[outside])
    again = editor.edit(source,render,"a barrier",4)
    assert again.tobytes() == final.tobytes() and first_raw == pixel_sha256(editor.last_raw_output)
    assert first_trace == editor.last_audit["sampling_steps"]
    editor.edit(source,moved,"a barrier",4)
    assert first_raw != pixel_sha256(editor.last_raw_output)
    prepare = xl.prepare_mask_latents
    def drop(*a,**kw):
        m,z = prepare(*a,**kw)
        return m*0,z
    xl.prepare_mask_latents = drop
    try:
        editor.edit(source,render,"a barrier",4)
        raise AssertionError("dropped SDXL mask accepted")
    except RuntimeError as exc:
        assert "SDXL latent Render mask differs" in str(exc)
    finally:
        xl.prepare_mask_latents = prepare
print(json.dumps(dict(actual_nine_channel_sampling=True, outside_render_exact=True,
                      actual_sdxl_sampling=True, sdxl_dropped_mask_rejected=True,
                      base_model_rejected=True, trained_checkpoint_used=False, counts_as_pilot=False)))
