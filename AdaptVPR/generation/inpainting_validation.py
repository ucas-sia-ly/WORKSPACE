"""Actual local-checkpoint loading/zero/location/determinism gates before pilot."""

import copy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .targeted_editor import canonical_mask, file_sha256, pixel_sha256


def validate_backend(editor, source, render, prompt, seed, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    def write(path, data):
        path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+"\n")
    report = dict(status="RUNNING", tests={}, trained_checkpoint=True, pilot_filtering=False,
                  config=asdict(editor.config), checkpoint=editor.checkpoint_audit)
    write(output/"summary.json", report)
    try:
        pipe = editor._load_pipeline()
        expected = editor.checkpoint_audit["pipeline_class"]
        if type(pipe).__name__ != expected or pipe.unet.config.in_channels != 9 or pipe.unet.conv_in.weight.shape[1] != 9:
            raise RuntimeError("Loaded class/weight shape is not the inspected inpainting model")
        import torch
        if pipe.unet.dtype != getattr(torch, editor.config.dtype):
            raise RuntimeError("Loaded dtype differs from requested sampling config")
        report["tests"]["loading"] = dict(passed=True, actual_pipeline_class=type(pipe).__name__,
            unet_conv_in_weight_shape=list(pipe.unet.conv_in.weight.shape), unet_dtype=str(pipe.unet.dtype),
            vae_dtype=str(pipe.vae.dtype), scheduler=type(pipe.scheduler).__name__, local_files_only=True,
            automatic_download=False, batch_size=1, model_path=editor.checkpoint_audit["model_path"])
        print("Backend gate: local checkpoint loading PASS", flush=True)

        def save(name, mask, final):
            directory = output/name
            directory.mkdir()
            source.save(directory/"source.png")
            mask.save(directory/"render_mask.png")
            editor.last_raw_output.save(directory/"raw_output.png")
            editor.last_padded_raw_output.save(directory/"raw_output_padded.png")
            final.save(directory/"final_output.png")
            audit = copy.deepcopy(editor.last_audit)
            outside = np.asarray(canonical_mask(mask,source.size)) == 0
            if not np.array_equal(np.asarray(source)[outside],np.asarray(final)[outside]):
                raise RuntimeError("Outside Render RGB changed during backend test")
            audit["outside_render_exact_rgb_verified"] = True
            audit["files_sha256"] = {p.name:file_sha256(p) for p in directory.iterdir()}
            write(directory/"audit.json", audit)
            return audit

        zero = Image.new("L", source.size, 0)
        original_sample = editor._sample
        def forbidden(*args, **kwargs):
            raise RuntimeError("zero-mask must not call sampling")
        editor._sample = forbidden
        try:
            final = editor.edit(source, zero, prompt, seed)
        finally:
            editor._sample = original_sample
        save("zero",zero,final)
        if final.tobytes() != source.convert("RGB").tobytes() or editor.last_audit["generated"]:
            raise RuntimeError("Zero-mask identity failed")
        report["tests"]["zero_mask"] = dict(passed=True, exact_source_identity=True, sampling_called=False)
        print("Backend gate: zero-mask PASS", flush=True)

        first = editor.edit(source, render, prompt, seed)
        first_audit = save("original_mask",render,first)
        repeated = editor.edit(source, render, prompt, seed)
        repeat_audit = save("repeat",render,repeated)
        if (pixel_sha256(first) != pixel_sha256(repeated)
                or first_audit["raw_output_pixel_sha256"] != repeat_audit["raw_output_pixel_sha256"]
                or first_audit["sampling_steps"] != repeat_audit["sampling_steps"]):
            raise RuntimeError("Same-environment same-seed raw/final/latent determinism failed")
        report["tests"]["determinism"] = dict(passed=True, raw_exact=True, final_exact=True, latent_trace_exact=True)
        print("Backend gate: determinism PASS", flush=True)

        mask = np.asarray(canonical_mask(render,source.size))>0
        ys,xs = np.nonzero(mask)
        top,left,bottom,right = int(ys.min()),int(xs.min()),int(ys.max())+1,int(xs.max())+1
        height,width = bottom-top,right-left
        locations = [(y,x) for y in (0,source.height-height) for x in (0,source.width-width) if (y,x)!=(top,left)]
        if not locations:
            raise RuntimeError("Render has no legal nonzero translation for sensitivity test")
        y,x = max(locations,key=lambda pt:(abs(pt[0]-top)+abs(pt[1]-left),pt))
        shifted = np.zeros_like(mask)
        shifted[y:y+height,x:x+width] = mask[top:bottom,left:right]
        np.testing.assert_array_equal(np.argwhere(shifted),np.argwhere(mask)+[y-top,x-left])
        moved_mask = Image.fromarray(shifted.astype(np.uint8)*255)
        moved = editor.edit(source,moved_mask,prompt,seed)
        moved_audit = save("moved_mask",moved_mask,moved)
        if (first_audit["raw_output_pixel_sha256"] == moved_audit["raw_output_pixel_sha256"]
                or first_audit["sampling_steps"][-1]["latent_sha256"] == moved_audit["sampling_steps"][-1]["latent_sha256"]):
            raise RuntimeError("Moving Render did not change raw generation and denoising latents")
        report["tests"]["mask_location_sensitivity"] = dict(passed=True, raw_changed=True, latent_trace_changed=True,
            offset_dy_dx=[y-top,x-left], pixel_exact_translation=True, prompt_seed_source_held_fixed=True)
        print("Backend gate: mask-location sensitivity PASS", flush=True)
        report.update(status="PASS", nonzero_sampling_runs=3, steps_per_run=editor.config.num_inference_steps,
                      outside_render_exact_all_outputs=True, scope="backend integrity; no realism/hardness acceptance")
    except Exception as exc:
        report.update(status="FAIL",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write(output/"summary.json",report)
    return report
