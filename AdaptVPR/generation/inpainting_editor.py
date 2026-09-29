"""Local trained SD inpainting only: nine-channel UNet, Render Mask sampling.

The legacy targeted_editor API and its 4-channel baseline remain unchanged.
This backend never falls back to base SD, LightX2V or an unmasked editor.
"""

from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import threading

from .targeted_editor import DiffusersMaskedEditor, MaskedEditorConfig, canonical_mask, file_sha256, pixel_sha256

MODEL_ENV = "INPAINTING_MODEL_PATH"


@dataclass(frozen=True)
class TrainedInpaintingConfig(MaskedEditorConfig):
    def __post_init__(self):
        super().__post_init__()
        configured = os.environ.get(MODEL_ENV, "").strip()
        if not configured:
            raise ValueError(f"Set {MODEL_ENV} to an existing trained inpainting Diffusers directory; downloads forbidden")
        if Path(self.model_path).expanduser().resolve() != Path(configured).expanduser().resolve():
            raise ValueError(f"model_path must equal the explicit {MODEL_ENV}")
        if self.local_files_only is not True:
            raise ValueError("local_files_only=False is forbidden")

    @classmethod
    def from_env(cls):
        path = os.environ.get(MODEL_ENV, "").strip()
        if not path:
            raise ValueError(f"Set {MODEL_ENV}; no automatic model discovery or download")
        device = os.environ.get("INPAINTING_DEVICE", "cuda")
        return cls(model_path=path, device=device,
                   dtype=os.environ.get("INPAINTING_DTYPE", "float32" if device == "cpu" else "float16"),
                   num_inference_steps=int(os.environ.get("INPAINTING_STEPS", "20")),
                   guidance_scale=float(os.environ.get("INPAINTING_GUIDANCE", "7.5")),
                   negative_prompt=os.environ.get("INPAINTING_NEGATIVE_PROMPT", ""))


def inspect_checkpoint(config, *, hash_weights=True):
    """Fail closed on base/foreign/custom pipelines before loading any weights.

    Nine channels verify the trained-inpainting *architecture*, not training
    history. The operator supplies trained weights; their local bytes are pinned.
    Tiny random components in unit tests are never a production fallback.
    """
    config.__post_init__()
    root = Path(config.model_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"{MODEL_ENV} must be an existing local Diffusers directory: {root}")
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    index = read(root/"model_index.json")
    pipeline_class = index.get("_class_name")
    if pipeline_class not in {"StableDiffusionInpaintPipeline", "StableDiffusionXLInpaintPipeline"}:
        raise ValueError("Require SD or SDXL InpaintPipeline metadata; base/remote pipelines unsupported")
    is_xl = pipeline_class == "StableDiffusionXLInpaintPipeline"
    expected = {"unet": ["diffusers", "UNet2DConditionModel"],
                "vae": ["diffusers", "AutoencoderKL"],
                "text_encoder": ["transformers", "CLIPTextModel"],
                "tokenizer": ["transformers", "CLIPTokenizer"]}
    if is_xl:
        expected.update(text_encoder_2=["transformers", "CLIPTextModelWithProjection"],
                        tokenizer_2=["transformers", "CLIPTokenizer"])
    for name, component in expected.items():
        if index.get(name) != component or not (root/name).is_dir():
            raise ValueError(f"Require local standard {name} component; no custom code")
    schedulers = {"DDIMScheduler", "PNDMScheduler", "LMSDiscreteScheduler", "EulerDiscreteScheduler",
                  "EulerAncestralDiscreteScheduler", "DPMSolverMultistepScheduler", "DDPMScheduler",
                  "UniPCMultistepScheduler", "HeunDiscreteScheduler"}
    scheduler = index.get("scheduler", [])
    if len(scheduler) != 2 or scheduler[0] != "diffusers" or scheduler[1] not in schedulers:
        raise ValueError("Require a standard local Diffusers scheduler")
    for name, allowed in {"safety_checker": [[None, None], ["stable_diffusion", "StableDiffusionSafetyChecker"]],
                          "feature_extractor": [[None, None], ["transformers", "CLIPImageProcessor"],
                                                ["transformers", "CLIPFeatureExtractor"]]}.items():
        if index.get(name, [None, None]) not in allowed:
            raise ValueError(f"Unsupported {name}; no custom components")
    for name, value in index.items():
        if isinstance(value, list) and name not in {*expected, "scheduler", "safety_checker", "feature_extractor"}:
            raise ValueError(f"Unsupported pipeline component: {name}")
    unet, vae = read(root/"unet/config.json"), read(root/"vae/config.json")
    if unet.get("in_channels") != 9 or unet.get("out_channels") != 4 or vae.get("latent_channels") != 4:
        raise ValueError("Require trained inpainting architecture: UNet 9 input/4 output channels and VAE 4 latent channels; 4-channel base SD forbidden")
    if is_xl and (unet.get("addition_embed_type") != "text_time" or unet.get("cross_attention_dim") != 2048
                  or unet.get("projection_class_embeddings_input_dim") != 2816):
        raise ValueError("Require SDXL dual-encoder text/time conditioning config")
    variant = "fp16" if is_xl and config.dtype == "float16" else None
    components = [*expected, "scheduler"]
    for name in ("safety_checker", "feature_extractor"):
        if index.get(name, [None, None])[0] is not None:
            components.append(name)
    paths = [root/"model_index.json"]
    for name in components:
        directory = root/name
        if not directory.is_dir():
            raise ValueError(f"Missing local component directory: {name}")
        files = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix in {".json", ".txt", ".safetensors", ".model"}
                       and (p.suffix != ".safetensors" or (".fp16." in p.name) == (variant == "fp16")))
        if name in {"unet", "vae", "text_encoder", "text_encoder_2", "safety_checker"} and not any(p.suffix == ".safetensors" for p in files):
            raise ValueError(f"Missing local safetensors weights for {name}; no download or pickle fallback")
        paths.extend(files)
    if (root/"README.md").is_file():
        paths.append(root/"README.md")
    hashes = {p.relative_to(root).as_posix(): file_sha256(p) for p in paths if hash_weights or p.suffix != ".safetensors"}
    return dict(model_path=str(root), pipeline_class=index["_class_name"], unet_in_channels=9,
                architecture="noisy_latents(4) + render_mask(1) + masked_source_latents(4)",
                checkpoint_files_sha256=hashes, weights_hashed=hash_weights,
                training_origin="operator-supplied trained local checkpoint; architecture alone does not prove training history",
                model_config={p.relative_to(root).as_posix(): read(p) for p in paths if p.name.endswith("config.json") or p.name == "model_index.json"},
                weight_variant=variant, local_files_only=True, use_safetensors=True, automatic_download=False)


class TrainedInpaintingEditor(DiffusersMaskedEditor):
    """Same PIL edit contract, with the second argument explicitly a Render Mask.

    Core is deliberately absent from this API; it is saved by the pilot runner
    solely as the scientific targeting footprint. Parent instrumentation verifies
    mask/masked-image UNet channels at every denoising step and restores exact RGB
    outside Render after sampling. Raw output remains in last_raw_output.
    """

    def __init__(self, config: TrainedInpaintingConfig):
        if not isinstance(config, TrainedInpaintingConfig):
            raise TypeError("Require TrainedInpaintingConfig.from_env()")
        super().__init__(config)
        self.checkpoint_audit = inspect_checkpoint(config)
        self._render_lock = threading.Lock()
        self.last_padded_raw_output = None

    def _load_pipeline(self):
        if self._pipe is None:
            # Pin local loading even if global hub settings permit network access.
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            import torch
            from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline
            if inspect_checkpoint(self.config) != self.checkpoint_audit:
                raise ValueError("Checkpoint bytes changed after inspection")
            if self.config.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("Configured CUDA device is unavailable")
            is_xl = self.checkpoint_audit["pipeline_class"] == "StableDiffusionXLInpaintPipeline"
            pipeline_type = StableDiffusionXLInpaintPipeline if is_xl else StableDiffusionInpaintPipeline
            extras = dict(variant=self.checkpoint_audit["weight_variant"], add_watermarker=False) if is_xl else {}
            pipe = pipeline_type.from_pretrained(
                self.checkpoint_audit["model_path"], torch_dtype=getattr(torch, self.config.dtype),
                local_files_only=True, use_safetensors=True, **extras,
            )
            if pipe.unet.config.in_channels != 9 or pipe.unet.config.out_channels != 4 or pipe.vae.config.latent_channels != 4:
                raise ValueError("Loaded checkpoint is not a 9-channel inpainting model")
            if type(pipe) is not pipeline_type:
                raise TypeError("Actual loaded pipeline class differs from the inspected inpainting class")
            if is_xl:
                if pipe.text_encoder_2 is None or pipe.tokenizer_2 is None or pipe.unet.conv_in.weight.shape[1] != 9:
                    raise ValueError("Loaded SDXL conditioning/UNet weights do not match inpainting architecture")
            else:
                pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            for component in (pipe.unet, pipe.vae, pipe.text_encoder, getattr(pipe,"text_encoder_2",None)):
                if component is not None:
                    component.eval().requires_grad_(False)
            self._pipe = pipe.to(self.config.device)
            self._pipe.set_progress_bar_config(disable=True)
        return self._pipe

    def _sample(self, pipe, source, mask_image, prompt, seed, audit):
        if pipe.unet.config.in_channels != 9:
            raise ValueError("4-channel latent blending is forbidden in trained-inpainting backend")
        is_xl = self.checkpoint_audit["pipeline_class"] == "StableDiffusionXLInpaintPipeline"
        audit.update(backend="trained_sdxl_inpainting_9channel" if is_xl else "trained_sd_inpainting_9channel", checkpoint=self.checkpoint_audit,
                     sampling_mask_role="render_mask", core_mask_used_for_sampling=False,
                     strength=1.0, eta=0.0, scheduler=type(pipe.scheduler).__name__, batch_size=1,
                     outside_render_exact_rgb_restoration=True)
        if is_xl:
            return self._sample_xl(pipe, source, mask_image, prompt, seed, audit)
        return super()._sample(pipe, source, mask_image, prompt, seed, audit)

    def _sample_xl(self, pipe, source, mask_image, prompt, seed, audit):
        """Verify the actual SDXL UNet mask inputs, not merely final compositing."""
        import torch
        import diffusers
        from diffusers import StableDiffusionXLInpaintPipeline
        if type(pipe) is not StableDiffusionXLInpaintPipeline or pipe.unet.config.in_channels != 9:
            raise TypeError("Require actual StableDiffusionXLInpaintPipeline with 9-channel UNet")
        audit.update(actual_pipeline_class=type(pipe).__name__, unet_in_channels=9,
                     unet_conv_in_weight_shape=list(pipe.unet.conv_in.weight.shape),
                     sampling_mechanism="mask_and_masked_image_unet_channels",
                     diffusers_version=diffusers.__version__, torch_version=torch.__version__,
                     pipeline_source_sha256=file_sha256(inspect.getfile(type(pipe))),
                     scheduler_source_sha256=file_sha256(inspect.getfile(type(pipe.scheduler))),
                     scheduler_config=dict(pipe.scheduler.config), watermark_enabled=pipe.watermark is not None,
                     determinism="checkpoint scheduler; CPU per-call RNG; deterministic kernels; math SDPA; TF32 off",
                     micro_conditioning=dict(original_size=[source.height,source.width],
                                             target_size=[source.height,source.width], crops_coords_top_left=[0,0]))
        lh, lw = source.height//pipe.vae_scale_factor, source.width//pipe.vae_scale_factor
        full = torch.tensor(bytearray(mask_image.tobytes()), dtype=torch.float32).reshape(source.height,source.width)/255
        expected = full[(torch.arange(lh)*source.height//lh)[:,None], (torch.arange(lw)*source.width//lw)[None,:]]
        if not expected.any():
            raise ValueError("Render disappears at latent resolution")
        trace = {"unet_calls": 0}
        prepare_masks = pipe.prepare_mask_latents

        def capture_masks(*args, **kwargs):
            mask, masked_image = prepare_masks(*args, **kwargs)
            wanted = expected.to(mask.device, mask.dtype)[None,None].expand_as(mask)
            if not torch.equal(mask, wanted):
                raise RuntimeError("SDXL latent Render mask differs from paired input")
            trace.update(mask=mask, masked_image=masked_image)
            return mask, masked_image

        def verify_unet(module, args):
            sample = args[0]
            if (sample.shape[1] != 9 or not torch.equal(sample[:,4:5], trace["mask"])
                    or not torch.equal(sample[:,5:9], trace["masked_image"])):
                raise RuntimeError("SDXL UNet did not receive Render and masked-source latents")
            trace["unet_calls"] += 1

        def step_end(pipeline, index, timestep, values):
            if (trace["unet_calls"] != index+1 or not torch.equal(values["mask"], trace["mask"])
                    or not torch.isfinite(values["latents"]).all()):
                raise RuntimeError("SDXL sampling mask dropped/changed or nonfinite latents")
            audit["sampling_steps"].append(dict(step=index,timestep=float(timestep),
                latent_mask_pixels=int(trace["mask"][0].sum()), latent_mask_shape=[lh,lw],mask_operation_verified=True,
                latent_sha256=hashlib.sha256(values["latents"].detach().cpu().contiguous().numpy().tobytes()).hexdigest()))
            return values

        def check_decode(module, args, result):
            decoded = result[0] if isinstance(result, tuple) else result.sample
            if not torch.isfinite(decoded).all():
                raise RuntimeError("Nonfinite VAE decode; refusing a silently black image")

        hook = pipe.unet.register_forward_pre_hook(verify_unet)
        # VAE.decode is called directly, so audit its result with an instance wrapper.
        original_decode = pipe.vae.decode
        def decode(*args, **kwargs):
            result = original_decode(*args, **kwargs)
            check_decode(pipe.vae, args, result)
            return result
        pipe.prepare_mask_latents, pipe.vae.decode = capture_masks, decode
        try:
            with torch.inference_mode():
                result = pipe(prompt=prompt, image=source.copy(), mask_image=mask_image.copy(),
                    height=source.height, width=source.width, strength=1.0, num_images_per_prompt=1,
                    num_inference_steps=self.config.num_inference_steps, guidance_scale=self.config.guidance_scale,
                    negative_prompt=self.config.negative_prompt, generator=torch.Generator(device="cpu").manual_seed(seed),
                    eta=0.0, output_type="pil", original_size=(source.height,source.width),
                    target_size=(source.height,source.width), crops_coords_top_left=(0,0),
                    callback_on_step_end=step_end, callback_on_step_end_tensor_inputs=["latents","mask"])
            if len(audit["sampling_steps"]) != self.config.num_inference_steps or len(result.images) != 1:
                raise RuntimeError("Incomplete SDXL sampling trace or unexpected image batch")
            return result.images[0]
        finally:
            pipe.prepare_mask_latents, pipe.vae.decode = prepare_masks, original_decode
            hook.remove()

    def edit(self, source_image, render_mask, prompt, seed):
        # Native masks are never resized. Right/bottom edge padding supports
        # e.g. 400x300 dev images without stretching their scientific footprints.
        import numpy as np
        from PIL import Image
        with self._render_lock:
            self.last_audit = self.last_raw_output = self.last_padded_raw_output = None
            if not isinstance(source_image, Image.Image):
                raise ValueError("source_image must be a PIL image")
            source = source_image.convert("RGB")
            render = canonical_mask(render_mask, source.size)
            right, bottom = (-source.width) % 8, (-source.height) % 8
            padded_source = Image.fromarray(np.pad(np.asarray(source), ((0,bottom),(0,right),(0,0)), mode="edge"))
            padded_render = Image.fromarray(np.pad(np.asarray(render), ((0,bottom),(0,right)), mode="constant"))
            sampled = super().edit(padded_source, padded_render, prompt, seed)
            self.last_padded_raw_output = self.last_raw_output.copy()
            box = (0, 0, source.width, source.height)
            self.last_raw_output = self.last_raw_output.crop(box)
            final = Image.composite(sampled.crop(box), source, render)
            self.last_audit.update(
                backend="trained_sdxl_inpainting_9channel" if self.checkpoint_audit["pipeline_class"] == "StableDiffusionXLInpaintPipeline" else "trained_sd_inpainting_9channel", sampling_mask_role="render_mask",
                core_mask_used_for_sampling=False, original_size=list(source.size),
                sampling_source_size=list(padded_source.size), padding_right_bottom=[right,bottom],
                padding_policy="RGB edge replication; Render zero padding; no resize; crop to original after sampling",
                source_pixel_sha256=pixel_sha256(source), mask_pixel_sha256=pixel_sha256(render),
                source_size=list(source.size), padded_raw_output_pixel_sha256=pixel_sha256(self.last_padded_raw_output),
                raw_output_pixel_sha256=pixel_sha256(self.last_raw_output), output_pixel_sha256=pixel_sha256(final),
                outside_render_exact_rgb_restoration=True,
            )
            return final
