"""IC-Light loading, LoRA utilities, and released Global-route generation."""
from __future__ import annotations

import io
from pathlib import Path
import warnings

import numpy as np
import torch
from PIL import Image

from AdaptVPR.adapters import iclight_sd15_fc as adapter
from AdaptVPR.prompts.rules import global_negative_prompt, normalize_weather


LORA_TARGET_MODULES = ("to_q", "to_k", "to_v", "to_out.0")
LORA_FORMAT_VERSION = 1


def _base_metadata():
    return {
        "base_model_id": adapter.BASE_MODEL_ID,
        "base_model_revision": adapter.BASE_MODEL_REVISION,
        "checkpoint_id": adapter.CHECKPOINT_ID,
        "checkpoint_revision": adapter.CHECKPOINT_REVISION,
        "checkpoint_filename": adapter.CHECKPOINT_FILENAME,
    }


def load_iclight():
    return adapter.load_pipeline()


def attach_lora(unet, rank: int = 8, alpha: int = 8):
    """Attach LoRA adapters and keep their trainable weights in fp32.

    The released IC-Light UNet is loaded in fp16. If LoRA parameters are left in
    fp16, AdamW's moment estimates/epsilon can underflow after the first update,
    which commonly turns the adapter weights into Inf/NaN on step 1. Keeping the
    tiny set of trainable LoRA parameters in fp32 preserves the fp16 base model
    memory footprint while making optimizer state numerically stable.
    """
    from peft import LoraConfig

    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("LoRA rank must be a positive integer")
    if isinstance(alpha, bool) or not isinstance(alpha, int) or alpha <= 0:
        raise ValueError("LoRA alpha must be a positive integer")
    if getattr(unet, "peft_config", None):
        raise ValueError("UNet already has an adapter; load LoRA into a fresh IC-Light UNet")
    unet.requires_grad_(False)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=list(LORA_TARGET_MODULES),
    )
    unet.add_adapter(config)
    trainable = [p for p in unet.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no LoRA parameters became trainable")
    unexpected = [name for name, parameter in unet.named_parameters()
                  if parameter.requires_grad and "lora_" not in name]
    if unexpected:
        raise RuntimeError(f"non-LoRA parameters became trainable: {unexpected[:10]}")

    # Important: the base IC-Light UNet is fp16, but AdamW should update LoRA
    # weights in fp32. PEFT casts the LoRA branch input to the adapter weight
    # dtype internally and casts the branch output back to the base-layer dtype.
    for p in trainable:
        p.data = p.data.float()

    bad = [str(p.dtype) for p in trainable if p.dtype != torch.float32]
    if bad:
        raise RuntimeError(f"LoRA parameters must be fp32 for training, found: {bad[:5]}")
    return trainable


def lora_state_dict(unet):
    return {k: v.detach().cpu() for k, v in unet.state_dict().items() if "lora_" in k}


def save_lora(unet, path: Path, *, rank: int, alpha: int, extra: dict | None = None):
    config = getattr(unet, "peft_config", {}).get("default")
    if config is None or config.r != rank or config.lora_alpha != alpha:
        raise ValueError("checkpoint rank/alpha do not match the attached default LoRA")
    if set(config.target_modules) != set(LORA_TARGET_MODULES):
        raise ValueError("attached LoRA target modules do not match IC-Light attention modules")
    state = lora_state_dict(unet)
    if not state or not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("cannot save empty or non-finite LoRA weights")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": LORA_FORMAT_VERSION,
        "rank": rank,
        "alpha": alpha,
        "target_modules": list(LORA_TARGET_MODULES),
        "base": _base_metadata(),
        "state_dict": state,
        "extra": extra or {},
    }, path)


def load_lora(unet, path: Path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not {"rank", "alpha", "state_dict"} <= payload.keys():
        raise ValueError("LoRA checkpoint must contain rank, alpha and state_dict")
    if "format_version" in payload:
        if payload["format_version"] != LORA_FORMAT_VERSION:
            raise ValueError(f"unsupported LoRA format version: {payload['format_version']}")
        if payload.get("target_modules") != list(LORA_TARGET_MODULES):
            raise ValueError("LoRA target modules do not match IC-Light attention modules")
        if payload.get("base") != _base_metadata():
            raise ValueError("LoRA base-model/IC-Light checkpoint metadata mismatch")
    else:
        warnings.warn("legacy LoRA checkpoint has no base/module metadata; validating all adapter keys and shapes",
                      UserWarning, stacklevel=2)
    state = payload["state_dict"]
    if not isinstance(state, dict) or not state:
        raise ValueError("LoRA checkpoint has an empty or invalid state_dict")
    if any(not isinstance(key, str) or "lora_" not in key for key in state):
        raise ValueError("LoRA checkpoint contains non-adapter keys")
    for name, value in state.items():
        if not torch.is_tensor(value) or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"invalid/non-finite LoRA weight: {name}")
    attach_lora(unet, payload["rank"], payload["alpha"])
    expected = lora_state_dict(unet)
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing or unexpected:
        raise ValueError(f"LoRA key mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}")
    for name, value in state.items():
        if value.shape != expected[name].shape:
            raise ValueError(f"LoRA shape mismatch for {name}: {tuple(value.shape)} != {tuple(expected[name].shape)}")
    # Include the unchanged base state so the complete UNet load is strict.
    # The adapter-only key equality above prevents a checkpoint from changing it.
    full_state = unet.state_dict()
    full_state.update(state)
    unet.load_state_dict(full_state, strict=True)
    return payload


def conditioning_source(source: Image.Image) -> Image.Image:
    """Match the original ICLightGenerator HTTP transport's JPEG-95 source."""
    buffer = io.BytesIO()
    source.convert("RGB").save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def released_negative_prompt(value: str | None = None) -> str:
    """Use the original Global agent default; allow an explicit experiment input."""
    return value or global_negative_prompt()


def sampling_policy(condition: str) -> dict:
    weather = normalize_weather(condition)
    if weather is None:
        raise ValueError(f"invalid Global weather/time-of-day condition: {condition!r}")
    strength = adapter.RAIN_HIGHRES_DENOISE if weather == "rain" else adapter.DEFAULT_HIGHRES_DENOISE
    return {
        "scheduler": "DDIMScheduler",
        "highres_scale": adapter.DEFAULT_HIGHRES_SCALE,
        "highres_denoise": strength,
        "num_inference_steps": adapter.DEFAULT_INFERENCE_STEPS,
        "highres_steps": adapter.DEFAULT_HIGHRES_STEPS,
        "stage2_num_inference_steps": max(1, int(adapter.DEFAULT_HIGHRES_STEPS / strength)),
        # Both pipeline calls inherit the released diffusers default explicitly.
        "guidance_scale": 7.5,
        "conditioning_preprocessing": "original_iclight_http_rgb_jpeg95_v1",
        "prompt_policy": "released_frozen_prompt_verbatim",
    }


def encode_image_latent(image: Image.Image, vae, width: int, height: int):
    array = np.asarray(image.convert("RGB").resize((width, height))).astype("float32") / 127.5 - 1.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(vae.device, dtype=vae.dtype)
    with torch.no_grad():
        return vae.encode(tensor).latent_dist.mode() * vae.config.scaling_factor


def image_tensor_01(image: Image.Image, width: int, height: int, device="cuda"):
    array = np.asarray(image.convert("RGB").resize((width, height))).astype("float32") / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def decode_latent_01(latent, vae):
    decoded = vae.decode((latent / vae.config.scaling_factor).to(dtype=vae.dtype), return_dict=False)[0]
    return (decoded.float() / 2.0 + 0.5).clamp(0, 1)


def encode_prompt(pipe, prompt: str):
    tokens = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                            truncation=True, return_tensors="pt")
    with torch.no_grad():
        return pipe.text_encoder(tokens.input_ids.to(pipe.device), return_dict=False)[0]


def generate_released(pipe_t2i, pipe_i2i, vae, source, prompt, negative_prompt, seed, condition):
    if pipe_t2i.unet is not pipe_i2i.unet:
        raise ValueError("IC-Light stages must share the same UNet so LoRA affects both")
    policy = sampling_policy(condition)
    source = conditioning_source(source)
    negative_prompt = released_negative_prompt(negative_prompt)
    width, height = adapter._valid_size(source)
    strength = policy["highres_denoise"]
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.inference_mode():
        base_cond = adapter._concat_condition(source, vae, width, height)
        lowres = pipe_t2i(
            prompt=prompt, negative_prompt=negative_prompt,
            num_inference_steps=adapter.DEFAULT_INFERENCE_STEPS, width=width, height=height,
            guidance_scale=policy["guidance_scale"],
            generator=generator, cross_attention_kwargs={"concat_conds": base_cond},
        ).images[0]
        target_w = int(width * adapter.DEFAULT_HIGHRES_SCALE) // 8 * 8
        target_h = int(height * adapter.DEFAULT_HIGHRES_SCALE) // 8 * 8
        high_cond = adapter._concat_condition(source, vae, target_w, target_h)
        return pipe_i2i(
            prompt=prompt, negative_prompt=negative_prompt,
            image=lowres.resize((target_w, target_h)), strength=strength,
            num_inference_steps=policy["stage2_num_inference_steps"],
            guidance_scale=policy["guidance_scale"],
            generator=generator, cross_attention_kwargs={"concat_conds": high_cond},
        ).images[0]
