"""IC-Light loading, LoRA utilities, and released Global-route generation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from AdaptVPR.adapters import iclight_sd15_fc as adapter


def load_iclight():
    return adapter.load_pipeline()


def attach_lora(unet, rank: int = 8, alpha: int = 8):
    from peft import LoraConfig
    unet.requires_grad_(False)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(config)
    trainable = [p for p in unet.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no LoRA parameters became trainable")
    return trainable


def lora_state_dict(unet):
    return {k: v.detach().cpu() for k, v in unet.state_dict().items() if "lora_" in k}


def save_lora(unet, path: Path, *, rank: int, alpha: int, extra: dict | None = None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"rank": rank, "alpha": alpha, "state_dict": lora_state_dict(unet), "extra": extra or {}}, path)


def load_lora(unet, path: Path):
    payload = torch.load(Path(path), map_location="cpu")
    attach_lora(unet, int(payload["rank"]), int(payload["alpha"]))
    result = unet.load_state_dict(payload["state_dict"], strict=False)
    bad = [k for k in result.unexpected_keys if "lora_" in k]
    if bad:
        raise RuntimeError(f"unexpected LoRA keys: {bad[:10]}")
    return payload


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
    width, height = adapter._valid_size(source)
    strength = adapter.RAIN_HIGHRES_DENOISE if condition.lower() == "rain" else adapter.DEFAULT_HIGHRES_DENOISE
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.inference_mode():
        base_cond = adapter._concat_condition(source, vae, width, height)
        lowres = pipe_t2i(
            prompt=prompt, negative_prompt=negative_prompt or adapter.DEFAULT_NEGATIVE_PROMPT,
            num_inference_steps=adapter.DEFAULT_INFERENCE_STEPS, width=width, height=height,
            generator=generator, cross_attention_kwargs={"concat_conds": base_cond},
        ).images[0]
        target_w = int(width * adapter.DEFAULT_HIGHRES_SCALE) // 8 * 8
        target_h = int(height * adapter.DEFAULT_HIGHRES_SCALE) // 8 * 8
        high_cond = adapter._concat_condition(source, vae, target_w, target_h)
        return pipe_i2i(
            prompt=prompt, negative_prompt=negative_prompt or adapter.DEFAULT_NEGATIVE_PROMPT,
            image=lowres.resize((target_w, target_h)), strength=strength,
            num_inference_steps=max(1, int(adapter.DEFAULT_HIGHRES_STEPS / strength)),
            generator=generator, cross_attention_kwargs={"concat_conds": high_cond},
        ).images[0]
