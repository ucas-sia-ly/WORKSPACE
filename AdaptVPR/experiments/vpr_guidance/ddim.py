"""Explicit DDIM sampling. No pipeline __call__, HTTP, or parameter training."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from .vpr import freeze


@dataclass(frozen=True)
class Guidance:
    scale: float = 0.03
    every: int = 5
    last_n: int = 0  # 0 = all steps, counted across both released stages

    def selected(self, index: int, total: int) -> bool:
        return self.scale > 0 and (index + 1) % self.every == 0 and (
            self.last_n == 0 or index >= total - self.last_n)


def normalized_gradient_step(latent, grad, scale):
    grad = grad.float()
    if not torch.isfinite(grad).all():
        raise FloatingPointError('Non-finite VPR gradient; reduce guidance or inspect precision')
    rms = grad.square().mean(dim=tuple(range(1, grad.ndim)), keepdim=True).sqrt()
    if (rms == 0).any():
        raise RuntimeError('Zero VPR gradient: check SALAD detach and VAE/UNet precision')
    return (latent.float() - scale * grad / rms.clamp_min(1e-12)).detach(), rms


def denoise(latent, timesteps, scheduler, predict_noise, decode, descriptor,
            source_descriptor, guidance, offset=0, total=None, stage='base'):
    """Differentiate one x_t -> UNet -> x0 -> VAE -> SALAD graph at a time."""
    total = len(timesteps) if total is None else total
    trace = []
    for local_index, timestep in enumerate(timesteps):
        index = offset + local_index
        latent = latent.detach()
        if guidance.selected(index, total):
            with torch.enable_grad():
                # Float32 leaf keeps normalized updates representable; models stay frozen.
                current_latent = latent.float().requires_grad_(True)
                noise = predict_noise(current_latent, timestep)
                x0_pred = scheduler.step(noise, timestep, current_latent, eta=0).pred_original_sample
                desc_pred = descriptor(decode(x0_pred))
                loss_vpr = (1 - (desc_pred * source_descriptor).sum(dim=-1)).mean()
                if not torch.isfinite(loss_vpr):
                    raise FloatingPointError('Non-finite SALAD loss')
                grad, = torch.autograd.grad(loss_vpr, current_latent)
                latent, rms = normalized_gradient_step(current_latent, grad, guidance.scale)
                trace.append({'stage': stage, 'step': index + 1, 'timestep': int(timestep),
                              'loss_vpr': float(loss_vpr.detach()),
                              'gradient_rms': float(rms.mean())})
                del current_latent, noise, x0_pred, desc_pred, loss_vpr, grad, rms
        with torch.no_grad():
            # Recompute epsilon after changing x_t; never reuse stale predictions.
            latent = latent.to(dtype=predict_noise.dtype)
            noise = predict_noise(latent, timestep)
            latent = scheduler.step(noise, timestep, latent, eta=0).prev_sample.detach()
    return latent, trace


class ICLightExperiment:
    def __init__(self):
        # Reuse the released loader (SD1.5 + additive FC checkpoint + concat hook
        # + DDIM config); it constructs pipelines but we never call their __call__.
        from AdaptVPR.adapters import iclight_sd15_fc as adapter
        self.adapter = adapter
        self.t2i, self.i2i, self.vae = adapter.load_pipeline()
        for model in (self.vae, self.t2i.unet, self.t2i.text_encoder, self.i2i.text_encoder):
            freeze(model)
        self.device = self.vae.device
        self.dtype = self.vae.dtype

    def decode(self, latent):
        decoded = self.vae.decode((latent / self.vae.config.scaling_factor).to(self.dtype),
                                  return_dict=False)[0]
        return (decoded.float() / 2 + 0.5).clamp(0, 1)

    def to_pil(self, latent, pipe):
        # Released VAE dtype and postprocessing, without guidance-time float casts.
        decoded = self.vae.decode(latent / self.vae.config.scaling_factor, return_dict=False)[0]
        return pipe.image_processor.postprocess(decoded, output_type='pil')[0]

    def condition(self, source, width, height):
        import numpy as np
        # Same PIL resize, [-1, 1] pixels and VAE mode as adapter._concat_condition.
        array = np.asarray(source.resize((width, height))).astype('float32') / 127.5 - 1
        tensor = torch.from_numpy(array).permute(2, 0, 1)[None].to(self.device, self.dtype)
        with torch.no_grad():
            return self.vae.encode(tensor).latent_dist.mode() * self.vae.config.scaling_factor

    def generate(self, source, prompt, negative_prompt, seed, strength, descriptor,
                 source_descriptor, guidance):
        from diffusers import DDIMScheduler
        adapter = self.adapter
        width, height = adapter._valid_size(source)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        scheduler = DDIMScheduler.from_config(self.t2i.scheduler.config)
        scheduler.set_timesteps(adapter.DEFAULT_INFERENCE_STEPS, device=self.device)
        base_timesteps = scheduler.timesteps
        refinement_steps = max(1, int(adapter.DEFAULT_HIGHRES_STEPS / strength))
        # Diffusers img2img strength truncates, so 0.30 gives 19 actual steps.
        highres_count = int(refinement_steps * strength)
        total = len(base_timesteps) + highres_count
        if guidance.scale > 0 and not any(guidance.selected(i, total) for i in range(total)):
            raise ValueError('No guidance steps selected; adjust every/last-n')
        with torch.no_grad():
            positive, negative = self.t2i.encode_prompt(
                prompt, self.device, 1, True, negative_prompt=negative_prompt)
            embeddings = torch.cat([negative, positive])
            condition = self.condition(source, width, height)
            # Four latent channels even though IC-Light's conv_in has eight.
            latent = self.t2i.prepare_latents(1, self.vae.config.latent_channels,
                                            height, width, embeddings.dtype,
                                            self.device, generator)

        def predict_noise(latent, timestep):
            model_input = scheduler.scale_model_input(torch.cat([latent] * 2).to(self.dtype), timestep)
            noise = self.t2i.unet(model_input, timestep, encoder_hidden_states=embeddings,
                                  cross_attention_kwargs={'concat_conds': condition},
                                  return_dict=False)[0]
            unconditional, conditional = noise.chunk(2)
            return unconditional + 7.5 * (conditional - unconditional)  # released CFG default

        predict_noise.dtype = self.dtype
        latent, trace = denoise(latent, base_timesteps, scheduler, predict_noise, self.decode,
                                descriptor, source_descriptor, guidance, total=total)
        with torch.no_grad():
            # Preserve the released PIL/8-bit round trip between stages, and its
            # random VAE posterior sample + noise using the SAME generator stream.
            lowres = self.to_pil(latent, self.t2i)
            target_width = int(width * adapter.DEFAULT_HIGHRES_SCALE) // 8 * 8
            target_height = int(height * adapter.DEFAULT_HIGHRES_SCALE) // 8 * 8
            condition = self.condition(source, target_width, target_height)
            scheduler.set_timesteps(refinement_steps, device=self.device)
            self.i2i.scheduler = scheduler
            timesteps, _ = self.i2i.get_timesteps(refinement_steps, strength, self.device)
            image = self.i2i.image_processor.preprocess(lowres.resize((target_width, target_height)))
            latent = self.i2i.prepare_latents(image, timesteps[:1], 1, 1,
                                             embeddings.dtype, self.device, generator)
        latent, highres_trace = denoise(latent, timesteps, scheduler, predict_noise, self.decode,
                                        descriptor, source_descriptor, guidance,
                                        offset=len(base_timesteps), total=total, stage='refinement')
        with torch.no_grad():
            result = self.to_pil(latent, self.i2i)
        return result, trace + highres_trace
