"""Exercise real Diffusers model/latent helpers on CPU without pipeline __call__."""
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from AdaptVPR.adapters import iclight_sd15_fc as adapter
from AdaptVPR.experiments.vpr_guidance.ddim import Guidance, ICLightExperiment
from AdaptVPR.experiments.vpr_guidance.vpr import Descriptor


class Helpers:
    # Reuse actual released Diffusers helpers, but replace only prompt encoding.
    prepare_latents = StableDiffusionPipeline.prepare_latents
    def encode_prompt(self, prompt, device, count, cfg, negative_prompt=None):
        return torch.zeros(1, 8, 8), torch.zeros(1, 8, 8)


class ImgHelpers:
    prepare_latents = StableDiffusionImg2ImgPipeline.prepare_latents
    get_timesteps = StableDiffusionImg2ImgPipeline.get_timesteps


class TwoStageTest(unittest.TestCase):
    def test_two_stage_seed_and_frozen_graph(self):
        torch.manual_seed(42)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            vae = AutoencoderKL(block_out_channels=(8, 8), norm_num_groups=4,
                                down_block_types=('DownEncoderBlock2D', 'DownEncoderBlock2D'),
                                up_block_types=('UpDecoderBlock2D', 'UpDecoderBlock2D'),
                                latent_channels=4, sample_size=16)
            unet = UNet2DConditionModel(
                sample_size=8, in_channels=8, out_channels=4,
                block_out_channels=(8, 16), norm_num_groups=4,
                down_block_types=('CrossAttnDownBlock2D', 'DownBlock2D'),
                up_block_types=('UpBlock2D', 'CrossAttnUpBlock2D'),
                layers_per_block=1, cross_attention_dim=8, attention_head_dim=2)
            original_forward = unet.forward
            def concat_forward(sample, timestep, encoder_hidden_states, **kwargs):
                concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
                concat = torch.cat([concat] * (sample.shape[0] // concat.shape[0]))
                kwargs['cross_attention_kwargs'] = {}
                return original_forward(torch.cat([sample, concat], dim=1), timestep,
                                        encoder_hidden_states, **kwargs)
            unet.forward = concat_forward
            scheduler = DDIMScheduler(clip_sample=False, steps_offset=1)
            text = torch.nn.Linear(1, 1)
            t2i, i2i = Helpers(), ImgHelpers()
            for pipe in (t2i, i2i):
                pipe.vae = vae
                pipe.unet = unet
                pipe.text_encoder = text
                pipe.scheduler = scheduler
                pipe.vae_scale_factor = 2
                pipe.image_processor = VaeImageProcessor(vae_scale_factor=2)
            with patch.object(adapter, 'load_pipeline', return_value=(t2i, i2i, vae)):
                experiment = ICLightExperiment()
            class MeanModel(torch.nn.Module):
                def forward(self, image):
                    return image.mean((2, 3))
            salad = Descriptor(MeanModel(), device='cpu')
            source = Image.fromarray(np.random.default_rng(42).integers(0, 256, (16, 16, 3), dtype=np.uint8))
            with torch.no_grad():
                source_desc = salad.from_pil(source).detach()
            baseline, _ = experiment.generate(source, 'snow', 'distortion', 42, .30,
                                               salad, source_desc, Guidance(0))
            repeated, _ = experiment.generate(source, 'snow', 'distortion', 42, .30,
                                               salad, source_desc, Guidance(0))
            np.testing.assert_array_equal(np.asarray(baseline), np.asarray(repeated))
            self.assertEqual(baseline.size, (16, 16))
            guided, trace = experiment.generate(source, 'snow', 'distortion', 42, .30,
                                                salad, source_desc, Guidance(.1, 5, 10))
            self.assertEqual([r['step'] for r in trace], [35, 40])
            self.assertTrue(all(r['stage'] == 'refinement' for r in trace))
            self.assertTrue(all(p.grad is None and not p.requires_grad
                                for m in (vae, unet, text) for p in m.parameters()))
            self.assertGreater(np.abs(np.asarray(guided).astype(float) - np.asarray(baseline)).sum(), 0)
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == '__main__':
    unittest.main()
