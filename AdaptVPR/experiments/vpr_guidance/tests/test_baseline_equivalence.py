"""CPU diagnostics, including the untouched adapter with small real pipelines."""
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline, UNet2DConditionModel
from transformers import CLIPTextConfig, CLIPTextModel

from AdaptVPR.adapters import iclight_sd15_fc as adapter
from AdaptVPR.experiments.vpr_guidance.check_baseline_equivalence import (
    comparison_metrics, diagnostic_warnings, mean_metrics, observe_sampling,
    pixel_metrics, released_generation, sampling_configuration)
from AdaptVPR.experiments.vpr_guidance.ddim import Guidance, ICLightExperiment
from AdaptVPR.experiments.vpr_guidance.vpr import Descriptor


class BaselineEquivalenceTests(unittest.TestCase):
    def test_pixel_metrics_and_perfect_match_json(self):
        black = Image.new('RGB', (64, 64), (0, 0, 0))
        white = Image.new('RGB', (64, 64), (255, 255, 255))
        metrics = pixel_metrics(black, white)
        self.assertEqual(metrics['pixel_mae'], 255)
        self.assertEqual(metrics['pixel_rmse'], 255)
        self.assertEqual(metrics['psnr'], 0)
        identical = pixel_metrics(black, black)
        self.assertIsNone(identical['psnr'])
        self.assertTrue(identical['psnr_is_infinite'])
        json.dumps(identical, allow_nan=False)
        with self.assertRaises(ValueError):
            pixel_metrics(black, Image.new('RGB', (8, 8)))

    def test_perceptual_metric_inputs_warnings_and_means(self):
        class LPIPSInputCheck(torch.nn.Module):
            def forward(self, a, b):
                self.assertions = (float(a.min()), float(a.max()), float(b.min()), float(b.max()))
                return (a - b).abs().mean()
        class MeanModel(torch.nn.Module):
            def forward(self, image):
                return image.mean((2, 3))
        salad = Descriptor(MeanModel(), device='cpu')
        lpips = LPIPSInputCheck()
        metrics = comparison_metrics(Image.new('RGB', (64, 64), (0, 0, 0)),
                                     Image.new('RGB', (64, 64), (255, 255, 255)),
                                     salad, lpips, 'cpu')
        self.assertEqual(lpips.assertions, (-1., -1., 1., 1.))
        self.assertEqual(len(diagnostic_warnings(metrics)), 2)
        self.assertEqual(diagnostic_warnings(metrics, -1., 3.), [])
        self.assertEqual(len(diagnostic_warnings(metrics, -1., 3., 1., 30.)), 2)
        means = mean_metrics([metrics, dict(metrics, pixel_mae=0)])
        self.assertEqual(means['pixel_mae'], 127.5)
        self.assertEqual(means['psnr'], 0)
        json.dumps(means, allow_nan=False)

    def test_untouched_released_adapter_vs_explicit_real_cpu_pipelines(self):
        torch.manual_seed(42)
        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            vae = AutoencoderKL(block_out_channels=(8, 8), norm_num_groups=4,
                                down_block_types=('DownEncoderBlock2D', 'DownEncoderBlock2D'),
                                up_block_types=('UpDecoderBlock2D', 'UpDecoderBlock2D'),
                                latent_channels=4, sample_size=16)
            unet = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
                                       block_out_channels=(8, 16), norm_num_groups=4,
                                       down_block_types=('CrossAttnDownBlock2D', 'DownBlock2D'),
                                       up_block_types=('UpBlock2D', 'CrossAttnUpBlock2D'),
                                       layers_per_block=1, cross_attention_dim=8, attention_head_dim=2)
            # Match the adapter: conv_in takes 8, while config.in_channels stays 4.
            unet.conv_in = torch.nn.Conv2d(8, 8, 3, padding=1)
            original_forward = unet.forward
            def concat_forward(sample, timestep, encoder_hidden_states, **kwargs):
                concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
                concat = torch.cat([concat] * (sample.shape[0] // concat.shape[0]))
                kwargs['cross_attention_kwargs'] = {}
                return original_forward(torch.cat([sample, concat], dim=1), timestep,
                                        encoder_hidden_states, **kwargs)
            unet.forward = concat_forward
            text = CLIPTextModel(CLIPTextConfig(vocab_size=8, hidden_size=8, intermediate_size=16,
                                                num_hidden_layers=1, num_attention_heads=2,
                                                max_position_embeddings=8))
            scheduler = DDIMScheduler(clip_sample=False, steps_offset=1)
            components = dict(vae=vae, unet=unet, text_encoder=text, tokenizer=None,
                              scheduler=scheduler, safety_checker=None, feature_extractor=None,
                              requires_safety_checker=False)
            t2i, i2i = StableDiffusionPipeline(**components), StableDiffusionImg2ImgPipeline(**components)
            def encode_prompt(self, prompt, device, count, cfg, negative_prompt=None, **kwargs):
                return torch.zeros(1, 8, 8), torch.zeros(1, 8, 8)
            for pipe in (t2i, i2i):
                pipe.encode_prompt = types.MethodType(encode_prompt, pipe)
                pipe.set_progress_bar_config(disable=True)
            with patch.object(adapter, 'load_pipeline', return_value=(t2i, i2i, vae)):
                experiment = ICLightExperiment()
            source = Image.fromarray(np.random.default_rng(42).integers(0, 256, (16, 16, 3), dtype=np.uint8))
            original_generator = torch.Generator
            def cpu_generator(device='cpu'):
                return original_generator(device='cpu')
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source_path = root / 'source.png'
                source.save(source_path)
                settings = {'ICLIGHT_BASE_MODEL_PATH': str(root / 'sd15'),
                            'ICLIGHT_MODEL_PATH': str(root / 'iclight_sd15_fc.safetensors')}
                previous_state = (adapter.state.pipe_t2i, adapter.state.pipe_i2i, adapter.state.vae)
                with patch.dict(os.environ, settings), \
                        patch.object(torch, 'Generator', new=cpu_generator), \
                        patch.object(adapter, '_concat_condition',
                                     side_effect=lambda image, vae, width, height:
                                     experiment.condition(image, width, height)):
                    # Exercise both released refinement policies.
                    for strength in (.30, .22):
                        with observe_sampling(experiment) as released_stages:
                            result = released_generation(experiment, source_path, 'snow', 'warp',
                                                         42, strength, root)
                        with Image.open(result) as image:
                            released = image.convert('RGB')
                        with observe_sampling(experiment) as explicit_stages:
                            explicit, trace = experiment.generate(source, 'snow', 'warp', 42, strength,
                                                                  None, None, Guidance(0))
                        self.assertEqual(trace, [])
                        # Small CPU models can agree exactly; CUDA diagnostic has no equality gate.
                        self.assertLessEqual(pixel_metrics(released, explicit)['pixel_mae'], 1.)
                        config_a = sampling_configuration(experiment, source, source_path, 'snow',
                                                           'warp', 42, strength, released_stages, True)
                        config_b = sampling_configuration(experiment, source, source_path, 'snow',
                                                           'warp', 42, strength, explicit_stages)
                        self.assertEqual(config_a, config_b)
                        self.assertEqual(len(explicit_stages[0]['used_timesteps']), 25)
                        self.assertEqual(len(explicit_stages[1]['used_timesteps']), 19)
                        self.assertIs(unet.forward, concat_forward)
                        self.assertEqual((adapter.state.pipe_t2i, adapter.state.pipe_i2i, adapter.state.vae),
                                         previous_state)
        finally:
            torch.set_num_threads(old_threads)


if __name__ == '__main__':
    unittest.main()
