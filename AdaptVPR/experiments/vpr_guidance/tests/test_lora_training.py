"""CPU regressions for actual IC-Light conditioning and recoverable LoRA training."""

import contextlib
import io
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import safetensors.torch as sf
import torch
from diffusers import DDPMScheduler
from PIL import Image

GUIDANCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GUIDANCE_ROOT))

from common import write_jsonl  # noqa: E402
from lora_utils import (  # noqa: E402
    freeze_non_lora_parameters, get_lora_parameters, inject_lora_into_unet,
    load_lora_checkpoint, lora_state_dict, read_lora_metadata,
    save_lora_checkpoint, unfreeze_lora_parameters,
)
from train_lora import (  # noqa: E402
    CyclingBatchSampler, denoising_loss, diffusion_target, encode_prompts,
    filter_training_rows, load_training_rows, load_training_state, main,
    lora_delta_norm, make_lr_scheduler, parse_args, save_training_state, to_pixels, train_steps,
)
from adapters.iclight_sd15_fc import _concat_condition, _configure_unet  # noqa: E402


class TinyConditionalUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = torch.nn.Linear(8, 4)

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs):
        joined = torch.cat([sample, cross_attention_kwargs["concat_conds"]], dim=1)
        return SimpleNamespace(sample=self.to_q(joined.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))


class DataAndSamplerTests(unittest.TestCase):
    def test_only_verified_strictly_mined_rows_are_eligible(self):
        rows = [{"passed": True, "utility": 0.0}, {"passed": True, "utility": 0.4},
                {"passed": False, "utility": 4.0}, {"passed": "true", "utility": 5.0}]
        self.assertEqual(filter_training_rows(rows), [rows[1]])
        self.assertEqual(filter_training_rows(rows, 0.4), [])
        self.assertEqual(filter_training_rows([{**rows[1], "eligible_for_training": False}]), [])
        self.assertEqual(filter_training_rows([{**rows[1], "eligible_for_training": True}]), [
            {**rows[1], "eligible_for_training": True}])
        for value in (float("nan"), float("inf"), True, "1"):
            with self.assertRaises(ValueError):
                filter_training_rows([{"passed": True, "utility": value}])

    def test_check_data_empty_selection_does_not_load_cuda_or_models(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected.jsonl"
            write_jsonl(path, [{"passed": True, "utility": 0.0}])
            with patch("torch.cuda.is_available", side_effect=AssertionError("CUDA consulted")):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    main(["--selected", str(path), "--check-data"])
            self.assertEqual(json.loads(output.getvalue())["training_examples"], 0)
            with self.assertRaisesRegex(SystemExit, "retain the previous generator"):
                load_training_rows([path], 0.0)

    def test_target_resolution_and_required_fields_are_checked_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source.png", root / "target.png"
            Image.new("RGB", (32, 27)).save(source)
            Image.new("RGB", (32, 24)).save(target)
            row = {"source_path": str(source), "output_path": str(target), "prompt": "rainy street",
                   "passed": True, "utility": 0.3}
            path = root / "selected.jsonl"
            write_jsonl(path, [row])
            self.assertEqual(load_training_rows([path], 0.0)[2], (32, 24))
            Image.new("RGB", (40, 24)).save(target)
            with self.assertRaisesRegex(ValueError, "differs from target"):
                load_training_rows([path], 0.0)
            write_jsonl(path, [{**row, "prompt": None}])
            with self.assertRaisesRegex(ValueError, "prompt"):
                load_training_rows([path], 0.0)

    def test_sampler_small_dataset_and_saved_remaining_order(self):
        self.assertEqual(CyclingBatchSampler(1, 7, 3).next_batch(), [0] * 7)
        first = CyclingBatchSampler(5, 3, 3)
        first.next_batch()
        restored = CyclingBatchSampler(5, 3, 99)
        restored.load_state_dict(first.state_dict())
        for _ in range(8):
            self.assertEqual(first.next_batch(), restored.next_batch())
        with self.assertRaises(ValueError):
            CyclingBatchSampler(6, 3, 3).load_state_dict(first.state_dict())

    def test_duplicate_outputs_are_deduplicated_and_conflicting_rows_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source.png", root / "target.png"
            Image.new("RGB", (32, 24)).save(source)
            Image.new("RGB", (32, 24)).save(target)
            row = {"source_path": str(source), "output_path": str(target), "prompt": "rainy street",
                   "passed": True, "utility": 0.3}
            path = root / "selected.jsonl"
            write_jsonl(path, [row, row])
            rows, usable, _ = load_training_rows([path], 0)
            self.assertEqual((len(rows), len(usable)), (2, 1))
            write_jsonl(path, [row, {**row, "prompt": "snowy street"}])
            with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
                load_training_rows([path], 0)

    def test_real_source_targets_and_missing_images_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, second, target = (root / name for name in ("source.png", "second.png", "target.png"))
            for image in (source, second, target):
                Image.new("RGB", (32, 24)).save(image)
            row = {"source_path": str(source), "output_path": str(target), "prompt": "rainy street",
                   "passed": True, "utility": 0.3}
            path = root / "selected.jsonl"
            write_jsonl(path, [{**row, "output_path": str(source)}])
            with self.assertRaisesRegex(ValueError, "generated image"):
                load_training_rows([path], 0)
            write_jsonl(path, [{**row, "output_path": str(second)}, {**row, "source_path": str(second)}])
            with self.assertRaisesRegex(ValueError, "another selected row"):
                load_training_rows([path], 0)
            write_jsonl(path, [{**row, "output_path": str(root / "missing.png")}])
            with self.assertRaisesRegex(ValueError, "not a readable image file"):
                load_training_rows([path], 0)

    def test_cli_rejects_invalid_optimizer_and_step_configuration(self):
        for flags in (["--steps", "0"], ["--batch-size", "0"], ["--alpha", "nan"],
                      ["--learning-rate", "0"], ["--warmup-steps", "-1"], ["--save-every", "0"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--selected", "selected.jsonl", "--output", "lora.safetensors", *flags])


class IcLightContractTests(unittest.TestCase):
    class BaseUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_in = torch.nn.Conv2d(4, 4, 1)
            self.last_kwargs = None

        def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
            self.last_kwargs = kwargs
            return SimpleNamespace(sample=self.conv_in(sample))

    def offset_for(self, unet):
        return {"conv_in.weight": torch.full((4, 8, 1, 1), 0.25),
                "conv_in.bias": torch.full_like(unet.conv_in.bias, 0.1)}

    def test_offset_applied_to_8_channel_conv_and_cfg_condition_repeated(self):
        model = self.BaseUNet()
        original = model.conv_in.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "offset.safetensors"
            sf.save_file(self.offset_for(model), path)
            _configure_unet(model, path)
            self.assertEqual(model.conv_in.in_channels, 8)
            self.assertTrue(torch.equal(model.conv_in.weight[:, :4], original + 0.25))
            self.assertTrue(torch.equal(model.conv_in.weight[:, 4:], torch.full_like(original, 0.25)))
            sample = torch.randn(4, 4, 2, 2)
            condition = torch.randn(2, 4, 2, 2)
            kwargs = {"concat_conds": condition, "scale": 0.7}
            output = model(sample, 0, None, cross_attention_kwargs=kwargs).sample
            expected = model.conv_in(torch.cat([sample, condition.repeat(2, 1, 1, 1)], dim=1))
            self.assertTrue(torch.equal(output, expected))
            self.assertEqual(model.last_kwargs["cross_attention_kwargs"], {"scale": 0.7})
            self.assertIs(kwargs["concat_conds"], condition)
            with self.assertRaisesRegex(ValueError, "compatible batch"):
                model(torch.randn(3, 4, 2, 2), 0, None, cross_attention_kwargs=kwargs)
            with self.assertRaisesRegex(ValueError, "concat_conds"):
                model(sample, 0, None)
            with self.assertRaisesRegex(ValueError, "configured once"):
                _configure_unet(model, path)

    def test_offset_rejects_unknown_keys_and_wrong_shapes(self):
        for wrong_shape in (False, True):
            model = self.BaseUNet()
            state = self.offset_for(model)
            if wrong_shape:
                state["conv_in.weight"] = torch.zeros(4, 4, 1, 1)
            else:
                state["unknown.weight"] = torch.zeros(1)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "offset.safetensors"
                sf.save_file(state, path)
                with self.assertRaises(ValueError):
                    _configure_unet(model, path)

    def test_source_pixels_match_serving_and_follow_vae_dtype_device(self):
        class FakeVAE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.marker = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
                self.config = SimpleNamespace(scaling_factor=0.5)

            def encode(self, tensor):
                self.tensor = tensor
                return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: tensor))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.png"
            Image.fromarray(np.arange(32 * 24 * 3, dtype=np.uint8).reshape(24, 32, 3)).save(path)
            vae = FakeVAE()
            with Image.open(path) as image:
                actual = _concat_condition(image, vae, 32, 24)
            expected = to_pixels(path, (32, 24)).unsqueeze(0).to(torch.bfloat16)
            self.assertEqual(vae.tensor.device.type, "cpu")
            self.assertTrue(torch.equal(vae.tensor, expected))
            self.assertTrue(torch.equal(actual, expected * 0.5))

    def test_text_encoding_matches_pipeline_attention_mask_policy(self):
        class Tokenizer:
            model_max_length = 77

            def __call__(self, prompts, **kwargs):
                self.prompts, self.kwargs = prompts, kwargs
                return SimpleNamespace(input_ids=torch.ones(len(prompts), 77, dtype=torch.long),
                                       attention_mask=torch.ones(len(prompts), 77, dtype=torch.long))

        class Encoder:
            config = SimpleNamespace(use_attention_mask=True)

            def __call__(self, ids, attention_mask):
                self.ids, self.mask = ids, attention_mask
                return (ids.float().unsqueeze(-1),)

        tokenizer, encoder = Tokenizer(), Encoder()
        prompts = ["rainy scene " * 100, "snowy scene"]
        output = encode_prompts(prompts, tokenizer, encoder, "cpu")
        self.assertEqual(tokenizer.prompts, prompts)
        self.assertEqual(tokenizer.kwargs["max_length"], 77)
        self.assertTrue(tokenizer.kwargs["truncation"])
        self.assertEqual(output.shape, (2, 77, 1))
        self.assertIsNotNone(encoder.mask)
        encoder.config.use_attention_mask = False
        encode_prompts(prompts, tokenizer, encoder, "cpu")
        self.assertIsNone(encoder.mask)


class CheckpointAndObjectiveTests(unittest.TestCase):
    def model_and_layers(self, dtype=torch.float32):
        model = TinyConditionalUNet().to(dtype)
        return model, inject_lora_into_unet(model, rank=2, alpha=4.0, dtype=dtype)

    def test_strict_loading_checks_scaling_and_is_atomic_on_bad_weights(self):
        _, layers = self.model_and_layers()
        state = lora_state_dict(layers)
        before = {key: value.clone() for key, value in state.items()}
        metadata = {"lora_rank": "2", "lora_alpha": "4.0", "num_layers": "1"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lora.safetensors"
            sf.save_file(state, path, metadata={**metadata, "lora_alpha": "8.0"})
            with self.assertRaisesRegex(ValueError, "rank/alpha"):
                load_lora_checkpoint(layers, path)
            state["to_q.lora_A"] = torch.full_like(state["to_q.lora_A"], 7.0)
            state["to_q.lora_B"][0, 0] = float("nan")
            sf.save_file(state, path, metadata=metadata)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_lora_checkpoint(layers, path)
        for key, value in lora_state_dict(layers).items():
            self.assertTrue(torch.equal(value, before[key]))

    def test_bad_metadata_and_inference_overflow_are_rejected(self):
        _, layers = self.model_and_layers(torch.float16)
        state = lora_state_dict(layers)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lora.safetensors"
            sf.save_file(state, path, metadata={"lora_rank": "0", "lora_alpha": "4"})
            with self.assertRaises(ValueError):
                read_lora_metadata(path)
            state["to_q.lora_B"].fill_(1e10)
            sf.save_file(state, path, metadata={"lora_rank": "2", "lora_alpha": "4", "num_layers": "1"})
            with self.assertRaisesRegex(ValueError, "overflow"):
                load_lora_checkpoint(layers, path)

    def test_fp32_checkpoint_runs_on_serving_dtype_and_registered_module_moves(self):
        _, source = self.model_and_layers()
        with torch.no_grad():
            source["to_q"].lora_B.fill_(0.2)
        model, target = self.model_and_layers(torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lora.safetensors"
            save_lora_checkpoint(source, path)
            load_lora_checkpoint(target, path)
        self.assertEqual(model.to_q(torch.ones(2, 8, dtype=torch.bfloat16)).dtype, torch.bfloat16)
        model.to(torch.float64)
        self.assertEqual(target["to_q"].lora_A.dtype, torch.float64)

    def test_epsilon_and_velocity_targets_and_bf16_forward(self):
        latent = torch.randn(2, 4, 2, 2, dtype=torch.bfloat16)
        noise = torch.randn(2, 4, 2, 2)
        timesteps = torch.tensor([0, 7])
        condition = torch.zeros_like(latent)
        for kind in ("epsilon", "v_prediction"):
            scheduler = DDPMScheduler(num_train_timesteps=10, prediction_type=kind)
            alpha = scheduler.alphas_cumprod[timesteps].reshape(2, 1, 1, 1)
            expected = noise if kind == "epsilon" else alpha.sqrt() * noise - (1 - alpha).sqrt() * latent.float()
            self.assertTrue(torch.allclose(diffusion_target(scheduler, latent, noise, timesteps), expected))

            class Oracle:
                def __call__(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs):
                    self.sample = sample
                    return SimpleNamespace(sample=expected)

            oracle = Oracle()
            loss = denoising_loss(oracle, scheduler, latent, condition, None, noise=noise, timesteps=timesteps)
            self.assertEqual(loss.item(), 0.0)
            self.assertEqual(oracle.sample.dtype, torch.bfloat16)
            noisy = (alpha.sqrt() * latent.float() + (1 - alpha).sqrt() * noise).to(torch.bfloat16)
            self.assertTrue(torch.equal(oracle.sample, noisy))
        scheduler = DDPMScheduler(prediction_type="sample")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            diffusion_target(scheduler, latent, noise, timesteps)

    def test_nonfinite_gradient_stops_before_optimizer_update(self):
        _, layers = self.model_and_layers()
        optimizer = torch.optim.AdamW(get_lora_parameters(layers), lr=1e-3)
        args = SimpleNamespace(steps=1, warmup_steps=0, grad_clip=1.0, log_every=1, save_every=1)
        schedule = make_lr_scheduler(optimizer, args)
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            train_steps(layers, optimizer, schedule, CyclingBatchSampler(1, 1, 0),
                        lambda _: layers["to_q"].lora_B[0, 0].sqrt(), args)
        self.assertEqual(torch.count_nonzero(layers["to_q"].lora_B).item(), 0)

    def test_delta_norm_includes_alpha_scaling(self):
        _, layers = self.model_and_layers()
        layer = layers["to_q"]
        with torch.no_grad():
            layer.lora_B.normal_()
            expected = float(torch.linalg.vector_norm((layer.lora_B @ layer.lora_A) * layer.scaling))
        self.assertAlmostEqual(lora_delta_norm(layers), expected, places=5)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.args = SimpleNamespace(steps=7, warmup_steps=2, grad_clip=1.0, log_every=4, save_every=3)
        self.config = {"steps": 7, "batch_size": 4, "selected_sha256": {"selected.jsonl": "abc"}}

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    def setup_training(self):
        torch.manual_seed(11)
        model = TinyConditionalUNet()
        freeze_non_lora_parameters(model)
        layers = inject_lora_into_unet(model, rank=2, alpha=4.0)
        unfreeze_lora_parameters(layers)
        optimizer = torch.optim.AdamW(get_lora_parameters(layers), lr=1e-2)
        schedule = make_lr_scheduler(optimizer, self.args)
        sampler = CyclingBatchSampler(3, 4, 13)
        scheduler = DDPMScheduler(num_train_timesteps=10)

        def loss(indices):
            latent = torch.randn(4, 4, 2, 2) + random.random() + float(np.random.normal())
            condition = torch.randn_like(latent) + torch.tensor(indices).reshape(4, 1, 1, 1)
            return denoising_loss(model, scheduler, latent, condition, None)

        return layers, optimizer, schedule, sampler, loss

    def seed_training(self):
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)

    def test_interrupted_training_matches_uninterrupted_weights_optimizer_and_history(self):
        full = self.setup_training()
        self.seed_training()
        with contextlib.redirect_stdout(io.StringIO()):
            reference_history = train_steps(*full, self.args)
        reference_random = (torch.rand(1), random.random(), float(np.random.normal()))
        interrupted = self.setup_training()
        self.seed_training()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lora.training.pt"

            def save_then_interrupt(step, history, running):
                save_training_state(path, *interrupted[:4], self.config, step, history, running)
                raise RuntimeError("simulated interruption")

            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "simulated"):
                train_steps(*interrupted, self.args, save_callback=save_then_interrupt)
            restored = self.setup_training()
            start, history, running = load_training_state(path, *restored[:4], self.config)
            self.assertEqual(start, 3)
            self.assertEqual(len(running), 3)  # A logging window straddles the recovery boundary.
            with contextlib.redirect_stdout(io.StringIO()):
                actual_history = train_steps(*restored, self.args, start_step=start, history=history, running=running)
        self.assertEqual(actual_history, reference_history)
        for name, weight in lora_state_dict(restored[0]).items():
            self.assertTrue(torch.equal(weight, lora_state_dict(full[0])[name]), name)
        for index, state in full[1].state_dict()["state"].items():
            for key, value in state.items():
                self.assertTrue(torch.equal(value, restored[1].state_dict()["state"][index][key]))
        self.assertEqual(full[2].state_dict(), restored[2].state_dict())
        self.assertEqual(full[3].state_dict(), restored[3].state_dict())
        actual_random = (torch.rand(1), random.random(), float(np.random.normal()))
        self.assertTrue(torch.equal(reference_random[0], actual_random[0]))
        self.assertEqual(reference_random[1:], actual_random[1:])

    def test_resume_rejects_changed_configuration_before_loading_weights(self):
        training = self.setup_training()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lora.training.pt"
            save_training_state(path, *training[:4], self.config, 0, [], [])
            before = lora_state_dict(training[0])
            with self.assertRaisesRegex(ValueError, "selected_sha256"):
                load_training_state(path, *training[:4], {**self.config, "selected_sha256": {"x": "changed"}})
            for key, value in lora_state_dict(training[0]).items():
                self.assertTrue(torch.equal(value, before[key]))

    def test_saved_state_uses_safe_loading_and_legacy_pickle_is_rejected(self):
        training = self.setup_training()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lora.training.pt"
            save_training_state(path, *training[:4], self.config, 0, [], [])
            state = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(state["format_version"], 2)
            self.assertIsInstance(state["numpy_rng"]["state"], list)
            state["format_version"], state["numpy_rng"] = 1, np.random.get_state()
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "Legacy or unsafe"):
                load_training_state(path, *training[:4], self.config)


if __name__ == "__main__":
    unittest.main()
