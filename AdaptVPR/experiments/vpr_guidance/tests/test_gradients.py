"""CPU correctness tests; tiny models test mathematics, not model quality."""
from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, UNet2DConditionModel
from PIL import Image

from AdaptVPR.experiments.vpr_guidance.iclight import (
    attach_lora, decode_latent_01, load_lora, lora_state_dict, sampling_policy, save_lora,
)
from AdaptVPR.experiments.vpr_guidance import teacher
from AdaptVPR.experiments.vpr_guidance.teacher import SaladTeacher, file_sha256, pil_tensor
from AdaptVPR.experiments.vpr_guidance.train_generator import (
    first_order_guidance_proxy, predict_x0, prepare_training_log, read_manifest, restore_training_state,
    prepare_tensorboard, training_state, validate_args, write_tensorboard_record,
)


class TinyTokens(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = torch.nn.Conv2d(3, 4, 14, stride=14)
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
        self.norm = torch.nn.LayerNorm(4)

    def prepare_tokens_with_masks(self, image):
        patches = self.patch(image).flatten(2).transpose(1, 2)
        return torch.cat((patches.mean(dim=1, keepdim=True), patches), dim=1)


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyTokens()
        self.norm_layer = True
        self.return_token = True
        self.num_channels = 4

    def forward(self, image):
        raise AssertionError("teacher must install its differentiable backbone forward")


class TinyAggregator(torch.nn.Module):
    num_clusters, cluster_dim, token_dim = 1, 4, 2

    def forward(self, pair):
        features, token = pair
        return torch.cat((features.mean(dim=(-1, -2)), token[:, :2]), dim=-1)


class TinySalad(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TinyBackbone()
        self.aggregator = TinyAggregator()

    def forward(self, image):
        return self.aggregator(self.backbone(image))


class TinyVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Conv2d(4, 3, 1)
        self.config = SimpleNamespace(scaling_factor=0.18215)

    @property
    def dtype(self):
        return self.proj.weight.dtype

    def decode(self, latent, return_dict=False):
        return (self.proj(latent).tanh(),)


def tiny_unet():
    return UNet2DConditionModel(
        sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
        block_out_channels=(8, 16), norm_num_groups=4,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=8, attention_head_dim=2,
    )


class TeacherLoadingTests(unittest.TestCase):
    def check_loading(self, repo, expected_repo, expected_source):
        calls = []
        model = object()

        def upstream_load(repo_or_dir, name, *args, **kwargs):
            calls.append((repo_or_dir, name, args, kwargs))
            if name == "dinov2_salad":
                # Reproduce upstream SALAD's unqualified nested dependency.
                torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            return model

        with patch.object(torch.hub, "load", side_effect=upstream_load) as hub_load, \
                patch.object(teacher, "SaladTeacher") as constructor:
            result = teacher.load_salad(device="cpu", repo=repo)
            self.assertIs(result, constructor.return_value)
            constructor.assert_called_once_with(model, device="cpu")
            self.assertIs(torch.hub.load, hub_load)
        self.assertEqual(calls, [
            (expected_repo, "dinov2_salad", (),
             {"pretrained": True, "trust_repo": True, "source": expected_source}),
            ("facebookresearch/dinov2:main", "dinov2_vitb14", (), {}),
        ])

    def test_default_and_nested_repositories_have_explicit_refs(self):
        self.check_loading(teacher.SALAD_REPO, "serizba/salad:main", "github")

    def test_local_salad_still_pins_nested_dinov2(self):
        with tempfile.TemporaryDirectory() as repo:
            self.check_loading(repo, repo, "local")

    def test_explicit_salad_ref_is_preserved(self):
        self.check_loading("serizba/salad:custom", "serizba/salad:custom", "github")

    def test_hub_loader_restored_after_nested_failure(self):
        def upstream_load(repo_or_dir, name, **kwargs):
            if name == "dinov2_salad":
                return torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            raise RuntimeError("backbone failed")

        with patch.object(torch.hub, "load", side_effect=upstream_load) as hub_load:
            with self.assertRaisesRegex(RuntimeError, "backbone failed"):
                teacher.load_salad(device="cpu")
            self.assertIs(torch.hub.load, hub_load)

    def test_explicit_ref_avoids_torch_hub_branch_probe(self):
        with patch.object(torch.hub, "urlopen", side_effect=AssertionError("network probe")):
            self.assertEqual(torch.hub._parse_repo_info("facebookresearch/dinov2:main"),
                             ("facebookresearch", "dinov2", "main"))



class GradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(42)

    def test_frozen_teacher_retains_generated_pixel_gradient(self):
        teacher = SaladTeacher(TinySalad(), device="cpu")
        generated = torch.rand(1, 3, 40, 64, requires_grad=True)
        with torch.no_grad():
            source = teacher(torch.rand_like(generated))
        descriptor = teacher(generated)
        (1 - (descriptor * source).sum(-1)).mean().backward()
        self.assertIsNotNone(generated.grad)
        self.assertGreater(generated.grad.norm().item(), 0)
        self.assertTrue(torch.isfinite(generated.grad).all())
        self.assertFalse(any(p.requires_grad or p.grad is not None for p in teacher.model.parameters()))

    def test_source_and_generated_use_identical_preprocessing(self):
        teacher = SaladTeacher(TinySalad(), device="cpu")
        image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (43, 61, 3), dtype=np.uint8))
        with torch.no_grad():
            torch.testing.assert_close(teacher.from_pil(image), teacher(pil_tensor(image)), rtol=0, atol=0)

    def test_scheduler_x0_inverse_and_prediction_type(self):
        scheduler = DDIMScheduler(beta_schedule="scaled_linear", beta_start=0.00085,
                                  beta_end=0.012, clip_sample=False, steps_offset=1)
        scheduler.set_timesteps(25)
        original, noise = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
        for timestep in scheduler.timesteps:
            noisy = scheduler.add_noise(original, noise, timestep.reshape(1))
            torch.testing.assert_close(predict_x0(scheduler, noisy, noise, timestep), original,
                                       rtol=2e-5, atol=3e-6)
        scheduler.register_to_config(prediction_type="v_prediction")
        with self.assertRaisesRegex(ValueError, "epsilon"):
            predict_x0(scheduler, original, noise, 1)

    def test_legacy_two_pass_utility_matches_full_backward(self):
        unet = tiny_unet()
        trainable = attach_lora(unet, rank=2, alpha=2)
        unet.enable_gradient_checkpointing()
        unet.train()
        teacher = SaladTeacher(TinySalad(), device="cpu")
        vae = TinyVAE().requires_grad_(False).eval()
        scheduler = DDIMScheduler(beta_schedule="scaled_linear", beta_start=0.00085,
                                  beta_end=0.012, clip_sample=False)
        latent, noise = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
        timestep, text = torch.tensor([161]), torch.randn(1, 5, 8)
        noisy = scheduler.add_noise(latent, noise, timestep)
        with torch.no_grad():
            source_descriptor = teacher(torch.rand(1, 3, 8, 8))
            baseline = decode_latent_01(latent, vae)

        def guidance(x0):
            image = decode_latent_01(x0.float(), vae)
            vpr = (1 - (teacher(image) * source_descriptor).sum(-1)).mean()
            return 0.1 * vpr + 0.05 * F.l1_loss(image, baseline), vpr

        eps = unet(noisy, timestep, encoder_hidden_states=text).sample
        x0 = predict_x0(scheduler, noisy, eps, int(timestep))
        _, vpr = guidance(x0)
        vpr.backward()
        self.assertGreater(sum(p.grad.square().sum().item() for p in trainable), 0)
        unet.zero_grad(set_to_none=True)

        eps = unet(noisy, timestep, encoder_hidden_states=text).sample
        guide, _ = guidance(predict_x0(scheduler, noisy, eps, int(timestep)))
        (F.mse_loss(eps, noise) + guide).backward()
        direct = [p.grad.clone() for p in trainable]
        unet.zero_grad(set_to_none=True)

        with torch.no_grad():
            eps_a = unet(noisy, timestep, encoder_hidden_states=text).sample
            predicted = predict_x0(scheduler, noisy, eps_a, int(timestep))
        leaf = predicted.detach().float().requires_grad_(True)
        guide, _ = guidance(leaf)
        grad_x0, = torch.autograd.grad(guide, leaf)
        eps_b = unet(noisy, timestep, encoder_hidden_states=text).sample
        torch.testing.assert_close(eps_a, eps_b, rtol=0, atol=0)
        proxy = first_order_guidance_proxy(
            predict_x0(scheduler, noisy, eps_b, int(timestep)), grad_x0,
        )
        (F.mse_loss(eps_b, noise) + proxy).backward()
        for expected, parameter in zip(direct, trainable):
            torch.testing.assert_close(parameter.grad, expected, rtol=1e-6, atol=1e-7)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertFalse(any(p.grad is not None for n, p in unet.named_parameters() if "lora_" not in n))
        self.assertFalse(any(p.grad is not None for p in teacher.model.parameters()))
        self.assertFalse(any(p.grad is not None for p in vae.parameters()))

    def test_descriptor_cache_rejects_stale_weights_source_and_legacy_tensor(self):
        teacher = SaladTeacher(TinySalad(), device="cpu")
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.png"
            cache = Path(temporary) / "source.pt"
            Image.new("RGB", (31, 35), "white").save(source)
            with torch.no_grad():
                descriptor = teacher.from_pil(Image.open(source))
            teacher.save_source_descriptor(cache, descriptor, source)
            torch.testing.assert_close(teacher.load_source_descriptor(cache, source), descriptor)
            changed_teacher = SaladTeacher(TinySalad(), device="cpu")
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                changed_teacher.load_source_descriptor(cache, source)
            Image.new("RGB", (31, 35), "black").save(source)
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                teacher.load_source_descriptor(cache, source)
            torch.save(descriptor, cache)
            with self.assertRaisesRegex(ValueError, "no validated metadata"):
                teacher.load_source_descriptor(cache, source)

    def test_invalid_training_windows_and_zero_loss_are_rejected(self):
        arguments = dict(max_steps=1, rank=2, alpha=2, timestep_window=10, save_every=1,
                         lr=1e-4, grad_clip=1, lambda_diff=1, lambda_meta=1., lambda_vpr=None, lambda_keep=0.05,
                         meta_inner_lr=1e-3, meta_inner_steps=1, meta_places=4,
                         meta_support_real_per_place=1, meta_query_real_per_place=2,
                         meta_image_size=224, meta_train_backbone_blocks=0, generator_grad_scale=65536.)
        validate_args(SimpleNamespace(**arguments))
        for name, value in (("timestep_window", 0), ("timestep_window", 26),
                            ("save_every", 0), ("max_steps", 0), ("lambda_meta", -1)):
            with self.assertRaises(ValueError):
                validate_args(SimpleNamespace(**{**arguments, name: value}))
        with self.assertRaises(ValueError):
            validate_args(SimpleNamespace(**{**arguments, "lambda_diff": 0,
                                              "lambda_meta": 0, "lambda_keep": 0}))

    def test_resume_restores_lora_optimizer_and_both_sampling_rngs(self):
        unet = tiny_unet()
        base_state = {name: value.clone() for name, value in unet.state_dict().items()}
        trainable = attach_lora(unet, rank=2, alpha=2)
        optimizer = torch.optim.AdamW(trainable, lr=1e-3)
        sample_rng = random.Random(42)
        random.seed(42)
        config = {"manifest_sha256": "test", "teacher_sha256": "test"}

        def step(model, opt, rng):
            noisy, text = torch.randn(1, 4, 8, 8), torch.randn(1, 5, 8)
            scale = rng.random() + random.random()
            opt.zero_grad(set_to_none=True)
            loss = scale * model(noisy, torch.tensor([161]), encoder_hidden_states=text).sample.square().mean()
            loss.backward()
            opt.step()
            return loss.detach()

        step(unet, optimizer, sample_rng)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "step_000001.pt"
            save_lora(unet, checkpoint, rank=2, alpha=2,
                      extra=training_state(1, config, optimizer, sample_rng))
            expected_loss = step(unet, optimizer, sample_rng)
            expected_weights = {name: value.clone() for name, value in lora_state_dict(unet).items()}
            restored = tiny_unet()
            restored.load_state_dict(base_state, strict=True)
            payload = load_lora(restored, checkpoint)
            restored_optimizer = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=1e-3)
            restored_rng = random.Random(999)
            self.assertEqual(restore_training_state(payload["extra"], config, restored_optimizer,
                                                    restored_rng, max_steps=3), 1)
            actual_loss = step(restored, restored_optimizer, restored_rng)
            torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
            for name, value in lora_state_dict(restored).items():
                torch.testing.assert_close(value, expected_weights[name], rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "settings differ"):
                restore_training_state(payload["extra"], {"manifest_sha256": "changed"},
                                       restored_optimizer, restored_rng, max_steps=3)

    def test_tensorboard_rebuilds_history_without_duplicate_or_stale_steps(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        config = {"lambda_diff": 1., "lambda_vpr": .1, "lambda_keep": .05, "lr": 1e-4}
        record = {"condition": "snow", "timestep": 161, "loss_diff": 1., "loss_vpr": .2,
                  "loss_keep": .3, "loss_total": 1.035, "loss_proxy": -.1,
                  "salad_cosine": .8, "grad_norm": .4, "guidance_x0_grad_norm": .05}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            history = root / "train.jsonl"
            history.write_text("".join(json.dumps({**record, "step": step}) + "\n"
                                       for step in (1, 2)))
            python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
            writer = prepare_tensorboard(root / "events", history, 2, config)
            self.assertEqual(random.getstate(), python_rng)
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
            # Simulate events flushed after the checkpoint but before a crash.
            write_tensorboard_record(writer, {**record, "step": 4}, config)
            writer.close()
            writer = prepare_tensorboard(root / "events", history, 2, config)
            write_tensorboard_record(writer, {**record, "step": 3}, config)
            writer.close()
            events = EventAccumulator(str(root / "events"), size_guidance={"scalars": 0}).Reload()
            for tag in ("loss/total", "quality/salad_cosine", "weighted_loss/vpr"):
                self.assertEqual([event.step for event in events.Scalars(tag)], [1, 2, 3])
            self.assertAlmostEqual(events.Scalars("weighted_loss/vpr")[-1].value, .02)
            self.assertAlmostEqual(events.Scalars("loss/guidance_proxy")[-1].value, -.1)

    def test_training_manifest_rejects_legacy_and_changed_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, baseline, descriptor = root / "source.png", root / "g1.png", root / "g1.pt"
            Image.new("RGB", (8, 8), "white").save(source)
            Image.new("RGB", (8, 8), "gray").save(baseline)
            torch.save(torch.ones(1, 6), descriptor)
            row = {"sample_id": "g1", "route": "global", "condition": "snow", "prompt": "snow",
                   "city": "test", "place_id": "0000001", "source_path": str(source),
                   "baseline_path": str(baseline), "source_descriptor": str(descriptor),
                   "baseline_sha256": file_sha256(baseline), "source_sha256": file_sha256(source),
                   "sampling_policy": sampling_policy("snow")}
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps(row), encoding="utf-8")
            self.assertEqual(read_manifest(manifest)[0]["sample_id"], "g1")
            legacy = {k: v for k, v in row.items() if k != "baseline_sha256"}
            manifest.write_text(json.dumps(legacy), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "rerun prepare_data"):
                read_manifest(manifest)
            manifest.write_text(json.dumps(row), encoding="utf-8")
            Image.new("RGB", (8, 8), "black").save(baseline)
            with self.assertRaisesRegex(ValueError, "baseline content mismatch"):
                read_manifest(manifest)

    def test_resume_rejects_logs_ahead_of_checkpoint_and_other_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {"manifest_sha256": "one"}
            with prepare_training_log(root, 0, config, False) as log:
                log.write(json.dumps({"step": 1}) + "\n" + json.dumps({"step": 2}) + "\n")
            with self.assertRaisesRegex(ValueError, "does not end"):
                prepare_training_log(root, 1, config, True)
            with self.assertRaisesRegex(ValueError, "different generator run"):
                prepare_training_log(root, 2, {"manifest_sha256": "two"}, True)
            with prepare_training_log(root, 2, config, True):
                pass
            checkpoints = root / "checkpoints"
            checkpoints.mkdir()
            (checkpoints / "step_000003.pt").touch()
            with self.assertRaisesRegex(ValueError, "checkpoints after"):
                prepare_training_log(root, 2, config, True)


if __name__ == "__main__":
    unittest.main()
