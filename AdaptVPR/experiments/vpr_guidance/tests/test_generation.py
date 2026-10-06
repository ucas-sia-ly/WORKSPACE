from __future__ import annotations

import copy
import csv
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import warnings
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from AdaptVPR.adapters import iclight_sd15_fc as adapter
from AdaptVPR.experiments.vpr_guidance import generate_dataset, iclight
from AdaptVPR.prompts.rules import global_negative_prompt
from AdaptVPR.verification.evaluator import DualTraitEvaluator, EvalResult


class TinyAttentionUNet(nn.Module):
    """Exercise the installed PEFT implementation with actual attention names."""

    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(4, 4)
        self.to_k = nn.Linear(4, 4)
        self.to_v = nn.Linear(4, 4)
        self.to_out = nn.Sequential(nn.Linear(4, 4))
        self.conv_in = nn.Linear(4, 4)

    def add_adapter(self, config):
        from peft import inject_adapter_in_model
        inject_adapter_in_model(config, self)

    def forward(self, value):
        return self.to_out(self.to_q(value) + self.to_k(value) + self.to_v(value))


class RecordingPipeline:
    def __init__(self, unet):
        self.unet = unet
        self.calls = []

    def __call__(self, **kwargs):
        kwargs.setdefault("guidance_scale", 7.5)
        self.calls.append(kwargs)
        image = kwargs.get("image") or Image.new("RGB", (kwargs["width"], kwargs["height"]), (80, 100, 120))
        return SimpleNamespace(images=[image])


class LoRACheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "lora.pt"
        self.unet = TinyAttentionUNet().half()
        self.base_state = copy.deepcopy(self.unet.state_dict())
        trainable = iclight.attach_lora(self.unet, 2, 4)
        self.assertTrue(trainable)
        for name, value in self.unet.named_parameters():
            self.assertEqual(value.requires_grad, "lora_" in name)
            if value.requires_grad:
                self.assertEqual(value.dtype, torch.float32)
                value.data.fill_(0.05)
        iclight.save_lora(self.unet, self.path, rank=2, alpha=4, extra={"step": 1})
        self.payload = torch.load(self.path, weights_only=True)

    def fresh(self):
        unet = TinyAttentionUNet().half()
        unet.load_state_dict(self.base_state, strict=True)
        return unet

    def test_real_peft_round_trip_keeps_base_and_output(self):
        target = self.fresh()
        payload = iclight.load_lora(target, self.path)
        self.assertEqual(payload["target_modules"], list(iclight.LORA_TARGET_MODULES))
        for name, value in target.state_dict().items():
            self.assertTrue(torch.equal(value, self.unet.state_dict()[name]), name)
        x = torch.randn(2, 4, dtype=torch.float16)
        self.assertTrue(torch.equal(target(x), self.unet(x)))
        self.assertTrue(all(p.requires_grad and p.dtype == torch.float32
                            for name, p in target.named_parameters() if "lora_" in name))

    def test_corrupt_checkpoints_fail_closed(self):
        cases = {}
        cases["empty"] = copy.deepcopy(self.payload)
        cases["empty"]["state_dict"] = {}
        cases["missing"] = copy.deepcopy(self.payload)
        cases["missing"]["state_dict"].pop(next(iter(cases["missing"]["state_dict"])))
        cases["base"] = copy.deepcopy(self.payload)
        cases["base"]["state_dict"]["conv_in.weight"] = torch.zeros(4, 4)
        cases["unexpected"] = copy.deepcopy(self.payload)
        cases["unexpected"]["state_dict"]["extra.lora_A.default.weight"] = torch.zeros(2, 4)
        cases["shape"] = copy.deepcopy(self.payload)
        key = next(iter(cases["shape"]["state_dict"]))
        cases["shape"]["state_dict"][key] = torch.zeros(3, 3)
        cases["nan"] = copy.deepcopy(self.payload)
        cases["nan"]["state_dict"][key].fill_(float("nan"))
        cases["rank"] = copy.deepcopy(self.payload)
        cases["rank"]["rank"] = 3
        cases["modules"] = copy.deepcopy(self.payload)
        cases["modules"]["target_modules"] = ["to_q"]
        cases["base_revision"] = copy.deepcopy(self.payload)
        cases["base_revision"]["base"]["base_model_revision"] = "wrong"
        for name, payload in cases.items():
            with self.subTest(case=name):
                torch.save(payload, self.path)
                with self.assertRaises(ValueError):
                    iclight.load_lora(self.fresh(), self.path)

    def test_legacy_checkpoint_is_validated_with_explicit_warning(self):
        payload = {key: self.payload[key] for key in ("rank", "alpha", "state_dict", "extra")}
        torch.save(payload, self.path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            iclight.load_lora(self.fresh(), self.path)
        self.assertTrue(any("legacy LoRA" in str(item.message) for item in caught))


class SamplingPolicyTests(unittest.TestCase):
    def test_two_stage_calls_match_original_adapter_endpoint(self):
        source = Image.fromarray(np.random.default_rng(10).integers(0, 256, (19, 27, 3), dtype=np.uint8))
        prompt = "Frozen released prompt without appended text."
        unet = nn.Identity()
        t2i, i2i = RecordingPipeline(unet), RecordingPipeline(unet)
        original_generator = torch.Generator
        def cpu_generator(**kwargs):
            return original_generator(device="cpu")
        def concat(image, vae, width, height):
            return torch.tensor(np.asarray(image.resize((width, height))).copy())
        for condition in ("snow", "rain", "rainy_night"):
            with self.subTest(condition=condition), tempfile.TemporaryDirectory() as tmp:
                source_path = Path(tmp) / "source.jpg"
                source.save(source_path, format="JPEG", quality=95)
                policy = iclight.sampling_policy(condition)
                with patch.object(torch, "Generator", side_effect=cpu_generator), \
                     patch.object(adapter, "_concat_condition", side_effect=concat), \
                     patch.object(adapter.state, "pipe_t2i", t2i), \
                     patch.object(adapter.state, "pipe_i2i", i2i), \
                     patch.object(adapter.state, "vae", object()), \
                     patch.dict(os.environ, {"ICLIGHT_OUTPUT_DIR": tmp}):
                    adapter.generate(adapter.GenerateRequest(
                        image_path=str(source_path), prompt=prompt,
                        negative_prompt=global_negative_prompt(), seed=123,
                        highres_denoise=policy["highres_denoise"],
                    ))
                    iclight.generate_released(t2i, i2i, object(), source, prompt, None, 123, condition)
                for pipe in (t2i, i2i):
                    original, experiment = pipe.calls[-2:]
                    self.assertEqual(original.keys(), experiment.keys())
                    for key in original:
                        if key == "generator":
                            self.assertEqual(original[key].initial_seed(), experiment[key].initial_seed())
                        elif key == "cross_attention_kwargs":
                            self.assertTrue(torch.equal(original[key]["concat_conds"], experiment[key]["concat_conds"]))
                        elif key == "image":
                            self.assertEqual(original[key].tobytes(), experiment[key].tobytes())
                        else:
                            self.assertEqual(original[key], experiment[key], key)
                    self.assertEqual(experiment["prompt"], prompt)

    def test_explicit_negative_prompt_and_invalid_condition(self):
        self.assertEqual(iclight.released_negative_prompt("custom"), "custom")
        self.assertEqual(iclight.released_negative_prompt(), global_negative_prompt())
        with self.assertRaises(ValueError):
            iclight.sampling_policy("snow+vehicle")

    def test_disjoint_stage_unets_fail(self):
        with self.assertRaisesRegex(ValueError, "share"):
            iclight.generate_released(RecordingPipeline(nn.Identity()), RecordingPipeline(nn.Identity()),
                                      None, Image.new("RGB", (8, 8)), "snow", None, 1, "snow")


class VerifierTests(unittest.TestCase):
    def checker(self):
        checker = DualTraitEvaluator(mock=True)
        checker.mock = False
        checker.torch, checker.functional = torch, F
        checker._compute_s_geo = lambda ref, gen: 1.0
        return checker

    def test_nan_clip_similarity_raises_instead_of_passing(self):
        checker = self.checker()
        checker._extract_clip_feature = lambda image: torch.full((1, 3), float("nan"))
        image = Image.new("RGB", (8, 8))
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            checker.evaluate(image, image, entry={"route": "global"})

    def test_invalid_clip_embeddings_fail_before_normalization(self):
        class Inputs(dict):
            def to(self, device):
                return self
        image = Image.new("RGB", (8, 8))
        for feature in (torch.zeros(1, 3), torch.full((1, 3), float("nan")),
                        torch.full((1, 3), float("inf"))):
            with self.subTest(feature=feature):
                checker = self.checker()
                checker.processor = lambda **kwargs: Inputs()
                checker.model = SimpleNamespace(get_image_features=lambda **kwargs: feature,
                                                config=SimpleNamespace(projection_dim=3))
                with self.assertRaises(FloatingPointError):
                    checker._extract_clip_feature(image)

    def test_nonfinite_raw_scores_and_inconsistent_pass_flag_rejected(self):
        checker = self.checker()
        checker._compute_s_div = lambda ref, gen: float("nan")
        image = Image.new("RGB", (8, 8))
        with self.assertRaises(FloatingPointError):
            checker.evaluate(image, image, route="global")
        with self.assertRaises(ValueError):
            generate_dataset._accepted_global_result(EvalResult(s_geo=0.9, s_div=0.01, passed=True))
        self.assertTrue(generate_dataset._accepted_global_result(EvalResult(s_geo=0.9, s_div=0.2)))


class GenerationEntryPointTests(unittest.TestCase):
    def test_generation_needs_no_salad_and_manifest_contains_only_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "Images" / "Bangkok"
            images.mkdir(parents=True)
            dataframes = root / "Dataframes"
            dataframes.mkdir()
            rows, prompts = [], []
            for place in (1, 2):
                row = {"city_id": "Bangkok", "place_id": place, "year": 2017, "month": 1,
                       "northdeg": 0, "lat": 13.0, "lon": 100.0, "panoid": f"pano_{place}"}
                rows.append(row)
                name = f"Bangkok_{place:07d}_2017_01_000_13.0_100.0_pano_{place}.jpg"
                Image.new("RGB", (16, 16), (place * 40, 80, 100)).save(images / name)
                prompts.append({"sample_id": f"s{place}", "source_id": name, "route": "global",
                                "condition": "snow", "prompt": "frozen snow prompt"})
            with (dataframes / "Bangkok.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            prompt_path = root / "prompts.jsonl"
            prompt_path.write_text("\n".join(json.dumps(row) for row in prompts), encoding="utf-8")
            out = root / "generated"
            unet = nn.Identity()
            pipes = (RecordingPipeline(unet), RecordingPipeline(unet), object())
            verifier = SimpleNamespace(
                _load_matcher=lambda: None, matcher_name="test", img_size=512, n_kpts=2048,
                model=SimpleNamespace(config=SimpleNamespace(_name_or_path="test-clip")),
                evaluate=lambda source, generated, entry: EvalResult(
                    s_geo=0.9, s_div=0.2 if np.asarray(source)[0, 0, 0] < 60 else 0.1,
                    passed=bool(np.asarray(source)[0, 0, 0] < 60)),
            )
            argv = ["generate_dataset", "--prompts", str(prompt_path), "--image-root", str(root / "Images"),
                    "--output-dir", str(out)]
            with patch("sys.argv", argv), patch.object(generate_dataset, "load_iclight", return_value=pipes), \
                 patch.object(generate_dataset, "DualTraitEvaluator", return_value=verifier), \
                 patch.object(generate_dataset, "generate_released", return_value=Image.new("RGB", (16, 16))), \
                 patch.object(generate_dataset, "load_salad", side_effect=AssertionError("SALAD must not load")):
                generate_dataset.main()
                with self.assertRaises(FileExistsError):
                    generate_dataset.main()
            records = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]
            accepted = [json.loads(line) for line in (out / "synthetic_manifest.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual([row["sample_id"] for row in accepted], ["s1"])
            self.assertEqual(accepted[0]["place_id"], "0000001")
            self.assertIsNone(accepted[0]["salad_preservation_cosine"])
            self.assertTrue(Path(accepted[0]["generated_path"]).is_file())
            self.assertEqual(accepted[0]["verifier_policy"]["matcher_name"], "test")
            c_out = root / "generated_c"
            lora_path = root / "trained.pt"
            torch.save({"extra": {}}, lora_path)
            c_argv = list(argv)
            c_argv[-1] = str(c_out)
            c_argv.extend(["--lora", str(lora_path)])
            with patch("sys.argv", c_argv), patch.object(generate_dataset, "load_iclight", return_value=pipes), \
                 patch.object(generate_dataset, "DualTraitEvaluator", return_value=verifier), \
                 patch.object(generate_dataset, "generate_released", return_value=Image.new("RGB", (16, 16))), \
                 patch.object(generate_dataset, "load_salad", side_effect=AssertionError("SALAD must not load")), \
                 patch.object(generate_dataset, "load_lora") as load_adapter:
                generate_dataset.main()
                load_adapter.assert_called_once_with(unet, lora_path)
            c_records = [json.loads(line) for line in (c_out / "records.jsonl").read_text().splitlines()]
            for original, aware in zip(records, c_records):
                self.assertEqual(original["generator_variant"], "released")
                self.assertEqual(aware["generator_variant"], "vpr_lora")
                for field in ("sample_id", "source_id", "source_sha256", "city", "place_id", "condition",
                              "prompt", "negative_prompt", "seed", "sampling_policy", "verifier_policy", "passed"):
                    self.assertEqual(original[field], aware[field], field)


if __name__ == "__main__":
    unittest.main()
