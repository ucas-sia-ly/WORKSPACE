"""Regression tests for online sampling, gate invariants and boundary journals."""
from collections import Counter
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from AdaptVPR.experiments.vpr_guidance.data import balanced_chunks, sample_seed
from AdaptVPR.experiments.vpr_guidance.train_online_generator import (
    recover_journal, sample_training_entry, summarize_round,
)
from AdaptVPR.experiments.vpr_guidance.train_generator import train_accepted_step


class OnlineTests(unittest.TestCase):
    def test_seed_cross_process_and_final_pairing(self):
        expected = sample_seed(42, "sample1")
        code = "from AdaptVPR.experiments.vpr_guidance.data import sample_seed; print(sample_seed(42, 'sample1'))"
        actual = subprocess.check_output([sys.executable, "-c", code], text=True)
        self.assertEqual(expected, int(actual))
        self.assertNotEqual(expected, sample_seed(42, "sample2"))
        self.assertNotEqual(sample_seed(42, "sample1", 0, 0), sample_seed(42, "sample1", 1, 0))
        self.assertNotEqual(sample_seed(42, "sample1", 0, 0), sample_seed(42, "sample1", 0, 1))

    def test_balanced_passes_cover_every_sample_once(self):
        rows = [{"sample_id": f"{c}{i}", "condition": c} for c, n in
                (("snow", 30), ("night", 30), ("rain", 30), ("fog", 4)) for i in range(n)]
        first = list(balanced_chunks(rows, 16, 42, 0))
        self.assertEqual(Counter(r["condition"] for r in first[0]), dict(snow=4, night=4, rain=4, fog=4))
        self.assertEqual({r["sample_id"] for chunk in first for r in chunk}, {r["sample_id"] for r in rows})
        self.assertEqual(sum(map(len, first)), len(rows))
        self.assertEqual(first, list(balanced_chunks(rows, 16, 42, 0)))
        self.assertNotEqual(first, list(balanced_chunks(rows, 16, 42, 1)))

    def test_legacy_single_row_sampler_and_strict_acceptance(self):
        current = [{"sample_id": "new", "passed": True, "eligible_for_training": True}]
        old = [{"sample_id": "old", "passed": True, "eligible_for_training": True}]
        rng = random.Random(42)
        n = sum(sample_training_entry(current, [old], rng)["sample_id"] == "new" for _ in range(2000))
        self.assertTrue(900 < n < 1100)
        self.assertEqual(sample_training_entry(current, [], rng), current[0])
        with self.assertRaises(ValueError):
            sample_training_entry([], [old], rng)
        rejected = {"passed": False, "eligible_for_training": False}
        with self.assertRaises(ValueError):
            train_accepted_step(rejected, step=1, args=None, t2i=None, vae=None, unet=None,
                                teacher=None, scheduler=None, timesteps=None, opt=None, trainable=None, rng=None)

    def test_round_audit_includes_rejected_scores(self):
        records = [{"condition": c, "passed": passed, "s_geo": .9, "s_div": div,
                    "salad_preservation_cosine": .8} for c, passed, div in
                   (("snow", True, .2), ("night", False, .1))]
        stats = summarize_round(records, 0, 1, 1)
        self.assertEqual(stats["accepted_count"], 1)
        self.assertEqual(stats["pass_rate"], .5)
        self.assertEqual(stats["conditions"]["night"]["pass_rate"], 0)

    def test_recovery_archives_uncommitted_journal_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            path.write_text('{"step":1}\n{"step":2}\n{"step":3}\n{"ste')
            recover_journal(path, "step", 2)
            self.assertEqual([json.loads(s)["step"] for s in path.read_text().splitlines()], [1, 2])
            self.assertEqual(len(list(Path(tmp).glob("train.jsonl.interrupted-*"))), 1)


if __name__ == "__main__":
    unittest.main()


class MatchedExposureTests(unittest.TestCase):
    def test_b_and_c_have_identical_per_place_counts_with_different_pools(self):
        from AdaptVPR.experiments.vpr_guidance.tests.test_data_pipeline import make_mixed_fixture
        from AdaptVPR.experiments.vpr_guidance.mixed_salad import MixedGSVCitiesDataset
        with tempfile.TemporaryDirectory() as tmp:
            root, b_manifest, rows = make_mixed_fixture(tmp, places=9)
            c_manifest = Path(tmp) / "c.jsonl"
            c_manifest.write_text("\n".join(json.dumps(r) for r in rows[:5]))
            plan = Path(tmp) / "plan.json"
            plan.write_text(json.dumps({"total_places": 9, "seed": 42, "requested_ratio": [8, 1],
                                        "capacities": [{"city": "Bangkok", "place_id": i, "capacity": 1}
                                                       for i in range(1, 6)]}))
            b = MixedGSVCitiesDataset(root, b_manifest, ["Bangkok"], shared_mix_plan=plan)
            c = MixedGSVCitiesDataset(root, c_manifest, ["Bangkok"], shared_mix_plan=plan)
            for epoch in range(3):
                b.set_epoch(epoch); c.set_epoch(epoch)
                self.assertEqual(b.mix_stats, c.mix_stats)
                self.assertEqual(b._synthetic_quota, c._synthetic_quota)


class OnlineOrchestrationTests(unittest.TestCase):
    def test_refresh_uses_updated_generator_and_boundary_resume_is_exact(self):
        import argparse
        from contextlib import ExitStack, redirect_stdout
        from io import StringIO
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch
        from torch import nn
        from PIL import Image
        from AdaptVPR.experiments.vpr_guidance import train_online_generator as online
        from AdaptVPR.experiments.vpr_guidance.iclight import lora_state_dict
        from AdaptVPR.experiments.vpr_guidance.teacher import file_sha256, PREPROCESSING_VERSION
        from AdaptVPR.experiments.vpr_guidance.tests.test_generation import TinyAttentionUNet
        from AdaptVPR.verification.evaluator import EvalResult

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from AdaptVPR.experiments.vpr_guidance.tests.test_data_pipeline import make_mixed_fixture
            from AdaptVPR.experiments.vpr_guidance.tests.test_bilevel import meta_args
            gsv_root, _, fixture_rows = make_mixed_fixture(tmp, places=4)
            rows = []
            for i in range(4):
                source = Path(fixture_rows[i]['source_path'])
                descriptor = root / f"s{i}.pt"
                torch.save(torch.ones(1, 1), descriptor)
                rows.append({"sample_id": f"s{i}", "source_path": str(source),
                             "source_sha256": file_sha256(source), "source_descriptor": str(descriptor),
                             "route": "global", "condition": "snow", "prompt": "snow", "negative_prompt": "negative",
                             "city": "Bangkok", "place_id": str(i + 1), "teacher_sha256": "fixed",
                             "preprocessing_version": PREPROCESSING_VERSION,
                             "sampling_policy": online.sampling_policy("snow")})
            manifest = root / "source_manifest.jsonl"
            manifest.write_text("\n".join(json.dumps(r) for r in rows))
            class Teacher:
                model_fingerprint = "fixed"
                device = "cpu"
                model = nn.Identity()
                def load_source_descriptor(self, path, source):
                    return torch.load(path, weights_only=True)
                def from_pil(self, image):
                    return torch.ones(1, 1)
            class Scheduler:
                timesteps = torch.arange(25)
                def set_timesteps(self, *args, **kwargs):
                    pass
            verifier = SimpleNamespace(_load_matcher=lambda: None, matcher_name="test", img_size=512,
                                       n_kpts=2048, model=None,
                                       evaluate=lambda *a, **kw: EvalResult(s_geo=.9, s_div=.2, passed=True))
            traces = []
            def pipeline():
                unet = TinyAttentionUNet()
                unet.enable_gradient_checkpointing = lambda: None
                pipe = SimpleNamespace(unet=unet, text_encoder=nn.Identity(), scheduler=SimpleNamespace(config={}))
                return pipe, pipe, nn.Identity()
            def generate(pipe, *args):
                traces.append(float(sum(value.sum() for value in lora_state_dict(pipe.unet).values())))
                return Image.new("RGB", (8, 8), (20, 40, 60))
            fail = [False]
            def update(episode, *, step, unet, rng, **kwargs):
                episode.validate()
                row = episode.places[0].synthetic_row
                if fail[0] and step == 2:
                    raise RuntimeError("simulated interrupted round")
                increment = rng.random() + float(torch.rand(()))
                with torch.no_grad():
                    next(p for p in unet.parameters() if p.requires_grad).add_(increment)
                return {"step": step, "sample_ids": [p.synthetic_row["sample_id"] for p in episode.places],
                        "generator/loss_diff": 1., "generator/loss_meta": .1,
                        "generator/loss_keep": .1, "generator/loss_total": 1.1,
                        "generator/lora_grad_norm": 1., "generator/meta_only_lora_grad_norm": 1.,
                        "generator/timestep": 1}
            def run(out, resume=None, meta_places=2):
                args = SimpleNamespace(prompts=manifest, gsv_root=gsv_root, salad_root=root,
                                       output_dir=out, source_manifest=manifest, conditions=["snow"],
                                       generation_passes=1, chunk_size=2, train_steps_per_chunk=1,
                                       replay_rounds=2, rank=8, alpha=8, timestep_window=10, seed=42,
                                       lr=1e-4, lambda_diff=1., lambda_keep=.05,
                                       grad_clip=1., tensorboard_dir=None, disable_tensorboard=True, resume=resume)
                for name, value in vars(meta_args()).items():
                    if name.startswith('meta_') or name in ('lambda_meta', 'lambda_vpr', 'generator_grad_scale'):
                        setattr(args, name, value)
                args.meta_places = meta_places
                parser = argparse.ArgumentParser()
                parser.parse_args = lambda: args
                with ExitStack() as stack:
                    for target, replacement in [("args_parser", lambda: parser), ("load_iclight", pipeline),
                                                ("load_salad", lambda **kw: Teacher()),
                                                ("DualTraitEvaluator", lambda: verifier),
                                                ("generate_released", generate), ("train_bilevel_step", update),
                                                ("build_fresh_salad", lambda *a, **kw: nn.Identity()),
                                                ("meta_identity", lambda *a: {'salad_initialization_sha256': 'fresh',
                                                                              'meta_trainable_names': []})]:
                        stack.enter_context(patch.object(online, target, replacement))
                    stack.enter_context(patch.object(online.DDIMScheduler, "from_config", return_value=Scheduler()))
                    stack.enter_context(patch.object(torch.cuda, "is_available", return_value=True))
                    stack.enter_context(patch.object(torch.cuda, "get_rng_state_all", return_value=[]))
                    stack.enter_context(patch.object(torch.cuda, "set_rng_state_all"))
                    stack.enter_context(redirect_stdout(StringIO()))
                    online.main()
            complete = root / "complete"
            run(complete)
            self.assertEqual(len(traces), 4)
            self.assertEqual(traces[0], traces[1])
            self.assertNotEqual(traces[1], traces[2])
            reference = torch.load(complete / "final_lora.pt", weights_only=True)
            interrupted = root / "interrupted"
            fail[0] = True
            with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
                run(interrupted)
            checkpoint = Path(json.loads((interrupted / "latest_checkpoint.json").read_text())["path"])
            saved = torch.load(checkpoint, weights_only=True)
            self.assertEqual(saved["extra"]["global_step"], 1)
            fail[0] = False
            run(interrupted, checkpoint)
            recovered = torch.load(interrupted / "final_lora.pt", weights_only=True)
            for key, value in reference["state_dict"].items():
                self.assertTrue(torch.equal(value, recovered["state_dict"][key]))
            self.assertEqual(len(list((interrupted / "rounds").glob("*.interrupted-*"))), 1)
            self.assertEqual(recovered["extra"]["global_step"], 2)
            with self.assertRaises(FileExistsError):
                run(interrupted, checkpoint)
            skipped = root / 'skipped'
            # First chunk has two places: default four-place episode is invalid.
            # Replay supplies the other two places for the second chunk.
            run(skipped, meta_places=4)
            stats = [json.loads(line) for line in (skipped / 'round_metrics.jsonl').read_text().splitlines()]
            self.assertEqual([r['updates'] for r in stats], [0, 1])
            self.assertEqual([r['skipped_meta_updates'] for r in stats], [1, 0])
            self.assertTrue(stats[0]['meta_skip_reasons'])
            # All-rejected current pools cannot update even with replay available.
            verifier.evaluate = lambda *a, **kw: EvalResult(s_geo=.1, s_div=.2, passed=False)
            rejected = root / 'rejected'
            with self.assertRaisesRegex(RuntimeError, 'no verified samples were trained'):
                run(rejected)
            stats = [json.loads(line) for line in (rejected / 'round_metrics.jsonl').read_text().splitlines()]
            self.assertEqual([r['updates'] for r in stats], [0, 0])
            self.assertEqual([r['skipped_meta_updates'] for r in stats], [1, 1])
            self.assertFalse((rejected / 'final_lora.pt').exists())
