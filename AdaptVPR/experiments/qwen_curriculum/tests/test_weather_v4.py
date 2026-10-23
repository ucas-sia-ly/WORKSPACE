"""CPU-only v4 policy, frozen-plan isolation and training-eligibility checks."""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
import random
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

ADAPTVPR_ROOT = Path(__file__).resolve().parents[3]
if str(ADAPTVPR_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTVPR_ROOT))

from experiments.qwen_curriculum import common, night_first_v3, night_first_v4, prompt_bank, prompt_bank_v4


class PromptPolicyTests(unittest.TestCase):
    def test_legacy_policy_and_engine_are_unchanged_by_v4(self):
        before = prompt_bank.definition()
        legacy_loader = night_first_v3.load_plan
        legacy_verifier = night_first_v3.QwenQualityVerifierV3
        engine = night_first_v4._engine()
        self.assertIsNot(engine, night_first_v3)
        self.assertIs(engine.prompt_bank, prompt_bank_v4)
        self.assertIs(night_first_v3.prompt_bank, prompt_bank)
        self.assertIs(night_first_v3.load_plan, legacy_loader)
        self.assertIs(night_first_v3.QwenQualityVerifierV3, legacy_verifier)
        self.assertEqual(before, prompt_bank.definition())
        self.assertIn("shopfronts", prompt_bank.VARIANTS["night"])
        self.assertNotIn("shopfronts", prompt_bank_v4.VARIANTS["night"])

    def test_weighted_choices_are_stateless_with_ten_percent_blue_hour(self):
        initial_rng = random.getstate()
        choices = [prompt_bank_v4.choose_variant("night", f"source-{index}") for index in range(20000)]
        self.assertEqual(initial_rng, random.getstate())
        self.assertEqual(choices, list(reversed([
            prompt_bank_v4.choose_variant("night", f"source-{index}") for index in reversed(range(20000))])))
        self.assertAlmostEqual(Counter(choices)["blue_hour"] / len(choices), 0.10, delta=0.01)
        self.assertEqual(sum(prompt_bank_v4.VARIANT_WEIGHTS["night"].values()), 1000)
        self.assertEqual(prompt_bank_v4.VARIANT_WEIGHTS["night"]["blue_hour"], 100)

    def test_all_night_variants_require_dry_roads_and_existing_sources(self):
        for variant in prompt_bank_v4.VARIANTS["night"]:
            text = prompt_bank_v4.prompt("night", variant)
            self.assertIn("Dry road", text)
            self.assertIn("already visible", text)
            self.assertIn("do not create one", text)
            self.assertIn("muted, natural colours", text)
            self.assertIn("do not invent or rewrite text", text)
        self.assertIn("no global orange tint", prompt_bank_v4.prompt("night", "sodium_lamps"))

    def test_old_outputs_cannot_be_relabelled_as_v4(self):
        old = {"condition": "night", "prompt_variant": "sodium_lamps",
               "prompt_policy": prompt_bank.POLICY_VERSION, "prompt": prompt_bank.prompt("night", "sodium_lamps")}
        self.assertFalse(prompt_bank_v4.matches_record(old))
        old["prompt_policy"] = prompt_bank_v4.POLICY_VERSION
        self.assertFalse(prompt_bank_v4.matches_record(old))
        old["prompt"] = prompt_bank_v4.prompt("night", "sodium_lamps")
        self.assertTrue(prompt_bank_v4.matches_record(old))


class PlanTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sources = []
        for index in range(100):
            path = self.root / "images" / f"source{index}.png"
            path.parent.mkdir(exist_ok=True)
            Image.new("RGB", (20, 16), (index, 80, 30)).save(path)
            self.sources.append({"city": "City", "place_id": index,
                "source_path": str(path), "source_sha256": common.file_sha256(path), "source_dimensions": [20, 16]})

    def test_source_order_invariance_seed_and_quota(self):
        first, reused = night_first_v4.make_plan(self.sources)
        self.assertFalse(reused)
        self.assertEqual(first, night_first_v4.make_plan(list(reversed(self.sources)))[0])
        self.assertNotEqual(first, night_first_v4.make_plan(self.sources, seed=43)[0])
        self.assertEqual(Counter(job["condition"] for job in first), {
            "night": 44, "rainy_night": 14, "rain": 15, "snow": 14, "sun": 8, "fog": 5})
        self.assertEqual(len({job["sample_id"] for job in first}), len(first))
        self.assertTrue(all(prompt_bank_v4.matches_record(job) for job in first))
        legacy, _ = night_first_v3.make_plan(self.sources, {})
        self.assertFalse({job["sample_id"] for job in legacy} & {job["sample_id"] for job in first})

    def test_duplicate_or_reused_candidates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Each place"):
            night_first_v4.make_plan(self.sources + [self.sources[0]])
        duplicate = dict(self.sources[0], place_id=100)
        with self.assertRaisesRegex(ValueError, "Duplicate source"):
            night_first_v4.make_plan(self.sources + [duplicate])
        with self.assertRaisesRegex(ValueError, "historical"):
            night_first_v4.make_plan(self.sources, {("City", 0, "night"): [{}]})

    def test_new_root_is_required(self):
        for root in (common.WORKSPACE_ROOT / "outputs/weather_v3", common.WORKSPACE_ROOT / "outputs/qwen_curriculum"):
            with self.assertRaisesRegex(ValueError, "historical"):
                night_first_v4._safe_roots(root / "v4", self.root)
            with self.assertRaisesRegex(ValueError, "historical"):
                night_first_v4._safe_roots(self.root, root)

    def test_prepare_resume_freezes_v4_without_network_or_model_loading(self):
        engine = night_first_v4._engine()
        legacy_root = self.root / "legacy"
        common.write_json(legacy_root / "state/all_city_campaign_config.json", {
            "reference_execution": {"service": {"model_id": "test-only-no-service"}}})
        args = SimpleNamespace(output_dir=self.root / "v4/state", image_root=self.root / "v4",
            real_data=self.root / "gsv", seed=42, min_images=4, workers=1)
        with ExitStack() as stack:
            stack.enter_context(patch.object(engine, "LEGACY_ROOT", legacy_root))
            stack.enter_context(patch.object(night_first_v4.source_screen_v3, "screen_sources", return_value=self.sources))
            stack.enter_context(patch.object(night_first_v4.source_reliability_v3, "refine_sources", return_value=(self.sources, {})))
            stack.enter_context(patch.object(engine.requests, "Session", side_effect=AssertionError("no network")))
            stack.enter_context(patch.object(engine, "QwenQualityVerifierV3", side_effect=AssertionError("no models")))
            config = night_first_v4.prepare(args)
            self.assertEqual(config, night_first_v4.prepare(args))
            checked, jobs = night_first_v4.load_plan(args.output_dir)
        self.assertEqual(config, checked)
        self.assertEqual(len(jobs), 100)
        self.assertEqual((args.output_dir / "training_manifest.jsonl").read_text(), "")
        self.assertFalse((args.image_root / "accepted").exists())
        self.assertEqual(set(config["files_sha256"]).intersection({str(Path(prompt_bank.__file__).resolve())}),
                         {str(Path(prompt_bank.__file__).resolve())})
        # The v3 loader must not accidentally accept this new plan.
        with self.assertRaisesRegex(ValueError, "Prompt bank changed"):
            night_first_v3.load_plan(args.output_dir)

    def test_geometry_pass_never_creates_trainable_candidates(self):
        original = {"passed": True, "eligible_for_training": True, "geometry_ok": True,
                    "weather_ok": True, "estimated_H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "status": "passed"}
        verifier = night_first_v4.CandidateVerifier.__new__(night_first_v4.CandidateVerifier)
        with patch.object(night_first_v4.quality_v3.QwenQualityVerifierV3, "evaluate", return_value=dict(original)):
            verdict = verifier.evaluate(None, None, "night")
        self.assertTrue(verdict["base_quality_passed"])
        self.assertFalse(verdict["eligible_for_training"])
        self.assertEqual(verdict["estimated_H"], original["estimated_H"])
        output = self.root / "candidate"
        output.mkdir()
        night_first_v4._engine().rebuild_manifest(output, {"image_root": str(output)}, [verdict])
        self.assertEqual((output / "training_manifest.jsonl").read_text(), "")
        self.assertFalse((output / "accepted").exists())

    def test_mocked_generation_resumes_without_calls_or_legacy_mutation(self):
        # The existing CPU fake exercises the exact service sidecar/reservation
        # protocol used by the original durable engine, without starting it.
        tests_root = str(Path(__file__).resolve().parent)
        if tests_root not in sys.path:
            sys.path.insert(0, tests_root)
        from test_night_first_v3 import FakeVerifier, HEALTH, Session

        class CandidateFake(FakeVerifier):
            definition = night_first_v4.quality_definition_v4(clip_device="cpu")

            def evaluate(self, *args):
                result = super().evaluate(*args)
                result.update(base_quality_passed=result["passed"], eligible_for_training=False,
                    content_status="pending_automated_audit", quality_version="4-candidate")
                return result

        engine = night_first_v4._engine()
        legacy_verifier = night_first_v3.QwenQualityVerifierV3
        directory, service = self.root / "generated/state", self.root / "service"
        service.mkdir()
        legacy_root = self.root / "legacy"
        common.write_json(legacy_root / "state/all_city_campaign_config.json", {
            "reference_execution": {"service": night_first_v4.run.service_identity(HEALTH)}})
        args = SimpleNamespace(output_dir=directory, image_root=self.root / "generated",
            real_data=self.root / "gsv", seed=42, min_images=4, workers=1, domain="night",
            device="cpu", qwen_url="http://127.0.0.1:8001/generate", request_timeout=20.0, max_calls=1)
        session, verifier = Session(directory, service), CandidateFake()
        with ExitStack() as stack:
            stack.enter_context(patch.object(engine, "LEGACY_ROOT", legacy_root))
            stack.enter_context(patch.object(night_first_v4.source_screen_v3, "screen_sources", return_value=self.sources[:10]))
            stack.enter_context(patch.object(night_first_v4.source_reliability_v3, "refine_sources", return_value=(self.sources[:10], {})))
            config = night_first_v4.prepare(args)
            stack.enter_context(patch.object(engine.requests, "Session", return_value=session))
            stack.enter_context(patch.object(engine.run, "service_output_directory", return_value=service))
            stack.enter_context(patch.object(engine, "QwenQualityVerifierV3", return_value=verifier))
            self.assertEqual(engine.generate(args)["state"], "call_limit")
            self.assertEqual(len(session.posts), 1)
            args.max_calls = 0
            self.assertEqual(engine.generate(args)["state"], "domain_complete")
            num_night = config["domain_counts"]["night"]
            self.assertEqual(len(session.posts), num_night)
            engine.generate(args)
            self.assertEqual(len(session.posts), num_night)
        self.assertIs(night_first_v3.QwenQualityVerifierV3, legacy_verifier)
        results = common.read_jsonl(directory / "results.jsonl")
        self.assertEqual(len(results), config["domain_counts"]["night"])
        self.assertTrue(all(row["base_quality_passed"] and not row["eligible_for_training"] for row in results))
        self.assertTrue(all(row["prompt_policy"] == prompt_bank_v4.POLICY_VERSION for row in results))
        self.assertEqual((directory / "training_manifest.jsonl").read_text(), "")
        self.assertFalse((args.image_root / "accepted").exists())


if __name__ == "__main__":
    unittest.main()
