"""CPU integration checks for the bounded, paired v5 night pilot.

Images, service replies and verifier outputs are fixtures. No model, GPU,
network service or generation process is started by these tests.
"""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import requests

ADAPTVPR_ROOT = Path(__file__).resolve().parents[3]
if str(ADAPTVPR_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTVPR_ROOT))
TESTS_ROOT = str(Path(__file__).resolve().parent)
if TESTS_ROOT not in sys.path:
    sys.path.insert(0, TESTS_ROOT)

from experiments.qwen_curriculum import (campaign, common, night_first_v3,
    night_first_v5 as pilot, prompt_bank_v4, run)
from test_night_first_v3 import FakeVerifier, HEALTH, Session


class CandidateFake(FakeVerifier):
    definition = pilot.quality_definition_v5(clip_device="cpu")

    def evaluate(self, *args):
        verdict = super().evaluate(*args)
        verdict.update(geometry_weather_passed=True, base_quality_passed=verdict["passed"],
            eligible_for_training=False, content_status="pending_automated_audit",
            quality_version="5-candidate", night_appearance={
                "appearance_status": "candidate" if self.accept else "unknown"})
        if not self.accept:
            verdict["status"] = "unknown"
        return verdict


class PilotFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images = self.root / "pilot"
        self.directory = self.images / "state"
        self.real = self.root / "gsv"
        self.service = self.root / "service"
        self.service.mkdir()
        self.source_manifest = self.root / "v4-results.jsonl"
        self.exclude = self.root / "excluded_places.json"
        self.service_config = self.root / "service-config.json"
        self.sources = []
        variants = ("led_lamps", "sodium_lamps", "sparse_lamps", "existing_headlights", "blue_hour")
        for index, variant in enumerate(variants):
            source = self.real / "Images/City" / f"City_{index:07d}_fixture.png"
            source.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (20, 16), (90 + index, 100, 110)).save(source)
            old_output = self.root / "old-v4" / f"output{index}.png"
            old_output.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (20, 16), (30 + index, 40, 50)).save(old_output)
            self.sources.append(run.sealed({
                "city": "City", "place_id": index, "sample_id": f"qwen4_{index:024x}",
                "condition": "night", "source_path": str(source),
                "source_sha256": common.file_sha256(source), "source_dimensions": [20, 16],
                "output_path": str(old_output), "output_sha256": common.file_sha256(old_output),
                "prompt_variant": variant, "prompt_policy": prompt_bank_v4.POLICY_VERSION,
                "prompt": prompt_bank_v4.prompt("night", variant), "negative_prompt": "",
                "seed": 200 + index, "eligible_for_training": False,
            }))
        common.write_jsonl(self.source_manifest, self.sources)
        common.write_json(self.exclude, [["City", 3]])
        campaign.frozen(self.service_config, {
            "reference_execution": {"service": run.service_identity(HEALTH)}})
        (self.real / "Dataframes").mkdir()
        (self.real / "Dataframes/City.csv").write_text("place_id\n0\n1\n2\n3\n4\n")
        self.args = SimpleNamespace(output_dir=self.directory, image_root=self.images,
            source_manifest=self.source_manifest, exclude_places=self.exclude,
            service_config=self.service_config, real_data=self.real, max_places=3,
            sample_ids=[], seed=42, request_budget=None, domain="night", device="cpu",
            qwen_url="http://127.0.0.1:8001/generate", request_timeout=20.0, max_calls=1)
        self.session, self.verifier = Session(self.directory, self.service), CandidateFake()

    def prepare(self):
        # Preparing must not touch the network or instantiate a model.
        engine = pilot._engine()
        with patch.object(engine.requests, "Session", side_effect=AssertionError("no network")), \
                patch.object(engine, "QwenQualityVerifierV3", side_effect=AssertionError("no model")):
            return pilot.prepare(self.args)

    def generate(self):
        engine = pilot._engine()
        with ExitStack() as stack:
            stack.enter_context(patch.object(engine.requests, "Session", return_value=self.session))
            stack.enter_context(patch.object(engine.run, "service_output_directory", return_value=self.service))
            stack.enter_context(patch.object(engine, "QwenQualityVerifierV3", return_value=self.verifier))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return engine.generate(self.args)

    def rows(self, name="results"):
        path = self.directory / f"{name}.jsonl"
        return common.read_jsonl(path) if path.exists() else []

    def test_plan_preserves_paired_source_variant_seed_and_has_finite_budget(self):
        excluded = {("City", 3)}
        jobs = pilot.make_plan(self.sources, 2, [], excluded, seed=42)
        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs, pilot.make_plan(list(reversed(self.sources)), 2, [], excluded, seed=42))
        original = {row["sample_id"]: row for row in self.sources}
        for job in jobs:
            old = original[job["comparison"]["sample_id"]]
            for key in ("source_path", "source_sha256", "source_dimensions", "prompt_variant", "seed"):
                self.assertEqual(job[key], old[key])
            self.assertNotEqual(job["prompt"], old["prompt"])
            self.assertNotEqual(job["sample_id"], old["sample_id"])
            self.assertEqual(job["negative_prompt"], "")
            self.assertEqual(job["condition"], "night")
            self.assertNotEqual(job["place_id"], 3)
            self.assertNotEqual(job["prompt_variant"], "blue_hour")
        for budget in (0, -1, True, 1.5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                pilot.make_plan(self.sources, budget, [], excluded)

    def test_explicit_comparisons_reject_exclusions_blue_hour_and_oversized_requests(self):
        excluded = {("City", 3)}
        for ids in ([self.sources[3]["sample_id"]], [self.sources[4]["sample_id"]],
                    ["missing"], [self.sources[0]["sample_id"]] * 2,
                    [row["sample_id"] for row in self.sources[:3]]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                pilot.make_plan(self.sources, 2, ids, excluded)
        ordered = [self.sources[2]["sample_id"], self.sources[0]["sample_id"]]
        jobs = pilot.make_plan(self.sources, 2, ordered, excluded)
        self.assertEqual([job["comparison"]["sample_id"] for job in jobs], ordered)

    def test_historical_output_roots_and_ancestors_are_protected(self):
        for name in ("weather_v3", "weather_v4", "qwen_curriculum"):
            old = common.WORKSPACE_ROOT / "outputs" / name
            for unsafe in (old, old / "new-v5", old.parent):
                with self.subTest(unsafe=unsafe):
                    with self.assertRaisesRegex(ValueError, "independent"):
                        pilot.safe_roots(unsafe, self.images)
                    with self.assertRaisesRegex(ValueError, "independent"):
                        pilot.safe_roots(self.directory, unsafe)

    def test_prepare_resume_freezes_dependencies_and_does_not_modify_comparisons(self):
        previous = self.source_manifest.read_bytes()
        config = self.prepare()
        self.assertEqual(config, self.prepare())
        checked, jobs = pilot.load_plan(self.directory)
        self.assertEqual(config, checked)
        self.assertEqual(len(jobs), 3)
        self.assertEqual(previous, self.source_manifest.read_bytes())
        required = {str(Path(module.__file__).resolve()) for module in
                    (pilot, pilot._engine(), pilot.prompt_bank, pilot.appearance,
                     pilot.quality_v3, campaign, common, run, pilot.weather_io)}
        required |= {str(self.exclude), str(self.service_config),
                     str(self.directory / "comparison_snapshot.jsonl"),
                     str(self.real / "Dataframes/City.csv")}
        self.assertTrue(required <= config["files_sha256"].keys())
        self.assertEqual((self.directory / "training_manifest.jsonl").read_text(), "")
        self.assertFalse((self.images / "accepted").exists())
        self.args.max_places = 2
        with self.assertRaisesRegex(ValueError, "selection differs"):
            self.prepare()

    def test_prepare_rejects_unsealed_or_changed_comparison_images(self):
        self.sources[0]["seed"] += 1
        common.write_jsonl(self.source_manifest, self.sources)
        with self.assertRaisesRegex(ValueError, "seal"):
            self.prepare()
        self.sources[0] = run.sealed(self.sources[0])
        common.write_jsonl(self.source_manifest, self.sources)
        Image.new("RGB", (20, 16), "red").save(self.sources[0]["output_path"])
        with self.assertRaises(ValueError):
            self.prepare()

    def test_resume_rejects_split_and_comparison_snapshot_changes(self):
        self.prepare()
        for target in (self.exclude, self.directory / "comparison_snapshot.jsonl"):
            before = target.read_bytes()
            target.write_bytes(before + b"\n")
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "Frozen input"):
                pilot.load_plan(self.directory)
            target.write_bytes(before)
        pilot.load_plan(self.directory)

    def test_mock_generation_budget_resume_and_unknown_remain_nontraining(self):
        legacy_verifier = night_first_v3.QwenQualityVerifierV3
        self.prepare()
        first = self.generate()
        self.assertEqual(first["state"], "call_limit")
        self.assertEqual(len(self.session.posts), 1)
        self.verifier.accept = False
        self.args.max_calls = 2
        finished = self.generate()
        self.assertEqual(finished["state"], "domain_complete")
        self.assertEqual(len(self.session.posts), 3)
        self.generate()
        self.assertEqual(len(self.session.posts), 3)
        self.assertEqual(len(self.verifier.calls), 3)
        self.assertEqual(len(self.rows("attempts")), 3)
        self.assertEqual(finished["candidate_passed"], 1)
        self.assertEqual(finished["appearance_unknown"], 2)
        self.assertEqual(finished["accepted"], 0)
        self.assertTrue(all(row["eligible_for_training"] is False for row in self.rows()))
        self.assertEqual((self.directory / "training_manifest.jsonl").read_text(), "")
        self.assertFalse((self.images / "accepted").exists())
        self.assertIs(night_first_v3.QwenQualityVerifierV3, legacy_verifier)

    def test_interrupted_reply_recovers_same_raw_without_duplicate_request(self):
        self.args.max_places = 1
        self.prepare()
        self.session.error_after_save = requests.Timeout("saved raw but reply interrupted")
        with self.assertRaises(requests.Timeout):
            self.generate()
        self.assertEqual(len(self.session.posts), 1)
        self.assertEqual(len(self.rows("attempts")), 1)
        self.assertFalse(self.rows("generated"))
        self.session.error_after_save = None
        summary = self.generate()
        self.assertEqual(summary["state"], "domain_complete")
        self.assertEqual(len(self.session.posts), 1)
        self.assertEqual(len(self.rows()), 1)
        self.assertFalse(list(self.service.glob("*.png")))

    def test_real_candidate_verifier_does_not_promote_unknown_or_candidate_to_training(self):
        for appearance_status in ("candidate", "unknown"):
            verifier = pilot.CandidateVerifier.__new__(pilot.CandidateVerifier)
            verifier.appearance = SimpleNamespace(measure=lambda *_: {"appearance_status": appearance_status})
            original = {"passed": True, "eligible_for_training": True,
                        "status": "passed", "rejection_reasons": []}
            with patch.object(pilot.quality_v3.QwenQualityVerifierV3, "evaluate", return_value=dict(original)), \
                    patch.object(verifier, "_ensure_weather"):
                verdict = verifier.evaluate(None, None, "night")
            self.assertFalse(verdict["eligible_for_training"])
            self.assertEqual(verdict["passed"], appearance_status == "candidate")
            self.assertTrue(verdict["geometry_weather_passed"])
            self.assertEqual(verdict["content_status"], "pending_automated_audit")

    def test_failed_call_consumes_cumulative_budget_on_resume(self):
        self.args.max_places = 1
        self.prepare()
        self.session.error_after_save = requests.Timeout("incomplete raw")
        self.session.incomplete_raw = True
        with self.assertRaises(requests.Timeout):
            self.generate()
        self.session.error_after_save = None
        self.session.incomplete_raw = False
        with self.assertRaisesRegex(RuntimeError, "cumulative request budget"):
            self.generate()
        self.assertEqual(len(self.rows("attempts")), 1)
        self.assertEqual(len(self.session.posts), 1)

    def test_comparison_dry_run_validates_pairs_without_loading_models(self):
        from experiments.qwen_curriculum.compare_night_v5 import compare
        self.args.max_places = 1
        self.prepare()
        self.generate()
        output = self.root / 'comparison'
        with patch.object(pilot.quality_v3, 'QwenQualityVerifierV3', side_effect=AssertionError('no model')):
            result = compare(self.directory, output)
        self.assertEqual(result, dict(planned=1, completed=1, model_loaded=False))
        self.assertFalse(output.exists())
        row = self.rows()[0]
        Image.new('RGB', (20, 16), 'red').save(row['comparison']['output_path'])
        with self.assertRaisesRegex(ValueError, 'missing or changed'):
            compare(self.directory, output)


if __name__ == "__main__":
    unittest.main()
