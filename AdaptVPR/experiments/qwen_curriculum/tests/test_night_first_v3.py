"""CPU integration checks for place coverage, weather quotas and durable resume."""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack, redirect_stdout
import io
import json
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

from experiments.qwen_curriculum import common, prompt_bank, run  # noqa: E402


HEALTH = {"status": "ok", "model_loaded": True, "generator_ready": True,
          "source_modified": False, "error": None, "model_id": "Qwen/Qwen-Image-Edit-2511",
          "canvas_policy": "source_aspect_v1", "sampling": {"infer_steps": 4, "guidance_scale": 1.0},
          "canvas": {"target_shape_order": "height,width", "target_pixels": 720,
                     "multiple": 1, "min_side": 1, "max_side": 128,
                     "rounding": "nearest_multiple", "unsupported_aspect_ratio": "reject"}}


class Response:
    def __init__(self, value):
        self.value = value

    def raise_for_status(self):
        pass

    def json(self):
        return self.value


class FakeVerifier:
    definition = {"version": "cpu-test", "prompt_bank": prompt_bank.POLICY_VERSION}

    def __init__(self):
        self.calls = []
        self.error = None
        self.accept = True

    def evaluate(self, source, output, condition):
        self.calls.append((source.size, output.size, condition))
        if self.error is not None:
            raise self.error
        return {"status": "passed" if self.accept else "rejected", "passed": self.accept,
                "eligible_for_training": self.accept, "s_geo": 0.9, "weather_shift": 12.0,
                "rejection_reasons": [] if self.accept else ["weather"]}


class Session:
    def __init__(self, directory, service):
        self.directory, self.service = directory, service
        self.posts, self.gets = [], []
        self.error_after_save = None
        self.incomplete_raw = False
        self.trust_env = True

    def get(self, url, timeout):
        self.gets.append((url, timeout))
        return Response(dict(HEALTH))

    def post(self, url, json, timeout):
        # Assert that reservation is durable at the exact outbound-call boundary.
        ledger = common.read_jsonl(self.directory / "attempts.jsonl")
        latest = ledger[-1]
        assert latest["sample_id"] == Path(json["image_path"]).stem
        body = {k: v for k, v in latest.items() if k != "attempt_sha256"}
        assert latest["attempt_sha256"] == common.fingerprint(body)
        assert latest["prompt"] == json["prompt"] and latest["seed"] == json["seed"]
        assert len(ledger) == len(self.posts) + 1
        assert Path(json["image_path"]).is_file()
        self.posts.append((url, json, timeout))
        raw = self.service / f"response{len(self.posts)}.png"
        Image.new("RGB", (30, 24), (40, 70, 90 + len(self.posts))).save(raw)
        sidecar = raw.with_suffix(".json")
        metadata = {"source_path": json["image_path"], "seed": json["seed"],
            "result_path": str(raw), "metadata_path": str(sidecar),
            "canvas_policy": "source_aspect_v1", "source_dimensions": [20, 16],
            "raw_dimensions": [30, 24], "target_shape": [24, 30]}
        common.write_json(sidecar, metadata)
        if self.incomplete_raw:
            raw.write_bytes(b"interrupted PNG bytes")
        if self.error_after_save is not None:
            raise self.error_after_save
        return Response(metadata)

    def close(self):
        pass


class NightCampaignFixture(unittest.TestCase):
    def setUp(self):
        from experiments.qwen_curriculum import night_first_v3
        self.module = night_first_v3
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.legacy_root = self.root / "outputs/qwen_curriculum"
        self.images = self.root / "outputs/weather_v3"
        self.directory = self.images / "state"
        self.service = self.root / "service"
        self.service.mkdir()
        self.gsv = self.root / "gsv"
        self.sources = []
        for index in range(100):
            city = "LargeCity" if index < 90 else "SmallCity"
            source = self.gsv / "Images" / city / f"{city}_{index:07d}.png"
            source.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (20, 16), (index, 80, 30)).save(source)
            self.sources.append({"city": city, "place_id": index, "source_path": str(source),
                "source_sha256": common.file_sha256(source), "source_dimensions": [20, 16],
                "passed": True, "reasons": [], "metrics": {}})
        self.verifier = FakeVerifier()
        self.session = Session(self.directory, self.service)
        self.args = SimpleNamespace(output_dir=self.directory, image_root=self.images,
            real_data=self.gsv, seed=42, min_images=4, workers=1, reuse_results=[],
            domain="night", device="cpu", qwen_url="http://127.0.0.1:8001/generate",
            request_timeout=20.0, max_calls=0)
        common.write_json(self.legacy_root / "state/all_city_campaign_config.json", {
            "reference_execution": {"service": run.service_identity(HEALTH)}})

    def prepare(self, sources=None, candidates=None):
        with ExitStack() as stack:
            stack.enter_context(patch.object(self.module, "LEGACY_ROOT", self.legacy_root))
            stack.enter_context(patch.object(self.module.source_screen_v3, "screen_sources",
                                            return_value=self.sources if sources is None else sources))
            stack.enter_context(patch.object(self.module.source_reliability_v3, "refine_sources",
                side_effect=lambda selected, *args, **kwargs: (selected, {"unreliable_source_paths": []})))
            stack.enter_context(patch.object(self.module.source_screen_v3, "screen_image", side_effect=lambda p: {
                "passed": True, "source_sha256": common.file_sha256(p)}))
            stack.enter_context(patch.object(self.module.v3_reuse, "collect_candidates",
                                            return_value=(candidates or {}, {"compatible": 0})))
            stack.enter_context(patch.object(self.module.quality_v3, "quality_definition_v3",
                                            return_value=FakeVerifier.definition))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return self.module.prepare(self.args)

    def generate(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(self.module.requests, "Session", return_value=self.session))
            stack.enter_context(patch.object(run, "service_output_directory", return_value=self.service))
            stack.enter_context(patch.object(self.module, "QwenQualityVerifierV3", return_value=self.verifier))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return self.module.generate(self.args)

    def rows(self, name="results"):
        path = self.directory / f"{name}.jsonl"
        return common.read_jsonl(path) if path.exists() else []

    def test_defaults_keep_v3_images_and_state_in_a_separate_output_root(self):
        root = common.WORKSPACE_ROOT / "outputs/weather_v3"
        self.assertEqual(self.module.ROOT, root)
        self.assertEqual(self.module.DEFAULT_IMAGES, root)
        self.assertEqual(self.module.DEFAULT_OUTPUT, root / "state")
        self.assertEqual(self.module.LEGACY_ROOT, root.with_name("qwen_curriculum"))
        config = {"image_root": str(root)}
        for domain in self.module.DOMAINS:
            output, raw = self.module.paths(config, {"condition": domain, "sample_id": "sample"})
            self.assertEqual(output, root / domain / "sample.png")
            self.assertEqual(raw, root / "raw" / domain / "sample.png")
        self.assertNotIn("overcast", self.module.DOMAINS)

    def test_prepared_plan_freezes_shared_io_and_only_reads_legacy_reference(self):
        config = self.prepare(sources=self.sources[:2])
        from experiments.qwen_curriculum import weather_io
        path = Path(weather_io.__file__).resolve()
        self.assertEqual(config["files_sha256"][str(path)], common.file_sha256(path))
        self.assertEqual(config["image_root"], str(self.images))
        self.assertTrue((self.directory / "plan_config.json").is_file())
        self.assertEqual(sorted(p.name for p in (self.legacy_root / "state").iterdir()),
                         ["all_city_campaign_config.json"])

    def test_plan_covers_every_place_once_with_exact_quotas_and_global_shuffle(self):
        jobs, reused = self.module.make_plan(self.sources, {}, seed=42)
        self.assertFalse(reused)
        self.assertEqual(len(jobs), 100)
        self.assertEqual(Counter(j["condition"] for j in jobs), {
            "night": 44, "rainy_night": 14, "rain": 15, "snow": 14, "sun": 8, "fog": 5})
        self.assertEqual({(j["city"], j["place_id"]) for j in jobs},
                         {(j["city"], j["place_id"]) for j in self.sources})
        self.assertEqual(Counter(j["city"] for j in jobs), {"LargeCity": 90, "SmallCity": 10})
        # An equal-city round robin would start with two different cities and
        # exhaust SmallCity in the first 20 jobs; global place shuffling does not.
        self.assertEqual(jobs[0]["city"], jobs[1]["city"])
        self.assertTrue(any(j["city"] == "SmallCity" for j in jobs[20:]))
        self.assertNotEqual([j["place_id"] for j in jobs], list(range(100)))
        self.assertEqual(jobs, self.module.make_plan(list(reversed(self.sources)), {}, seed=42)[0])
        self.assertNotEqual(jobs, self.module.make_plan(self.sources, {}, seed=43)[0])

    def test_duplicate_places_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Each place"):
            self.module.make_plan(self.sources + [self.sources[0]], {})

    def test_generate_posts_only_night_and_uses_separate_domain_paths(self):
        config = self.prepare()
        original = {r["source_path"]: Path(r["source_path"]).read_bytes() for r in self.sources}
        summary = self.generate()
        self.assertEqual(len(self.session.posts), 44)
        self.assertFalse(self.session.trust_env)
        self.assertEqual(summary["state"], "domain_complete")
        self.assertEqual(summary["by_domain"]["night"]["remaining_generation"], 0)
        self.assertEqual(len(self.rows()), 44)
        self.assertEqual(len(self.rows("training_manifest")), 44)
        self.assertEqual({r["condition"] for r in self.rows()}, {"night"})
        for row in self.rows():
            output, raw = self.module.paths(config, row)
            self.assertEqual(output.parent, self.images / "night")
            self.assertEqual(raw.parent, self.images / "raw/night")
            self.assertTrue(output.is_file() and raw.is_file())
            accepted = self.images / "accepted/night" / output.name
            self.assertTrue(accepted.is_symlink())
            self.assertEqual(accepted.resolve(), output.resolve())
            with Image.open(output) as normalized:
                self.assertEqual(normalized.size, (20, 16))
        self.assertFalse((self.images / "rain").exists())
        self.assertEqual(original, {p: Path(p).read_bytes() for p in original})
        self.generate()
        self.assertEqual(len(self.session.posts), 44)
        self.assertEqual(len(self.verifier.calls), 44)

    def test_saved_raw_after_interrupted_post_resumes_without_duplicate_call(self):
        self.prepare(sources=self.sources[:2])
        self.session.error_after_save = requests.Timeout("HTTP reply was interrupted after save")
        with self.assertRaises(requests.Timeout):
            self.generate()
        self.assertEqual(len(self.session.posts), 1)
        self.assertEqual(len(self.rows("attempts")), 1)
        self.assertFalse(self.rows("generated"))
        pending = list((self.directory / "requests").glob("*/source_inputs/*.png"))
        self.assertEqual(len(pending), 1)
        self.session.error_after_save = None
        self.generate()
        self.assertEqual(len(self.session.posts), 1)
        self.assertEqual(len(self.rows("generated")), 1)
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(len(self.verifier.calls), 1)
        self.assertFalse(pending[0].exists())
        self.assertFalse(list(self.service.glob("*.png")))

    def test_relocated_pending_request_recovers_saved_service_image_without_post(self):
        self.prepare(sources=self.sources[:2])
        self.session.error_after_save = requests.Timeout("interrupted before reply")
        with self.assertRaises(requests.Timeout):
            self.generate()
        original_plan = (self.directory / "plan.jsonl").read_bytes()
        raw_bytes = (self.service / "response1.png").read_bytes()
        old_images, old_directory = self.images, self.directory
        self.images = old_images.with_name("relocated_weather_v3")
        old_images.rename(self.images)
        self.directory = self.images / "state"
        self.args.output_dir, self.args.image_root = self.directory, self.images
        self.session.directory = self.directory

        def relocated(path):
            return self.images / Path(path).relative_to(old_images)

        attempts = self.rows("attempts")
        attempts[0]["workspace"] = str(relocated(attempts[0]["workspace"]))
        body = {k: v for k, v in attempts[0].items() if k != "attempt_sha256"}
        attempts[0]["attempt_sha256"] = common.fingerprint(body)
        common.write_jsonl(self.directory / "attempts.jsonl", attempts)
        sidecar = self.service / "response1.json"
        metadata = json.loads(sidecar.read_text())
        metadata["source_path"] = str(relocated(metadata["source_path"]))
        common.write_json(sidecar, metadata)
        config_path = self.directory / "plan_config.json"
        config = json.loads(config_path.read_text())
        config["image_root"] = str(self.images)
        config["files_sha256"] = {
            str(relocated(path) if Path(path).is_relative_to(old_images) else path): digest
            for path, digest in config["files_sha256"].items()}
        body = {k: v for k, v in config.items() if k != "fingerprint"}
        common.write_json(config_path, {**body, "fingerprint": common.fingerprint(body)})

        self.session.error_after_save = None
        self.generate()
        self.assertEqual(len(self.session.posts), 1)
        self.assertEqual(len(self.rows("generated")), 1)
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual((self.directory / "plan.jsonl").read_bytes(), original_plan)
        saved = self.rows("generated")[0]
        self.assertEqual(Path(saved["raw_output_path"]).read_bytes(), raw_bytes)
        self.assertEqual(saved["attempt_sha256"], attempts[0]["attempt_sha256"])
        self.assertFalse(old_directory.exists())
        self.assertFalse(Path(attempts[0]["workspace"]).exists())
        self.assertFalse(list(self.service.glob("*.png")))

    def test_saved_image_is_reverified_without_new_generation_after_verifier_failure(self):
        self.prepare(sources=self.sources[:2])
        self.verifier.error = RuntimeError("CPU verifier interrupted")
        with self.assertRaisesRegex(RuntimeError, "verifier interrupted"):
            self.generate()
        saved = self.rows("generated")
        self.assertEqual(len(saved), 1)
        self.assertFalse(self.rows())
        before = {saved[0][key]: Path(saved[0][key]).read_bytes() for key in ("output_path", "raw_output_path")}
        self.verifier.error = None
        self.generate()
        self.assertEqual(len(self.session.posts), 1)
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(before, {p: Path(p).read_bytes() for p in before})

    def test_rejected_output_is_retained_but_not_accepted_or_regenerated(self):
        self.prepare(sources=self.sources[:2])
        self.verifier.accept = False
        self.generate()
        self.generate()
        self.assertEqual(len(self.session.posts), 1)
        self.assertEqual(len(self.rows()), 1)
        self.assertFalse(self.rows()[0]["passed"])
        self.assertTrue(Path(self.rows()[0]["output_path"]).is_file())
        self.assertFalse(self.rows("training_manifest"))
        self.assertFalse(list((self.images / "accepted").rglob("*.png")))

    def test_torn_journal_tail_is_repaired_without_regeneration(self):
        self.prepare(sources=self.sources[:2])
        self.generate()
        for name in ("generated", "results", "attempts"):
            with (self.directory / f"{name}.jsonl").open("a") as handle:
                handle.write('{"interrupted":')
        self.generate()
        for name in ("generated", "results", "attempts"):
            self.assertEqual(len(self.rows(name)), 1)
        self.assertEqual(len(self.session.posts), 1)

    def test_changed_source_and_changed_plan_fail_before_post(self):
        self.prepare(sources=self.sources[:2])
        _, jobs = self.module.load_plan(self.directory)
        job = next(j for j in jobs if j["condition"] == "night")
        Path(job["source_path"]).write_bytes(b"changed source")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            self.generate()
        self.assertFalse(self.session.posts)
        with (self.directory / "plan.jsonl").open("a") as handle:
            handle.write("{}\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.generate()
        self.assertFalse(self.session.posts)

    def test_reuse_copies_exact_bytes_without_modifying_or_aliasing_old_images(self):
        jobs, _ = self.module.make_plan(self.sources[:2], {})
        job = next(j for j in jobs if j["condition"] == "night")
        old = self.root / "old_pilot"
        old.mkdir()
        output, raw = old / "train.png", old / "raw.png"
        Image.new("RGB", (20, 16), (90, 50, 20)).save(output)
        Image.new("RGB", (30, 24), (90, 50, 20)).save(raw)
        candidate = {**job, "output_path": str(output), "raw_output_path": str(raw),
            "output_sha256": common.file_sha256(output), "raw_output_sha256": common.file_sha256(raw),
            "status": "passed", "passed": True, "eligible_for_training": True, "s_geo": 0.92,
            "reuse_provenance": {"ledger_path": str(old / "results.jsonl")}}
        before = {p: p.read_bytes() for p in (output, raw)}
        key = (job["city"], job["place_id"], job["condition"])
        config = self.prepare(sources=self.sources[:2], candidates={key: [candidate]})
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(len(self.rows("generated")), 1)
        self.assertFalse(self.rows("attempts"))
        copied, copied_raw = self.module.paths(config, job)
        self.assertEqual(copied.read_bytes(), before[output])
        self.assertEqual(copied_raw.read_bytes(), before[raw])
        self.assertFalse(copied.is_symlink() or copied_raw.is_symlink())
        self.assertNotEqual(copied.stat().st_ino, output.stat().st_ino)
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.prepare(sources=self.sources[:2], candidates={key: [candidate]})
        self.generate()
        self.assertEqual(len(self.rows()), 1)
        self.assertFalse(self.session.posts)

    def test_incomplete_raw_does_not_block_one_fresh_request_on_resume(self):
        self.prepare(sources=self.sources[:2])
        self.session.incomplete_raw = True
        self.session.error_after_save = requests.Timeout("interrupted while raw was being written")
        with self.assertRaises(requests.Timeout):
            self.generate()
        self.assertEqual(len(self.rows("attempts")), 1)
        self.session.incomplete_raw = False
        self.session.error_after_save = None
        self.generate()
        self.assertEqual(len(self.session.posts), 2)
        self.assertEqual(len(self.rows("attempts")), 2)
        self.assertEqual(len(self.rows()), 1)
        self.assertNotEqual(self.rows("attempts")[0]["workspace"], self.rows("attempts")[1]["workspace"])
        self.assertEqual(len(self.verifier.calls), 1)

    def test_resealed_attempt_with_changed_prompt_is_rejected(self):
        self.prepare(sources=self.sources[:2])
        self.generate()
        attempts = self.rows("attempts")
        attempts[0]["prompt"] = "another prompt that was never planned"
        body = {k: v for k, v in attempts[0].items() if k != "attempt_sha256"}
        attempts[0]["attempt_sha256"] = common.fingerprint(body)
        common.write_jsonl(self.directory / "attempts.jsonl", attempts)
        with self.assertRaises(ValueError):
            self.generate()
        self.assertEqual(len(self.session.posts), 1)

    def test_worker_lock_prevents_concurrent_post(self):
        self.prepare(sources=self.sources[:2])
        with run.run_lock(self.directory):
            with self.assertRaisesRegex(RuntimeError, "Another worker"):
                self.generate()
        self.assertFalse(self.session.posts)

    def test_two_matching_service_outputs_fail_closed_without_new_post(self):
        self.prepare(sources=self.sources[:2])
        self.session.error_after_save = requests.Timeout("reply interrupted")
        with self.assertRaises(requests.Timeout):
            self.generate()
        first = self.service / "response1.png"
        second = self.service / "duplicate.png"
        second.write_bytes(first.read_bytes())
        metadata = json.loads(first.with_suffix(".json").read_text())
        metadata.update(result_path=str(second), metadata_path=str(second.with_suffix(".json")))
        common.write_json(second.with_suffix(".json"), metadata)
        self.session.error_after_save = None
        with self.assertRaisesRegex(ValueError, "Multiple raw outputs"):
            self.generate()
        self.assertEqual(len(self.session.posts), 1)
        self.assertFalse(self.rows("generated"))

    def test_interruption_before_generated_commit_recovers_copies_without_new_post(self):
        self.prepare(sources=self.sources[:2])
        original_append = self.module._append
        def append(path, row):
            if path.name == "generated.jsonl":
                raise OSError("power failure before generated commit")
            return original_append(path, row)
        with patch.object(self.module, "_append", side_effect=append):
            with self.assertRaisesRegex(OSError, "generated commit"):
                self.generate()
        self.assertEqual(len(self.session.posts), 1)
        self.assertFalse(self.rows("generated"))
        self.generate()
        self.assertEqual(len(self.session.posts), 1)
        self.assertEqual(len(self.rows("generated")), 1)
        self.assertEqual(len(self.rows()), 1)

    def test_interruption_after_result_commit_rebuilds_accepted_and_manifest(self):
        self.prepare(sources=self.sources[:2])
        with patch.object(self.module, "_accepted", side_effect=OSError("accepted link interrupted")):
            with self.assertRaisesRegex(OSError, "accepted link"):
                self.generate()
        self.assertEqual(len(self.rows()), 1)
        self.assertFalse(self.rows("training_manifest"))
        self.generate()
        row = self.rows()[0]
        accepted = self.images / "accepted/night" / f"{row['sample_id']}.png"
        self.assertEqual(accepted.resolve(), Path(row["output_path"]).resolve())
        self.assertEqual(len(self.rows("training_manifest")), 1)
        self.assertEqual(len(self.session.posts), 1)

    def test_reuse_source_conflict_falls_back_without_losing_place_coverage(self):
        sources = self.sources[:2]
        original, _ = self.module.make_plan(sources, {})
        job = next(j for j in original if j["condition"] == "night")
        other = next(r for r in sources if r["place_id"] != job["place_id"])
        alternate = self.root / f"{job['city']}_{job['place_id']:07d}_alternate.png"
        alternate.write_bytes(Path(other["source_path"]).read_bytes())
        candidate = {**job, "source_path": str(alternate), "source_sha256": common.file_sha256(alternate)}
        key = (job["city"], job["place_id"], job["condition"])
        with patch.object(self.module.source_screen_v3, "screen_image", return_value={
                "passed": True, "source_sha256": candidate["source_sha256"]}):
            planned, reused = self.module.make_plan(sources, {key: [candidate]})
        self.assertEqual(planned, original)
        self.assertFalse(reused)
        self.assertEqual(len({j["source_sha256"] for j in planned}), 2)

    def test_changed_saved_sidecar_is_rejected_before_cleanup_or_new_post(self):
        self.prepare(sources=self.sources[:2])
        self.verifier.error = RuntimeError("quality interrupted")
        with self.assertRaises(RuntimeError):
            self.generate()
        row = self.rows("generated")[0]
        sidecar = Path(row["raw_output_path"]).with_suffix(".json")
        sidecar.write_text("{}")
        self.verifier.error = None
        with self.assertRaisesRegex(ValueError, "sidecar missing or changed"):
            self.generate()
        self.assertEqual(len(self.session.posts), 1)
        self.assertFalse(self.rows())


if __name__ == "__main__":
    unittest.main()
