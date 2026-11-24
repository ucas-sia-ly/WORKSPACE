"""No-model tests for the HTTP budget, saved-image verification and durable resume."""

import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack, redirect_stdout
from unittest.mock import patch

from PIL import Image
import requests

ADAPTVPR_ROOT = Path(__file__).resolve().parents[3]
if str(ADAPTVPR_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTVPR_ROOT))

from experiments.qwen_curriculum import common, run  # noqa: E402


HEALTH = {"status": "ok", "model_loaded": True, "generator_ready": True,
          "source_modified": False, "error": None, "model_id": "Qwen/Qwen-Image-Edit-2511",
          "canvas_policy": "source_aspect_v1", "sampling": {"infer_steps": 4, "guidance_scale": 1.0}}


class Response:
    def __init__(self, value):
        self.value = value

    def raise_for_status(self):
        pass

    def json(self):
        return self.value


class Session:
    def __init__(self, directory, service_dir):
        self.directory, self.service_dir = directory, service_dir
        self.gets, self.posts = [], []
        self.error = None
        self.trust_env = True

    def get(self, url, timeout):
        self.gets.append((url, timeout))
        return Response(dict(HEALTH))

    def post(self, url, json, timeout):
        # This observes durable disk state at the exact outbound-call boundary.
        ledger = common.read_jsonl(self.directory / "attempts.jsonl")
        assert ledger[-1]["sample_id"] == Path(json["image_path"]).stem
        assert ledger[-1]["attempt_sha256"] == run.sealed_attempt(ledger[-1])["attempt_sha256"]
        assert len(ledger) == len(self.posts) + 1
        self.posts.append((url, json, timeout))
        if self.error is not None:
            raise self.error
        self.service_dir.mkdir(exist_ok=True)
        output = self.service_dir / f"response{len(self.posts)}.png"
        with Image.open(json["image_path"]) as source:
            dimensions = list(source.size)
        Image.new("RGB", (30, 24), (50, 70, 90 + len(self.posts))).save(output)
        return Response({"result_path": str(output), "canvas_policy": "source_aspect_v1",
                         "source_dimensions": dimensions, "raw_dimensions": [30, 24],
                         "target_shape": [24, 30]})


class Verifier:
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
                "condition": condition, "rejection_reasons": [] if self.accept else ["weather"]}


class RunTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "run"
        self.directory.mkdir()
        self.service_dir = self.root / "service"
        self.sources, self.jobs = [], []
        for index, condition in enumerate(("night", "snow", "fog")):
            source = self.root / f"Bangkok_{index:07d}_{index:02d}_18.0_100.0.jpg"
            Image.new("RGB", (20, 16), (index * 70, 80, 30)).save(source)
            self.sources.append(source)
            job = {"sample_id": "qwen_" + common.fingerprint(index)[:24],
                   "source_path": str(source), "source_sha256": common.file_sha256(source),
                   "source_dimensions": [20, 16], "city": "Bangkok", "place_id": index,
                   "condition": condition, "prompt": f"change to {condition}", "negative_prompt": "", "seed": index + 1}
            self.jobs.append({**job, "record_sha256": common.fingerprint(job)})
        common.write_jsonl(self.directory / "plan.jsonl", self.jobs)
        plan_config = {"num_images": 3, "plan_sha256": common.file_sha256(self.directory / "plan.jsonl")}
        common.write_json(self.directory / "plan_config.json", {**plan_config, "fingerprint": common.fingerprint(plan_config)})
        self.session = Session(self.directory, self.service_dir)
        self.verifier = Verifier()
        self.args = run.parse_args(["generate", "--run-dir", str(self.directory), "--device", "cpu"])

    def test_new_quality_policy_cannot_overwrite_a_frozen_legacy_run(self):
        legacy = run.execution_definition(self.args)
        legacy["quality"]["version"] = 1
        legacy["quality"]["thresholds"]["min_matches"] = 64
        run.freeze(self.directory, legacy)
        config_path = self.directory / "execution_config.json"
        original = config_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "Immutable execution settings changed"):
            run.freeze(self.directory, run.execution_definition(self.args))
        self.assertEqual(config_path.read_bytes(), original)

    def generate(self, verifier_factory=None):
        with ExitStack() as stack:
            stack.enter_context(patch.object(run.requests, "Session", return_value=self.session))
            stack.enter_context(patch.object(run, "service_output_directory", return_value=self.service_dir))
            stack.enter_context(patch.object(run, "QwenQualityVerifier", side_effect=verifier_factory,
                                            return_value=self.verifier))
            stack.enter_context(redirect_stdout(io.StringIO()))
            run.generate(self.args)

    def rows(self, name="results.jsonl"):
        return run.read_rows(self.directory, name)

    def summary(self):
        return json.loads((self.directory / "summary.json").read_text())

    def test_post_reservations_budget_and_complete_resume(self):
        self.args.max_calls = 2
        original = [path.read_bytes() for path in self.sources]
        self.generate()
        self.assertEqual(len(self.session.posts), 2)
        self.assertFalse(self.session.trust_env)
        self.assertEqual(self.summary()["state"], "call_budget_exhausted")
        self.assertEqual(self.summary()["http_calls_reserved"], 2)
        self.assertEqual(len(self.rows("training_manifest.jsonl")), 2)
        self.assertEqual([path.read_bytes() for path in self.sources], original)
        self.assertEqual(self.verifier.calls, [((20, 16), (20, 16), "night"), ((20, 16), (20, 16), "snow")])
        self.generate()
        self.assertEqual(len(self.session.posts), 2)
        self.assertEqual(len(self.verifier.calls), 2)
        self.assertEqual(len(self.rows()), 2)
        self.args.max_calls = 3
        with self.assertRaisesRegex(ValueError, "Immutable execution settings"):
            self.generate()
        self.assertEqual(len(self.session.posts), 2)

    def test_quality_rejection_spends_exactly_one_call_per_source(self):
        self.verifier.accept = False
        self.generate()
        self.generate()
        self.assertEqual(len(self.session.posts), 3)
        self.assertEqual(len(self.rows("training_manifest.jsonl")), 0)
        self.assertEqual(self.summary()["rejected"], 3)
        self.assertEqual(self.summary()["state"], "complete")

    def test_http_timeout_never_retries_reserved_job(self):
        self.args.max_calls = 1
        self.session.error = requests.Timeout("outcome unknown")
        with self.assertRaises(requests.Timeout):
            self.generate()
        self.assertEqual(self.summary()["state"], "stopped_on_generation_error")
        self.assertEqual(self.rows()[0]["status"], "generation_error")
        self.session.error = None
        self.generate()
        self.assertEqual(len(self.session.posts), 1)
        self.assertEqual(len(self.verifier.calls), 0)
        self.assertEqual(self.summary()["http_calls_reserved"], 1)

    def test_quality_failure_preserves_generated_image_and_zero_call_reverification(self):
        self.args.max_calls = 1
        self.verifier.error = RuntimeError("matcher failed")
        with self.assertRaisesRegex(RuntimeError, "matcher failed"):
            self.generate()
        image = Path(self.rows("generated.jsonl")[0]["output_path"])
        before = image.read_bytes()
        self.assertEqual(len(self.rows()), 0)
        self.assertEqual(self.summary()["state"], "stopped_on_verification_error")
        self.verifier.error = None
        self.generate()
        self.assertEqual(len(self.session.posts), 1)
        self.assertEqual(len(self.verifier.calls), 2)
        self.assertEqual(image.read_bytes(), before)
        self.assertEqual(len(self.rows("training_manifest.jsonl")), 1)

    def test_verifier_constructor_failure_also_reuses_generated_bytes(self):
        self.args.max_calls = 1
        with self.assertRaisesRegex(RuntimeError, "weights unavailable"):
            self.generate(verifier_factory=RuntimeError("weights unavailable"))
        self.assertEqual(self.summary()["state"], "stopped_on_verification_error")
        self.assertEqual(len(self.rows("generated.jsonl")), 1)
        self.generate()
        self.assertEqual(len(self.session.posts), 1)
        self.assertEqual(len(self.verifier.calls), 1)

    def test_interrupted_reservation_is_terminal_without_repeating_http(self):
        self.args.max_calls = 1
        self.verifier.error = RuntimeError("stop after save")
        with self.assertRaises(RuntimeError):
            self.generate()
        (self.directory / "generated.jsonl").unlink()
        self.verifier.error = None
        self.generate()
        self.assertEqual(len(self.session.posts), 1)
        self.assertEqual(self.rows()[0]["status"], "generation_error")
        self.assertIn("no repeat POST", self.rows()[0]["error"])

    def test_corrupt_ledger_and_missing_reservation_reject_before_post(self):
        self.args.max_calls = 1
        self.generate()
        attempts = self.rows("attempts.jsonl")
        attempts[0]["call_number"] = 2
        common.write_jsonl(self.directory / "attempts.jsonl", attempts)
        with self.assertRaisesRegex(ValueError, "Invalid durable call ledger"):
            self.generate()
        common.write_jsonl(self.directory / "attempts.jsonl", [])
        with self.assertRaisesRegex(ValueError, "no durable call reservation"):
            self.generate()
        self.assertEqual(len(self.session.posts), 1)

    def test_corrupt_result_and_resealed_wrong_job_reject_before_post(self):
        self.args.max_calls = 1
        self.generate()
        rows = self.rows()
        rows[0]["condition"] = "rain"
        common.write_jsonl(self.directory / "results.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.generate()
        common.write_jsonl(self.directory / "results.jsonl", [run.sealed(rows[0])])
        with self.assertRaisesRegex(ValueError, "exact planned job"):
            self.generate()
        self.assertEqual(len(self.session.posts), 1)

    def test_changed_generated_image_and_source_are_detected(self):
        self.args.max_calls = 1
        self.generate()
        output = Path(self.rows()[0]["output_path"])
        before = output.read_bytes()
        output.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Recorded image missing or changed"):
            self.generate()
        output.write_bytes(before)
        self.sources[0].write_bytes(b"changed original")
        with self.assertRaisesRegex(ValueError, "Recorded image missing or changed"):
            self.generate()
        self.assertEqual(len(self.session.posts), 1)

    def test_torn_jsonl_and_changed_plan_refuse_new_calls(self):
        self.args.max_calls = 1
        self.generate()
        (self.directory / "attempts.jsonl").write_text('{"call_number":')
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            self.generate()
        common.write_jsonl(self.directory / "plan.jsonl", self.jobs[:-1])
        with self.assertRaisesRegex(ValueError, "Plan manifest checksum"):
            self.generate()
        self.assertEqual(len(self.session.posts), 1)

    def test_lock_prevents_another_worker(self):
        with run.run_lock(self.directory):
            with self.assertRaisesRegex(RuntimeError, "Another worker"):
                self.generate()
        self.assertEqual(len(self.session.gets), 0)
        self.assertEqual(len(self.session.posts), 0)

    def test_failed_reservation_write_prevents_outbound_call(self):
        real_write = common.write_jsonl
        def fail_ledger(path, rows):
            if Path(path).name == "attempts.jsonl":
                raise OSError("disk full before reservation")
            return real_write(path, rows)
        with patch.object(common, "write_jsonl", side_effect=fail_ledger):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.generate()
        self.assertEqual(len(self.session.posts), 0)

    def test_cli_hard_budget_local_service_and_cpu_threads(self):
        for extra in (("--max-calls", "1001"), ("--max-calls", "0"), ("--cpu-threads", "0"),
                      ("--qwen-url", "http://external.example/generate")):
            with self.subTest(extra=extra), redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit):
                    run.parse_args(["generate", "--run-dir", str(self.directory), *extra])
        self.assertEqual(self.args.cpu_threads, 4)
        with patch("torch.set_num_threads") as threads, patch.object(run, "generate") as generate:
            run.main(["generate", "--run-dir", str(self.directory), "--cpu-threads", "2"])
        threads.assert_called_once_with(2)
        generate.assert_called_once()

    def test_canonical_import_ignores_unrelated_bare_common_module(self):
        fake = types.ModuleType("common")
        with patch.dict(sys.modules, {"common": fake}):
            imported = importlib.reload(run)
            self.assertIs(imported.common, common)
            self.assertEqual(imported.QwenQualityVerifier.__module__, "experiments.qwen_curriculum.quality")


if __name__ == "__main__":
    unittest.main()
