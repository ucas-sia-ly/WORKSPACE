"""Exercise adaptive main, real artifact seals and interruption recovery on CPU."""

import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

GUIDANCE_ROOT = Path(__file__).resolve().parents[1]
if str(GUIDANCE_ROOT) not in sys.path:
    sys.path.insert(0, str(GUIDANCE_ROOT))

import adaptive_candidates as adaptive  # noqa: E402
import online_scoring  # noqa: E402
import weather_signal  # noqa: E402
from adapters.iclight_sd15_fc import GenerateRequest  # noqa: E402
from common import ADAPTVPR_ROOT, SALAD_ROOT, candidate_seed, read_jsonl, write_jsonl  # noqa: E402


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest_record(record, checksum_field):
    payload = {key: value for key, value in record.items() if key != checksum_field}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class AdaptiveLifecycleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.real_data = self.root / "gsv"
        self.image_root = self.real_data / "Images"
        city_images = self.image_root / "City"
        city_images.mkdir(parents=True)
        dataframes = self.real_data / "Dataframes"
        dataframes.mkdir()
        (dataframes / "City.csv").write_text("fixture metadata\n", encoding="utf-8")
        self.entries = []
        for index, sample_id in enumerate(("a", "b")):
            source = city_images / f"street{index}.jpg"
            Image.new("RGB", (31, 23), (80 + index * 30, 130, 190)).save(source)
            self.entries.append({
                "sample_id": sample_id, "source_id": source.name, "city": "City",
                "condition": "rain", "route": "global", "prompt": "frozen released rain prompt",
            })
        self.prompts = self.root / "prompts.jsonl"
        write_jsonl(self.prompts, self.entries)
        self.checkpoint = self.root / "student.ckpt"
        self.checkpoint.write_bytes(b"injected CPU student checkpoint")
        base_weights = self.root / "base-model.bin"
        base_weights.write_bytes(b"injected diffusion weights")
        ic_weights = self.root / "iclight.bin"
        ic_weights.write_bytes(b"injected IC-Light weights")
        self.output = self.root / "adaptive"
        self.attempts = []
        self.successful = []
        self.fail_at = None
        self.utility_offset = 0.0
        self.actual_generate_one = adaptive._generate_one
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.dict(os.environ, {
            "ICLIGHT_BASE_MODEL_PATH": str(base_weights), "ICLIGHT_MODEL_PATH": str(ic_weights),
            "ADAPTVPR_LORA_CHECKPOINT": "preserved-session-lora",
        }))
        self.select_prompts = stack.enter_context(patch.object(
            adaptive, "select_prompts", side_effect=lambda *args: copy.deepcopy(self.entries),
        ))
        self.backend_init = stack.enter_context(patch.object(
            adaptive, "ExperimentalICBackend", side_effect=self.make_backend,
        ))
        self.evaluator = SimpleNamespace(evaluate=Mock(return_value=SimpleNamespace(
            s_geo=0.9, s_div=0.3, passed=True,
        )))
        self.evaluator_init = stack.enter_context(patch.object(
            weather_signal, "WeatherSignalEvaluator", return_value=self.evaluator,
        ))
        self.weather = SimpleNamespace(measure=Mock(return_value={
            "weather_shift": 10.0, "weather_source_logit": 1.0, "weather_generated_logit": 11.0,
        }))
        self.weather_init = stack.enter_context(patch.object(
            weather_signal, "WeatherSignal", return_value=self.weather,
        ))
        self.scorer = SimpleNamespace(
            score=Mock(side_effect=self.score), metadata=Mock(return_value={
                "checkpoint": str(self.checkpoint), "checkpoint_sha256": digest_file(self.checkpoint),
                "score_settings": {"device": "cpu", "seed": 42}, "plausibility_calibration": None,
            }),
        )
        self.scorer_init = stack.enter_context(patch.object(
            online_scoring, "OnlineScorer", return_value=self.scorer,
        ))
        self.generate_one = stack.enter_context(patch.object(
            adaptive, "_generate_one", side_effect=self.generate,
        ))

    def make_backend(self, strategy, scratch):
        backend = SimpleNamespace(scratch=scratch, GenerateRequest=GenerateRequest)

        def generate(request):
            result = scratch / "backend.png"
            Image.new("RGB", (21, 27), (
                request.seed % 256, (request.seed // 256) % 256, (request.seed // 65536) % 256,
            )).save(result)
            return {"result_path": str(result)}

        backend.generate = generate
        return backend

    def generate(self, backend, evaluator, source, source_path, entry, index, config, scratch, output_dir):
        key = (entry["sample_id"], index)
        self.attempts.append(key)
        if key == self.fail_at:
            raise RuntimeError("injected generation interruption")
        row = self.actual_generate_one(
            backend, evaluator, source, source_path, entry, index, config, scratch, output_dir,
        )
        self.successful.append(key)
        return row

    def score(self, row):
        utility = (0.0 if row["sample_id"] == "a" and row["candidate_index"] == 0 else 0.5)
        return {
            **row, "utility": utility + self.utility_offset, "mined_pairs": 1.0,
            "mining_probability": 0.5, "positive_pairs": 3, "available_real_views": 4,
            "mean_positive_similarity": 0.4, "min_positive_similarity": 0.2,
            "expected_hardest_negative": 0.5, "identity_margin": -0.1,
            "plausible": True, "eligible_for_training": True,
        }

    def invoke(self, policy="adaptive"):
        adaptive.main([
            "--prompts", str(self.prompts), "--image-root", str(self.image_root),
            "--real-data", str(self.real_data), "--checkpoint", str(self.checkpoint),
            "--output-dir", str(self.output), "--num-sources", "2", "--num-candidates", "4",
            "--cities", "City", "--sampling-policy", policy, "--score-args", "--device cpu --num-workers 0",
        ])

    def assert_artifacts(self, *, complete=True):
        config = json.loads((self.output / "generation_config.json").read_text())
        self.assertEqual(config["fingerprint"], digest_record(config, "fingerprint"))
        self.assertEqual(config["prompts_sha256"], digest_file(self.prompts))
        self.assertEqual(config["checkpoint_sha256"], digest_file(self.checkpoint))
        for name, expected in config["implementation_sha256"].items():
            self.assertEqual(expected, digest_file(ADAPTVPR_ROOT / name))
        for name, expected in config["salad_implementation_sha256"].items():
            self.assertEqual(expected, digest_file(SALAD_ROOT / name))
        selected_sources = {row["sample_id"]: row for row in config["selected_prompts"]}
        rows = read_jsonl(self.output / "candidates.jsonl")
        self.assertEqual(len(rows), len({(row["sample_id"], row["candidate_index"]) for row in rows}))
        for row in rows:
            self.assertEqual(row["record_sha256"], digest_record(row, "record_sha256"))
            self.assertEqual(row["config_fingerprint"], config["fingerprint"])
            self.assertEqual(row["seed"], candidate_seed(42, row["sample_id"], row["candidate_index"]))
            self.assertEqual(row["source_sha256"], digest_file(row["source_path"]))
            self.assertEqual(row["source_sha256"], selected_sources[row["sample_id"]]["source_sha256"])
            self.assertEqual(row["output_sha256"], digest_file(row["output_path"]))
            with Image.open(row["output_path"]) as image:
                self.assertEqual(image.format, "JPEG")
                image.verify()
        context = json.loads((self.output / "scoring_context.json").read_text())
        self.assertEqual(context["config_fingerprint"], config["fingerprint"])
        if complete:
            marker = json.loads((self.output / "generation_complete.json").read_text())
            self.assertEqual(marker["config_fingerprint"], config["fingerprint"])
            self.assertEqual(marker["metadata"], context["metadata"])
            self.assertEqual(set(marker["outputs"]), {
                "candidates.jsonl", "scored.jsonl", "selected.jsonl", "summary.json",
            })
            for name, expected in marker["outputs"].items():
                self.assertEqual(expected, digest_file(self.output / name))
            def by_identity(records):
                return {(row["sample_id"], row["candidate_index"]): row for row in records}

            self.assertEqual(by_identity(read_jsonl(self.output / "scored.jsonl")), by_identity(rows))
        self.assertEqual(os.environ["ADAPTVPR_LORA_CHECKPOINT"], "preserved-session-lora")
        self.assertFalse(list(self.output.glob(".adaptive_tmp_*")))
        return rows

    def reset_model_constructor_mocks(self):
        for constructor in (self.backend_init, self.evaluator_init, self.weather_init, self.scorer_init):
            constructor.reset_mock()

    def assert_no_models_loaded(self):
        for constructor in (self.backend_init, self.evaluator_init, self.weather_init, self.scorer_init):
            constructor.assert_not_called()

    def test_adaptive_early_stop_saves_calls_and_preserves_best_eligible_candidate(self):
        self.invoke()
        rows = self.assert_artifacts()
        self.assertEqual(self.successful, [("a", 0), ("a", 1), ("b", 0)])
        self.assertEqual(len(rows), 3)
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(summary["fixed_budget_candidates"], 8)
        self.assertEqual(summary["generation_calls_saved"], 5)
        self.assertEqual(summary["stop_reasons"], {"target_reached": 2})
        self.assertEqual([(r["sample_id"], r["candidate_index"]) for r in read_jsonl(self.output / "selected.jsonl")],
                         [("a", 1), ("b", 0)])
        self.backend_init.assert_called_once()
        self.evaluator_init.assert_called_once()
        self.weather_init.assert_called_once_with(self.evaluator)
        self.scorer_init.assert_called_once()

    def test_fixed_policy_exhausts_all_candidates_despite_qualifying_early_rows(self):
        self.invoke("fixed")
        rows = self.assert_artifacts()
        self.assertEqual(self.successful, [(sample, index) for sample in ("a", "b") for index in range(4)])
        self.assertEqual(len(rows), 8)
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(summary["generation_calls_saved"], 0)
        self.assertEqual(summary["stop_reasons"], {"budget_exhausted": 2})

    def test_interruption_resumes_only_missing_prefix_suffixes(self):
        self.fail_at = ("b", 1)
        with self.assertRaisesRegex(RuntimeError, "generation interruption"):
            self.invoke("fixed")
        prefix = self.assert_artifacts(complete=False)
        self.assertEqual([(r["sample_id"], r["candidate_index"]) for r in prefix],
                         [("a", 0), ("a", 1), ("a", 2), ("a", 3), ("b", 0)])
        self.assertFalse((self.output / "generation_complete.json").exists())
        # Model a torn append after the durable prefix, including interrupted
        # UTF-8. The real manifest reader must repair it before appending.
        with (self.output / "candidates.jsonl").open("ab") as handle:
            handle.write(b'{"sample_id":"b","candidate_index":1,"torn":"\xe4')
        attempts_before = len(self.attempts)
        self.fail_at = None
        self.invoke("fixed")
        rows = self.assert_artifacts()
        self.assertEqual(self.attempts[attempts_before:], [("b", 1), ("b", 2), ("b", 3)])
        self.assertEqual(rows[:len(prefix)], prefix)
        self.assertTrue((self.output / "candidates.jsonl").read_bytes().endswith(b"\n"))

    def test_changed_jpeg_discards_and_regenerates_only_its_group_suffix(self):
        self.invoke("fixed")
        original = self.assert_artifacts()
        corrupt = next(row for row in original if row["sample_id"] == "a" and row["candidate_index"] == 1)
        Image.new("RGB", (21, 27), (1, 2, 3)).save(corrupt["output_path"])
        self.assertNotEqual(corrupt["output_sha256"], digest_file(corrupt["output_path"]))
        calls_before = len(self.attempts)
        self.utility_offset = 1.0
        self.invoke("fixed")
        rows = self.assert_artifacts()
        self.assertEqual(self.attempts[calls_before:], [("a", 1), ("a", 2), ("a", 3)])
        original_by_key = {(row["sample_id"], row["candidate_index"]): row for row in original}
        rows_by_key = {(row["sample_id"], row["candidate_index"]): row for row in rows}
        self.assertEqual(rows_by_key[("a", 0)], original_by_key[("a", 0)])
        for index in range(4):
            self.assertEqual(rows_by_key[("b", index)], original_by_key[("b", index)])
        for index in range(1, 4):
            row = rows_by_key[("a", index)]
            self.assertEqual(row["utility"], 1.5)
            self.assertNotEqual(row["record_sha256"], original_by_key[("a", index)]["record_sha256"])

    def test_complete_run_and_missing_marker_republication_skip_every_model_constructor(self):
        self.invoke()
        original = self.assert_artifacts()
        before = {name: (self.output / name).read_bytes() for name in (
            "candidates.jsonl", "scored.jsonl", "selected.jsonl", "summary.json", "generation_complete.json",
        )}
        generation_calls = len(self.attempts)
        self.reset_model_constructor_mocks()
        self.invoke()
        self.assert_no_models_loaded()
        for name, contents in before.items():
            self.assertEqual((self.output / name).read_bytes(), contents)
        (self.output / "generation_complete.json").unlink()
        (self.output / "selected.jsonl").unlink()
        self.invoke()
        self.assert_no_models_loaded()
        self.assertEqual(len(self.attempts), generation_calls)
        self.assertEqual(self.assert_artifacts(), original)
        for name in ("candidates.jsonl", "scored.jsonl", "selected.jsonl"):
            self.assertEqual((self.output / name).read_bytes(), before[name])

    def test_tampered_row_seal_rejects_resume_before_model_loading(self):
        self.invoke()
        rows = self.assert_artifacts()
        rows[0]["utility"] += 10
        write_jsonl(self.output / "candidates.jsonl", rows)
        self.reset_model_constructor_mocks()
        with self.assertRaisesRegex(ValueError, "record checksum"):
            self.invoke()
        self.assert_no_models_loaded()

    def test_tampered_config_hash_rejects_resume_before_model_loading(self):
        self.invoke()
        self.assert_artifacts()
        config_path = self.output / "generation_config.json"
        config = json.loads(config_path.read_text())
        config["stop"]["utility_strictly_above"] = 2.0
        config_path.write_text(json.dumps(config), encoding="utf-8")
        self.reset_model_constructor_mocks()
        with self.assertRaisesRegex(ValueError, "fingerprint is invalid"):
            self.invoke()
        self.assert_no_models_loaded()


if __name__ == "__main__":
    unittest.main()
