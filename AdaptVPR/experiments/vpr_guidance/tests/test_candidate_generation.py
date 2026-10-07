"""CPU/mock regression checks for deterministic and recoverable generation."""

import json
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

GUIDANCE_ROOT = Path(__file__).resolve().parents[1]
if str(GUIDANCE_ROOT) not in sys.path:
    sys.path.insert(0, str(GUIDANCE_ROOT))

import generate_candidates as generation  # noqa: E402
from common import candidate_seed, read_jsonl, write_json, write_jsonl  # noqa: E402
import adapters.iclight_sd15_fc as adapter  # noqa: E402
import verification.evaluator as verification  # noqa: E402
from prompts.rules import global_negative_prompt  # noqa: E402


class PromptSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "prompts.jsonl"
        self.rows = [
            {"sample_id": f"id_{index}", "route": "global", "city": "A", "condition": "rain", "prompt": "fixed"}
            for index in range(6)
        ]
        write_jsonl(self.path, self.rows + [{"route": "dual"}, {"route": "global", "city": "B"}])

    def test_filters_before_fixed_shuffle_and_round_slices_do_not_overlap(self):
        full = generation.select_prompts(self.path, ["A"], ["rain"], 42, 0, 6)
        first = generation.select_prompts(self.path, ["A"], ["rain"], 42, 0, 3)
        second = generation.select_prompts(self.path, ["A"], ["rain"], 42, 3, 3)
        self.assertEqual(full, first + second)
        self.assertFalse({row["sample_id"] for row in first} & {row["sample_id"] for row in second})

    def test_invalid_slice_is_rejected(self):
        for offset, count in [(-1, 1), (0, 0), (5, 2)]:
            with self.subTest(offset=offset, count=count), self.assertRaises(ValueError):
                generation.select_prompts(self.path, ["A"], None, 42, offset, count)

    def test_duplicate_id_and_sanitized_filename_collision_are_rejected_before_slicing(self):
        for identities in [("same", "same"), ("a/b", "a_b")]:
            rows = [{**row, "sample_id": identities[index % 2]} for index, row in enumerate(self.rows[:2])]
            write_jsonl(self.path, rows)
            with self.subTest(identities=identities), self.assertRaises(ValueError):
                generation.select_prompts(self.path, None, None, 42, 0, 1)

    def test_seed_is_stable_and_depends_on_candidate_identity(self):
        expected = candidate_seed(42, "adapt_000317", 0)
        self.assertEqual(expected, 802961716)  # Existing Claude candidate provenance.
        self.assertEqual(expected, candidate_seed(42, "adapt_000317", 0))
        self.assertNotEqual(expected, candidate_seed(42, "adapt_000317", 1))
        self.assertNotEqual(expected, candidate_seed(43, "adapt_000317", 0))
        self.assertTrue(1 <= expected <= 2_000_000_000)


class CandidateGenerationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.images = self.root / "sources"
        (self.images / "A").mkdir(parents=True)
        self.source = self.images / "A" / "street.jpg"
        Image.new("RGB", (16, 16), (20, 40, 60)).save(self.source)
        self.prompts = self.root / "prompts.jsonl"
        self.entries = [{"sample_id": "sample", "source_id": "street.jpg", "city": "A",
                         "route": "global", "condition": "rain", "prompt": "released frozen rain prompt"}]
        write_jsonl(self.prompts, self.entries)
        self.output = self.root / "output"
        self.manifest = self.output / "candidates.jsonl"
        self.requests = []
        self.evaluated = []
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.load = self.stack.enter_context(patch.object(adapter, "load_pipeline", return_value=(None, None, None)))
        self.stack.enter_context(patch.object(adapter, "generate", side_effect=self.generate))
        evaluator = SimpleNamespace(evaluate=self.evaluate)
        self.evaluator_init = self.stack.enter_context(patch.object(verification, "DualTraitEvaluator", return_value=evaluator))

    def generate(self, request):
        self.requests.append(request)
        path = Path(os.environ["ICLIGHT_OUTPUT_DIR"]) / f"candidate_{len(self.requests)}.png"
        image = Image.new("RGB", (16, 16))
        image.putdata([((request.seed + index * 9) % 256, index % 256, index * 3 % 256) for index in range(256)])
        image.save(path)
        return {"result_path": str(path)}

    def evaluate(self, source, generated, entry):
        self.evaluated.append((generated.tobytes(), entry))
        return SimpleNamespace(s_geo=0.9, s_div=0.2, passed=True)

    def invoke(self, *extra):
        generation.main(["--prompts", str(self.prompts), "--image-root", str(self.images),
                         "--output-dir", str(self.output), "--num-sources", "1", "--num-candidates", "2", *extra])

    def test_reuses_frozen_prompt_rain_policy_global_verifier_and_saved_jpeg(self):
        self.invoke()
        self.assertEqual(len(self.requests), 2)
        for index, request in enumerate(self.requests):
            self.assertEqual(request.prompt, self.entries[0]["prompt"])
            self.assertEqual(request.negative_prompt, global_negative_prompt())
            self.assertEqual(request.highres_denoise, adapter.RAIN_HIGHRES_DENOISE)
            self.assertEqual(request.seed, candidate_seed(42, "sample", index))
            row = read_jsonl(self.manifest)[index]
            with Image.open(row["output_path"]) as saved:
                self.assertEqual(saved.convert("RGB").tobytes(), self.evaluated[index][0])
            self.assertEqual(self.evaluated[index][1], {"route": "global", "weather": "rain"})
            self.assertTrue(row["passed"])
        config = json.loads((self.output / "generation_config.json").read_text())
        complete = json.loads((self.output / "generation_complete.json").read_text())
        self.assertEqual(complete["config_fingerprint"], config["fingerprint"])
        self.assertEqual(complete["candidate_count"], 2)
        self.assertEqual(complete["passed_count"], 2)
        self.assertFalse(list(self.output.glob(".adapter_tmp_*")))

    def test_complete_resume_does_not_load_models_or_duplicate_rows(self):
        self.invoke()
        before = self.manifest.read_bytes()
        self.invoke()
        self.assertEqual(self.load.call_count, 1)
        self.assertEqual(self.evaluator_init.call_count, 1)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.manifest.read_bytes(), before)

    def test_interrupted_final_json_row_resumes_exact_candidate_seed(self):
        self.invoke()
        first = read_jsonl(self.manifest)[0]
        write_jsonl(self.manifest, [first])
        with self.manifest.open("ab") as handle:
            handle.write(b'{"sample_id": "sam\xe4')  # Interrupted UTF-8 and JSON append.
        self.invoke()
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(self.requests[-1].seed, candidate_seed(42, "sample", 1))
        self.assertEqual(len(read_jsonl(self.manifest)), 2)
        self.assertTrue(self.manifest.read_bytes().endswith(b"\n"))

    def test_valid_unterminated_last_row_is_kept_and_new_append_has_newline(self):
        self.invoke()
        first = read_jsonl(self.manifest)[0]
        self.manifest.write_text(json.dumps(first), encoding="utf-8")
        self.invoke()
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(len(read_jsonl(self.manifest)), 2)

    def test_missing_or_changed_jpeg_regenerates_only_its_candidate(self):
        self.invoke()
        second = read_jsonl(self.manifest)[1]
        Path(second["output_path"]).unlink()
        self.invoke()
        self.assertEqual(len(self.requests), 3)
        Image.new("RGB", (16, 16), "white").save(second["output_path"])
        self.invoke()
        self.assertEqual(len(self.requests), 4)
        self.assertTrue(all(request.seed == candidate_seed(42, "sample", 1) for request in self.requests[2:]))

    def test_changed_config_is_rejected_without_overwriting_artifacts(self):
        self.invoke()
        config = (self.output / "generation_config.json").read_bytes()
        manifest = self.manifest.read_bytes()
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            self.invoke("--num-candidates", "3")
        self.assertEqual((self.output / "generation_config.json").read_bytes(), config)
        self.assertEqual(self.manifest.read_bytes(), manifest)
        self.assertEqual(self.load.call_count, 1)

    def test_changed_prompt_or_source_content_prevents_wrong_resume(self):
        self.invoke()
        self.entries[0]["prompt"] = "changed frozen prompt"
        write_jsonl(self.prompts, self.entries)
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            self.invoke()
        self.entries[0]["prompt"] = "released frozen rain prompt"
        write_jsonl(self.prompts, self.entries)
        Image.new("RGB", (16, 16), "green").save(self.source)
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            self.invoke()
        self.assertEqual(self.load.call_count, 1)

    def test_lora_checkpoint_identity_is_content_bound(self):
        lora = self.root / "lora.safetensors"
        lora.write_bytes(b"weights_v1")
        self.invoke("--lora", str(lora))
        lora.write_bytes(b"weights_v2")
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            self.invoke("--lora", str(lora))
        self.assertEqual(self.load.call_count, 1)

    def test_verifier_implementation_change_prevents_reusing_old_candidates(self):
        self.invoke()
        original_hash = generation.file_sha256

        def changed_verifier(path):
            return "changed-verifier-code" if Path(path).name == "evaluator.py" else original_hash(path)

        with patch.object(generation, "file_sha256", side_effect=changed_verifier):
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                self.invoke()
        self.assertEqual(self.load.call_count, 1)

    def test_legacy_manifest_is_strictly_validated_and_upgraded_without_regeneration(self):
        self.invoke()
        config_path = self.output / "generation_config.json"
        config = json.loads(config_path.read_text())
        legacy_keys = ("prompts", "image_root", "output_dir", "cities", "conditions", "offset", "num_sources",
                       "num_candidates", "lora", "seed", "generator", "sample_ids")
        write_json(config_path, {key: config[key] for key in legacy_keys})
        rows = [{key: value for key, value in row.items() if key not in
                 {"source_sha256", "output_sha256", "config_fingerprint", "source_id"}} for row in read_jsonl(self.manifest)]
        write_jsonl(self.manifest, rows)
        self.invoke()
        self.assertEqual(self.load.call_count, 1)
        self.assertIn("fingerprint", json.loads(config_path.read_text()))
        self.assertTrue(all("output_sha256" in row for row in read_jsonl(self.manifest)))
        self.assertTrue(all(row["provenance"] == "legacy_identity_validated" for row in read_jsonl(self.manifest)))
        complete = json.loads((self.output / "generation_complete.json").read_text())
        self.assertEqual(complete["legacy_candidate_count"], 2)
        self.assertIn("not recorded", complete["legacy_provenance_note"])

    def test_duplicate_or_mismatched_candidate_identity_is_rejected(self):
        self.invoke()
        rows = read_jsonl(self.manifest)
        write_jsonl(self.manifest, [*rows, rows[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.invoke()
        rows[0]["seed"] += 1
        write_jsonl(self.manifest, rows)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.invoke()

    def test_invalid_interior_json_is_not_silently_recovered(self):
        self.invoke()
        self.manifest.write_text('{broken}\n' + self.manifest.read_text(), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid candidate JSON"):
            self.invoke()

    def test_source_path_escape_and_missing_lora_are_rejected_before_model_loading(self):
        outside = self.root / "outside.jpg"
        Image.new("RGB", (16, 16), "red").save(outside)
        self.entries[0]["source_id"] = str(outside)
        write_jsonl(self.prompts, self.entries)
        with self.assertRaisesRegex(ValueError, "outside --image-root"):
            self.invoke()
        with self.assertRaisesRegex(ValueError, "existing checkpoint"):
            self.invoke("--lora", str(self.root / "missing.safetensors"))
        self.load.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_two_different_prompt_ids_may_share_one_source(self):
        write_jsonl(self.prompts, [*self.entries, {**self.entries[0], "sample_id": "other", "condition": "night"}])
        self.invoke("--num-sources", "2")
        rows = read_jsonl(self.manifest)
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["sample_id"] for row in rows}, {"sample", "other"})
        self.assertEqual(len({row["source_path"] for row in rows}), 1)
        night_request = next(request for request in self.requests if request.highres_denoise == adapter.DEFAULT_HIGHRES_DENOISE)
        self.assertEqual(night_request.prompt, self.entries[0]["prompt"])

    def test_failed_verification_does_not_commit_candidate_and_restores_environment(self):
        with patch.dict(os.environ, {"ICLIGHT_OUTPUT_DIR": "original-output", "ADAPTVPR_LORA_CHECKPOINT": "original-lora"}), \
                patch.object(verification, "DualTraitEvaluator", return_value=SimpleNamespace(evaluate=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("verifier failure")))):
            with self.assertRaisesRegex(RuntimeError, "verifier failure"):
                self.invoke()
            self.assertEqual(os.environ["ICLIGHT_OUTPUT_DIR"], "original-output")
            self.assertEqual(os.environ["ADAPTVPR_LORA_CHECKPOINT"], "original-lora")
        self.assertEqual(read_jsonl(self.manifest), [])
        self.assertFalse(list((self.output / "images").iterdir()))
        self.assertFalse((self.output / "generation_complete.json").exists())
        self.assertFalse(list(self.output.glob(".adapter_tmp_*")))


class AtomicJsonTests(unittest.TestCase):
    def test_failed_serialization_preserves_previous_jsonl_and_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            write_jsonl(path, [{"before": True}])
            before = path.read_bytes()
            with self.assertRaises(TypeError):
                write_jsonl(path, [{"first": True}, {"unserializable": object()}])
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_jsonl_decode_error_includes_path_and_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text('{}\n{bad}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"rows.jsonl:2: invalid JSON"):
                read_jsonl(path)


if __name__ == "__main__":
    unittest.main()
