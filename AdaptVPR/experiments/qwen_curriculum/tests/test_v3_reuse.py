"""CPU checks for exact-prompt, source-preserving weather-v3 reuse."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

ADAPTVPR_ROOT = Path(__file__).resolve().parents[3]
if str(ADAPTVPR_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTVPR_ROOT))

from experiments.qwen_curriculum import common, prompt_bank, v3_reuse


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.ledger = self.root / "results.jsonl"
        self.service = dict(model_id="Qwen/Qwen-Image-Edit-2511", model_revision="modelrev",
                            lora_id="lightx2v/Qwen-Image-Edit-2511-Lightning", lora_revision="lorarev",
                            lora_filename="4steps-bf16.safetensors", canvas_policy="source_aspect_v1",
                            sampling=dict(infer_steps=4, guidance_scale=1.0, attn_mode="torch_sdpa"),
                            performance=dict(profile="resident_bf16_48g", resident_dit_blocks=32),
                            canvas=dict(target_shape_order="height,width", target_pixels=720, multiple=1,
                                        min_side=1, max_side=128, rounding="nearest_multiple",
                                        unsupported_aspect_ratio="reject"))
        self.quality = dict(version="3", prompt_bank=prompt_bank.POLICY_VERSION,
                            thresholds=dict(min_s_geo=.7, min_matches=16),
                            weather_signal=dict(weather_text={"night": "a street photo at night"}))
        self.jobs, self.rows = [], []
        self.add()

    def add(self, sample_id=None, place=1, domain="night", variant="sodium_lamps"):
        sample_id = sample_id or f"pilot_{len(self.jobs)}"
        source = self.root / f"{sample_id}.jpg"
        output = self.root / f"{sample_id}.png"
        raw = self.root / f"{sample_id}_raw.png"
        Image.new("RGB", (20, 16), (140, 120, 90)).save(source)
        Image.new("RGB", (20, 16), (20, 15, 10)).save(output)
        Image.new("RGB", (30, 24), (20, 15, 10)).save(raw)
        job = dict(sample_id=sample_id, city="City", place_id=place, condition=domain,
                   prompt_variant=variant, prompt=prompt_bank.prompt(domain, variant), seed=123456789,
                   source_path=str(source), source_sha256=common.file_sha256(source),
                   source_dimensions=[20, 16], prompt_policy="qwen_weather_bank_v3")
        row = dict(**job, output_path=str(output), output_sha256=common.file_sha256(output),
                   raw_output_path=str(raw), passed=True, eligible_for_training=True,
                   geometry_ok=True, weather_ok=True, status="passed", quality_version="3",
                   sampling=dict(canvas_policy="source_aspect_v1", source_dimensions=[20, 16],
                                 target_shape=[24, 30], raw_dimensions=[30, 24]))
        self.jobs.append(job)
        self.rows.append(row)
        return row

    def write(self, *, service=None, quality=None):
        common.write_jsonl(self.root / "plan.jsonl", self.jobs)
        plan_sha = common.file_sha256(self.root / "plan.jsonl")
        config = dict(stage="weather_v3_plan", plan_sha256=plan_sha)
        common.write_json(self.root / "plan_config.json", {**config, "fingerprint": common.fingerprint(config)})
        common.write_json(self.root / "execution_config.json",
                          dict(plan_sha256=plan_sha, service=service or self.service, quality=quality or self.quality))
        common.write_jsonl(self.ledger, self.rows)

    def collect(self, **kwargs):
        return v3_reuse.collect_candidates([self.ledger], expected_service=self.service,
                                           expected_quality=kwargs.pop("expected_quality", self.quality), **kwargs)

    def test_exact_prompt_reuses_across_policy_label_and_preserves_seed(self):
        old_quality = {**self.quality, "prompt_bank": "qwen_weather_bank_v3"}
        self.write(quality=old_quality)
        before = {p: p.read_bytes() for p in self.root.iterdir() if p.is_file()}
        candidates, report = self.collect()
        row = candidates["City", 1, "night"][0]
        self.assertEqual(report["accepted_candidates"], 1)
        self.assertEqual(row["seed"], 123456789)
        self.assertEqual(row["sample_id"], "pilot_0")
        self.assertEqual(row["prompt_policy"], "qwen_weather_bank_v3")
        self.assertEqual(row["negative_prompt"], "")
        self.assertEqual(row["raw_output_sha256"], common.file_sha256(Path(row["raw_output_path"])))
        evidence = row["reuse_provenance"]
        self.assertEqual(evidence["original_seed"], 123456789)
        self.assertEqual(evidence["ledger_sha256"], common.file_sha256(self.ledger))
        self.assertEqual(evidence["execution_sha256"], common.file_sha256(self.root / "execution_config.json"))
        self.assertEqual(evidence["ledger_row_sha256"], common.fingerprint(self.rows[0]))
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.iterdir() if p.is_file()})

    def test_changed_full_prompt_rejected_despite_current_version_label(self):
        self.rows[0]["prompt"] += " different geometry request"
        self.rows[0]["prompt_policy"] = prompt_bank.POLICY_VERSION
        self.write()
        with patch.object(common, "file_sha256", side_effect=AssertionError("no image or plan hashes needed")):
            candidates, report = self.collect()
        self.assertEqual(candidates, {})
        self.assertEqual(report["scanned_rows"], 1)
        self.assertEqual(report["exact_prompt_rows"], 0)

    def test_disabled_domain_and_removed_variant_not_reused(self):
        self.add("overcast", domain="overcast", variant="flat_grey")
        bad = copy.deepcopy(self.rows[0])
        bad.update(sample_id="old_mist", condition="fog", prompt_variant="morning_mist", prompt="old fog prompt")
        self.rows.append(bad)
        self.write()
        _, report = self.collect()
        self.assertEqual(report["scanned_rows"], 3)
        self.assertEqual(report["accepted_candidates"], 1)

    def test_malformed_variant_is_counted_and_skipped(self):
        self.rows[0]["prompt_variant"] = {"bad": "value"}
        self.write()
        candidates, report = self.collect()
        self.assertEqual(candidates, {})
        self.assertEqual(report["scanned_rows"], 1)
        self.assertEqual(report["rejected"]["prompt_not_current_or_domain_disabled"], 1)

    def test_train_disguised_as_png_rejected(self):
        path = Path(self.rows[0]["output_path"])
        Image.new("RGB", (20, 16), "gray").save(path, format="JPEG")
        self.rows[0]["output_sha256"] = common.file_sha256(path)
        self.write()
        self.assertEqual(self.collect()[1]["rejected"]["output_not_png"], 1)

    def test_model_lora_canvas_performance_and_sampling_mismatch_rejected(self):
        for path, value in [(('model_revision',), 'other'), (('lora_revision',), 'other'),
                            (('canvas', 'target_pixels'), 800), (('performance', 'resident_dit_blocks'), 20),
                            (('sampling', 'attn_mode'), 'other')]:
            with self.subTest(path=path):
                changed = copy.deepcopy(self.service)
                target = changed
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                self.write(service=changed)
                candidates, report = self.collect()
                self.assertEqual(candidates, {})
                self.assertEqual(report["rejected"]["service_definition_mismatch"], 1)

    def test_quality_threshold_or_clip_text_mismatch_rejected(self):
        for key in ("thresholds", "weather_signal"):
            changed = copy.deepcopy(self.quality)
            changed[key]["changed"] = True
            self.write(quality=changed)
            self.assertEqual(self.collect()[1]["rejected"]["quality_definition_mismatch"], 1)

    def test_plan_seed_source_and_prompt_must_match_ledger(self):
        original = copy.deepcopy(self.rows[0])
        for key, value in [("seed", 991), ("source_sha256", "a" * 64), ("place_id", 999)]:
            with self.subTest(key=key):
                self.rows[0] = {**original, key: value}
                self.write()
                self.assertEqual(self.collect()[1]["rejected"]["result_differs_from_plan"], 1)

    def test_changed_source_train_and_raw_hash_rejected(self):
        original = copy.deepcopy(self.rows[0])
        for key in ("source_sha256", "output_sha256", "raw_output_sha256"):
            with self.subTest(key=key):
                self.rows[0] = {**original, key: "a" * 64}
                if key == "source_sha256":
                    self.jobs[0][key] = "a" * 64
                else:
                    self.jobs[0]["source_sha256"] = original["source_sha256"]
                self.write()
                self.assertEqual(self.collect()[1]["rejected"]["image_checksum_mismatch"], 1)

    def test_truncated_png_rejected_even_with_updated_hash(self):
        path = Path(self.rows[0]["output_path"])
        path.write_bytes(path.read_bytes()[:-12])
        self.rows[0]["output_sha256"] = common.file_sha256(path)
        self.write()
        candidates, report = self.collect()
        self.assertEqual(candidates, {})
        self.assertEqual(report["rejected"]["candidate_invalid_or_corrupt"], 1)

    def test_missing_raw_rejected(self):
        Path(self.rows[0]["raw_output_path"]).unlink()
        self.write()
        self.assertEqual(self.collect()[1]["rejected"]["image_missing"], 1)

    def test_incorrect_source_train_raw_dimensions_rejected(self):
        for key in ("source_path", "output_path", "raw_output_path"):
            with self.subTest(key=key):
                path = Path(self.rows[0][key])
                original = path.read_bytes()
                Image.new("RGB", (18, 14), "gray").save(path)
                hash_key = {"source_path": "source_sha256", "output_path": "output_sha256",
                            "raw_output_path": "raw_output_sha256"}[key]
                old_hash = self.rows[0].get(hash_key)
                self.rows[0][hash_key] = common.file_sha256(path)
                if key == "source_path":
                    self.jobs[0][hash_key] = self.rows[0][hash_key]
                self.write()
                self.assertEqual(self.collect()[1]["rejected"]["image_dimensions_mismatch"], 1)
                path.write_bytes(original)
                if old_hash is None:
                    self.rows[0].pop(hash_key)
                else:
                    self.rows[0][hash_key] = old_hash
                self.jobs[0]["source_sha256"] = self.rows[0]["source_sha256"]

    def test_response_canvas_and_sampling_cannot_override_execution(self):
        self.rows[0]["sampling"]["infer_steps"] = 8
        self.write()
        self.assertEqual(self.collect()[1]["rejected"]["sampling_parameter_mismatch"], 1)
        self.rows[0]["sampling"].pop("infer_steps")
        self.rows[0]["sampling"]["target_shape"] = [30, 24]
        self.write()
        self.assertEqual(self.collect()[1]["rejected"]["sampling_canvas_mismatch"], 1)

    def test_negative_prompt_cannot_be_imported(self):
        self.jobs[0]["negative_prompt"] = "remove cars"
        self.rows[0]["negative_prompt"] = "remove cars"
        self.write()
        self.assertEqual(self.collect()[1]["rejected"]["negative_prompt_mismatch"], 1)

    def test_multiple_views_per_place_remain_candidates(self):
        self.add("view2", place=1)
        self.add("view3", place=2)
        self.write()
        candidates, report = self.collect()
        self.assertEqual(len(candidates["City", 1, "night"]), 2)
        self.assertEqual(len(candidates["City", 2, "night"]), 1)
        self.assertEqual(report["accepted_candidates"], 3)

    def test_rejected_result_never_inspects_image_bytes(self):
        self.rows[0]["passed"] = False
        self.write()
        with patch.object(common, "file_sha256", side_effect=AssertionError("no image hashes needed")):
            candidates, report = self.collect()
        self.assertEqual(candidates, {})
        self.assertEqual(report["rejected"]["not_accepted"], 1)

    def test_plan_digest_tampering_rejected(self):
        self.write()
        with (self.root / "plan.jsonl").open("a") as handle:
            handle.write("\n")
        self.assertEqual(self.collect()[1]["rejected"]["plan_checksum_mismatch"], 1)

    def test_duplicate_path_argument_not_counted_twice(self):
        self.write()
        candidates, report = v3_reuse.collect_candidates([self.ledger, self.ledger],
                                                        expected_service=self.service, expected_quality=self.quality)
        self.assertEqual(len(candidates["City", 1, "night"]), 1)
        self.assertEqual(report["scanned_rows"], 1)


if __name__ == "__main__":
    unittest.main()
