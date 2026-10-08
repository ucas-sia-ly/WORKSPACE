"""Cache-only extension preserves sealed jobs and rejects provenance drift."""
import json
from collections import Counter
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ADAPTVPR_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ADAPTVPR_ROOT))
from experiments.qwen_curriculum import common, extend_plan, mine_sources, plan, run


def seal(value):
    return {**value, "fingerprint": common.fingerprint(value)}


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.parent = self.root / "original"
        self.output = self.root / "campaign"
        self.mining = self.root / "mining"
        self.cache = self.mining / "cache"
        self.cache.mkdir(parents=True)
        self.real_data = self.root / "real"
        (self.real_data / "Dataframes").mkdir(parents=True)
        self.records = []
        self.cities = ["Alpha", "Beta"]
        for city_index, city in enumerate(self.cities):
            (self.real_data / "Dataframes" / f"{city}.csv").write_text(f"city={city}\n")
            for i in range(10):
                image = self.real_data / f"{city}_{i}.png"
                Image.new("RGB", (12, 10), (i * 20, city_index * 90, 77)).save(image)
                self.records.append({"source_path": str(image), "source_id": image.name,
                    "city": city, "place_id": i // 2, "source_index": len(self.records),
                    "place_index": city_index * 5 + i // 2, "latitude": 0., "longitude": float(i)})
        self.scores = {"hardness": np.array([1. - i / 10 for i in range(10)] * 2),
            "own_loo_positive_similarity": np.full(20, .5),
            "hardest_negative_similarity": np.full(20, .6),
            "centroid_positive_rank": np.ones(20, dtype=np.int64),
            "num_negative_places": np.full(20, 9, dtype=np.int64),
            "reliable": np.ones(20, dtype=bool)}
        selected, audit = mine_sources.select_hard_sources(self.records, self.scores, 4,
            min_quantile=0, max_quantile=1)
        checkpoint = self.root / "teacher.pt"
        checkpoint.write_bytes(b"teacher")
        request = seal({"real_data": str(self.real_data), "cities": self.cities,
            "min_images_per_place": 2, "checkpoint": str(checkpoint),
            "checkpoint_sha256": common.file_sha256(checkpoint),
            "metadata_sha256": {city: common.file_sha256(self.real_data / "Dataframes" / f"{city}.csv")
                                for city in self.cities},
            "source_order_sha256": mine_sources.sequence_fingerprint(self.records),
            "image_inventory": mine_sources.image_inventory(self.records)})
        scoring = seal({"descriptor_fingerprint": request["fingerprint"]})
        mining_config = seal({"descriptor_request": request, "scoring": scoring,
            "selection": {"num_sources": 4, **audit}, "benchmark_data_used": False})
        for row in selected:
            row["source_sha256"] = common.file_sha256(Path(row["source_path"]))
            row["mining_fingerprint"] = mining_config["fingerprint"]
        common.write_jsonl(self.mining / "sources.jsonl", selected)
        common.write_json(self.mining / "mining_config.json", mining_config)
        common.write_json(self.mining / "summary.json", {
            "config_fingerprint": mining_config["fingerprint"],
            "sources_sha256": common.file_sha256(self.mining / "sources.jsonl"),
            "descriptor_cache": str(self.cache)})
        common.write_json(self.cache / "descriptor_config.json", request)
        np.save(self.cache / "descriptors.npy", np.ones((20, 1), dtype=np.float32))
        stat = (self.cache / "descriptors.npy").stat()
        common.write_json(self.cache / "descriptor_complete.json", {
            "config_fingerprint": request["fingerprint"], "shape": [20, 1],
            "file_stat": {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}})
        np.savez(self.cache / "source_scores.npz", **self.scores)
        common.write_json(self.cache / "scores_complete.json", {
            "config_fingerprint": scoring["fingerprint"],
            "sha256": common.file_sha256(self.cache / "source_scores.npz")})
        config, self.jobs = plan.build_plan(SimpleNamespace(sources=self.mining / "sources.jsonl",
            output_dir=self.parent, num_images=4, seed=42, domain_weights=None, domain_stats=None))
        common.write_json(self.parent / "plan_config.json", config)
        common.write_jsonl(self.parent / "plan.jsonl", self.jobs)
        self.execution = run.freeze(self.parent, {"stage": "generate", "max_calls": 4,
            "plan_fingerprint": config["fingerprint"], "implementation_sha256": {},
            "service": {"source_modified": False, "model_id": "Qwen/Qwen-Image-Edit-2511",
                        "canvas_policy": "source_aspect_v1", "sampling": {"infer_steps": 4, "guidance_scale": 1.}}})
        attempts = []
        run.reserve(self.parent, attempts, self.jobs[0], 4, self.execution)
        image = self.root / "saved.png"
        Image.new("RGB", (12, 10), (70, 60, 50)).save(image)
        result = run.sealed({**self.jobs[0], "execution_fingerprint": self.execution["fingerprint"],
            "output_path": str(image), "output_sha256": common.file_sha256(image),
            "passed": True, "eligible_for_training": True, "status": "passed"})
        common.write_jsonl(self.parent / "results.jsonl", [result])
        self.metadata_patch = patch.object(mine_sources, "load_training_sources", return_value=(self.records, self.cities))
        self.metadata_patch.start()
        self.addCleanup(self.metadata_patch.stop)

    def prepare(self, total=8):
        with patch.object(mine_sources, "extract_descriptor_cache", side_effect=AssertionError("No re-extraction")), \
             patch.object(run, "generate", side_effect=AssertionError("No generation")), \
             patch.object(run, "QwenQualityVerifier", side_effect=AssertionError("No quality model")):
            return extend_plan.prepare_extension(self.parent, self.output, total)

    def test_preserves_original_prefix_saved_images_and_disjoint_source_identity(self):
        before = {p: p.read_bytes() for p in self.parent.rglob("*") if p.is_file()}
        result = self.prepare()
        self.assertEqual(result["total_images"], 8)
        self.assertEqual(result["city_quotas"], {"Alpha": 4, "Beta": 4})
        combined = Path(result["combined_plan"]).read_bytes()
        self.assertTrue(combined.startswith(before[self.parent / "plan.jsonl"]))
        rows = common.read_jsonl(Path(result["combined_plan"]))
        self.assertEqual(rows[:4], self.jobs)
        self.assertEqual(len({r["source_path"] for r in rows}), 8)
        self.assertEqual(len({r["source_sha256"] for r in rows}), 8)
        self.assertLessEqual(max(Counter((r["city"], r["place_id"]) for r in rows).values()), 2)
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertFalse(result["mining_provenance"]["descriptor_extraction_performed"])
        additional_config, additional = run.load_plan(Path(result["additional_run_dir"]))
        self.assertEqual(len(additional), 4)
        self.assertEqual(additional_config["seed"], 42)

    def test_idempotent_freeze_and_changed_budget_refused(self):
        first = self.prepare()
        before = {p: p.read_bytes() for p in self.output.rglob("*") if p.is_file()}
        self.assertEqual(first, self.prepare())
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        with self.assertRaisesRegex(ValueError, "campaign configuration changed"):
            self.prepare(10)
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_changed_score_cache_and_metadata_refused(self):
        with (self.cache / "source_scores.npz").open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "Difficulty score cache"):
            self.prepare()
        self.assertFalse(self.output.exists())
        np.savez(self.cache / "source_scores.npz", **self.scores)
        (self.real_data / "Dataframes" / "Alpha.csv").write_text("changed")
        with self.assertRaisesRegex(ValueError, "GSV training metadata"):
            self.prepare()

    def test_changed_training_image_inventory_refused(self):
        Path(self.records[-1]["source_path"]).touch()
        with self.assertRaisesRegex(ValueError, "image inventory"):
            self.prepare()

    def test_existing_result_tamper_and_invalid_budgets_refused(self):
        for total in (4, 1005, True, 8.0):
            with self.subTest(total=total), self.assertRaises(ValueError):
                self.prepare(total)
        results = common.read_jsonl(self.parent / "results.jsonl")
        Path(results[0]["output_path"]).write_bytes(b"bad image")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            self.prepare()

    def test_non_nested_selection_and_duplicate_content_refused(self):
        selected, _ = mine_sources.select_hard_sources(self.records, self.scores, 8, min_quantile=0, max_quantile=1)
        for row in selected:
            row["source_sha256"] = common.file_sha256(Path(row["source_path"]))
        with self.assertRaisesRegex(ValueError, "preserve every original"):
            extend_plan._additional_sources(self.jobs, selected[1:], 2)
        new = next(row for row in selected if row["source_path"] not in {r["source_path"] for r in self.jobs})
        new["source_sha256"] = self.jobs[0]["source_sha256"]
        with self.assertRaisesRegex(ValueError, "duplicates an original"):
            extend_plan._additional_sources(self.jobs, selected, 2)

    def test_tampered_combined_manifest_is_not_republished(self):
        self.prepare()
        path = self.output / "combined_plan.jsonl"
        with path.open("ab") as handle:
            handle.write(b"\n")
        with self.assertRaisesRegex(ValueError, "Frozen extension artifact changed"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
