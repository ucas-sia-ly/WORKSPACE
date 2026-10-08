"""Model-free curriculum planning, byte provenance and frozen rerun tests."""

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ADAPTVPR_ROOT = Path(__file__).resolve().parents[3]
if str(ADAPTVPR_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTVPR_ROOT))

from experiments.qwen_curriculum import plan  # noqa: E402
from experiments.qwen_curriculum.common import candidate_seed, read_jsonl, write_jsonl  # noqa: E402
from prompts.rules import build_structured_prompt  # noqa: E402


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AllocationTests(unittest.TestCase):
    def test_default_exact_thousand_image_prior_and_shuffle(self):
        labels = plan.allocate_domains(1000, plan.DEFAULT_DOMAIN_WEIGHTS, 42)
        self.assertEqual(Counter(labels), {"night": 500, "snow": 200, "fog": 200, "rain": 100})
        self.assertEqual(labels, plan.allocate_domains(1000, plan.DEFAULT_DOMAIN_WEIGHTS, 42))
        self.assertNotEqual(labels, plan.allocate_domains(1000, plan.DEFAULT_DOMAIN_WEIGHTS, 43))
        self.assertNotEqual(labels, sorted(labels))

    def test_largest_remainder_normalizes_weights_and_breaks_ties_stably(self):
        self.assertEqual(Counter(plan.allocate_domains(7, {"night": 5, "snow": 2, "fog": 2, "rain": 1}, 4)),
                         {"night": 4, "fog": 1, "snow": 1, "rain": 1})
        self.assertEqual(Counter(plan.allocate_domains(3, {"night": 1, "snow": 1, "fog": 1, "rain": 1}, 4)),
                         {"fog": 1, "night": 1, "rain": 1})
        self.assertEqual(plan.allocate_domains(0, {"night": 1}, 3), [])
        self.assertEqual(Counter(plan.allocate_domains(6, {"night": 1e308, "snow": 1e308}, 4)),
                         {"night": 3, "snow": 3})

    def test_unknown_zero_negative_nonfinite_and_duplicate_domains_fail(self):
        for value in ("sun=1", "overcast=1", "night=0", "night=-1", "night=nan", "night=inf",
                      "night=1,night=2", "night", {}, {"rain": True}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                plan.allocate_domains(10, value, 42)


class PlanningTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sources = self.root / "sources.jsonl"
        self.output = self.root / "plan"
        self.rows = []
        for city_index, city in enumerate(("Alpha", "Beta")):
            for index in range(10):
                source = self.root / city / f"view{index}.jpg"
                source.parent.mkdir(exist_ok=True)
                Image.new("RGB", (40, 30), (15 + index * 20, 20 + city_index * 100, 170)).save(source)
                self.rows.append({
                    "source_path": str(source), "source_sha256": file_digest(source),
                    "city": city, "place_id": index // 4, "hardness": 1.0 - index / 10,
                    "source_id": f"{city}/{source.name}", "reliable": True,
                    "mining_fingerprint": "frozen-miner", "own_loo_positive_similarity": 0.3,
                })
        self.write_sources()

    def write_sources(self):
        write_jsonl(self.sources, self.rows)

    def arguments(self, *extra):
        return ["--sources", str(self.sources), "--output-dir", str(self.output), "--num-images", "10", *extra]

    def build(self, *extra):
        return plan.build_plan(plan.parse_args(self.arguments(*extra)))

    def test_build_is_model_free_read_only_reproducible_and_city_balanced(self):
        before = {row["source_path"]: Path(row["source_path"]).read_bytes() for row in self.rows}
        with patch.dict(sys.modules, {"torch": None, "diffusers": None, "transformers": None}):
            first = self.build()
            second = self.build()
        self.assertEqual(first, second)
        self.assertFalse(self.output.exists())
        config, rows = first
        self.assertEqual(len(rows), 10)
        self.assertEqual(len({row["source_path"] for row in rows}), 10)
        self.assertEqual(len({row["image_hash"] for row in rows}), 10)
        self.assertEqual(len({row["sample_id"] for row in rows}), 10)
        self.assertEqual(Counter(row["city"] for row in rows), {"Alpha": 5, "Beta": 5})
        for city in ("Alpha", "Beta"):
            city_rows = [row for row in rows if row["city"] == city]
            self.assertEqual([row["hardness"] for row in city_rows], [1.0 - i / 10 for i in range(5)])
        self.assertEqual(config["domain_weight_origin"], "proposed_prior_not_gift_exact")
        self.assertIn("not GIFT", config["prior_interpretation"])
        self.assertEqual(config["source_pool"], "mined_real_training_pool_no_evaluation_images")
        self.assertEqual(config["variants_per_source"], 1)
        self.assertEqual({row["source_path"]: Path(row["source_path"]).read_bytes() for row in self.rows}, before)

    def test_rows_use_released_prompts_actual_hashes_dimensions_and_deterministic_seeds(self):
        config, rows = self.build()
        self.assertEqual(Counter(row["condition"] for row in rows), config["domain_quotas"])
        for row in rows:
            self.assertIn(row["condition"], {"night", "snow", "fog", "rain"})
            self.assertEqual(row["domain"], row["condition"])
            self.assertEqual(row["prompt"], build_structured_prompt(route="global", weather=row["condition"], occlusion=None))
            self.assertEqual(row["negative_prompt"], "")
            self.assertEqual(row["source_dimensions"], [40, 30])
            self.assertEqual(row["source_sha256"], file_digest(row["source_path"]))
            self.assertEqual(row["image_hash"], row["source_sha256"])
            self.assertEqual(row["seed"], candidate_seed(42, row["sample_id"], 0))
            self.assertEqual(row["mining_info"]["hardness"], row["hardness"])
            self.assertEqual(row["record_sha256"], plan._fingerprint({k: v for k, v in row.items() if k != "record_sha256"}))
        self.assertEqual(config["fingerprint"], plan._fingerprint({k: v for k, v in config.items() if k != "fingerprint"}))
        self.assertEqual(config["sources_sha256"], file_digest(self.sources))
        for name, digest in config["implementation_sha256"].items():
            self.assertEqual(digest, file_digest(ADAPTVPR_ROOT / name))

    def test_write_and_exact_rerun_skip_preserve_all_bytes_and_hashes(self):
        plan.main(self.arguments())
        config_path, rows_path = self.output / "plan_config.json", self.output / "plan.jsonl"
        config = json.loads(config_path.read_text())
        self.assertEqual(config["plan_sha256"], file_digest(rows_path))
        before = (config_path.read_bytes(), rows_path.read_bytes())
        plan.main(self.arguments())
        self.assertEqual(before, (config_path.read_bytes(), rows_path.read_bytes()))
        self.assertEqual(len(read_jsonl(rows_path)), 10)
        rows_path.unlink()
        plan.main(self.arguments())
        self.assertEqual(before, (config_path.read_bytes(), rows_path.read_bytes()))

    def test_request_changes_refuse_to_overwrite_frozen_plan(self):
        plan.main(self.arguments())
        before = (self.output / "plan.jsonl").read_bytes()
        for extra in (("--seed", "43"), ("--num-images", "8"), ("--domain-weights", "night=1,rain=1")):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "configuration changed"):
                plan.main(self.arguments(*extra))
        self.assertEqual((self.output / "plan.jsonl").read_bytes(), before)
        self.rows[0]["hardness"] = 99.0
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            plan.main(self.arguments())
        self.assertEqual((self.output / "plan.jsonl").read_bytes(), before)

    def test_tampered_config_and_plan_checksums_refuse_republication(self):
        plan.main(self.arguments())
        rows_path = self.output / "plan.jsonl"
        with rows_path.open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "plan checksum"):
            plan.main(self.arguments())
        config_path = self.output / "plan_config.json"
        config = json.loads(config_path.read_text())
        config["seed"] = 9
        config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "fingerprint is invalid"):
            plan.main(self.arguments())

    def test_domain_stats_are_validated_hashed_and_can_be_explicitly_overridden(self):
        stats = self.root / "stats.json"
        stats.write_text(json.dumps({"domain_weights": {"night": 2, "fog": 1}, "provenance": "measured training gaps"}))
        config, rows = self.build("--domain-stats", str(stats))
        self.assertEqual(config["domain_weight_origin"], "domain_stats")
        self.assertEqual(config["domain_stats_sha256"], file_digest(stats))
        self.assertEqual(config["domain_stats_provenance"], "measured training gaps")
        self.assertEqual(Counter(row["condition"] for row in rows), {"night": 7, "fog": 3})
        config, rows = self.build("--domain-stats", str(stats), "--domain-weights", "rain=1")
        self.assertEqual(config["domain_weight_origin"], "explicit_user_weights")
        self.assertEqual(Counter(row["condition"] for row in rows), {"rain": 10})
        stats.write_text(json.dumps({"domain_weights": {"overcast": 1}}))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            self.build("--domain-stats", str(stats), "--domain-weights", "rain=1")

    def test_budget_exceeding_distinct_sources_fails_before_any_write(self):
        with self.assertRaisesRegex(ValueError, "only 20 distinct"):
            plan.main(self.arguments("--num-images", "21"))
        self.assertFalse(self.output.exists())

    def test_changed_source_bytes_or_recorded_dimensions_are_rejected(self):
        source = Path(self.rows[0]["source_path"])
        Image.new("RGB", (40, 30), (0, 0, 0)).save(source)
        with self.assertRaisesRegex(ValueError, "image bytes"):
            self.build()
        self.rows[0]["source_sha256"] = file_digest(source)
        self.rows[0]["source_dimensions"] = [400, 300]
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "source_dimensions"):
            self.build()

    def test_duplicate_paths_content_and_conflicting_place_labels_are_rejected(self):
        first = self.rows[0]
        self.rows.append(first.copy())
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "duplicate source path"):
            self.build()
        duplicate = self.root / "same-content.jpg"
        shutil.copyfile(first["source_path"], duplicate)
        self.rows[-1] = {**first, "source_path": str(duplicate)}
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "duplicate source image content"):
            self.build()
        self.rows[-1]["place_id"] += 1
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "conflicting place labels"):
            self.build()

    def test_image_hash_is_mandatory_in_plan_and_invalid_input_hashes_fail(self):
        self.rows[0]["source_sha256"] = "invalid"
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.build()
        self.rows[0]["source_sha256"] = file_digest(self.rows[0]["source_path"])
        self.rows[0]["image_hash"] = "0" * 64
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "image bytes"):
            self.build()


if __name__ == "__main__":
    unittest.main()
