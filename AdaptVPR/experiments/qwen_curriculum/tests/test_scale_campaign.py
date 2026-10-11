from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.qwen_curriculum import common, run, plan, fast_campaign, scale_campaign as scale
from experiments.qwen_curriculum.train_scaled import admission_rows


class ScaleCampaignTests(unittest.TestCase):
    def test_bounded_stages_and_exact_weather_total(self):
        self.assertEqual(scale.stage_budgets(5277), [1000] * 5 + [277])
        weights = plan._weights(plan.DEFAULT_DOMAIN_WEIGHTS)
        target = Counter(plan.allocate_domains(7277, weights, 42))
        used = Counter(plan.allocate_domains(2000, weights, 42))
        for n in scale.stage_budgets(5277):
            used.update(plan.allocate_domains(n, weights if n == 1000 else dict(target - used), 42))
        self.assertEqual(used, target)
        self.assertEqual(sum(used.values()), 7277)

    def test_duplicates_are_rejected_across_stages(self):
        rows = [{"sample_id": "a", "source_path": "one", "source_sha256": "1"},
                {"sample_id": "b", "source_path": "two", "source_sha256": "2"}]
        scale.validate_unique(rows)
        for key in rows[0]:
            changed = deepcopy(rows)
            changed[1][key] = changed[0][key]
            with self.assertRaisesRegex(ValueError, key):
                scale.validate_unique(changed)

    def test_training_selection_keeps_original_quality_evidence(self):
        rows = [{"sample_id": f"id{i}", "source_path": f"s{i}", "output_path": f"o{i}",
            "source_sha256": "a", "output_sha256": "b", "city": "city", "place_id": 1,
            "condition": "night", "passed": passed, "eligible_for_training": passed,
            "status": "passed" if passed else "rejected", "result_sha256": f"seal{i}",
            "execution_fingerprint": "exec"} for i, passed in enumerate((True, False))]
        original = deepcopy(rows)
        all_rows = admission_rows(rows, "all-successful")
        self.assertEqual(len(all_rows), 2)
        self.assertFalse(all_rows[1]["original_quality_passed"])
        self.assertEqual(all_rows[1]["original_result_sha256"], "seal1")
        self.assertEqual(len(admission_rows(rows, "accepted-only")), 1)
        self.assertEqual(rows, original)

    def test_stage_settings_budget_reservation_and_image_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            source = directory / "source.png"
            Image.new("RGB", (8, 6), "red").save(source)
            common.write_jsonl(directory / "sources.jsonl", [{"source_path": str(source),
                "source_sha256": common.file_sha256(source), "city": "Test", "place_id": 1}])
            cfg, jobs = plan.build_plan(SimpleNamespace(sources=directory / "sources.jsonl",
                output_dir=directory, num_images=1, seed=42, domain_weights={"night": 1}, domain_stats=None))
            common.write_json(directory / "plan_config.json", cfg)
            common.write_jsonl(directory / "plan.jsonl", jobs)
            reference = {**run.execution_definition(SimpleNamespace(run_dir=directory, device="cuda", cpu_threads=4)),
                "stage": "generate", "plan_fingerprint": cfg["fingerprint"], "max_calls": 1,
                "qwen_url": "http://127.0.0.1:8001/generate", "request_timeout": 600.0,
                "service": {"sampling": {"infer_steps": 4}}, "call_policy": "one POST per job",
                "source_encoding": "lossless RGB PNG decoded from original"}
            execution = run.freeze(directory, reference)
            stage = {"run_dir": str(directory), "max_calls": 1, "num_images": 1}
            row = run.sealed({**jobs[0], "execution_fingerprint": execution["fingerprint"],
                "output_path": str(source), "output_sha256": common.file_sha256(source),
                "status": "rejected", "passed": False, "eligible_for_training": False})
            for name in ("results.jsonl", "generated.jsonl"):
                common.write_jsonl(directory / name, [row])
            with self.assertRaisesRegex(ValueError, "reservation"):
                scale.check_stage(stage, reference, complete=True)
            run.reserve(directory, [], jobs[0], 1, execution)
            self.assertEqual(scale.check_stage(stage, reference, complete=True), [row])
            altered = deepcopy(reference)
            altered["service"]["sampling"]["infer_steps"] = 3
            with self.assertRaisesRegex(ValueError, "settings differ"):
                scale.check_stage(stage, altered)
            Image.new("RGB", (8, 6), "blue").save(source)
            with self.assertRaisesRegex(ValueError, "changed"):
                scale.check_stage(stage, reference)


if __name__ == "__main__":
    unittest.main()
