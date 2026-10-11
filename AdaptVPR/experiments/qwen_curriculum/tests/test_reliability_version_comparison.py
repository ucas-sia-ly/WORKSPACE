"""CPU-only comparisons against independent original and module-version snapshots."""
from copy import deepcopy
from pathlib import Path
import shutil
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.qwen_curriculum import common, reliability_comparison as compare
from experiments.qwen_curriculum.tests import test_reliability_comparison as fixtures


class ReliabilityVersionComparisonTests(unittest.TestCase):
    # Reuse the complete four-arm fixture, including independently sealed native
    # evaluations, without inheriting and rerunning the historical test methods.
    setUp = fixtures.ReliabilityComparisonTests.setUp
    freeze = fixtures.ReliabilityComparisonTests.freeze
    new_experiment = fixtures.ReliabilityComparisonTests.new_experiment

    def version_one(self):
        directory = self.root / "v1"
        shutil.copytree(self.baseline, directory)
        config = {k: deepcopy(v) for k, v in self.config.items() if k != "fingerprint"}
        config["reliability"] = {"enabled": True, "version": 1}
        config = compare._sealed(config)
        common.write_json(directory / "experiment_config.json", config)
        report = {k: deepcopy(v) for k, v in self.report.items() if k != "fingerprint"}
        report["experiment_fingerprint"] = config["fingerprint"]
        initial = {k: v for k, v in report["shared_initialization"].items() if k != "fingerprint"}
        initial.update(experiment_fingerprint=config["fingerprint"],
                       standard_aggregator_state_sha256="same standard SALAD initialization",
                       initial_aggregator_state_sha256="V1 full aggregator including reliability")
        report["shared_initialization"] = compare._sealed(initial)
        for arm in compare.ARMS:
            common.write_json(directory / arm / "initialization.json", report["shared_initialization"])
            for domain in compare.DOMAINS:
                path = directory / arm / "evaluation" / f"SVOX_{domain}.json"
                row = compare._read_sealed(path)
                row.pop("fingerprint")
                row["experiment_fingerprint"] = config["fingerprint"]
                common.write_json(path, compare._sealed(row))
        common.write_json(directory / "comparison.json", compare._sealed(report))
        return directory

    def rebind(self, config, report):
        config = {k: deepcopy(v) for k, v in config.items() if k != "fingerprint"}
        config = compare._sealed(config)
        report = {k: deepcopy(v) for k, v in report.items() if k != "fingerprint"}
        report["experiment_fingerprint"] = config["fingerprint"]
        initial = report["shared_initialization"]
        initial.pop("fingerprint")
        initial["experiment_fingerprint"] = config["fingerprint"]
        report["shared_initialization"] = compare._sealed(initial)
        return config, compare._sealed(report)

    def version_two(self):
        config, report = self.new_experiment()
        previous = compare.freeze_baseline(self.version_one(), self.output, self.args,
                                           snapshot_subdir="previous_baseline")
        config["previous_baseline"] = previous
        return self.rebind(config, report)

    def test_original_default_and_legacy_snapshot_directory(self):
        config, report = self.new_experiment()
        self.assertEqual(config["baseline"]["snapshot_subdir"], "baseline")
        comparison = compare.build_module_comparison(config, report, self.output)
        self.assertEqual(comparison["state"], "complete")
        # Older persisted descriptors omit snapshot_subdir and still use baseline.
        config["baseline"].pop("snapshot_subdir")
        config, report = self.rebind(config, report)
        legacy = compare.build_module_comparison(config, report, self.output)
        self.assertEqual(legacy["comparisons"], comparison["comparisons"])

    def test_new_module_full_initialization_can_differ_with_same_standard_and_backbone(self):
        config, report = self.version_two()
        old = compare._read_sealed(self.output / "previous_baseline/comparison.json")
        self.assertNotEqual(old["shared_initialization"]["initial_aggregator_state_sha256"],
                            report["shared_initialization"]["initial_aggregator_state_sha256"])
        comparison = compare.build_module_comparison(config, report, self.output,
                                                     descriptor_key="previous_baseline")
        self.assertEqual(comparison["baseline_experiment_fingerprint"], old["experiment_fingerprint"])
        compare._verify_sealed(comparison, "version comparison")
        for domains in comparison["comparisons"].values():
            for row in domains.values():
                for metric in compare.METRICS:
                    self.assertAlmostEqual(row["arms"]["true"]["delta_percentage_points"][metric], 2)
                    self.assertAlmostEqual(row["arms"]["generated"]["delta_percentage_points"][metric], 5)
                    self.assertAlmostEqual(row["paired_benefit_percentage_points"]["benefit_change"][metric], 3)

    def test_changed_standard_or_backbone_initialization_is_rejected(self):
        config, report = self.version_two()
        for key in ("standard_aggregator_state_sha256", "initial_backbone_state_sha256"):
            with self.subTest(key=key):
                changed = deepcopy(report)
                changed.pop("fingerprint")
                initial = changed["shared_initialization"]
                initial.pop("fingerprint")
                initial[key] = "different initialization"
                changed["shared_initialization"] = compare._sealed(initial)
                with self.assertRaisesRegex(ValueError, "initial weights differ"):
                    compare.build_module_comparison(config, compare._sealed(changed), self.output,
                                                    descriptor_key="previous_baseline")

    def test_original_and_version_snapshots_remain_separate_and_immutable(self):
        config, report = self.version_two()
        original = (self.output / "baseline/comparison.json").read_bytes()
        previous = (self.output / "previous_baseline/comparison.json").read_bytes()
        self.assertNotEqual(original, previous)
        for key in ("baseline", "previous_baseline"):
            descriptor = config[key]
            for name, digest in descriptor["snapshot_files_sha256"].items():
                self.assertEqual(common.file_sha256(self.output / descriptor["snapshot_subdir"] / name), digest)
        compare.build_module_comparison(config, report, self.output)
        compare.build_module_comparison(config, report, self.output, descriptor_key="previous_baseline")
        (self.output / "previous_baseline/comparison.json").write_text("corrupted V1 snapshot")
        with self.assertRaisesRegex(ValueError, "baseline snapshot"):
            compare.build_module_comparison(config, report, self.output, descriptor_key="previous_baseline")
        self.assertEqual((self.output / "baseline/comparison.json").read_bytes(), original)

    def test_snapshot_subdirectory_cannot_escape_output(self):
        for name in ("../old", "/absolute", "nested/dir", ".", "..", ""):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "single directory name"):
                    compare.freeze_baseline(self.baseline, self.output, self.args, snapshot_subdir=name)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
