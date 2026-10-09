"""CPU-only historical snapshot validation and module benefit comparisons."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.qwen_curriculum import common, reliability_comparison as compare


class ReliabilityComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.baseline = self.root / "old"
        self.baseline.mkdir()
        self.output = self.root / "new"
        real = self.root / "real"
        (real / "Dataframes").mkdir(parents=True)
        metadata = real / "Dataframes/City.csv"
        metadata.write_text("metadata")
        weights = self.root / "backbone.pth"
        weights.write_bytes(b"backbone weights; no torch deserialization")
        self.args = SimpleNamespace(generation_run_dir=self.root / "generation", real_data=real,
            backbone_weights=weights, backbone_repo=self.root / "backbone", dataset_root=self.root / "svox",
            num_images=4, epochs=2, learning_rate=6e-5, trainable_blocks=4, batch_size=8, seed=42, device="cuda")
        self.inventory, self.generated = [], []
        sources = []
        for i in range(36):
            path = self.root / f"source{i}.jpg"
            path.write_bytes(f"source{i}".encode())
            sources.append(str(path))
            self.inventory.append({"path": str(path), "sha256": common.file_sha256(path), "kind": "source"})
        for i in range(4):
            output = self.root / f"generated{i}.png"
            output.write_bytes(f"generated{i}".encode())
            row = {"source_path": sources[i * 4], "source_sha256": self.inventory[i * 4]["sha256"],
                   "output_path": str(output), "output_sha256": common.file_sha256(output)}
            self.generated.append({**row, "result_sha256": common.fingerprint(row)})
            self.inventory.append({"path": str(output), "sha256": row["output_sha256"], "kind": "generated"})
        groups = [{"sources": sources[i:i + 4], "group_id": i // 4, "label": i // 4}
                  for i in range(0, len(sources), 4)]
        common.write_jsonl(self.baseline / "groups_8to1.jsonl", groups)
        common.write_jsonl(self.baseline / "groups_4to1.jsonl", groups[:5])
        common.write_jsonl(self.baseline / "generated_700.jsonl", self.generated)
        common.write_jsonl(self.baseline / "image_inventory.jsonl", self.inventory)
        self.svox = {"fixture": "native inventory mocked without building 35k images"}
        self.config = compare._sealed({"request": compare._request_from_args(self.args),
            "arms": {arm: {"ratio": ratio, "replace": replace, "source_slots": 4 * (ratio + 1),
                      "generated_per_epoch": 4 if replace else 0,
                      "true_per_epoch": 4 * (ratio if replace else ratio + 1),
                      "schedule": f"groups_{ratio}to1.jsonl"} for arm, (ratio, replace) in compare.ARMS.items()},
            "files_sha256": {name: common.file_sha256(self.baseline / name) for name in compare.FROZEN_FILES},
            "backbone_weights_sha256": common.file_sha256(weights),
            "metadata_sha256": {"City": common.file_sha256(metadata)}, "svox_inventory": self.svox,
            "implementation_sha256": {"deleted_historical_implementation.py": "old implementation SHA"},
            "training": {"views_per_group": 4, "precision": "32", "workers": 0, "augment": False},
            "svox_domains": compare.DOMAINS, "evaluation_protocol": "full native test; 25m UTM positives; final checkpoint only"})
        common.write_json(self.baseline / "experiment_config.json", self.config)
        initialization = compare._sealed({"experiment_fingerprint": self.config["fingerprint"],
            "init_policy": "random_aggregator_pretrained_backbone",
            "backbone_weights_sha256": self.config["backbone_weights_sha256"],
            "initial_backbone_state_sha256": "same backbone initialization",
            "initial_aggregator_state_sha256": "same standard SALAD initialization"})
        self.report = {"state": "complete", "experiment_fingerprint": self.config["fingerprint"],
            "comparisons": {}, "checkpoint_sha256": {}, "shared_initialization": initialization}
        for ratio in (8, 4):
            self.report["comparisons"][str(ratio)] = {}
            for domain in compare.DOMAINS:
                real = {"R@1": 0.4, "R@5": 0.6, "R@10": 0.8}
                generated = {key: value + .1 for key, value in real.items()}
                self.report["comparisons"][str(ratio)][domain] = {"true": real, "generated": generated,
                    "delta_percentage_points": {metric: 100 * (generated[metric] - real[metric])
                                                for metric in compare.METRICS}}
        for arm, (ratio, replace) in compare.ARMS.items():
            folder = self.baseline / arm
            (folder / "evaluation").mkdir(parents=True)
            (folder / "checkpoint.pt").write_bytes(arm.encode())
            self.report["checkpoint_sha256"][arm] = common.file_sha256(folder / "checkpoint.pt")
            spec = self.config["arms"][arm]
            common.write_jsonl(folder / "training_log.jsonl", [{"epoch": epoch,
                "real_exposure": spec["true_per_epoch"], "synthetic_exposure": spec["generated_per_epoch"]}
                for epoch in range(1, self.args.epochs + 1)])
            common.write_json(folder / "initialization.json", initialization)
            for domain, query_folder in compare.DOMAINS.items():
                common.write_json(folder / "evaluation" / f"SVOX_{domain}.json", compare._sealed({
                    "checkpoint_sha256": self.report["checkpoint_sha256"][arm],
                    "experiment_fingerprint": self.config["fingerprint"],
                    "num_references": 17166, "num_queries": compare.QUERY_COUNTS[domain],
                    "num_evaluated_queries": compare.QUERY_COUNTS[domain], "num_queries_without_positives": 0,
                    "image_size": [224, 224], "query_folder": query_folder,
                    "protocol": {"ground_truth": "utm_radius", "positive_radius_meters": 25,
                                 "split": "test", "query_subdirs": [query_folder]},
                    "recall": self.report["comparisons"][str(ratio)][domain]["generated" if replace else "true"]}))
        self.report = compare._sealed(self.report)
        common.write_json(self.baseline / "comparison.json", self.report)
        self.inventory_mock = patch.object(compare, "_svox_inventory", return_value=self.svox)
        self.inventory_mock.start()
        self.addCleanup(self.inventory_mock.stop)

    def freeze(self):
        return compare.freeze_baseline(self.baseline, self.output, self.args)

    def new_experiment(self):
        descriptor = self.freeze()
        config = {k: deepcopy(v) for k, v in self.config.items() if k != "fingerprint"}
        config.update(baseline=descriptor, reliability={"enabled": True})
        config["implementation_sha256"] = {"newcode.py": "new code hash"}
        config["training"]["source_companions"] = "auxiliary supervision only"
        config = compare._sealed(config)
        report = {k: deepcopy(v) for k, v in self.report.items() if k != "fingerprint"}
        report["experiment_fingerprint"] = config["fingerprint"]
        initial = {k: v for k, v in report["shared_initialization"].items() if k != "fingerprint"}
        initial.update(experiment_fingerprint=config["fingerprint"],
                       initial_aggregator_state_sha256="new SALAD and reliability head",
                       standard_aggregator_state_sha256="same standard SALAD initialization")
        report["shared_initialization"] = compare._sealed(initial)
        for domains in report["comparisons"].values():
            for row in domains.values():
                row["true"] = {key: value + .02 for key, value in row["true"].items()}
                row["generated"] = {key: value + .05 for key, value in row["generated"].items()}
                row["delta_percentage_points"] = {key: 100 * (row["generated"][key] - value)
                                                  for key, value in row["true"].items()}
        return config, compare._sealed(report)

    def test_freeze_verifies_all_artifacts_without_loading_historical_code(self):
        descriptor = self.freeze()
        self.assertEqual(self.freeze(), descriptor)
        self.assertEqual(descriptor["experiment_fingerprint"], self.config["fingerprint"])
        self.assertEqual(len(descriptor["snapshot_files_sha256"]), 6)
        for name, digest in descriptor["snapshot_files_sha256"].items():
            self.assertEqual(common.file_sha256(self.output / "baseline" / name), digest)

    def test_existing_changed_snapshot_is_never_overwritten(self):
        self.freeze()
        path = self.output / "baseline/comparison.json"
        path.write_text("changed snapshot")
        with self.assertRaisesRegex(ValueError, "existing baseline snapshot"):
            self.freeze()
        self.assertEqual(path.read_text(), "changed snapshot")

    def test_training_budget_changed_and_checkpoint_changed_are_rejected(self):
        self.args.epochs += 1
        with self.assertRaisesRegex(ValueError, "option differs: epochs"):
            self.freeze()
        self.args.epochs -= 1
        (self.baseline / "generated_4to1/checkpoint.pt").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "generated_4to1 checkpoint"):
            self.freeze()

    def test_image_and_csv_changes_are_rejected(self):
        source = Path(self.inventory[0]["path"])
        original = source.read_bytes()
        source.write_bytes(b"modified image")
        with self.assertRaisesRegex(ValueError, "Baseline bytes changed"):
            self.freeze()
        source.write_bytes(original)
        (self.args.real_data / "Dataframes/City.csv").write_text("modified metadata")
        with self.assertRaisesRegex(ValueError, "City metadata"):
            self.freeze()

    def test_logs_must_contain_every_epoch_with_exact_exposure(self):
        path = self.baseline / "generated_8to1/training_log.jsonl"
        rows = common.read_jsonl(path)
        rows[0]["synthetic_exposure"] -= 1
        common.write_jsonl(path, rows)
        with self.assertRaisesRegex(ValueError, "epochs/exposure"):
            self.freeze()

    def test_sealed_evaluation_with_wrong_native_protocol_is_rejected(self):
        path = self.baseline / "true_4to1/evaluation/SVOX_night.json"
        row = compare._read_sealed(path)
        row.pop("fingerprint")
        row["protocol"]["positive_radius_meters"] = 50
        common.write_json(path, compare._sealed(row))
        with self.assertRaisesRegex(ValueError, "SVOX evaluation differs"):
            self.freeze()

    def test_module_deltas_and_paired_benefit_change_are_percentage_points(self):
        config, report = self.new_experiment()
        result = compare.build_module_comparison(config, report, self.output)
        compare._verify_sealed(result, "result")
        for domains in result["comparisons"].values():
            for row in domains.values():
                for metric in compare.METRICS:
                    self.assertAlmostEqual(row["arms"]["true"]["delta_percentage_points"][metric], 2)
                    self.assertAlmostEqual(row["arms"]["generated"]["delta_percentage_points"][metric], 5)
                    self.assertAlmostEqual(row["paired_benefit_percentage_points"]["old"][metric], 10)
                    self.assertAlmostEqual(row["paired_benefit_percentage_points"]["new"][metric], 13)
                    self.assertAlmostEqual(row["paired_benefit_percentage_points"]["benefit_change"][metric], 3)

    def test_module_comparison_rejects_new_pool_or_original_weight_change(self):
        config, report = self.new_experiment()
        config.pop("fingerprint")
        config["files_sha256"]["groups_4to1.jsonl"] = "another pool"
        config = compare._sealed(config)
        report.pop("fingerprint")
        report["experiment_fingerprint"] = config["fingerprint"]
        report = compare._sealed(report)
        with self.assertRaisesRegex(ValueError, "protocols differ: files_sha256"):
            compare.build_module_comparison(config, report, self.output)
        config, report = self.new_experiment()
        report.pop("fingerprint")
        initial = report["shared_initialization"]
        initial.pop("fingerprint")
        initial["standard_aggregator_state_sha256"] = "different initialization"
        report["shared_initialization"] = compare._sealed(initial)
        with self.assertRaisesRegex(ValueError, "initial weights differ"):
            compare.build_module_comparison(config, compare._sealed(report), self.output)


if __name__ == "__main__":
    unittest.main()
