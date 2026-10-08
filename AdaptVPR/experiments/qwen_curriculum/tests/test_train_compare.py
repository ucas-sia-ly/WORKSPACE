"""CPU-only frozen comparison, completed-generation and service defenses."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

ADAPTVPR_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ADAPTVPR_ROOT))
from experiments.qwen_curriculum import common, train_compare as compare


def sealed(row, field):
    return {**row, field: common.fingerprint(row)}


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.run_dir = self.root / "generation"
        self.run_dir.mkdir()
        self.real = self.root / "real"
        (self.real / "Dataframes").mkdir(parents=True)
        self.sources = []
        for city_number, city in enumerate(compare.CITIES):
            folder = self.real / "Images" / city
            folder.mkdir(parents=True)
            rows = []
            for view in range(4):
                row = {"place_id": "0", "year": "2020", "month": "1", "northdeg": str(view),
                       "city_id": city, "lat": "1.0", "lon": "2.0", "panoid": f"p{view}"}
                rows.append(row)
                path = folder / f"{city}_0000000_2020_01_{view:03d}_1.0_2.0_p{view}.jpg"
                Image.new("RGB", (12, 8), (city_number * 50, view * 50, 100)).save(path)
                if view == 0:
                    self.sources.append(path)
            with (self.real / "Dataframes" / f"{city}.csv").open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        self.checkpoint = self.root / "initial.pt"
        self.checkpoint.write_bytes(b"fake checkpoint; prepare must never deserialize this")
        self.backbone = self.root / "backbone"
        self.backbone.mkdir()
        (self.backbone / "hubconf.py").write_text("# local backbone")
        self.jobs = [sealed({"sample_id": f"qwen_{index:024x}", "source_path": str(source),
                            "source_sha256": common.file_sha256(source), "source_dimensions": [12, 8],
                            "city": compare.CITIES[index], "place_id": 0, "condition": "night",
                            "prompt": "released prompt", "negative_prompt": "", "seed": 42}, "record_sha256")
                     for index, source in enumerate(self.sources)]
        common.write_jsonl(self.run_dir / "plan.jsonl", self.jobs)
        self.plan = sealed({"num_images": len(self.jobs),
                            "plan_sha256": common.file_sha256(self.run_dir / "plan.jsonl")}, "fingerprint")
        common.write_json(self.run_dir / "plan_config.json", self.plan)
        self.execution = sealed({"stage": "generate", "plan_fingerprint": self.plan["fingerprint"],
                                 "max_calls": len(self.jobs)}, "fingerprint")
        common.write_json(self.run_dir / "execution_config.json", self.execution)
        self.results, self.attempts = [], []
        for index, job in enumerate(self.jobs):
            output = self.run_dir / f"output{index}.png"
            raw = self.run_dir / f"raw{index}.png"
            Image.new("RGB", (12, 8), (1, 2, index)).save(output)
            Image.new("RGB", (24, 16), (1, 2, index)).save(raw)
            self.results.append(sealed({**job, "execution_fingerprint": self.execution["fingerprint"],
                "output_path": str(output), "output_sha256": common.file_sha256(output),
                "raw_output_path": str(raw), "raw_output_sha256": common.file_sha256(raw),
                "status": "accepted" if index < 2 else "rejected", "passed": index < 2,
                "eligible_for_training": index < 2, "geometry_ok": True, "weather_ok": index < 2},
                "result_sha256"))
            self.attempts.append(sealed({"call_number": index + 1, "sample_id": job["sample_id"],
                "record_sha256": job["record_sha256"], "execution_fingerprint": self.execution["fingerprint"]},
                "attempt_sha256"))
        self.summary = {"state": "complete", "planned": 4, "completed": 4, "accepted": 2,
                        "http_calls_reserved": 4}
        self.publish()

    def publish(self):
        common.write_json(self.run_dir / "summary.json", self.summary)
        common.write_jsonl(self.run_dir / "results.jsonl", self.results)
        common.write_jsonl(self.run_dir / "attempts.jsonl", self.attempts)
        common.write_jsonl(self.run_dir / "training_manifest.jsonl", self.results[:2])

    def args(self, *extra):
        return compare.parse_args(["prepare", "--generation-run-dir", str(self.run_dir),
            "--output-dir", str(self.root / "comparison"), "--real-data", str(self.real),
            "--checkpoint", str(self.checkpoint), "--backbone-repo", str(self.backbone), *extra])

    def test_prepare_model_free_frozen_and_same_broad_real_pool(self):
        args = self.args()
        with patch.dict(sys.modules, {"torch": None, "diffusers": None, "transformers": None}):
            config = compare.prepare(args)
            before = (args.output_dir / "accepted_manifest.jsonl").read_bytes()
            self.assertEqual(compare.prepare(args), config)
            self.assertEqual((args.output_dir / "accepted_manifest.jsonl").read_bytes(), before)
        self.assertEqual(config["accepted_images"], 2)
        self.assertEqual(config["dataset_summaries"]["real"]["num_places"], 4)
        self.assertEqual(config["dataset_summaries"]["replace"]["num_places"], 4)
        self.assertEqual(config["dataset_summaries"]["real"]["num_real_images"], 16)
        commands = config["training_commands"]
        for arm, command in commands.items():
            self.assertNotIn("--synthetic-places-only", command)
            for flag, value in {"--epochs": "4", "--num-trainable-blocks": "0", "--learning-rate": "1e-6",
                                "--precision": "32", "--num-workers": "0", "--save-every": "4"}.items():
                self.assertEqual(command[command.index(flag) + 1], value)
            self.assertEqual(command[command.index("--init-checkpoint") + 1], str(self.checkpoint))
        self.assertNotIn("--synthetic-manifest", commands["real"])
        self.assertIn("--synthetic-mode", commands["replace"])
        self.assertEqual(commands["replace"][commands["replace"].index("--synthetic-mode") + 1], "replace")
        self.assertEqual(commands["replace"][commands["replace"].index("--synthetic-fraction") + 1], "0.5")

    def test_incomplete_or_partial_generation_refuses_before_writing(self):
        for state in ("running", "call_budget_exhausted", "stopped_on_generation_error"):
            self.summary["state"] = state
            self.publish()
            with self.assertRaisesRegex(ValueError, "not complete"):
                compare.prepare(self.args())
            self.assertFalse(self.args().output_dir.exists())
        self.summary["state"], self.summary["completed"] = "complete", 3
        self.publish()
        with self.assertRaisesRegex(ValueError, "cover every"):
            compare.prepare(self.args())

    def test_changed_source_output_raw_or_record_seal_refuses(self):
        for field in ("source_path", "output_path", "raw_output_path"):
            path = Path(self.results[0][field])
            old = path.read_bytes()
            path.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                compare.prepare(self.args())
            path.write_bytes(old)
        self.results[0]["prompt"] = "changed"
        self.publish()
        with self.assertRaisesRegex(ValueError, "checksum"):
            compare.prepare(self.args())

    def test_accepted_subset_and_quality_flags_are_defended(self):
        common.write_jsonl(self.run_dir / "training_manifest.jsonl", self.results[:1])
        with self.assertRaisesRegex(ValueError, "accepted sealed results"):
            compare.prepare(self.args())
        self.results[0].pop("result_sha256")
        self.results[0]["plausible"] = False
        self.results[0] = sealed(self.results[0], "result_sha256")
        self.publish()
        with self.assertRaisesRegex(ValueError, "quality flags"):
            compare.prepare(self.args())

    def test_exact_source_label_and_call_ledger_are_defended(self):
        self.jobs[0].pop("record_sha256")
        self.jobs[0]["place_id"] = 99
        self.jobs[0] = sealed(self.jobs[0], "record_sha256")
        common.write_jsonl(self.run_dir / "plan.jsonl", self.jobs)
        self.plan.pop("fingerprint")
        self.plan["plan_sha256"] = common.file_sha256(self.run_dir / "plan.jsonl")
        self.plan = sealed(self.plan, "fingerprint")
        common.write_json(self.run_dir / "plan_config.json", self.plan)
        self.execution.pop("fingerprint")
        self.execution["plan_fingerprint"] = self.plan["fingerprint"]
        self.execution = sealed(self.execution, "fingerprint")
        common.write_json(self.run_dir / "execution_config.json", self.execution)
        for index, row in enumerate(self.results):
            row.pop("result_sha256")
            row.update(self.jobs[index])
            row["execution_fingerprint"] = self.execution["fingerprint"]
            self.results[index] = sealed(row, "result_sha256")
            self.attempts[index].pop("attempt_sha256")
            self.attempts[index]["record_sha256"] = self.jobs[index]["record_sha256"]
            self.attempts[index]["execution_fingerprint"] = self.execution["fingerprint"]
            self.attempts[index] = sealed(self.attempts[index], "attempt_sha256")
        self.publish()
        with self.assertRaisesRegex(ValueError, "exact GSV metadata"):
            compare.prepare(self.args())
        self.attempts.pop()
        self.publish()
        with self.assertRaisesRegex(ValueError, "call ledger"):
            compare.prepare(self.args())

    def test_protocol_or_snapshot_changes_refuse_frozen_directory(self):
        args = self.args()
        compare.prepare(args)
        for extra in (("--seed", "43"), ("--train-batch-size", "4")):
            with self.assertRaisesRegex(ValueError, "Immutable comparison"):
                compare.prepare(self.args(*extra))
        common.write_jsonl(args.output_dir / "accepted_manifest.jsonl", self.results[:1])
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            compare.prepare(args)

    def test_hash_bound_visual_review_curates_snapshot_without_changing_generation(self):
        review = self.root / "review.jsonl"
        row = self.results[0]
        exclusion = {key: row[key] for key in ("sample_id", "source_sha256", "output_sha256")}
        exclusion["reason"] = "Invented facade in visually reviewed source blur"
        common.write_jsonl(review, [exclusion])
        before = (self.run_dir / "results.jsonl").read_bytes()
        args = self.args("--review-exclusions", str(review))
        config = compare.prepare(args)
        self.assertEqual(config["machine_accepted_images"], 2)
        self.assertEqual(config["accepted_images"], 1)
        self.assertEqual(config["review_exclusions"]["excluded_images"], 1)
        self.assertEqual(config["review_exclusions"]["sha256"], common.file_sha256(review))
        self.assertEqual(common.read_jsonl(args.output_dir / "accepted_manifest.jsonl"), [self.results[1]])
        self.assertEqual((self.run_dir / "results.jsonl").read_bytes(), before)
        self.assertEqual(config["dataset_summaries"]["replace"]["num_synthetic_images"], 1)
        exclusion["reason"] = "Updated human review reason"
        common.write_jsonl(review, [exclusion])
        with self.assertRaisesRegex(ValueError, "Immutable comparison"):
            compare.prepare(args)

    def test_review_unknown_duplicate_and_changed_image_hashes_fail(self):
        review = self.root / "review.jsonl"
        row = self.results[0]
        exclusion = {key: row[key] for key in ("sample_id", "source_sha256", "output_sha256")}
        exclusion["reason"] = "Known visual failure"
        for rows, message in (([{**exclusion, "sample_id": "unknown"}], "unknown"),
                              ([exclusion, exclusion], "Duplicate"),
                              ([{**exclusion, "output_sha256": "0" * 64}], "hash differs"),
                              ([{**exclusion, "source_sha256": "0" * 64}], "hash differs")):
            common.write_jsonl(review, rows)
            with self.assertRaisesRegex(ValueError, message):
                compare.prepare(self.args("--review-exclusions", str(review)))
            self.assertFalse(self.args().output_dir.exists())

    def test_fixture_svox_cannot_claim_native_evaluation(self):
        folder = self.root / "svox/images/test/gallery"
        folder.mkdir(parents=True)
        Image.new("RGB", (12, 8)).save(folder / "fixture.jpg")
        with self.assertRaisesRegex(ValueError, "expects 17166"):
            compare.prepare(self.args("--evaluate-svox", "--dataset-root", str(self.root / "svox")))

    def test_wait_is_lightweight_and_worker_failure_stops(self):
        args = self.args()
        args.generation_service = "qwen-curriculum-1000.service"
        self.summary["state"] = "running"
        self.publish()
        active, inactive = Mock(stdout="active\n"), Mock(stdout="inactive\n")
        def complete(_):
            self.summary["state"] = "complete"
            self.publish()
        with patch.dict(sys.modules, {"torch": None}), patch.object(compare.subprocess, "run", side_effect=[active, inactive]), \
                patch.object(compare.time, "sleep", side_effect=complete) as sleep:
            compare.wait_for_generation(args)
            sleep.assert_called_once_with(60)
        self.summary["state"] = "running"
        self.publish()
        with patch.object(compare.subprocess, "run", return_value=Mock(stdout="failed\n")), \
                patch.object(compare.time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "training will not start"):
                compare.wait_for_generation(args)
            sleep.assert_not_called()

    def test_service_stop_checks_workspace_ownership_and_exact_unit(self):
        with patch.object(compare.subprocess, "run", return_value=Mock(stdout="unrelated adapter")) as execute:
            with self.assertRaisesRegex(RuntimeError, "Refusing to stop"):
                compare.stop_qwen_service()
            self.assertEqual(execute.call_count, 1)
        adapter = str(common.ADAPTVPR_ROOT / "adapters/lightx2v_qwen_image_edit.py")
        with patch.object(compare.subprocess, "run", return_value=Mock(stdout=adapter)) as execute:
            compare.stop_qwen_service()
            self.assertEqual(execute.call_args.args[0], ["systemctl", "--user", "stop", "adaptvpr-lightx2v.service"])

    def test_training_resumes_epoch_boundaries_and_completed_report_skips_models(self):
        args = self.args()
        args.command = "run"
        args.stop_qwen_service = True
        config = compare.prepare(args)
        for arm in ("real", "replace"):
            folder = args.output_dir / arm
            folder.mkdir()
            (folder / "checkpoint.pt").write_bytes(arm.encode())
        commands = []
        def launch(command, *_):
            commands.append(command)
        with patch.object(compare, "_check_final", side_effect=[{"epoch": 2}, {"epoch": 4}, {"epoch": 4}]), \
                patch.object(compare, "_subprocess", side_effect=launch), \
                patch.object(compare, "stop_qwen_service") as stop:
            compare.run(args, config)
            stop.assert_called_once()
        self.assertEqual(len(commands), 1)
        self.assertIn("--resume", commands[0])
        self.assertNotIn("--init-checkpoint", commands[0])
        report = json.loads((args.output_dir / "comparison.json").read_text())
        self.assertEqual(report["state"], "training_complete_evaluation_disabled")
        with patch.object(compare, "_check_final") as check, patch.object(compare, "_subprocess") as launch, \
                patch.object(compare, "stop_qwen_service") as stop:
            compare.run(args, config)
            check.assert_not_called()
            launch.assert_not_called()
            stop.assert_not_called()
        (args.output_dir / "real/checkpoint.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            compare.run(args, config)

    def test_main_waits_then_validates_before_service_stop_or_training(self):
        order = []
        args = self.args()
        args.command, args.wait_for_generation = "run", True
        with patch.object(compare, "parse_args", return_value=args), \
                patch.object(compare, "wait_for_generation", side_effect=lambda _: order.append("wait")), \
                patch.object(compare, "prepare", side_effect=lambda _: order.append("validate") or {"ready": True}), \
                patch.object(compare, "run", side_effect=lambda *_: order.append("train")):
            compare.main([])
        self.assertEqual(order, ["wait", "validate", "train"])
        with patch.object(compare, "parse_args", return_value=args), \
                patch.object(compare, "wait_for_generation"), \
                patch.object(compare, "prepare", side_effect=ValueError("bad source bytes")), \
                patch.object(compare, "run") as run:
            with self.assertRaisesRegex(ValueError, "bad source"):
                compare.main([])
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
