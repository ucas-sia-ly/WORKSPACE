"""CPU checks for RandAugment, per-anchor auxiliary scaling, stratified batches and Stage 5."""

import json
from pathlib import Path
import random
import tempfile
import unittest

import torch
from PIL import Image

import build_weather_validation
import run_weather_round as stage5
import test_source_replacement as fixtures
from workflow.augmentation import randaugment
from workflow.synthetic_losses import asymmetric_multi_positive_infonce
from workflow.training_data import StratifiedCoverageBatchSampler


def pattern(seed):
    generator = random.Random(seed)
    image = Image.new("RGB", (32, 24))
    image.putdata([tuple(generator.randrange(256) for _ in range(3)) for _ in range(32 * 24)])
    return image


class AugmentationTests(unittest.TestCase):
    def test_keyed_draw_is_reproducible_shared_and_leaves_global_rng(self):
        image = pattern(1)
        torch.manual_seed(5)
        expected_next = torch.rand(1)
        torch.manual_seed(5)
        first, paired = randaugment([image, image.copy()], 123)
        self.assertTrue(torch.equal(torch.rand(1), expected_next))
        again, = randaugment([image], 123)
        self.assertEqual(first.tobytes(), paired.tobytes())
        self.assertEqual(first.tobytes(), again.tobytes())
        others = {randaugment([image], key)[0].tobytes() for key in range(8)}
        self.assertGreater(len(others), 1)


class AuxiliaryScalingTests(unittest.TestCase):
    def test_denominator_sums_queries_over_main_batch(self):
        torch.manual_seed(0)
        references, labels = torch.randn(8, 5), torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        queries, query_labels = torch.randn(2, 5), torch.tensor([1, 3])
        mean = asymmetric_multi_positive_infonce(queries, query_labels, references, labels)
        scaled = asymmetric_multi_positive_infonce(queries, query_labels, references, labels, denominator=8)
        self.assertAlmostEqual(float(scaled), float(mean) * 2 / 8, places=6)


class StratifiedSamplerTests(unittest.TestCase):
    def test_slots_uniqueness_coverage_and_determinism(self):
        eligible = [i % 5 == 0 for i in range(53)]
        sampler = StratifiedCoverageBatchSampler(eligible, batch_size=6, slots=2, seed=9)
        others = [i for i, flag in enumerate(eligible) if not flag]
        for epoch in range(3):
            sampler.set_epoch(epoch)
            batches = list(sampler)
            self.assertEqual(len(batches), len(sampler))
            seen = []
            for batch in batches:
                places = [index for _, index in batch]
                self.assertEqual(len(places), len(set(places)))
                self.assertEqual(sum(eligible[i] for i in places), 2)
                self.assertTrue(all(e == epoch for e, _ in batch))
                seen += [i for i in places if not eligible[i]]
            self.assertEqual(sorted(seen), others)
            self.assertEqual(batches, list(sampler))

    def test_rejects_impossible_slot_counts(self):
        with self.assertRaises(ValueError):
            StratifiedCoverageBatchSampler([True, False], batch_size=4, slots=2, seed=0)
        with self.assertRaises(ValueError):
            StratifiedCoverageBatchSampler([True, True, False], batch_size=2, slots=2, seed=0)


class SharedMainBatchTests(unittest.TestCase):
    setUp = fixtures.SourceReplacementTests.setUp
    candidate = fixtures.SourceReplacementTests.candidate

    def test_real_views_get_identical_tensors_in_a_and_m(self):
        self.candidate((1, 4), "night")
        make = lambda mode, fraction: fixtures.SourceReplacementTests.dataset(
            self, mode=mode, fraction=fraction, real_sampling="coverage_pk", prefer_synthetic_sources=True)
        control, replaced = make("mix", 0.0), make("replace", 1.0)
        control.augment = replaced.augment = True
        for epoch in range(3):
            def by_path(dataset):
                dataset.set_epoch(epoch)
                images, _, kinds = dataset[0]
                paths = [p for p, _ in dataset.sample_paths(0, epoch=epoch)]
                return {p: images[i] for i, p in enumerate(paths) if not kinds[i]}
            real, mixed = by_path(control), by_path(replaced)
            self.assertEqual(len(mixed), len(real) - 1)
            for path, tensor in mixed.items():
                self.assertTrue(torch.equal(tensor, real[path]))


class PreferredSourceTests(unittest.TestCase):
    setUp = fixtures.SourceReplacementTests.setUp
    candidate = fixtures.SourceReplacementTests.candidate

    def dataset(self, fraction):
        return fixtures.SourceReplacementTests.dataset(
            self, mode="replace", fraction=fraction, real_sampling="coverage_pk", prefer_synthetic_sources=True)

    def test_variant_sources_are_selected_and_real_stream_matches_control(self):
        self.candidate((1, 4), "night")
        self.candidate((1, 5), "rain")
        control, replaced = self.dataset(0.0), self.dataset(1.0)
        wanted = {self.sources[1, 4], self.sources[1, 5]}
        for epoch in range(5):
            real = [path for path, _ in control.sample_paths(0, epoch=epoch)]
            self.assertTrue(wanted <= set(real))
            mixed = replaced.sample_paths(0, epoch=epoch)
            self.assertEqual(sum(flag for _, flag in mixed), 2)
            restored = {path if not flag else replaced.places[0].source_by_synthetic_path[path] for path, flag in mixed}
            self.assertEqual(restored, set(real))

    def test_requires_coverage_sampling(self):
        with self.assertRaisesRegex(ValueError, "coverage_pk"):
            fixtures.SourceReplacementTests.dataset(self, prefer_synthetic_sources=True)


class WeatherValidationTests(unittest.TestCase):
    def test_buffer_drops_queries_near_held_out_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "svox"
            def put(folder, x, y):
                path = root / "images" / folder / f"@{x:.2f}@{y:.2f}@x@.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (4, 4)).save(path)
            for x in (0, 100, 200):
                put("train/gallery", x, 0)
            put("train/queries_night", 1, 0)       # 199 m from test: kept
            put("train/queries_night", 101, 0)     # 10 m from val: dropped
            put("train/queries_rain", 199, 0)      # 1 m from test: dropped
            for domain in ("snow", "sun", "overcast"):
                put(f"train/queries_{domain}", 2, 0)
            put("test/gallery", 200, 0)
            put("test/queries", 200, 5)
            put("val/gallery", 110, 0)
            put("val/queries", 110, 1)
            protocol = build_weather_validation.build(root, Path(temporary) / "out", 50.0)
            self.assertEqual(protocol["queries_per_domain"]["night"], 1)
            self.assertEqual(protocol["queries_per_domain"]["rain"], 0)
            self.assertEqual(protocol["dropped_per_domain"], dict(night=1, rain=1, snow=0, sun=0, overcast=0))
            manifest = json.loads(Path(protocol["manifest"]).read_text())
            night = next(q for q in manifest["queries"] if "queries_night" in q["id"])
            self.assertEqual(night["positives"], ["images/train/gallery/@0.00@0.00@x@.jpg"])
            with self.assertRaises(ValueError):
                build_weather_validation.build(root, Path(temporary) / "out", 60.0)


class Stage5Tests(unittest.TestCase):
    def config(self):
        return stage5.load_config(stage5.ROOT / "configs/weather_round.json")

    def test_shipped_config_and_method_commands(self):
        c = self.config()
        plan = dict(config=c)
        commands = {m: stage5.command(plan, m, 43, Path("/tmp/run"), 1, False) for m in c["methods"]}
        for cmd in commands.values():
            self.assertIn("--prefer-synthetic-sources", cmd)
            self.assertEqual(cmd[cmd.index("--synthetic-place-slots") + 1], "8")
            self.assertEqual(cmd[cmd.index("--init-checkpoint") + 1], c["checkpoint"])
        flag = lambda m, name: commands[m][commands[m].index(name) + 1]
        self.assertEqual((flag("A", "--synthetic-mode"), flag("A", "--synthetic-fraction")), ("mix", "0.0"))
        self.assertEqual((flag("M", "--synthetic-mode"), flag("M", "--synthetic-fraction")), ("replace", "1.0"))
        self.assertIn("--synthetic-aux", commands["B0"])
        self.assertEqual(flag("B0", "--synthetic-mode"), "mix")
        self.assertNotIn("--synthetic-aux", commands["A"])
        self.assertEqual(flag("A_retention", "--teacher-checkpoint"), c["checkpoint"])
        self.assertNotIn("--teacher-checkpoint", commands["M"])
        resumed = stage5.command(plan, "A", 43, Path("/tmp/run"), 2, True)
        self.assertEqual(resumed[resumed.index("--resume") + 1], "/tmp/run/checkpoint.pt")

    def test_trainer_accepts_every_method_command(self):
        import train_salad
        c = self.config()
        for method in c["methods"]:
            cmd = stage5.command(dict(config=c), method, 43, Path("/tmp/run"), 1, False)
            args = train_salad.parse_args(cmd[2:])
            self.assertEqual(args.synthetic_place_slots, 8)

    def test_groups_split_weather_domains_and_recompute_recall(self):
        result = {"query_results": [
            {"query_id": "images/train/queries_night/a.jpg", "rank": 1},
            {"query_id": "images/train/queries_night/b.jpg", "rank": 7},
            {"query_id": "images/train/queries_sun/c.jpg", "rank": 2}],
            "recall": {"R@1": 1 / 3, "R@5": 2 / 3, "R@10": 1.0}}
        groups = stage5.split_groups("svox_train_weather", result)
        self.assertEqual(set(groups), {"night", "sun"})
        self.assertEqual(groups["night"]["recall"], {"R@1": .5, "R@5": .5, "R@10": 1.0})
        self.assertEqual(set(stage5.split_groups("svox_val", result)), {"all"})
        tokyo = stage5.split_groups("tokyo247_reduced", {**result, "query_results": [
            {"query_id": "N/1.jpg", "rank": 1}, {"query_id": "D/2.jpg", "rank": 1}, {"query_id": "D/3.jpg", "rank": 1}],
            "recall": {"R@1": 1.0, "R@5": 1.0, "R@10": 1.0}})
        self.assertEqual(set(tokyo), {"N", "D"})


if __name__ == "__main__":
    unittest.main()


class Stage5ReportTests(unittest.TestCase):
    def test_summarize_pairs_seeds_against_initial_and_reference(self):
        import run_ccr_feasibility as ccr
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            initial = out / "initial.ckpt"
            initial.write_bytes(b"initial")
            evaluations = [dict(name="svox_val"), dict(name="svox_train_weather")]
            config = dict(output_dir=str(out), checkpoint=str(initial), reference_method="A", seeds=[43],
                          methods={"A": {}, "M": {}}, training=dict(epochs=1))
            plan = dict(config=config, evaluations=evaluations, snapshot=dict(accepted=1, places=1, conditions={}))
            queries = {"svox_val": [f"images/val/queries/{i}.jpg" for i in range(4)],
                       "svox_train_weather": [f"images/train/queries_night/{i}.jpg" for i in range(4)]}
            def write(directory, checkpoint, correct):
                directory.mkdir(parents=True, exist_ok=True)
                for evaluation in evaluations:
                    ranks = [1 if i < correct[evaluation["name"]] else 20 for i in range(4)]
                    result = dict(query_results=[dict(query_id=q, rank=r) for q, r in zip(queries[evaluation["name"]], ranks)],
                                  recall={f"R@{k}": sum(r <= k for r in ranks) / 4 for k in (1, 5, 10)})
                    path = directory / (evaluation["name"] + ".json")
                    ccr.write_json(path, result)
                    ccr.write_json(directory / (evaluation["name"] + ".receipt.json"), dict(
                        checkpoint_sha256=ccr.digest(checkpoint), protocol=evaluation, result_sha256=ccr.digest(path)))
            write(out / "initial", initial, dict(svox_val=3, svox_train_weather=2))
            for method, night in (("A", 1), ("M", 3)):
                run = out / "runs/43" / method
                run.mkdir(parents=True)
                checkpoint = run / "checkpoint_epoch_001.pt"
                checkpoint.write_bytes(method.encode())
                (run / "training_log.jsonl").write_text(json.dumps(dict(
                    epoch=1, loss=.2, synthetic_fraction_observed=.1 if method == "M" else 0.,
                    sampling=dict(synthetic_weather_exposures={"night": 3} if method == "M" else {}),
                    mining=dict(active_anchor_fraction=.2, clip_batch_fraction=0.))) + "\n")
                write(run / "validation/1", checkpoint, dict(svox_val=3, svox_train_weather=night))
            report = stage5.summarize(plan)
            self.assertEqual(report["status"], "complete")
            pick = lambda method, reference, evaluation: next(
                r for r in report["paired_statistics"] if (r["method"], r["reference"], r["evaluation"], r["k"])
                == (method, reference, evaluation, 1))
            self.assertEqual(pick("M", "A", "svox_train_weather")["delta_pp"], 50.0)
            self.assertEqual(pick("A", "initial", "svox_train_weather")["delta_pp"], -25.0)
            self.assertEqual(pick("M", "initial", "svox_val")["delta_pp"], 0.0)
            self.assertIn("p_holm", pick("M", "A", "svox_train_weather"))
            text = (out / "report.md").read_text()
            self.assertIn("svox_train_weather/night", text)
            self.assertIn("+50.00 (1/1)", text)
