"""CPU checks for source-preserving synthetic replacement during training."""

import csv
import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from workflow.training_data import MixedGSVCitiesDataset, PlaceImages


class SourceReplacementTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.gsv = self.root / "gsv"
        (self.gsv / "Dataframes").mkdir(parents=True)
        images = self.gsv / "Images" / "City"
        images.mkdir(parents=True)
        self.sources = {}
        self.metadata = []
        for place, count in ((1, 6), (2, 4), (3, 3)):
            for view in range(count):
                row = {"place_id": str(place), "year": "2020", "month": "1", "northdeg": str(view),
                       "city_id": "City", "lat": "1.0", "lon": "2.0", "panoid": f"p{place}v{view}"}
                path = images / f"City_{place:07d}_2020_01_{view:03d}_1.0_2.0_p{place}v{view}.jpg"
                Image.new("RGB", (13, 11), (20 + view * 20, 50 + place * 20, 100)).save(path)
                self.sources[(place, view)] = path
                self.metadata.append(row)
        with (self.gsv / "Dataframes" / "City.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, self.metadata[0].keys())
            writer.writeheader()
            writer.writerows(self.metadata)
        self.manifest = self.root / "selected.jsonl"
        self.source_of = {path: path for path in self.sources.values()}
        self.domain_of = {}
        self.entries = []

    def candidate(self, source_key=(1, 0), condition="rain", *, name=None, **flags):
        source = self.sources[source_key]
        output = self.root / (name or f"synthetic{len(self.entries)}.jpg")
        Image.new("RGB", (13, 11), (200, 20 + len(self.entries) % 100, 90)).save(output)
        row = {"passed": True, "eligible_for_training": True, "source_path": str(source),
               "output_path": str(output), **flags}
        if condition is not None:
            row["condition"] = condition
        self.entries.append(row)
        self.source_of[output] = source
        self.domain_of[output] = condition if condition else str(output)
        return row

    def dataset(self, *, mode="replace", fraction=0.5, **kwargs):
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in self.entries), encoding="utf-8")
        return MixedGSVCitiesDataset(
            self.gsv, self.manifest, synthetic_mode=mode, synthetic_fraction=fraction,
            images_per_place=4, min_images_per_place=4, image_size=(7, 9), augment=False, **kwargs,
        )

    def sample(self, dataset, index=0, *, seed=None):
        if seed is not None:
            random.seed(seed)
        paths = []
        actual_open = Image.open

        def open_image(path, *args, **kwargs):
            paths.append(Path(path))
            return actual_open(path, *args, **kwargs)

        with patch("PIL.Image.open", side_effect=open_image):
            tensors, labels, kinds = dataset[index]
        self.assertEqual(len(paths), 4)
        self.assertEqual(tuple(tensors.shape), (4, 3, 7, 9))
        self.assertEqual(tensors.dtype, torch.float32)
        self.assertEqual(labels.dtype, torch.long)
        self.assertEqual(kinds.dtype, torch.bool)
        self.assertEqual(labels.unique().tolist(), [dataset.places[index].label])
        for path, synthetic in zip(paths, kinds.tolist()):
            self.assertIs(synthetic, path != self.source_of[path])
        return paths, tensors, labels, kinds

    def test_mapping_preserves_exact_source_and_domain_and_existing_dataclass_construction(self):
        rain = self.candidate((1, 0), "rain")
        fog = self.candidate((1, 0), "fog")
        other_view = self.candidate((1, 1), "rain")
        unlabelled = self.candidate((1, 0), None)
        dataset = self.dataset()
        place = dataset.places[0]
        domains = place.synthetic_by_source[self.sources[(1, 0)]]
        self.assertEqual(domains["rain"], [Path(rain["output_path"])])
        self.assertEqual(domains["fog"], [Path(fog["output_path"])])
        self.assertEqual(domains[unlabelled["output_path"]], [Path(unlabelled["output_path"])])
        self.assertEqual(place.synthetic_by_source[self.sources[(1, 1)]]["rain"],
                         [Path(other_view["output_path"])])
        legacy = PlaceImages("City", 1, 8, [self.sources[(1, 0)]], [Path(rain["output_path"])])
        self.assertEqual(legacy.label, 8)
        self.assertEqual(legacy.synthetic_by_source, {})

    def test_replacement_uses_original_distinct_source_draw_and_never_pairs_real_with_own_variant(self):
        for view in range(6):
            self.candidate((1, view), "rain")
            self.candidate((1, view), "fog")
        dataset = self.dataset(fraction=1)
        originals_before = {path: path.read_bytes() for path in self.sources.values()}
        for seed in range(35):
            random.seed(seed)
            expected_sources = set(random.sample(dataset.places[0].real_paths, 4))
            paths, _, _, kinds = self.sample(dataset, seed=seed)
            actual_sources = [self.source_of[path] for path in paths]
            self.assertEqual(set(actual_sources), expected_sources)
            self.assertEqual(len(set(actual_sources)), 4)
            self.assertEqual(int(kinds.sum()), 3)
            self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual({path: path.read_bytes() for path in self.sources.values()}, originals_before)

    def test_missing_variant_keeps_selected_source_and_omits_unselected_source_variants(self):
        self.candidate((1, 0), "rain")
        self.candidate((1, 0), "fog")
        dataset = self.dataset(fraction=1)
        selected_source = self.sources[(1, 0)]
        seen_present, seen_absent = False, False
        for seed in range(40):
            random.seed(seed)
            expected = set(random.sample(dataset.places[0].real_paths, 4))
            paths, _, _, kinds = self.sample(dataset, seed=seed)
            self.assertEqual({self.source_of[path] for path in paths}, expected)
            if selected_source in expected:
                self.assertEqual(int(kinds.sum()), 1)
                self.assertNotIn(selected_source, paths)
                seen_present = True
            else:
                self.assertEqual(int(kinds.sum()), 0)
                seen_absent = True
            for path in paths:
                if self.source_of[path] != selected_source:
                    self.assertEqual(path, self.source_of[path])
        self.assertTrue(seen_present and seen_absent)
        self.assertEqual(int(self.sample(dataset, index=1, seed=3)[3].sum()), 0)

    def test_probability_is_independent_per_source_and_does_not_round_slots(self):
        for view in range(6):
            self.candidate((1, view), "rain")
        dataset = self.dataset(fraction=0.25)
        # Both zero and two replacements are possible despite floor(4*.25)=1.
        with patch("workflow.training_data.random.random", side_effect=[0.9, 0.8, 0.7, 0.6]) as draws:
            paths, _, _, kinds = self.sample(dataset, seed=17)
        self.assertEqual(int(kinds.sum()), 0)
        self.assertEqual(draws.call_count, 4)
        with patch("workflow.training_data.random.random", side_effect=[0.1, 0.8, 0.9, 0.2]) as draws:
            paths, _, _, kinds = self.sample(dataset, seed=17)
        self.assertEqual(int(kinds.sum()), 2)
        self.assertEqual(draws.call_count, 4)
        self.assertEqual(len({self.source_of[path] for path in paths}), 4)

    def test_all_replaced_draw_restores_one_uniformly_chosen_source(self):
        for view in range(6):
            self.candidate((1, view), "rain")
        dataset = self.dataset(fraction=1)
        random.seed(11)
        sources = random.sample(dataset.places[0].real_paths, 4)
        with patch("workflow.training_data.random.randrange", return_value=2) as retain:
            paths, _, _, kinds = self.sample(dataset, seed=11)
        retain.assert_called_once_with(4)
        self.assertEqual(int(kinds.sum()), 3)
        self.assertEqual([path for path, synthetic in zip(paths, kinds.tolist()) if not synthetic], [sources[2]])

    def test_domain_sampling_is_uniform_despite_unequal_variant_counts(self):
        for variant in range(9):
            self.candidate((1, 0), "rain")
        self.candidate((1, 0), "fog")
        dataset = self.dataset(fraction=1)
        counts = {"rain": 0, "fog": 0}
        random.seed(73)
        for _ in range(500):
            paths, _, _, kinds = self.sample(dataset)
            for path, synthetic in zip(paths, kinds.tolist()):
                if synthetic:
                    counts[self.domain_of[path]] += 1
        self.assertGreater(sum(counts.values()), 250)
        self.assertLess(abs(counts["rain"] - counts["fog"]) / sum(counts.values()), 0.15)

    def test_sampling_is_deterministic_with_random_seed_and_zero_probability_is_all_real(self):
        for view in range(6):
            self.candidate((1, view), "rain")
            self.candidate((1, view), "night")
        dataset = self.dataset(fraction=0.65)
        first = self.sample(dataset, seed=91)
        second = self.sample(dataset, seed=91)
        self.assertEqual(first[0], second[0])
        for left, right in zip(first[1:], second[1:]):
            self.assertTrue(torch.equal(left, right))
        no_replacement = self.dataset(fraction=0)
        paths, _, _, kinds = self.sample(no_replacement, seed=91)
        self.assertFalse(bool(kinds.any()))
        self.assertTrue(all(path == self.source_of[path] for path in paths))

    def test_false_or_nonboolean_optional_acceptance_flags_are_ignored(self):
        accepted = self.candidate((1, 0), "rain", plausible=True, weather_ok=True)
        self.candidate((1, 1), "rain", plausible=False)
        self.candidate((1, 2), "rain", weather_ok=False)
        self.candidate((1, 3), "rain", plausible="true")
        self.candidate((1, 4), "rain", weather_ok=1)
        legacy = self.candidate((1, 5), "fog")
        dataset = self.dataset()
        self.assertEqual(dataset.summary["manifest"]["accepted_rows"], 2)
        self.assertEqual(dataset.summary["manifest"]["ignored_rows"], 4)
        self.assertEqual(set(dataset.places[0].synthetic_paths),
                         {Path(accepted["output_path"]), Path(legacy["output_path"])})

    def test_one_generated_image_cannot_claim_different_views_even_with_same_place(self):
        first = self.candidate((1, 0), "rain")
        self.entries.append({**first, "source_path": str(self.sources[(1, 1)])})
        with self.assertRaisesRegex(ValueError, "conflicting exact source"):
            self.dataset()
        self.entries[-1] = {**first, "condition": "fog"}
        with self.assertRaisesRegex(ValueError, "conflicting condition domains"):
            self.dataset()

    def test_real_outputs_and_unknown_sources_remain_strictly_rejected(self):
        first = self.candidate((1, 0), "rain")
        first["output_path"] = str(self.sources[(1, 1)])
        with self.assertRaisesRegex(ValueError, "generated image, not a real"):
            self.dataset()
        first["output_path"] = str(self.root / "synthetic0.jpg")
        unknown = self.root / "unknown.jpg"
        Image.new("RGB", (13, 11)).save(unknown)
        first["source_path"] = str(unknown)
        with self.assertRaisesRegex(ValueError, "not a selected GSV metadata image"):
            self.dataset()

    def test_synthetics_do_not_rescue_place_with_too_few_distinct_real_sources(self):
        for view in range(3):
            for condition in ("rain", "fog", "snow"):
                self.candidate((3, view), condition)
        dataset = self.dataset(fraction=1)
        self.assertEqual([place.place_id for place in dataset.places], [1, 2])
        self.assertEqual(dataset.summary["excluded_places"], 1)
        self.assertEqual(dataset.summary["num_synthetic_images"], 0)

    def test_summary_defines_probability_source_and_domain_semantics(self):
        self.candidate((1, 0), "rain")
        self.candidate((1, 0), "fog")
        self.candidate((1, 1), None)
        dataset = self.dataset(fraction=0.35)
        summary = dataset.summary
        self.assertEqual(summary["synthetic_mode"], "replace")
        self.assertEqual(summary["synthetic_fraction_semantics"], "per_selected_source_replacement_probability")
        self.assertEqual(summary["replacement_probability"], 0.35)
        self.assertEqual(summary["replacement_domain_sampling"], "uniform_domain_then_uniform_variant")
        self.assertEqual(summary["replacement_domain_field"], "condition_or_output_path")
        self.assertEqual(summary["minimum_real_views_per_sample"], 1)
        self.assertEqual(summary["max_synthetic_views_per_sample"], 3)
        self.assertEqual(summary["replacement_real_retention"], "restore_one_uniform_source_if_all_replaced")
        self.assertEqual(summary["num_sources_with_synthetic"], 2)
        self.assertEqual(summary["num_source_domains"], 3)
        json.dumps(summary, allow_nan=False)

    def test_default_mix_keeps_legacy_floor_sampler_and_summary_fields(self):
        for _ in range(5):
            self.candidate((1, 0), "rain")
        explicit_mix = self.dataset(mode="mix", fraction=0.5)
        default_mix = MixedGSVCitiesDataset(
            self.gsv, self.manifest, synthetic_fraction=0.5,
            images_per_place=4, min_images_per_place=4, image_size=(7, 9), augment=False,
        )
        self.assertEqual(default_mix.synthetic_mode, "mix")
        first = self.sample(explicit_mix, seed=7)
        second = self.sample(default_mix, seed=7)
        self.assertEqual(first[0], second[0])
        self.assertEqual(int(first[3].sum()), 2)
        self.assertEqual(default_mix.summary, explicit_mix.summary)
        self.assertEqual(set(default_mix.summary), {
            "real_data", "synthetic_manifest", "cities", "num_places", "num_real_images",
            "num_synthetic_images", "places_with_synthetic", "excluded_places", "duplicate_real_rows",
            "manifest", "images_per_place", "min_images_per_place", "synthetic_fraction", "image_size",
        })

    def test_invalid_replacement_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "synthetic_mode"):
            self.dataset(mode="overwrite")


class ReplacementTrainingCLITests(unittest.TestCase):
    """Exercise persisted mode and legacy resume using the existing tiny fixture."""

    def setUp(self):
        import test_pretrained_initialization as fixtures

        fixtures.PretrainedInitializationTests.setUp(self)

    def arguments(self, output, *extra):
        import test_pretrained_initialization as fixtures

        return fixtures.PretrainedInitializationTests.arguments(self, output, *extra)

    def train_fixture(self, args):
        from train_salad import train

        with patch("workflow.metric_loss.multi_similarity_loss",
                   side_effect=lambda descriptors, labels, **kwargs: descriptors.sum() * 0):
            with contextlib.redirect_stdout(io.StringIO()):
                train(args)

    def test_replace_mode_is_saved_and_resumes_with_original_real_pool(self):
        from workflow.model import read_checkpoint

        initial = self.root / "pretrained.pt"
        torch.save({"state_dict": self.state, "model_config": self.config}, initial)
        output = self.root / "replace_training"
        args = self.arguments(output, "--init-checkpoint", str(initial),
                              "--synthetic-mode", "replace", "--synthetic-fraction", "0.5")
        self.train_fixture(args)
        final = read_checkpoint(output / "checkpoint.pt")
        self.assertEqual(final["training_config"]["synthetic_mode"], "replace")
        self.assertEqual(final["dataset_summary"]["num_places"], 3)
        self.assertEqual(final["dataset_summary"]["num_real_images"], 6)
        self.assertEqual(final["dataset_summary"]["replacement_probability"], 0.5)
        self.assertGreaterEqual(final["metrics"]["real_exposure"], 2)
        run_config = json.loads((output / "training_config.json").read_text())
        self.assertEqual(run_config["synthetic_mode"], "replace")
        resumed = self.root / "replace_resumed"
        resume_args = self.arguments(resumed, "--resume", str(output / "checkpoint_epoch_001.pt"),
                                     "--synthetic-mode", "replace", "--synthetic-fraction", "0.5")
        self.train_fixture(resume_args)
        resumed_checkpoint = read_checkpoint(resumed / "checkpoint.pt")
        self.assertEqual(resumed_checkpoint["training_config"], final["training_config"])
        for name, value in final["state_dict"].items():
            torch.testing.assert_close(resumed_checkpoint["state_dict"][name], value, rtol=0, atol=0)

    def test_legacy_checkpoint_missing_mode_resumes_as_default_mix(self):
        from workflow.model import read_checkpoint

        initial = self.root / "pretrained.pt"
        torch.save({"state_dict": self.state, "model_config": self.config}, initial)
        output = self.root / "mixed_training"
        self.train_fixture(self.arguments(output, "--init-checkpoint", str(initial)))
        checkpoint = read_checkpoint(output / "checkpoint_epoch_001.pt")
        checkpoint["training_config"].pop("synthetic_mode")
        self.assertNotIn("synthetic_mode", checkpoint["dataset_summary"])
        legacy = self.root / "legacy_checkpoint.pt"
        torch.save(checkpoint, legacy)
        resumed = self.root / "legacy_resumed"
        self.train_fixture(self.arguments(resumed, "--resume", str(legacy)))
        restored = read_checkpoint(resumed / "checkpoint.pt")
        self.assertEqual(restored["epoch"], 2)
        self.assertEqual(restored["training_config"]["synthetic_mode"], "mix")
        self.assertEqual(restored["dataset_summary"], checkpoint["dataset_summary"])


if __name__ == "__main__":
    unittest.main()
