"""Exact-source companions without changing the existing view sampler."""

import random
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image, ImageOps

import test_source_replacement as fixtures


class ReliabilityPairsTests(unittest.TestCase):
    setUp = fixtures.SourceReplacementTests.setUp
    candidate = fixtures.SourceReplacementTests.candidate

    def dataset(self, **kwargs):
        return fixtures.SourceReplacementTests.dataset(self, **kwargs)

    def read_item(self, dataset, seed=71):
        paths = []
        original_open = Image.open

        def capture(path, *args, **kwargs):
            paths.append(Path(path))
            return original_open(path, *args, **kwargs)

        random.seed(seed)
        with patch("PIL.Image.open", side_effect=capture):
            result = dataset[0]
        return result, paths

    def test_exact_sources_in_mix_and_replace_including_multiple_domain_variants(self):
        for view in range(6):
            self.candidate((1, view), "night")
            self.candidate((1, view), "rain")
        for mode in ("mix", "replace"):
            with self.subTest(mode=mode):
                dataset = self.dataset(mode=mode, fraction=1, reliability_pairs=True)
                (images, labels, kinds, companions, valid), paths = self.read_item(dataset)
                self.assertEqual(tuple(companions.shape), (4, 3, 7, 9))
                self.assertEqual(companions.dtype, images.dtype)
                self.assertEqual(valid.dtype, torch.bool)
                self.assertTrue(torch.equal(valid, kinds))
                self.assertEqual(labels.unique().tolist(), [dataset.places[0].label])
                cursor = 0
                for slot, synthetic in enumerate(kinds.tolist()):
                    selected = paths[cursor]
                    cursor += 1
                    if synthetic:
                        source = paths[cursor]
                        cursor += 1
                        self.assertEqual(source, self.source_of[selected])
                        self.assertEqual(dataset.source_index[source], ("City", 1))
                        self.assertEqual(dataset.places[0].source_by_synthetic_path[selected], source)
                        self.assertFalse(torch.equal(images[slot], companions[slot]))
                    else:
                        self.assertTrue(torch.equal(images[slot], companions[slot]))
                self.assertEqual(cursor, len(paths))

    def test_all_real_companions_reuse_selected_tensor_without_extra_decodes(self):
        self.candidate((1, 0), "night")
        dataset = self.dataset(fraction=0, reliability_pairs=True)
        (images, _, kinds, companions, valid), paths = self.read_item(dataset)
        self.assertEqual(len(paths), 4)
        self.assertFalse(bool(kinds.any()))
        self.assertFalse(bool(valid.any()))
        self.assertTrue(torch.equal(images, companions))

    def test_default_schema_summary_pixels_and_rng_are_unchanged_with_pairing_disabled(self):
        for view in range(6):
            self.candidate((1, view), "snow")
        default = self.dataset(mode="mix", fraction=0.5)
        explicit = self.dataset(mode="mix", fraction=0.5, reliability_pairs=False)
        enabled = self.dataset(mode="mix", fraction=0.5, reliability_pairs=True)
        for dataset in (default, explicit, enabled):
            dataset.augment = True
        self.assertEqual(default.summary, explicit.summary)
        self.assertNotIn("reliability_pairs", default.summary)
        random.seed(137)
        first = default[0]
        after_default = random.getstate()
        random.seed(137)
        second = explicit[0]
        after_explicit = random.getstate()
        random.seed(137)
        paired = enabled[0]
        after_paired = random.getstate()
        self.assertEqual(len(first), 3)
        self.assertEqual(len(second), 3)
        self.assertEqual(len(paired), 5)
        for plain, disabled, with_pair in zip(first, second, paired):
            self.assertTrue(torch.equal(plain, disabled))
            self.assertTrue(torch.equal(plain, with_pair))
        self.assertEqual(after_default, after_explicit)
        self.assertEqual(after_default, after_paired)

    def identical_asymmetric_pairs(self):
        # The marker makes a horizontal flip observable. Byte-identical inputs
        # must remain identical after every shared spatial/photometric transform.
        for view in range(6):
            source = self.sources[(1, view)]
            image = Image.new("RGB", (13, 11), (20, 30, 40))
            for x in range(4):
                for y in range(11):
                    image.putpixel((x, y), (220, 70, 30))
            image.save(source)
            row = self.candidate((1, view), "fog")
            Path(row["output_path"]).write_bytes(source.read_bytes())
        dataset = self.dataset(mode="mix", fraction=1, reliability_pairs=True)
        dataset.augment = True
        return dataset

    def test_horizontal_flip_is_shared_and_not_drawn_independently(self):
        dataset = self.identical_asymmetric_pairs()
        original_mirror = ImageOps.mirror
        with patch("workflow.training_data.random.random", side_effect=[0.1, 0.9] * 4) as draws:
            with patch("PIL.ImageOps.mirror", wraps=original_mirror) as flips:
                (images, _, kinds, companions, valid), _ = self.read_item(dataset)
        self.assertEqual(draws.call_count, 8)
        self.assertEqual(flips.call_count, 4 + int(valid.sum()))
        self.assertTrue(torch.equal(kinds, valid))
        self.assertTrue(torch.equal(images, companions))
        self.assertTrue(bool((images[:, 0, :, -1].mean(1) > images[:, 0, :, 0].mean(1)).all()))

    def test_color_jitter_is_shared_without_extra_random_draws(self):
        dataset = self.identical_asymmetric_pairs()
        with patch("workflow.training_data.random.random", side_effect=[0.1, 0.1] * 4) as draws:
            with patch("workflow.training_data.random.uniform", return_value=1.15) as factors:
                (images, _, _, companions, valid), _ = self.read_item(dataset)
        self.assertEqual(draws.call_count, 8)
        self.assertEqual(factors.call_count, 12)
        self.assertEqual(int(valid.sum()), 3)
        self.assertTrue(torch.equal(images, companions))

    def test_dataloader_collates_pair_validity_and_exact_companions(self):
        for view in range(6):
            self.candidate((1, view), "rain")
        dataset = self.dataset(mode="replace", fraction=1, reliability_pairs=True)
        batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=0)))
        self.assertEqual(len(batch), 5)
        images, labels, kinds, companions, valid = batch
        self.assertEqual(tuple(images.shape), (2, 4, 3, 7, 9))
        self.assertEqual(tuple(companions.shape), tuple(images.shape))
        self.assertEqual(tuple(labels.shape), (2, 4))
        self.assertTrue(torch.equal(kinds, valid))
        self.assertTrue(torch.equal(images[~valid], companions[~valid]))

    def test_pairing_summary_is_opt_in_and_documents_shared_transforms(self):
        self.candidate((1, 0), "rain")
        dataset = self.dataset(reliability_pairs=True)
        self.assertTrue(dataset.summary["reliability_pairs"])
        self.assertEqual(dataset.summary["reliability_pair_source"], "exact_manifest_source")
        self.assertEqual(dataset.summary["reliability_pair_augmentation"], "shared_spatial_and_color_jitter")

    def test_pairing_flag_rejects_non_boolean_value(self):
        with self.assertRaisesRegex(ValueError, "reliability_pairs must be a bool"):
            self.dataset(reliability_pairs="false")


if __name__ == "__main__":
    unittest.main()
