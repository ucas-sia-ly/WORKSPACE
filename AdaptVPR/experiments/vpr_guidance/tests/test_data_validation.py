from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from AdaptVPR.experiments.vpr_guidance.data import (
    GSVLabelIndex, canonical_place_id, gsv_image_name, image_index,
    read_global_prompts, require_empty_output, validate_source_label,
)


class SourceIdentityTests(unittest.TestCase):
    def fixture(self, root):
        city = "City_With_Underscores"
        row = dict(city_id=city, place_id="0000123", year="2017", month="01",
                   northdeg="090", lat="13.0", lon="100.0", panoid="pano_with_parts")
        dataframes = root / "Dataframes"
        dataframes.mkdir()
        with (dataframes / f"{city}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        images = root / "Images" / city
        images.mkdir(parents=True)
        source = images / gsv_image_name(row)
        Image.new("RGB", (8, 8)).save(source)
        return GSVLabelIndex(dataframes), source, city

    def test_labels_come_from_dataframe_full_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            index, source, city = self.fixture(Path(tmp))
            self.assertEqual(index.key_for_source(source), (city, "0000123"))
            self.assertEqual(validate_source_label({"city": city, "place_id": 123}, source, index),
                             (city, "0000123"))
            with self.assertRaisesRegex(ValueError, "place_id disagrees"):
                validate_source_label({"place_id": "0000124"}, source, index)
            with self.assertRaisesRegex(ValueError, "city disagrees"):
                validate_source_label({"city": "OtherCity"}, source, index)

    def test_source_filename_with_unlisted_panorama_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            index, source, _ = self.fixture(Path(tmp))
            unknown = source.with_name(source.name.replace("pano_with_parts", "unlisted_pano"))
            Image.new("RGB", (8, 8)).save(unknown)
            with self.assertRaisesRegex(ValueError, "dataframe row"):
                index.key_for_source(unknown)

    def test_integer_and_leading_zero_labels_are_canonical(self):
        self.assertEqual(canonical_place_id(123), "0000123")
        self.assertEqual(canonical_place_id("0000000123"), "0000123")
        for bad in (True, "123.0", "-1", "1e2", ""):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                canonical_place_id(bad)

    def test_ids_cannot_escape_cache_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.jsonl"
            for sample_id in ("../other", "/tmp/escape", "folder/id", ".."):
                row = dict(sample_id=sample_id, source_id="source.jpg", route="global",
                           condition="snow", prompt="snow")
                path.write_text(json.dumps(row))
                with self.subTest(sample_id=sample_id), self.assertRaisesRegex(ValueError, "unsafe"):
                    read_global_prompts(path)

    def test_duplicate_image_names_fail_and_uppercase_suffix_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (8, 8)).save(root / "source.JPG")
            self.assertIn("source.JPG", image_index(root))
            (root / "other").mkdir()
            Image.new("RGB", (8, 8)).save(root / "other" / "source.JPG")
            with self.assertRaisesRegex(ValueError, "duplicate image basename"):
                image_index(root)

    def test_output_reuse_is_explicitly_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "new"
            require_empty_output(out)
            out.mkdir()
            require_empty_output(out)
            (out / "old.jsonl").write_text("old")
            with self.assertRaises(FileExistsError):
                require_empty_output(out)


if __name__ == "__main__":
    unittest.main()
