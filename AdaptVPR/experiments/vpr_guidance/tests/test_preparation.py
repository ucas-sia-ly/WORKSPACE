"""CPU orchestration tests; generation and teacher are controlled test doubles."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from AdaptVPR.experiments.vpr_guidance import prepare_data
from AdaptVPR.experiments.vpr_guidance.data import gsv_image_name
from AdaptVPR.experiments.vpr_guidance.train_generator import read_manifest


class PreparationTests(unittest.TestCase):
    def test_preparation_keeps_source_baseline_descriptor_and_label_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gsv = root / "gsv"
            (gsv / "Dataframes").mkdir(parents=True)
            images = gsv / "Images" / "Bangkok"
            images.mkdir(parents=True)
            meta, prompts = [], []
            for place, color in ((1, (50, 20, 10)), (2, (180, 80, 60))):
                row = dict(city_id="Bangkok", place_id=place, year=2017, month=1,
                           northdeg=90, lat=13.0, lon=100.0, panoid=f"pano_{place}")
                meta.append(row)
                name = gsv_image_name(row)
                Image.new("RGB", (16, 16), color).save(images / name)
                prompts.append(dict(sample_id=f"sample{place}", source_id=name, prompt=f"snow {place}",
                                    route="global", condition="snow"))
            with (gsv / "Dataframes/Bangkok.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(meta[0]))
                writer.writeheader()
                writer.writerows(meta)
            prompt_path = root / "prompts.jsonl"
            prompt_path.write_text("\n".join(json.dumps(row) for row in prompts))
            out = root / "output"

            class TeacherDouble:
                def from_pil(self, image):
                    return torch.tensor(image.getpixel((0, 0)), dtype=torch.float32)

                def save_source_descriptor(self, path, descriptor, source_path):
                    torch.save({"descriptor": descriptor, "source_path": str(source_path)}, path)

            # Controlled baseline retains the source identity while recording a
            # domain request. This validates bookkeeping, not model quality.
            def generate_double(t2i, i2i, vae, source, prompt, negative, seed, condition):
                self.assertEqual(condition, "snow")
                self.assertEqual(seed, 42)
                return source.copy()

            argv = ["prepare_data", "--prompts", str(prompt_path), "--image-root", str(gsv / "Images"),
                    "--output-dir", str(out)]
            with patch("sys.argv", argv), patch.object(prepare_data, "load_iclight", return_value=(None, None, None)), \
                    patch.object(prepare_data, "load_salad", return_value=TeacherDouble()), \
                    patch.object(prepare_data, "generate_released", side_effect=generate_double), \
                    redirect_stdout(StringIO()):
                prepare_data.main()
            manifest = out / "generator_train.jsonl"
            rows = read_manifest(manifest)
            self.assertFalse((out / "generator_train.jsonl.partial").exists())
            self.assertEqual(len(rows), 2)
            for place, row in enumerate(rows, 1):
                self.assertEqual(row["place_id"], f"{place:07d}")
                self.assertEqual(row["city"], "Bangkok")
                self.assertEqual(row["prompt"], f"snow {place}")
                payload = torch.load(row["source_descriptor"], weights_only=True)
                self.assertEqual(payload["source_path"], row["source_path"])
                with Image.open(row["baseline_path"]) as baseline:
                    self.assertTrue(torch.equal(payload["descriptor"],
                                                torch.tensor(baseline.getpixel((0, 0)), dtype=torch.float32)))
            # Replacing a baseline with another sample is detected before training.
            Path(rows[0]["baseline_path"]).write_bytes(Path(rows[1]["baseline_path"]).read_bytes())
            with self.assertRaisesRegex(ValueError, "baseline content mismatch"):
                read_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
