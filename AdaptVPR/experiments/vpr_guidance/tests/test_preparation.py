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
from AdaptVPR.experiments.vpr_guidance.train_online_generator import read_source_manifest
from AdaptVPR.experiments.vpr_guidance.teacher import file_sha256, PREPROCESSING_VERSION


class PreparationTests(unittest.TestCase):
    def test_preparation_caches_sources_without_generating_targets(self):
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
                model_fingerprint = "fixed-test-teacher"

                def load_source_descriptor(self, path, source_path):
                    payload = torch.load(path, weights_only=True)
                    if payload["source_sha256"] != file_sha256(source_path):
                        raise ValueError("source content mismatch")
                    return payload["descriptor"]

                def from_pil(self, image):
                    return torch.tensor(image.getpixel((0, 0)), dtype=torch.float32)

                def save_source_descriptor(self, path, descriptor, source_path):
                    torch.save({"descriptor": descriptor, "source_path": str(source_path),
                                "source_sha256": file_sha256(source_path)}, path)

            argv = ["prepare_data", "--prompts", str(prompt_path), "--gsv-root", str(gsv), "--salad-root", str(root / "salad"),
                    "--output-dir", str(out)]
            with patch("sys.argv", argv), patch.object(prepare_data, "load_salad", return_value=TeacherDouble()), \
                    redirect_stdout(StringIO()):
                prepare_data.main()
            manifest = out / "source_manifest.jsonl"
            rows, _ = read_source_manifest(manifest, TeacherDouble())
            self.assertFalse((out / "source_manifest.jsonl.partial").exists())
            self.assertEqual(len(rows), 2)
            for place, row in enumerate(rows, 1):
                self.assertEqual(row["place_id"], f"{place:07d}")
                self.assertEqual(row["city"], "Bangkok")
                self.assertEqual(row["prompt"], f"snow {place}")
                payload = torch.load(row["source_descriptor"], weights_only=True)
                self.assertEqual(payload["source_path"], row["source_path"])
                self.assertNotIn("baseline_path", row)
                self.assertEqual(row["teacher_sha256"], TeacherDouble.model_fingerprint)
                self.assertEqual(row["preprocessing_version"], PREPROCESSING_VERSION)
            self.assertFalse((out / "baseline").exists())
            with patch("sys.argv", argv), patch.object(prepare_data, "load_salad", return_value=TeacherDouble()), \
                    patch.object(TeacherDouble, "from_pil", side_effect=AssertionError("cache must be reused")), \
                    redirect_stdout(StringIO()):
                prepare_data.main()
            Path(rows[0]["source_path"]).write_bytes(Path(rows[1]["source_path"]).read_bytes())
            with self.assertRaisesRegex(ValueError, "source content changed"):
                read_source_manifest(manifest, TeacherDouble())


if __name__ == "__main__":
    unittest.main()
