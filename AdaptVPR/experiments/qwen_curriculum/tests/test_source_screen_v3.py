"""CPU fixtures for durable, unbiased per-place source screening."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.qwen_curriculum import source_screen_v3 as screen


def textured(path, seed=0):
    rng = np.random.default_rng(seed)
    image = Image.fromarray(rng.integers(20, 235, (150, 200, 3), dtype=np.uint8))
    image.save(path)


class ScreeningTests(unittest.TestCase):
    def make_gsv(self, root, places=3, cities=("City",)):
        gsv = root / "gsv"
        (gsv / "Dataframes").mkdir(parents=True)
        paths = {}
        for ci, city in enumerate(cities):
            (gsv / "Images" / city).mkdir(parents=True)
            rows = []
            for place in range(places):
                views = []
                for view in range(4):
                    row = dict(city_id=city, place_id=place, year=2020, month=3, northdeg=view,
                               lat="1.5", lon="2.5", panoid=f"p{place}v{view}")
                    rows.append(row)
                    path = gsv / "Images" / city / f"{city}_{place:07d}_2020_03_{view:03d}_1.5_2.5_p{place}v{view}.jpg"
                    textured(path, seed=ci * 10000 + place * 10 + view)
                    views.append(path)
                paths[city, place] = views
            with (gsv / "Dataframes" / f"{city}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, rows[0])
                writer.writeheader()
                writer.writerows(rows)
        return gsv, paths

    def test_image_extremes_corruption_blur_and_texture(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.png"
            textured(path)
            verdict = screen.screen_image(path)
            self.assertTrue(verdict["passed"], verdict)
            self.assertEqual(verdict["source_dimensions"], [200, 150])
            self.assertEqual(len(verdict["source_sha256"]), 64)
            for color, reason in [(0, "source_extremely_dark"), (255, "source_extremely_bright"),
                                  (120, "source_nearly_blank_or_flat")]:
                Image.new("L", (200, 150), color).save(path)
                self.assertIn(reason, screen.screen_image(path)["reasons"])
            gradient = np.broadcast_to(np.linspace(20, 235, 200, dtype=np.uint8), (150, 200))
            Image.fromarray(gradient).filter(ImageFilter.GaussianBlur(8)).save(path)
            self.assertIn("source_extremely_blurred_or_textureless", screen.screen_image(path)["reasons"])
            path.write_bytes(b"invalid image")
            self.assertIn("source_missing_or_decode_failed", screen.screen_image(path)["reasons"])
            self.assertFalse(screen.screen_image(path.parent / "missing.png")["passed"])

    def test_fallback_skip_deduplicate_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gsv, paths = self.make_gsv(root, places=4)
            ordered = {key: screen.ordered_views(p, *key, 42) for key, p in paths.items()}
            # A missing first choice falls back to another real view.
            ordered["City", 0][0].unlink()
            # One place has no usable view, yet does not abort the campaign.
            for path in paths["City", 1]:
                Image.new("RGB", (200, 150), "black").save(path)
            # The next place must reject exact content already chosen earlier.
            chosen0 = ordered["City", 0][1]
            ordered["City", 2][0].write_bytes(chosen0.read_bytes())
            output = root / "screen"
            rows = screen.screen_sources(gsv, output, workers=1)
            self.assertEqual({r["place_id"] for r in rows}, {0, 2, 3})
            self.assertEqual(len({r["source_sha256"] for r in rows}), 3)
            summary = screen.status(output)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["skipped_places"], 1)
            self.assertEqual(summary["rejection_reasons"]["source_content_already_selected"], 1)
            with patch.object(screen, "_screen_place", side_effect=AssertionError("must reuse checkpoint")):
                self.assertEqual(screen.screen_sources(gsv, output, workers=1), rows)

    def test_interruption_resumes_without_rescanning_committed_places(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gsv, _ = self.make_gsv(root)
            output = root / "screen"
            original = screen._screen_place
            calls = []
            def interrupt(task):
                calls.append(task[1])
                if len(calls) == 2:
                    raise KeyboardInterrupt()
                return original(task)
            with patch.object(screen, "_screen_place", side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    screen.screen_sources(gsv, output, workers=1)
            self.assertEqual(screen.status(output)["screened_places"], 1)
            resumed = []
            def record(task):
                resumed.append(task[1])
                return original(task)
            with patch.object(screen, "_screen_place", side_effect=record):
                rows = screen.screen_sources(gsv, output, workers=1)
            self.assertEqual(resumed, [1, 2])
            self.assertEqual(len(rows), 3)

    def test_frozen_inputs_and_selected_contents_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gsv, _ = self.make_gsv(root)
            output = root / "screen"
            rows = screen.screen_sources(gsv, output, workers=1)
            with self.assertRaisesRegex(ValueError, "Frozen"):
                screen.screen_sources(gsv, output, seed=43, workers=1)
            metadata = gsv / "Dataframes/City.csv"
            content = metadata.read_bytes()
            metadata.write_bytes(content + b"\n")
            with self.assertRaisesRegex(ValueError, "Frozen"):
                screen.screen_sources(gsv, output, workers=1)
            metadata.write_bytes(content)
            Path(rows[0]["source_path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "selected source changed"):
                screen.screen_sources(gsv, output, workers=1)

    def test_parallel_results_and_seed_order_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gsv, paths = self.make_gsv(root, cities=("Alpha", "Beta"))
            sequential = screen.screen_sources(gsv, root / "one", workers=1)
            parallel = screen.screen_sources(gsv, root / "two", workers=2)
            self.assertEqual(sequential, parallel)
            self.assertEqual(len(sequential), 6)
            self.assertEqual(len({(r["city"], r["place_id"]) for r in parallel}), 6)
            candidates = paths["Alpha", 0]
            orders = {tuple(screen.ordered_views(candidates, "Alpha", 0, seed)) for seed in range(10)}
            self.assertGreater(len(orders), 1)

    def test_csv_duplicates_do_not_satisfy_minimum_views(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gsv, _ = self.make_gsv(root, places=1)
            metadata = gsv / "Dataframes/City.csv"
            lines = metadata.read_text().splitlines()
            metadata.write_text("\n".join([lines[0]] + [lines[1]] * 4) + "\n")
            self.assertEqual(screen.eligible_places(gsv), {})


if __name__ == "__main__":
    unittest.main()
