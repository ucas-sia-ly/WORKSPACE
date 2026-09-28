"""Mask-only geometry tests; no numpy/torch/model dependencies."""

import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageChops, ImageDraw

from targeted.mask_adapter import adapt_generation_mask

ROOT = Path(__file__).resolve().parents[1]


class GenerationMaskAdapterTest(unittest.TestCase):
    def cross(self):
        mask = Image.new("L", (160, 120), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle((30, 45, 129, 74), fill=1)
        draw.rectangle((65, 10, 94, 109), fill=1)
        return mask

    def assert_success(self, original, generation, diagnostic, ratio=.06, overlap=.70):
        self.assertIsNotNone(generation, diagnostic)
        self.assertEqual(diagnostic["status"], "success")
        self.assertIsNone(diagnostic["failure_reason"])
        self.assertEqual(generation.size, original.size)
        self.assertEqual(generation.mode, "L")
        self.assertEqual({i for i, n in enumerate(generation.histogram()) if n}, {0, 1})
        area = generation.histogram()[1]
        vuln = original.convert("L").point(lambda x: 1 if x else 0)
        measured = ImageChops.darker(generation, vuln).histogram()[1]
        self.assertEqual(diagnostic["generation_area"], area)
        self.assertEqual(diagnostic["overlap_pixels"], measured)
        self.assertAlmostEqual(diagnostic["overlap_ratio"], measured / area)
        self.assertGreaterEqual(measured / area, overlap)
        self.assertLessEqual(abs(area / (original.width * original.height) - ratio), ratio * .05)
        self.assertGreaterEqual(diagnostic["generation_area_ratio"], .04)
        self.assertLessEqual(diagnostic["generation_area_ratio"], .08)
        self.assertEqual(diagnostic["num_components_after"], 1)
        self.assertGreaterEqual(diagnostic["compactness_after"] + 1e-12, diagnostic["compactness_before"])
        self.assertGreaterEqual(diagnostic["centroid_distance_normalized"], 0)
        self.assertLessEqual(diagnostic["centroid_distance_normalized"], 1)

    def test_contiguous_compact_area_overlap_and_unchanged_input(self):
        original = self.cross()
        before = original.tobytes()
        generation, diagnostic = adapt_generation_mask(original, image_height=120, image_width=160)
        self.assert_success(original, generation, diagnostic)
        self.assertEqual(original.tobytes(), before)
        self.assertEqual(diagnostic["num_components_before"], 1)
        self.assertGreater(diagnostic["compactness_after"], diagnostic["compactness_before"])
        self.assertLess(diagnostic["generation_area"], diagnostic["vulnerability_area"])
        self.assertNotEqual(generation.getbbox(), original.getbbox())

    def test_fragmented_dense_region_becomes_one_component(self):
        original = Image.new("L", (160, 120), 0)
        for x in (30, 65, 100):
            original.paste(1, (x, 20, x + 30, 100))
        generation, diagnostic = adapt_generation_mask(original)
        self.assert_success(original, generation, diagnostic)
        self.assertEqual(diagnostic["num_components_before"], 3)

    def test_sparse_fragmented_region_fails_explicitly(self):
        original = Image.new("L", (160, 120), 0)
        for y in range(0, 120, 12):
            for x in range(0, 160, 12):
                original.paste(1, (x, y, min(x + 4, 160), min(y + 4, 120)))
        generation, diagnostic = adapt_generation_mask(original)
        self.assertIsNone(generation)
        self.assertEqual(diagnostic["status"], "failure")
        self.assertEqual(diagnostic["failure_reason"], "no_compact_candidate_meets_overlap_in_configured_search")
        self.assertEqual(diagnostic["generation_area"], 0)
        self.assertLess(diagnostic["best_candidate_overlap_ratio"], .70)

    def test_thin_region_with_enough_total_area_still_fails_geometry(self):
        original = Image.new("L", (160, 120), 0)
        original.paste(1, (0, 55, 160, 65))
        generation, diagnostic = adapt_generation_mask(original)
        self.assertIsNone(generation)
        self.assertGreaterEqual(diagnostic["vulnerability_area"], .7 * diagnostic["allowed_generation_area"][0])
        self.assertEqual(diagnostic["failure_reason"], "no_compact_candidate_meets_overlap_in_configured_search")

    def test_border_touching_mask_is_not_clipped_outside_image(self):
        original = Image.new("L", (160, 120), 0)
        original.paste(1, (0, 0, 80, 100))
        generation, diagnostic = adapt_generation_mask(original)
        self.assert_success(original, generation, diagnostic)
        self.assertEqual(diagnostic["overlap_ratio"], 1)
        x, y, w, h = diagnostic["ellipse_bounds_xywh"]
        self.assertTrue(0 <= x <= x + w <= 160 and 0 <= y <= y + h <= 120)

    def test_impossible_small_and_empty_region(self):
        for area in (0, 25):
            original = Image.new("L", (160, 120), 0)
            if area:
                original.paste(1, (70, 50, 75, 55))
            generation, diagnostic = adapt_generation_mask(original)
            self.assertIsNone(generation)
            expected = "insufficient_vulnerability_area_for_required_overlap" if area else "empty_vulnerability_mask"
            self.assertEqual(diagnostic["failure_reason"], expected)
            self.assertEqual(diagnostic["vulnerability_area"], area)
            self.assertIsNone(diagnostic["centroid_generation"])
            self.assertEqual(diagnostic["num_components_after"], 0)

    def test_4_to_8_percent_configurable_with_exact_binary_output(self):
        original = Image.new("L", (160, 120), 1)
        for ratio in (.04, .06, .08):
            with self.subTest(ratio=ratio):
                generation, diagnostic = adapt_generation_mask(original, target_ratio=ratio)
                self.assert_success(original, generation, diagnostic, ratio=ratio)

    def test_stricter_overlap_threshold(self):
        original = Image.new("L", (160, 120), 0)
        original.paste(1, (20, 20, 140, 100))
        generation, diagnostic = adapt_generation_mask(original, min_overlap=1.0)
        self.assert_success(original, generation, diagnostic, overlap=1.0)

    def test_deterministic_and_does_not_touch_global_rng(self):
        original = self.cross()
        state = random.getstate()
        a, da = adapt_generation_mask(original)
        b, db = adapt_generation_mask(original)
        self.assertEqual(a.tobytes(), b.tobytes())
        self.assertEqual(da, db)
        self.assertEqual(random.getstate(), state)

    def test_0_255_and_one_bit_formats_equivalent(self):
        original = self.cross()
        a, da = adapt_generation_mask(original)
        for other in (original.point(lambda x: x * 255), original.point(lambda x: x * 255).convert("1")):
            b, db = adapt_generation_mask(other)
            self.assertEqual(a.tobytes(), b.tobytes())
            self.assertEqual(da, db)

    def test_invalid_input_policy_and_no_resizing(self):
        original = self.cross()
        for bad in (Image.new("RGB", original.size), Image.new("L", original.size, 127)):
            with self.assertRaises(ValueError):
                adapt_generation_mask(bad)
        for kwargs in (dict(target_ratio=.03), dict(target_ratio=.09), dict(min_overlap=1.1),
                       dict(image_height=119, image_width=160), dict(image_height=120),
                       dict(search_stride=0), dict(area_tolerance=-1), dict(aspect_ratios=(5,))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                adapt_generation_mask(original, **kwargs)

    def test_adapter_never_resizes_binary_masks(self):
        with patch.object(Image.Image, "resize", side_effect=AssertionError("mask resizing forbidden")):
            generation, diagnostic = adapt_generation_mask(self.cross())
        self.assertEqual(diagnostic["status"], "success")
        self.assertEqual(generation.size, (160, 120))

    def test_visualizer_success_failure_and_no_generation_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "City").mkdir()
            (root / "masks").mkdir()
            records = []
            for i in range(2):
                source = root / f"City/{i}.png"
                Image.new("RGB", (160, 120), "gray").save(source)
                mask = self.cross() if i == 0 else Image.new("L", (160, 120), 0)
                if i == 1:
                    mask.putpixel((80, 60), 1)  # Valid nonempty target, geometrically too small.
                mask.save(root / f"masks/{i}.png")
                records.append(dict(schema_version=1, sample_id=f"sample_{i}", image_key=f"City/{i}.png",
                                    place_key=f"City:{i}", source_path=str(source), mask_original_path=f"masks/{i}.png",
                                    target_type="attention", mask_ratio=.15, clean_margin=.2,
                                    mask_mode="connected_topk", mask_token_count=38, checkpoint_path="/recorded/model",
                                    checkpoint_sha256="a" * 64, stage2_commit="b" * 40, seed=0))
            manifest = root / "targets.jsonl"
            manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
            original = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            guard = """
import importlib.abc, runpy, sys
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ('torch', 'diffusers', 'transformers', 'openai', 'generation.targeted_editor',
                   'generation.agent', 'generation.lightx2v', 'generation.router', 'verification', 'prompts')
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('forbidden generation import: ' + fullname)
sys.meta_path.insert(0, NoModels())
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
"""
            output = root / "visualization"
            command = [sys.executable, "-c", guard, str(ROOT / "scripts/visualize_generation_masks.py"),
                       str(manifest), "--count", "2", "--seed", "7", "--output", str(output)]
            result = subprocess.run(command, cwd=temporary, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual((summary["success_count"], summary["failure_count"]), (1, 1))
            self.assertEqual(summary["generation_area_ratio_distribution"]["count"], 1)
            self.assertTrue(summary["no_diffusion_called"])
            self.assertTrue((output / "contact_sheet.png").is_file())
            for row in summary["samples"]:
                directory = output / row["directory"]
                self.assertTrue((directory / "source.png").is_file())
                self.assertTrue((directory / "vulnerability_overlay.png").is_file())
                self.assertTrue((directory / "side_by_side.png").is_file())
                self.assertEqual((directory / "generation_mask.png").exists(), row["diagnostic"]["status"] == "success")
            for path, data in original.items():
                self.assertEqual(Path(path).read_bytes(), data)

    def test_impossible_compactness_gain_is_failure(self):
        generation, diagnostic = adapt_generation_mask(self.cross(), min_compactness_gain=1.0)
        self.assertIsNone(generation)
        self.assertEqual(diagnostic["failure_reason"], "no_template_meets_area_size_and_compactness_constraints")


if __name__ == "__main__":
    unittest.main()
