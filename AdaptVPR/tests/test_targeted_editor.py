"""Contract tests plus isolated actual tiny Diffusers 4/9-channel sampling."""

from dataclasses import replace
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from generation.targeted_editor import (
    DiffusersMaskedEditor, MaskedEditorConfig, TargetedEditor, canonical_mask, load_task_pair, pixel_sha256,
)

ROOT = Path(__file__).resolve().parents[1]


class RecordingEditor(DiffusersMaskedEditor):
    """Contract-only spy. Not used by CLI, smoke or actual sampling tests."""
    def _load_pipeline(self):
        return object()

    def _sample(self, pipe, source, mask_image, prompt, seed, audit):
        self.received = dict(source=source.copy(), mask=mask_image.copy(), prompt=prompt, seed=seed)
        return Image.new("RGB", source.size, "red")


class TargetedEditorTest(unittest.TestCase):
    def setUp(self):
        # Only contract spies run here; real torch sampling is checked in a clean subprocess.
        scope = patch("generation.targeted_editor._deterministic_sampling", side_effect=lambda: nullcontext())
        scope.start()
        self.addCleanup(scope.stop)
        self.config = MaskedEditorConfig(model_path="explicit-test-path")
        self.source = Image.new("RGB", (32, 24), (20, 40, 60))
        self.mask = Image.new("L", self.source.size, 0)
        self.mask.paste(1, (8, 4, 24, 20))

    def test_unified_interface(self):
        with self.assertRaises(TypeError):
            TargetedEditor()
        self.assertIsInstance(DiffusersMaskedEditor(self.config), TargetedEditor)

    def test_zero_mask_is_exact_identity_without_loading_backend(self):
        editor = DiffusersMaskedEditor(self.config)
        with patch.object(editor, "_load_pipeline", side_effect=AssertionError("zero mask loaded a model")):
            output = editor.edit(self.source, Image.new("L", self.source.size), "edit", 5)
        self.assertEqual(output.tobytes(), self.source.tobytes())
        self.assertIsNot(output, self.source)
        self.assertFalse(editor.last_audit["generated"])
        self.assertFalse(editor.last_audit["mask_participated_in_sampling"])

    def test_mask_source_prompt_and_seed_reach_backend_together(self):
        editor = RecordingEditor(self.config)
        before_source, before_mask = self.source.tobytes(), self.mask.tobytes()
        output = editor.edit(self.source, self.mask, "a barrier", 37)
        self.assertEqual(editor.received["source"].tobytes(), before_source)
        self.assertEqual(editor.received["mask"].tobytes(), canonical_mask(self.mask, self.source.size).tobytes())
        self.assertEqual((editor.received["prompt"], editor.received["seed"]), ("a barrier", 37))
        self.assertEqual(output.getpixel((10, 10)), (255, 0, 0))
        self.assertEqual(output.getpixel((0, 0)), self.source.getpixel((0, 0)))
        self.assertEqual(self.source.tobytes(), before_source)
        self.assertEqual(self.mask.tobytes(), before_mask)

    def test_binary_formats_have_same_canonical_bytes(self):
        white = self.mask.point(lambda x: x * 255)
        bit = white.convert("1")
        outputs = [canonical_mask(m, self.source.size).tobytes() for m in (self.mask, white, bit)]
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], outputs[2])

    def test_nonbinary_rgb_and_wrong_size_rejected_before_backend(self):
        editor = DiffusersMaskedEditor(self.config)
        bad = [Image.new("L", (31, 24)), Image.new("RGB", self.source.size), Image.new("L", self.source.size, 127)]
        mixed = self.mask.copy()
        mixed.putpixel((8, 4), 255)
        bad.append(mixed)
        with patch.object(editor, "_load_pipeline", side_effect=AssertionError("loaded before validation")):
            for mask in bad:
                with self.subTest(mode=mask.mode, size=mask.size), self.assertRaises(ValueError):
                    editor.edit(self.source, mask, "edit", 0)

    def test_no_hidden_resize_for_nonzero_mask(self):
        editor = DiffusersMaskedEditor(self.config)
        with self.assertRaisesRegex(ValueError, "multiples of 8"):
            editor.edit(Image.new("RGB", (31, 24)), Image.new("L", (31, 24), 1), "edit", 0)

    def test_invalid_prompt_and_seed(self):
        editor = DiffusersMaskedEditor(self.config)
        for prompt, seed in (("", 0), (None, 0), ("edit", -1), ("edit", True), ("edit", 2**63)):
            with self.subTest(prompt=prompt, seed=seed), self.assertRaises(ValueError):
                editor.edit(self.source, self.mask, prompt, seed)

    def test_config_has_no_implicit_model_path(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "TARGETED_EDITOR_MODEL_PATH"):
                MaskedEditorConfig.from_env()
        with patch.dict(os.environ, {"TARGETED_EDITOR_MODEL_PATH": "/configured/model", "TARGETED_EDITOR_DEVICE": "cpu",
                                     "TARGETED_EDITOR_DTYPE": "float32", "TARGETED_EDITOR_STEPS": "3"}):
            config = MaskedEditorConfig.from_env()
            self.assertEqual((config.model_path, config.device, config.num_inference_steps), ("/configured/model", "cpu", 3))
        with self.assertRaises(ValueError):
            replace(self.config, device="cpu", dtype="float16")

    def test_bad_backend_output_rejected_and_audit_not_stale(self):
        editor = RecordingEditor(self.config)
        editor.edit(self.source, self.mask, "edit", 1)
        with patch.object(editor, "_sample", return_value=Image.new("RGB", (8, 8))):
            with self.assertRaisesRegex(RuntimeError, "wrong dimensions"):
                editor.edit(self.source, self.mask, "edit", 2)
        self.assertIsNone(editor.last_audit)
        self.assertIsNone(editor.last_raw_output)

    def test_pairing_hashes_reject_source_or_mask_swap(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "City").mkdir()
            source_path, mask_path = root / "City/source.png", root / "mask.png"
            self.source.save(source_path)
            self.mask.save(mask_path)
            task = dict(sample_id="sample", image_key="City/source.png", source_path=str(source_path),
                        mask_original_path=str(mask_path), source_width=32, source_height=24,
                        source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                        mask_original_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest())
            source, mask = load_task_pair(task)
            self.assertEqual(pixel_sha256(source), pixel_sha256(self.source))
            self.assertEqual(pixel_sha256(mask), pixel_sha256(self.mask))
            Image.new("L", self.mask.size, 1).save(mask_path)
            with self.assertRaisesRegex(ValueError, "mask_original_sha256 mismatch"):
                load_task_pair(task)
            self.mask.save(mask_path)
            Image.new("RGB", self.source.size, "red").save(source_path)
            with self.assertRaisesRegex(ValueError, "source_sha256 mismatch"):
                load_task_pair(task)

    def test_actual_diffusers_sampling_masks_and_determinism(self):
        # Separate process avoids the older test suite's global numpy module stub.
        result = subprocess.run([sys.executable, str(ROOT / "tests/targeted_editor_sampling_check.py")],
                                cwd=ROOT, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual([r["unet_channels"] for r in report], [4, 9])
        self.assertTrue(all(r["actual_sampling"] and r["moved_mask_changes_raw"] and r["deterministic_repeat"]
                            and r["dropped_mask_rejected"] for r in report))


if __name__ == "__main__":
    unittest.main()
