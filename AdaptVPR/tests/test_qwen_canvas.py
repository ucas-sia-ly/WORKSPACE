import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from PIL import Image

from adapters import lightx2v_qwen_image_edit as adapter


class RecordingPipeline:
    """CPU contract double; real model inference is checked separately."""

    def __init__(self, wrong_size=False):
        self.calls = []
        self.wrong_size = wrong_size

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        height, width = kwargs["target_shape"]
        size = (400, 300) if self.wrong_size else (width, height)
        Image.new("RGB", size).save(kwargs["save_result_path"])


class QwenCanvasTest(unittest.TestCase):
    def test_four_three_is_source_scaled_not_sixteen_nine(self):
        for size in ((400, 300), (800, 600), (40, 30)):
            self.assertEqual(adapter.source_target_shape(*size), [1104, 1472])

    def test_portrait_and_square(self):
        self.assertEqual(adapter.source_target_shape(300, 400), [1472, 1104])
        height, width = adapter.source_target_shape(100, 100)
        self.assertEqual(height, width)

    def test_nonstandard_ratios_stay_close_and_within_runner_limits(self):
        for width, height in ((403, 301), (1201, 799), (1920, 1080), (1080, 1920), (650, 100), (100, 650), (1, 1)):
            out_height, out_width = adapter.source_target_shape(width, height)
            for side in (out_height, out_width):
                self.assertGreaterEqual(side, 256)
                self.assertLessEqual(side, 1664)
                self.assertEqual(side % 16, 0)
            self.assertLess(abs((out_width / out_height) / (width / height) - 1), .03)

    def test_impossible_ratios_and_invalid_dimensions_are_rejected(self):
        for size in ((651, 100), (100, 651), (0, 1), (1, 0), (-1, 10)):
            with self.assertRaises(ValueError):
                adapter.source_target_shape(*size)

    def run_request(self, source_size=(400, 300), wrong_size=False, unreadable=False):
        pipeline = RecordingPipeline(wrong_size=wrong_size)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            if unreadable:
                source.write_text("not an image")
            else:
                Image.new("RGB", source_size).save(source)
            with patch.object(adapter.state, "pipe", pipeline), patch.dict(os.environ, {"LIGHTX2V_OUTPUT_DIR": temporary}):
                response = adapter.generate(adapter.GenerateRequest(
                    image_path=str(source), prompt="rainy street", seed=123,
                ))
                metadata = json.loads(Path(response["metadata_path"]).read_text())
            return response, metadata, pipeline.calls

    def test_generate_forwards_shape_and_records_actual_canvas(self):
        response, metadata, calls = self.run_request()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["target_shape"], [1104, 1472])
        self.assertEqual(calls[0]["prompt"], "rainy street")
        self.assertEqual(calls[0]["seed"], 123)
        self.assertEqual(response["source_dimensions"], [400, 300])
        self.assertEqual(response["raw_dimensions"], [1472, 1104])
        self.assertEqual(response["canvas_policy"], "source_aspect_v1")
        self.assertEqual(metadata["target_shape"], response["target_shape"])
        self.assertEqual(metadata["seed"], 123)

    def test_wrong_actual_output_cannot_be_hidden_by_client_resize(self):
        with self.assertRaises(HTTPException) as error:
            self.run_request(wrong_size=True)
        self.assertEqual(error.exception.status_code, 500)
        self.assertIn("differs from requested", error.exception.detail)

    def test_sequential_requests_do_not_reuse_previous_canvas(self):
        pipeline = RecordingPipeline()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.png"
            with patch.object(adapter.state, "pipe", pipeline), patch.dict(os.environ, {"LIGHTX2V_OUTPUT_DIR": temporary}):
                for size in ((400, 300), (300, 400), (400, 300)):
                    Image.new("RGB", size).save(source)
                    response = adapter.generate(adapter.GenerateRequest(image_path=str(source), prompt="fog"))
                    self.assertEqual(response["source_dimensions"], list(size))
        self.assertEqual([call["target_shape"] for call in pipeline.calls],
                         [[1104, 1472], [1472, 1104], [1104, 1472]])

    def test_unreadable_source_and_unsupported_ratio_http_errors(self):
        for kwargs, status in (({"unreadable": True}, 400), ({"source_size": (1000, 10)}, 422)):
            with self.assertRaises(HTTPException) as error:
                self.run_request(**kwargs)
            self.assertEqual(error.exception.status_code, status)

    def test_health_preserves_sampling_and_exposes_canvas_policy(self):
        with patch.object(adapter.state, "pipe", object()), patch.object(adapter.state, "error", None):
            health = adapter.health()
        self.assertEqual(health["status"], "ok")
        self.assertTrue(health["generator_ready"])
        self.assertEqual(health["sampling"]["infer_steps"], 4)
        self.assertEqual(health["sampling"]["guidance_scale"], 1.0)
        self.assertEqual(health["canvas_policy"], "source_aspect_v1")
        self.assertEqual(health["canvas"]["target_shape_order"], "height,width")


if __name__ == "__main__":
    unittest.main()
