"""CPU checks for experiment pairing, recovery and separated failure reporting."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from . import runner


class RunnerProtocolTests(unittest.TestCase):
    def test_qwen_canvas_metadata_is_preserved_and_legacy_response_supported(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.png"
            result_path = Path(directory) / "generated.png"
            source = Image.new("RGB", (400, 300), "white")
            source.save(source_path)
            Image.new("RGB", (800, 600), "grey").save(result_path)
            metadata = {"canvas_policy": "source_aspect_ratio_v1", "source_dimensions": [400, 300],
                        "raw_dimensions": [800, 600], "target_shape": [600, 800]}
            # Exercise HTTP response handling without service health or model loading.
            backend = runner.QwenBackend.__new__(runner.QwenBackend)
            backend.url, backend.timeout, backend.session = "http://test/generate", 30, Mock()
            response = backend.session.post.return_value
            for fields in (metadata, {}):
                with self.subTest(legacy=not fields):
                    response.json.return_value = {"result_path": str(result_path), **fields}
                    image, sampling = backend.generate(source, source_path, "weather", "", "rain", 42, "qwen_edit")
                    self.assertEqual(image.size, (800, 600))
                    self.assertEqual({key: sampling[key] for key in metadata if key in sampling}, fields)
                    self.assertEqual((sampling["infer_steps"], sampling["guidance_scale"]), (4, 1.0))
                    self.assertEqual(sampling["adapter_result_path"], str(result_path))
                    backend.session.post.assert_called_with(
                        backend.url, json={"image_path": str(source_path), "prompt": "weather", "negative_prompt": "",
                                           "seed": 42, "infer_steps": 4, "guidance_scale": 1.0}, timeout=30)

    def test_old_cohort_stride_and_seed_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for index in range(16):
                source = root / f"{index:02}.png"
                source.touch()
                sources.extend([str(source)] * 4)
            manifest = root / "candidates.jsonl"
            manifest.write_text("".join(json.dumps({"source_path": path}) + "\n"
                                        for path in reversed(sources)), encoding="utf-8")
            selected = runner.select_sources(manifest, 3)
            self.assertEqual(selected, sorted(set(sources))[::5][:3])
            # Known published formula; seed changes neither by prompt nor strategy.
            self.assertEqual(runner.candidate_seed(42, "adapt_000317", 0), 802961716)
            self.assertEqual(runner.candidate_seed(42, selected[0] + "|night", 0),
                             runner.candidate_seed(42, selected[0] + "|night", 0))

    def test_torn_append_recovery_keeps_complete_results(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "results.jsonl"
            first = {"src": 0, "passed": False}
            manifest.write_bytes(json.dumps(first).encode() + b'\n{"src":1,"pa')
            self.assertEqual(runner.read_checkpoint(manifest), [first])
            runner.append_checkpoint(manifest, {"src": 1})
            self.assertEqual(runner.read_checkpoint(manifest), [first, {"src": 1}])
            manifest.write_bytes(b'{"src":0}\nmalformed\n{"src":1}\n')
            with self.assertRaises(ValueError):
                runner.read_checkpoint(manifest)

    def test_valid_unterminated_final_row_gets_newline_before_append(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "results.jsonl"
            manifest.write_text('{"src":0}', encoding="utf-8")
            self.assertEqual(runner.read_checkpoint(manifest), [{"src": 0}])
            runner.append_checkpoint(manifest, {"src": 1})
            self.assertEqual(runner.read_checkpoint(manifest), [{"src": 0}, {"src": 1}])

    def test_negation_arm_removes_only_complete_forbid_sentences(self):
        self.assertEqual(runner.prompt_for("night", "no_negations"),
                         runner.prompt_for("night", "released").replace(
                             "Avoid structural changes, blur or deformation.", "").strip())
        rain = runner.prompt_for("rain", "released")
        forbid = ("Do not add heavy rain, flooding, haze, darkness, strong blur, large reflections, "
                  "new vehicles, or structural changes.\n\n")
        self.assertEqual(runner.prompt_for("rain", "no_negations"), rain.replace(forbid, "").strip())
        self.assertEqual(runner.prompt_for("overcast", "no_negations"),
                         runner.prompt_for("overcast", "released"))

    def test_summary_separates_geometry_diversity_and_errors(self):
        def row(index, geo, div):
            return {"src": index, "cond": "night", "strat": "released", "prompt_variant": "released",
                    "method": "iclight_released_released", "status": "ok", "s_geo": 0.9 if geo else 0.6,
                    "s_div": 0.2 if div else 0.1, "geo_ok": geo, "div_ok": div, "passed": geo and div,
                    "failure_reasons": ([] if geo else ["geometry"]) + ([] if div else ["diversity"])}
        rows = [row(0, True, False), row(1, False, True), row(2, True, True)]
        failed_attempt = {**row(2, False, False), "status": "error", "failure_reasons": ["execution_error"]}
        rows.insert(0, failed_attempt)
        summary = runner.summarize(rows)
        aggregate = next(group for group in summary["groups"] if group["cond"] == "all")
        self.assertEqual(aggregate["attempted"], 3)
        self.assertEqual((aggregate["geo_ok"], aggregate["div_ok"], aggregate["passed"]), (2, 2, 1))
        self.assertEqual(aggregate["failure_reasons"], {"geometry": 1, "diversity": 1})
        self.assertEqual(aggregate["errors"], 0)


if __name__ == "__main__":
    unittest.main()
