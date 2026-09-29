"""Pilot must pass Render, preserve Core, and never filter generated outcomes."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from generation.inpainting_pilot import (
    FAMILY_OBJECTS, file_sha256, prompt_for_family, read_json, run_prepared, select_candidates, write_json,
)


class InpaintingPilotTest(unittest.TestCase):
    def test_relaxation_is_explicit_and_only_weighted_gate(self):
        def row(cid, reasons, weight=.2):
            return dict(candidate_id=cid, image_key="A/image", family="parked_vehicle", render={"feasible": True},
                        thresholds={"0.3": dict(passes=not reasons, rejection_reasons=reasons)},
                        metrics=dict(vulnerability_weighted_coverage=weight, target_precision=.8,
                                     area_fraction=.06, centroid_distance_normalized=.1))
        samples = [dict(target={"image_key": "A/image"}, scene_decision={"editable": True}, families=["parked_vehicle"])]
        rows = [row("good", ["weighted_coverage_below_threshold"]),
                row("bad", ["weighted_coverage_below_threshold", "core_footprint_clipped_at_image_boundary"], .9)]
        self.assertEqual(select_candidates(samples, rows, mode="frozen", threshold=.3), [])
        chosen = select_candidates(samples, rows, mode="render_semantics", threshold=.3)
        self.assertEqual(chosen[0][1]["candidate_id"], "good")
        self.assertEqual(chosen, select_candidates(samples, rows[::-1], mode="render_semantics", threshold=.3))

    def test_prompts_use_closed_families(self):
        for family in FAMILY_OBJECTS:
            self.assertIn("masked region", prompt_for_family(family))
        with self.assertRaises(KeyError):
            prompt_for_family("arbitrary_vlm_object")

    def test_all_outputs_preserved_and_only_render_sent_to_editor(self):
        class EditorSpy:
            calls = []
            def __init__(self, config): pass
            def edit(self, source, render, prompt, seed):
                self.calls.append((np.asarray(render).copy(), prompt, seed))
                if seed == 1:
                    raise RuntimeError("test failure must be retained, not replaced")
                self.last_raw_output = Image.new("RGB", source.size, "red")
                self.last_audit = {"mask_participated_in_sampling": True}
                return Image.composite(self.last_raw_output, source, render)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            records = []
            for i in range(10):
                directory = out/str(i)
                directory.mkdir()
                Image.new("RGB", (32, 32), (1, 2, 3)).save(directory/"source.png")
                core, render = Image.new("L", (32, 32)), Image.new("L", (32, 32))
                core.paste(255, (12, 12, 20, 20))
                render.paste(255, (8, 8, 24, 24))
                core.save(directory/"core_mask.png")
                render.save(directory/"render_mask.png")
                records.append(dict(index=i, directory=str(i), candidate_id=str(i), family="parked_vehicle", seed=i,
                                    prompt="a parked car", source_identity={"place_key": str(i)},
                                    files_sha256={p.name:file_sha256(p) for p in directory.iterdir()}))
            write_json(out/"planned_edits.json", records)
            write_json(out/"protocol.json", dict(selected_count=10, requested_count=10, inputs_sha256={}, code_sha256={},
                       planned_edits_sha256=file_sha256(out/"planned_edits.json")))
            write_json(out/"summary.json", dict(status="PREPARED", generated_count=0, diffusion_called=False))
            from generation.targeted_editor import MaskedEditorConfig
            with patch("generation.inpainting_pilot.TrainedInpaintingConfig.from_env", return_value=MaskedEditorConfig("test")), \
                 patch("generation.inpainting_pilot.inspect_checkpoint", return_value={"pipeline_class":"StableDiffusionInpaintPipeline", "test_spy": True}), \
                 patch("generation.inpainting_pilot.validate_backend", return_value={"status":"PASS"}), \
                 patch("generation.inpainting_pilot.TrainedInpaintingEditor", EditorSpy):
                result = run_prepared(out)
            self.assertEqual(result["status"], "COMPLETE_WITH_ERRORS")
            self.assertEqual((result["attempted_count"], result["generated_count"], result["failed_count"]), (10, 9, 1))
            self.assertEqual(len(EditorSpy.calls), 10)
            self.assertTrue(all((m>0).sum() == 256 for m, _, _ in EditorSpy.calls))
            self.assertEqual(read_json(out/"1/audit.json")["status"], "ERROR")
            for i in (0, 2, 3, 4, 5, 6, 7, 8, 9):
                self.assertTrue(read_json(out/f"{i}/audit.json")["outside_render_exact_rgb"])
                self.assertTrue((out/f"{i}/raw_output.png").exists())
                self.assertTrue((out/f"{i}/final_output.png").exists())
                self.assertEqual(file_sha256(out/f"{i}/core_mask.png"), records[i]["files_sha256"]["core_mask.png"])
            with self.assertRaisesRegex(ValueError, "untouched"):
                run_prepared(out)


if __name__ == "__main__":
    unittest.main()
