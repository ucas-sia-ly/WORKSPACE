"""Independent scene contract, three-view boundary and audit tests; no VLM load."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from targeted.scene_planner import (
    FAMILIES, IMAGE_LABELS, SCENE_SCHEMA, SYSTEM_PROMPT, SceneInput, ScenePlanner,
    SceneValidationError, parse_scene_decision, pixel_sha256, prepare_scene_views,
)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("scene_audit", ROOT / "scripts/stage3_audit_scene_planner.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def decision(**changes):
    return dict(dict(editable=True, region_type="mixed", support_surface="paved_ground",
                     feasible_families=["construction_barrier", "traffic_cones"], confidence=.9,
                     reason="The nearby sidewalk and road provide visible paved support."), **changes)


def sample():
    source = Image.new("RGB", (320, 240), (60, 100, 140))
    roi = np.zeros((16, 16), dtype=bool)
    roi[5:11, 5:11] = True
    roi[4, 5:7] = True
    views, metadata = prepare_scene_views(source, roi)
    return SceneInput(dict(image_key="City/source.jpg", place_key="City:1"), *views, metadata), roi


class FakeClient:
    metadata = dict(backend="test_double")

    def __init__(self, replies=None):
        self.replies = replies or [json.dumps(decision())]
        self.calls = []

    def decide(self, **kwargs):
        self.calls.append(kwargs)
        return self.replies[min(len(self.calls)-1, len(self.replies)-1)]


class ScenePlannerTests(unittest.TestCase):
    def make_dev(self, root):
        dev = root / "dev"
        maps = dev / "vulnerability/maps"
        maps.mkdir(parents=True)
        source = root / "source.png"
        Image.new("RGB", (32, 24), "gray").save(source)
        _, roi = sample()
        cohort, records = [], []
        for index in range(50):
            identity = dict(image_key=f"City/{index}.png", place_key=f"City:{index}", city_id="City",
                            local_place_id=index, role="SOURCE")
            row = dict(identity, source_identity=identity, cohort="dev", source_role="SOURCE",
                       source_path=str(source), source_sha256=audit.digest(source))
            cohort.append(row)
            artifact = maps / f"{index}.npz"
            np.savez(artifact, image_key=identity["image_key"], place_key=identity["place_key"],
                     source_identity_json=json.dumps(identity), source_width=32, source_height=24,
                     attention_roi_token_mask=roi)
            records.append(dict(row, source_width=32, source_height=24, token_grid=[16,16], mask_ratio=.15,
                                mask_tokens=38, primary_roi="attention_roi_token_mask", mask_mode="connected_topk",
                                numerical_artifact=f"maps/{index}.npz", numerical_artifact_sha256=audit.digest(artifact)))
        for path, rows in ((dev/"cohort.jsonl", cohort), (dev/"vulnerability/vulnerability.jsonl", records)):
            path.write_text("".join(json.dumps(row)+"\n" for row in rows))
        audit.write_json(dev/"summary.json", dict(cohort="dev", dev=50, cohort_sha256=audit.digest(dev/"cohort.jsonl"),
                                                  stage2_audit=dict(excluded_place_keys=["City:100"])))
        audit.write_json(dev/"vulnerability/summary.json", dict(cohort="dev", status="COMPLETE", record_count=50,
            records_sha256=audit.digest(dev/"vulnerability/vulnerability.jsonl"),
            inputs_sha256={str(dev/name):audit.digest(dev/name) for name in ("cohort.jsonl", "summary.json")}))
        return dev

    def test_loader_uses_first_twenty_dev_and_rejects_artifact_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dev = self.make_dev(root)
            inputs, provenance = audit.load_inputs(dev)
            self.assertEqual([scene.identity["place_key"] for scene in inputs], [f"City:{i}" for i in range(20)])
            self.assertFalse(provenance["generation_mask_read"])
            self.assertEqual(len(inputs[0].views), 3)
            (dev/"vulnerability/maps/0.npz").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "NPZ hash mismatch"):
                audit.load_inputs(dev)

    def test_loader_rejects_identity_mismatch_even_with_updated_manifest_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            dev = self.make_dev(Path(temp))
            path = dev/"vulnerability/vulnerability.jsonl"
            records = audit.read_jsonl(path)
            records[0]["place_key"] = "City:999"
            path.write_text("".join(json.dumps(row)+"\n" for row in records))
            summary_path = dev/"vulnerability/summary.json"
            summary = json.loads(summary_path.read_text())
            summary["records_sha256"] = audit.digest(path)
            audit.write_json(summary_path, summary)
            with self.assertRaisesRegex(ValueError, "identities/order"):
                audit.load_inputs(dev)

    def test_six_fields_closed_multi_family_schema(self):
        parsed = parse_scene_decision(json.dumps(decision()))
        self.assertEqual(set(parsed.to_dict()), set(SCENE_SCHEMA["required"]))
        self.assertEqual(len(parsed.feasible_families), 2)
        self.assertEqual(len(FAMILIES), 7)
        self.assertTrue(parse_scene_decision(json.dumps(decision(feasible_families=list(FAMILIES)))).editable)
        self.assertFalse(parse_scene_decision(json.dumps(decision(editable=False, feasible_families=[]))).editable)

    def test_rejects_extra_coordinates_masks_prompts_objects(self):
        for field in ("coordinates", "bbox", "mask", "prompt", "object_name", "position", "occluder_family"):
            with self.subTest(field=field), self.assertRaises(SceneValidationError):
                parse_scene_decision(json.dumps(decision(**{field: "forbidden"})))
        for families in (["bus"], ["dumpster"], ["none"], ["traffic_cones", "traffic_cones"], "traffic_cones", [4], []):
            with self.subTest(families=families), self.assertRaises(SceneValidationError):
                parse_scene_decision(json.dumps(decision(feasible_families=families)))

    def test_strict_types_duplicates_missing_fields_and_reason(self):
        bad = [decision(confidence=True), decision(confidence=float("nan")), decision(confidence=1.1),
               decision(editable=1), decision(editable=False), decision(reason=""), decision(region_type="anything")]
        for reason in ("Use bbox (1, 2, 3, 4).", "Place a car on the left.", "Generate a photorealistic street.",
                       "The coordinates are x and y.", "Edit the mask.", "prompt: realistic car", "Support spans ４ meters."):
            bad.append(decision(reason=reason))
        raw = json.dumps(decision())
        missing = decision()
        del missing["confidence"]
        for text in [json.dumps(value) for value in bad + [missing]] + [
                "```json\n"+raw+"\n```", raw+raw, raw[:-1]+',"editable":true}', "[]"]:
            with self.subTest(text=text), self.assertRaises(SceneValidationError):
                parse_scene_decision(text)

    def test_display_overlay_projection_crop_and_source_unchanged(self):
        scene, roi = sample()
        original = np.asarray(scene.original)
        overlay = np.asarray(scene.vulnerability_overlay)
        projected = roi[np.arange(240)[:, None]*16//240, np.arange(320)[None, :]*16//320]
        np.testing.assert_array_equal(original[~projected], overlay[~projected])
        self.assertTrue(np.any(original[projected] != overlay[projected]))
        box = scene.view_metadata["context_crop_xyxy"]
        self.assertEqual(scene.roi_context_crop.tobytes(), scene.original.crop(box).tobytes())
        self.assertFalse(scene.view_metadata["generation_mask_used"])
        self.assertGreater(scene.roi_context_crop.width * scene.roi_context_crop.height, int(projected.sum()))
        self.assertFalse(hasattr(scene, "generation_mask"))

    def test_invalid_roi_budget_and_type_rejected(self):
        scene, roi = sample()
        for invalid in (roi.astype(np.uint8), np.zeros((16,16), bool), np.zeros((224,224), bool)):
            with self.assertRaises(ValueError):
                prepare_scene_views(scene.original, invalid)

    def test_only_three_rgb_copies_seed_and_semantic_instructions(self):
        scene, _ = sample()
        client = FakeClient()
        before = [pixel_sha256(image) for image in scene.views]
        row = ScenePlanner(client).plan(scene, seed=17)
        call = client.calls[0]
        self.assertEqual(call["seed"], 17)
        self.assertEqual(len(call["images"]), 3)
        for sent, original in zip(call["images"], scene.views):
            self.assertIsNot(sent, original)
            self.assertEqual(sent.mode, "RGB")
        call["images"][0].paste("black", (0,0,10,10))
        self.assertEqual(before, [pixel_sha256(image) for image in scene.views])
        self.assertNotIn("prompts", row)
        self.assertFalse(row["spatial_placement_decided"])
        self.assertIn("NOT an editing boundary", IMAGE_LABELS[1])
        self.assertNotIn("ONLY editable", " ".join(IMAGE_LABELS))
        self.assertIn("Do not require an object or its shadow to fit inside magenta", SYSTEM_PROMPT)

    def test_schema_retry_preserves_failed_raw_and_never_fabricates(self):
        scene, _ = sample()
        client = FakeClient([json.dumps(decision(bbox=[1,2,3,4])), json.dumps(decision())])
        row = ScenePlanner(client).plan(scene)
        self.assertEqual(row["status"], "editable")
        self.assertEqual(len(row["response_attempts"]), 2)
        self.assertIsNotNone(row["response_attempts"][0]["validation_error"])
        failed = ScenePlanner(FakeClient(["bad JSON"])).plan(scene)
        self.assertEqual(failed["status"], "schema_error")
        self.assertIsNone(failed["decision"])

    def test_audit_artifacts_and_schema_errors_count_separately(self):
        scene, _ = sample()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/"audit"
            summary = audit.run_audit([scene], output, ScenePlanner(FakeClient()), seed=0)
            self.assertEqual(summary["status"], "COMPLETE")
            self.assertFalse(summary["diffusion_called"])
            self.assertEqual(summary["feasible_family_counts"]["traffic_cones"], 1)
            self.assertFalse(list(output.rglob("*generation_mask*")))
            self.assertTrue((output/"human_review.csv").is_file())
            self.assertTrue((output/"contact_sheet.jpg").is_file())
            with self.assertRaises(FileExistsError):
                audit.run_audit([scene], output, ScenePlanner(FakeClient()))
            error = audit.run_audit([scene], Path(temp)/"error", ScenePlanner(FakeClient(["bad"])))
            self.assertEqual(error["schema_error_count"], 1)
            self.assertEqual(error["editable_count"]+error["rejected_count"], 0)

    def test_import_and_help_do_not_load_models_or_generation(self):
        code = "import sys; import targeted.scene_planner; assert 'torch' not in sys.modules; assert not any(k.startswith('generation') for k in sys.modules)"
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
        result = subprocess.run([sys.executable, str(ROOT/"scripts/stage3_audit_scene_planner.py"), "--help"],
                                capture_output=True, text=True, check=True)
        self.assertIn("--count", result.stdout)


if __name__ == "__main__":
    unittest.main()
