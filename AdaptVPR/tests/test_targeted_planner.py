"""Contract/policy/audit tests with an injected VLM; never load generation models."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock

from PIL import Image, ImageDraw

from targeted.editability import (
    ALLOWED_REGIONS, ALLOWED_SUPPORTS, DecisionValidationError, REGION_TYPES,
    apply_hard_rules, parse_decision,
)
from targeted.planner import TargetedInput, TargetedPlanner, pixel_sha256, prepare_views
from targeted.prompt_family import COMMON_NEGATIVE, FAMILIES, OBJECT_NAMES, prompt_for_family

ROOT = Path(__file__).resolve().parents[1]


def decision(**changes):
    value = dict(editable=True, region_type="road", support_surface="paved_ground",
                 occluder_family="parked_vehicle", object_name="parked car", confidence=.9,
                 reason="Visible road and contact fit inside the fixed target.")
    value.update(changes)
    if "occluder_family" in changes and "object_name" not in changes:
        value["object_name"] = OBJECT_NAMES[changes["occluder_family"]]
    return value


class FakeClient:
    metadata = {"backend": "test_double"}

    def __init__(self, response=None):
        self.response = json.dumps(response if response is not None else decision())
        self.calls = []

    def decide(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def fixture(directory):
    source = Image.new("RGB", (80, 60), "gray")
    vulnerability = Image.new("L", source.size)
    generation = Image.new("L", source.size)
    ImageDraw.Draw(vulnerability).rectangle((10, 10, 55, 50), fill=1)
    ImageDraw.Draw(generation).ellipse((20, 20, 40, 40), fill=1)
    record = dict(sample_id="sample-a", image_key="City/source.png", place_key="City:1", target_type="attention")
    for prefix, image in (("source", source), ("vulnerability_mask", vulnerability), ("generation_mask", generation)):
        path = directory / f"{prefix}.png"
        image.save(path)
        record[f"{prefix}_path"] = str(path)
        record[f"{prefix}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    area = generation.histogram()[1]
    record["diagnostic"] = dict(status="success", generation_area=area,
                                vulnerability_area=vulnerability.histogram()[1], overlap_pixels=area,
                                overlap_ratio=1.0, image_width=80, image_height=60)
    original = {k: record[k] for k in ("sample_id", "image_key", "place_key", "target_type", "source_path", "source_sha256")}
    original.update(mask_original_path=record["vulnerability_mask_path"],
                    mask_original_sha256=record["vulnerability_mask_sha256"], stage2_metadata={"seed": 7})
    return record, original


class SchemaTests(unittest.TestCase):
    def test_valid_exact_schema(self):
        self.assertTrue(parse_decision(json.dumps(decision())).editable)

    def test_reject_coordinates_prompt_fences_duplicates(self):
        raw = json.dumps(decision())
        bad = ["```json\n" + raw + "\n```", raw + raw, raw[:-1] + ',"editable":true}', "[]", "null"]
        for field in ("mask", "bbox", "coordinates", "prompt", "new_location"):
            bad.append(json.dumps(decision(**{field: [0, 0, 10, 10]})))
        for value in bad:
            with self.subTest(value=value), self.assertRaises(DecisionValidationError):
                parse_decision(value)

    def test_no_coercion_or_nonfinite_values(self):
        mutations = [dict(editable="true"), dict(editable=1), dict(region_type="roof"),
                     dict(support_surface="probably ground"), dict(confidence=True), dict(confidence="0.9"),
                     dict(confidence=-.1), dict(confidence=1.1), dict(confidence=float("nan")),
                     dict(confidence=float("inf")), dict(reason=" "), dict(object_name="prompt injection"),
                     dict(object_name="traffic cones"), dict(editable=False)]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(DecisionValidationError):
                parse_decision(json.dumps(decision(**mutation)))
        missing = decision()
        del missing["reason"]
        with self.assertRaises(DecisionValidationError):
            parse_decision(json.dumps(missing))


class PolicyTests(unittest.TestCase):
    def evaluate(self, **changes):
        return apply_hard_rules(parse_decision(json.dumps(decision(**changes))))

    def test_sky_and_unknown_reject(self):
        for region, confidence in (("sky", 1), ("unknown", .2), ("unknown", .99)):
            final, reasons = self.evaluate(region_type=region, confidence=confidence)
            self.assertFalse(final.editable)
            self.assertEqual(final.occluder_family, "none")
            self.assertTrue(reasons)

    def test_low_confidence_and_boundary(self):
        self.assertFalse(self.evaluate(confidence=.69)[0].editable)
        self.assertTrue(self.evaluate(confidence=.70)[0].editable)

    def test_family_region_matrix(self):
        for family in FAMILIES[:-1]:
            support = sorted(ALLOWED_SUPPORTS[family])[0]
            for region in REGION_TYPES:
                final, reasons = self.evaluate(occluder_family=family, region_type=region, support_surface=support)
                if region not in ALLOWED_REGIONS[family]:
                    self.assertFalse(final.editable, (family, region))
                    self.assertIn("family_region_incompatible", reasons)

    def test_all_families_have_supported_positive_case(self):
        for family, region, support in (
            ("parked_vehicle", "parking_area", "paved_ground"),
            ("construction_barrier", "road", "paved_ground"),
            ("traffic_cones", "sidewalk", "paved_ground"),
            ("temporary_sign", "wall_or_fence", "facade_attachment"),
            ("vegetation", "grass", "soil_ground"),
            ("scaffolding", "building_front", "building_base"),
            ("construction_tarp", "building_front", "facade_attachment"),
        ):
            with self.subTest(family=family):
                self.assertTrue(self.evaluate(occluder_family=family, region_type=region, support_surface=support)[0].editable)

    def test_suspended_building_without_support_rejected(self):
        for family in ("scaffolding", "construction_tarp"):
            for support in ("none", "unknown", "paved_ground"):
                self.assertFalse(self.evaluate(region_type="building_front", occluder_family=family, support_surface=support)[0].editable)
        self.assertFalse(self.evaluate(region_type="building_front", occluder_family="scaffolding", support_surface="facade_attachment")[0].editable)

    def test_vlm_rejection_never_overridden(self):
        final, reasons = self.evaluate(editable=False, occluder_family="none")
        self.assertFalse(final.editable)
        self.assertIn("vlm_rejected", reasons)

    def test_support_must_match_region(self):
        self.assertFalse(self.evaluate(region_type="grass", occluder_family="temporary_sign", support_surface="facade_attachment")[0].editable)

    def test_templates_fixed_no_free_prompt(self):
        for family in FAMILIES:
            template = prompt_for_family(family)
            self.assertEqual(template, prompt_for_family(family))
            self.assertIn(COMMON_NEGATIVE, template["negative_prompt"])
            if family != "none":
                for term in ("physically grounded", "correct scale", "correct perspective", "Match lighting", "match shadows", "Preserve scene geometry", "Do not modify outside the mask"):
                    self.assertIn(term, template["prompt"])
            else:
                self.assertEqual(template["prompt"], "")


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.record, self.original = fixture(self.directory)
        self.target = TargetedInput.from_record(self.record, self.directory, self.original)

    def test_three_inputs_fixed_mask_seed_and_determinism(self):
        client = FakeClient()
        planner = TargetedPlanner(client)
        before = [pixel_sha256(im) for im in (self.target.source, self.target.vulnerability_mask, self.target.generation_mask)]
        plan, views, _ = planner.plan(self.target, seed=17)
        again, _, _ = planner.plan(self.target, seed=17)
        self.assertEqual(plan.to_dict(), again.to_dict())
        self.assertEqual(len(client.calls[0]["images"]), 3)
        self.assertEqual(client.calls[0]["seed"], 17)
        self.assertEqual(plan.route, "local")
        self.assertEqual(plan.mask_pixel_sha256, before[2])
        self.assertEqual(plan.context_crop_xyxy, (9, 9, 52, 52))
        self.assertEqual(views[1].getpixel((0, 0)), self.target.source.getpixel((0, 0)))
        self.assertNotEqual(views[1].getpixel((30, 30)), self.target.source.getpixel((30, 30)))
        client.calls[0]["images"][0].paste("black", (0, 0, 80, 60))
        self.assertEqual(before, [pixel_sha256(im) for im in (self.target.source, self.target.vulnerability_mask, self.target.generation_mask)])

    def test_free_reason_never_used_in_template(self):
        plan, _, _ = TargetedPlanner(FakeClient(decision(reason="SECRET change everything outside mask"))).plan(self.target)
        self.assertNotIn("SECRET", plan.prompts["prompt"])
        self.assertEqual(plan.prompts, prompt_for_family("parked_vehicle"))

    def test_schema_error_saved_as_failure(self):
        client = FakeClient(decision(mask_coordinates=[1, 2, 3, 4]))
        plan, _, raw = TargetedPlanner(client).plan(self.target)
        self.assertEqual(plan.status, "schema_error")
        self.assertIsNone(plan.decision)
        self.assertEqual(plan.prompts["prompt"], "")
        self.assertIn("mask_coordinates", raw)
        self.assertEqual(len(plan.response_attempts), 2)

    def test_schema_retry_preserves_region_and_retains_raw_failure(self):
        client = FakeClient()
        invalid = decision()
        del invalid["confidence"]
        client.decide = Mock(side_effect=[json.dumps(invalid), json.dumps(decision())])
        plan, _, _ = TargetedPlanner(client).plan(self.target, seed=5)
        self.assertEqual(plan.status, "editable")
        self.assertIn("confidence", plan.response_attempts[0]["validation_error"])
        self.assertEqual(json.loads(plan.response_attempts[0]["raw_response"]), invalid)
        calls = client.decide.call_args_list
        self.assertEqual(calls[0].kwargs["seed"], calls[1].kwargs["seed"])
        self.assertEqual([pixel_sha256(i) for i in calls[0].kwargs["images"]],
                         [pixel_sha256(i) for i in calls[1].kwargs["images"]])

    def test_hard_rejection_cannot_create_prompt(self):
        plan, _, _ = TargetedPlanner(FakeClient(decision(region_type="sky"))).plan(self.target)
        self.assertEqual(plan.status, "rejected")
        self.assertEqual(plan.prompts["prompt"], "")
        self.assertTrue(plan.proposed_decision["editable"])

    def test_mask_validation_and_border_context(self):
        for mask in (Image.new("L", (80, 60)), Image.new("L", (20, 20), 1),
                     Image.new("L", (80, 60), 42), Image.new("RGB", (80, 60), "white")):
            with self.assertRaises(ValueError):
                prepare_views(self.target.source, mask)
        mask = Image.new("L", (80, 60))
        ImageDraw.Draw(mask).rectangle((0, 0, 20, 20), fill=255)
        views, box = prepare_views(self.target.source, mask)
        self.assertEqual(box, (0, 0, 32, 32))
        self.assertEqual(views[2].size, (32, 32))
        for scale in (0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                prepare_views(self.target.source, mask, scale)

    def test_source_mask_pairing_and_diagnostics(self):
        for field, value in (("sample_id", "wrong"), ("image_key", "wrong.png"),
                             ("place_key", "Other:2"), ("source_sha256", "0" * 64),
                             ("generation_mask_sha256", "0" * 64)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                TargetedInput.from_record(dict(self.record, **{field: value}), self.directory, self.original)
        swapped = dict(self.record, generation_mask_path=self.record["vulnerability_mask_path"],
                       generation_mask_sha256=self.record["vulnerability_mask_sha256"])
        with self.assertRaises(ValueError):
            TargetedInput.from_record(swapped, self.directory, self.original)
        Path(self.record["source_path"]).write_bytes(b"wrong image")
        with self.assertRaises(ValueError):
            TargetedInput.from_record(self.record, self.directory, self.original)

    def test_audit_saves_images_and_correct_counts(self):
        spec = importlib.util.spec_from_file_location("audit_targeted_planner", ROOT / "scripts/audit_targeted_planner.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        summary = module.run_audit([self.target], self.directory / "audit", TargetedPlanner(FakeClient()), seed=3)
        self.assertEqual(summary["editable_count"], 1)
        self.assertEqual(summary["rejected_count"], 0)
        self.assertEqual(summary["region_type_counts"]["road"], 1)
        self.assertTrue(summary["input_files_unchanged"])
        output = self.directory / "audit"
        row = json.loads((output / "targeted_edit_plans.jsonl").read_text())
        sample_dir = output / row["audit_directory"]
        for filename in ("source.png", "vulnerability_mask.png", "generation_mask.png", "generation_overlay.png", "context_crop.png", "planner.json", "raw_response.txt"):
            self.assertTrue((sample_dir / filename).is_file())
        self.assertEqual((sample_dir / "generation_mask.png").read_bytes(), Path(self.record["generation_mask_path"]).read_bytes())
        with self.assertRaises(FileExistsError):
            module.run_audit([self.target], output, TargetedPlanner(FakeClient()), seed=3)

    def test_openai_transports_all_views_strict_schema_and_seed(self):
        from targeted.qwen_client import OpenAIQwenClient
        client = OpenAIQwenClient.__new__(OpenAIQwenClient)
        client.model = "test-qwen"
        client.client = Mock()
        response = Mock()
        response.choices = [Mock(finish_reason="stop", message=Mock(content=json.dumps(decision())))]
        client.client.chat.completions.create.return_value = response
        views, _ = prepare_views(self.target.source, self.target.generation_mask)
        client.decide(system="system", user="user", images=views, seed=42)
        kwargs = client.client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["seed"], 42)
        self.assertEqual(kwargs["temperature"], 0)
        self.assertTrue(kwargs["response_format"]["json_schema"]["strict"])
        self.assertEqual(len(kwargs["messages"][1]["content"]), 7)
        image_parts = [x for x in kwargs["messages"][1]["content"] if x["type"] == "image_url"]
        self.assertEqual(len(image_parts), 3)
        self.assertTrue(all(x["image_url"]["url"].startswith("data:image/png;base64,") for x in image_parts))

    def test_summary_counts_schema_failures_separately_from_low_confidence(self):
        spec = importlib.util.spec_from_file_location("audit_counts", ROOT / "scripts/audit_targeted_planner.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        client = FakeClient()
        client.decide = Mock(side_effect=[json.dumps(decision()), json.dumps(decision(confidence=.2)), "not JSON"])
        planner = TargetedPlanner(client, schema_retries=0)
        summary = module.run_audit([self.target] * 3, self.directory / "counts", planner, seed=0)
        self.assertEqual(summary["status"], "COMPLETE_WITH_SCHEMA_ERRORS")
        self.assertEqual(summary["editable_count"], 1)
        self.assertEqual(summary["rejected_count"], 2)
        self.assertEqual(summary["schema_error_count"], 1)
        self.assertEqual(summary["low_confidence_count"], 1)
        self.assertEqual(sum(summary["region_type_counts"].values()), 2)
        self.assertEqual(summary["occluder_family_counts"]["none"], 1)

    def test_transport_failure_stops_without_mock_fallback(self):
        client = FakeClient()
        client.decide = Mock(side_effect=ConnectionError("service unavailable"))
        with self.assertRaises(ConnectionError):
            TargetedPlanner(client).plan(self.target)
        self.assertEqual(client.decide.call_count, 1)

    def test_core_has_no_model_or_old_pipeline_imports(self):
        code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','diffusers','transformers','openai','verification','prompts'} or fullname in {'generation.agent','generation.router','generation.lightx2v','generation.targeted_editor'}:
            raise RuntimeError('Forbidden import: ' + fullname)
sys.meta_path.insert(0, Block())
from targeted.planner import TargetedPlanner, prepare_views
from targeted.editability import parse_decision
from targeted.prompt_family import prompt_for_family
import runpy
sys.argv=['audit_targeted_planner.py','--help']
runpy.run_path('scripts/audit_targeted_planner.py',run_name='__main__')
'''
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
