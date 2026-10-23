"""CPU-only safety semantics, geometry and automatic-control calibration checks."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.qwen_curriculum import content_audit_v4 as audit
from experiments.qwen_curriculum import audit_content_v4 as cli


def fixture(value=0., *, place="city/1", label="benign"):
    row = {"sample_id": f"{place}/{label}", "place_key": place,
           "source_path": "/approved-training/source.png", "output_path": "/controls/output.png",
           "source_sha256": "source", "output_sha256": "output", "control": label,
           "control_recipe": {"programmatic": True}, "dataset": "gsv-cities",
           "origin_split": "train", "split": "calibration", "estimated_H": np.eye(3).tolist(),
           "matcher_image_size": 448, "geometry_ok": True, "weather_ok": True, "passed": True}
    pipeline = audit.pipeline_definition({"model": "mock-local"}, 448, 1)
    signals = {"sample_id": row["sample_id"], "source_sha256": "source", "output_sha256": "output",
               "pipeline": pipeline, "pipeline_sha256": audit.digest(pipeline),
               "grid_shape": [4, 4], "dino_distance": np.full((4, 4), value).tolist(),
               "structure_distance": np.full((4, 4), value).tolist(),
               "valid": np.ones((4, 4), bool).tolist(), "structure_valid": np.ones((4, 4), bool).tolist(),
               "ocr": {"available": False},
               "geometry_sha256": audit.digest({"estimated_H": row["estimated_H"], "matcher_image_size": 448})}
    return row, signals


def calibrated(separate=True):
    controls = []
    for index in range(128):
        controls.append(fixture(0., place=f"city/{index}", label="benign"))
        controls.append(fixture(1. if separate else 0., place=f"city/{index}", label="controlled_negative"))
    return audit.calibrate_controls(controls, .05, min_pairs=5)


class ContentAuditTests(unittest.TestCase):
    def test_same_image_and_translation_with_correct_homography(self):
        source = np.eye(16, dtype=np.float32).reshape(4, 4, 16)
        generated = np.zeros_like(source)
        generated[:, 1:] = source[:, :-1]
        h = np.array([[1., 0, .25], [0, 1, 0], [0, 0, 1]])
        distance, valid = audit.patch_distances(source, generated, h, radius=0)
        self.assertEqual(int(valid.sum()), 12)
        self.assertTrue(np.allclose(distance[valid], 0))
        self.assertFalse(valid[:, 0].any())
        wrong, wrong_valid = audit.patch_distances(source, generated, np.eye(3), radius=0)
        self.assertGreater(wrong[wrong_valid].mean(), .9)

    def test_resize_coordinates_and_boundary_footprints(self):
        source = np.eye(8, dtype=np.float32).reshape(2, 4, 8)
        generated = source.repeat(2, 0).repeat(2, 1)
        distance, valid = audit.patch_distances(source, generated, np.eye(3), radius=0)
        self.assertTrue(valid.all())
        self.assertTrue(np.allclose(distance, 0))
        _, shifted = audit.patch_distances(source, generated, [[1, 0, .03], [0, 1, 0], [0, 0, 1]])
        self.assertFalse(shifted[:, 0].any())  # center in-frame cannot override partial footprint.

    def test_square_matcher_h_is_normalized_without_original_dimensions_assumption(self):
        row = {"estimated_H": [[1, 0, 64], [0, 1, 0], [0, 0, 1]], "matcher_image_size": 512,
               "source_dimensions": [400, 300]}
        value = audit.normalized_homography(row)
        value /= value[2, 2]
        self.assertAlmostEqual(value[0, 2], .125)
        with self.assertRaises(ValueError):
            audit.normalized_homography({"estimated_H": np.eye(3).tolist()})
        with self.assertRaises(ValueError):
            audit.validate_homography([[1, 0, 0], [0, 1, 0], [2, 0, -1]])

    def test_local_contrast_preserves_linear_photometric_shift(self):
        array = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
        source, generated = Image.fromarray(array), Image.fromarray(array * 2 + 20)
        distance, valid = audit.structural_distances(source, generated, np.eye(3), (4, 4), 64)
        self.assertTrue(valid.all())
        self.assertLess(float(distance.mean()), .003)
        _, blank_valid = audit.structural_distances(Image.new("L", (64, 64)), Image.new("L", (64, 64)),
                                                    np.eye(3), (4, 4), 64)
        self.assertFalse(blank_valid.any())

    def test_machine_controls_separate_or_abstain_without_manual_labels(self):
        good, bad = calibrated(True), calibrated(False)
        self.assertTrue(good["usable"])
        self.assertFalse(bad["usable"])
        self.assertGreater(good["heldout_controls"]["controlled_negative"]["reject"], 0)
        row, signals = fixture()
        self.assertEqual(audit.audit_pair(row, signals, bad)["decision"], "unknown")
        paired = good["heldout_separation_test"]
        self.assertEqual(paired["name"], "exact_paired_discordance_one_sided")
        self.assertEqual(paired["benign_only_reject"], 0)
        self.assertAlmostEqual(paired["p"], 2. ** -paired["negative_only_reject"])
        negative, ns = fixture(1., label="controlled_negative")
        negative["source_sha256"] = ns["source_sha256"] = "different-source"
        with self.assertRaisesRegex(ValueError, "exact source"):
            audit.calibrate_controls([fixture(), (negative, ns)], .05, min_pairs=2)
        with self.assertRaisesRegex(ValueError, "Exactly one"):
            audit.calibrate_controls([fixture(), fixture()], .05, min_pairs=2)

    def test_missing_required_evidence_and_bad_hash_fail_closed(self):
        row, signals = fixture()
        cal = calibrated()
        for sig, artifact in ((None, cal), (signals, None)):
            result = audit.audit_pair(row, sig, artifact)
            self.assertEqual(result["decision"], "unknown")
            self.assertFalse(result["eligible_for_training"])
        signals["source_sha256"] = "different"
        self.assertEqual(audit.audit_pair(row, signals, cal)["decision"], "unknown")

    def test_brightness_and_ocr_novelty_alone_never_mean_hallucination(self):
        row, signals = fixture()
        ocr = audit.ocr_diagnostics({"source": [], "generated": [
            {"text": "newly visible existing sign", "box_xyxy_normalized": [.2, .2, .5, .4]}]}, np.eye(3))
        self.assertEqual(ocr["newly_detected"], 1)
        self.assertFalse(ocr["rejection_evidence"])
        signals["ocr"] = ocr
        result = audit.audit_pair(row, signals, calibrated())
        self.assertEqual(result["decision"], "pass")
        signals["dino_distance"] = np.ones((4, 4)).tolist()
        # Low DINO agreement without independent structural change is UNKNOWN.
        result = audit.audit_pair(row, signals, calibrated())
        self.assertEqual(result["decision"], "unknown")

    def test_base_pending_v4_is_reinstated_only_after_complete_audit(self):
        row, signals = fixture()
        row.update(base_quality_passed=True, eligible_for_training=False, content_status="pending_automated_audit")
        self.assertTrue(audit.audit_pair(row, signals, calibrated())["eligible_for_training"])
        row["prompt_variant"] = "shopfronts"
        self.assertEqual(audit.audit_pair(row, signals, calibrated())["decision"], "reject")

    def test_legacy_explicit_rejection_cannot_be_resurrected(self):
        row, signals = fixture()
        row["passed"] = False
        self.assertEqual(audit.audit_pair(row, signals, calibrated())["decision"], "unknown")

    def test_unknown_no_overlap_and_nonfinite_signals(self):
        row, signals = fixture()
        signals["valid"] = np.zeros((4, 4), bool).tolist()
        self.assertEqual(audit.audit_pair(row, signals, calibrated())["decision"], "unknown")
        signals["dino_distance"][0][0] = float("nan")
        self.assertEqual(audit.audit_pair(row, signals, calibrated())["decision"], "unknown")

    def test_eval_corpora_and_test_paths_cannot_calibrate(self):
        row, _ = fixture()
        for dataset in ("Tokyo247", "SF-XL", "RobotCar", "SVOX", "Nordland"):
            with self.subTest(dataset=dataset), self.assertRaises(ValueError):
                audit.require_training_calibration({**row, "dataset": dataset})
        for path in ("/gsv/test/a.jpg", "/dataset/tokyo247/a.jpg"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                audit.require_training_calibration({**row, "source_path": path})

    def test_export_never_rewrites_old_manifest_and_checks_real_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, generated = root / "source.png", root / "output.png"
            Image.new("RGB", (10, 10), "red").save(source)
            Image.new("RGB", (10, 10), "blue").save(generated)
            row, signals = fixture()
            row.update(source_path=str(source), output_path=str(generated),
                       source_sha256=audit.sha256(source), output_sha256=audit.sha256(generated),
                       result_sha256="historical-generation-seal")
            signals.update(source_sha256=row["source_sha256"], output_sha256=row["output_sha256"])
            manifest, signal_file, calibration = root / "old.jsonl", root / "signal.jsonl", root / "calibration.json"
            cli._new_file(manifest, [row], True)
            cli._new_file(signal_file, [signals], True)
            cli._new_file(calibration, calibrated())
            before = audit.sha256(manifest)
            Image.new("RGB", (10, 10), "green").save(generated)
            args = Namespace(manifest=str(manifest), signals=str(signal_file), calibration=str(calibration),
                             output_dir=str(root / "new"))
            cli.export_audit(args)
            self.assertEqual(before, audit.sha256(manifest))
            self.assertEqual((root / "new/training_manifest.jsonl").read_text(), "")
            verdict = json.loads((root / "new/audited_manifest.jsonl").read_text())
            self.assertEqual(verdict["content_status"], "unknown")
            self.assertNotIn("result_sha256", verdict)
            self.assertEqual(verdict["original_result_sha256"], "historical-generation-seal")
            self.assertEqual(verdict["input_row_sha256"], audit.digest(row))
            self.assertEqual(verdict["audit_record_sha256"],
                             audit.digest({k: v for k, v in verdict.items() if k != "audit_record_sha256"}))
            with self.assertRaises(FileExistsError):
                cli.export_audit(args)

    def test_automatic_controls_use_disjoint_places_and_preserve_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "training"
            training.mkdir()
            rows = []
            for index in range(32):
                source = training / f"{index}.png"
                image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
                image = np.roll(image, index, axis=1)
                Image.fromarray(image).save(source)
                rows.append({"sample_id": str(index), "city": "city", "place_id": index,
                             "source_path": str(source), "source_sha256": audit.sha256(source)})
            manifest, exclude = root / "train.jsonl", root / "excluded.json"
            cli._new_file(manifest, rows, True)
            cli._new_file(exclude, [["city", 0]])
            before = audit.sha256(manifest)
            args = Namespace(training_root=str(training), exclude_places=str(exclude),
                             manifest=str(manifest), dataset="gsv-cities", count=16,
                             seed=42, side=32, output_dir=str(root / "controls"))
            cli.create_controls(args)
            controls = cli._rows(root / "controls/controls.jsonl")
            self.assertEqual(len(controls), 32)
            self.assertEqual(before, audit.sha256(manifest))
            self.assertNotIn("city/0", {row["place_key"] for row in controls})
            source_to_key = {row["source_path"]: row["place_key"] for row in controls}
            for row in controls:
                audit.require_training_calibration(row)
                self.assertEqual(row["source_sha256"], audit.sha256(row["source_path"]))
                self.assertEqual(len(row["control_recipe"]["smooth_spatial_illumination"]), 3)
                self.assertIn("localized_clipping_gain", row["control_recipe"])
                if row["control"] == "controlled_negative":
                    donor = source_to_key[row["control_recipe"]["donor_path"]]
                    self.assertNotEqual(row["place_key"], donor)
                    own_partition = int(audit.digest(row["place_key"])[:8], 16) % 4 == 0
                    donor_partition = int(audit.digest(donor)[:8], 16) % 4 == 0
                    self.assertEqual(own_partition, donor_partition)

    def test_signal_extraction_binds_optional_ocr_to_exact_image_bytes(self):
        class Provider:
            definition = {"model": "mock-local"}
            def __call__(self, image, side):
                return np.eye(16, dtype=np.float32).reshape(4, 4, 16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            texture = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
            Image.fromarray(texture).save(path)
            row = {"sample_id": "same", "source_path": str(path), "output_path": str(path),
                   "estimated_H": np.eye(3).tolist(), "matcher_image_size": 64}
            signals = audit.pair_signals(row, Provider(), side=64)
            self.assertEqual(signals["source_sha256"], audit.sha256(path))
            self.assertTrue(np.allclose(signals["dino_distance"], 0))
            with self.assertRaises(ValueError):
                audit.pair_signals(row, Provider(), side=64,
                                   ocr_cache={"source_sha256": "stale", "output_sha256": "stale"})


if __name__ == "__main__":
    unittest.main()
