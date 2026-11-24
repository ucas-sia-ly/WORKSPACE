"""CPU/cache-only v4.2 regression checks; no model or real-pair tuning."""
from __future__ import annotations

import copy
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

from experiments.qwen_curriculum import content_audit_v4 as legacy
from experiments.qwen_curriculum import content_audit_v42 as decision
from experiments.qwen_curriculum import audit_content_v42 as cli


def controls(separate=True):
    pipeline = legacy.pipeline_definition({"architecture": "dinov2_vitb14", "normalization": "ImageNet",
        "checkpoint_sha256": "a" * 64, "repository_python_sha256": "b" * 64,
        "torch_version": "mock_cpu"}, 448, 1)
    pairs = []
    for i in range(64):
        for label in ("benign", "controlled_negative"):
            row = {"sample_id": f"city/{i}/{label}", "place_key": f"city/{i}", "control": label,
                   "source_path": f"/approved-training/{i}.png", "output_path": f"/machine-controls/{i}_{label}.png",
                   "source_sha256": f"source{i}", "output_sha256": f"output{i}{label}",
                   "control_recipe": {"programmatic_benign": True}, "dataset": "gsv-cities",
                   "split": "calibration", "origin_split": "train", "estimated_H": np.eye(3).tolist(),
                   "matcher_image_size": 448, "geometry_ok": True, "weather_ok": True, "passed": True}
            array = np.full((32, 32), .05 + i / 1000, dtype=float)
            if label == "controlled_negative":
                row["control_recipe"]["insertion_xyxy"] = [112, 112, 224, 224]
                if separate:
                    array[8:16, 8:16] = 1.2
            signal = {"sample_id": row["sample_id"], "source_sha256": row["source_sha256"],
                      "output_sha256": row["output_sha256"], "pipeline": pipeline,
                      "pipeline_sha256": legacy.digest(pipeline), "grid_shape": [32, 32],
                      "dino_distance": array.tolist(), "structure_distance": array.tolist(),
                      "valid": np.ones((32, 32), bool).tolist(),
                      "structure_valid": np.ones((32, 32), bool).tolist(),
                      "geometry_sha256": legacy.digest({"estimated_H": row["estimated_H"], "matcher_image_size": 448})}
            pairs.append((row, signal))
    return pairs


class V42Tests(unittest.TestCase):
    def test_local_joint_detects_insertions_without_global_area_conjunction(self):
        pairs = controls()
        artifact = decision.calibrate_controls_v42(pairs, .05, min_pairs=3)
        self.assertTrue(artifact["usable"])
        self.assertEqual(artifact["heldout_controls"]["benign"]["reject"], 0)
        self.assertEqual(artifact["heldout_controls"]["controlled_negative"]["pass"], 0)
        self.assertEqual(artifact["heldout_controls"]["controlled_negative"]["reject"],
                         artifact["heldout_controls"]["controlled_negative"]["n"])
        # Reject threshold is the actual FIT benign maximum, not a tuned lower limit.
        fit_benign = [decision.local_joint_summary(s, artifact["rules"]["window_fractions"])["score"]
                      for r, s in pairs if r["control"] == "benign" and decision._partition(r["place_key"]) == "fit"]
        self.assertEqual(artifact["rules"]["thresholds"]["reject_above"], max(fit_benign))
        self.assertEqual(decision.audit_pair_v42(*pairs[1], artifact)["decision"], "reject")

    def test_heldout_changes_cannot_change_fit_rules_only_veto_detector(self):
        pairs = controls()
        original = decision.calibrate_controls_v42(pairs, .05, min_pairs=3)
        changed = copy.deepcopy(pairs)
        for row, signals in changed:
            if decision._partition(row["place_key"]) == "heldout":
                signals["dino_distance"] = np.zeros((32, 32)).tolist()
                signals["structure_distance"] = np.zeros((32, 32)).tolist()
        after = decision.calibrate_controls_v42(changed, .05, min_pairs=3)
        self.assertEqual(original["rules"], after["rules"])
        self.assertEqual(original["frozen_fit_rules_sha256"], after["frozen_fit_rules_sha256"])
        self.assertFalse(after["usable"])

    def test_unseparated_controls_abstain_not_accept_all(self):
        pairs = controls(False)
        artifact = decision.calibrate_controls_v42(pairs, .05, min_pairs=3)
        self.assertFalse(artifact["usable"])
        self.assertEqual(decision.audit_pair_v42(*pairs[0], artifact)["decision"], "unknown")

    def test_only_dino_or_ocr_change_cannot_reject(self):
        pairs = controls()
        artifact = decision.calibrate_controls_v42(pairs, .05, min_pairs=3)
        row, signals = copy.deepcopy(pairs[0])
        signals["dino_distance"] = np.full((32, 32), 1.8).tolist()
        signals["ocr"] = {"available": True, "newly_detected": 20}
        self.assertEqual(decision.audit_pair_v42(row, signals, artifact)["decision"], "pass")
        signals["structure_distance"] = np.full((32, 32), 1.8).tolist()
        self.assertEqual(decision.audit_pair_v42(row, signals, artifact)["decision"], "reject")

    def test_missing_nonfinite_coverage_and_incompatible_caches_fail_closed(self):
        pairs = controls()
        artifact = decision.calibrate_controls_v42(pairs, .05, min_pairs=3)
        row, signals = copy.deepcopy(pairs[0])
        self.assertEqual(decision.audit_pair_v42(row, None, artifact)["decision"], "unknown")
        for key, value in (("valid", np.zeros((32, 32), bool).tolist()),
                           ("dino_distance", np.full((32, 32), np.nan).tolist()),
                           ("source_sha256", "stale")):
            altered = copy.deepcopy(signals)
            altered[key] = value
            self.assertEqual(decision.audit_pair_v42(row, altered, artifact)["decision"], "unknown")
        altered = copy.deepcopy(signals)
        altered["pipeline"]["version"] = "unrecognized"
        altered["pipeline_sha256"] = legacy.digest(altered["pipeline"])
        self.assertEqual(decision.audit_pair_v42(row, altered, artifact)["decision"], "unknown")

    def test_training_pair_and_known_insertion_geometry_required(self):
        pairs = controls()
        broken = copy.deepcopy(pairs)
        broken[1][0]["source_sha256"] = broken[1][1]["source_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "identical source"):
            decision.calibrate_controls_v42(broken, .05, min_pairs=3)
        broken = copy.deepcopy(pairs)
        for row, _ in broken:
            if row["control"] == "controlled_negative":
                row["control_recipe"].pop("insertion_xyxy")
        with self.assertRaisesRegex(ValueError, "known insertion"):
            decision.calibrate_controls_v42(broken, .05, min_pairs=3)
        pairs[0][0]["dataset"] = "Tokyo247"
        with self.assertRaises(ValueError):
            decision.calibrate_controls_v42(pairs, .05, min_pairs=3)

    def test_code_guard_pending_base_and_legacy_rejection(self):
        pairs = controls()
        artifact = decision.calibrate_controls_v42(pairs, .05, min_pairs=3)
        row, signals = copy.deepcopy(pairs[0])
        row.update(passed=False, eligible_for_training=False, base_quality_passed=True)
        self.assertTrue(decision.audit_pair_v42(row, signals, artifact)["eligible_for_training"])
        row.pop("base_quality_passed")
        self.assertFalse(decision.audit_pair_v42(row, signals, artifact)["eligible_for_training"])
        artifact["decision_implementation_sha256"] = "modified"
        artifact["artifact_sha256"] = legacy.digest({k: v for k, v in artifact.items() if k != "artifact_sha256"})
        row["base_quality_passed"] = True
        self.assertEqual(decision.audit_pair_v42(row, signals, artifact)["decision"], "unknown")

    def test_export_new_seals_and_no_old_result_mutation(self):
        pairs = controls()
        artifact = decision.calibrate_controls_v42(pairs, .05, min_pairs=3)
        row, signals = copy.deepcopy(pairs[0])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "same.png"
            Image.new("RGB", (10, 10), "red").save(image)
            row.update(source_path=str(image), output_path=str(image), source_sha256=legacy.sha256(image),
                       output_sha256=legacy.sha256(image), result_sha256="historical")
            signals.update(source_sha256=row["source_sha256"], output_sha256=row["output_sha256"])
            manifest, cache, cal = root / "input.jsonl", root / "signals.jsonl", root / "cal.json"
            cli._new_file(manifest, [row], True)
            cli._new_file(cache, [signals], True)
            cli._new_file(cal, artifact)
            original_sha = legacy.sha256(manifest)
            cli.export(Namespace(manifest=str(manifest), signals=str(cache), calibration=str(cal),
                                 output_dir=str(root / "new")))
            result = json.loads((root / "new/training_manifest.jsonl").read_text())
            self.assertEqual(result["original_result_sha256"], "historical")
            self.assertNotIn("result_sha256", result)
            self.assertEqual(result["audit_record_sha256"], legacy.digest({k: v for k, v in result.items()
                                                                          if k != "audit_record_sha256"}))
            self.assertEqual(original_sha, legacy.sha256(manifest))


if __name__ == "__main__":
    unittest.main()
