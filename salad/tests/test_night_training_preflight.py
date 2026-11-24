"""No GPU/model/subprocess required for audited night training command plans."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

from train_salad import parse_args
from workflow.night_training_preflight import build_training_preflight


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NightTrainingPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.real = self.root / "gsv"
        (self.real / "Dataframes").mkdir(parents=True)
        (self.real / "Dataframes" / "Town.csv").write_text("place_id\n")
        (self.real / "Images" / "Town").mkdir(parents=True)
        self.backbone = self.root / "repo"; self.backbone.mkdir()
        (self.backbone / "hubconf.py").write_text("# Never imported by preflight\n")
        self.checkpoint = self.root / "checkpoint.pt"; self.checkpoint.write_bytes(b"not loaded by preflight")
        self.excluded = self.root / "excluded.json"; self.excluded.write_text("[[\"Town\",99]]")
        self.profile = self.root / "camera.json"
        self.profile.write_text(json.dumps(dict(version=1,status="exploratory",conditions=["night","rainy_night"],
            probability=1.,operations={"gain_gamma":dict(gain=[1.,1.3],gamma=[.8,1.])})))
        self.audit = self.root / "audit"; self.audit.mkdir()
        self.manifest, self.report = self.audit / "training_manifest.jsonl", self.audit / "report.json"
        self.calibration = self.root / "calibration.json"
        provider = dict(checkpoint_sha256=sha(self.checkpoint), repository_python_sha256=digest(
            {"hubconf.py":sha(self.backbone / "hubconf.py")}))
        pipeline = dict(version="content_safety_v4.1",provider=provider,
                        mapping="inverse_full_patch_footprint",
                        structure="aligned_local_standardized_gray_cosine_v1")
        rules = dict(version="content_safety_v4.2",window_fractions=[.125,.25],
            thresholds={"reject_above":.3,"pass_at_most":.2,
                        "min_valid_fraction":.8,"min_structure_fraction":.2},
            score="max_local_mean_min_dino_structure",fit_places=64)
        implementation_path = Path(__file__).resolve().parents[2] / \
            "AdaptVPR/experiments/qwen_curriculum/content_audit_v42.py"
        self.calibration_data = dict(version="content_safety_v4.2",usable=True,
            feature_pipeline=pipeline,feature_pipeline_sha256=digest(pipeline),
            decision_implementation_sha256=sha(implementation_path),
            rules=rules,frozen_fit_rules_sha256=digest(rules),
            control_provenance=[dict(place_key="Town/10",partition="fit"),
                                dict(place_key="Town/11",partition="heldout")],
            holdout_limit="reused_v4.1_control_split_exploratory_not_fresh_external_validation")
        self.calibration_data["artifact_sha256"] = digest(self.calibration_data)
        self.calibration.write_text(json.dumps(self.calibration_data))
        self.rows = []
        for place in range(3):
            source = self.real / "Images" / "Town" / f"Town_{place:07d}_source.jpg"
            output = self.root / f"generated_{place}.jpg"
            Image.new("RGB",(16,24),(50,60,70)).save(source)
            Image.new("RGB",(16,24),(10,20,30)).save(output)
            row = dict(sample_id=f"sample{place}",city="Town",place_id=place,
                source_path=str(source),output_path=str(output),source_sha256=sha(source),output_sha256=sha(output),
                condition="night",prompt_variant="led_lamps",passed=True,eligible_for_training=True,
                base_quality_passed=True,content_status="pass",content_audit=dict(
                    version="content_safety_v4.2",decision="pass",eligible_for_training=True,
                    calibration_sha256=self.calibration_data["artifact_sha256"]))
            self.rows.append(row)
        self.save()

    def save(self, reseal=True):
        for row in self.rows:
            if reseal:
                row["audit_record_sha256"] = digest({k:v for k,v in row.items() if k!="audit_record_sha256"})
        self.manifest.write_text("".join(json.dumps(row)+"\n" for row in self.rows))
        self.report.write_text(json.dumps(dict(training_manifest_sha256=sha(self.manifest),
            calibration_sha256=sha(self.calibration),counts=dict(**{"pass":len(self.rows)}))))

    def run_preflight(self, **kw):
        return build_training_preflight(manifest=self.manifest,exclude_places=self.excluded,real_root=self.real,
            checkpoint=self.checkpoint,backbone_repo=self.backbone,profile=self.profile,
            output_root=self.root/"planned",python=sys.executable,batch_size=4,synthetic_place_slots=2,**kw)

    def assert_blocked(self, report, text):
        self.assertEqual(report["status"],"blocked")
        self.assertEqual(report["commands"],[])
        self.assertEqual(report["data_check_commands"],[])
        self.assertIn(text," ".join(report["errors"]))

    def test_ready_publishes_fair_cpu_checked_commands_without_output_creation(self):
        result = self.run_preflight()
        self.assertEqual(result["status"],"ready_for_data_check",result["errors"])
        self.assertEqual(result["counts"]["distinct_synthetic_places"],3)
        self.assertEqual(result["camera_profile"]["profile_sha256"],sha(self.profile))
        self.assertFalse((self.root/"planned").exists())
        configs = {row["method"]:parse_args(row["argv"][3:]) for row in result["commands"]}
        self.assertEqual(configs["A"].synthetic_mode,"mix")
        self.assertEqual(configs["A"].synthetic_fraction,0)
        self.assertEqual(configs["M"].synthetic_mode,"replace")
        self.assertEqual(configs["M_camera"].synthetic_fraction,1)
        self.assertIsNone(configs["M"].synthetic_camera_profile)
        self.assertEqual(configs["M_camera"].synthetic_camera_profile,self.profile)
        for name in ("seed","max_batches_per_epoch","stop_after_epoch","epochs","init_checkpoint",
                     "synthetic_place_slots","prefer_synthetic_sources","learning_rate","num_trainable_blocks"):
            self.assertEqual(len({str(getattr(a,name)) for a in configs.values()}),1)
        self.assertEqual(result["gpu_status"],"not_checked_cpu_only_preflight")
        self.assertEqual(result["calibration_controls"]["distinct_places"],2)
        self.assertEqual(result["checks"]["calibration_control_place_exclusion"],"verified")
        self.assertTrue(any("reuse the earlier v4.1 split" in w for w in result["warnings"]))
        self.assertTrue(any("not complete semantic validation" in w for w in result["warnings"]))

    def test_unknown_or_ineligible_rows_never_leak_even_with_valid_seals(self):
        for key,value in (("passed",False),("content_status","unknown"),("condition","unknown"),
                          ("prompt_variant","shopfronts"),("place_id",1.2)):
            with self.subTest(key=key):
                old=self.rows[0][key]; self.rows[0][key]=value; self.save()
                self.assertEqual(self.run_preflight()["status"],"blocked")
                self.rows[0][key]=old
        self.save()

    def test_source_hash_seal_receipt_and_checkpoint_tampering(self):
        self.rows[0]["condition"]="rainy_night"; self.save(reseal=False)
        self.assert_blocked(self.run_preflight(),"audit record seal")
        self.save()
        self.checkpoint.write_bytes(b"stale checkpoint")
        self.assert_blocked(self.run_preflight(),"checkpoint has changed")
        self.checkpoint.write_bytes(b"not loaded by preflight")
        Path(self.rows[0]["output_path"]).write_bytes(b"changed image")
        self.assert_blocked(self.run_preflight(),"output_sha256")

    def test_heldout_leakage_and_insufficient_distinct_places(self):
        self.excluded.write_text('[["Town",0]]')
        self.assert_blocked(self.run_preflight(),"excluded calibration/validation place")
        self.excluded.write_text('[["Town",99]]')
        self.rows=self.rows[:1];self.save()
        self.assert_blocked(self.run_preflight(),"at least 2 distinct")

    def test_empty_unusable_and_output_reuse_block(self):
        output=self.root/"planned"/"M_camera";output.mkdir(parents=True)
        self.assert_blocked(self.run_preflight(),"already exists")
        output.rmdir();output.parent.rmdir()
        self.rows=[];self.save()
        self.assert_blocked(self.run_preflight(),"manifest is empty")
        self.calibration_data["usable"]=False
        self.calibration_data["artifact_sha256"]=digest({k:v for k,v in self.calibration_data.items() if k!="artifact_sha256"})
        self.calibration.write_text(json.dumps(self.calibration_data));self.save()
        self.assert_blocked(self.run_preflight(),"calibration is unusable")

    def test_profile_is_not_silently_missing_and_recipe_rejects_bad_budget(self):
        self.profile.write_text('{}')
        self.assert_blocked(self.run_preflight(),"version must be 1")
        self.assert_blocked(self.run_preflight(smoke_batches=0),"budget")

    def test_legacy_v41_pipeline_schema_still_validates(self):
        data=self.calibration_data
        data["version"]="content_safety_v4.1"
        data["pipeline"]=data.pop("feature_pipeline")
        data["pipeline_sha256"]=data.pop("feature_pipeline_sha256")
        for field in ("decision_implementation_sha256","rules","frozen_fit_rules_sha256"):
            data.pop(field)
        data["artifact_sha256"]=digest({k:v for k,v in data.items() if k!="artifact_sha256"})
        self.calibration.write_text(json.dumps(data))
        for row in self.rows:
            row["content_audit"].update(version=data["version"],calibration_sha256=data["artifact_sha256"])
        self.save()
        self.assertEqual(self.run_preflight()["status"],"ready_for_data_check")

    def test_v42_code_and_fit_rule_seals_cannot_be_bypassed_by_resealing_artifact(self):
        for field in ("decision_implementation_sha256","frozen_fit_rules_sha256"):
            old=self.calibration_data[field]
            self.calibration_data[field]="0"*64
            self.calibration_data["artifact_sha256"]=digest({k:v for k,v in self.calibration_data.items() if k!="artifact_sha256"})
            self.calibration.write_text(json.dumps(self.calibration_data));self.save()
            expected="decision implementation" if field=="decision_implementation_sha256" else "FIT decision rules"
            self.assert_blocked(self.run_preflight(),expected)
            self.calibration_data[field]=old

    def test_current_exclusions_reject_fit_and_heldout_controls_even_when_training_places_are_disjoint(self):
        # None of the generated training rows belong to control places 10/11.
        # A new exclusion protocol still cannot reuse either control partition.
        for place in (10,11):
            with self.subTest(place=place):
                self.excluded.write_text(json.dumps([["Town",place]]))
                self.assert_blocked(self.run_preflight(),"Audit control place overlaps")

    def test_missing_or_malformed_control_provenance_cannot_be_resealed_into_eligibility(self):
        for controls in ([],[dict(place_key="Town/not-an-id",partition="fit")],
                         [dict(place_key="Town/10",partition="test")]):
            with self.subTest(controls=controls):
                self.calibration_data["control_provenance"]=controls
                self.calibration_data["artifact_sha256"]=digest({k:v for k,v in self.calibration_data.items() if k!="artifact_sha256"})
                self.calibration.write_text(json.dumps(self.calibration_data));self.save()
                self.assertEqual(self.run_preflight()["status"],"blocked")


if __name__ == "__main__":unittest.main()
