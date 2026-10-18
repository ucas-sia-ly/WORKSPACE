from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.qwen_curriculum import common, finetune_all_cities as training


class TrainingInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.real_data = self.root / "gsv-cities"
        (self.real_data / "Dataframes").mkdir(parents=True)
        self.checkpoint = self.root / "dino_salad.ckpt"
        self.checkpoint.write_bytes(b"original pretrained weights")
        self.backbone = self.root / "dinov2"
        self.backbone.mkdir()
        self.cities = ["A", "B"]
        for city in self.cities:
            (self.real_data / "Dataframes" / f"{city}.csv").write_text("city,place_id\n" + city + ",1\n")
            (self.real_data / "Images" / city).mkdir(parents=True)
        files = [self.checkpoint] + list((self.real_data / "Dataframes").glob("*.csv"))
        self.request = {"schema_version": 1, "real_data": str(self.real_data), "checkpoint": str(self.checkpoint),
            "backbone_repo": str(self.backbone), "cities": self.cities,
            "files_sha256": {str(path): common.file_sha256(path) for path in files}}
        self.path = self.root / "data_config.json"
        self.write_request()
        self.campaign = {"training_inputs": str(self.path), "cities": self.cities}

    def write_request(self):
        common.write_json(self.path, {**self.request, "fingerprint": common.fingerprint(self.request)})

    def test_new_data_snapshot_loads_without_any_old_mining_path(self):
        actual = training.load_training_inputs(self.campaign)
        self.assertEqual(actual["checkpoint"], str(self.checkpoint))
        self.assertEqual(actual["cities"], self.cities)
        self.assertEqual(training.DEFAULT_OUTPUT, common.WORKSPACE_ROOT / "outputs/qwen_curriculum/state/training")

    def test_old_mining_directories_are_not_a_training_input_fallback(self):
        with self.assertRaisesRegex(ValueError, "requires a frozen training_inputs snapshot"):
            training.load_training_inputs({"mining_dir": "/old/mining"})

    def test_checkpoint_and_csv_content_changes_fail_without_old_fallback(self):
        for file in (self.checkpoint, self.real_data / "Dataframes/A.csv"):
            original = file.read_bytes()
            file.write_bytes(b"changed bytes")
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                training.load_training_inputs(self.campaign)
            file.write_bytes(original)

    def test_data_config_seal_and_missing_csv_hash_cannot_be_bypassed(self):
        invalid = json.loads(self.path.read_text())
        invalid["checkpoint"] = "/some/other/model"
        common.write_json(self.path, invalid)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            training.load_training_inputs(self.campaign)
        self.request["files_sha256"].pop(str(self.real_data / "Dataframes/A.csv"))
        self.write_request()
        with self.assertRaisesRegex(ValueError, "every city CSV"):
            training.load_training_inputs(self.campaign)

    def test_partial_city_snapshot_cannot_replace_full_generation_dataset(self):
        self.request["cities"] = ["A"]
        self.write_request()
        with self.assertRaisesRegex(ValueError, "complete generation campaign"):
            training.load_training_inputs(self.campaign)

    def test_missing_local_backbone_fails_before_training_initialization(self):
        self.backbone.rmdir()
        with self.assertRaisesRegex(ValueError, "backbone repository"):
            training.load_training_inputs(self.campaign)

    def test_unsupported_schema_or_relative_paths_fail_without_legacy_fallback(self):
        for version in (None, True, 2):
            self.request["schema_version"] = version
            self.write_request()
            with self.assertRaisesRegex(ValueError, "schema"):
                training.load_training_inputs(self.campaign)
        self.request["schema_version"] = 1
        self.request["real_data"] = "relative/gsv"
        self.write_request()
        with self.assertRaisesRegex(ValueError, "absolute strings"):
            training.load_training_inputs(self.campaign)

    def test_city_inventory_requires_all_23_csvs_and_image_directories(self):
        with self.assertRaisesRegex(ValueError, "Expected 23"):
            training.discover_cities(self.real_data)
        for number in range(21):
            city = f"City{number:02d}"
            (self.real_data / "Dataframes" / f"{city}.csv").touch()
            (self.real_data / "Images" / city).mkdir()
        expected = sorted(["A", "B"] + [f"City{number:02d}" for number in range(21)])
        self.assertEqual(training.discover_cities(self.real_data), expected)
        (self.real_data / "Images" / "A").rmdir()
        with self.assertRaisesRegex(ValueError, "Images directory"):
            training.discover_cities(self.real_data)

    def test_training_needs_approved_weather_checked_records_in_every_city(self):
        rows = [{"city": city, "passed": True, "eligible_for_training": True, "weather_ok": True}
                for city in self.cities]
        self.assertEqual(training.accepted_cities(rows, self.cities), rows)
        rows[0]["eligible_for_training"] = False
        with self.assertRaisesRegex(ValueError, "No approved.*A"):
            training.accepted_cities(rows, self.cities)
        rows[0]["eligible_for_training"] = True
        rows[0]["weather_ok"] = False
        with self.assertRaisesRegex(ValueError, "weather check"):
            training.accepted_cities(rows, self.cities)

    def test_unfinished_generation_cannot_prepare_training_or_write_a_snapshot(self):
        directory = self.root / "training"
        with patch.object(training.campaign, "load_collection", side_effect=ValueError("generation incomplete")):
            with self.assertRaisesRegex(ValueError, "generation incomplete"):
                training.prepare(directory, self.root / "state", 10)
        self.assertFalse(directory.exists())

    def test_frozen_training_validates_both_source_and_generated_image_bytes(self):
        source = self.root / "source.png"
        generated = self.root / "generated.png"
        source.write_bytes(b"source bytes")
        generated.write_bytes(b"generated bytes")
        manifest = self.root / "accepted.jsonl"
        common.write_jsonl(manifest, [{"source_path": str(source), "source_sha256": common.file_sha256(source),
            "output_path": str(generated), "output_sha256": common.file_sha256(generated)}])
        config = {"accepted_manifest": str(manifest),
            "files_sha256": {str(manifest): common.file_sha256(manifest)}}
        training.base.check_inputs(config)
        for file in (source, generated):
            original = file.read_bytes()
            file.write_bytes(b"changed bytes")
            with self.assertRaisesRegex(ValueError, "training image missing or changed"):
                training.base.check_inputs(config)
            file.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
