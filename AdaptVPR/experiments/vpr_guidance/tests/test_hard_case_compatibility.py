"""Check the evaluation -> extraction -> generator-data manifest contract."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


GUIDANCE_ROOT = Path(__file__).resolve().parents[1]
if str(GUIDANCE_ROOT) not in sys.path:
    sys.path.insert(0, str(GUIDANCE_ROOT))

from extract_hard_cases import extract_hard_cases_from_salad_eval
from hard_cases import filter_cases_with_source, load_hard_cases


class HardCaseCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.gsv_root = self.root / "gsv"
        self.source = self.gsv_root / "Images" / "Bangkok" / "source.jpg"
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b"source-path fixture")

    def extract(self, **overrides):
        query = {
            "query_id": "external-benchmark/query.jpg",
            "query_path": "/benchmark/query.jpg",
            "ground_truth": "reference/correct.jpg",
            "predicted": "reference/wrong.jpg",
            "rank": 17,
            "distance_pred": 0.1,
            "distance_gt": 0.9,
        }
        query.update(overrides)
        evaluation = self.root / "evaluation.json"
        evaluation.write_text(
            json.dumps({"dataset": "custom", "error_queries": [query]}),
            encoding="utf-8",
        )
        hard_cases = self.root / "hard_cases.json"
        with contextlib.redirect_stdout(io.StringIO()):
            extract_hard_cases_from_salad_eval(evaluation, hard_cases)
        return hard_cases

    def test_explicit_source_and_rank_survive_full_pipeline(self):
        manifest = self.extract(source_id="Bangkok/source.jpg")
        cases = load_hard_cases(manifest, self.gsv_root)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].source_path, str(self.source))
        self.assertEqual(cases[0].retrieval_rank, 17)
        self.assertEqual(cases[0].distance_to_wrong, 0.1)
        self.assertEqual(cases[0].distance_to_correct, 0.9)
        self.assertEqual(filter_cases_with_source(cases), cases)

    def test_missing_external_source_is_not_invented(self):
        manifest = self.extract()
        record = json.loads(manifest.read_text())["error_cases"][0]
        self.assertIsNone(record["source_id"])
        cases = load_hard_cases(manifest, self.gsv_root)
        self.assertIsNone(cases[0].source_path)
        self.assertEqual(filter_cases_with_source(cases), [])

    def test_null_and_invalid_source_ids_are_safe(self):
        for source_id in (None, "", "  ", 42, [], str(self.source), "../outside.jpg"):
            with self.subTest(source_id=source_id):
                cases = load_hard_cases(self.extract(source_id=source_id), self.gsv_root)
                self.assertIsNone(cases[0].source_path)
                self.assertEqual(filter_cases_with_source(cases), [])

    def test_rank_falls_back_to_legacy_field(self):
        manifest = self.extract(source_id="Bangkok/source.jpg")
        data = json.loads(manifest.read_text())
        record = data["error_cases"][0]
        record.pop("retrieval_rank")
        record["rank"] = 23
        manifest.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(load_hard_cases(manifest)[0].retrieval_rank, 23)


if __name__ == "__main__":
    unittest.main()
