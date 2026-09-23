import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from generation.service_health import probe_service
from generation.lightx2v import _wait_service_ready
from generation.preflight import load_environment


class ServiceHealthTest(unittest.TestCase):
    def test_relative_service_paths_are_independent_of_working_directory(self):
        cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"LIGHTX2V_ROOT": "../LightX2V"}
        ), patch("dotenv.load_dotenv"):
            try:
                os.chdir(temporary)
                load_environment()
                expected = Path(__file__).resolve().parents[2] / "LightX2V"
                self.assertEqual(os.environ["LIGHTX2V_ROOT"], str(expected))
            finally:
                os.chdir(cwd)

    def session(self, payload):
        session = Mock()
        session.get.return_value.json.return_value = payload
        return session

    def test_ready_service(self):
        session = self.session({"status": "ok", "model_loaded": True, "generator_ready": True})
        self.assertTrue(probe_service(session, "http://localhost:8001/generate")[0])
        session.get.assert_called_once_with("http://localhost:8001/health", timeout=2)

    def test_legacy_health_without_generator_flag(self):
        self.assertTrue(probe_service(self.session({"status": "ok", "model_loaded": True}), "http://x/generate")[0])

    def test_generator_must_be_ready(self):
        session = self.session({"status": "ok", "model_loaded": True, "generator_ready": False})
        ready, detail = probe_service(session, "http://x/generate")
        self.assertFalse(ready)
        self.assertIn("generator_ready=False", detail)

    def test_loading_error_is_reported_without_sleeping(self):
        session = self.session({"status": "error", "error": "CUDA out of memory"})
        with patch("generation.lightx2v.time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
                _wait_service_ready(session, "http://x/generate")
            sleep.assert_not_called()

    def test_connection_failure_preserves_reason(self):
        session = Mock()
        session.get.side_effect = requests.ConnectionError("Connection refused")
        ready, detail = probe_service(session, "http://x/generate")
        self.assertFalse(ready)
        self.assertIn("Connection refused", detail)

    def test_invalid_json_shape(self):
        self.assertFalse(probe_service(self.session([]), "http://x/generate")[0])

    def test_http_failure_is_not_ready(self):
        session = self.session({})
        session.get.return_value.raise_for_status.side_effect = requests.HTTPError("503 unavailable")
        self.assertIn("503", probe_service(session, "http://x/generate")[1])

    def test_default_wait_is_bounded(self):
        session = Mock()
        session.get.side_effect = requests.ConnectionError("Connection refused")
        with patch.dict(os.environ, {}, clear=True), patch(
            "generation.lightx2v.time.monotonic", side_effect=[0, 601]
        ), patch("generation.lightx2v.time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "after 600s.*Connection refused"):
                _wait_service_ready(session, "http://x/generate")
            sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
