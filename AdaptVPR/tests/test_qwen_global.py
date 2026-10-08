"""Production Global routing uses the existing Qwen recipe and no IC service."""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from generation.agent import SceneAugmentAgent
from generation.lightx2v import Lightx2vGenerator
from generation.preflight import check_environment
from generation.router import normalize_decision


class QwenGlobalTests(unittest.TestCase):
    def test_public_constructor_never_constructs_iclight(self):
        client, evaluator = Mock(), Mock()
        with patch("generation.iclight.ICLightGenerator") as historical_ic:
            with patch("generation.agent.Lightx2vGenerator", return_value=client) as qwen:
                with patch("generation.agent.DualTraitEvaluator", return_value=evaluator):
                    agent = SceneAugmentAgent(planning_enabled=False, reflection_enabled=False)
        historical_ic.assert_not_called()
        qwen.assert_called_once_with(api_url=None)
        self.assertIs(agent.lightx2v, client)

    def test_global_preserves_prompt_and_uses_empty_negative_without_ic_denoise(self):
        agent = SceneAugmentAgent.__new__(SceneAugmentAgent)
        agent.lightx2v = Mock()
        source = Image.new("RGB", (40, 30))
        prompt = "Task: frozen weather edit. Preserve: the source. Forbidden: changed geometry."
        result = agent._generate(source, prompt, "global", decision={"weather": "rain"}, seed=987)
        agent.lightx2v.generate_global.assert_called_once_with(source, prompt, negative_prompt="", seed=987)
        self.assertIs(result, agent.lightx2v.generate_global.return_value)
        self.assertNotIn("highres_denoise", agent.lightx2v.generate_global.call_args.kwargs)

    def test_router_reports_qwen_and_does_not_append_ic_constraints(self):
        prompt = "Task: weather edit. Preserve: source identity. Forbidden: viewpoint changes."
        decision = normalize_decision({"route": "global", "weather": "night", "prompt": prompt})
        self.assertEqual(decision["selected_model"], "Qwen-Image-Edit-2511")
        self.assertEqual(decision["prompt"], prompt)

    def test_client_forwards_existing_four_step_recipe_and_restores_source_dimensions(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw.png"
            Image.new("RGB", (1472, 1104), (50, 100, 170)).save(raw)
            response = Mock()
            response.json.return_value = {"result_path": str(raw)}
            session = Mock()
            session.post.return_value = response
            client = Lightx2vGenerator.__new__(Lightx2vGenerator)
            client.api_url, client._session = "http://qwen.example/generate", session
            source = Image.new("RGB", (400, 300), (80, 130, 170))
            with patch.dict(os.environ, {"ADAPTVPR_DISABLE_MOCK": "1", "LIGHTX2V_INFER_STEPS": "4",
                                         "LIGHTX2V_GUIDANCE_SCALE": "1.0"}):
                with patch("generation.lightx2v._wait_service_ready"):
                    generated = client.generate_global(source, "released global prompt", seed=17)
            payload = session.post.call_args.kwargs["json"]
            self.assertEqual(payload["prompt"], "released global prompt")
            self.assertEqual(payload["negative_prompt"], "")
            self.assertEqual(payload["seed"], 17)
            self.assertEqual(payload["infer_steps"], 4)
            self.assertEqual(payload["guidance_scale"], 1.0)
            self.assertNotIn("highres_denoise", payload)
            self.assertFalse(Path(payload["image_path"]).exists())
            self.assertEqual(generated.size, source.size)
            self.assertTrue(raw.is_file())

    def test_strict_global_failure_cannot_fabricate_mock_image(self):
        client = Lightx2vGenerator.__new__(Lightx2vGenerator)
        with patch.object(client, "_call_api", return_value=None):
            with patch.dict(os.environ, {"ADAPTVPR_DISABLE_MOCK": "1"}):
                with self.assertRaisesRegex(RuntimeError, "Global mock fallback"):
                    client.generate_global(Image.new("RGB", (40, 30)), "prompt")


class QwenPreflightTests(unittest.TestCase):
    def preflight(self, *, with_iclight=False):
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: True)
        torch.__version__, torch.version = "test", types.SimpleNamespace(cuda="test")
        vismatch = types.ModuleType("vismatch")
        session = Mock()
        session.__enter__ = Mock(return_value=session)
        session.__exit__ = Mock(return_value=None)
        with patch.dict(os.environ, {"LIGHTX2V_API_URL": "http://qwen.example/generate"}, clear=True):
            with patch.dict(sys.modules, {"torch": torch, "vismatch": vismatch}):
                with patch("generation.preflight.load_environment"):
                    with patch("generation.preflight.importlib.util.find_spec", return_value=object()) as modules:
                        with patch("generation.preflight.requests.Session", return_value=session):
                            with patch("generation.preflight.probe_service", return_value=(True, "ready")) as probe:
                                errors = check_environment(planner=False, with_iclight=with_iclight)
        return errors, probe, modules

    def test_default_preflight_requires_only_qwen_and_no_planner_sdk(self):
        errors, probe, modules = self.preflight()
        self.assertEqual(errors, [])
        self.assertEqual(probe.call_count, 1)
        self.assertEqual(probe.call_args.args[1], "http://qwen.example/generate")
        self.assertNotIn("openai", [call.args[0] for call in modules.call_args_list])

    def test_historical_iclight_check_is_explicit_opt_in(self):
        errors, probe, _ = self.preflight(with_iclight=True)
        self.assertEqual(errors, ["ICLIGHT_API_URL is not configured"])
        self.assertEqual(probe.call_count, 1)


if __name__ == "__main__":
    unittest.main()
