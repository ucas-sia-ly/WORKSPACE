"""Exercise the real launcher with short-lived adapters; never load models."""

import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import uuid


ROOT = Path(__file__).resolve().parents[1]


class GenerationServiceStartupTest(unittest.TestCase):
    def check_launcher(self, use_tmux, wait=False, existing_server=False, has_lora=True):
        with tempfile.TemporaryDirectory(prefix="adaptvpr-startup-") as directory:
            temporary = Path(directory)
            project = temporary / "AdaptVPR's test"
            for name in ("scripts", "generation", "adapters"):
                (project / name).mkdir(parents=True)
            shutil.copy2(ROOT / "scripts/start_generation_services.sh", project / "scripts")
            for name in ("__init__.py", "preflight.py", "service_health.py"):
                shutil.copy2(ROOT / "generation" / name, project / "generation")
            for name in ("IC-Light", "LightX2V"):
                (temporary / name).mkdir()
            for name in ("iclight_sd15_fc.py", "lightx2v_qwen_image_edit.py"):
                (project / "adapters" / name).write_text(
                    "import json, os\n"
                    "print(json.dumps({key: os.environ.get(key) for key in "
                    "['LIGHTX2V_ROOT', 'LIGHTX2V_MODEL_PATH', 'LIGHTX2V_DISK_OFFLOAD', 'ADAPTVPR_LORA_CHECKPOINT', 'PYTHONPATH', 'PLATFORM']}), flush=True)\n"
                    + ("from http.server import HTTPServer, BaseHTTPRequestHandler\n"
                       "class Handler(BaseHTTPRequestHandler):\n"
                       " def log_message(self, *args): pass\n"
                       " def do_GET(self):\n"
                       "  self.send_response(200); self.end_headers()\n"
                       "  self.wfile.write(b'{\"status\":\"ok\",\"model_loaded\":true,\"generator_ready\":true}')\n"
                       f"server=HTTPServer(('127.0.0.1', int(os.environ['{'ICLIGHT_PORT' if name.startswith('iclight') else 'LIGHTX2V_PORT'}'])), Handler)\n"
                       "server.timeout=5\nserver.handle_request()\nserver.server_close()\n" if wait else "")
                )
            (project / ".env").write_text(
                "ICLIGHT_ROOT=../IC-Light\nLIGHTX2V_ROOT=../LightX2V\n"
                "LIGHTX2V_MODEL_PATH=../model-from-dotenv\nLIGHTX2V_DISK_OFFLOAD=1\n"
                + ("ADAPTVPR_LORA_CHECKPOINT=outputs/round_0/lora/lora_final.safetensors\n" if has_lora else "")
            )
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith(("ICLIGHT_", "LIGHTX2V_", "ADAPTVPR_"))}
            sockets = [socket.socket(), socket.socket()]
            try:
                for sock in sockets:
                    sock.bind(("127.0.0.1", 0))
                ports = [sock.getsockname()[1] for sock in sockets]
            finally:
                for sock in sockets:
                    sock.close()
            env.update({
                "ADAPTVPR_HEALTHCHECK_PYTHON": sys.executable,
                "ADAPTVPR_DISABLE_TMUX": "0" if use_tmux else "1",
                "ICLIGHT_PORT": str(ports[0]), "LIGHTX2V_PORT": str(ports[1]),
                "ADAPTVPR_SERVICE_READY_TIMEOUT": "3",
            })
            tmux = shutil.which("tmux")
            server = "adaptvpr-test-" + uuid.uuid4().hex
            if use_tmux:
                config = temporary / "tmux.conf"
                config.write_text("set-option -g default-shell " + shlex.quote(shutil.which("fish")) + "\n")
                wrapper = temporary / "bin/tmux"
                wrapper.parent.mkdir()
                wrapper.write_text(
                    "#!/bin/bash\nexec " + shlex.join([tmux, "-L", server, "-f", str(config)]) + ' "$@"\n'
                )
                wrapper.chmod(0o755)
                env["PATH"] = str(wrapper.parent) + os.pathsep + env["PATH"]
            try:
                if existing_server:
                    stale = dict(env, LIGHTX2V_MODEL_PATH="/stale/path", LIGHTX2V_DISK_OFFLOAD="0",
                                 ADAPTVPR_LORA_CHECKPOINT="/stale/lora.safetensors")
                    subprocess.run([tmux, "-L", server, "-f", str(config), "new-session", "-d", "-s", "keeper",
                                    "/bin/sleep", "8"], env=stale, check=True, timeout=3)
                result = subprocess.run(
                    ["bash", str(project / "scripts/start_generation_services.sh"), *(["--wait"] if wait else [])],
                    cwd=temporary, env=env, capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                if wait:
                    self.assertIn("ICLIGHT ready", result.stdout)
                    self.assertIn("LIGHTX2V ready", result.stdout)
                logs = [project / "tmp/logs" / f"{name}_{port}.log"
                        for name, port in zip(("iclight", "lightx2v"), ports)]
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    if all(log.exists() and log.stat().st_size for log in logs):
                        break
                    time.sleep(0.05)
                for log, upstream in zip(logs, ("IC-Light", "LightX2V")):
                    self.assertTrue(log.exists(), f"Missing {log}: {result.stdout} {result.stderr}")
                    data = json.loads(log.read_text())
                    self.assertEqual(data["LIGHTX2V_ROOT"], str(temporary / "LightX2V"))
                    self.assertEqual(data["LIGHTX2V_MODEL_PATH"], str(temporary / "model-from-dotenv"))
                    self.assertEqual(data["PYTHONPATH"].split(os.pathsep)[0], str(temporary / upstream))
                    self.assertEqual(data["PLATFORM"], "cuda")
                    self.assertEqual(data["LIGHTX2V_DISK_OFFLOAD"], "1")
                    expected_lora = str(project / "outputs/round_0/lora/lora_final.safetensors") if has_lora else ""
                    self.assertEqual(data["ADAPTVPR_LORA_CHECKPOINT"], expected_lora)
            finally:
                if use_tmux:
                    subprocess.run([tmux, "-L", server, "kill-server"], capture_output=True, timeout=3)

    def test_nohup_launches_adapters_with_dotenv_paths(self):
        self.check_launcher(use_tmux=False)

    @unittest.skipUnless(shutil.which("tmux") and shutil.which("fish"), "requires tmux and fish")
    def test_tmux_launches_adapters_when_default_shell_is_fish(self):
        self.check_launcher(use_tmux=True)

    @unittest.skipUnless(shutil.which("tmux") and shutil.which("fish"), "requires tmux and fish")
    def test_existing_tmux_server_gets_updated_model_settings_and_waits(self):
        self.check_launcher(use_tmux=True, wait=True, existing_server=True)

    @unittest.skipUnless(shutil.which("tmux") and shutil.which("fish"), "requires tmux and fish")
    def test_existing_tmux_server_clears_unconfigured_lora_checkpoint(self):
        self.check_launcher(use_tmux=True, existing_server=True, has_lora=False)


if __name__ == "__main__":
    unittest.main()
