"""Bounded demo diagnostics: no weight loading, downloads, or inference."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlparse

import requests

from generation.service_health import probe_service


ROOT = Path(__file__).resolve().parents[1]
PATH_KEYS = (
    "ICLIGHT_ROOT", "ICLIGHT_BASE_MODEL_PATH", "ICLIGHT_MODEL_PATH",
    "LIGHTX2V_ROOT", "LIGHTX2V_MODEL_PATH", "LIGHTX2V_LORA_PATH", "VISMATCH_ROOT",
    "LIGHTX2V_DISK_MODEL_PATH",
)


def load_environment() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    for key in PATH_KEYS:
        if os.getenv(key):
            path = Path(os.environ[key]).expanduser()
            os.environ[key] = str((ROOT / path).resolve())


def check_environment(*, planner: bool = True) -> list[str]:
    """Return all actionable failures instead of stopping at the first one."""
    load_environment()
    errors = []
    for module in ("openai", "torch", "transformers", "PIL"):
        if importlib.util.find_spec(module) is None:
            errors.append(f"{sys.executable}: missing Python package {module}")
    if importlib.util.find_spec("torch"):
        import torch

        if not torch.cuda.is_available():
            errors.append(
                f"{sys.executable}: CUDA unavailable (torch={torch.__version__}, "
                f"built for CUDA={torch.version.cuda}); check the GPU driver/PyTorch build"
            )
    matcher_root = os.getenv("VISMATCH_ROOT")
    if matcher_root and matcher_root not in sys.path:
        sys.path.insert(0, matcher_root)
    try:
        import vismatch  # noqa: F401
    except Exception as exc:
        errors.append(f"VisMatch import failed: {type(exc).__name__}: {exc}")

    with requests.Session() as session:
        session.trust_env = False
        for name in ("ICLIGHT", "LIGHTX2V"):
            url = os.getenv(f"{name}_API_URL", "")
            if not url:
                errors.append(f"{name}_API_URL is not configured")
                continue
            try:
                ready, detail = probe_service(session, url)
            except RuntimeError as exc:
                ready, detail = False, str(exc)
            if ready:
                continue
            errors.append(f"{name} not ready: {detail}")
            if urlparse(url).hostname not in {"127.0.0.1", "localhost", "0.0.0.0"}:
                continue
            for key in PATH_KEYS:
                if key == "LIGHTX2V_DISK_MODEL_PATH" and os.getenv("LIGHTX2V_DISK_OFFLOAD", "0").lower() not in {"1", "true", "yes"}:
                    continue
                if key.startswith(name + "_"):
                    value = os.getenv(key, "")
                    if not value or not Path(value).exists():
                        errors.append(f"{key} does not exist: {value or '(unset)'}")
            if name == "LIGHTX2V" and os.getenv("LIGHTX2V_DISK_OFFLOAD", "0").lower() in {"1", "true", "yes"}:
                from generation.qwen_disk_weights import validate_prepared_weights

                try:
                    validate_prepared_weights(os.environ["LIGHTX2V_MODEL_PATH"], os.environ["LIGHTX2V_LORA_PATH"],
                                              os.environ.get("LIGHTX2V_DISK_MODEL_PATH", ""))
                except (OSError, ValueError, RuntimeError, KeyError) as exc:
                    errors.append(f"LightX2V disk weights: {exc}")
            python = os.getenv(f"{name}_PYTHON", sys.executable)
            try:
                result = subprocess.run(
                    [python, "-c", "import diffusers, fastapi, uvicorn, accelerate"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode:
                    detail = (result.stderr or result.stdout).strip()
                    detail = detail.splitlines()[-1] if detail else f"exit code {result.returncode}"
                    errors.append(f"{name} dependencies ({python}): {detail}")
            except (OSError, subprocess.TimeoutExpired) as exc:
                errors.append(f"{name} Python check failed: {exc}")
            if name == "LIGHTX2V" and os.getenv("LIGHTX2V_ROOT"):
                # Keep diagnostics usable before FastAPI/service dependencies exist.
                import ast

                source_tree = ast.parse((ROOT / "adapters/lightx2v_qwen_image_edit.py").read_text())
                LIGHTX2V_COMMIT = next(
                    ast.literal_eval(node.value) for node in source_tree.body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "LIGHTX2V_COMMIT" for t in node.targets)
                )

                source = Path(os.environ["LIGHTX2V_ROOT"])
                try:
                    if (source / ".git").exists():
                        revision = subprocess.check_output(
                            ["git", "-C", str(source), "rev-parse", "HEAD"],
                            text=True, stderr=subprocess.STDOUT, timeout=3,
                        ).strip()
                    else:
                        revision = (source / ".adaptvpr-source-revision").read_text().strip()
                    if revision != LIGHTX2V_COMMIT:
                        errors.append(f"LIGHTX2V_ROOT revision {revision}; required {LIGHTX2V_COMMIT}")
                except (OSError, subprocess.SubprocessError) as exc:
                    errors.append(f"LightX2V source check failed: {exc}")

        if planner:
            base = os.getenv("ADAPTVPR_PLANNER_API_BASE", "").rstrip("/")
            model = os.getenv("ADAPTVPR_PLANNER_MODEL", "")
            try:
                response = session.get(
                    base + "/models", timeout=5,
                    headers={"Authorization": "Bearer " + os.getenv("ADAPTVPR_PLANNER_API_KEY", "")},
                )
                response.raise_for_status()
                models = [item["id"] for item in response.json()["data"]]
                if model not in models:
                    errors.append(f"Planner model {model!r} is not served; available: {models}")
            except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                errors.append(f"Planner check failed: {exc}")
    return errors
