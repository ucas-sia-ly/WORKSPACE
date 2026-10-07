#!/bin/bash
set -euo pipefail

ADAPTVPR_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# Read dotenv with Python (not shell evaluation) and resolve paths before cd.
if [ "${ADAPTVPR_SERVICE_ENV_LOADED:-0}" != "1" ]; then
  exec "${ADAPTVPR_HEALTHCHECK_PYTHON:-python}" - "$ADAPTVPR_ROOT" "$@" <<'PY'
import os
from pathlib import Path
import sys

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from generation.preflight import load_environment

load_environment()
for name in ("ICLIGHT", "LIGHTX2V"):
    os.environ.setdefault(name + "_PYTHON", sys.executable)
    os.environ.setdefault(name + "_ROOT", str(root.parent / ("IC-Light" if name == "ICLIGHT" else "LightX2V")))
for key in ("ADAPTVPR_SERVICE_LOG_DIR", "ADAPTVPR_TMP_DIR"):
    if os.getenv(key):
        os.environ[key] = str((root / Path(os.environ[key]).expanduser()).resolve())
os.environ["ADAPTVPR_SERVICE_ENV_LOADED"] = "1"
os.execv("/bin/bash", ["bash", str(root / "scripts/start_generation_services.sh"), *sys.argv[2:]])
PY
fi
WAIT_FOR_READY=0
if [ "${1:-}" = "--wait" ] && [ "$#" = "1" ]; then
  WAIT_FOR_READY=1
elif [ "$#" != "0" ]; then
  echo "Usage: bash scripts/start_generation_services.sh [--wait]" >&2
  exit 2
fi
LOG_DIR=${ADAPTVPR_SERVICE_LOG_DIR:-$ADAPTVPR_ROOT/tmp/logs}
TMP_DIR=${ADAPTVPR_TMP_DIR:-$ADAPTVPR_ROOT/tmp}
HEALTHCHECK_PYTHON=${ADAPTVPR_HEALTHCHECK_PYTHON:-python}
ICLIGHT_ROOT=${ICLIGHT_ROOT:-${ICLIGHT_WORKDIR:-../IC-Light}}
ICLIGHT_PYTHON=${ICLIGHT_PYTHON:-python}
LIGHTX2V_ROOT=${LIGHTX2V_ROOT:-${LIGHTX2V_WORKDIR:-../LightX2V}}
LIGHTX2V_PYTHON=${LIGHTX2V_PYTHON:-python}
ICLIGHT_ADAPTER=$ADAPTVPR_ROOT/adapters/iclight_sd15_fc.py
LIGHTX2V_ADAPTER=$ADAPTVPR_ROOT/adapters/lightx2v_qwen_image_edit.py

mkdir -p "$LOG_DIR" "$TMP_DIR"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export ADAPTVPR_DISABLE_MOCK=${ADAPTVPR_DISABLE_MOCK:-1}
export ICLIGHT_ROOT LIGHTX2V_ROOT

is_port_open() {
  local port="$1"
  "$HEALTHCHECK_PYTHON" - "$port" <<'PY'
import socket
import sys

sock = socket.socket()
sock.settimeout(1)
try:
    sock.connect(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
}

start_service() {
  local name="$1"
  local port="$2"
  local workdir="$3"
  local python_bin="$4"
  local entrypoint="$5"
  local log_file="$LOG_DIR/${name}_${port}.log"
  local cuda_devices="${CUDA_VISIBLE_DEVICES:-0}"
  local port_name

  if [ ! -d "$workdir" ]; then
    echo "[$name] missing upstream Git checkout: $workdir" >&2
    return 1
  fi
  if [ ! -f "$entrypoint" ]; then
    echo "[$name] missing AdaptVPR adapter: $entrypoint" >&2
    return 1
  fi
  if [ "$name" = "iclight" ]; then
    port_name=ICLIGHT_PORT
  else
    port_name=LIGHTX2V_PORT
  fi

  if [ "${ADAPTVPR_FORCE_RESTART:-0}" != "1" ] && is_port_open "$port"; then
    echo "[$name] port $port already open"
    return 0
  fi
  if [ "${ADAPTVPR_FORCE_RESTART:-0}" = "1" ] && is_port_open "$port"; then
    echo "[$name] stopping existing process on port $port"
    fuser -k "${port}/tcp" >/dev/null 2>&1 || true
    sleep 2
  fi

  echo "[$name] starting pinned adapter on port $port"
  # Multiple tmux command arguments bypass its default shell (which may be fish).
  # Use the same explicit Bash command for tmux and nohup, with paths as arguments.
  local -a service_command=(
    /bin/bash -c '
      exec >>"$8" 2>&1
      cd "$1" || exit 1
      export CUDA_VISIBLE_DEVICES="$3"
      export "$5=$6"
      export PLATFORM=cuda
      export SKIP_PLATFORM_CHECK=True
      export PYTHONPATH="$1:${PYTHONPATH:-}"
      export ICLIGHT_OUTPUT_DIR="$7/service_outputs/iclight"
      export LIGHTX2V_OUTPUT_DIR="$7/service_outputs/lightx2v"
      exec "$2" "$4"
    ' _ "$workdir" "$python_bin" "$cuda_devices" "$entrypoint" "$port_name" "$port" "$TMP_DIR" "$log_file"
  )
  if [ "${ADAPTVPR_DISABLE_TMUX:-0}" != "1" ] && command -v tmux >/dev/null 2>&1; then
    local session="adaptvpr_${name}_${port}"
    if tmux has-session -t "$session" 2>/dev/null; then
      if [ "${ADAPTVPR_FORCE_RESTART:-0}" != "1" ]; then
        echo "[$name] tmux session already running/loading; log=$log_file"
        return 0
      fi
      tmux kill-session -t "$session"
    fi
    # An existing tmux server retains its old environment. Forward current model
    # settings explicitly so changing .env really enables disk offload.
    local -a tmux_environment=(
      -e "ADAPTVPR_LORA_CHECKPOINT=${ADAPTVPR_LORA_CHECKPOINT:-}"
      -e "LIGHTX2V_DISK_OFFLOAD=${LIGHTX2V_DISK_OFFLOAD:-0}"
      -e "LIGHTX2V_CPU_OFFLOAD=${LIGHTX2V_CPU_OFFLOAD:-1}"
    )
    local key
    for key in ICLIGHT_ROOT ICLIGHT_BASE_MODEL_PATH ICLIGHT_MODEL_PATH \
               LIGHTX2V_ROOT LIGHTX2V_GIT_COMMIT LIGHTX2V_MODEL_PATH \
               LIGHTX2V_LORA_PATH LIGHTX2V_DISK_MODEL_PATH; do
      if [[ -v "$key" ]]; then
        tmux_environment+=(-e "$key=${!key}")
      fi
    done
    tmux new-session -d -s "$session" "${tmux_environment[@]}" "${service_command[@]}"
    echo "[$name] tmux=$session log=$log_file"
  else
    nohup "${service_command[@]}" </dev/null >/dev/null 2>&1 &
    echo "[$name] pid=$! log=$log_file"
  fi
}

start_service iclight "${ICLIGHT_PORT:-8002}" "$ICLIGHT_ROOT" "$ICLIGHT_PYTHON" "$ICLIGHT_ADAPTER"
start_service lightx2v "${LIGHTX2V_PORT:-8001}" "$LIGHTX2V_ROOT" "$LIGHTX2V_PYTHON" "$LIGHTX2V_ADAPTER"

echo "Generation service startup requested. Check /health until model_loaded and generator_ready are true."
if [ "$WAIT_FOR_READY" = "1" ]; then
  "$HEALTHCHECK_PYTHON" - "$ADAPTVPR_ROOT" "$LOG_DIR" <<'PY'
import os
import sys
import time
import requests

sys.path.insert(0, sys.argv[1])
from generation.service_health import probe_service

pending = {
    name: os.getenv(f"{name}_API_URL", f"http://127.0.0.1:{os.getenv(name + '_PORT', port)}/generate")
    for name, port in (("ICLIGHT", "8002"), ("LIGHTX2V", "8001"))
}
deadline = time.monotonic() + float(os.getenv("ADAPTVPR_SERVICE_READY_TIMEOUT", "900"))
with requests.Session() as session:
    session.trust_env = False
    while pending:
        for name, url in list(pending.items()):
            try:
                ready, detail = probe_service(session, url)
            except RuntimeError as exc:
                raise SystemExit(f"{name} initialization failed: {exc}; logs: {sys.argv[2]}")
            if ready:
                print(f"{name} ready", flush=True)
                del pending[name]
        if not pending:
            break
        if time.monotonic() >= deadline:
            raise SystemExit(f"Timed out waiting for {', '.join(pending)}; logs: {sys.argv[2]}")
        print(f"Waiting for {', '.join(pending)} to load; logs: {sys.argv[2]}", flush=True)
        time.sleep(5)
PY
fi
