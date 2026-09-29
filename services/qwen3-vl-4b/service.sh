#!/usr/bin/env bash
set -Eeuo pipefail
SERVICE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER=${QWEN_CONTAINER_NAME:-qwen3-vl-4b-instruct}
IMAGE='vllm/vllm-openai:v0.11.0@sha256:014a95f21c9edf6abe0aea6b07353f96baa4ec291c427bb1176dc7c93a85845c'
MODEL_REVISION=ebb281ec70b05090aa6165b016eac8ec08e71b17
BASE_URL=http://127.0.0.1:23002
export QWEN_API_KEY="${QWEN_API_KEY:-local-placeholder}"

case "${1:-start}" in
  start)
    mkdir -p "$SERVICE_DIR/logs" "$SERVICE_DIR/cache"
    exec > >(tee -a "$SERVICE_DIR/logs/deploy.log") 2>&1
    trap 'rc=$?; if (( rc != 0 )); then echo "Deployment failed (exit $rc). See logs/deploy.log and logs/server.log."; docker logs --tail 100 "$CONTAINER" > "$SERVICE_DIR/logs/server.log" 2>&1 || true; fi' EXIT
    command -v docker >/dev/null
    command -v curl >/dev/null
    command -v python3 >/dev/null
    docker info >/dev/null
    resource_args=()
    if [[ -n "${QWEN_MEMORY_LIMIT:-}" ]]; then
      resource_args+=(--memory "$QWEN_MEMORY_LIMIT" --memory-swap "${QWEN_MEMORY_SWAP_LIMIT:-$QWEN_MEMORY_LIMIT}")
    fi
    if [[ -n "${QWEN_CPU_LIMIT:-}" ]]; then
      resource_args+=(--cpus "$QWEN_CPU_LIMIT")
    fi
    if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
      if [[ "$(docker inspect -f '{{index .Config.Labels "local.qwen3-vl.service"}}' "$CONTAINER")" != "$SERVICE_DIR" ]]; then
        echo "Container name is already used by another deployment: $CONTAINER"; exit 1
      fi
      echo "Starting existing container (existing configuration retained)."
      if (( ${#resource_args[@]} )); then
        docker update "${resource_args[@]}" --restart "${QWEN_RESTART_POLICY:-unless-stopped}" "$CONTAINER" >/dev/null
      fi
      docker start "$CONTAINER" >/dev/null
    else
      python3 - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 23002))
PY
      echo "Downloading pinned vLLM image. This can take tens of minutes."
      docker pull "$IMAGE"
      proxy_args=()
      for proxy_var in HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy HF_ENDPOINT HF_TOKEN; do
        if [[ -n "${!proxy_var:-}" ]]; then proxy_args+=(--env "$proxy_var"); fi
      done
      docker run --detach \
        --name "$CONTAINER" \
        --label "local.qwen3-vl.service=$SERVICE_DIR" \
        --restart "${QWEN_RESTART_POLICY:-unless-stopped}" \
        "${resource_args[@]}" \
        --gpus device=0 \
        --network host \
        --shm-size 4g \
        --log-opt max-size=20m --log-opt max-file=3 \
        --volume "$SERVICE_DIR/cache:/root/.cache/huggingface" \
        --env "NO_PROXY=localhost,127.0.0.1,${NO_PROXY:-}" \
        --env "no_proxy=localhost,127.0.0.1,${no_proxy:-}" \
        "${proxy_args[@]}" \
        "$IMAGE" \
        --model Qwen/Qwen3-VL-4B-Instruct \
        --revision "$MODEL_REVISION" \
        --served-model-name qwen3-vl-4b-instruct-remote Qwen/Qwen3-VL-4B-Instruct \
        --api-key "$QWEN_API_KEY" \
        --host 127.0.0.1 --port 23002 \
        --dtype bfloat16 \
        --gpu-memory-utilization "${QWEN_GPU_MEMORY_UTILIZATION:-0.35}" \
        --max-model-len 16384 \
        --max-num-seqs 2 \
        --max-num-batched-tokens 4096 \
        --limit-mm-per-prompt '{"image":4,"video":0}' \
        --mm-processor-kwargs '{"min_pixels":4096,"max_pixels":1048576}' \
        --enforce-eager
    fi
    echo "Waiting for model download and initialization; progress: docker logs -f $CONTAINER"
    deadline=$((SECONDS + ${QWEN_STARTUP_TIMEOUT:-3600}))
    until curl --noproxy '*' --silent --fail --max-time 3 "$BASE_URL/health" >/dev/null; do
      if (( SECONDS >= deadline )); then
        echo "Startup wait expired. Container remains available for diagnosis; use service.sh logs or stop."
        exit 1
      fi
      if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" != true ]]; then
        echo "Container is not running; inspect logs/server.log."; exit 1
      fi
      sleep 10
    done
    python3 "$SERVICE_DIR/smoke_test.py"
    docker logs --tail 100 "$CONTAINER" > "$SERVICE_DIR/logs/server.log" 2>&1
    echo "READY: $BASE_URL/v1 | model=qwen3-vl-4b-instruct-remote"
    ;;
  stop) docker stop "$CONTAINER" ;;
  status) docker ps -a --filter "name=^/${CONTAINER}$"; curl --noproxy '*' --fail --silent --show-error --max-time 3 "$BASE_URL/health" ;;
  logs) docker logs --tail 100 "$CONTAINER" ;;
  check) python3 "$SERVICE_DIR/smoke_test.py" ;;
  *) echo "Usage: $0 {start|stop|status|logs|check}" >&2; exit 2 ;;
esac
