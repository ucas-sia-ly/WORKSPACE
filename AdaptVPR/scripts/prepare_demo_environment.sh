#!/bin/bash
# Run manually in the intended Python environment; downloads can take many minutes.
set -euo pipefail
ADAPTVPR_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ADAPTVPR_ROOT"
DEMO_PYTHON=${ADAPTVPR_SETUP_PYTHON:-python}
PIN=522609ecc121b49c20d201b3f00c3dc052821bce
SOURCE_DIR="$ADAPTVPR_ROOT/../LightX2V"
CONSTRAINTS="$ADAPTVPR_ROOT/configs/demo-cu128-constraints.txt"

"$DEMO_PYTHON" - <<'PY'
import sys
if not (3, 10) <= sys.version_info[:2] <= (3, 12):
    raise SystemExit("Use the AdaptVPR Python 3.10-3.12 environment, not the base environment.")
print("Installing demo dependencies into", sys.executable, flush=True)
PY

# Use the single shared checkout. Never recreate a second worktree.
if [ ! -e "$SOURCE_DIR" ]; then
  git clone https://github.com/ModelTC/LightX2V.git "$SOURCE_DIR"
  git -C "$SOURCE_DIR" checkout --detach "$PIN"
fi
# Source copies inside a workspace repository use the revision marker instead
# of the workspace's own Git HEAD. Preserve local compatibility patches.
if [ -e "$SOURCE_DIR/.git" ]; then
  SOURCE_REVISION=$(git -C "$SOURCE_DIR" rev-parse HEAD)
elif [ -f "$SOURCE_DIR/.adaptvpr-source-revision" ]; then
  SOURCE_REVISION=$(cat "$SOURCE_DIR/.adaptvpr-source-revision")
else
  SOURCE_REVISION=""
fi
if [ "$SOURCE_REVISION" != "$PIN" ]; then
  echo "Unexpected source revision in $SOURCE_DIR" >&2
  echo "Expected $PIN; preserve local changes before changing the checkout." >&2
  exit 1
fi

"$DEMO_PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.8.0' 'torchvision==0.23.0' 'torchaudio==2.8.0'
"$DEMO_PYTHON" -m pip install -c "$CONSTRAINTS" \
  -r requirements.txt -r adapters/requirements.txt \
  -e "$SOURCE_DIR" -e "$ADAPTVPR_ROOT/../vismatch"

"$DEMO_PYTHON" - "$SOURCE_DIR" <<'PY'
import os
from pathlib import Path
import sys

from dotenv import set_key
from huggingface_hub import hf_hub_download, snapshot_download
from generation.preflight import load_environment

load_environment()
root = Path.cwd()
env_file = root / ".env"
source = Path(sys.argv[1]).resolve()
models = Path(os.environ["LIGHTX2V_MODEL_PATH"]).parent
lora = hf_hub_download(
    "lightx2v/Qwen-Image-Edit-2511-Lightning",
    "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    revision="d74eba145674fd7e31b949324e148e21e7118abd",
    local_dir=models,
)
snapshot_download("openai/clip-vit-base-patch32")
for key, value in {
    "LIGHTX2V_ROOT": str(source),
    "LIGHTX2V_LORA_PATH": str(Path(lora).resolve()),
    "ICLIGHT_PYTHON": sys.executable,
    "LIGHTX2V_PYTHON": sys.executable,
}.items():
    set_key(str(env_file), key, value)

import torch
if not torch.cuda.is_available():
    raise SystemExit(f"CUDA still unavailable with torch={torch.__version__}")
os.environ["PLATFORM"] = "cuda"
os.environ["SKIP_PLATFORM_CHECK"] = "True"
sys.path.insert(0, str(source))
from lightx2v import LightX2VPipeline
from vismatch import get_matcher
from transformers import CLIPModel, CLIPProcessor
from diffusers import StableDiffusionPipeline

# Download/check the small matcher weights without loading generation models.
get_matcher("superpoint-lightglue", device="cpu", max_num_keypoints=2048)
print("Dependency imports, CUDA and matcher initialization passed.")
PY
echo "Setup finished. Start the generators: bash scripts/start_generation_services.sh"
echo "Then check readiness: python tests/run_demo_10.py --gsvcities-root ../dataset/gsv-cities --output ./outputs/demo_10 --check-only"
