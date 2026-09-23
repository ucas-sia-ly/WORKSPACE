1.测试 demo_10
```bash
python tests/run_demo_10.py \
  --gsvcities-root ../dataset/gsv-cities \
  --output ./outputs/demo_10

python tests/run_demo_10.py \
  --gsvcities-root ../dataset/gsv-cities \
  --output ./outputs/demo_10 \
  --strict
```


2.下载相关的预训练权重
```bash
 download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir ./models/Qwen3-VL-4B-Instruct

hf download stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --revision 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 \
  --local-dir ./models/stable-diffusion-v1-5

hf download lllyasviel/ic-light \
  --revision 9cad1878695f546a7fb9eaca14e2a89131ba5ffe \
  --include "iclight_sd15_fc.safetensors" \
  --local-dir ./models/iclight-ckpt

hf download Qwen/Qwen-Image-Edit-2511 \
  --revision 6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9 \
  --local-dir ./models/Qwen-Image-Edit-2511
```

3.重新准备 demo 环境
```bash
cd AdaptVPR  # from the workspace directory
set -o pipefail

ADAPTVPR_SETUP_PYTHON=/home/admin123/miniconda3/envs/AdaptVPR/bin/python \
ADAPTVPR_TORCH_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
ADAPTVPR_PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
bash scripts/prepare_demo_environment.sh 2>&1 | tee /tmp/adaptvpr-setup.log

```
