# AdaptVPR

本工作区当前推荐使用 [Qwen 困难源图增广流程](experiments/qwen_curriculum/README.md)：离线筛选 GSV-Cities 困难训练图，按 night/snow/fog/rain 预算仅调用 Qwen，再以同源图概率替换接入 SALAD。该入口针对单台 48GB 显存、32GB 内存机器，分阶段加载模型；1000 次新增调用按 night 500、snow 200、fog 200、rain 100 分配，取消 overcast 生成和生成器 LoRA 闭环。`run.py` 的 Global、Local、Dual 路由现在也全部使用 Qwen，但保留独立的路由调度、反思和双指标验证协议。

Official repository for [AdaptVPR: Route-Aware Hard Positive Generation for Robust Visual Place Recognition](https://arxiv.org/abs/2609.04369).

<p align="center">
  <a href="https://arxiv.org/abs/2609.04369"><img src="https://img.shields.io/badge/arXiv-2609.04369-D32F2F?style=flat-square&amp;labelColor=444444" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/shunpeng/AdaptCities"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-AdaptCities-FFD21E?style=flat-square&amp;labelColor=444444" alt="Hugging Face Dataset"></a>
</p>

<p align="center">
  <img src="assets/matching_demos/01_global16_prague_5s_loop.gif" width="48%" />
  <img src="assets/matching_demos/02_local08_madrid_5s_loop.gif" width="48%" />
</p>

<p align="center">
  <img src="assets/matching_demos/03_local00007_bangkok_5s_loop.gif" width="48%" />
  <img src="assets/matching_demos/04_dual_demo02_bangkok_5s_loop.gif" width="48%" />
</p>

<p align="center">
  <img src="assets/matching_demos/05_dual_demo07_bangkok_5s_loop.gif" width="48%" />
  <img src="assets/matching_demos/06_dual06_lisbon_5s_loop.gif" width="48%" />
</p>

## 📢 News

- **2026-09-10** — ⚡ Improved planning reproducibility and verification efficiency with fully documented scheduler configuration and content-validated CLIP reference-feature caching.
- **2026-09** — 📄 Paper released on arXiv: [AdaptVPR: Route-Aware Hard Positive Generation for Robust Visual Place Recognition](https://arxiv.org/abs/2609.04369).
- **2026-09** — 🚀 Released the AdaptVPR generation code, verification pipeline, prompt templates, and reproducibility documentation.
- **2026-09** — 📦 Released the AdaptCities prompts, annotations, and metadata on [Hugging Face](https://huggingface.co/datasets/shunpeng/AdaptCities).

## 📝 Method overview

![Overview of the AdaptVPR framework](assets/Method.png)
> **Method overview.** AdaptVPR combines scene understanding, generation tailored to each route, selective prompt reflection, and verification of geometric consistency and appearance diversity to improve VPR robustness under changes in weather, illumination, and occlusion.

| Route | Edit | Generator | Policy |
|---|---|---|---|
| Global | Weather / illumination / time | [Qwen-LightX2V](https://github.com/ModelTC/LightX2V) | 1 attempt; reject on failure |
| Local | Local occlusion | [Qwen-LightX2V](https://github.com/ModelTC/LightX2V) | ≤4 attempts; ≤3 refinements |
| Dual | Global + local changes | [Qwen-LightX2V](https://github.com/ModelTC/LightX2V) | ≤4 attempts; ≤3 refinements |
| Skip | Unsuitable input | None | No generation |

The table describes the current `run.py` route agent. Its `DualTraitEvaluator`
accepts candidates when both $s_{\mathrm{geo}} \ge \tau_{\mathrm{geo}}$ and
$s_{\mathrm{div}} \ge \tau_{\mathrm{div}}$, using the route thresholds below.
The paper diagram and historical IC-Light diagnosis remain available for
reproduction.

Scene planning is implemented in `generation/agent.py`, while prompt refinement is handled by `generation/reflection_controller.py`; both are used by the public entry point.

| Threshold | Global | Local | Dual | Dual (rain + vehicle) |
|:---:|:---:|:---:|:---:|:---:|
| $\tau_{\mathrm{geo}}$ | 0.78 | 0.82 | 0.72 | 0.72 |
| $\tau_{\mathrm{div}}$ | 0.15 | 0.09 | 0.20 | 0.12 |

The recommended [staged Qwen curriculum](experiments/qwen_curriculum/README.md)
uses source mining, one weather edit per source and separate structure/weather
checks. Its acceptance rules include match evidence, spatial coverage,
homography displacement and CLIP weather contrasts; `s_div` is descriptive.
Its four-domain generation quota is independent of the route-agent scheduler.
The configurable first-run checks remain experimental and require visual review.

## 📁 Expected workspace layout

```text
workspace/
├── Gsvcities/
│   ├── Images/
│   └── Dataframes/
└── AdaptVPR/
    ├── adapters/                      # Generator HTTP services and dependencies
    ├── assets/
    │   └── Method.png                 # Paper method overview
    ├── configs/
    │   └── default.env.example        # Public configuration template
    ├── docs/
    │   ├── EXTERNAL_COMPONENTS.md     # Third-party setup and integration map
    │   ├── API_CONTRACTS.md           # Generator HTTP service contracts
    │   └── SCHEDULER.md               # Route quotas, parameters, and run semantics
    ├── examples/
    │   ├── generated_prompts.example.jsonl
    │   └── annotations_metadata.example.jsonl
    ├── experiments/
    │   ├── qwen_curriculum/           # Recommended staged augmentation workflow
    │   └── generation_diagnosis/      # Historical generation/verification studies
    ├── generation/                    # Planning, generation, routing and reflection
    ├── preprocessing/                 # Accepted-sample manifest processing
    ├── prompts/                       # Prompt templates and construction rules
    ├── requirements.txt               # Main AdaptVPR dependencies
    ├── run.py                         # Planner and frozen-prompt batch entry point
    ├── scripts/
    │   ├── build_manifest.py          # Export accepted training candidates
    │   └── start_generation_services.sh
    ├── tests/                         # Unit tests and 10-source demo files
    └── verification/                  # Geometry and appearance verification
```

## 🛠️ Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/chenshunpeng/AdaptVPR.git
cd AdaptVPR
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp configs/default.env.example .env
```

To run the included HTTP adapters, install their service dependencies separately:

```bash
pip install -r adapters/requirements.txt
```

Install and configure Qwen-LightX2V, VisMatch and the CLIP weights separately.
The route-agent planner and reflection also require a Qwen3-VL-4B-Instruct
endpoint. The staged curriculum uses the SALAD checkout and its released
checkpoint for source mining and training. IC-Light is optional for historical
diagnosis. Set local paths, endpoints and credentials in `.env`.

## 🧩 External components

| Role | Component | Integration | Public default |
|---|---|---|---|
| Planner and prompt refiner | [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | OpenAI-compatible API | `127.0.0.1:23002/v1` |
| Global/Local/Dual generator | [Qwen-LightX2V](https://github.com/ModelTC/LightX2V) | Local HTTP adapter | `127.0.0.1:8001/generate` |
| Historical generator (optional) | [IC-Light](https://github.com/lllyasviel/IC-Light) | Local HTTP adapter | `127.0.0.1:8002/generate` |
| Appearance verifier | [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) | Transformers local loading | `openai/clip-vit-base-patch32` |
| Geometry verifier | [VisMatch (SuperPoint + LightGlue)](https://github.com/gmberton/vismatch) | Local Python import | `superpoint-lightglue` |

See [Scheduler specification](docs/SCHEDULER.md) for the route-agent quota rule,
[External components](docs/EXTERNAL_COMPONENTS.md) for setup, and
[API contracts](docs/API_CONTRACTS.md) for the default Qwen service and optional
historical IC-Light service. Curriculum commands and quotas are documented in
[qwen_curriculum](experiments/qwen_curriculum/README.md).

## ⚡ Quick Demo

Use the ten GSV-Cities paths listed in `tests/demo_10.csv` to check the route
agent's planning, generation, reflection and dual-trait verification. The
recommended staged curriculum has its own mining and generation commands.
Source images are not included; run:

Source-to-generated image comparisons for this 10-image Quick Demo set are
available in [`tests/output`](tests/output).

```bash
python tests/run_demo_10.py \
  --gsvcities-root /path/to/Gsvcities \
  --output ./outputs/demo_10
```

This command uses Qwen3-VL-4B planning and enables up to three prompt-reflection attempts by default.
The planner may legitimately choose Skip. By default the demo validates that
all ten sources have completed records and any generated images exist, and
reports generation/Skip counts and observed routes. Add `--strict` to generate
all ten sources through the public released-prompt path: it uses the bundled
official prompts instead of replanning routes, so planner Skip cannot prevent
generation. These prompts cover 2 Global, 4 Local, and 4 Dual tasks. Their dataset
revision and SHA-256 are recorded in `tests/demo_10_prompts.source.json`.
Strict outputs go to `strict_released_reflection_on/` (or `_off/`), independently
of earlier planning runs. `--resume` reuses completed outputs within that run.
Image verification and acceptance thresholds are unchanged: a generated image
can still fail verification and be retained under `rejected/`.

```bash
python tests/run_demo_10.py --gsvcities-root /path/to/Gsvcities \
  --output ./outputs/demo_10 --strict --resume
```
It checks input files, CUDA, dependencies, and service health before planning.
Add `--check-only` to run these checks without loading generation models or starting inference.
Start Qwen with `bash scripts/start_generation_services.sh --wait`; the launcher
and demo preflight check the LightX2V `/health` endpoint by default. The startup
script reads `.env` and resolves relative dependency paths against this repository.
Historical IC-Light runs can add `--with-iclight` to the launcher.

## 🚀 Generation

The commands below exercise the `run.py` route agent, using Qwen for all three
generation routes. Local and Dual routes use one initial generation and up to
three reflection rounds; Global uses one generation. For the recommended
1000-call augmentation and SALAD replacement workflow, follow
[qwen_curriculum](experiments/qwen_curriculum/README.md).

Accepted final images (`passed=true` and `eligible_for_training=true`) are saved under `<output-root>/<route>/`, while failed candidates are retained for audit under `<output-root>/rejected/<route>/` with a `__rejected.jpg` suffix. Training should use the manifest generated by `scripts/build_manifest.py`, which includes only accepted candidates; generated images are not distributed by this repository.

```bash
# Plan routes and prompts with Qwen3-VL-4B-Instruct.
python run.py /path/to/Gsvcities/Images --mode plan --output ./outputs/planner \
  --reflection on --max-reflections 3

# Generate directly from released initial prompts.
python run.py /path/to/AdaptCities/prompts/by_city/Bangkok.jsonl --mode prompt \
  --image-root ./Gsvcities/Images \
  --output ./outputs/prompt \
  --reflection on --max-reflections 3
```

## 📦 Data Availability and License

We release the AdaptVPR generation code, processing pipeline, and prompt templates. Original GSV-Cities and generated AdaptCities images are not redistributed; obtain [GSV-Cities](https://github.com/amaralibey/gsv-cities) separately for reproduction.

The released AdaptCities prompts, annotations, and metadata are available on [Hugging Face](https://huggingface.co/datasets/shunpeng/AdaptCities).

## 🙏 Acknowledgements

This project builds on [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), [IC-Light](https://github.com/lllyasviel/IC-Light), [Qwen-LightX2V](https://github.com/ModelTC/LightX2V), [VisMatch](https://github.com/gmberton/vismatch), [CLIP](https://huggingface.co/openai/clip-vit-base-patch32), and [GSV-Cities](https://github.com/amaralibey/gsv-cities).

We also thank the authors of [BoQ](https://github.com/amaralibey/Bag-of-Queries), [ImAge](https://github.com/buptzjy/ImAge), [SALAD](https://github.com/serizba/salad), and [EDTFormer](https://github.com/buptzjy/EDTFormer) for their public implementations.

## 📌 Citation

If you find this repository useful for your research, please consider giving it a ⭐ and citing our [paper](https://arxiv.org/abs/2609.04369):

```bibtex
@article{chen2026adaptvpr,
  title   = {AdaptVPR: Route-Aware Hard Positive Generation for Robust Visual Place Recognition},
  author  = {Chen, Shunpeng and Zhang, Jingyi and Wang, Changwei and Xu, Shengpeng and Song, Yukun and Pei, Xingtian and Lin, Jinzhou and Guo, Li and Xu, Shibiao},
  journal = {arXiv preprint arXiv:2609.04369},
  year    = {2026}
}
```
