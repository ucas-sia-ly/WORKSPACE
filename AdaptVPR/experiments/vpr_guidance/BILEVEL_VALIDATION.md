# Bilevel SALAD implementation validation

Validated on 2026-10-07 in `ucas-sia-ly/WORKSPACE`, branch
`test/vpr-loss-generator`, using the existing `AdaptVPR` conda environment
(PyTorch 2.8.0+cu128, diffusers 0.36.0, pytorch-metric-learning 2.9.0).

## Changed implementation

- `salad_factory.py`: shared fresh DINOv2 + random SALAD construction; unchanged
  downstream configuration; meta parameter policy, differentiable image
  preprocessing, pure metric loss, eager second-order DINO attention.
- `bilevel.py`: official-dataframe-validated real GSV index, current-anchored
  distinct-place episodes, disjoint support/query, functional K-step SGD.
- `train_generator.py`: differentiable accepted-target x0/VAE prediction,
  actual bilevel objective, LoRA-only optimizer checks and meta-only gradient
  audit. Old teacher/VJP helpers are isolated legacy utilities.
- `train_online_generator.py`: accepted current + bounded replay episodes,
  actual/skipped update counts, diagnostics, objective-version resume checks,
  deterministic fresh initialization and hashes.
- `train_full.py`: meta CLI/config forwarding and objective-version checks.
- `train_salad.py`: shared factory; final downstream training behavior preserved.
- `teacher.py`: diagnostic/cache role documented.
- `tests/test_bilevel.py`, `tests/test_gradients.py`, `tests/test_online.py`:
  added/updated correctness and orchestration regressions.
- `smoke_bilevel_gpu.py`: opt-in actual-checkpoint, meta-only integration smoke.
- `README.md`: objective, mathematical gradient path, CLI, resume, memory and
  truncation limitations.

## Gradient path

```text
held-out real GSV query MultiSimilarity loss
  -> functional SALAD with theta_K(phi)
  -> K SGD updates with autograd.grad(create_graph=True)
  -> real + differentiable generated support MultiSimilarity loss
  -> tensor resize and ImageNet normalization
  -> generated RGB from frozen VAE decode
  -> predicted x0 from gradient-enabled IC-Light UNet
  -> fp32 IC-Light LoRA parameters phi
```

Miner selection uses official MultiSimilarity semantics in support and query.
The teacher has no contribution to this graph. The generator total adds the
existing diffusion MSE and accepted-target L1 regularizers. Base SALAD weights
stay fixed; each episode starts from their original initialization.

## CPU suite

```bash
/home/admin123/miniconda3/envs/AdaptVPR/bin/python \
  -m unittest discover -s AdaptVPR/experiments/vpr_guidance/tests -v
```

Result: **81 tests passed, no skips or failures**. Coverage includes:

- nonzero meta-only gradient, detached-inner-update negative control;
- K=2 finite-difference agreement and repeatable fresh starts;
- a real tiny diffusers LoRA UNet -> x0 -> frozen tiny VAE -> bilevel gradient;
- synthetic input gradient through frozen/all-four-block meta backbone policies;
- exact official metric/miner helper equivalence and shared factory config;
- teacher diagnostics leaving LoRA gradients unchanged;
- optimizer isolation and unchanged/unaccumulated base SALAD weights;
- dataframe labels, deterministic episodes, disjoint real support/query,
  rejected/insufficient/duplicate place handling and current-anchor requirements;
- online refresh, exact interrupted-boundary resume, replay enabling a later
  valid episode, skipped-update accounting, and all-rejected rounds;
- full orchestration forwarding every meta setting;
- existing data, generation, verifier, cache, evaluation and downstream trainer
  regressions, including optimizer/scheduler/RNG/worker-epoch restoration.

`git diff --check` also passed.

## Actual GPU/checkpoint smoke

Hardware: NVIDIA GeForce RTX 4090 as reported by this host, 49,140 MiB VRAM.
Used the existing cached pretrained DINOv2 and released IC-Light weights, with
640x480 verifier-approved targets from
`outputs/full_run/generator/rounds/pass_00_round_000/records.jsonl` and separate
real support/query files from `dataset/gsv-cities`.

```bash
export ICLIGHT_ROOT="$PWD/IC-Light"
export ICLIGHT_BASE_MODEL_PATH="$PWD/models/stable-diffusion-v1-5"
export ICLIGHT_MODEL_PATH="$PWD/models/iclight-ckpt/iclight_sd15_fc.safetensors"

/home/admin123/miniconda3/envs/AdaptVPR/bin/python \
  -m AdaptVPR.experiments.vpr_guidance.smoke_bilevel_gpu \
  --accepted-records outputs/full_run/generator/rounds/pass_00_round_000/records.jsonl \
  --gsv-root dataset/gsv-cities --salad-root salad \
  --meta-places 2 --meta-train-backbone-blocks 0
```

Repeated the command with `--meta-train-backbone-blocks 4`, and with the default
four-place, zero-backbone-block episode. All three runs used K=1, inner_lr=1e-3,
224x224 SALAD inputs, lambda_meta=1 and **lambda_diff=lambda_keep=0**.

| Places | Trainable backbone blocks | Inner before -> after | Outer loss | Meta-only LoRA norm | Peak CUDA allocated |
| --- | --- | --- | --- | --- | --- |
| 2 | 0 | 1.240868 -> 1.239952 | 1.264127 | 8.31664e-5 | 11.05 GiB |
| 2 | 4 | 1.240868 -> 1.232752 | 1.261740 | 8.03971e-4 | 11.84 GiB |
| 4 (default) | 0 | 0.934417 -> 0.934011 | 1.262228 | 1.86429e-5 | 18.57 GiB |

All smoke runs asserted a positive meta-only LoRA norm, positive total LoRA
norm, no base SALAD `.grad` accumulation, and an unchanged initialization hash:
`ca149448d52485c65ba2fbde29eba2c29b7a53bded704400356bcf42639f6fdb`.
No teacher or source descriptor was required. No accepted/source images or
existing experiment checkpoints were modified; smoke LoRA updates were only
in memory.

An initial unscaled real fp16 IC-Light smoke exposed hypergradient underflow:
inner adaptation and outer loss were finite, but LoRA norm was zero. Fixed
outer backward scaling (65536), followed by unscaling fp32 LoRA gradients,
resolved it. Inner/outer SALAD losses and SGD arithmetic remain fp32, with no
AMP or graph detach. CPU tests also exercise this scaling path.

## Limits of validation

The smoke measures one update, and excludes resident teacher/verifier memory
from a full online process. Larger K, more places, larger accepted diffusion
images, and trainable backbone blocks increase cost. The explicit meta-only
norm audit performs an extra backward traversal. Reduce episode size or K for
memory pressure; never detach inner gradients. Fixed scaling can require tuning
if gradients overflow; finite assertions stop the run rather than silently
changing the trainable parameter policy.

No GPU/checkpoint test was blocked by unavailable dependencies in this
environment. A full released Generate -> real Verify -> multi-round Train run,
4000-step downstream A/B/C training, and final-domain benchmarking were not
executed. The real smoke reused already approved targets; verifier gating and
online boundary logic were exercised by the package regression suite. This is
correctness validation of truncated bilevel optimization, not evidence of final
retrieval improvements or an exact solution of downstream training.
