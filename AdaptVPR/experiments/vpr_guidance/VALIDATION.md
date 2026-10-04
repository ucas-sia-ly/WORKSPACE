# Evaluation infrastructure validation

Repository: `ucas-sia-ly/WORKSPACE`, branch `test/vpr-loss-generator`.
Starting HEAD: `df2ad17a2e471495c2648a609e70808dce6c53fc`.
Runtime: AdaptVPR environment, torch `2.8.0+cu128`, diffusers `0.36.0`.
CUDA is unavailable; LPIPS is not yet installed in that environment.

## Files

Added:

- `check_baseline_equivalence.py`
- `tests/test_baseline_equivalence.py`
- `tests/test_retrieval_reporting.py`
- `VALIDATION.md`

Modified:

- `evaluate_retrieval.py`
- `requirements.txt` (LPIPS; optional SSIM dependency comment)
- `README.md`

No changes to `ddim.py`, `vpr.py`, `generate.py`, `gsv_pairs.py`, the original
adapter, model weights, prompts, or existing tests. No Phase 2 code.

## Exact checks executed

Existing suite before changes: **9 passed**.
Expanded suite: **16 passed**, including the original nine.
Focused equivalence suite: **3 passed**.

```bash
/home/admin123/miniconda3/envs/AdaptVPR/bin/python -m unittest discover -s AdaptVPR/experiments/vpr_guidance/tests -v
/home/admin123/miniconda3/envs/AdaptVPR/bin/python -m unittest discover -s AdaptVPR/experiments/vpr_guidance/tests -p 'test_baseline_equivalence.py' -v
/home/admin123/miniconda3/envs/AdaptVPR/bin/python -m compileall -q AdaptVPR/experiments/vpr_guidance
```

The equivalence integration test calls the existing adapter function with small
real Diffusers UNet/VAE/pipelines on CPU, substituting only prompt embeddings
and CUDA-specific plumbing. Both 0.30 and 0.22 refinement policies pass matching
configuration/timestep and low pixel-MAE checks. This is an infrastructure test,
not a measurement on the pretrained IC-Light snow outputs. Metric tests check
uint8 scale, finite JSON encoding for infinite PSNR, LPIPS tensor range, and
nonfatal warnings. Retrieval integration uses synthetic images/descriptors to
check all three query sets, optional BoQ columns, per-condition/skipped results,
source exclusion, recovery ratios, rank ties and saved JSONL/table output.

Both real-input dry runs succeeded and selected only the two Global snow rows:

```bash
/home/admin123/miniconda3/envs/AdaptVPR/bin/python -m AdaptVPR.experiments.vpr_guidance.check_baseline_equivalence --conditions snow --limit 2 --image-root dataset/gsv-cities/Images/Bangkok --output-dir /tmp/vpr-equivalence-smoke --dry-run > /tmp/vpr-equivalence-inputs.json
/home/admin123/miniconda3/envs/AdaptVPR/bin/python -m AdaptVPR.experiments.vpr_guidance.generate --conditions snow --limit 2 --image-root dataset/gsv-cities/Images/Bangkok --output-dir /tmp/vpr-guided-smoke --seed 42 --guidance-scale 0.003 0.01 0.03 --guidance-every 5 --guidance-last-n 10 --dry-run > /tmp/vpr-guided-inputs.json
```

Real generation attempts stopped explicitly at the CUDA check:

```bash
/home/admin123/miniconda3/envs/AdaptVPR/bin/python -m AdaptVPR.experiments.vpr_guidance.check_baseline_equivalence --conditions snow --limit 2 --image-root dataset/gsv-cities/Images/Bangkok --output-dir /tmp/vpr-equivalence-smoke --seed 42
/home/admin123/miniconda3/envs/AdaptVPR/bin/python -m AdaptVPR.experiments.vpr_guidance.generate --conditions snow --limit 2 --image-root dataset/gsv-cities/Images/Bangkok --output-dir /tmp/vpr-guided-smoke --seed 42 --guidance-scale 0.003 0.01 0.03 --guidance-every 5 --guidance-last-n 10 --geometry
```

The README's small-database preparation was exercised using a temporary script:

```bash
PYTHONPATH="$PWD" /home/admin123/miniconda3/envs/AdaptVPR/bin/python /tmp/prepare_vpr_smoke_database.py
```

It created `/tmp/vpr-smoke-database` with **50 capture symlinks**, including all
captures of both demo places and 32 distractors. Assertions confirmed **3**
alternate positives for `adapt_000066` and **13** for `adapt_000071`, with each
exact source capture excluded. `/tmp/vpr-smoke-database/manifest.json` records
those counts. The setup script's self-contained equivalent is in the README.
This subset establishes that positives exist; two queries cannot support a
claim of general retrieval improvement.

## Checks that remain on a CUDA machine

Install the updated requirements in the existing environment, then run README
Steps B–D: real snow baseline equivalence, guided smoke generation at scales
0.003/0.01/0.03, and source/baseline/guided retrieval against the fixed small
subset with held-out BoQ and geometry. The temporary database already exists
here; use it directly or prepare a fresh directory when repeating the workflow.

No real snow MAE/RMSE/PSNR/LPIPS/SALAD equivalence metrics were produced. No
real guided images, retrieval recalls or recovery ratios were produced. Retrieval
on generated smoke outputs could not run because Step C has no CUDA outputs.
No numbers from synthetic tests are presented as scientific results.

## Remaining risks

- Full pretrained CUDA outputs may differ between released pipeline calls and
  the explicit sampler despite passing small CPU integration checks. Review
  recorded scheduler configurations/timesteps and diagnostic metrics first.
- Floating-point kernels and future Diffusers default changes may alter results.
  Warning thresholds are diagnostic, not acceptance criteria or an equality gate.
- Equivalence calls the original adapter directly with identical source pixels.
  It does not reproduce the HTTP client's extra JPEG re-encoding, which would
  otherwise change the source conditioning.
- The two-query, 50-capture subset is a smoke test with sorted distractors, not a
  representative evaluation set. Missing BoQ/geometry is insufficient evidence;
  SALAD-only improvement could reflect SALAD-specific optimization.

No guidance bug was found or fixed. The current algorithm and Phase 2 TODO
remain intact.
