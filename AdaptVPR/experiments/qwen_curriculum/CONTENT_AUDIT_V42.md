# V4.2 CPU continuation from existing distance caches

V4.1 remains frozen. Its run-1 artifact correctly marked the detector unusable:
the required global DINO-anomaly area exceeded 0.9366, even though local joint
anomalies already required both DINO and structural changes. No held-out
insertion control was rejected. An unusable detector exports no training images;
this does not establish that every generated image is bad.

V4.2 introduces a separate decision module and calibration artifact. It reuses
the exact v4.1 feature maps, provider/weights/code fingerprints and geometry.
No teacher/DINO features, image generation, GPU or training is required for
the cached-control calibration or cached-pair export stages.

## Decision and calibration

At each valid patch, the local joint score is the minimum of DINO distance and
aligned structural distance. A single disagreeing channel cannot produce high
joint evidence. The image score is the maximum spatial-window mean. Window
fractions are the half-width quantiles of known **FIT-control** insertion
rectangles. The algorithm uses their local interior to avoid interpolation at
the insertion boundary. Invalid patch evidence contributes zero; insufficient
overall overlap abstains.

All limits are learned only from the original training-place FIT controls:

* Reject strictly above the maximum benign FIT score, preserving zero FIT
  false rejects instead of lowering a cutoff to force outcomes.
* Pass within the central benign envelope (the declared alpha quantile), also
  below every known FIT insertion score.
* Between these bounds, or below FIT-derived valid-evidence floors: Unknown.

The decision rule is fingerprinted before any held-out evaluation. Held-out
controls only veto usability: exact one-sided paired tests must distinguish
insertions for rejection and benign transformations for passing, and empirical
benign false rejection/insertion false pass rates must not exceed alpha.
One benign and one insertion control must share each source/geometry/recipe.

The v4.2 score family was chosen after FIT-only exploratory comparison of raw
joint, local residual joint, normalized-residual and product scores. Qwen pair
outputs and real evaluation/test images did not select it. The original
held-out control split is reused; therefore its check is exploratory and
**does not constitute fresh external validation**. Success on these machine
controls cannot prove semantic correctness of every Qwen edit or equivalence
to real night-camera imagery. A rejected image is unsafe for this policy, not
a guaranteed hallucination label. Missing evidence stays Unknown, excluded.

## Standalone commands

Choose a fresh output location. V4.1 files and images are never overwritten.

```bash
PY=/home/admin123/miniconda3/envs/AdaptVPR/bin/python
CLI=AdaptVPR/experiments/qwen_curriculum/audit_content_v42.py
OLD=outputs/night_content_audit_v4_run1
NEW=outputs/night_content_audit_v42_run1

"$PY" "$CLI" calibrate --manifest "$OLD/controls/controls.jsonl" \
  --signals "$OLD/control_signals.jsonl" --output "$NEW/calibration.json" \
  --false-alarm-rate 0.05 --min-pairs 20

# Proceed only after calibration.json reports usable=true. If not, do not
# change limits by inspecting generated-pair or benchmark/test outcomes.
"$PY" "$CLI" audit --manifest outputs/weather_v3/state/results.jsonl \
  --signals "$OLD/pair_signals.jsonl" --calibration "$NEW/calibration.json" \
  --output-dir "$NEW/audit"
```

The manifest passed to `audit` must be exactly the generation manifest matching
the signal cache. Image bytes, homography, model, preprocessing and code hashes
are checked. Failed/missing cache rows remain Unknown; `shopfronts` remains
explicitly quarantined. Only the new `audit/training_manifest.jsonl` is eligible
for subsequent training. Original result seals are provenance fields; new
records get their own audit hashes. No formal training is launched.

CPU tests:

```bash
"$PY" -m unittest discover -s AdaptVPR/experiments/qwen_curriculum/tests \
  -p test_content_audit_v42.py -v
```
