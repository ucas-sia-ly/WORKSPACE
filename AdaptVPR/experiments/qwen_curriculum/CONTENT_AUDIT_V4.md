# Automated content safety envelope (v4)

The v3 geometry/weather ledger is immutable. `audit_content_v4.py` produces a
new audit ledger and a **pass-only** training manifest. Existing images are
referenced and SHA256 verified; they are never moved, rewritten or deleted.
V4 generator candidates are ineligible until this independent audit passes.

## Semantics

* **Pass**: base geometry/weather passed, exact image/geometry/model pipeline
  provenance matches, sufficient valid overlap, and signals fall within the
  training-control envelope. This is not proof of semantic fidelity or of real
  night-camera realism.
* **Reject**: `shopfronts` is explicitly quarantined, or a spatially coherent
  **joint** DINO/structural outlier also exceeds the calibrated DINO-area limit.
  This means unsafe for this training policy, not a hallucination ground truth.
* **Unknown**: missing calibration/features/geometry, insufficient valid overlap,
  mismatched hashes, an unvalidated detector, or ambiguous isolated signals.
  Unknown never enters the exported training manifest.

Low cosine alone is unknown, never a destruction/Negative label. Local normalized
contrast is a structural diagnostic, not semantic evidence. OCR cache boxes are
projected using the homography, but newly visible or changed text is diagnostic
only: an existing sign may become visible at night. OCR cannot by itself reject
an image. No manual labels or review stage is involved.

## Automatic calibration

`controls` derives known pairs from approved training sources: geometric
translations plus relighting/gamma/noise/blur/JPEG controls, and identical
controls with a known foreign-region insertion from another training place.
Benign controls include smooth spatial multiplicative illumination fields and
gain-only localized clipping centered on existing source brightness. These
change existing surfaces without drawing new fixtures, signs or objects;
their parameters are preserved in every machine control recipe.
Fit/held-out assignment is deterministic by place hash; insertion donors stay
within the same partition. Full machine recipes, exclusion-list hash, source
hashes and donor hashes are archived. These synthetic corruption ranges are
detector stress tests, **not** measurements of a real night camera.

The DINO/structural patch limits and connected-component/area envelope come from
benign controls, not hardcoded cosine constants. The required false-alarm alpha
is an explicit research operating choice. Held-out controls must show a
significant one-sided exact paired-discordance separation (binomial/McNemar)
and no more than the requested
empirical benign false-alarm rate. Otherwise `usable=false`; final audits
abstain. The artifact reports counts and separation p, not quality guarantees.
Performance on these controls does not establish performance on all Qwen edits.
Every training place must supply exactly one benign control and one insertion
control with the same source bytes, geometry, and benign transform recipe.

Only `split=calibration, origin_split=train` controls can fit the detector.
Known evaluation corpora (SVOX, RobotCar, Tokyo, SF-XL, Nordland, Aachen) and
test/val paths are blocked, including the Tokyo/SF tests inspected previously.
`--exclude-places` must include all held-out real validation/calibration places
from the existing strict split protocol. Do not weaken isolation to make the
detector pass.

## Commands (nothing below launches generation or training)

Run from the workspace root. Use new output locations each time. This is a
template: choose the explicit operating alpha before inspecting outcomes.

```bash
PY=/home/admin123/miniconda3/envs/AdaptVPR/bin/python
AUDIT=AdaptVPR/experiments/qwen_curriculum/audit_content_v4.py
BASE=outputs/qwen_curriculum/state/content_audit_v4_01

# 128 approved training places, two machine controls per place.
"$PY" "$AUDIT" controls \
  --manifest outputs/qwen_curriculum/state/finetuning/am_multiseed_v1/train_manifest.jsonl \
  --training-root dataset/gsv-cities --dataset gsv-cities \
  --exclude-places outputs/qwen_curriculum/state/finetuning/am_multiseed_v1/real_pool/excluded_places.json \
  --output-dir "$BASE/controls" --count 128

# Explicit local repository/weights; supports a native DINO state dictionary
# or a complete SALAD checkpoint containing backbone.model.* parameters.
"$PY" "$AUDIT" signals --manifest "$BASE/controls/controls.jsonl" \
  --output "$BASE/control_signals.jsonl" \
  --dino-repo /home/admin123/.cache/torch/hub/facebookresearch_dinov2_main \
  --dino-checkpoint salad/checkpoint/dino_salad.ckpt --device cuda --execute

# Example alpha, a stated study decision rather than a calibrated guarantee.
"$PY" "$AUDIT" calibrate --manifest "$BASE/controls/controls.jsonl" \
  --signals "$BASE/control_signals.jsonl" --output "$BASE/calibration.json" \
  --false-alarm-rate 0.05

"$PY" "$AUDIT" signals --manifest outputs/weather_v4/state/results.jsonl \
  --output "$BASE/candidate_signals.jsonl" \
  --dino-repo /home/admin123/.cache/torch/hub/facebookresearch_dinov2_main \
  --dino-checkpoint salad/checkpoint/dino_salad.ckpt --device cuda --execute

"$PY" "$AUDIT" audit --manifest outputs/weather_v4/state/results.jsonl \
  --signals "$BASE/candidate_signals.jsonl" --calibration "$BASE/calibration.json" \
  --output-dir "$BASE/audited"
```

All phases refuse to overwrite outputs. `signals` without `--execute` only
prints a plan. Features are computed online; only small distance/validity maps
are saved, never a many-GB teacher-feature cache. Provider fingerprints include
weight SHA256, architecture, local DINO Python implementation, Torch version,
resize, normalization, and search radius. Any mismatch requires recalibration.

For optional precomputed OCR evidence use `signals --ocr-cache DIRECTORY`.
Each filename is `sha256(JSON-encoded sample_id).json` using the module's
`digest(sample_id)`, with `source_sha256`, `output_sha256`, and lists `source`
and `generated`. Each detection needs `box_xyxy_normalized: [x0,y0,x1,y1]` and
may include `text`. Coordinates refer to the original image, not an OCR resize.
Mismatched image hashes make that pair unknown. No OCR model is downloaded.

`audited/audited_manifest.jsonl` retains every candidate and decision reason.
`audited/training_manifest.jsonl` includes only passes, with `passed=true` and
`eligible_for_training=true`; pre-content flags and source row fingerprints are
preserved. Point subsequent training explicitly to this separate manifest.
Historical generation seals are retained as `original_result_sha256`, not as
stale top-level `result_sha256`. Each new row has `audit_record_sha256` covering
all its other fields, as well as the original input row fingerprint.

CPU verification:

```bash
"$PY" -m unittest discover -s AdaptVPR/experiments/qwen_curriculum/tests \
  -p test_content_audit_v4.py -v
```
