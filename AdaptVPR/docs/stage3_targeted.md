# Stage3 targeted input reader

This is an independent, model-free dry-run entrypoint requiring only Pillow and
the Python standard library. It never imports the
planner, scheduler, generation backends, VLM, reflection or verifier. The existing
`run.py` plan/prompt modes and their contracts remain unchanged.

From the AdaptVPR directory:

```bash
python scripts/run_targeted.py ../../Bag-of-Queries/outputs/stage3/targets/targets.jsonl --check-only
python scripts/run_targeted.py ../../Bag-of-Queries/outputs/stage3/targets/targets.jsonl --seed 0
python scripts/run_targeted.py ../../Bag-of-Queries/outputs/stage3/targets/targets.jsonl --seed 0 --resume
```

The default destination is `AdaptVPR/outputs/stage3_targeted/tasks.jsonl`, resolved
relative to the script's repository, independently of the current directory.
`--output DIRECTORY` selects a different destination. All invocations are dry-run:
no image generation or prompt construction is implemented.

- `--check-only`: validate records and decode selected sources/masks; write
  nothing and create no output directory. With `--resume`, also validate the
  existing task prefix. Without `--resume`, an existing output is left untouched.
- `--limit N`: select the first N records in manifest order. Zero means all.
  Schema and unique sample IDs are checked across the entire manifest before
  applying the limit; source/mask decoding is limited to selected records.
- `--seed N`: a nonnegative task seed, default 0. It does not resample or change
  Stage2 masks. The original Stage2 seed is retained in `stage2_metadata.seed`.
- `--resume`: revalidate inputs and require existing tasks to be an identical
  prefix (same input hash, seed, image/mask hashes and normalized fields).
  Increasing a previous limit appends the remaining tasks through an atomic
  rewrite; rerunning a complete export writes nothing. Decreasing the limit
  below the existing task count or changing input/seed fails explicitly. An
  existing output without `--resume` is never overwritten.

## Input and validation

Accepts Bag-of-Queries target schema **integer 1**. Each row requires `sample_id`,
`image_key`, `place_key`, `source_path`, `mask_original_path`, `target_type`,
`mask_ratio`, `clean_margin`, `mask_mode`, `mask_token_count`, `checkpoint_path`,
`checkpoint_sha256`, `stage2_commit`, and Stage2 `seed`, in addition to
`schema_version`. Missing metadata is an error. `attention` is primary; `fused`
is supplementary. Unknown types or contradictory roles are rejected.

Absolute file paths remain absolute; relative source/mask paths resolve against
the directory containing `targets.jsonl`, not the working directory. SOURCE path
components must end with the unchanged relative `image_key`. Files must exist and
decode successfully. Mask dimensions must equal the raw decoded source dimensions
(no EXIF transpose). Masks must be nonempty single-channel PNGs with values 0/1,
as specified by the Bag-of-Queries schema 1 exporter. There is no thresholding,
resizing or automatic conversion of 0/255 masks. Declared dimensions, pixel area
and optional source/mask SHA256 values are checked when present.

A sibling `export_manifest.json`, if present, must have supported schema, matching
record count and matching `targets_sha256`. Standalone JSONL is supported too.
Empty manifests, duplicate IDs/JSON fields, malformed JSON, nonfinite metrics and
invalid ratios fail before any output is published. The reader does not load or
require the checkpoint file; it preserves its Stage2 identity as metadata.

## Normalized output

Each task preserves `sample_id`, `image_key`, `place_key`, target type/role,
requested ratio and clean margin, and records absolute source/mask paths,
measured dimensions, original mask pixel area, and source/mask/manifest hashes.
Additional original metadata, including Stage2 seed, checkpoint identity, commit,
mask mode/token count and Bag-of-Queries HEAD, is retained in `stage2_metadata`.

Fixed task fields are `mode="targeted"`, `route="local"`,
`targeting="provided_mask"`, `status="validated"`, `dry_run=true`,
`generated=false`, and `task_schema_version=1`. There is no generated prompt or
inferred position. Source images and masks are only read, never copied/modified.
For fixed inputs and task seed, output bytes are deterministic. Publication is
atomic; validation failures leave no partial tasks file. Concurrent writers to
the same output directory are not supported.

```bash
python -m unittest discover -s tests -p test_targeted_inputs.py -v
```
