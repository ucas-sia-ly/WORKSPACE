# AdaptCities 160K local regeneration

The official release contains 160,000 prompts, plus **example-only** metadata.
This pipeline creates new annotations from real local generation and verification.
Passing images are a subset of the 160,000 tasks; failed quality checks remain
completed tasks with images under `images/rejected/`.

Use the existing AdaptVPR Conda Python. Local overrides live in
`workspace/AdaptCities/runtime.env`; they point to the existing GSV-Cities/model
files and dedicate service temporary files to this run. No originals are copied.

```bash
PYTHON=/home/admin123/miniconda3/envs/AdaptVPR/bin/python
# Start services, prepare fixed shards, and run/resume in the background.
$PYTHON scripts/start_adaptcities.py
# Inspect durable progress, including whether the recorded process is alive.
$PYTHON scripts/adaptcities.py status
# Explicit preparation or foreground resume, after services are ready.
$PYTHON scripts/adaptcities.py prepare
# Inference must run inside the protected service; direct unbounded runs refuse.
# Export/revalidate a stopped or finished run.
$PYTHON scripts/adaptcities.py export
```

`prepare`, `run`, and `export` accept `--data-dir`, `--output`, `--image-root`,
and `--env-file`. Output defaults to `workspace/AdaptCities/run`. A process lock
prevents concurrent writers. `run` always resumes; it verifies configuration,
code, model file inventory, source hashes, output hashes and image decoding before
skipping records. Changing configuration requires a separate output directory.

The first shard contains the first prompt for each city/route combination (69
tasks). Remaining shards preserve official order, at most 1,000 tasks each. IDs
are never deduplicated by source image. Base seed is zero; the underlying agent
records each actual seed. Models, reflection policy and thresholds match the
existing public pipeline. All initial prompts remain unchanged.

`progress.json` records the active sample, counts, measured generation time,
remaining-time estimate and free disk. `pilot_report.json` estimates full-run
storage with a 1.5 safety factor and a 30 GiB reserve. The runner stops before
that reserve is exhausted. Infrastructure exceptions receive three retries
after the initial attempt, then stop the queue with `stopped_incomplete`.
SIGINT/SIGTERM exports completed work; SIGKILL/power loss is recovered from
atomically committed per-sample records. Uncommitted images are regenerated.

The process runs continuously while this machine is on. After reboot or a
stopped run, invoke `start_adaptcities.py` again. Do not interrupt unrelated GPU
processes to make room. Service logs and the supervisor log are in
`workspace/AdaptCities/logs/`.

## 32 GB host RAM protection

The launcher uses a dedicated user systemd service, separate from the editor's
cgroup. All host generators, verifier, batch process and descendants together
have `MemoryHigh=8G`, `MemoryMax=10G`, `MemorySwapMax=0`, and four CPU cores worth
of quota. Qwen's Docker container has a separate 6 GiB RAM limit, zero swap, and
four CPU cores worth of quota. Automatic container restart is disabled.
Torch compilation is disabled; compiler concurrency is limited to one and CPU
math threads to two. These limits do not change the published prompts, model
weights, generation rounds or acceptance thresholds.

A separate 128 MiB watchdog checks system available RAM every two seconds. It
stops this task and its Qwen container after three checks below 6 GiB, or
immediately below 3 GiB. It also stops Qwen when the main service exits. Monitor
`AdaptCities/memory_watchdog.json`; it reports cgroup usage, peaks, limits and OOM
events. Starting requires at least 16 GiB available RAM. The existing system
swap is not reset; this task's cgroups are prohibited from adding swapped pages.

```bash
systemctl --user status adaptcities-generation.service
systemctl --user stop adaptcities-generation.service
```

Resource-profile changes are recorded in the experiment provenance. Explicitly
validated earlier profiles can appear in `compatible_profile_fingerprints`;
their complete prior manifests are archived under `provenance/`. Old sample
records keep their original fingerprint and measured values.

Exports are atomically replaced after the pilot and on normal completion or a
handled stop: `annotations.jsonl`, `metadata.jsonl`, `train_manifest.jsonl`, and
`summary.json`. Per-sample records are committed throughout the run. Metadata
contains original/actual prompts, round seeds and scores, image SHA-256s, and a
configuration fingerprint linking to model/code provenance in `experiment.json`.
Official example files remain byte-for-byte intact and are tagged as
`example_only_files` in the release provenance; they never enter real exports.
`summary.json.complete` is true only for 160,000 valid records and zero remaining
infrastructure errors. Exact reconstruction of unpublished author images is not
claimed.

Model loading in this local profile uses `HF_HUB_OFFLINE=1`; all required weights
are already cached. The runner initializes SuperPoint/LightGlue before the first
generation and rejects a fallback matcher, so a network check or missing cached
weight cannot silently change the verification method.
