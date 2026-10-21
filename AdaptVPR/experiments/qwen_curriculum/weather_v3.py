"""Generate a v3 weather set: several sources and domains per place, varied lighting.

Separate from the sealed 23-city campaign (its plan, journals and frozen v2 code
are untouched). Commands:

  plan      choose places and write plan.jsonl (no model is loaded)
  generate  POST each unfinished job to the local Qwen service, verify, append
  manifest  rewrite training_manifest.jsonl from results.jsonl
  reverify  re-score every saved output with the current verifier (no Qwen calls)
  requalify re-accept v2 campaign images that pass the v3 geometry limits
  status    print counts per domain/variant

Design. A place gets ``sources_per_place`` distinct real views, each edited into
``domains_per_source`` different domains, so one training visit can hold the same
place under several conditions (M replaces exact sources; B0 queries them).
Domain quotas follow ``prompt_bank.DOMAIN_WEIGHTS`` exactly (largest remainder);
variants are uniform within a domain and keyed by job, so a pilot can calibrate
variants with salad/probe_synthetic_difficulty.py before the full run.

A job interrupted after its POST is regenerated on the next start: its seed and
prompt are fixed, and one Qwen call costs ~25 s.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import random
import sys
import time

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import common
common.use_adaptvpr()
from experiments.qwen_curriculum import prompt_bank

DEFAULT_RUN = common.WORKSPACE_ROOT / "outputs/qwen_curriculum/weather_v3"
MANIFEST_FIELDS = ("sample_id", "city", "place_id", "condition", "prompt_variant", "source_path", "output_path",
                   "source_sha256", "output_sha256", "passed", "eligible_for_training", "geometry_ok", "weather_ok")


def read_rows(path):
    """Read JSONL; a torn final line from an interrupted append is dropped."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if number != len(lines):
                raise ValueError(f"{path}:{number}: malformed JSON before the final line")
    return rows


def append_row(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def quotas(total, weights):
    raw = {k: total * w for k, w in weights.items()}
    result = {k: int(v) for k, v in raw.items()}
    for key in sorted(raw, key=lambda k: (result[k] - raw[k], k))[:total - sum(result.values())]:
        result[key] += 1
    return result


def eligible_places(real_data, min_images, excluded):
    places = defaultdict(list)
    for csv_path in sorted((real_data / "Dataframes").glob("*.csv")):
        city = csv_path.stem
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                place = (city, int(row["place_id"]))
                if place in excluded:
                    continue
                # Same spelling as salad/workflow/training_data.py::_metadata_filename.
                name = (f"{row['city_id'].strip()}_{int(row['place_id']):07d}_{int(row['year']):04d}_"
                        f"{int(row['month']):02d}_{int(row['northdeg']):03d}_{row['lat']}_{row['lon']}_{row['panoid']}.jpg")
                path = real_data / "Images" / city / name
                if not path.is_file():
                    raise FileNotFoundError(f"GSV metadata image missing: {path}")
                places[place].append(path)
    return {k: sorted(v) for k, v in places.items() if len(v) >= min_images}


def plan(args):
    from PIL import Image
    run_dir = args.run_dir
    if (run_dir / "plan.jsonl").exists():
        raise ValueError(f"Plan already exists in {run_dir}; use a new --run-dir")
    excluded = set()
    if args.exclude_places:
        excluded = {tuple(row) for row in json.loads(args.exclude_places.read_text())}
    if args.skip_manifest:
        for line in args.skip_manifest.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                excluded.add((row["city"], int(row["place_id"])))
    places = eligible_places(args.real_data, args.min_images, excluded)
    rng = random.Random(args.seed)
    by_city = defaultdict(list)
    for key in sorted(places):
        by_city[key[0]].append(key)
    for keys in by_city.values():
        rng.shuffle(keys)
    order, cursor = [], 0
    while len(order) < args.places and any(cursor < len(v) for v in by_city.values()):
        order += [v[cursor] for _, v in sorted(by_city.items()) if cursor < len(v)]
        cursor += 1
    order = order[:args.places]
    slots = len(order) * args.sources_per_place * args.domains_per_source
    domains = [d for d, n in sorted(quotas(slots, prompt_bank.DOMAIN_WEIGHTS).items()) for _ in range(n)]
    rng.shuffle(domains)
    jobs = []
    for city, place_id in order:
        sources = rng.sample(places[city, place_id], args.sources_per_place)
        for source in sources:
            picked = []
            for _ in range(args.domains_per_source):
                # Prefer a domain this source does not have yet.
                index = next((i for i, d in enumerate(domains) if d not in picked), 0)
                picked.append(domains.pop(index))
            for domain in picked:
                key = f"{source.name}/{domain}"
                variant = prompt_bank.choose_variant(domain, key)
                sample_id = "qwen3_" + common.fingerprint([prompt_bank.POLICY_VERSION, key])[:24]
                with Image.open(source) as image:
                    dimensions = list(image.size)
                jobs.append(dict(sample_id=sample_id, city=city, place_id=place_id, source_path=str(source),
                                 source_sha256=common.file_sha256(source), source_dimensions=dimensions,
                                 condition=domain, prompt_variant=variant,
                                 prompt=prompt_bank.prompt(domain, variant),
                                 seed=common.candidate_seed(args.seed, sample_id, 0),
                                 prompt_policy=prompt_bank.POLICY_VERSION))
    config = dict(schema_version=1, stage="weather_v3_plan", real_data=str(args.real_data), seed=args.seed,
                  places=len(order), sources_per_place=args.sources_per_place,
                  domains_per_source=args.domains_per_source, min_images=args.min_images,
                  excluded_places=len(excluded), num_jobs=len(jobs), prompt_bank=prompt_bank.definition(),
                  domain_counts=dict(Counter(j["condition"] for j in jobs)),
                  variant_counts=dict(Counter(f"{j['condition']}/{j['prompt_variant']}" for j in jobs)))
    run_dir.mkdir(parents=True, exist_ok=True)
    common.write_jsonl(run_dir / "plan.jsonl", jobs)
    config["plan_sha256"] = common.file_sha256(run_dir / "plan.jsonl")
    common.write_json(run_dir / "plan_config.json", {**config, "fingerprint": common.fingerprint(config)})
    print(json.dumps({k: config[k] for k in ("places", "num_jobs", "domain_counts")}, indent=1))


def load_plan(run_dir):
    config = json.loads((run_dir / "plan_config.json").read_text())
    expected = config.pop("fingerprint")
    if common.fingerprint(config) != expected or common.file_sha256(run_dir / "plan.jsonl") != config["plan_sha256"]:
        raise ValueError("Plan checksum mismatch")
    return config, read_rows(run_dir / "plan.jsonl")


def generate(args):
    import requests
    from PIL import Image
    from experiments.qwen_curriculum import run
    from experiments.qwen_curriculum.quality_v3 import QwenQualityVerifierV3

    run_dir = args.run_dir
    with run.run_lock(run_dir):
        config, jobs = load_plan(run_dir)
        session = requests.Session()
        session.trust_env = False
        health = session.get(args.qwen_url.rsplit("/", 1)[0] + "/health", timeout=15)
        health.raise_for_status()
        service = run.service_identity(health.json())
        verifier = QwenQualityVerifierV3(device=args.device, clip_device="cpu")
        execution = dict(plan_sha256=config["plan_sha256"],
                         service=service, quality=verifier.definition)
        execution_path = run_dir / "execution_config.json"
        if execution_path.exists() and json.loads(execution_path.read_text()) != execution:
            raise ValueError("Service or quality settings changed; use a new --run-dir")
        common.write_json(execution_path, execution)
        results_path = run_dir / "results.jsonl"
        rows = read_rows(results_path)
        common.write_jsonl(results_path, rows)  # drop a torn final line before appending
        done = {row["sample_id"] for row in rows}
        for name in ("images", "raw", "inputs"):
            (run_dir / name).mkdir(exist_ok=True)
        calls = 0
        for job in jobs:
            if job["sample_id"] in done:
                continue
            if args.max_calls and calls >= args.max_calls:
                break
            calls += 1
            started = time.monotonic()
            input_path = run_dir / "inputs" / f"{job['sample_id']}.png"
            with Image.open(job["source_path"]) as image:
                source = image.convert("RGB")
            if common.file_sha256(Path(job["source_path"])) != job["source_sha256"]:
                raise ValueError(f"Source changed: {job['source_path']}")
            source.save(input_path)
            try:
                response = session.post(args.qwen_url, timeout=args.request_timeout, json=dict(
                    image_path=str(input_path), prompt=job["prompt"], negative_prompt="", seed=job["seed"],
                    infer_steps=4, guidance_scale=1.0))
                response.raise_for_status()
                sampling = response.json()
            finally:
                input_path.unlink(missing_ok=True)
            result = Path(sampling["result_path"]).resolve()
            if result.parent != run.service_output_directory() or not result.is_file():
                raise ValueError("Qwen returned a path outside its output directory")
            raw_path = run_dir / "raw" / f"{job['sample_id']}.png"
            output_path = run_dir / "images" / f"{job['sample_id']}.png"
            with Image.open(result) as image:
                output = image.convert("RGB").resize(tuple(job["source_dimensions"]), Image.Resampling.LANCZOS)
            result.replace(raw_path)
            result.with_suffix(".json").unlink(missing_ok=True)
            temporary = output_path.with_suffix(".tmp.png")
            output.save(temporary)
            temporary.replace(output_path)
            verdict = verifier.evaluate(source, output, job["condition"])
            row = {**job, **verdict, "output_path": str(output_path), "output_sha256": common.file_sha256(output_path),
                   "raw_output_path": str(raw_path), "generation_seconds": time.monotonic() - started,
                   "sampling": {k: v for k, v in sampling.items() if k != "result_path"}}
            append_row(results_path, row)
            done.add(job["sample_id"])
            print(f"[{len(done)}/{len(jobs)}] {job['city']} {job['condition']}/{job['prompt_variant']} "
                  f"{row['status']} geo={row['s_geo']:.3f} shift={row['weather_shift']:.2f} "
                  f"luma={row['luminance_generated']:.0f}", flush=True)
        write_manifest(run_dir)


def reverify(args):
    from PIL import Image
    from experiments.qwen_curriculum import run
    from experiments.qwen_curriculum.quality_v3 import QwenQualityVerifierV3

    run_dir = args.run_dir
    with run.run_lock(run_dir):
        rows = read_rows(run_dir / "results.jsonl")
        verifier = QwenQualityVerifierV3(device=args.device, clip_device="cpu")
        updated = []
        for row in rows:
            if common.file_sha256(Path(row["output_path"])) != row["output_sha256"]:
                raise ValueError(f"Image changed: {row['output_path']}")
            with Image.open(row["source_path"]) as source, Image.open(row["output_path"]) as output:
                updated.append({**row, **verifier.evaluate(source.convert("RGB"), output.convert("RGB"),
                                                           row["condition"])})
        execution = json.loads((run_dir / "execution_config.json").read_text())
        common.write_json(run_dir / "execution_config.json", {**execution, "quality": verifier.definition})
        common.write_jsonl(run_dir / "results.jsonl", updated)
    print(json.dumps(write_manifest(run_dir), indent=1))


def write_manifest(run_dir):
    rows = read_rows(run_dir / "results.jsonl")
    accepted = [{k: row[k] for k in MANIFEST_FIELDS if k in row} for row in rows
                if row.get("passed") is True and row.get("eligible_for_training") is True]
    common.write_jsonl(run_dir / "training_manifest.jsonl", accepted)
    summary = dict(results=len(rows), accepted=len(accepted),
                   by_variant={f"{c}/{v}": dict(results=n, accepted=sum(
                       1 for r in accepted if (r["condition"], r["prompt_variant"]) == (c, v)))
                       for (c, v), n in sorted(Counter((r["condition"], r["prompt_variant"]) for r in rows).items())},
                   rejection_reasons=dict(Counter(x for r in rows for x in r.get("rejection_reasons", []))),
                   places_with_accepted=len({(r["city"], r["place_id"]) for r in accepted}),
                   updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    common.write_json(run_dir / "summary.json", summary)
    return summary


def requalify(args):
    """Apply v3 geometry limits to recorded v2 measurements; no image is re-scored.

    Only rows with a saved output, passing v2 weather checks and failing v2 only
    on geometry are considered. Output rows keep their v2 audit fields.
    """
    from experiments.qwen_curriculum.quality_v3 import THRESHOLDS
    accepted, counts = [], Counter()
    for row in read_rows(args.results):
        if not row.get("output_path") or "s_geo" not in row or row.get("passed") is True:
            continue
        displacement = row.get("estimated_H_grid_displacement_median_px")
        geometry = (row["s_geo"] >= THRESHOLDS["min_s_geo"] and row["num_matched"] >= 16
                    and displacement is not None
                    and displacement <= THRESHOLDS["max_H_grid_median_px_at_512"] * row["matcher_image_size"] / 512)
        if geometry and row.get("weather_ok") is True and "geometry_canvas_dimensions_differ" not in row["rejection_reasons"]:
            if common.file_sha256(Path(row["output_path"])) != row["output_sha256"]:
                raise ValueError(f"Image changed: {row['output_path']}")
            accepted.append({**{k: row[k] for k in MANIFEST_FIELDS if k in row}, "passed": True,
                             "eligible_for_training": True, "geometry_ok": True,
                             "requalified": "v3_geometry_limits", "prompt_variant": f"v2_{row['condition']}"})
            counts[row["condition"]] += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    common.write_jsonl(args.output, accepted)
    print(json.dumps(dict(output=str(args.output), requalified=len(accepted), by_condition=dict(counts)), indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "generate", "manifest", "reverify", "requalify", "status"):
        child = sub.add_parser(name)
        child.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
        if name == "requalify":
            child.add_argument("--results", type=Path,
                               default=common.WORKSPACE_ROOT / "outputs/qwen_curriculum/state/results.jsonl")
            child.add_argument("--output", type=Path,
                               default=common.WORKSPACE_ROOT / "outputs/qwen_curriculum/state/requalified_v3_manifest.jsonl")
        if name == "plan":
            child.add_argument("--real-data", type=Path, default=common.WORKSPACE_ROOT / "dataset/gsv-cities")
            child.add_argument("--exclude-places", type=Path, help="JSON list of [city, place_id] held out from training")
            child.add_argument("--skip-manifest", type=Path, help="Skip places already in this training manifest")
            child.add_argument("--places", type=int, required=True)
            child.add_argument("--sources-per-place", type=int, default=2)
            child.add_argument("--domains-per-source", type=int, default=2)
            child.add_argument("--min-images", type=int, default=4)
            child.add_argument("--seed", type=int, default=3)
        if name in ("generate", "reverify"):
            child.add_argument("--device", default="cuda")
        if name == "generate":
            child.add_argument("--qwen-url", default="http://127.0.0.1:8001/generate")
            child.add_argument("--request-timeout", type=float, default=600.0)
            child.add_argument("--max-calls", type=int, default=0, help="0 = all unfinished jobs")
    args = parser.parse_args(argv)
    args.run_dir = args.run_dir.expanduser().resolve()
    if args.command == "plan":
        if args.places < 1 or not 1 <= args.domains_per_source <= len(prompt_bank.VARIANTS):
            parser.error("--places must be positive and --domains-per-source within the number of domains")
        if args.sources_per_place < 1 or args.sources_per_place >= args.min_images:
            parser.error("--sources-per-place must leave at least one untouched real view (< --min-images)")
        plan(args)
    elif args.command == "generate":
        generate(args)
    elif args.command == "reverify":
        reverify(args)
    elif args.command == "requalify":
        requalify(args)
    elif args.command == "manifest":
        print(json.dumps(write_manifest(args.run_dir), indent=1))
    else:
        print((args.run_dir / "summary.json").read_text() if (args.run_dir / "summary.json").exists()
              else json.dumps(load_plan(args.run_dir)[0]["domain_counts"]))


if __name__ == "__main__":
    main()
