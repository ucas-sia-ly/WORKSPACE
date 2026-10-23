"""One suitable source per randomly ordered place; finish the night quota first.

The old campaign and pilot are read-only. Domain folders and the independent
sealed plan live in outputs/weather_v3 and its state subdirectory.
Reservations precede every POST; saved raw/sidecar pairs are recovered before
another request, and saved images are verified without regeneration.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
from types import SimpleNamespace

import requests
from PIL import Image

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import (common, run, campaign, prompt_bank, quality_v3,
    resume_recovery, resume_control, weather_io, source_screen_v3, v3_reuse)
from experiments.qwen_curriculum import source_reliability_v3
from experiments.qwen_curriculum.quality_v3 import QwenQualityVerifierV3

LEGACY_ROOT = common.WORKSPACE_ROOT / "outputs/qwen_curriculum"
ROOT = common.WORKSPACE_ROOT / "outputs/weather_v3"
DEFAULT_OUTPUT = ROOT / "state"
DEFAULT_IMAGES = ROOT
DOMAINS = tuple(d for d, weight in prompt_bank.DOMAIN_WEIGHTS.items() if weight > 0)


class IncompleteRaw(ValueError):
    """A reserved output is unusable; an explicit resume may try again once."""


def _freeze(path, value):
    return campaign.frozen(path, value)


def _rows(path):
    return weather_io.read_rows(path)


def _append(path, row):
    weather_io.append_row(path, row)
    resume_recovery._sync_directory(path.parent)


def _copy(source, target, digest):
    target.parent.mkdir(parents=True, exist_ok=True)
    resume_recovery._copy_atomic(Path(source), target, digest)


def paths(config, job):
    root = Path(config["image_root"])
    name = f"{job['sample_id']}.png"
    return root / job["condition"] / name, root / "raw" / job["condition"] / name


def make_plan(sources, candidates, seed=42):
    """Assign domains independently of city and source ordering, without replacement."""
    ordered = sorted(sources, key=lambda r: (r["city"], int(r["place_id"])))
    if len({(r["city"], int(r["place_id"])) for r in ordered}) != len(ordered):
        raise ValueError("Each place must appear exactly once")
    rng = random.Random(seed)
    rng.shuffle(ordered)
    quotas = weather_io.quotas(len(ordered), prompt_bank.DOMAIN_WEIGHTS)
    domains = [d for d, count in sorted(quotas.items()) for _ in range(count)]
    rng.shuffle(domains)
    jobs, reused = [], {}
    original_owners = {r["source_sha256"]: (r["city"], int(r["place_id"])) for r in sources}
    used_hashes = set()
    for source, domain in zip(ordered, domains):
        source = dict(source)
        selected = None
        key = (source["city"], int(source["place_id"]), domain)
        for candidate in candidates.get(key, []):
            owner = original_owners.get(candidate["source_sha256"])
            if (owner is not None and owner != key[:2]) or candidate["source_sha256"] in used_hashes:
                continue
            screened = source_screen_v3.screen_image(Path(candidate["source_path"]))
            if screened["passed"] and screened["source_sha256"] == candidate["source_sha256"]:
                selected = candidate
                source.update({k: candidate[k] for k in ("source_path", "source_sha256", "source_dimensions")})
                break
        variant_key = f"{Path(source['source_path']).name}/{domain}"
        variant = selected["prompt_variant"] if selected else prompt_bank.choose_variant(domain, variant_key)
        sid = "qwen3_" + common.fingerprint([prompt_bank.POLICY_VERSION, variant_key, source["source_sha256"]])[:24]
        job = {k: source[k] for k in ("city", "place_id", "source_path", "source_sha256", "source_dimensions")}
        job.update(sample_id=sid, condition=domain, prompt_variant=variant,
            prompt=prompt_bank.prompt(domain, variant), negative_prompt="", prompt_policy=prompt_bank.POLICY_VERSION,
            seed=selected["seed"] if selected else common.candidate_seed(seed, sid, 0))
        job["record_sha256"] = common.fingerprint(job)
        jobs.append(job)
        used_hashes.add(job["source_sha256"])
        if selected:
            reused[sid] = selected
    if len({j["source_sha256"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate source content across places after reuse selection")
    return jobs, reused


def prepare(args):
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    with run.run_lock(directory):
        if (directory / "plan_config.json").exists():
            config, jobs = load_plan(directory)
            if config["seed"] != args.seed or config["real_data"] != str(args.real_data):
                raise ValueError("Existing plan uses another source pool or seed")
            finish_import(directory, config, jobs)
            return config
        sources = source_screen_v3.screen_sources(args.real_data, directory / "source_screen",
            seed=args.seed, min_images=args.min_images, workers=args.workers)
        if not sources:
            raise ValueError("No suitable places; no generation calls will be made")
        sources, reliability = source_reliability_v3.refine_sources(
            sources, args.real_data, directory / "source_reliability", seed=args.seed)
        if not sources:
            raise ValueError("No sources pass both CPU and reliability gates")
        expected = json.loads((LEGACY_ROOT / "state/all_city_campaign_config.json").read_text())["reference_execution"]["service"]
        candidates, reuse_report = v3_reuse.collect_candidates(args.reuse_results, expected_service=expected,
            expected_quality=quality_v3.quality_definition_v3(clip_device="cpu"))
        unreliable = set(reliability["unreliable_source_paths"])
        candidates = {key: [r for r in rows if r["source_path"] not in unreliable] for key, rows in candidates.items()}
        jobs, reused = make_plan(sources, candidates, args.seed)
        common.write_jsonl(directory / "plan.jsonl", jobs)
        # A restart can finish the import from these authenticated candidates.
        common.write_jsonl(directory / "reuse_candidates.jsonl", [{"sample_id": sid, "candidate": row} for sid, row in reused.items()])
        common.write_json(directory / "reuse_report.json", reuse_report)
        modules = [sys.modules[__name__], prompt_bank, quality_v3, source_screen_v3, source_reliability_v3,
                   v3_reuse, run, common, resume_recovery, weather_io]
        static = [Path(m.__file__).resolve() for m in modules]
        definition = run.execution_definition(SimpleNamespace(run_dir=directory, device="cuda", cpu_threads=4))
        static += [common.ADAPTVPR_ROOT / name for name in definition["implementation_sha256"]]
        static.append(common.ADAPTVPR_ROOT / "adapters/lightx2v_qwen_image_edit.py")
        static += [directory / "reuse_candidates.jsonl", directory / "reuse_report.json"]
        static += [p for p in (directory / "source_screen/screen_config.json",
                              directory / "source_screen/selected_sources.jsonl") if p.is_file()]
        static += [p for p in (directory / "source_reliability/reliability_config.json",
                              directory / "source_reliability/reliability_report.json",
                              directory / "source_reliability/refined_sources.jsonl") if p.is_file()]
        static += sorted((args.real_data / "Dataframes").glob("*.csv"))
        config = _freeze(directory / "plan_config.json", {
            "schema_version": 1, "stage": "place_coverage_weather_v3", "seed": args.seed,
            "real_data": str(args.real_data), "image_root": str(args.image_root),
            "num_places": len(jobs), "num_jobs": len(jobs), "views_per_place": 1,
            "selection": "global seeded shuffle of suitable places without replacement; independent shuffled domain quotas",
            "source_quality": "CPU exposure/decoding/blur/texture checks plus archived original-SALAD positive reliability >= 0.15; no hardness quantiles",
            "domain_counts": dict(Counter(j["condition"] for j in jobs)), "prompt_bank": prompt_bank.definition(),
            "service": expected, "quality": quality_v3.quality_definition_v3(clip_device="cpu"),
            "default_domain": "night", "plan_sha256": common.file_sha256(directory / "plan.jsonl"),
            "files_sha256": {str(p): common.file_sha256(p) for p in static}})
        finish_import(directory, config, jobs)
        return config


def load_plan(directory):
    config = campaign.config_file(directory / "plan_config.json")
    if config["prompt_bank"] != prompt_bank.definition():
        raise ValueError("Prompt bank changed; preserve this plan or use a new campaign")
    if common.file_sha256(directory / "plan.jsonl") != config["plan_sha256"]:
        raise ValueError("Plan checksum mismatch")
    for path, digest in config["files_sha256"].items():
        if common.file_sha256(Path(path)) != digest:
            raise ValueError(f"Frozen input/code changed: {path}")
    jobs = _rows(directory / "plan.jsonl")
    if (len(jobs) != config["num_jobs"] or len({(j["city"], j["place_id"]) for j in jobs}) != len(jobs)
            or Counter(j["condition"] for j in jobs) != Counter(config["domain_counts"])):
        raise ValueError("Place coverage or domain quota changed")
    for job in jobs:
        body = {k: v for k, v in job.items() if k != "record_sha256"}
        if (job["record_sha256"] != common.fingerprint(body) or job["condition"] not in DOMAINS
                or job["prompt"] != prompt_bank.prompt(job["condition"], job["prompt_variant"])):
            raise ValueError("Invalid planned job")
    return config, jobs


def finish_import(directory, config, jobs):
    if (directory / "import_complete.json").exists():
        return
    generated = {r["sample_id"]: r for r in _rows(directory / "generated.jsonl")}
    results = {r["sample_id"]: r for r in _rows(directory / "results.jsonl")}
    candidates = {r["sample_id"]: r["candidate"] for r in _rows(directory / "reuse_candidates.jsonl")}
    for job in jobs:
        sid = job["sample_id"]
        if sid not in candidates or sid in results:
            continue
        source = candidates[sid]
        output, raw = paths(config, job)
        _copy(Path(source["output_path"]), output, source["output_sha256"])
        _copy(Path(source["raw_output_path"]), raw, source["raw_output_sha256"])
        saved = run.sealed({**job, "output_path": str(output), "output_sha256": source["output_sha256"],
            "raw_output_path": str(raw), "raw_output_sha256": source["raw_output_sha256"],
            "origin": "compatible_prompt_cached_copy", "reuse_provenance": source["reuse_provenance"]})
        if sid not in generated:
            _append(directory / "generated.jsonl", saved)
            generated[sid] = saved
        # Keep measurements, while making the new planned source/seed explicit.
        verdict = {k: v for k, v in source.items() if k not in set(job) | {
            "result_sha256", "output_path", "raw_output_path", "output_sha256", "raw_output_sha256", "reuse_provenance"}}
        result = run.sealed({**verdict, **saved})
        _append(directory / "results.jsonl", result)
        results[sid] = result
    rebuild_manifest(directory, config, list(results.values()))
    publish(directory, config, jobs, generated, results, [], "prepared", "night")
    common.write_json(directory / "import_complete.json", {"reused": len(candidates), "new_generation_calls": 0})


def _accepted(config, row):
    if row.get("passed") is not True or row.get("eligible_for_training") is not True:
        return
    path = Path(config["image_root"]) / "accepted" / row["condition"] / f"{row['sample_id']}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    target = f"../../{row['condition']}/{row['sample_id']}.png"
    if path.is_symlink() and os.readlink(path) == target:
        return
    if path.exists() or path.is_symlink():
        raise ValueError(f"Conflicting accepted entry: {path}")
    path.symlink_to(target)
    resume_recovery._sync_directory(path.parent)


def rebuild_manifest(directory, config, results):
    accepted = [r for r in results if r.get("passed") is True and r.get("eligible_for_training") is True]
    common.write_jsonl(directory / "training_manifest.jsonl", accepted)
    for row in accepted:
        _accepted(config, row)


def publish(directory, config, jobs, generated, results, attempts, state, domain, error=None):
    counts = {}
    for condition, planned in config["domain_counts"].items():
        saved = [r for r in generated.values() if r["condition"] == condition]
        evaluated = [r for r in results.values() if r["condition"] == condition]
        counts[condition] = {"planned": planned, "saved": len(saved), "evaluated": len(evaluated),
            "accepted": sum(r.get("passed") is True for r in evaluated),
            "reused": sum(r.get("origin") == "compatible_prompt_cached_copy" for r in saved),
            "remaining_generation": planned - len(saved)}
    summary = {"state": state, "active_domain": domain, "planned": len(jobs), "places": len(jobs),
        "saved": len(generated), "evaluated": len(results), "accepted": sum(r.get("passed") is True for r in results.values()),
        "calls_reserved": len(attempts), "by_domain": counts, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if error:
        summary["error"] = error
    common.write_json(directory / "summary.json", summary)
    return summary


def _read_state(directory, jobs, config):
    by_id = {j["sample_id"]: j for j in jobs}
    journals = {}
    verified = set()
    for name in ("generated", "results", "attempts"):
        rows = _rows(directory / f"{name}.jsonl")
        # Remove a torn final append before any subsequent durable append.
        common.write_jsonl(directory / f"{name}.jsonl", rows)
        for row in rows:
            job = by_id.get(row.get("sample_id"))
            field = "attempt_sha256" if name == "attempts" else "result_sha256"
            body = {k: v for k, v in row.items() if k != field}
            if job is None or common.fingerprint(body) != row.get(field):
                raise ValueError(f"Invalid sealed {name} row")
            if name != "attempts":
                if any(row.get(k) != v for k, v in job.items()) or paths(config, job) != (Path(row["output_path"]), Path(row["raw_output_path"])):
                    raise ValueError("Saved record differs from plan/domain paths")
                for field in ("source_path", "output_path", "raw_output_path"):
                    hash_field = field.replace("_path", "_sha256")
                    key = row[field], row[hash_field]
                    if key not in verified:
                        run.validate_image(row, field, hash_field)
                        verified.add(key)
                if row.get("origin") == "reserved_qwen_request":
                    sidecar = Path(row["raw_output_path"]).with_suffix(".json")
                    if not sidecar.is_file() or common.file_sha256(sidecar) != row.get("metadata_sha256"):
                        raise ValueError("Recorded service sidecar missing or changed")
            elif (row.get("record_sha256") != job["record_sha256"] or row.get("seed") != job["seed"]
                    or row.get("prompt") != job["prompt"] or row.get("source_sha256") != job["source_sha256"]
                    or row.get("plan_sha256") != config["plan_sha256"]
                    or row.get("call_number") != len(journals.get("checked_attempts", [])) + 1
                    or row.get("workspace") != str(directory / "requests" / f"a{row['call_number']:08d}")):
                raise ValueError("Request reservation differs from the frozen job")
            if name == "attempts":
                journals.setdefault("checked_attempts", []).append(row)
        if name != "attempts" and len({r["sample_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate committed image")
        journals[name] = rows
    generated = {r["sample_id"]: r for r in journals["generated"]}
    results = {r["sample_id"]: r for r in journals["results"]}
    if not set(results) <= set(generated):
        raise ValueError("Quality result has no saved image")
    reserved = {r["attempt_sha256"] for r in journals["attempts"]}
    for row in generated.values():
        if row.get("origin") != "compatible_prompt_cached_copy" and row.get("attempt_sha256") not in reserved:
            raise ValueError("Generated image has no durable request reservation")
    return generated, results, journals["attempts"]


def _saved_from_raw(directory, config, job, attempt, metadata_path, metadata, sampling=None):
    output, raw = paths(config, job)
    service_dir = run.service_output_directory()
    resume_recovery._validate_metadata(metadata, Path(attempt["workspace"]), service_dir, job, {"service": config["service"]})
    original = Path(metadata["result_path"])
    sources = [p for p in (raw, original) if p.is_file()]
    if not sources:
        return None
    digests = {common.file_sha256(p) for p in sources}
    if len(digests) != 1:
        raise ValueError("Conflicting interrupted raw copies")
    raw_sha = next(iter(digests))
    # Decode before publishing any canonical file. Partial bytes remain in their
    # original service location as evidence and do not poison future retries.
    try:
        with Image.open(sources[0]) as image:
            image.load()
            if image.format != "PNG" or list(image.size) != metadata["raw_dimensions"]:
                raise IncompleteRaw("Reserved raw has an invalid format or canvas")
            normalized = image.convert("RGB").resize(tuple(job["source_dimensions"]), Image.Resampling.LANCZOS)
    except OSError as exc:
        raise IncompleteRaw(f"Reserved raw cannot be completely decoded: {exc}") from exc
    _copy(sources[0], raw, raw_sha)
    sidecar = raw.with_suffix(".json")
    metadata_sha = common.file_sha256(metadata_path)
    _copy(metadata_path, sidecar, metadata_sha)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.png")
    normalized.save(temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(output)
    resume_recovery._sync_directory(output.parent)
    return run.sealed({**job, "output_path": str(output), "output_sha256": common.file_sha256(output),
        "raw_output_path": str(raw), "raw_output_sha256": raw_sha, "metadata_sha256": metadata_sha,
        "attempt_sha256": attempt["attempt_sha256"], "sampling": sampling or {},
        "origin": "reserved_qwen_request", "recovery_proof": "source-input path, seed and canvas sidecar; request ledger records prompt"})


def recover_saved(directory, config, job, attempts):
    output, raw = paths(config, job)
    for attempt in reversed([a for a in attempts if a["sample_id"] == job["sample_id"]]):
        matches = {}
        for sidecar in [raw.with_suffix(".json"), *sorted(run.service_output_directory().glob("*.json"))]:
            if not sidecar.is_file():
                continue
            try:
                metadata = json.loads(sidecar.read_text())
            except (OSError, ValueError):
                continue
            if not isinstance(metadata, dict):
                continue
            expected = str(Path(attempt["workspace"]) / "source_inputs" / f"{job['sample_id']}.png")
            if metadata.get("source_path") == expected and metadata.get("seed") == job["seed"]:
                key = metadata.get("metadata_path")
                if key in matches and matches[key][1] != metadata:
                    raise ValueError("Conflicting recovery sidecars")
                matches[key] = (sidecar, metadata)
        if len(matches) > 1:
            raise ValueError("Multiple raw outputs for one reserved request")
        if matches:
            sidecar, metadata = next(iter(matches.values()))
            try:
                recovered = _saved_from_raw(directory, config, job, attempt, sidecar, metadata)
            except IncompleteRaw as exc:
                _append(directory / "recovery_rejections.jsonl", {"sample_id": job["sample_id"],
                    "attempt_sha256": attempt["attempt_sha256"], "metadata_path": str(sidecar), "reason": str(exc)})
                continue
            if recovered is not None:
                return recovered
    return None


def _cleanup_request(config, job, attempt):
    if not attempt:
        return
    _, raw = paths(config, job)
    metadata = json.loads(raw.with_suffix(".json").read_text())
    resume_recovery._validate_metadata(metadata, Path(attempt["workspace"]), run.service_output_directory(),
                                       job, {"service": config["service"]})
    for field in ("result_path", "metadata_path"):
        path = Path(metadata[field])
        if path.parent == run.service_output_directory():
            path.unlink(missing_ok=True)
    workspace = Path(attempt["workspace"])
    (workspace / "source_inputs" / f"{job['sample_id']}.png").unlink(missing_ok=True)
    for path in (workspace / "source_inputs", workspace):
        try:
            path.rmdir()
        except OSError:
            pass


def generate(args, *, locked=False):
    directory = args.output_dir
    with nullcontext() if locked else run.run_lock(directory):
        config, jobs = load_plan(directory)
        generated, results, attempts = _read_state(directory, jobs, config)
        rebuild_manifest(directory, config, list(results.values()))
        domain = args.domain
        selected = [j for j in jobs if j["condition"] == domain]
        if domain not in DOMAINS:
            raise ValueError("Overcast/unknown domains cannot be generated")
        if all(j["sample_id"] in results for j in selected):
            return publish(directory, config, jobs, generated, results, attempts, "domain_complete", domain)
        verifier, session, calls = None, None, 0
        try:
            for job in selected:
                sid = job["sample_id"]
                if sid in results:
                    continue
                run.validate_image(job, "source_path", "source_sha256")
                saved = generated.get(sid)
                attempt = None
                if saved is None:
                    saved = recover_saved(directory, config, job, attempts)
                    if saved is not None:
                        _append(directory / "generated.jsonl", saved)
                        generated[sid] = saved
                if saved is None:
                    if args.max_calls and calls >= args.max_calls:
                        return publish(directory, config, jobs, generated, results, attempts, "call_limit", domain)
                    if shutil.disk_usage(directory).free < 5_000_000_000 + 3_000_000 * sum(j["sample_id"] not in generated for j in selected):
                        raise RuntimeError("Insufficient disk for the complete remaining domain")
                    if session is None:
                        session = requests.Session()
                        session.trust_env = False
                        response = session.get(args.qwen_url.rsplit("/", 1)[0] + "/health", timeout=15)
                        response.raise_for_status()
                        if run.service_identity(response.json()) != config["service"]:
                            raise ValueError("Qwen service differs from the frozen proven profile")
                    workspace = directory / "requests" / f"a{len(attempts) + 1:08d}"
                    input_path = workspace / "source_inputs" / f"{sid}.png"
                    input_path.parent.mkdir(parents=True, exist_ok=True)
                    with Image.open(job["source_path"]) as source:
                        source.convert("RGB").save(input_path)
                    with input_path.open("rb") as handle:
                        os.fsync(handle.fileno())
                    resume_recovery._sync_directory(input_path.parent)
                    body = {"sample_id": sid, "record_sha256": job["record_sha256"], "workspace": str(workspace),
                        "call_number": len(attempts) + 1, "seed": job["seed"], "prompt": job["prompt"],
                        "source_sha256": job["source_sha256"], "plan_sha256": config["plan_sha256"],
                        "reserved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                    attempt = {**body, "attempt_sha256": common.fingerprint(body)}
                    _append(directory / "attempts.jsonl", attempt)
                    attempts.append(attempt)
                    calls += 1
                    publish(directory, config, jobs, generated, results, attempts, "generating", domain)
                    response = session.post(args.qwen_url, timeout=args.request_timeout, json={
                        "image_path": str(input_path), "prompt": job["prompt"], "negative_prompt": "",
                        "seed": job["seed"], "infer_steps": 4, "guidance_scale": 1.0})
                    response.raise_for_status()
                    sampling = response.json()
                    raw_path = Path(sampling["result_path"])
                    if raw_path.parent != run.service_output_directory():
                        raise ValueError("Qwen returned an unowned path")
                    sidecar = raw_path.with_suffix(".json")
                    saved = _saved_from_raw(directory, config, job, attempt, sidecar, json.loads(sidecar.read_text()),
                        {k: v for k, v in sampling.items() if k not in {"result_path", "metadata_path"}})
                    if saved is None:
                        raise ValueError("Qwen did not save a complete raw image")
                    _append(directory / "generated.jsonl", saved)
                    generated[sid] = saved
                if saved.get("origin") == "reserved_qwen_request":
                    own = next((a for a in attempts if a["attempt_sha256"] == saved["attempt_sha256"]), None)
                    _cleanup_request(config, job, own)
                if verifier is None:
                    verifier = QwenQualityVerifierV3(device=args.device, clip_device="cpu")
                    if verifier.definition != config["quality"]:
                        raise ValueError("Quality verifier differs from the frozen plan")
                with Image.open(job["source_path"]) as source, Image.open(saved["output_path"]) as output:
                    verdict = verifier.evaluate(source.convert("RGB"), output.convert("RGB"), job["condition"])
                result = run.sealed({**saved, **verdict})
                _append(directory / "results.jsonl", result)
                results[sid] = result
                _accepted(config, result)
                if result.get("passed") is True and result.get("eligible_for_training") is True:
                    _append(directory / "training_manifest.jsonl", result)
                publish(directory, config, jobs, generated, results, attempts, "generating", domain)
                print(f"{domain}: {sum(r['condition'] == domain for r in results.values())}/{len(selected)} {sid} {result['status']}", flush=True)
            return publish(directory, config, jobs, generated, results, attempts, "domain_complete", domain)
        except BaseException as exc:
            publish(directory, config, jobs, generated, results, attempts,
                "paused" if isinstance(exc, KeyboardInterrupt) else "stopped", domain, f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if session is not None:
                session.close()


def execute(args):
    """Own the Qwen service lifetime; pause always releases both services."""
    with run.run_lock(args.output_dir):
        campaign.stop_qwen()
        config, jobs = load_plan(args.output_dir)
        done = {r["sample_id"] for r in _rows(args.output_dir / "results.jsonl")}
        if all(j["sample_id"] in done for j in jobs if j["condition"] == args.domain):
            return generate(args, locked=True)
        try:
            campaign.start_qwen({"reference_execution": {"service": config["service"]}})
            return generate(args, locked=True)
        except BaseException as exc:
            path = args.output_dir / "summary.json"
            if path.exists():
                report = json.loads(path.read_text())
                report.update(state="paused" if isinstance(exc, KeyboardInterrupt) else "stopped",
                              error=f"{type(exc).__name__}: {exc}")
                common.write_json(path, report)
            raise
        finally:
            campaign.stop_qwen()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("prepare", "run", "generate", "check", "status"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--real-data", type=Path, default=common.WORKSPACE_ROOT / "dataset/gsv-cities")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-images", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reuse-results", nargs="+", type=Path, default=[LEGACY_ROOT / "weather_v3_pilot/results.jsonl", LEGACY_ROOT / "state/results.jsonl"])
    parser.add_argument("--domain", choices=DOMAINS, default="night")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--qwen-url", default="http://127.0.0.1:8001/generate")
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--max-calls", type=int, default=0)
    args = parser.parse_args(argv)
    for name in ("output_dir", "image_root", "real_data"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    if args.command == "prepare":
        config = prepare(args)
        print(json.dumps({"planned": config["num_jobs"], "domain_counts": config["domain_counts"]}, indent=2))
    elif args.command in {"run", "generate"}:
        execute(args)
    elif args.command == "check":
        with run.run_lock(args.output_dir):
            config, jobs = load_plan(args.output_dir)
            generated, results, attempts = _read_state(args.output_dir, jobs, config)
            print(json.dumps({"planned": len(jobs), "saved": len(generated), "evaluated": len(results), "calls_reserved": len(attempts)}))
    else:
        print((args.output_dir / "summary.json").read_text())


if __name__ == "__main__":
    main()
