"""Interruptible continuation of the frozen 23-city Qwen campaign.

Original journals remain immutable. Each explicit start grants at most one new
attempt per unfinished job; failed attempts are retained, never erased. Saved
bytes are verified without calling Qwen, before planning any new requests.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import requests

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import (common, run, all_city_campaign,
    scale_campaign, fast_campaign, plan, resume_recovery)
from experiments.qwen_curriculum.campaign import config_file

DEFAULT_OUTPUT = all_city_campaign.DEFAULT_OUTPUT
QWEN_UNIT = "adaptvpr-lightx2v.service"
JOURNALS = ("execution_config.json", "results.jsonl", "generated.jsonl", "attempts.jsonl")


def snapshot_originals(config):
    return {str(Path(s["run_dir"]) / name):
            common.file_sha256(Path(s["run_dir"]) / name)
            if (Path(s["run_dir"]) / name).exists() else None
            for s in config["stages"] for name in JOURNALS}


def prepare(directory, campaign):
    """Called only after the old coordinator and Qwen request have stopped."""
    path = directory / "resume_config.json"
    if path.exists():
        cfg = config_file(path)
    else:
        files = [Path(__file__), Path(resume_recovery.__file__), Path(run.__file__),
                 directory / "all_city_campaign_config.json"]
        cfg = fast_campaign.frozen(path, {
            "schema_version": 1, "campaign_fingerprint": campaign["fingerprint"],
            "original_journals": snapshot_originals(campaign),
            "retry_policy": "one attempt per unresolved job per explicit start; stop on failure",
            "saved_image_policy": "recover or verify existing bytes before any new POST",
            "files_sha256": {str(p.resolve()): common.file_sha256(p) for p in files}})
    if cfg["campaign_fingerprint"] != campaign["fingerprint"]:
        raise ValueError("Resume belongs to another campaign")
    for path, digest in cfg["files_sha256"].items():
        if common.file_sha256(Path(path)) != digest:
            raise ValueError(f"Frozen resume implementation changed: {path}")
    if cfg["original_journals"] != snapshot_originals(campaign):
        raise ValueError("An original journal changed after migration; do not start the old coordinator")
    return cfg


def stop_qwen():
    subprocess.run(["systemctl", "--user", "stop", QWEN_UNIT], check=True)
    pid = subprocess.run(["systemctl", "--user", "show", QWEN_UNIT,
                          "--property=MainPID", "--value"],
                         check=True, capture_output=True, text=True).stdout.strip()
    if pid != "0":
        raise RuntimeError("Qwen still has a running process; recovery is unsafe")


class Catalog:
    """Audit append-only stage journals, caching unchanged files for progress."""
    def __init__(self, directory, campaign):
        self.directory, self.campaign = directory, campaign
        self.jobs = common.read_jsonl(Path(campaign["combined_plan"]))
        self.by_id = {j["sample_id"]: j for j in self.jobs}
        scale_campaign.validate_unique(self.jobs)
        if len(self.by_id) != campaign["total_images"]:
            raise ValueError("Combined plan length changed")
        self.parent, _ = scale_campaign.load_collection(Path(campaign["parent_run_dir"]))
        self.cache = {}
        self.image_cache = {}
        self.refresh()

    def validate_rows(self, rows, execution, jobs):
        """Hash newly saved/changed bytes, not every PNG on every progress tick."""
        planned = {j["sample_id"]: j for j in jobs}
        ids = set()
        for row in rows:
            sid = row["sample_id"]
            job = planned.get(sid)
            if (sid in ids or job is None or row.get("execution_fingerprint") != execution["fingerprint"]
                    or row.get("result_sha256") != run.sealed(row)["result_sha256"]
                    or any(row.get(k) != v for k, v in job.items())):
                raise ValueError("Saved result differs from its sealed execution/job")
            ids.add(sid)
            for field, hash_field in (("source_path", "source_sha256"),
                                      ("output_path", "output_sha256"), ("raw_output_path", "raw_output_sha256")):
                if row.get(field):
                    path = Path(row[field])
                    stat = path.stat()
                    key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, row[hash_field])
                    if key not in self.image_cache:
                        run.validate_image(row, field, hash_field)
                        self.image_cache[key] = True

    def audit_stage(self, stage):
        directory = Path(stage["run_dir"])
        paths = [directory / n for n in (*JOURNALS, "plan_config.json", "plan.jsonl")]
        signature = tuple((p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns)
                          if p.exists() else None for p in paths)
        if directory in self.cache and self.cache[directory][0] == signature:
            return self.cache[directory][1]
        pcfg = config_file(directory / "plan_config.json")
        if (pcfg["fingerprint"] != stage["plan_fingerprint"]
                or common.file_sha256(directory / "plan.jsonl") != pcfg["plan_sha256"]):
            raise ValueError("Continuation plan changed")
        jobs = run.read_rows(directory, "plan.jsonl")
        if len(jobs) != stage["num_images"] or not 1 <= len(jobs) <= 1000:
            raise ValueError("Stage exceeds the original per-execution budget")
        if any(self.by_id.get(j["sample_id"]) != j for j in jobs):
            raise ValueError("Continuation changed an exact original job")
        results, generated, attempts = [run.read_rows(directory, n) for n in
                                        ("results.jsonl", "generated.jsonl", "attempts.jsonl")]
        execution = None
        if (directory / "execution_config.json").exists():
            execution = config_file(directory / "execution_config.json")
            expected = {**self.campaign["reference_execution"],
                **run.execution_definition(SimpleNamespace(run_dir=directory, device="cuda", cpu_threads=4)),
                "max_calls": stage["max_calls"], "plan_fingerprint": pcfg["fingerprint"]}
            expected.pop("fingerprint", None)
            if execution != {**expected, "fingerprint": common.fingerprint(expected)}:
                raise ValueError("Sampling or quality differs from the original generation")
            self.validate_rows(results, execution, jobs)
            self.validate_rows(generated, execution, jobs)
            reserved = run.validate_attempts(attempts, jobs, execution)
            if not {r["sample_id"] for r in results + generated} <= reserved:
                raise ValueError("Image has no durable request reservation")
            generated_ids = {r["sample_id"] for r in generated}
            if any(r.get("status") not in {"passed", "rejected", "generation_error"}
                   or (r.get("status") != "generation_error" and r["sample_id"] not in generated_ids)
                   for r in results):
                raise ValueError("Invalid result or evaluated image without saved generation")
        elif results or generated or attempts:
            raise ValueError("Journal exists without frozen execution")
        value = {"directory": directory, "jobs": jobs, "execution": execution,
                 "results": results, "generated": generated, "attempts": attempts}
        self.cache[directory] = (signature, value)
        return value

    def refresh(self):
        self.stages = [self.audit_stage(s) for s in self.campaign["stages"]]
        state_dir = self.directory / "resume_state"
        for path in sorted((state_dir / "continuations").glob("*/stage_config.json")):
            stage = config_file(path)
            if stage["campaign_fingerprint"] != self.campaign["fingerprint"]:
                raise ValueError("Continuation belongs to another campaign")
            for name, digest in stage["files_sha256"].items():
                if common.file_sha256(Path(name)) != digest:
                    raise ValueError("Continuation inputs changed")
            info = self.audit_stage(stage)
            earlier = {(str(s["directory"]), a["sample_id"]): (s, a)
                       for s in self.stages for a in s["attempts"]}
            for sid, link in stage["retry_of"].items():
                previous = earlier.get((link["run_dir"], sid))
                if previous is None or previous[1]["attempt_sha256"] != link["attempt_sha256"]:
                    raise ValueError("Retry has no matching original reservation")
            self.stages.append(info)
        self.results = {r["sample_id"]: r for r in self.parent}
        self.generated = {r["sample_id"]: r for r in
                          run.read_rows(Path(self.campaign["parent_run_dir"]), "generated.jsonl")}
        self.origins, self.attempts = {}, {}
        for info in self.stages:
            for attempt in info["attempts"]:
                self.attempts.setdefault(attempt["sample_id"], []).append((info, attempt))
            for row in info["generated"]:
                sid = row["sample_id"]
                if sid in self.generated:
                    raise ValueError("Multiple saved generations for one source; inspect provenance")
                self.generated[sid] = row
                self.origins[sid] = info
            for row in info["results"]:
                if row.get("status") in {"passed", "rejected"}:
                    if row["sample_id"] in self.results:
                        raise ValueError("Multiple evaluated results for one source")
                    self.results[row["sample_id"]] = row
        for path in sorted((state_dir / "recoveries").glob("*/*/recovery_config.json")):
            directory = path.parent
            # A frozen intent may exist before the atomic generated commit.
            if not (directory / "generated.jsonl").exists():
                continue
            row, execution = resume_recovery.validate_recovery(directory)
            sid = row["sample_id"]
            links = self.attempts.get(sid, [])
            matching = [(s, a) for s, a in links
                        if s["execution"]["fingerprint"] == execution["recovery_provenance"]["parent_execution_fingerprint"]]
            if len(matching) != 1:
                raise ValueError("Recovered raw has no unique original request")
            resume_recovery.validate_recovery(directory, self.by_id[sid],
                matching[0][0]["execution"], matching[0][1])
            self.add_saved(directory, row, execution)
        for path in sorted((state_dir / "verifications").glob("*/*/execution_config.json")):
            directory, execution = path.parent, config_file(path)
            inputs = run.read_rows(directory, "input.jsonl")
            if len(inputs) != 1 or common.file_sha256(directory / "input.jsonl") != execution["input_sha256"]:
                raise ValueError("Verification input changed")
            original = inputs[0]
            sid = original["sample_id"]
            info = self.origins.get(sid)
            if info is None or original != self.generated[sid]:
                raise ValueError("Verification has no exact saved original image")
            expected = verification_definition(directory, original, info["execution"])
            if execution != {**expected, "fingerprint": common.fingerprint(expected)}:
                raise ValueError("Verification settings changed")
            saved = run.read_rows(directory, "generated.jsonl")
            if not (directory / "generated.jsonl").exists():
                # Interrupted between execution freeze and generated commit.
                # recover_saved() reconstructs this exact immutable input.
                continue
            if len(saved) != 1 or saved[0] != run.sealed({**original, "execution_fingerprint": execution["fingerprint"]}):
                raise ValueError("Verification changed saved image bytes")
            self.add_saved(directory, saved[0], execution)
        return self

    def add_saved(self, directory, row, execution):
        sid = row["sample_id"]
        if self.by_id.get(sid) is None:
            raise ValueError("Unplanned recovered image")
        run.validate_records([row], execution, [self.by_id[sid]])
        results = run.read_rows(directory, "results.jsonl")
        run.validate_records(results, execution, [self.by_id[sid]])
        if len(results) > 1 or any(r.get("status") not in {"passed", "rejected"} for r in results):
            raise ValueError("Invalid verification result")
        if any(any(r.get(k) != v for k, v in row.items() if k != "result_sha256") for r in results):
            raise ValueError("Verification changed its saved generated image")
        if results:
            if sid in self.results:
                raise ValueError("Duplicate original/recovered evaluated result")
            self.results[sid] = results[0]
        self.generated[sid] = row

    def pending(self):
        return [j for j in self.jobs if j["sample_id"] not in self.generated]

    def publish(self, state, error=None):
        rows = [self.results[j["sample_id"]] for j in self.jobs if j["sample_id"] in self.results]
        generated = [self.generated[j["sample_id"]] for j in self.jobs if j["sample_id"] in self.generated]
        accepted = [r for r in rows if r.get("passed") is True and r.get("eligible_for_training") is True]
        unresolved = set(self.attempts) - set(self.generated)
        retry_calls = sum(max(0, len(values) - 1) for values in self.attempts.values())
        failures = [r for s in self.stages for r in s["results"] if r.get("status") == "generation_error"]
        report = {"state": state, "target_images": len(self.jobs), "generated_images": len(generated),
            "quality_evaluated": len(rows), "accepted": len(accepted),
            "rejected": sum(r["status"] == "rejected" for r in rows),
            "generation_errors": len(unresolved & {r["sample_id"] for r in failures}),
            "reserved_without_saved_image": len(unresolved), "historical_failure_records": len(failures),
            "quality_pending": len(generated) - len(rows),
            "remaining_images": len(self.jobs) - len(rows), "retry_calls_reserved": retry_calls,
            "new_calls_reserved": sum(len(v) for v in self.attempts.values()),
            "new_generation_calls_remaining": self.campaign["additional_images"] - len(self.attempts),
            "recovered_without_generation": sum(r.get("recovery", {}).get("http_calls") == 0 for r in generated),
            "conditions_generated": dict(Counter(r["condition"] for r in generated)),
            "cities_generated": dict(Counter(r["city"] for r in generated)),
            "cities_accepted": dict(Counter(r["city"] for r in accepted)),
            "runtime_profile": self.campaign["runtime_profile"],
            "campaign_fingerprint": self.campaign["fingerprint"],
            "resume_fingerprint": config_file(self.directory / "resume_config.json")["fingerprint"],
            "required_free_bytes_estimate": len(self.pending()) * 3_000_000 + 5_000_000_000,
            "available_free_bytes": shutil.disk_usage(self.directory).free,
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if error:
            report["error"] = error
        for name, values in (("generated.jsonl", generated), ("results.jsonl", rows), ("training_manifest.jsonl", accepted)):
            common.write_jsonl(self.directory / name, values)
        common.write_json(self.directory / "summary.json", report)
        return report


def verification_definition(directory, original, execution):
    body = {k: v for k, v in execution.items() if k != "fingerprint"}
    body.update(run_dir=str(directory), stage="saved_image_verification", max_calls=0,
                original_execution_fingerprint=execution["fingerprint"],
                original_result_sha256=original["result_sha256"],
                input_sha256=common.file_sha256(directory / "input.jsonl"))
    return body


def make_verification(directory, original, execution):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "input.jsonl"
    if path.exists():
        if common.read_jsonl(path) != [original]:
            raise ValueError("Saved verification input differs from original generation")
    else:
        common.write_jsonl(path, [original])
    cfg = fast_campaign.frozen(directory / "execution_config.json",
                               verification_definition(directory, original, execution))
    saved = [run.sealed({**original, "execution_fingerprint": cfg["fingerprint"]})]
    if (directory / "generated.jsonl").exists():
        if run.read_rows(directory, "generated.jsonl") != saved:
            raise ValueError("Saved verification output changed")
    else:
        common.write_jsonl(directory / "generated.jsonl", saved)
    return directory


def verify_one(directory):
    with run.run_lock(directory):
        cfg = config_file(directory / "execution_config.json")
        if cfg.get("stage") not in {"saved_image_verification", "saved_raw_recovery"}:
            raise ValueError("Verify only independent recovery directories; never rewrite original journals")
        generated = run.read_rows(directory, "generated.jsonl")
        rows = run.read_rows(directory, "results.jsonl")
        if len(generated) != 1:
            raise ValueError("Expected one saved image")
        run.validate_records(generated, cfg)
        run.validate_records(rows, cfg)
        if rows:
            return
        import torch
        torch.set_num_threads(4)
        verifier = run.QwenQualityVerifier(device="cuda", clip_device="cpu")
        result = run.verify(generated[0], verifier, cfg)
        run.publish(directory, [result], 1, 0, "complete")


def run_child(command, catalog, state):
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
           "ADAPTVPR_CLIP_LOCAL_FILES_ONLY": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    with (catalog.directory / "generation.log").open("a") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            while child.poll() is None:
                catalog.refresh().publish(state)
                time.sleep(20)
        except BaseException:
            # SIGINT lets the worker preserve its already committed boundary.
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise
        if child.returncode:
            raise RuntimeError("Generation/verification stopped; resume explicitly to retry the unfinished job; see generation.log")
    catalog.refresh().publish(state)


def recover_saved(catalog):
    # Qwen must be stopped: there can be no late writes from the old request.
    for sid, values in list(catalog.attempts.items()):
        if sid in catalog.generated:
            continue
        for info, attempt in reversed(values):
            destination = (catalog.directory / "resume_state/recoveries" / sid /
                           info["execution"]["fingerprint"][:24])
            recovered = resume_recovery.recover(info["directory"], catalog.by_id[sid],
                info["execution"], attempt, run.service_output_directory(), destination,
                parent_locked=info["directory"] in getattr(catalog, "locked_generation_dirs", set()))
            if recovered is not None:
                break
    catalog.refresh()
    for job in catalog.jobs:
        sid = job["sample_id"]
        if sid in catalog.results or sid not in catalog.generated:
            continue
        row = catalog.generated[sid]
        if sid in catalog.origins:
            info = catalog.origins[sid]
            directory = (catalog.directory / "resume_state/verifications" / sid /
                         info["execution"]["fingerprint"][:24])
            make_verification(directory, row, info["execution"])
        else:
            directory = Path(row["output_path"]).parents[1]
        run_child([sys.executable, "-u", str(Path(__file__).resolve()), "verify-one",
                   "--verification-dir", str(directory)], catalog, "verifying_saved")


def continuation(catalog, jobs):
    base = catalog.directory / "resume_state/continuations"
    base.mkdir(parents=True, exist_ok=True)
    numbers = [int(p.name[1:]) for p in base.glob("c[0-9]*") if p.name[1:].isdigit()]
    directory = base / f"c{max(numbers, default=0) + 1:06d}"
    directory.mkdir()
    # Exact sealed rows: do not reassign domains, seed, prompt, or canvas.
    payload = plan._plan_bytes(jobs)
    common._atomic_write(directory / "plan.jsonl", [payload.decode("utf-8")])
    body = {k: v for k, v in config_file(catalog.directory / "additional_plan/plan_config.json").items()
            if k != "fingerprint"}
    body.update(output_dir=str(directory), num_images=len(jobs),
                plan_sha256=hashlib.sha256(payload).hexdigest(),
                domain_quotas=dict(Counter(j["condition"] for j in jobs)),
                allocation="exact_unfinished_original_jobs_without_reallocating_weather")
    pcfg = fast_campaign.frozen(directory / "plan_config.json", body)
    retry_of = {}
    for job in jobs:
        values = catalog.attempts.get(job["sample_id"])
        if values:
            info, attempt = values[-1]
            retry_of[job["sample_id"]] = {"run_dir": str(info["directory"]),
                                       "attempt_sha256": attempt["attempt_sha256"]}
    return fast_campaign.frozen(directory / "stage_config.json", {
        "run_dir": str(directory), "num_images": len(jobs), "max_calls": len(jobs),
        "plan_fingerprint": pcfg["fingerprint"], "campaign_fingerprint": catalog.campaign["fingerprint"],
        "retry_of": retry_of, "files_sha256": {str(directory / n): common.file_sha256(directory / n)
                                               for n in ("plan.jsonl", "plan_config.json")}})


def start_qwen(campaign):
    subprocess.run(["systemctl", "--user", "start", QWEN_UNIT], check=True)
    session = requests.Session()
    session.trust_env = False
    for _ in range(180):
        try:
            response = session.get("http://127.0.0.1:8001/health", timeout=5)
            response.raise_for_status()
            health = response.json()
            if health.get("error"):
                raise RuntimeError(health["error"])
            if health.get("generator_ready"):
                if run.service_identity(health) != campaign["reference_execution"]["service"]:
                    raise ValueError("Qwen settings differ from the original generation")
                return
        except requests.ConnectionError:
            pass
        time.sleep(2)
    raise RuntimeError("Qwen startup timed out")


def load_collection(directory, *, require_complete=True):
    directory = Path(directory).resolve()
    campaign = config_file(directory / "all_city_campaign_config.json")
    all_city_campaign.check_inputs(campaign)
    resume = prepare(directory, campaign)
    catalog = Catalog(directory, campaign)
    rows = [catalog.results[j["sample_id"]] for j in catalog.jobs if j["sample_id"] in catalog.results]
    summary = json.loads((directory / "summary.json").read_text())
    if (summary.get("resume_fingerprint") != resume["fingerprint"]
            or summary.get("campaign_fingerprint") != campaign["fingerprint"]
            or (require_complete and (summary["state"] != "complete" or len(rows) != len(catalog.jobs)))):
        raise ValueError("Finish the resumed campaign before training")
    if rows != run.read_rows(directory, "results.jsonl"):
        raise ValueError("Aggregate differs from signed original and continuation journals")
    generated = [catalog.generated[j["sample_id"]] for j in catalog.jobs if j["sample_id"] in catalog.generated]
    if generated != run.read_rows(directory, "generated.jsonl"):
        raise ValueError("Saved aggregate changed")
    if len(rows) == len(catalog.jobs):
        scale_campaign.validate_unique(rows)
        if (Counter(r["city"] for r in rows) != campaign["city_quotas"] or
                Counter(r["condition"] for r in rows) != campaign["domain_quotas"]):
            raise ValueError("Final resumed cohort changed original quotas")
    return rows, campaign


def execute(directory, campaign):
    catalog = None
    owns_qwen = False
    locks = ExitStack()
    try:
        # Nonblocking locks rule out independently launched original workers.
        workers = [Path(s["run_dir"]) for s in campaign["stages"]]
        workers += [p.parent for p in (directory / "resume_state/continuations").glob("*/stage_config.json")]
        for stage_dir in workers:
            locks.enter_context(run.run_lock(stage_dir))
        owns_qwen = True
        stop_qwen()
        prepare(directory, campaign)
        catalog = Catalog(directory, campaign)
        catalog.locked_generation_dirs = set(workers)
        if len(catalog.results) == len(catalog.jobs):
            summary = json.loads((directory / "summary.json").read_text())
            if summary.get("state") != "complete" or not summary.get("resume_fingerprint"):
                catalog.publish("complete")
            load_collection(directory)
            return
        recover_saved(catalog)
        report = catalog.publish("preflight")
        if report["available_free_bytes"] < report["required_free_bytes_estimate"]:
            raise RuntimeError("Insufficient disk for remaining lossless raw and training PNGs")
        if catalog.pending():
            start_qwen(campaign)
        while catalog.pending():
            stage = continuation(catalog, catalog.pending()[:1000])
            run_child([sys.executable, "-u", str(common.CURRICULUM_ROOT / "run.py"), "generate",
                "--run-dir", stage["run_dir"], "--max-calls", str(stage["max_calls"]),
                "--request-timeout", "600", "--device", "cuda", "--cpu-threads", "4"], catalog, "generating")
            results = catalog.audit_stage(stage)["results"]
            if len(results) != stage["num_images"] or any(r["status"] == "generation_error" for r in results):
                raise RuntimeError("Unfinished request; stopped without repeatedly retrying")
        catalog.publish("validating")
        load_collection(directory, require_complete=False)
        catalog.publish("complete")
    except BaseException as exc:
        if catalog is not None:
            catalog.refresh().publish("paused" if isinstance(exc, KeyboardInterrupt) else "stopped",
                                      f"{type(exc).__name__}: {exc}")
        raise
    finally:
        # Persistent Qwen is a separate cgroup: stopping the worker is insufficient.
        try:
            if owns_qwen:
                stop_qwen()
        finally:
            locks.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("run", "status", "validate", "verify-one"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verification-dir", type=Path)
    args = parser.parse_args(argv)
    directory = args.output_dir.resolve()
    if args.command == "status":
        print((directory / "summary.json").read_text())
        return
    if args.command == "verify-one":
        if args.verification_dir is None:
            parser.error("--verification-dir required")
        verify_one(args.verification_dir.resolve())
        return
    with run.run_lock(directory):
        if args.command == "validate":
            rows, _ = load_collection(directory)
            print(f"Validated {len(rows)} resumed 23-city results")
        else:
            signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
            campaign = config_file(directory / "all_city_campaign_config.json")
            all_city_campaign.check_inputs(campaign)
            execute(directory, campaign)


if __name__ == "__main__":
    main()
