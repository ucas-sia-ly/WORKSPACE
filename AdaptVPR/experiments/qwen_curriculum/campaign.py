"""Resume the fixed 23-city Qwen plan and collect committed images in one library.

The generator, sampling parameters and append-only stage journals stay sealed.
Saved interrupted images are recovered and verified before any new HTTP request;
a failed request is retried only after another explicit start.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import requests

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import common, run, resume_recovery, image_store

DEFAULT_OUTPUT = common.WORKSPACE_ROOT / "outputs/qwen_curriculum/state"
QWEN_UNIT = "adaptvpr-lightx2v.service"
JOURNALS = ("execution_config.json", "results.jsonl", "generated.jsonl", "attempts.jsonl")
run_lock = run.run_lock


def config_file(path):
    value = json.loads(Path(path).read_text())
    expected = value.pop("fingerprint")
    if common.fingerprint(value) != expected:
        raise ValueError(f"Configuration checksum mismatch: {path}")
    return {**value, "fingerprint": expected}


def frozen(path, value):
    value = {**value, "fingerprint": common.fingerprint(value)}
    if path.exists():
        if config_file(path) != value:
            raise ValueError(f"Immutable configuration changed: {path}")
    else:
        common.write_json(path, value)
    return value


def check_inputs(directory):
    """Validate the deployed runtime, without executing archived legacy code."""
    directory = Path(directory).resolve()
    campaign = config_file(directory / "all_city_campaign_config.json")
    runtime = config_file(directory / "runtime_config.json")
    if type(runtime.get("schema_version")) is not int or runtime["schema_version"] != 1:
        raise ValueError("Unsupported runtime schema")
    if runtime.get("campaign_fingerprint") != campaign["fingerprint"]:
        raise ValueError("Runtime belongs to a different campaign")
    if (campaign.get("stage") != "current_quality_23city_migration" or campaign.get("stages") != []
            or campaign.get("combined_plan") != str(directory / "combined_plan.jsonl")
            or campaign.get("parent_run_dir") != str(directory / "imported")
            or type(campaign.get("total_images")) is not int or campaign["total_images"] <= 0
            or not all(k in campaign for k in ("reference_execution", "city_quotas", "domain_quotas", "runtime_profile"))):
        raise ValueError("Expected the current fixed-plan 23-city snapshot")
    required = [directory / n for n in ("all_city_campaign_config.json", "combined_plan.jsonl",
                                       "additional_plan/plan_config.json")]
    required += [directory / "imported" / n for n in
                 ("execution_config.json", "generated.jsonl", "results.jsonl", "historical_attempts.jsonl")]
    required += [Path(module.__file__).resolve() for module in
                 (sys.modules[__name__], common, run, resume_recovery, image_store)]
    files = runtime.get("files_sha256")
    if not isinstance(files, dict) or not {str(p) for p in required} <= files.keys():
        raise ValueError("Runtime does not freeze every required static input")
    for name, digest in runtime["files_sha256"].items():
        if common.file_sha256(Path(name)) != digest:
            raise ValueError(f"Runtime input/code changed: {name}")
    return campaign


def ensure_stopped():
    from experiments.qwen_curriculum import resume_control
    for unit in (resume_control.COORDINATOR_UNIT, QWEN_UNIT):
        if not resume_control.stopped(resume_control.unit_state(unit)):
            raise RuntimeError("Pause generation first: resume_control.py pause")


def stop_qwen():
    subprocess.run(["systemctl", "--user", "stop", QWEN_UNIT], check=True)
    pid = subprocess.run(["systemctl", "--user", "show", QWEN_UNIT,
                          "--property=MainPID", "--value"],
                         check=True, capture_output=True, text=True).stdout.strip()
    if pid != "0":
        raise RuntimeError("Qwen still has a running process; recovery is unsafe")


class Catalog:
    """Audit signed imports and continuation journals; cache unchanged image hashes."""
    def __init__(self, directory, campaign):
        self.directory, self.campaign = directory, campaign
        self.jobs = common.read_jsonl(Path(campaign["combined_plan"]))
        self.by_id = {j["sample_id"]: j for j in self.jobs}
        if any(len({j[k] for j in self.jobs}) != len(self.jobs)
               for k in ("sample_id", "source_path", "source_sha256")):
            raise ValueError("Duplicate planned source")
        if len(self.by_id) != campaign["total_images"]:
            raise ValueError("Frozen plan length changed")
        parent = Path(campaign["parent_run_dir"])
        self.import_execution = config_file(parent / "execution_config.json")
        self.parent = run.read_rows(parent, "results.jsonl")
        self.imported_generated = run.read_rows(parent, "generated.jsonl")
        self.historical_attempts = {}
        for value in run.read_rows(parent, "historical_attempts.jsonl"):
            self.historical_attempts.setdefault(value["sample_id"], []).append(value)
        self.cache, self.image_cache = {}, {}
        self.validate_rows(self.parent, self.import_execution, self.jobs)
        self.validate_rows(self.imported_generated, self.import_execution, self.jobs)
        self.store = image_store.ImageStore(directory.parent / "images", scope_root=directory.parent)
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
        if (len(jobs) != stage["num_images"] or stage["max_calls"] != len(jobs)
                or not 1 <= len(jobs) <= 1000):
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
        self.stages, self.attempts = [], {}
        self.verifications = {}
        self.results = {r["sample_id"]: r for r in self.parent}
        self.generated = {r["sample_id"]: r for r in self.imported_generated}
        imported_origin = {"directory": Path(self.campaign["parent_run_dir"]), "execution": self.import_execution}
        self.origins = {sid: imported_origin for sid in self.generated}
        state_dir = self.directory / "resume_state"
        for path in sorted((state_dir / "continuations").glob("*/stage_config.json")):
            stage = config_file(path)
            if stage["campaign_fingerprint"] != self.campaign["fingerprint"]:
                raise ValueError("Continuation belongs to a different campaign")
            for name, digest in stage["files_sha256"].items():
                if common.file_sha256(Path(name)) != digest:
                    raise ValueError("Continuation inputs changed")
            for sid, link in stage.get("legacy_retry_of", {}).items():
                if not any(v["run_dir"] == link["run_dir"] and v["attempt"]["attempt_sha256"] == link["attempt_sha256"]
                           for v in self.historical_attempts.get(sid, [])):
                    raise ValueError("Migrated retry has no original request reservation")
            earlier = {(str(s["directory"]), a["sample_id"]): a for s in self.stages for a in s["attempts"]}
            for sid, link in stage["retry_of"].items():
                if earlier.get((link["run_dir"], sid), {}).get("attempt_sha256") != link["attempt_sha256"]:
                    raise ValueError("Retry has no matching prior request")
            info = self.audit_stage(stage)
            self.stages.append(info)
            for attempt in info["attempts"]:
                self.attempts.setdefault(attempt["sample_id"], []).append((info, attempt))
            for row in info["generated"]:
                sid = row["sample_id"]
                if sid in self.generated:
                    raise ValueError("An existing image was regenerated")
                self.generated[sid], self.origins[sid] = row, info
            for row in info["results"]:
                if row.get("status") in {"passed", "rejected"}:
                    if row["sample_id"] in self.results:
                        raise ValueError("Duplicate evaluated image")
                    self.results[row["sample_id"]] = row
        for path in sorted((state_dir / "recoveries").glob("*/*/recovery_config.json")):
            if not (path.parent / "generated.jsonl").exists():
                continue
            row, execution = resume_recovery.validate_recovery(path.parent)
            sid = row["sample_id"]
            expected_parent = execution["recovery_provenance"]["parent_execution_fingerprint"]
            links = [(s, a) for s, a in self.attempts.get(sid, []) if s["execution"]["fingerprint"] == expected_parent]
            if len(links) != 1:
                raise ValueError("Recovery has no unique migrated request")
            resume_recovery.validate_recovery(path.parent, self.by_id[sid], links[0][0]["execution"], links[0][1])
            self.add_saved(path.parent, row, execution)
        for path in sorted((state_dir / "verifications").glob("*/*/execution_config.json")):
            directory, execution = path.parent, config_file(path)
            inputs = run.read_rows(directory, "input.jsonl")
            if len(inputs) != 1 or common.file_sha256(directory / "input.jsonl") != execution["input_sha256"]:
                raise ValueError("Verification input changed")
            original, info = inputs[0], self.origins.get(inputs[0]["sample_id"])
            if info is None or self.generated[original["sample_id"]] != original:
                raise ValueError("Verification is not the exact saved image")
            expected = verification_definition(directory, original, info["execution"])
            if execution != {**expected, "fingerprint": common.fingerprint(expected)}:
                raise ValueError("Verification differs from current quality settings")
            if not (directory / "generated.jsonl").exists():
                continue
            saved = run.read_rows(directory, "generated.jsonl")
            if saved != [run.sealed({**original, "execution_fingerprint": execution["fingerprint"]})]:
                raise ValueError("Verification modified image provenance")
            self.add_saved(directory, saved[0], execution)
            self.verifications[original["sample_id"]] = directory
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

    def synchronize(self):
        generated = [self.generated[j["sample_id"]] for j in self.jobs if j["sample_id"] in self.generated]
        results = [self.results[j["sample_id"]] for j in self.jobs if j["sample_id"] in self.results]
        return self.store.sync(generated, results)

    def publish(self, state, error=None, *, sync=True):
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
            "new_generation_calls_remaining": len(self.pending()),
            "recovered_without_generation": sum(r.get("recovery", {}).get("http_calls") == 0 for r in generated),
            "conditions_generated": dict(Counter(r["condition"] for r in generated)),
            "cities_generated": dict(Counter(r["city"] for r in generated)),
            "cities_accepted": dict(Counter(r["city"] for r in accepted)),
            "runtime_profile": self.campaign["runtime_profile"],
            "quality_version": 2, "historical_images": len(self.imported_generated),
            "historical_calls_reserved": sum(len(v) for v in self.historical_attempts.values()),
            "legacy_interrupted_retries_reserved": sum(bool(self.historical_attempts.get(sid)) for sid in self.attempts),
            "campaign_fingerprint": self.campaign["fingerprint"],
            "runtime_fingerprint": config_file(self.directory / "runtime_config.json")["fingerprint"],
            "required_free_bytes_estimate": len(self.pending()) * 3_000_000 + 5_000_000_000,
            "available_free_bytes": shutil.disk_usage(self.directory).free,
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if error:
            report["error"] = error
        for name, values in (("generated.jsonl", generated), ("results.jsonl", rows), ("training_manifest.jsonl", accepted)):
            common.write_jsonl(self.directory / name, values)
        common.write_json(self.directory / "summary.json", report)
        if sync:
            self.synchronize()
        return report


def verification_definition(directory, original, execution):
    body = {k: v for k, v in execution.items() if k != "fingerprint"}
    # Import executions retain the original code hashes as historical evidence.
    # This new zero-call execution records the verifier actually used now.
    body.update(run.execution_definition(SimpleNamespace(run_dir=directory, device="cuda", cpu_threads=4)))
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
    cfg = frozen(directory / "execution_config.json", verification_definition(directory, original, execution))
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
        if sid in catalog.verifications:
            # The saved row already carries this independent verification's
            # fingerprint. Reuse its sealed input instead of wrapping it again.
            directory = catalog.verifications[sid]
        elif sid in catalog.origins:
            info = catalog.origins[sid]
            directory = (catalog.directory / "resume_state/verifications" / sid /
                         info["execution"]["fingerprint"][:24])
            make_verification(directory, row, info["execution"])
        else:
            directory = Path(row["output_path"]).parents[1]
        run_child([sys.executable, "-u", str(Path(__file__).resolve()), "verify-one",
                   "--verification-dir", str(directory)], catalog, "verifying_saved")


def continuation(catalog, jobs):
    if (not 1 <= len(jobs) <= 1000 or len({j["sample_id"] for j in jobs}) != len(jobs)
            or any(catalog.by_id.get(j["sample_id"]) != j or j["sample_id"] in catalog.generated for j in jobs)):
        raise ValueError("Continuation must contain 1-1000 exact original jobs")
    # Keep an explicit link when retrying a previously reserved request.
    root = catalog.directory / "resume_state/continuations"
    root.mkdir(parents=True, exist_ok=True)
    numbers = [int(p.name[1:]) for p in root.glob("c[0-9]*") if p.name[1:].isdigit()]
    directory = root / f"c{max(numbers, default=0) + 1:06d}"
    directory.mkdir()
    payload = "".join(json.dumps(j, ensure_ascii=False, allow_nan=False) + "\n" for j in jobs).encode("utf-8")
    common._atomic_write(directory / "plan.jsonl", [payload.decode()])
    body = {k: v for k, v in config_file(catalog.directory / "additional_plan/plan_config.json").items() if k != "fingerprint"}
    body.update(output_dir=str(directory), num_images=len(jobs), plan_sha256=hashlib.sha256(payload).hexdigest(),
                domain_quotas=dict(Counter(j["condition"] for j in jobs)), allocation="exact_unfinished_jobs_current_quality")
    pcfg = frozen(directory / "plan_config.json", body)
    retry, legacy = {}, {}
    for job in jobs:
        sid = job["sample_id"]
        if catalog.attempts.get(sid):
            info, attempt = catalog.attempts[sid][-1]
            retry[sid] = {"run_dir": str(info["directory"]), "attempt_sha256": attempt["attempt_sha256"]}
        elif catalog.historical_attempts.get(sid):
            value = catalog.historical_attempts[sid][-1]
            legacy[sid] = {"run_dir": value["run_dir"], "attempt_sha256": value["attempt"]["attempt_sha256"]}
    return frozen(directory / "stage_config.json", {
        "run_dir": str(directory), "num_images": len(jobs), "max_calls": len(jobs),
        "plan_fingerprint": pcfg["fingerprint"], "campaign_fingerprint": catalog.campaign["fingerprint"],
        "retry_of": retry, "legacy_retry_of": legacy,
        "files_sha256": {str(directory / n): common.file_sha256(directory / n) for n in ("plan.jsonl", "plan_config.json")}})


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
    cfg = check_inputs(directory)
    catalog = Catalog(directory, cfg)
    rows = [catalog.results[j["sample_id"]] for j in catalog.jobs if j["sample_id"] in catalog.results]
    generated = [catalog.generated[j["sample_id"]] for j in catalog.jobs if j["sample_id"] in catalog.generated]
    report = json.loads((directory / "summary.json").read_text())
    if (report.get("campaign_fingerprint") != cfg["fingerprint"] or len(rows) != len(catalog.jobs)
            or (require_complete and report["state"] != "complete")):
        raise ValueError("Finish all 23-city generation and quality review before training")
    if rows != run.read_rows(directory, "results.jsonl") or generated != run.read_rows(directory, "generated.jsonl"):
        raise ValueError("Migrated aggregation differs from signed stage journals")
    if len(rows) == len(catalog.jobs) and (Counter(r["condition"] for r in rows) != cfg["domain_quotas"]
                                        or Counter(r["city"] for r in rows) != cfg["city_quotas"]):
        raise ValueError("Original cities or weather quotas changed")
    return rows, cfg


def execute(directory, cfg):
    catalog = None
    with ExitStack() as locks:
        workers = [p.parent for p in (directory / "resume_state/continuations").glob("*/stage_config.json")]
        for worker in workers:
            locks.enter_context(run.run_lock(worker))
        try:
            stop_qwen()
            check_inputs(directory)
            catalog = Catalog(directory, cfg)
            catalog.locked_generation_dirs = set(workers)
            catalog.synchronize()
            if len(catalog.results) == len(catalog.jobs):
                summary = json.loads((directory / "summary.json").read_text())
                if summary.get("state") != "complete":
                    catalog.publish("complete")
                load_collection(directory)
                return
            recover_saved(catalog)
            report = catalog.publish("preflight")
            if report["available_free_bytes"] < report["required_free_bytes_estimate"]:
                raise RuntimeError("Insufficient disk for lossless raw and training PNGs")
            if catalog.pending():
                start_qwen(cfg)
            while catalog.pending():
                stage = continuation(catalog, catalog.pending()[:1000])
                run_child([sys.executable, "-u", str(common.CURRICULUM_ROOT / "run.py"), "generate",
                    "--run-dir", stage["run_dir"], "--max-calls", str(stage["max_calls"]),
                    "--request-timeout", "600", "--device", "cuda", "--cpu-threads", "4"], catalog, "generating")
                rows = catalog.audit_stage(stage)["results"]
                if len(rows) != stage["num_images"] or any(r["status"] == "generation_error" for r in rows):
                    raise RuntimeError("Request failed; resume explicitly to retry once")
            catalog.publish("validating")
            load_collection(directory, require_complete=False)
            catalog.publish("complete")
        except BaseException as exc:
            if catalog is not None:
                catalog.refresh().publish("paused" if isinstance(exc, KeyboardInterrupt) else "stopped",
                                          f"{type(exc).__name__}: {exc}", sync=False)
                try:
                    catalog.synchronize()
                except Exception as storage_error:
                    print(f"Image library will resume next start: {storage_error}", file=sys.stderr, flush=True)
            raise
        finally:
            stop_qwen()


def check(directory):
    campaign = check_inputs(directory)
    catalog = Catalog(directory, campaign)
    report = catalog.synchronize()
    report.update(state="preflight_verified", generated=len(catalog.generated),
                  evaluated=len(catalog.results), remaining=len(catalog.pending()), new_generation_calls=0)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("check", "run", "status", "verify-one"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verification-dir", type=Path)
    args = parser.parse_args(argv)
    directory = args.output_dir.expanduser().resolve()
    if args.command == "status":
        print((directory / "summary.json").read_text())
        return
    if args.command == "verify-one":
        if args.verification_dir is None:
            parser.error("--verification-dir required")
        verify_one(args.verification_dir.resolve())
        return
    with run.run_lock(directory):
        if args.command == "check":
            check(directory)
        else:
            signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
            execute(directory, check_inputs(directory))


if __name__ == "__main__":
    main()
