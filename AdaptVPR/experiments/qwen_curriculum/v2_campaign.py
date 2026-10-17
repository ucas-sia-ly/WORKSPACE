"""Migrate saved 23-city images to current quality code, then resume generation.

Historical code is audited against archived Git bytes, never represented as the
current executable. Original journals/images remain read-only. Imported quality
decisions have new seals and explicit links to their historical evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import signal
import sys
from types import SimpleNamespace

from PIL import Image

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import (common, run, plan, fast_campaign,
    resume_campaign as base, resume_recovery, history_v2, requalify_v2)
from experiments.qwen_curriculum.campaign import config_file

DEFAULT_SOURCE = base.DEFAULT_OUTPUT
DEFAULT_OUTPUT = common.WORKSPACE_ROOT / "outputs/qwen_curriculum/generation_23city_v2"


def freeze_rows(path, rows):
    if path.exists():
        if common.read_jsonl(path) != rows:
            raise ValueError(f"Immutable migrated input changed: {path}")
    else:
        common.write_jsonl(path, rows)


def ensure_stopped():
    from experiments.qwen_curriculum import resume_control
    for unit in (resume_control.COORDINATOR_UNIT, resume_control.QWEN_UNIT):
        if not resume_control.stopped(resume_control.unit_state(unit)):
            raise RuntimeError("Pause generation before migrating: resume_control.py pause")


def migration_code():
    from experiments.qwen_curriculum import quality, weather_signal
    from experiments.generation_diagnosis import verifier_controls
    return [Path(p.__file__).resolve() for p in
            (sys.modules[__name__], history_v2, requalify_v2, base, resume_recovery,
             run, common, plan, quality, weather_signal, verifier_controls)]


def check_inputs(directory):
    cfg = config_file(directory / "all_city_campaign_config.json")
    migration = config_file(directory / "migration_config.json")
    if (cfg.get("stage") != "current_quality_23city_migration"
            or cfg["migration_fingerprint"] != migration["fingerprint"]):
        raise ValueError("Invalid current-quality migration")
    for path, digest in {**migration["files_sha256"], **cfg["files_sha256"]}.items():
        if common.file_sha256(Path(path)) != digest:
            raise ValueError(f"Migration input/code changed: {path}; use an explicit new migration")
    return cfg


def legacy_snapshot(source):
    paths = []
    for pattern in ("*.json", "*.jsonl", "additional_*/*.json", "additional_*/*.jsonl",
                    "resume_state/**/*.json", "resume_state/**/*.jsonl"):
        paths.extend(source.glob(pattern))
    return {str(p.resolve()): common.file_sha256(p) for p in sorted(set(paths))
            if p.is_file() and p.name not in {"summary.json", "audit.json"}}


def saved_without_quality(job, row, execution):
    keys = ("output_path", "output_sha256", "raw_output_path", "raw_output_sha256",
            "sampling", "generation_seconds")
    return run.sealed({**job, **{k: row[k] for k in keys if k in row},
        "execution_fingerprint": execution["fingerprint"], "origin": "historical_image_current_quality_import",
        "migration": {"original_execution_fingerprint": row["execution_fingerprint"],
                      "original_result_sha256": row["result_sha256"], "new_generation_calls": 0}})


def recover_orphan(history, job, attempt, output, execution):
    """Copy a uniquely attributable saved raw using already audited old evidence."""
    source = Path(attempt["run_dir"])
    old_execution = attempt["execution"]
    intent, _ = resume_recovery._discover(source, job, old_execution,
                                         attempt["attempt"], run.service_output_directory())
    if intent is None:
        return None
    target = output / "raw" / f"{job['sample_id']}.png"
    sidecar = target.with_suffix(".json")
    resume_recovery._copy_atomic(Path(intent["source_raw_path"]), target, intent["raw_sha256"])
    resume_recovery._copy_atomic(Path(intent["source_metadata_path"]), sidecar, intent["metadata_sha256"])
    normalized = output / "images" / f"{job['sample_id']}.png"
    normalized.parent.mkdir(exist_ok=True)
    temporary = normalized.with_suffix(".tmp.png")
    with Image.open(target) as image:
        image.convert("RGB").resize(tuple(job["source_dimensions"]), Image.Resampling.LANCZOS).save(temporary)
    temporary.replace(normalized)
    fast_campaign.frozen(output / "orphan_intents" / f"{job['sample_id']}.json", intent)
    return run.sealed({**job, "execution_fingerprint": execution["fingerprint"],
        "output_path": str(normalized), "output_sha256": common.file_sha256(normalized),
        "raw_output_path": str(target), "raw_output_sha256": intent["raw_sha256"],
        "origin": "historical_orphan_raw_current_quality_import", "recovery": {"http_calls": 0},
        "migration": {"original_execution_fingerprint": old_execution["fingerprint"],
                      "original_attempt_sha256": attempt["attempt"]["attempt_sha256"],
                      "intent_sha256": intent["intent_sha256"], "new_generation_calls": 0,
                      "sidecar_does_not_prove": ["prompt", "service_identity"]}})


def migrate(source, directory):
    if (directory / "all_city_campaign_config.json").exists():
        cfg = check_inputs(directory)
        if cfg["source_run_dir"] != str(source):
            raise ValueError("Use a new directory for a different historical campaign")
        return cfg
    ensure_stopped()
    if source == directory or directory.is_relative_to(source) or source.is_relative_to(directory):
        raise ValueError("Use a separate migration output directory")
    directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as locks:
        locks.enter_context(run.run_lock(source))
        workers = [p.parent for p in source.glob("additional_*/plan_config.json")]
        workers += [p.parent for p in (source / "resume_state").rglob("execution_config.json")]
        for worker in sorted(set(workers)):
            locks.enter_context(run.run_lock(worker))
        print("Auditing historical seals, image bytes and archived source code...", flush=True)
        history = history_v2.audit_history(source)
        before = legacy_snapshot(source)
        parent = directory / "imported"
        parent.mkdir(exist_ok=True)
        # This is an independent current-code execution, not a modified old seal.
        definitions = {}
        for values in history.attempts.values():
            for value in values:
                old = value["execution"]
                definitions[old["fingerprint"]] = old["quality"]
        # Zero-call recovered quality executions may not own an HTTP attempt.
        for path in history.files_sha256:
            if Path(path).name == "execution_config.json":
                old = config_file(Path(path))
                if "quality" in old:
                    definitions[old["fingerprint"]] = old["quality"]
        execution = fast_campaign.frozen(parent / "execution_config.json", {
            **run.execution_definition(SimpleNamespace(run_dir=parent, device="cuda", cpu_threads=4)),
            "stage": "quality_v2_import", "max_calls": 0,
            "historical_campaign_fingerprint": history.campaign["fingerprint"],
            "historical_quality_definitions": definitions,
            "policy": "existing image bytes; current-code decisions from audited recorded metrics or fresh verification"})
        previous_results = {r["sample_id"]: r for r in history.results}
        previous_generated = {r["sample_id"]: r for r in history.generated}
        generated, results = [], []
        promoted, orphan_count = 0, 0
        for job in history.jobs:
            sid = job["sample_id"]
            if sid in previous_generated:
                original = previous_results.get(sid)
                current = requalify_v2.requalify(original, execution) if original is not None else None
                saved = current or saved_without_quality(job, previous_generated[sid], execution)
                generated.append(saved)
                if current is not None:
                    results.append(current)
                    promoted += original.get("passed") is not True and current.get("passed") is True
            elif sid in history.attempts:
                # A stopped request may have raw bytes even without a ledger row.
                for attempt in reversed(history.attempts[sid]):
                    saved = recover_orphan(history, job, attempt, parent, execution)
                    if saved is not None:
                        generated.append(saved)
                        orphan_count += 1
                        break
        if before != legacy_snapshot(source):
            raise ValueError("Historical journals changed during migration")
        freeze_rows(parent / "generated.jsonl", generated)
        freeze_rows(parent / "results.jsonl", results)
        attempts_rows = [{"sample_id": sid, **v} for sid, values in history.attempts.items() for v in values]
        freeze_rows(parent / "historical_attempts.jsonl", attempts_rows)
        freeze_rows(directory / "combined_plan.jsonl", history.jobs)
        master = directory / "additional_plan"
        master.mkdir(exist_ok=True)
        original_master = config_file(source / "additional_plan/plan_config.json")
        fast_campaign.frozen(master / "plan_config.json", {k: v for k, v in original_master.items() if k != "fingerprint"})
        common.write_json(directory / "historical_code_proofs.json", history.code_proofs)
        imported_files = [parent / n for n in ("execution_config.json", "generated.jsonl", "results.jsonl", "historical_attempts.jsonl")]
        imported_files += [directory / "combined_plan.jsonl", directory / "historical_code_proofs.json", master / "plan_config.json"]
        imported_files += sorted((parent / "orphan_intents").glob("*.json"))
        report = {"historical_generated": len(history.generated), "imported_generated": len(generated),
            "quality_evaluated": len(results), "accepted_v2": sum(r.get("passed") is True for r in results),
            "accepted_legacy": sum(r.get("passed") is True for r in history.results),
            "promoted_by_v2": promoted, "quality_pending": len(generated) - len(results),
            "orphan_recovered_without_generation": orphan_count, "new_generation_calls": 0}
        migration = fast_campaign.frozen(directory / "migration_config.json", {
            "schema_version": 1, "source_run_dir": str(source),
            "source_campaign_fingerprint": history.campaign["fingerprint"],
            "import_execution_fingerprint": execution["fingerprint"], "report": report,
            "files_sha256": {**history.files_sha256, **before,
                **{str(p): common.file_sha256(p) for p in imported_files + migration_code()}}})
        reference = {**history.campaign["reference_execution"],
                     **run.execution_definition(SimpleNamespace(run_dir=directory, device="cuda", cpu_threads=4))}
        reference.pop("fingerprint", None)
        body = {k: v for k, v in history.campaign.items() if k not in {"fingerprint", "files_sha256"}}
        body.update(stage="current_quality_23city_migration", source_run_dir=str(source),
                    migration_fingerprint=migration["fingerprint"], parent_run_dir=str(parent), stages=[],
                    combined_plan=str(directory / "combined_plan.jsonl"), reference_execution=reference,
                    additional_images=len(history.jobs) - len(generated),
                    files_sha256={str(directory / "migration_config.json"): common.file_sha256(directory / "migration_config.json")})
        cfg = fast_campaign.frozen(directory / "all_city_campaign_config.json", body)
        base.prepare(directory, cfg)
        catalog = Catalog(directory, cfg)
        catalog.publish("prepared")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return cfg


class Catalog(base.Catalog):
    def __init__(self, directory, campaign):
        self.directory, self.campaign = directory, campaign
        self.jobs = common.read_jsonl(Path(campaign["combined_plan"]))
        self.by_id = {j["sample_id"]: j for j in self.jobs}
        if len(self.by_id) != campaign["total_images"]:
            raise ValueError("Migrated plan length changed")
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
        self.refresh()

    def refresh(self):
        self.stages, self.attempts = [], {}
        self.results = {r["sample_id"]: r for r in self.parent}
        self.generated = {r["sample_id"]: r for r in self.imported_generated}
        imported_origin = {"directory": Path(self.campaign["parent_run_dir"]), "execution": self.import_execution}
        self.origins = {sid: imported_origin for sid in self.generated}
        state_dir = self.directory / "resume_state"
        for path in sorted((state_dir / "continuations").glob("*/stage_config.json")):
            stage = config_file(path)
            if stage["campaign_fingerprint"] != self.campaign["fingerprint"]:
                raise ValueError("Continuation belongs to a different migration")
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
                        raise ValueError("Duplicate evaluated migrated image")
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
            expected = base.verification_definition(directory, original, info["execution"])
            if execution != {**expected, "fingerprint": common.fingerprint(expected)}:
                raise ValueError("Verification differs from current quality settings")
            if not (directory / "generated.jsonl").exists():
                continue
            saved = run.read_rows(directory, "generated.jsonl")
            if saved != [run.sealed({**original, "execution_fingerprint": execution["fingerprint"]})]:
                raise ValueError("Verification modified image provenance")
            self.add_saved(directory, saved[0], execution)
        return self

    def publish(self, state, error=None):
        report = super().publish(state, error)
        report.update(quality_version=2, historical_images=len(self.imported_generated),
                      source_run_dir=self.campaign["source_run_dir"],
                      historical_calls_reserved=sum(len(v) for v in self.historical_attempts.values()))
        report["new_generation_calls_remaining"] = len(self.pending())
        report["legacy_interrupted_retries_reserved"] = sum(
            bool(self.historical_attempts.get(sid)) for sid in self.attempts)
        common.write_json(self.directory / "summary.json", report)
        return report


def continuation(catalog, jobs):
    # Mirror the immutable generator stage with an explicit old-request link.
    root = catalog.directory / "resume_state/continuations"
    root.mkdir(parents=True, exist_ok=True)
    numbers = [int(p.name[1:]) for p in root.glob("c[0-9]*") if p.name[1:].isdigit()]
    directory = root / f"c{max(numbers, default=0) + 1:06d}"
    directory.mkdir()
    payload = plan._plan_bytes(jobs)
    common._atomic_write(directory / "plan.jsonl", [payload.decode()])
    body = {k: v for k, v in config_file(catalog.directory / "additional_plan/plan_config.json").items() if k != "fingerprint"}
    body.update(output_dir=str(directory), num_images=len(jobs), plan_sha256=hashlib.sha256(payload).hexdigest(),
                domain_quotas=dict(Counter(j["condition"] for j in jobs)), allocation="exact_unfinished_jobs_current_quality")
    pcfg = fast_campaign.frozen(directory / "plan_config.json", body)
    retry, legacy = {}, {}
    for job in jobs:
        sid = job["sample_id"]
        if catalog.attempts.get(sid):
            info, attempt = catalog.attempts[sid][-1]
            retry[sid] = {"run_dir": str(info["directory"]), "attempt_sha256": attempt["attempt_sha256"]}
        elif catalog.historical_attempts.get(sid):
            value = catalog.historical_attempts[sid][-1]
            legacy[sid] = {"run_dir": value["run_dir"], "attempt_sha256": value["attempt"]["attempt_sha256"]}
    return fast_campaign.frozen(directory / "stage_config.json", {
        "run_dir": str(directory), "num_images": len(jobs), "max_calls": len(jobs),
        "plan_fingerprint": pcfg["fingerprint"], "campaign_fingerprint": catalog.campaign["fingerprint"],
        "retry_of": retry, "legacy_retry_of": legacy,
        "files_sha256": {str(directory / n): common.file_sha256(directory / n) for n in ("plan.jsonl", "plan_config.json")}})


def load_collection(directory, *, require_complete=True):
    directory = Path(directory).resolve()
    cfg = check_inputs(directory)
    base.prepare(directory, cfg)
    catalog = Catalog(directory, cfg)
    rows = [catalog.results[j["sample_id"]] for j in catalog.jobs if j["sample_id"] in catalog.results]
    generated = [catalog.generated[j["sample_id"]] for j in catalog.jobs if j["sample_id"] in catalog.generated]
    report = json.loads((directory / "summary.json").read_text())
    if (report.get("campaign_fingerprint") != cfg["fingerprint"] or len(rows) != len(catalog.jobs)
            or (require_complete and report["state"] != "complete")):
        raise ValueError("Finish current-quality generation before training")
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
            base.stop_qwen()
            check_inputs(directory)
            base.prepare(directory, cfg)
            catalog = Catalog(directory, cfg)
            catalog.locked_generation_dirs = set(workers)
            if len(catalog.results) == len(catalog.jobs):
                summary = json.loads((directory / "summary.json").read_text())
                if summary.get("state") != "complete":
                    catalog.publish("complete")
                load_collection(directory)
                return
            base.recover_saved(catalog)
            report = catalog.publish("preflight")
            if report["available_free_bytes"] < report["required_free_bytes_estimate"]:
                raise RuntimeError("Insufficient disk for lossless raw and training PNGs")
            if catalog.pending():
                base.start_qwen(cfg)
            while catalog.pending():
                stage = continuation(catalog, catalog.pending()[:1000])
                base.run_child([sys.executable, "-u", str(common.CURRICULUM_ROOT / "run.py"), "generate",
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
                                          f"{type(exc).__name__}: {exc}")
            raise
        finally:
            base.stop_qwen()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("migrate", "run", "check", "status", "validate"))
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    directory = args.output_dir.resolve()
    if args.command == "status":
        print((directory / "summary.json").read_text())
        return
    with run.run_lock(directory):
        if args.command == "migrate":
            migrate(args.source_dir.resolve(), directory)
        elif args.command == "check":
            cfg = check_inputs(directory)
            base.prepare(directory, cfg)
            catalog = Catalog(directory, cfg)
            print(json.dumps({"state": "preflight_verified", "quality_version": 2,
                "generated": len(catalog.generated), "evaluated": len(catalog.results),
                "remaining": len(catalog.pending()), "new_generation_calls": 0}, ensure_ascii=False), flush=True)
        elif args.command == "validate":
            print(f"Validated {len(load_collection(directory)[0])} current-quality results")
        else:
            signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
            execute(directory, check_inputs(directory))


if __name__ == "__main__":
    main()
