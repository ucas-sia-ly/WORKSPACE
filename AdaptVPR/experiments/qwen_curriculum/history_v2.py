"""Read-only audit of old Qwen data before a separately signed v2 migration.

Historical Python hashes may be proved by real Git blobs at the same repository
path. This proves that the frozen source bytes exist in history; it does not
claim that today's verifier is that old implementation. Data files and weights
must still match their frozen on-disk hashes. No old validator is imported.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import subprocess

from PIL import Image

from experiments.qwen_curriculum import common

DOMAINS = {"night", "snow", "fog", "rain"}


def seal(value, field="fingerprint"):
    body = {k: v for k, v in value.items() if k != field}
    return {**body, field: common.fingerprint(body)}


def check_seal(value, field="fingerprint"):
    if value.get(field) != seal(value, field)[field]:
        raise ValueError(f"Invalid historical {field} seal")


def unique(rows):
    for key in ("sample_id", "source_path", "source_sha256"):
        if len({row[key] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate historical {key}")


@dataclass
class History:
    campaign: dict
    jobs: list
    generated: list
    results: list
    attempts: dict
    files_sha256: dict
    code_proofs: list
    stages: list


class Auditor:
    def __init__(self, repo_root=common.WORKSPACE_ROOT):
        self.repo_root = Path(repo_root).resolve()
        self.adaptvpr_root = self.repo_root / "AdaptVPR"
        self.files_sha256 = {}
        self.code_proofs = []
        self._proofs = {}
        self._hashes = {}
        self.stages = []

    def digest(self, path, *, remember=True):
        path = Path(path).resolve()
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if key not in self._hashes:
            self._hashes[key] = common.file_sha256(path)
        digest = self._hashes[key]
        if remember:
            self.files_sha256[str(path)] = digest
        return digest

    def audit_file(self, path, expected):
        path = Path(path).resolve()
        actual = self.digest(path) if path.is_file() else None
        key = (str(path), expected, actual)
        if key in self._proofs:
            return self._proofs[key]
        if actual == expected:
            proof = {"path": str(path), "expected_sha256": expected, "proof": "current_bytes"}
        else:
            if path.suffix != ".py" or not path.is_relative_to(self.repo_root):
                raise ValueError(f"Historical data missing or changed: {path}")
            relative = path.relative_to(self.repo_root).as_posix()
            revisions = subprocess.run(["git", "-C", str(self.repo_root), "log", "--all",
                "--format=%H", "--", relative], check=True, text=True, capture_output=True).stdout.splitlines()
            found = None
            for revision in revisions:
                blob = subprocess.run(["git", "-C", str(self.repo_root), "show", f"{revision}:{relative}"],
                                      capture_output=True, check=False)
                if blob.returncode == 0:
                    import hashlib
                    if hashlib.sha256(blob.stdout).hexdigest() == expected:
                        found = revision
                        break
            if found is None:
                raise ValueError(f"No matching historical Python blob at {relative}: {expected}")
            proof = {"path": str(path), "expected_sha256": expected,
                "current_sha256": actual, "proof": "git_archived_source_bytes",
                "git_commit": found, "git_path": relative,
                "claim": "historical source bytes only; current execution uses a new v2 configuration"}
        self._proofs[key] = proof
        if path.suffix == ".py":
            self.code_proofs.append(proof)
        return proof

    def audit_files(self, values, *, base=None):
        for name, digest in values.items():
            path = Path(name)
            if not path.is_absolute():
                path = (Path(base) if base else self.adaptvpr_root) / path
            self.audit_file(path, digest)

    def config(self, path):
        path = Path(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        check_seal(value)
        self.digest(path)
        self.audit_files(value.get("files_sha256", {}))
        self.audit_files(value.get("implementation_sha256", {}))
        return value

    def rows(self, directory, name):
        path = Path(directory) / name
        if not path.exists():
            return []
        values = common.read_jsonl(path)
        self.digest(path)
        return values

    def image(self, path, digest, dimensions=None, *, png=False):
        path = Path(path)
        if self.digest(path, remember=False) != digest:
            raise ValueError(f"Historical image hash changed: {path}")
        # Recorded byte hashes protect the complete original file. Header reads
        # check dimensions without decoding every multi-megapixel raw repeatedly.
        with Image.open(path) as image:
            if png and image.format != "PNG":
                raise ValueError(f"Historical generated image is not PNG: {path}")
            if dimensions is not None and list(image.size) != dimensions:
                raise ValueError(f"Historical image dimensions changed: {path}")

    def validate_jobs(self, jobs):
        unique(jobs)
        for job in jobs:
            check_seal(job, "record_sha256")
            if not re.fullmatch(r"qwen_[0-9a-f]{24}", job["sample_id"]) or job["condition"] not in DOMAINS:
                raise ValueError("Unsafe historical job or weather")
            self.image(job["source_path"], job["source_sha256"], job["source_dimensions"])

    def plan(self, directory):
        directory = Path(directory)
        config = self.config(directory / "plan_config.json")
        self.audit_file(directory / "plan.jsonl", config["plan_sha256"])
        jobs = self.rows(directory, "plan.jsonl")
        if len(jobs) != config["num_images"]:
            raise ValueError("Historical plan length differs from budget")
        self.validate_jobs(jobs)
        return config, jobs

    def validate_rows(self, rows, execution, jobs):
        by_id = {row["sample_id"]: row for row in jobs}
        seen = set()
        for row in rows:
            check_seal(row, "result_sha256")
            sid = row["sample_id"]
            job = by_id.get(sid)
            if (sid in seen or job is None or row.get("execution_fingerprint") != execution["fingerprint"]
                    or any(row.get(k) != v for k, v in job.items())):
                raise ValueError("Historical result differs from sealed execution/job")
            seen.add(sid)
            self.image(row["source_path"], row["source_sha256"], row["source_dimensions"])
            if row.get("output_path"):
                self.image(row["output_path"], row["output_sha256"], row["source_dimensions"], png=True)
            if row.get("raw_output_path"):
                sampling = row.get("sampling", {})
                self.image(row["raw_output_path"], row["raw_output_sha256"], sampling.get("raw_dimensions"), png=True)

    @staticmethod
    def validate_attempts(attempts, execution, jobs):
        by_id = {job["sample_id"]: job for job in jobs}
        seen = set()
        for number, row in enumerate(attempts, 1):
            check_seal(row, "attempt_sha256")
            job = by_id.get(row["sample_id"])
            if (job is None or row["sample_id"] in seen or row.get("call_number") != number
                    or row.get("record_sha256") != job["record_sha256"]
                    or row.get("execution_fingerprint") != execution["fingerprint"]):
                raise ValueError("Invalid historical durable request reservation")
            seen.add(row["sample_id"])
        if len(attempts) > execution["max_calls"]:
            raise ValueError("Historical durable calls exceed frozen budget")
        return seen

    @staticmethod
    def service(execution):
        service = execution.get("service", {})
        if (service.get("model_id") != "Qwen/Qwen-Image-Edit-2511"
                or service.get("source_modified") is not False
                or service.get("canvas_policy") != "source_aspect_v1"
                or service.get("sampling", {}).get("infer_steps") != 4
                or service.get("sampling", {}).get("guidance_scale") != 1.0):
            raise ValueError("Historical sampling differs from Qwen source-aspect 4-step BF16 baseline")

    def audit_stage(self, directory, *, reference=None, stage=None, complete=False):
        directory = Path(directory).resolve()
        pcfg, jobs = self.plan(directory)
        if stage is not None and (pcfg["fingerprint"] != stage["plan_fingerprint"]
                                  or len(jobs) != stage["num_images"]):
            raise ValueError("Historical stage plan differs from campaign")
        results, generated, attempts = [self.rows(directory, name) for name in
            ("results.jsonl", "generated.jsonl", "attempts.jsonl")]
        execution = None
        if (directory / "execution_config.json").exists():
            execution = self.config(directory / "execution_config.json")
            if (execution.get("run_dir") != str(directory) or execution.get("stage") != "generate"
                    or execution.get("plan_fingerprint") != pcfg["fingerprint"]
                    or type(execution.get("max_calls")) is not int or not 1 <= execution["max_calls"] <= 1000):
                raise ValueError("Historical execution differs from its plan/budget")
            self.service(execution)
            if stage is not None and execution["max_calls"] != stage["max_calls"]:
                raise ValueError("Historical execution budget changed")
            if reference is not None:
                expected = {k: v for k, v in reference.items() if k != "fingerprint"}
                expected.update(run_dir=str(directory), plan_fingerprint=pcfg["fingerprint"],
                                max_calls=stage["max_calls"] if stage else execution["max_calls"])
                if execution != seal(expected):
                    raise ValueError("Historical execution changed frozen generator/quality settings")
            self.validate_rows(results, execution, jobs)
            self.validate_rows(generated, execution, jobs)
            reserved = self.validate_attempts(attempts, execution, jobs)
            if not {r["sample_id"] for r in results + generated} <= reserved:
                raise ValueError("Historical image has no durable reservation")
            saved = {r["sample_id"]: r for r in generated}
            for row in results:
                if row.get("status") not in {"passed", "rejected", "generation_error"}:
                    raise ValueError("Invalid historical result status")
                if row["status"] != "generation_error":
                    original = saved.get(row["sample_id"])
                    if original is None or any(row.get(k) != v for k, v in original.items() if k != "result_sha256"):
                        raise ValueError("Historical verification changed generated image")
        elif results or generated or attempts:
            raise ValueError("Historical journals exist without frozen execution")
        if complete and (len(results) != len(jobs) or len(generated) != len(jobs)
                         or any(row.get("status") not in {"passed", "rejected"} for row in results)):
            raise ValueError("Historical stage is incomplete")
        info = {"run_dir": str(directory), "directory": directory, "plan": pcfg, "jobs": jobs,
                "execution": execution, "results": results, "generated": generated, "attempts": attempts}
        self.stages.append(info)
        return info

    def metadata(self, metadata, parent_dir, job, execution):
        service = (common.ADAPTVPR_ROOT / "tmp/service_outputs/lightx2v").resolve()
        raw, side = Path(metadata["result_path"]).resolve(), Path(metadata["metadata_path"]).resolve()
        if (raw.parent != service or side != raw.with_suffix(".json")
                or metadata.get("source_path") != str(Path(parent_dir) / "source_inputs" / f"{job['sample_id']}.png")
                or metadata.get("seed") != job["seed"] or metadata.get("canvas_policy") != "source_aspect_v1"
                or metadata.get("source_dimensions") != job["source_dimensions"]):
            raise ValueError("Historical recovery metadata ownership or source changed")
        canvas = execution["service"]["canvas"]
        width, height = job["source_dimensions"]
        if (canvas.get("target_shape_order") != "height,width" or canvas.get("rounding") != "nearest_multiple"
                or canvas.get("unsupported_aspect_ratio") != "reject"
                or any(type(canvas.get(k)) is not int or canvas[k] <= 0
                       for k in ("target_pixels", "multiple", "min_side", "max_side"))
                or max(width, height) / min(width, height) > canvas["max_side"] / canvas["min_side"]):
            raise ValueError("Invalid historical source-aspect canvas")
        scale = min(math.sqrt(canvas["target_pixels"] / (width * height)), canvas["max_side"] / max(width, height))
        target = [max(canvas["min_side"], min(canvas["max_side"],
            round(side * scale / canvas["multiple"]) * canvas["multiple"])) for side in (height, width)]
        if metadata.get("target_shape") != target or metadata.get("raw_dimensions") != target[::-1]:
            raise ValueError("Historical recovery dimensions differ from frozen canvas")

    def zero_rows(self, directory, execution, jobs):
        generated = self.rows(directory, "generated.jsonl")
        results = self.rows(directory, "results.jsonl")
        if self.rows(directory, "attempts.jsonl"):
            raise ValueError("Zero-call recovery has a fabricated HTTP reservation")
        self.validate_rows(generated, execution, jobs)
        self.validate_rows(results, execution, jobs)
        by_id = {r["sample_id"]: r for r in generated}
        for row in results:
            original = by_id.get(row["sample_id"])
            if (row.get("status") not in {"passed", "rejected"} or original is None
                    or any(row.get(k) != v for k, v in original.items() if k != "result_sha256")):
                raise ValueError("Historical zero-call result changed saved image")
        return generated, results

    def fast(self, directory):
        directory = Path(directory)
        cfg = self.config(directory / "fast_campaign_config.json")
        self.audit_files(cfg["parent_files_sha256"], base=cfg["parent_run_dir"])
        self.audit_file(cfg["combined_plan"], cfg["combined_plan_sha256"])
        old = self.audit_stage(cfg["parent_run_dir"])
        new = self.audit_stage(cfg["generation_run_dir"], complete=True)
        if (old["execution"]["fingerprint"] != cfg["parent_execution_fingerprint"]
                or new["plan"]["fingerprint"] != cfg["new_plan_fingerprint"]):
            raise ValueError("Historical fast campaign parent or plan changed")
        recdir = Path(cfg["recovery_run_dir"])
        rexec = self.config(recdir / "execution_config.json")
        self.audit_file(recdir / "recovery_inputs.jsonl", cfg["recovery_inputs_sha256"])
        inputs = self.rows(recdir, "recovery_inputs.jsonl")
        if (rexec["fingerprint"] != cfg["recovery_execution_fingerprint"] or rexec["max_calls"] != 0
                or rexec["original_execution_fingerprint"] != old["execution"]["fingerprint"]
                or rexec["inputs_sha256"] != cfg["recovery_inputs_sha256"]):
            raise ValueError("Historical fast recovery execution changed")
        recjobs = [i["job"] for i in inputs]
        generated, results = self.zero_rows(recdir, rexec, recjobs)
        if (len(generated) != len(inputs) or len(results) != len(inputs)
                or set(cfg["recovery_ids"]) != {j["sample_id"] for j in recjobs}):
            raise ValueError("Historical fast recovery is incomplete")
        oldjobs = {j["sample_id"]: j for j in old["jobs"]}
        olderrors = {r["sample_id"]: r for r in old["results"] if r.get("status") == "generation_error"}
        oldattempts = {a["sample_id"] for a in old["attempts"]}
        for value in inputs:
            job, intent = value["job"], value["intent"]
            sid = job["sample_id"]
            check_seal(intent, "intent_sha256")
            if (oldjobs.get(sid) != job or sid not in oldattempts or sid not in olderrors
                    or value["original_error_result_sha256"] != olderrors[sid]["result_sha256"]
                    or intent["record_sha256"] != job["record_sha256"]
                    or intent["execution_fingerprint"] != old["execution"]["fingerprint"]):
                raise ValueError("Historical fast recovery has no original failed reservation")
            self.metadata(intent["metadata"], old["run_dir"], job, old["execution"])
            row = next(r for r in generated if r["sample_id"] == sid)
            if row["raw_output_sha256"] != intent["raw_sha256"] or row.get("recovery", {}).get("http_calls") != 0:
                raise ValueError("Historical fast recovery raw/provenance changed")
            self.audit_file(recdir / "raw" / f"{sid}.json", intent["metadata_sha256"])
            if json.loads((recdir / "raw" / f"{sid}.json").read_text()) != intent["metadata"]:
                raise ValueError("Historical fast recovery sidecar differs from its intent")
        jobs = self.rows(directory, "combined_plan.jsonl")
        if jobs != old["jobs"] + new["jobs"]:
            raise ValueError("Historical fast combined plan changed")
        generated = [r for r in old["generated"] if r["sample_id"] not in set(cfg["recovery_ids"])] + generated + new["generated"]
        results = [r for r in old["results"] if r["sample_id"] not in set(cfg["recovery_ids"])] + results + new["results"]
        self.compare_aggregate(directory, cfg, jobs, generated, results, complete=True)
        return self.rows(directory, "generated.jsonl"), self.rows(directory, "results.jsonl"), cfg

    def compare_aggregate(self, directory, cfg, jobs, generated, results, *, complete=False):
        unique(jobs)
        generated_by = {r["sample_id"]: r for r in generated}
        results_by = {r["sample_id"]: r for r in results}
        if len(generated_by) != len(generated) or len(results_by) != len(results):
            raise ValueError("Duplicate historical aggregate image/result")
        ids = {j["sample_id"] for j in jobs}
        if (set(generated_by) - ids or set(results_by) - ids
                or (complete and (set(results_by) != ids or set(generated_by) != ids))):
            raise ValueError("Historical aggregate is incomplete or contains unplanned images")
        ordered_gen = [generated_by[j["sample_id"]] for j in jobs if j["sample_id"] in generated_by]
        ordered_res = [results_by[j["sample_id"]] for j in jobs if j["sample_id"] in results_by]
        if (ordered_gen != self.rows(directory, "generated.jsonl")
                or ordered_res != self.rows(directory, "results.jsonl")):
            raise ValueError("Historical aggregate differs from signed stage journals")
        if (len(jobs) != cfg["total_images"] or Counter(j["condition"] for j in jobs) != cfg["domain_quotas"]
                or ("city_quotas" in cfg and Counter(j["city"] for j in jobs) != cfg["city_quotas"])):
            raise ValueError("Historical combined plan count or quotas changed")
        summary = json.loads((Path(directory) / "summary.json").read_text())
        self.digest(Path(directory) / "summary.json")
        if (summary.get("campaign_fingerprint") != cfg["fingerprint"]
                or (complete and summary.get("state") != "complete")
                or summary.get("generated_images") != len(generated)
                or summary.get("quality_evaluated") != len(results)):
            raise ValueError("Historical summary differs from audited journals")

    def scale(self, directory):
        directory = Path(directory)
        cfg = self.config(directory / "scale_campaign_config.json")
        generated, results, parent = self.fast(cfg["parent_run_dir"])
        if parent["fingerprint"] != cfg["parent_fingerprint"] or len(results) != cfg["original_images"]:
            raise ValueError("Historical 7277 parent changed")
        jobs = self.rows(Path(cfg["parent_run_dir"]), "combined_plan.jsonl")
        for stage in cfg["stages"]:
            info = self.audit_stage(stage["run_dir"], reference=cfg["reference_execution"], stage=stage, complete=True)
            jobs += info["jobs"]
            generated += info["generated"]
            results += info["results"]
        if jobs != self.rows(directory, "combined_plan.jsonl"):
            raise ValueError("Historical 7277 combined plan changed")
        self.compare_aggregate(directory, cfg, jobs, generated, results, complete=True)
        return generated, results, cfg

    def selection(self, campaign):
        directory = Path(campaign["mining_dir"])
        cfg, request = self.config(directory / "all_city_selection.json"), self.config(directory / "request_config.json")
        if cfg["request_fingerprint"] != request["fingerprint"]:
            raise ValueError("Historical source selection request changed")
        sources = self.rows(Path(cfg["additional_sources"]).parent, Path(cfg["additional_sources"]).name)
        if len(sources) != cfg["additional_images"] or set(r["city"] for r in sources) != set(request["new_cities"]):
            raise ValueError("Historical source selection lost a city")
        for name in cfg["city_selections"]:
            self.config(Path(name))
        return cfg, request

    def resumed_recovery(self, directory, jobs_by, attempts):
        directory = Path(directory)
        cfg = self.config(directory / "recovery_config.json")
        self.audit_file(common.CURRICULUM_ROOT / "resume_recovery.py", cfg["recovery_implementation_sha256"])
        intent = json.loads((directory / "intent.json").read_text())
        check_seal(intent, "intent_sha256")
        self.digest(directory / "intent.json")
        sid = cfg["sample_id"]
        links = [v for v in attempts.get(sid, []) if v["attempt"]["attempt_sha256"] == cfg["parent_attempt_sha256"]]
        if (len(links) != 1 or cfg.get("http_calls") != 0 or cfg.get("stage") != "zero_call_saved_raw_recovery"
                or cfg["intent_sha256"] != intent["intent_sha256"]):
            raise ValueError("Historical recovery has no exact zero-call parent")
        link, job = links[0], jobs_by[sid]
        parent = link["execution"]
        expected = {"parent_run_dir": link["run_dir"], "parent_execution_fingerprint": parent["fingerprint"],
                    "parent_attempt_sha256": link["attempt"]["attempt_sha256"],
                    "record_sha256": job["record_sha256"], "sample_id": sid, "http_calls": 0}
        if any(intent.get(k) != v or cfg.get(k) != v for k, v in expected.items()):
            raise ValueError("Historical recovery intent parent changed")
        self.metadata(intent["metadata"], link["run_dir"], job, parent)
        pcfg, pjobs = self.plan(directory)
        execution = self.config(directory / "execution_config.json")
        if pjobs != [job] or pcfg["fingerprint"] != cfg["plan_fingerprint"] or execution["fingerprint"] != cfg["execution_fingerprint"]:
            raise ValueError("Historical recovery job or execution changed")
        provenance = {"parent_run_dir": link["run_dir"], "parent_execution_fingerprint": parent["fingerprint"],
            "parent_attempt_sha256": link["attempt"]["attempt_sha256"], "intent_sha256": intent["intent_sha256"], "http_calls": 0}
        if intent.get("original_error_result_sha256"):
            provenance["original_error_result_sha256"] = intent["original_error_result_sha256"]
        body = {k: v for k, v in parent.items() if k != "fingerprint"}
        body.update(run_dir=str(directory), plan_fingerprint=pcfg["fingerprint"], max_calls=1,
                    stage="saved_raw_recovery", call_policy="zero POST; copied saved raw from an original durable reservation",
                    recovery_provenance=provenance)
        if execution != seal(body):
            raise ValueError("Historical recovery changed generation/quality settings")
        generated, results = self.zero_rows(directory, execution, [job])
        if not (directory / "generated.jsonl").exists():
            return [], [], execution
        if len(generated) != 1 or generated[0]["raw_output_sha256"] != intent["raw_sha256"]:
            raise ValueError("Historical recovery raw differs from intent")
        self.audit_file(directory / "raw" / f"{sid}.json", intent["metadata_sha256"])
        if json.loads((directory / "raw" / f"{sid}.json").read_text()) != intent["metadata"]:
            raise ValueError("Historical recovered sidecar changed")
        self.image(generated[0]["raw_output_path"], intent["raw_sha256"], intent["metadata"]["raw_dimensions"])
        return generated, results, execution


def audit_history(directory, *, repo_root=common.WORKSPACE_ROOT):
    """Return exact saved bytes, old signed outcomes and complete request links."""
    directory = Path(directory).resolve()
    audit = Auditor(repo_root)
    campaign = audit.config(directory / "all_city_campaign_config.json")
    audit.selection(campaign)
    generated, results, parent = audit.scale(campaign["parent_run_dir"])
    if parent["fingerprint"] != campaign["parent_fingerprint"]:
        raise ValueError("Historical all-city parent changed")
    jobs = audit.rows(directory, "combined_plan.jsonl")
    audit.validate_jobs(jobs)
    planned_by = {j["sample_id"]: j for j in jobs}
    expected_jobs = audit.rows(Path(campaign["parent_run_dir"]), "combined_plan.jsonl")
    originals = []
    for stage in campaign["stages"]:
        info = audit.audit_stage(stage["run_dir"], reference=campaign["reference_execution"], stage=stage)
        expected_jobs += info["jobs"]
        originals.append(info)
    if jobs != expected_jobs:
        raise ValueError("Historical all-city combined plan changed")
    resume_path = directory / "resume_config.json"
    if resume_path.exists():
        resume = audit.config(resume_path)
        if resume["campaign_fingerprint"] != campaign["fingerprint"]:
            raise ValueError("Historical resume belongs to another campaign")
        for name, digest in resume["original_journals"].items():
            if digest is None:
                if Path(name).exists():
                    raise ValueError("Original journal appeared after frozen resume snapshot")
            else:
                audit.audit_file(name, digest)
        for path in sorted((directory / "resume_state/continuations").glob("*/stage_config.json")):
            stage = audit.config(path)
            if stage["campaign_fingerprint"] != campaign["fingerprint"]:
                raise ValueError("Historical continuation belongs to another campaign")
            earlier = {(s["run_dir"], a["sample_id"]): a for s in originals for a in s["attempts"]}
            info = audit.audit_stage(stage["run_dir"], reference=campaign["reference_execution"], stage=stage)
            if any(planned_by.get(j["sample_id"]) != j for j in info["jobs"]):
                raise ValueError("Historical continuation changed exact original jobs")
            for sid, link in stage["retry_of"].items():
                old = earlier.get((link["run_dir"], sid))
                if old is None or old["attempt_sha256"] != link["attempt_sha256"]:
                    raise ValueError("Historical retry has no exact parent reservation")
            retry_ids = {j["sample_id"] for j in info["jobs"] if any(sid == j["sample_id"] for _, sid in earlier)}
            if retry_ids != set(stage["retry_of"]):
                raise ValueError("Historical retry omits or adds reservation provenance")
            originals.append(info)
    generated_by = {r["sample_id"]: r for r in generated}
    results_by = {r["sample_id"]: r for r in results}
    attempts, origins = {}, {}
    for info in audit.stages:
        for attempt in info["attempts"]:
            attempts.setdefault(attempt["sample_id"], []).append({"run_dir": info["run_dir"],
                "execution": info["execution"], "attempt": attempt})
    for info in originals:
        for row in info["generated"]:
            sid = row["sample_id"]
            if sid in generated_by:
                raise ValueError("Multiple historical saved images for one source")
            generated_by[sid], origins[sid] = row, info["execution"]
        for row in info["results"]:
            if row.get("status") in {"passed", "rejected"}:
                if row["sample_id"] in results_by:
                    raise ValueError("Duplicate historical evaluated result")
                results_by[row["sample_id"]] = row
    for path in sorted((directory / "resume_state/recoveries").glob("*/*/recovery_config.json")):
        gen, res, _ = audit.resumed_recovery(path.parent, planned_by, attempts)
        for row in gen:
            if row["sample_id"] in generated_by:
                raise ValueError("Historical recovery duplicates a saved image")
            generated_by[row["sample_id"]] = row
        for row in res:
            if row["sample_id"] in results_by:
                raise ValueError("Historical recovery duplicates an evaluated image")
            results_by[row["sample_id"]] = row
    for path in sorted((directory / "resume_state/verifications").glob("*/*/execution_config.json")):
        vdir, execution = path.parent, audit.config(path)
        inputs = audit.rows(vdir, "input.jsonl")
        if len(inputs) != 1:
            raise ValueError("Historical saved-image verification needs one original")
        original, sid = inputs[0], inputs[0]["sample_id"]
        parent_exec = origins.get(sid)
        if parent_exec is None or original != generated_by[sid]:
            raise ValueError("Historical verification has no exact saved original")
        body = {k: v for k, v in parent_exec.items() if k != "fingerprint"}
        body.update(run_dir=str(vdir), stage="saved_image_verification", max_calls=0,
            original_execution_fingerprint=parent_exec["fingerprint"],
            original_result_sha256=original["result_sha256"], input_sha256=audit.digest(vdir / "input.jsonl"))
        if execution != seal(body):
            raise ValueError("Historical verification execution changed")
        gen, res = audit.zero_rows(vdir, execution, [planned_by[sid]])
        if not (vdir / "generated.jsonl").exists():
            continue
        if gen != [seal({**original, "execution_fingerprint": execution["fingerprint"]}, "result_sha256")]:
            raise ValueError("Historical verification changed saved generated image")
        generated_by[sid] = gen[0]
        if res:
            if sid in results_by:
                raise ValueError("Historical verification duplicates an evaluated image")
            results_by[sid] = res[0]
    generated = [generated_by[j["sample_id"]] for j in jobs if j["sample_id"] in generated_by]
    results = [results_by[j["sample_id"]] for j in jobs if j["sample_id"] in results_by]
    audit.compare_aggregate(directory, campaign, jobs, generated, results)
    if resume_path.exists():
        summary = json.loads((directory / "summary.json").read_text())
        if summary.get("resume_fingerprint") != resume["fingerprint"]:
            raise ValueError("Historical aggregate belongs to another resume")
    return History(campaign, jobs, generated, results, attempts, audit.files_sha256,
                   audit.code_proofs, audit.stages)
