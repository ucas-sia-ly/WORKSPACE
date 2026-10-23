"""Validate reusable weather-bank outputs without loading generation models.

The original ledger, plan, execution settings and images remain untouched. A
candidate is indexed by place and domain so a new plan can choose one suitable
view without inheriting an old campaign's source ordering or domain allocation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import json
import math
from pathlib import Path

from PIL import Image

from experiments.qwen_curriculum import common, prompt_bank


_HEALTH_FIELDS = {"status", "model_loaded", "generator_ready", "error"}
_REQUIRED_SERVICE = {"model_id", "model_revision", "lora_id", "lora_revision", "lora_filename",
                     "canvas_policy", "canvas", "performance", "sampling"}


class _Rejected(ValueError):
    pass


def _identity(service):
    if not isinstance(service, dict) or not _REQUIRED_SERVICE <= service.keys():
        raise _Rejected("service_definition_missing")
    if (service["model_id"] != "Qwen/Qwen-Image-Edit-2511"
            or service["canvas_policy"] != "source_aspect_v1"
            or not isinstance(service["sampling"], dict)
            or not isinstance(service["performance"], dict)
            or service["sampling"].get("infer_steps") != 4
            or service["sampling"].get("guidance_scale") != 1.0):
        raise _Rejected("unsupported_service_definition")
    return {key: value for key, value in service.items() if key not in _HEALTH_FIELDS}


def _quality_settings(definition):
    if not isinstance(definition, dict) or not definition:
        raise _Rejected("quality_definition_missing")
    # v3.1 changes the planning ratios and drops morning_mist. The verifier's
    # actual texts/thresholds are unchanged; those must still compare exactly.
    return {key: value for key, value in definition.items() if key != "prompt_bank"}


def _object(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise _Rejected("metadata_not_object")
    return value


def _rows(path):
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise _Rejected("ledger_invalid_json") from exc
            if not isinstance(row, dict):
                raise _Rejected("ledger_row_not_object")
            yield number, row


def _sealed_config(config):
    if "fingerprint" in config:
        payload = {key: value for key, value in config.items() if key != "fingerprint"}
        if config["fingerprint"] != common.fingerprint(payload):
            raise _Rejected("configuration_checksum_mismatch")


def _ledger_context(ledger, expected_service, expected_quality):
    directory = ledger.parent
    plan_path = directory / "plan.jsonl"
    plan_config_path = directory / "plan_config.json"
    execution_path = directory / "execution_config.json"
    execution = _object(execution_path)
    _sealed_config(execution)
    if _identity(execution.get("service")) != expected_service:
        raise _Rejected("service_definition_mismatch")
    if _quality_settings(execution.get("quality")) != expected_quality:
        raise _Rejected("quality_definition_mismatch")
    plan_config = _object(plan_config_path)
    _sealed_config(plan_config)
    plan_sha = common.file_sha256(plan_path)
    if plan_config.get("plan_sha256") != plan_sha or execution.get("plan_sha256") != plan_sha:
        raise _Rejected("plan_checksum_mismatch")
    jobs = {}
    for _, job in _rows(plan_path):
        sample_id = job.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in jobs:
            raise _Rejected("plan_duplicate_or_missing_sample_id")
        jobs[sample_id] = job
    return dict(jobs=jobs, execution=execution, plan_config=plan_config, ledger_path=str(ledger),
                ledger_sha256=common.file_sha256(ledger), execution_path=str(execution_path),
                execution_sha256=common.file_sha256(execution_path), plan_path=str(plan_path),
                plan_sha256=plan_sha, plan_config_path=str(plan_config_path),
                plan_config_sha256=common.file_sha256(plan_config_path))


def _dimensions(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(type(side) is not int or side <= 0 for side in value)):
        raise _Rejected("invalid_dimensions")
    return list(value)


def _target_shape(dimensions, canvas):
    if (not isinstance(canvas, dict) or canvas.get("target_shape_order") != "height,width"
            or canvas.get("rounding") != "nearest_multiple"
            or canvas.get("unsupported_aspect_ratio") != "reject"
            or any(type(canvas.get(key)) is not int or canvas[key] <= 0
                   for key in ("target_pixels", "multiple", "min_side", "max_side"))):
        raise _Rejected("invalid_canvas_definition")
    width, height = dimensions
    if max(width, height) / min(width, height) > canvas["max_side"] / canvas["min_side"]:
        raise _Rejected("unsupported_source_aspect")
    scale = min(math.sqrt(canvas["target_pixels"] / (width * height)),
                canvas["max_side"] / max(width, height))
    return [max(canvas["min_side"], min(canvas["max_side"],
                round(side * scale / canvas["multiple"]) * canvas["multiple"])) for side in (height, width)]


def _image(path, dimensions, *, png=False, expected_sha=None, hashes):
    path = Path(path)
    if not path.is_file():
        raise _Rejected("image_missing")
    stat = path.stat()
    signature = stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    key = (str(path.resolve()), *signature)
    if key not in hashes:
        digest = common.file_sha256(path)
        with Image.open(path) as image:
            actual_dimensions, image_format = list(image.size), image.format
            image.verify()
        # verify() checks the PNG chunks; load() also catches an incomplete or
        # invalid compressed pixel stream, including JPEG source corruption.
        with Image.open(path) as image:
            image.load()
        after = path.stat()
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != signature:
            raise _Rejected("image_changed_during_validation")
        hashes[key] = digest, actual_dimensions, image_format
    digest, actual_dimensions, image_format = hashes[key]
    if actual_dimensions != dimensions:
        raise _Rejected("image_dimensions_mismatch")
    if png and image_format != "PNG":
        raise _Rejected("output_not_png")
    if expected_sha is not None and digest != expected_sha:
        raise _Rejected("image_checksum_mismatch")
    return digest


def _candidate(row, line, context, expected_service, hashes):
    job = context["jobs"].get(row.get("sample_id"))
    if job is None or any(row.get(key) != value for key, value in job.items()):
        raise _Rejected("result_differs_from_plan")
    if type(row.get("seed")) is not int or not 1 <= row["seed"] <= 2_000_000_000:
        raise _Rejected("invalid_seed")
    if type(row.get("place_id")) is not int or not isinstance(row.get("city"), str) or not row["city"]:
        raise _Rejected("invalid_place")
    if row.get("negative_prompt", "") != "" or job.get("negative_prompt", "") != "":
        raise _Rejected("negative_prompt_mismatch")
    if "negative_prompt" not in job and context["plan_config"].get("stage") != "weather_v3_plan":
        raise _Rejected("negative_prompt_not_proven")
    if row.get("quality_version") != context["execution"]["quality"].get("version"):
        raise _Rejected("result_quality_version_mismatch")
    if not all(row.get(field) is True for field in ("passed", "eligible_for_training", "geometry_ok", "weather_ok")):
        raise _Rejected("not_accepted")
    if "result_sha256" in row:
        unsigned = {key: value for key, value in row.items() if key != "result_sha256"}
        if row["result_sha256"] != common.fingerprint(unsigned):
            raise _Rejected("result_checksum_mismatch")
    dimensions = _dimensions(row.get("source_dimensions"))
    sampling = row.get("sampling")
    target_shape = _target_shape(dimensions, expected_service["canvas"])
    if (not isinstance(sampling, dict) or sampling.get("canvas_policy") != "source_aspect_v1"
            or sampling.get("source_dimensions") != dimensions
            or sampling.get("target_shape") != target_shape
            or sampling.get("raw_dimensions") != target_shape[::-1]):
        raise _Rejected("sampling_canvas_mismatch")
    for key, value in expected_service["sampling"].items():
        if key in sampling and sampling[key] != value:
            raise _Rejected("sampling_parameter_mismatch")
    if not isinstance(row.get("source_sha256"), str) or not isinstance(row.get("output_sha256"), str):
        raise _Rejected("image_checksum_missing")
    _image(row["source_path"], dimensions, expected_sha=row["source_sha256"], hashes=hashes)
    _image(row["output_path"], dimensions, png=True, expected_sha=row["output_sha256"], hashes=hashes)
    raw_sha = _image(row["raw_output_path"], target_shape[::-1], png=True,
                     expected_sha=row.get("raw_output_sha256"), hashes=hashes)
    provenance = {key: context[key] for key in ("ledger_path", "ledger_sha256", "execution_path", "execution_sha256",
                                               "plan_path", "plan_sha256", "plan_config_path", "plan_config_sha256")}
    provenance.update(ledger_line=line, ledger_row_sha256=common.fingerprint(row),
                      original_sample_id=row["sample_id"], original_seed=row["seed"],
                      original_prompt_policy=row.get("prompt_policy"), source_path=row["source_path"],
                      source_sha256=row["source_sha256"], output_path=row["output_path"],
                      output_sha256=row["output_sha256"], raw_output_path=row["raw_output_path"],
                      raw_output_sha256=raw_sha, service_fingerprint=common.fingerprint(expected_service),
                      original_quality_fingerprint=common.fingerprint(context["execution"]["quality"]),
                      negative_prompt_evidence="explicit_empty" if "negative_prompt" in job else
                      "weather_v3_plan_and_generator_fixed_empty")
    return {**copy.deepcopy(row), "negative_prompt": "", "raw_output_sha256": raw_sha, "reuse_provenance": provenance}


def collect_candidates(result_paths, *, expected_service, expected_quality=None):
    """Return ``{(city, place_id, domain): [validated original rows]}`` and counts.

    Only accepted, byte-verified current prompts become candidates. The supplied
    service identity must match in full. Quality definitions may differ only in
    the descriptive top-level ``prompt_bank`` label. Errors reject the affected
    row/ledger and are recorded; this function never writes or launches models.
    """
    expected_service = _identity(expected_service)
    if expected_quality is None:
        from experiments.qwen_curriculum.quality_v3 import quality_definition_v3
        expected_quality = quality_definition_v3(clip_device="cpu")
    expected_quality = _quality_settings(expected_quality)
    candidates, hashes = defaultdict(list), {}
    report = dict(scanned_rows=0, exact_prompt_rows=0, accepted_candidates=0, by_domain={}, rejected={}, per_ledger=[])
    domain_counts, reasons = Counter(), Counter()
    seen_ledgers = set()
    current_prompts = {(domain, variant): prompt_bank.prompt(domain, variant)
                       for domain, variants in prompt_bank.VARIANTS.items()
                       if prompt_bank.DOMAIN_WEIGHTS.get(domain, 0) > 0 for variant in variants}
    for raw_path in result_paths:
        ledger = Path(raw_path).expanduser().resolve()
        if ledger in seen_ledgers:
            continue
        seen_ledgers.add(ledger)
        local = dict(path=str(ledger), scanned_rows=0, exact_prompt_rows=0, accepted_candidates=0, rejected={})
        local_reasons = Counter()
        context, context_error = None, None
        seen_ids = set()
        try:
            for line, row in _rows(ledger):
                report["scanned_rows"] += 1
                local["scanned_rows"] += 1
                pair = row.get("condition"), row.get("prompt_variant")
                if (not all(isinstance(value, str) for value in pair)
                        or pair not in current_prompts or row.get("prompt") != current_prompts[pair]):
                    local_reasons["prompt_not_current_or_domain_disabled"] += 1
                    continue
                report["exact_prompt_rows"] += 1
                local["exact_prompt_rows"] += 1
                if row.get("passed") is not True or row.get("eligible_for_training") is not True:
                    local_reasons["not_accepted"] += 1
                    continue
                if context is None and context_error is None:
                    try:
                        context = _ledger_context(ledger, expected_service, expected_quality)
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        context_error = str(exc) if isinstance(exc, _Rejected) else "ledger_context_invalid"
                if context_error is not None:
                    local_reasons[context_error] += 1
                    continue
                try:
                    if row.get("sample_id") in seen_ids:
                        raise _Rejected("duplicate_result_sample_id")
                    seen_ids.add(row.get("sample_id"))
                    candidate = _candidate(row, line, context, expected_service, hashes)
                except (OSError, ValueError, KeyError, TypeError, SyntaxError) as exc:
                    local_reasons[str(exc) if isinstance(exc, _Rejected) else "candidate_invalid_or_corrupt"] += 1
                    continue
                key = candidate["city"], candidate["place_id"], candidate["condition"]
                candidates[key].append(candidate)
                domain_counts[candidate["condition"]] += 1
                report["accepted_candidates"] += 1
                local["accepted_candidates"] += 1
        except (OSError, ValueError, UnicodeError) as exc:
            local_reasons[str(exc) if isinstance(exc, _Rejected) else "ledger_unreadable"] += 1
        local["rejected"] = dict(sorted(local_reasons.items()))
        report["per_ledger"].append(local)
        reasons.update(local_reasons)
    report["by_domain"] = dict(sorted(domain_counts.items()))
    report["rejected"] = dict(sorted(reasons.items()))
    return dict(candidates), report
