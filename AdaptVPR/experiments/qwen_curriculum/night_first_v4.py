"""Opt-in v4 weather plans and candidates, separate from every frozen v3 plan.

Generation keeps the established durable request/resume implementation. It
cannot publish trainable images: a separate automated content audit must export
the final training manifest. Importing or preparing does not call Qwen.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import (campaign, common, prompt_bank_v4 as prompt_bank,
    quality_v3, run, source_reliability_v3, source_screen_v3, weather_io)

ROOT = common.WORKSPACE_ROOT / "outputs/weather_v4"
DEFAULT_OUTPUT = ROOT / "state"
DEFAULT_IMAGES = ROOT
DOMAINS = tuple(domain for domain, weight in prompt_bank.DOMAIN_WEIGHTS.items() if weight > 0)
_ENGINE = None


def quality_definition_v4(**kwargs):
    definition = quality_v3.quality_definition_v3(**kwargs)
    definition.update(version="4-candidate", prompt_bank=prompt_bank.POLICY_VERSION,
        training_eligibility="pending separate automated content audit; never inferred from geometry/CLIP")
    return definition


class CandidateVerifier(quality_v3.QwenQualityVerifierV3):
    """Geometry/CLIP acceptance is candidate status, never content approval."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.definition = quality_definition_v4(
            thresholds=self.thresholds, matcher_name=self.matcher_name, img_size=self.img_size,
            n_kpts=self.n_kpts, clip_device=self.clip_device, clip_model_name=self.clip_model_name)

    def evaluate(self, source, generated, condition):
        verdict = super().evaluate(source, generated, condition)
        verdict.update(base_quality_passed=verdict.get("passed") is True,
            eligible_for_training=False, content_status="pending_automated_audit",
            quality_version="4-candidate")
        return verdict


def _engine():
    """Private namespace reuses frozen engine bytes without mutating v3 globals.

    The older module freezes its own source SHA in existing plans. Loading that
    same implementation under a private name provides explicit v4 hooks while
    retaining its tested reservation, fsync, recovery and stop behaviour.
    """
    global _ENGINE
    if _ENGINE is None:
        name = "experiments.qwen_curriculum._night_first_v4_durable_engine"
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("night_first_v3.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        module.prompt_bank = prompt_bank
        module.ROOT, module.DEFAULT_OUTPUT, module.DEFAULT_IMAGES = ROOT, DEFAULT_OUTPUT, DEFAULT_IMAGES
        module.DOMAINS = DOMAINS
        module.load_plan = load_plan
        module.QwenQualityVerifierV3 = CandidateVerifier
        module._candidate_publish = module.publish
        module.publish = publish
        _ENGINE = module
    return _ENGINE


def publish(directory, config, jobs, generated, results, attempts, state, domain, error=None):
    """Expose geometry-passed candidates separately from trainable outputs."""
    report = _engine()._candidate_publish(
        directory, config, jobs, generated, results, attempts, state, domain, error)
    for condition, counts in report["by_domain"].items():
        rows = [row for row in results.values() if row["condition"] == condition]
        counts["geometry_weather_passed"] = counts["accepted"]
        counts["accepted"] = sum(row.get("eligible_for_training") is True for row in rows)
        counts["pending_content_audit"] = sum(row.get("base_quality_passed") is True
                                               and row.get("eligible_for_training") is not True for row in rows)
    report["geometry_weather_passed"] = report["accepted"]
    report["accepted"] = sum(row.get("eligible_for_training") is True for row in results.values())
    report["training_eligibility"] = "separate automated content-audit manifest required"
    common.write_json(directory / "summary.json", report)
    return report


def _safe_roots(directory, image_root):
    for path in (Path(directory).resolve(), Path(image_root).resolve()):
        for old in (common.WORKSPACE_ROOT / "outputs/weather_v3",
                    common.WORKSPACE_ROOT / "outputs/qwen_curriculum"):
            if path == old or old in path.parents:
                raise ValueError("v4 requires a new output root; historical campaigns are read-only")


def make_plan(sources, candidates=None, seed=42):
    """One source per place, deterministic quotas and independent weighted variant draws."""
    if candidates:
        raise ValueError("v4 does not import or relabel historical generation candidates")
    ordered = sorted(sources, key=lambda row: (row["city"], int(row["place_id"])))
    if len({(row["city"], int(row["place_id"])) for row in ordered}) != len(ordered):
        raise ValueError("Each place must appear exactly once")
    if len({row["source_sha256"] for row in ordered}) != len(ordered):
        raise ValueError("Duplicate source content across places")
    import random
    rng = random.Random(seed)
    rng.shuffle(ordered)
    quotas = weather_io.quotas(len(ordered), prompt_bank.DOMAIN_WEIGHTS)
    domains = [domain for domain, count in sorted(quotas.items()) for _ in range(count)]
    rng.shuffle(domains)
    jobs = []
    for source, domain in zip(ordered, domains):
        variant_key = f"{source['city']}/{source['place_id']}/{source['source_sha256']}/{domain}"
        variant = prompt_bank.choose_variant(domain, variant_key)
        sid = "qwen4_" + common.fingerprint([prompt_bank.POLICY_VERSION, variant_key])[:24]
        job = {key: source[key] for key in ("city", "place_id", "source_path", "source_sha256", "source_dimensions")}
        job.update(sample_id=sid, condition=domain, prompt_variant=variant,
            prompt=prompt_bank.prompt(domain, variant), negative_prompt="",
            prompt_policy=prompt_bank.POLICY_VERSION, seed=common.candidate_seed(seed, sid, 0))
        job["record_sha256"] = common.fingerprint(job)
        jobs.append(job)
    return jobs, {}


def prepare(args):
    directory = args.output_dir
    _safe_roots(directory, args.image_root)
    directory.mkdir(parents=True, exist_ok=True)
    engine = _engine()
    with run.run_lock(directory):
        if (directory / "plan_config.json").exists():
            config, jobs = load_plan(directory)
            if (config["seed"] != args.seed or config["real_data"] != str(args.real_data)
                    or config["image_root"] != str(args.image_root)):
                raise ValueError("Existing plan uses another seed, source pool or image root")
            engine.finish_import(directory, config, jobs)
            return config
        # Never merge an unsealed pre-existing generation directory into a new plan.
        if any((directory / name).exists() for name in ("plan.jsonl", "results.jsonl", "generated.jsonl", "attempts.jsonl")):
            raise ValueError("Unsealed generation state exists; use a new output directory")
        sources = source_screen_v3.screen_sources(args.real_data, directory / "source_screen",
            seed=args.seed, min_images=args.min_images, workers=args.workers)
        if not sources:
            raise ValueError("No suitable source places")
        sources, _ = source_reliability_v3.refine_sources(
            sources, args.real_data, directory / "source_reliability", seed=args.seed)
        if not sources:
            raise ValueError("No sources pass the source reliability gate")
        expected = json.loads((engine.LEGACY_ROOT / "state/all_city_campaign_config.json").read_text())["reference_execution"]["service"]
        jobs, _ = make_plan(sources, seed=args.seed)
        common.write_jsonl(directory / "plan.jsonl", jobs)
        common.write_jsonl(directory / "reuse_candidates.jsonl", [])
        common.write_json(directory / "reuse_report.json", {"policy": "no historical image reuse", "accepted_candidates": 0})
        modules = (sys.modules[__name__], engine, prompt_bank, prompt_bank.legacy, quality_v3, campaign, source_screen_v3,
                   source_reliability_v3, run, common, weather_io, engine.resume_recovery)
        static = {Path(module.__file__).resolve() for module in modules}
        definition = run.execution_definition(SimpleNamespace(run_dir=directory, device="cuda", cpu_threads=4))
        static.update(common.ADAPTVPR_ROOT / name for name in definition["implementation_sha256"])
        static.add(common.ADAPTVPR_ROOT / "adapters/lightx2v_qwen_image_edit.py")
        static.update((directory / "reuse_candidates.jsonl", directory / "reuse_report.json"))
        for child in ("source_screen", "source_reliability"):
            static.update((directory / child).glob("*.json"))
            static.update((directory / child).glob("*.jsonl"))
        static.update((args.real_data / "Dataframes").glob("*.csv"))
        config = campaign.frozen(directory / "plan_config.json", {
            "schema_version": 1, "stage": "place_coverage_weather_v4_candidates", "seed": args.seed,
            "real_data": str(args.real_data), "image_root": str(args.image_root),
            "num_places": len(jobs), "num_jobs": len(jobs), "views_per_place": 1,
            "selection": "seeded place shuffle, fixed domain quotas, stateless weighted variant draw",
            "domain_counts": dict(Counter(job["condition"] for job in jobs)),
            "prompt_bank": prompt_bank.definition(), "service": expected,
            "quality": quality_definition_v4(clip_device="cpu"), "default_domain": "night",
            "training_eligibility": "none before independent automated content audit",
            "plan_sha256": common.file_sha256(directory / "plan.jsonl"),
            "files_sha256": {str(path): common.file_sha256(path) for path in sorted(static)},
        })
        engine.finish_import(directory, config, jobs)
        return config


def load_plan(directory):
    config = campaign.config_file(directory / "plan_config.json")
    _safe_roots(directory, config["image_root"])
    if (config.get("stage") != "place_coverage_weather_v4_candidates"
            or config.get("prompt_bank") != prompt_bank.definition()
            or config.get("quality") != quality_definition_v4(clip_device="cpu")):
        raise ValueError("Not the frozen v4 candidate policy; use the original campaign entry point")
    if common.file_sha256(directory / "plan.jsonl") != config["plan_sha256"]:
        raise ValueError("Plan checksum mismatch")
    for path, digest in config["files_sha256"].items():
        if common.file_sha256(Path(path)) != digest:
            raise ValueError(f"Frozen input/code changed: {path}")
    jobs = weather_io.read_rows(directory / "plan.jsonl")
    if (len(jobs) != config["num_jobs"] or len({(job["city"], job["place_id"]) for job in jobs}) != len(jobs)
            or Counter(job["condition"] for job in jobs) != Counter(config["domain_counts"])):
        raise ValueError("Place coverage or domain quota changed")
    for job in jobs:
        body = {key: value for key, value in job.items() if key != "record_sha256"}
        if (job["record_sha256"] != common.fingerprint(body) or not prompt_bank.matches_record(job)
                or job["condition"] not in DOMAINS or not job["sample_id"].startswith("qwen4_")):
            raise ValueError("Invalid v4 planned job")
    return config, jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("prepare", "run", "generate", "check", "status"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--real-data", type=Path, default=common.WORKSPACE_ROOT / "dataset/gsv-cities")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-images", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--domain", choices=DOMAINS, default="night")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--qwen-url", default="http://127.0.0.1:8001/generate")
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--max-calls", type=int, default=0)
    args = parser.parse_args(argv)
    for name in ("output_dir", "image_root", "real_data"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    _safe_roots(args.output_dir, args.image_root)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    engine = _engine()
    if args.command == "prepare":
        config = prepare(args)
        print(json.dumps({"planned": config["num_jobs"], "domain_counts": config["domain_counts"],
            "trainable_before_content_audit": 0}, indent=2))
    elif args.command in {"run", "generate"}:
        engine.execute(args)
    elif args.command == "check":
        with run.run_lock(args.output_dir):
            config, jobs = load_plan(args.output_dir)
            generated, results, attempts = engine._read_state(args.output_dir, jobs, config)
            print(json.dumps({"planned": len(jobs), "saved": len(generated), "evaluated": len(results),
                "calls_reserved": len(attempts), "trainable_before_content_audit": 0}))
    else:
        print((args.output_dir / "summary.json").read_text())


if __name__ == "__main__":
    main()
