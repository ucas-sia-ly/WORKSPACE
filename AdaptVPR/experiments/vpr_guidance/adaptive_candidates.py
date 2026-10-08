"""Experimental in-process generation, weather verification and SALAD selection.

Fixed and adaptive budgets use the same sampler, gates and seeds. The adaptive
policy stops after a plausible weather candidate meets the current student's
utility/mining targets, retaining the best eligible candidate seen so far.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from common import (ADAPTVPR_ROOT, GUIDANCE_ROOT, SALAD_ROOT, file_sha256,
                    use_adaptvpr, write_json, write_jsonl)
from generate_candidates import (_check_config, _fingerprint, _generate_one,
                                 _read_manifest, _request_config, _validated_rows, select_prompts)

use_adaptvpr()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("prompts", "image-root", "real-data", "checkpoint", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--cities", nargs="+")
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--num-sources", type=int, required=True)
    parser.add_argument("--num-candidates", type=int, default=4, help="Maximum attempts per prompt")
    parser.add_argument("--min-candidates", type=int, default=1)
    parser.add_argument("--sampling-policy", choices=["adaptive", "fixed"], default="adaptive")
    parser.add_argument("--strategy", choices=["released", "sdedit_0.85"], default="sdedit_0.85")
    parser.add_argument("--prompt-variant", choices=["released", "positive"], default="positive")
    parser.add_argument("--weather-min-shift", type=float, default=6.0,
                        help="Exploratory CLIP shift floor, NOT a calibrated probability")
    parser.add_argument("--disable-weather-gate", action="store_true")
    parser.add_argument("--stop-utility", type=float, default=0.0, help="Stop requires utility strictly above this")
    parser.add_argument("--stop-mining-probability", type=float, default=0.25)
    parser.add_argument("--selection", choices=["hardness", "random"], default="hardness")
    parser.add_argument("--lora", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-args", default="", help="Extra scorer flags, e.g. --device cpu")
    parser.add_argument("--backbone-repo", type=Path)
    parser.add_argument("--plan-only", action="store_true", help="Print frozen config without writing/loading models")
    args = parser.parse_args(argv)
    if args.num_sources <= 0 or args.offset < 0 or not 1 <= args.min_candidates <= args.num_candidates:
        parser.error("Positive --num-sources and 1 <= --min-candidates <= --num-candidates required")
    if (not math.isfinite(args.weather_min_shift) or not math.isfinite(args.stop_utility)
            or args.stop_utility < 0 or not 0 <= args.stop_mining_probability <= 1):
        parser.error("Finite weather/utility thresholds, nonnegative utility and mining probability in [0,1] required")
    for name in ("prompts", "image_root", "real_data", "checkpoint", "output_dir", "lora", "backbone_repo"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    return args


def scorer_args(args):
    from run_loop import _extra_flags
    from score_candidates import parse_args as parse_score
    extra = _extra_flags(args.score_args, ["--candidates", "--checkpoint", "--real-data", "--cities",
                                          "--output-dir", "--selection", "--seed", "--backbone-repo", "--help"])
    flags = ["--candidates", str(args.output_dir / "candidates.jsonl"), "--checkpoint", str(args.checkpoint),
             "--real-data", str(args.real_data), "--output-dir", str(args.output_dir),
             "--selection", args.selection, "--seed", str(args.seed), *extra]
    if args.cities:
        flags += ["--cities", *args.cities]
    if args.backbone_repo:
        flags += ["--backbone-repo", str(args.backbone_repo)]
    return parse_score(flags)


def build_config(args, score_args, entries, sources):
    from adapters import iclight_sd15_fc as adapter
    from experiments.generation_diagnosis.runner import STRATEGIES
    from prompts.rules import global_negative_prompt
    from run_loop import _input_signature
    from verification.evaluator import ROUTE_THRESHOLDS
    from weather_signal import weather_definition

    config = _request_config(args, entries, sources, adapter, global_negative_prompt(), ROUTE_THRESHOLDS["global"])
    config.pop("fingerprint")
    config.update(schema_version=2, execution="experimental_online_salad",
                  checkpoint_sha256=file_sha256(args.checkpoint),
                  score_args={key: str(value) if isinstance(value, Path) else value
                              for key, value in vars(score_args).items() if key not in {"candidates", "output_dir"}},
                  sampling_policy=args.sampling_policy, min_candidates=args.min_candidates,
                  selection=args.selection, strategy=args.strategy, prompt_variant=args.prompt_variant,
                  weather_gate={"enabled": not args.disable_weather_gate, "min_shift": args.weather_min_shift,
                                "definition": weather_definition(), "comparison": "strictly_greater",
                                "calibration": "exploratory_agent_visual_labels"},
                  stop={"utility_strictly_above": args.stop_utility,
                        "mining_probability_at_least": args.stop_mining_probability})
    config["sampling"].update(STRATEGIES[args.strategy])
    init = STRATEGIES[args.strategy]["init_strength"]
    config["sampling"].update(
        low_num_inference_steps=adapter.DEFAULT_INFERENCE_STEPS if init is None
        else max(1, int(adapter.DEFAULT_INFERENCE_STEPS / init)),
        high_num_inference_steps={"rain": max(1, int(adapter.DEFAULT_HIGHRES_STEPS / adapter.RAIN_HIGHRES_DENOISE)),
                                  "other": max(1, int(adapter.DEFAULT_HIGHRES_STEPS / adapter.DEFAULT_HIGHRES_DENOISE))})
    cities = args.cities or sorted(path.stem for path in (args.real_data / "Dataframes").glob("*.csv"))
    inputs = [args.real_data / "Dataframes" / f"{city}.csv" for city in cities]
    inputs += [args.real_data / "Images" / city for city in cities]
    inputs += [Path(path) for path in config["model_paths"].values() if path]
    if args.backbone_repo:
        inputs.append(args.backbone_repo)
    config["input_signatures"] = {str(path): _input_signature(path, ignore_runtime=True) for path in inputs}
    config["implementation_sha256"].update({str(path.relative_to(ADAPTVPR_ROOT)): file_sha256(path)
        for path in [*GUIDANCE_ROOT.glob("*.py"), ADAPTVPR_ROOT / "experiments/generation_diagnosis/runner.py"]})
    config["salad_implementation_sha256"] = {str(path.relative_to(SALAD_ROOT)): file_sha256(path)
        for path in [*(SALAD_ROOT / "workflow").glob("*.py"), *(SALAD_ROOT / "models").rglob("*.py")]}
    config["fingerprint"] = _fingerprint(config)
    return config


def qualifies(row, args):
    return (row.get("eligible_for_training") is True and row.get("plausible") is True
            and row.get("utility", 0) > args.stop_utility
            and row.get("mining_probability", 0) >= args.stop_mining_probability)


def stopping_reason(rows, args):
    if (args.sampling_policy == "adaptive" and len(rows) >= args.min_candidates
            and any(qualifies(row, args) for row in rows)):
        return "target_reached"
    if len(rows) >= args.num_candidates:
        return "budget_exhausted"
    return None


def seal_row(row):
    row = {key: value for key, value in row.items() if key != "record_sha256"}
    return {**row, "record_sha256": _fingerprint(row)}


def validated_prefixes(rows, config, output_dir, args):
    """Validate all records, then resume only uninterrupted candidate prefixes.

    If an artifact is corrupt, discard its group's later records so early stop
    cannot depend on a different regenerated prefix. Existing images are kept.
    """
    for row in rows:
        if seal_row(row)["record_sha256"] != row.get("record_sha256"):
            raise ValueError("Adaptive candidate record checksum differs")
    # Reuse generation's identity/image/verifier validation. Experimental weather
    # and plausibility eligibility is distinct from the original Global verdict.
    copies = [{**row, "eligible_for_training": row.get("passed")} for row in rows]
    valid = _validated_rows(copies, config, output_dir, legacy=False)
    valid_keys = {(row["sample_id"], row["candidate_index"]) for row in valid}
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["sample_id"]][row["candidate_index"]] = row
    prefixes = {}
    for sample_id in config["sample_ids"]:
        prefix = []
        for index in range(args.num_candidates):
            if (sample_id, index) not in valid_keys:
                break
            if stopping_reason(prefix, args):
                raise ValueError(f"Candidate manifest contains records after adaptive stop: {sample_id}")
            prefix.append(grouped[sample_id][index])
        prefixes[sample_id] = prefix
    return prefixes


class ExperimentalICBackend:
    """Reuse the two-stage diagnostic sampler and cache one source VAE latent."""

    def __init__(self, strategy, scratch):
        from adapters import iclight_sd15_fc as adapter
        from experiments.generation_diagnosis.runner import STRATEGIES
        self.adapter, self.settings, self.scratch = adapter, STRATEGIES[strategy], scratch
        self.GenerateRequest = adapter.GenerateRequest
        self.pt2i, self.pi2i, self.vae = adapter.load_pipeline()
        self._source_key, self._condition = None, None

    def generate(self, request):
        import torch
        from PIL import Image
        from verification.evaluator import DualTraitEvaluator

        with Image.open(request.image_path) as image:
            source = image.convert("RGB")
        width, height = self.adapter._valid_size(source)
        key = (DualTraitEvaluator._clip_cache_key(source), width, height)
        init, cfg = self.settings["init_strength"], self.settings["guidance_scale"]
        with torch.inference_mode():
            if self._source_key != key:
                self._condition = self.adapter._concat_condition(source, self.vae, width, height)
                self._source_key = key
            generator = torch.Generator("cuda").manual_seed(request.seed)
            kwargs = dict(prompt=request.prompt, negative_prompt=request.negative_prompt, guidance_scale=cfg,
                          generator=generator, cross_attention_kwargs={"concat_conds": self._condition},
                          num_inference_steps=self.adapter.DEFAULT_INFERENCE_STEPS if init is None
                          else max(1, int(self.adapter.DEFAULT_INFERENCE_STEPS / init)))
            if init is None:
                low = self.pt2i(width=width, height=height, **kwargs).images[0]
            else:
                low = self.pi2i(image=source.resize((width, height)), strength=init, **kwargs).images[0]
            result = self.pi2i(prompt=request.prompt, negative_prompt=request.negative_prompt, image=low,
                              strength=request.highres_denoise,
                              num_inference_steps=max(1, int(self.adapter.DEFAULT_HIGHRES_STEPS / request.highres_denoise)),
                              guidance_scale=cfg, generator=generator,
                              cross_attention_kwargs={"concat_conds": self._condition}).images[0]
        path = self.scratch / "generated.png"
        result.convert("RGB").save(path)
        return {"result_path": str(path)}


def finish(output_dir, config, groups, args, metadata, elapsed):
    from feedback import select_per_group
    rows = [row for sample_id in config["sample_ids"] for row in groups[sample_id]]
    scored = [row for row in rows if row.get("score_status") == "scored"]
    selected = select_per_group([row for row in scored if row["eligible_for_training"]], args.selection, args.seed)
    budget = args.num_sources * args.num_candidates
    reasons = {sample_id: stopping_reason(group, args) for sample_id, group in groups.items()}
    summary = {**metadata, "status": "complete", "config_fingerprint": config["fingerprint"],
               "sampling_policy": args.sampling_policy, "selection": args.selection,
               "candidates": len(rows), "fixed_budget_candidates": budget,
               "generation_calls_saved": budget - len(rows), "saved_fraction": 1 - len(rows) / budget,
               "global_verified": sum(row["passed"] for row in rows),
               "weather_accepted": sum(row["weather_ok"] for row in rows),
               "eligible_candidates": sum(row["eligible_for_training"] for row in rows),
               "groups_selected": len(selected), "stop_reasons": dict(Counter(reasons.values())),
               "groups": [{"sample_id": key, "attempts": len(group), "stop_reason": reasons[key],
                           "target_found": any(qualifies(row, args) for row in group)} for key, group in groups.items()],
               "elapsed_this_invocation_seconds": elapsed,
               "comparison_scope": "budget experiment; adaptive search changes candidate and selection distributions"}
    write_jsonl(output_dir / "scored.jsonl", scored)
    write_jsonl(output_dir / "selected.jsonl", selected)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "generation_complete.json", {
        "config_fingerprint": config["fingerprint"], "metadata": metadata,
        "outputs": {name: file_sha256(output_dir / name)
                    for name in ("candidates.jsonl", "scored.jsonl", "selected.jsonl", "summary.json")}})
    print(f"[adaptive complete] candidates={len(rows)}/{budget} selected={len(selected)} "
          f"calls_saved={budget-len(rows)}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    score_args = scorer_args(args)
    from PIL import Image
    from generation.batch import resolve_prompt_source
    from experiments.generation_diagnosis.runner import prompt_for

    entries = select_prompts(args.prompts, args.cities, args.conditions, args.seed, args.offset, args.num_sources)
    sources = {}
    for entry in entries:
        source_path = resolve_prompt_source(entry, args.image_root).resolve()
        if not source_path.is_relative_to(args.image_root):
            raise ValueError(f"Source outside --image-root: {source_path}")
        with Image.open(source_path) as image:
            image.verify()
        sources[entry["sample_id"]] = source_path
        if args.prompt_variant == "positive":
            entry["prompt"] = prompt_for(entry["condition"], "positive")
    config = build_config(args, score_args, entries, sources)
    if args.plan_only:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return
    output_dir = args.output_dir
    manifest, config_path = output_dir / "candidates.jsonl", output_dir / "generation_config.json"
    rows, repair = _read_manifest(manifest)
    if _check_config(config_path, config, bool(rows)):
        raise ValueError("Adaptive experiments cannot migrate legacy fixed-K records")
    groups = validated_prefixes(rows, config, output_dir, args)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    if not config_path.exists():
        write_json(config_path, config)
    prefixes = [row for key in config["sample_ids"] for row in groups[key]]
    if repair or len(prefixes) != len(rows) or not manifest.exists():
        write_jsonl(manifest, prefixes)
    complete_path = output_dir / "generation_complete.json"
    context_path = output_dir / "scoring_context.json"
    if all(stopping_reason(group, args) for group in groups.values()) and complete_path.exists():
        complete = json.loads(complete_path.read_text())
        if complete.get("config_fingerprint") != config["fingerprint"]:
            raise ValueError("Adaptive completion fingerprint differs")
        if all((output_dir / name).is_file() and file_sha256(output_dir / name) == digest
               for name, digest in complete.get("outputs", {}).items()) and len(complete.get("outputs", {})) == 4:
            print("[skip] validated complete adaptive run; no models loaded", flush=True)
            return
    started = time.monotonic()
    complete_path.unlink(missing_ok=True)
    if all(stopping_reason(group, args) for group in groups.values()) and context_path.exists():
        context = json.loads(context_path.read_text())
        if context.get("config_fingerprint") != config["fingerprint"]:
            raise ValueError("Saved scoring context fingerprint differs")
        finish(output_dir, config, groups, args, context["metadata"], time.monotonic() - started)
        return
    from online_scoring import OnlineScorer
    scorer = OnlineScorer(score_args, sources.values())
    metadata = scorer.metadata()
    write_json(context_path, {"config_fingerprint": config["fingerprint"], "metadata": metadata})
    pending = any(not stopping_reason(group, args) for group in groups.values())
    if pending:
        from weather_signal import WeatherSignal, WeatherSignalEvaluator
        old_lora = os.environ.get("ADAPTVPR_LORA_CHECKPOINT")
        try:
            if args.lora:
                os.environ["ADAPTVPR_LORA_CHECKPOINT"] = str(args.lora)
            else:
                os.environ.pop("ADAPTVPR_LORA_CHECKPOINT", None)
            with tempfile.TemporaryDirectory(prefix=".adaptive_tmp_", dir=output_dir) as scratch_name:
                backend = ExperimentalICBackend(args.strategy, Path(scratch_name).resolve())
                evaluator = WeatherSignalEvaluator()
                weather = WeatherSignal(evaluator)
                with manifest.open("a", encoding="utf-8") as handle:
                    for entry in entries:
                        sample_id, group = entry["sample_id"], groups[entry["sample_id"]]
                        if stopping_reason(group, args):
                            continue
                        with Image.open(sources[sample_id]) as image:
                            source = image.convert("RGB")
                        while not stopping_reason(group, args):
                            tick = time.monotonic()
                            row = _generate_one(backend, evaluator, source, sources[sample_id], entry,
                                                len(group), config, backend.scratch, output_dir)
                            with Image.open(row["output_path"]) as image:
                                row.update(weather.measure(source, image.convert("RGB"), entry["condition"]))
                                row.update(source_dimensions=list(source.size), output_dimensions=list(image.size))
                            row["weather_ok"] = args.disable_weather_gate or row["weather_shift"] > args.weather_min_shift
                            if row["passed"] and row["weather_ok"]:
                                row.update(scorer.score(row), score_status="scored")
                                row["eligible_for_training"] = row["plausible"]
                            else:
                                row.update(eligible_for_training=False, plausible=False,
                                           score_status="rejected_by_verifier" if not row["passed"] else "rejected_by_weather")
                            row["elapsed_seconds"] = time.monotonic() - tick
                            row = seal_row(row)
                            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                            handle.flush()
                            os.fsync(handle.fileno())
                            group.append(row)
                        print(f"[adaptive] {sample_id}: {len(group)} attempts, {stopping_reason(group, args)}", flush=True)
        finally:
            if old_lora is None:
                os.environ.pop("ADAPTVPR_LORA_CHECKPOINT", None)
            else:
                os.environ["ADAPTVPR_LORA_CHECKPOINT"] = old_lora
    finish(output_dir, config, groups, args, metadata, time.monotonic() - started)


if __name__ == "__main__":
    main()
