"""Resume a matched IC-Light prompt ablation or Qwen weather-edit comparison.

This experiment does not change the production adapters or evaluator. Sources
and outputs are PNG snapshots; the production evaluator still uses its usual
JPEG temporary inputs for matching, preserving comparability with Claude's grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ADAPTVPR_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = ADAPTVPR_ROOT.parent
sys.path.insert(0, str(ADAPTVPR_ROOT))
from experiments.vpr_guidance.common import (  # noqa: E402
    candidate_seed, file_sha256, read_jsonl, use_adaptvpr, write_json, write_jsonl,
)

CONDITIONS = ["overcast", "rain", "snow", "fog", "night"]
STRATEGIES = {
    "released": {"init_strength": None, "guidance_scale": 7.5},
    "released_cfg3": {"init_strength": None, "guidance_scale": 3.0},
    "sdedit_0.85": {"init_strength": 0.85, "guidance_scale": 7.5},
    "sdedit_0.70": {"init_strength": 0.70, "guidance_scale": 7.5},
    "sdedit_0.55": {"init_strength": 0.55, "guidance_scale": 7.5},
}
# Weather first, short positive descriptions; these also strengthen the requested
# effect, so this arm is not interpreted as an isolated negation intervention.
POSITIVE_WEATHER = {
    "overcast": "Overcast street scene, uniform grey cloudy sky, soft diffuse daylight, muted contrast.",
    "rain": "Rainy street scene, overcast sky, visible light rain streaks, wet pavement, subtle road reflections.",
    "snow": "Snowy winter street scene, overcast sky, visible falling snowflakes, thin snow on horizontal surfaces.",
    "fog": "Foggy street scene, realistic atmospheric fog, reduced distant visibility, nearby geometry readable.",
    "night": "Urban night street scene, dark night sky, dim street lighting, reduced ambient brightness, readable details.",
}
POSITIVE_PRESERVE = (
    " Natural street-view photo. Preserve the original camera viewpoint, road layout, "
    "buildings, facades, signs, windows and vehicles."
)


def fingerprint(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def prompt_for(condition: str, variant: str) -> str:
    from prompts.rules import build_structured_prompt

    released = build_structured_prompt(route="global", weather=condition, occlusion=None)
    if variant == "released":
        return released
    if variant == "no_negations":
        # Delete only the complete Avoid / Do not sentences. Do not rewrite
        # remaining clauses (including overcast's "no strong shadows").
        return re.sub(r"(?:Avoid|Do not)\b[^.]*\.\s*", "", released).strip()
    if variant == "positive":
        return POSITIVE_WEATHER[condition] + POSITIVE_PRESERVE
    raise ValueError(f"Unknown prompt variant: {variant}")


def select_sources(manifest: Path, count: int) -> list[str]:
    if count <= 0:
        raise ValueError("--num-sources must be positive")
    # Preserve the original strings for Claude's seed formula, even if a path
    # could be normalized to an equivalent location.
    sources = sorted({r["source_path"] for r in read_jsonl(manifest)})[::5]
    if len(sources) < count:
        raise ValueError(f"Only {len(sources)} sources available after original [::5] selection")
    selected = sources[:count]
    for source in selected:
        if not Path(source).is_file():
            raise ValueError(f"Missing source: {source}")
    return selected


def read_checkpoint(path: Path) -> list[dict]:
    """Recover a torn final append, rejecting corrupt complete/interior rows."""
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    rows = []
    rewrite = bool(lines and not lines[-1].endswith(b"\n"))
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                break
            raise ValueError(f"Invalid complete checkpoint row {index + 1}: {path}")
        if not isinstance(row, dict):
            raise ValueError(f"Checkpoint row {index + 1} is not an object")
        rows.append(row)
    if rewrite:
        write_jsonl(path, rows)
    return rows


def row_key(row: dict) -> tuple:
    return row["src"], row["cond"], row["strat"], row["prompt_variant"]


def append_checkpoint(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def summarize(rows: list[dict]) -> dict:
    # A failed attempt may be retried on resume; use its most recent result.
    latest = {row_key(row): row for row in rows}
    groups = defaultdict(list)
    for row in latest.values():
        groups[(row["method"], row["cond"])].append(row)
        groups[(row["method"], "all")].append(row)
    summaries = []
    for (method, condition), group in sorted(groups.items()):
        evaluated = [row for row in group if row["status"] == "ok"]
        reasons = defaultdict(int)
        for row in evaluated:
            for reason in row["failure_reasons"]:
                reasons[reason] += 1
        summaries.append({
            "method": method, "cond": condition, "attempted": len(group),
            "evaluated": len(evaluated), "errors": len(group) - len(evaluated),
            "geo_ok": sum(row["geo_ok"] for row in evaluated),
            "div_ok": sum(row["div_ok"] for row in evaluated),
            "passed": sum(row["passed"] for row in evaluated),
            "median_geo": statistics.median(row["s_geo"] for row in evaluated) if evaluated else None,
            "median_div": statistics.median(row["s_div"] for row in evaluated) if evaluated else None,
            "failure_reasons": dict(reasons),
        })
    return {"groups": summaries, "rows": len(latest), "successful_evaluations":
            sum(row["status"] == "ok" for row in latest.values())}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["iclight", "qwen"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sources-manifest", type=Path,
                        default=WORKSPACE_ROOT / "outputs/vpr_guidance_smoke/cand_probe/candidates.jsonl")
    parser.add_argument("--num-sources", type=int, default=8)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument("--strategies", nargs="+", choices=[*STRATEGIES, "qwen_edit"])
    parser.add_argument("--prompt-variants", nargs="+", choices=["released", "no_negations", "positive"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qwen-url", default="http://127.0.0.1:8001/generate")
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--plan-only", action="store_true", help="Write frozen config without loading GPU models")
    args = parser.parse_args(argv)
    args.strategies = args.strategies or (["released", "sdedit_0.85"] if args.mode == "iclight" else ["qwen_edit"])
    args.prompt_variants = args.prompt_variants or (["released", "no_negations", "positive"]
                                                  if args.mode == "iclight" else ["released", "positive"])
    if (args.mode == "qwen" and args.strategies != ["qwen_edit"]
            or args.mode == "iclight" and "qwen_edit" in args.strategies):
        parser.error("IC-Light requires IC-Light strategies; Qwen requires only qwen_edit")
    for name in ("conditions", "strategies", "prompt_variants"):
        if len(getattr(args, name)) != len(set(getattr(args, name))):
            parser.error(f"--{name.replace('_', '-')} contains duplicates")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.sources_manifest = args.sources_manifest.expanduser().resolve()
    return args


def build_config(args, sources: list[str]) -> dict:
    from adapters import iclight_sd15_fc as adapter
    from prompts.rules import global_negative_prompt
    from verification.evaluator import ROUTE_THRESHOLDS

    tokenizer = None
    base = os.getenv("ICLIGHT_BASE_MODEL_PATH", "")
    if base and Path(base).is_dir():
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer", local_files_only=True)
    prompts = []
    for condition in args.conditions:
        for variant in args.prompt_variants:
            prompt = prompt_for(condition, variant)
            row = {"cond": condition, "prompt_variant": variant, "prompt": prompt}
            if tokenizer is not None:
                ids = tokenizer(prompt, truncation=False).input_ids
                row.update(token_count=len(ids), truncated=len(ids) > tokenizer.model_max_length,
                           clip_visible_text=tokenizer.decode(ids[1:tokenizer.model_max_length - 1]),
                           clip_truncated_text=tokenizer.decode(ids[tokenizer.model_max_length - 1:-1])
                           if len(ids) > tokenizer.model_max_length else "")
            prompts.append(row)
    payload = {
        "schema_version": 1, "mode": args.mode, "output_dir": str(args.output_dir),
        "sources_manifest": str(args.sources_manifest), "sources_manifest_sha256": file_sha256(args.sources_manifest),
        "source_selection": "sorted(unique(source_path))[::5][:num_sources]", "seed": args.seed,
        "seed_formula": "candidate_seed(seed, original_source_path + '|' + condition, 0)",
        "sources": [{"src": i, "source_path": path, "source_sha256": file_sha256(Path(path))}
                    for i, path in enumerate(sources)],
        "conditions": args.conditions, "strategies": args.strategies, "prompt_variants": args.prompt_variants,
        "prompts": prompts, "negative_prompt": global_negative_prompt() if args.mode == "iclight" else "",
        "sampling": ({
            "strategies": {key: STRATEGIES[key] for key in args.strategies},
            "low_steps_target": 25, "highres_steps_target": 20, "highres_scale": 1.0,
            "rain_highres_denoise": adapter.RAIN_HIGHRES_DENOISE,
            "default_highres_denoise": adapter.DEFAULT_HIGHRES_DENOISE,
            "scheduler": "DDIMScheduler",
        } if args.mode == "iclight" else {"infer_steps": 4, "guidance_scale": 1.0,
                                          "qwen_url": args.qwen_url}),
        "verification": {"thresholds": ROUTE_THRESHOLDS["global"], "output_format": "PNG",
                         "matcher_temp_format": "production JPEG95", "mock": False,
                         "matcher": os.getenv("ADAPTVPR_MATCHER_NAME", "superpoint-lightglue"),
                         "clip_model": os.getenv("ADAPTVPR_CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")},
        "model_paths": {key: os.getenv(key) for key in ("ICLIGHT_BASE_MODEL_PATH", "ICLIGHT_MODEL_PATH")}
        if args.mode == "iclight" else {},
        "implementation_sha256": {
            name: file_sha256(ADAPTVPR_ROOT / name) for name in (
                "experiments/generation_diagnosis/runner.py", "adapters/iclight_sd15_fc.py",
                "adapters/lightx2v_qwen_image_edit.py", "verification/evaluator.py", "prompts/rules.py")},
        "comparability_note": "Claude grid had JPEG95 final outputs; these use PNG final outputs. "
                              "Production verifier JPEG95 temporary matching is retained.",
        "qwen_output_normalization": "Retain raw PNG; LANCZOS resize to source dimensions before evaluation "
                                     "as generation/lightx2v.py does. Adaptive Qwen aspect ratio can change geometry."
                                     if args.mode == "qwen" else None,
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


class ICBackend:
    def __init__(self):
        from adapters import iclight_sd15_fc as adapter
        self.adapter = adapter
        self.pt2i, self.pi2i, self.vae = adapter.load_pipeline()
        self.health = {"status": "ok", "execution": "local load_pipeline", "model_loaded": True,
                       "expected_commit": adapter.ICLIGHT_COMMIT, "source_commit": adapter.state.source_commit,
                       "base_model_revision": adapter.BASE_MODEL_REVISION,
                       "checkpoint_revision": adapter.CHECKPOINT_REVISION}

    def generate(self, source, source_png, prompt, negative_prompt, condition, seed, strategy):
        import torch
        settings = STRATEGIES[strategy]
        width, height = self.adapter._valid_size(source)
        init, cfg = settings["init_strength"], settings["guidance_scale"]
        denoise = self.adapter.RAIN_HIGHRES_DENOISE if condition == "rain" else self.adapter.DEFAULT_HIGHRES_DENOISE
        sampling = {**settings, "width": width, "height": height, "highres_scale": 1.0,
                    "low_num_inference_steps": 25 if init is None else max(1, int(25 / init)),
                    "highres_num_inference_steps": max(1, int(20 / denoise)), "highres_denoise": denoise}
        with torch.inference_mode():
            generator = torch.Generator("cuda").manual_seed(seed)
            condition_latent = self.adapter._concat_condition(source, self.vae, width, height)
            kwargs = dict(prompt=prompt, negative_prompt=negative_prompt, guidance_scale=cfg,
                          generator=generator, cross_attention_kwargs={"concat_conds": condition_latent},
                          num_inference_steps=sampling["low_num_inference_steps"])
            if init is None:
                low = self.pt2i(width=width, height=height, **kwargs).images[0]
            else:
                low = self.pi2i(image=source.resize((width, height)), strength=init, **kwargs).images[0]
            result = self.pi2i(prompt=prompt, negative_prompt=negative_prompt, image=low,
                              strength=denoise, num_inference_steps=sampling["highres_num_inference_steps"],
                              guidance_scale=cfg, generator=generator,
                              cross_attention_kwargs={"concat_conds": condition_latent}).images[0]
        return result.convert("RGB"), sampling


class QwenBackend:
    def __init__(self, url: str, timeout: float):
        import requests
        from generation.service_health import health_url
        self.url, self.timeout = url, timeout
        self.session = requests.Session()
        self.session.trust_env = False
        response = self.session.get(health_url(url), timeout=10)
        response.raise_for_status()
        self.health = response.json()
        if (not isinstance(self.health, dict) or self.health.get("status") != "ok"
                or self.health.get("model_loaded") is not True
                or self.health.get("generator_ready") is not True or self.health.get("error")):
            raise RuntimeError(f"Qwen adapter is not ready: {self.health}")
        expected = {"infer_steps": 4, "guidance_scale": 1.0}
        if any(self.health.get("sampling", {}).get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Qwen adapter sampling mismatch: {self.health.get('sampling')}")

    def generate(self, source, source_png, prompt, negative_prompt, condition, seed, strategy):
        from PIL import Image
        payload = {"image_path": str(source_png), "prompt": prompt, "negative_prompt": negative_prompt,
                   "seed": seed, "infer_steps": 4, "guidance_scale": 1.0}
        response = self.session.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        result_path = Path(data.get("result_path", ""))
        if not result_path.is_file():
            raise RuntimeError(f"Qwen returned an unreadable result_path: {result_path}")
        with Image.open(result_path) as image:
            result = image.convert("RGB")
        sampling = {"infer_steps": 4, "guidance_scale": 1.0, "adapter_result_path": str(result_path)}
        # Dimensions are [W, H]; the upstream target_shape is [H, W].
        # Optional fields keep results from older adapters readable.
        sampling.update({key: data[key] for key in (
            "canvas_policy", "source_dimensions", "raw_dimensions", "target_shape"
        ) if key in data})
        return result, sampling


def main(argv=None):
    args = parse_args(argv)
    use_adaptvpr()
    if os.getenv("ADAPTVPR_LORA_CHECKPOINT", "").strip():
        raise ValueError("This released-model experiment requires ADAPTVPR_LORA_CHECKPOINT to be unset")
    os.environ["ADAPTVPR_DISABLE_MOCK"] = "1"
    sources = select_sources(args.sources_manifest, args.num_sources)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = build_config(args, sources)
    config_path = args.output_dir / "generation_config.json"
    manifest = args.output_dir / "results.jsonl"
    rows = read_checkpoint(manifest)
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Configuration changed; use a new --output-dir")
    elif rows:
        raise ValueError("Cannot resume without generation_config.json")
    else:
        write_json(config_path, config)
    if args.plan_only:
        print(json.dumps({"config": str(config_path), "planned_images": args.num_sources * len(args.conditions)
                          * len(args.strategies) * len(args.prompt_variants)}, indent=2), flush=True)
        return

    from PIL import Image
    from verification.evaluator import DualTraitEvaluator

    done = {}
    for row in rows:
        if row.get("config_fingerprint") != config["fingerprint"]:
            raise ValueError("Checkpoint config fingerprint mismatch")
        if row["status"] == "ok":
            path = Path(row["output_path"])
            if not path.is_file() or file_sha256(path) != row["output_sha256"]:
                raise ValueError(f"Checkpoint image changed or missing: {path}")
            done[row_key(row)] = row
    expected = args.num_sources * len(args.conditions) * len(args.strategies) * len(args.prompt_variants)
    if len(done) == expected:
        write_json(args.output_dir / "summary.json", summarize(rows))
        print(f"Already complete: {len(done)}/{expected}", flush=True)
        return
    backend = ICBackend() if args.mode == "iclight" else QwenBackend(args.qwen_url, args.request_timeout)
    write_json(args.output_dir / "service_health.json", backend.health)
    evaluator = DualTraitEvaluator(mock=False)
    images_dir = args.output_dir / "images"
    source_dir = args.output_dir / "sources"
    images_dir.mkdir(exist_ok=True)
    source_dir.mkdir(exist_ok=True)
    prompt_map = {(row["cond"], row["prompt_variant"]): row["prompt"] for row in config["prompts"]}
    for index, source_path in enumerate(sources):
        with Image.open(source_path) as image:
            source = image.convert("RGB")
        source_png = source_dir / f"s{index}.png"
        source.save(source_png, format="PNG")
        for condition in args.conditions:
            seed = candidate_seed(args.seed, f"{source_path}|{condition}", 0)
            for strategy in args.strategies:
                for variant in args.prompt_variants:
                    if (index, condition, strategy, variant) in done:
                        continue
                    output_path = images_dir / f"s{index}_{condition}_{strategy}_{variant}.png"
                    method = f"iclight_{variant}_{strategy}" if args.mode == "iclight" else f"qwen_{variant}"
                    row = {"mode": args.mode, "model": "iclight_sd15_fc" if args.mode == "iclight" else "qwen_image_edit_2511",
                           "src": index, "source_path": source_path, "source_png_path": str(source_png),
                           "cond": condition, "strat": strategy, "prompt_variant": variant,
                           "method": method, "seed": seed, "prompt": prompt_map[(condition, variant)],
                           "negative_prompt": config["negative_prompt"], "output_path": str(output_path),
                           "service_health": backend.health, "config_fingerprint": config["fingerprint"]}
                    started = time.monotonic()
                    try:
                        generated, sampling = backend.generate(source, source_png, row["prompt"],
                                                               row["negative_prompt"], condition, seed, strategy)
                        if args.mode == "qwen":
                            raw_path = output_path.with_name(output_path.stem + "_raw.png")
                            generated.save(raw_path, format="PNG")
                            row.update(raw_output_path=str(raw_path), raw_output_sha256=file_sha256(raw_path),
                                       raw_dimensions=list(generated.size), source_dimensions=list(source.size),
                                       resize_applied=generated.size != source.size,
                                       resize_method="LANCZOS to source size, matching generation/lightx2v.py")
                            if generated.size != source.size:
                                generated = generated.resize(source.size, Image.Resampling.LANCZOS)
                        generated.save(output_path, format="PNG")
                        with Image.open(output_path) as image:
                            generated = image.convert("RGB")
                        result = evaluator.evaluate(source, generated, entry={"route": "global", "weather": condition})
                        if not all(math.isfinite(value) for value in (result.s_geo, result.s_div)):
                            raise ValueError("Non-finite evaluator score")
                        row.update(status="ok", sampling=sampling, output_sha256=file_sha256(output_path),
                                   s_geo=result.s_geo, s_div=result.s_div, geo_ok=result.geo_ok,
                                   div_ok=result.div_ok, passed=result.passed,
                                   failure_reasons=(["geometry"] if not result.geo_ok else [])
                                   + (["diversity"] if not result.div_ok else []))
                    except Exception as exc:
                        row.update(status="error", error=f"{type(exc).__name__}: {exc}",
                                   failure_reasons=["execution_error"], elapsed_seconds=time.monotonic() - started)
                        append_checkpoint(manifest, row)
                        rows.append(row)
                        write_json(args.output_dir / "summary.json", summarize(rows))
                        raise
                    row["elapsed_seconds"] = time.monotonic() - started
                    append_checkpoint(manifest, row)
                    rows.append(row)
                    done[row_key(row)] = row
                    write_json(args.output_dir / "summary.json", summarize(rows))
                    print(f"[{len(done)}/{expected}] s{index} {condition} {strategy} {variant} "
                          f"geo={row['s_geo']:.3f} div={row['s_div']:.3f} passed={row['passed']} "
                          f"({row['elapsed_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
