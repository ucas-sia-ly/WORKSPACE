#!/usr/bin/env python3
"""Run the public AdaptVPR pipeline on the 10-source demo."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES_CSV = Path(__file__).with_name("demo_10.csv")
RELEASED_PROMPTS = Path(__file__).with_name("demo_10_prompts.jsonl")
RELEASED_PROVENANCE = Path(__file__).with_name("demo_10_prompts.source.json")
REQUIRED_ROUTES = {"global", "local", "dual"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GSV-Cities demo paths with Qwen3-VL-4B planning or released prompts."
    )
    parser.add_argument("--mode", choices=("qwen4b", "prompt", "all"), default="qwen4b")
    parser.add_argument(
        "--gsvcities-root",
        type=Path,
        required=True,
        help="GSV-Cities root containing Images/CITY/source.jpg.",
    )
    parser.add_argument(
        "--prompts-jsonl",
        type=Path,
        help="Released Prompt JSONL; required for prompt/all mode.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reflection", choices=("on", "off"), default="on")
    # --reflection为on时，--max-reflections表示最大反射次数，范围为0-3，默认值为3
    parser.add_argument("--max-reflections", type=int, choices=range(0, 4), default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Check inputs/environment without inference.")
    parser.add_argument(
        "--strict", action="store_true",
        help="Generate all 10 sources using fixed released prompts, bypassing planner Skip; require all three routes.",
        # strict 使用已发布的固定提示词，让10张图都进入生成流程；质量评分仍按原规则执行。
    )
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.gsvcities_root = args.gsvcities_root.resolve()
    if args.prompts_jsonl:
        args.prompts_jsonl = args.prompts_jsonl.resolve()
    if args.strict:
        if args.mode == "all":
            parser.error("--strict runs the released-prompt demo once; use --mode qwen4b (default) or prompt")
        if args.prompts_jsonl is None:
            args.prompts_jsonl = RELEASED_PROMPTS
    if args.mode in {"prompt", "all"} and args.prompts_jsonl is None:
        parser.error("--prompts-jsonl is required for prompt/all mode")
    return args


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_sources() -> list[dict[str, str]]:
    with SOURCES_CSV.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"city", "source_id", "gsvcities_path"}
    if len(rows) != 10:
        raise RuntimeError(f"{SOURCES_CSV} must contain exactly 10 rows")
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"{SOURCES_CSV} must contain columns {sorted(required)}")
    source_ids = [row["source_id"].strip() for row in rows]
    if len(set(source_ids)) != len(source_ids):
        raise RuntimeError(f"{SOURCES_CSV} contains duplicate source_id values")
    for row in rows:
        relative = Path(row["gsvcities_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"GSV-Cities path must be relative and contained: {relative}")
        if relative.name != row["source_id"]:
            raise RuntimeError(f"source_id/path mismatch: {row['source_id']} != {relative}")
    return rows


def prepare_inputs(
    args: argparse.Namespace, samples: list[dict[str, str]]
) -> tuple[Path, Path | None]:
    staging = args.output / "demo_inputs"
    staging.mkdir(parents=True, exist_ok=True)
    wanted = [sample["source_id"] for sample in samples]
    for sample in samples:
        source = args.gsvcities_root / sample["gsvcities_path"]
        if not source.is_file():
            raise FileNotFoundError(source)
        link = staging / sample["source_id"]
        if link.is_symlink() and link.resolve() == source.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise RuntimeError(f"staged input conflicts with fixed CSV: {link}")
        link.symlink_to(source.resolve())

    subset = None
    if args.prompts_jsonl:
        if args.prompts_jsonl.resolve() == RELEASED_PROMPTS.resolve():
            metadata = json.loads(RELEASED_PROVENANCE.read_text())
            if hashlib.sha256(RELEASED_PROMPTS.read_bytes()).hexdigest() != metadata["sha256"]:
                raise RuntimeError("Bundled released demo prompts differ from their recorded SHA-256")
            shutil.copyfile(RELEASED_PROVENANCE, args.output / "demo_10_prompts.source.json")
        prompt_rows = load_jsonl(args.prompts_jsonl)
        if args.strict:
            selected_ids = [str(row.get("source_id")) for row in prompt_rows if str(row.get("source_id")) in wanted]
            if len(selected_ids) != len(set(selected_ids)):
                raise RuntimeError("Strict demo needs exactly one released prompt per source_id")
        by_source = {
            str(row.get("source_id")): row for row in prompt_rows
        }
        missing = [name for name in wanted if name not in by_source]
        if missing:
            raise RuntimeError(f"released Prompt JSONL is missing demo sources: {missing}")
        if args.strict:
            sys.path.insert(0, str(ROOT))
            from generation.inputs import normalize_frozen_prompt_entry

            selected = [by_source[name] for name in wanted]
            ids = [row.get("sample_id") for row in selected]
            if any(not name for name in ids) or len(set(ids)) != 10:
                raise RuntimeError("Strict demo needs 10 distinct published sample_id values")
            for row in selected:
                normalize_frozen_prompt_entry(row, args.gsvcities_root / "Images" / row["city"] / row["source_id"])
            if {row["route"] for row in selected} != REQUIRED_ROUTES:
                raise RuntimeError("Strict demo prompts must cover Global, Local, and Dual before generation")
        subset = args.output / "demo_10_prompts.jsonl"
        subset.write_text(
            "".join(
                json.dumps(by_source[name], ensure_ascii=False) + "\n" for name in wanted
            ),
            encoding="utf-8",
        )
    return staging, subset


def execute(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, env=os.environ.copy())
    if result.returncode:
        raise SystemExit(
            f"Demo subprocess exited with status {result.returncode}; "
            "see its error/summary above. Existing results are retained; use --resume after fixing the cause."
        )


def validate(run: Path, *, strict: bool = False) -> dict:
    records = load_jsonl(run / "records.jsonl")
    if len(records) != 10:
        raise RuntimeError(f"{run}: expected 10 records, got {len(records)}")
    sample_ids = [record.get("sample_id") for record in records]
    if any(not value for value in sample_ids) or len(set(sample_ids)) != 10:
        raise RuntimeError(f"{run}: expected 10 distinct sample IDs")
    generated, skipped = [], []
    for record in records:
        status = record.get("status")
        if status == "skipped" and record.get("route") == "skip" and not record.get("generated"):
            skipped.append(record)
        elif status in {"passed", "failed"} and record.get("generated") is True and record.get("route") in REQUIRED_ROUTES:
            output = Path(record.get("output_path") or "")
            if not output.is_file():
                raise RuntimeError(f"{run}: generated output is missing: {record['sample_id']} ({output})")
            generated.append(record)
        else:
            raise RuntimeError(f"{run}: incomplete/error record: {record.get('sample_id')} status={status!r}")
    errors = sorted((run / "errors").glob("*.json"))
    if errors:
        raise RuntimeError(f"{run}: unresolved errors: {[path.stem for path in errors]}")
    routes = {record.get("route") for record in generated}
    missing_routes = REQUIRED_ROUTES - routes
    depths = sorted(
        {max(0, int(record.get("rounds_used", 1)) - 1) for record in generated}
    )
    report = {
        "records": len(records), "generated": len(generated), "skipped": len(skipped),
        "passed": sum(record.get("passed") is True for record in generated),
        "observed_routes": sorted(routes), "uncovered_routes": sorted(missing_routes),
        "observed_reflection_depths": depths, "strict": strict,
    }
    print(f"{run}: {json.dumps(report, ensure_ascii=False)}", flush=True)
    if strict and (skipped or missing_routes):
        raise RuntimeError(
            f"{run}: strict coverage failed: generated={len(generated)}/10, "
            f"skipped={len(skipped)}, missing_routes={sorted(missing_routes)}"
        )
    return report


def run_one(args: argparse.Namespace, public_mode: str, input_path: Path) -> Path:
    run_mode = "plan" if public_mode == "qwen4b" and not args.strict else "prompt"
    reflection = args.reflection if args.max_reflections > 0 else "off"
    max_reflections = args.max_reflections if reflection == "on" else 0
    output_mode = "strict_released" if args.strict else public_mode
    output = args.output / f"{output_mode}_reflection_{reflection}"
    command = [
        sys.executable,
        str(ROOT / "run.py"),
        str(input_path),
        "--mode",
        run_mode,
        "--output",
        str(output),
        "--reflection",
        reflection,
        "--max-reflections",
        str(max_reflections),
        "--seed",
        str(args.seed),
        "--fail-fast",
    ]
    if args.strict:
        command.append("--require-generated")
        print("Strict demo: generating all 10 sources from fixed released prompts (planner Skip is bypassed).", flush=True)
    if run_mode == "prompt":
        command += ["--image-root", str(args.gsvcities_root / "Images")]
    if args.resume:
        command.append("--resume")
    execute(command)
    validate(output, strict=args.strict)
    return output


def main() -> None:
    args = parse_args()
    samples = load_sources()
    missing = [str(args.gsvcities_root / row["gsvcities_path"]) for row in samples
               if not (args.gsvcities_root / row["gsvcities_path"]).is_file()]
    if missing:
        raise SystemExit("Missing demo inputs:\n- " + "\n- ".join(missing))
    sys.path.insert(0, str(ROOT))
    from generation.preflight import check_environment

    needs_planner = (not args.strict and args.mode != "prompt") or (args.reflection == "on" and args.max_reflections > 0)
    errors = check_environment(planner=needs_planner)
    if errors:
        raise SystemExit(
            "Demo preflight failed (no inference started):\n- " + "\n- ".join(errors)
            + "\nFix the environment, then run bash scripts/start_generation_services.sh."
        )
    print("Demo preflight passed: 10 inputs, CUDA, dependencies, and services are ready.", flush=True)
    if args.check_only:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    staged, prompts = prepare_inputs(args, samples)
    completed = []
    if args.mode in {"qwen4b", "all"}:
        completed.append(run_one(args, "qwen4b", prompts if args.strict else staged))
    if args.mode in {"prompt", "all"}:
        completed.append(run_one(args, "prompt", prompts))
    print("Demo-10 validation passed:", *(str(path) for path in completed), sep="\n- ")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Demo failed: {exc}") from None
