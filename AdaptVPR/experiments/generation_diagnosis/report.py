"""Aggregate generation diagnosis without loading generation or verification models.

Example (from the workspace root)::

    python AdaptVPR/experiments/generation_diagnosis/report.py \
        --run-dir outputs/gen_diagnosis/prompt_ablation/iclight \
        --run-dir outputs/gen_diagnosis/prompt_ablation/qwen

The old grid uses JPEG evidence and one seed per source/condition. PNG sheets
avoid another lossy encode; resizing for display does not restore JPEG detail.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


WORKSPACE = Path(__file__).resolve().parents[3]
CONDITIONS = ("overcast", "rain", "snow", "fog", "night")
LEGACY_THRESHOLDS = {"s_geo": 0.78, "s_div": 0.15}
MAIN_METHODS = (
    ("source", "Source"),
    ("iclight_released_released", "IC-Light: released / noise"),
    ("iclight_released_sdedit_0.85", "IC-Light: released / SDEdit .85"),
    ("iclight_positive_released", "IC-Light: positive / noise"),
    ("iclight_positive_sdedit_0.85", "IC-Light: positive / SDEdit .85"),
    ("qwen_released", "Qwen: released"),
    ("qwen_positive", "Qwen: positive"),
)
COMPARISONS = (
    ("iclight_legacy_released", "iclight_legacy_released_cfg3", "legacy_sampling"),
    ("iclight_legacy_released", "iclight_legacy_sdedit_0.85", "legacy_sampling"),
    ("iclight_legacy_released", "iclight_legacy_sdedit_0.70", "legacy_sampling"),
    ("iclight_legacy_released", "iclight_legacy_sdedit_0.55", "legacy_sampling"),
    ("iclight_legacy_released", "iclight_released_released", "serialization_and_rerun"),
    ("iclight_legacy_sdedit_0.85", "iclight_released_sdedit_0.85", "serialization_and_rerun"),
    ("iclight_released_released", "iclight_released_released_cfg3", "sampling"),
    ("iclight_released_released", "iclight_released_sdedit_0.85", "sampling"),
    ("iclight_released_released", "iclight_released_sdedit_0.70", "sampling"),
    ("iclight_released_released", "iclight_released_sdedit_0.55", "sampling"),
    ("iclight_released_released", "iclight_no_negations_released", "prompt"),
    ("iclight_released_released", "iclight_positive_released", "prompt"),
    ("iclight_released_sdedit_0.85", "iclight_no_negations_sdedit_0.85", "prompt"),
    ("iclight_released_sdedit_0.85", "iclight_positive_sdedit_0.85", "prompt"),
    ("iclight_positive_released", "iclight_positive_sdedit_0.85", "sampling"),
    ("qwen_released", "qwen_no_negations", "prompt"),
    ("qwen_released", "qwen_positive", "prompt"),
    ("iclight_released_released", "qwen_released", "model"),
    ("iclight_positive_sdedit_0.85", "qwen_positive", "model_and_sampling"),
)


def jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: expected a JSON object")
        rows.append(row)
    return rows


def deterministic_seed(source: str, condition: str) -> int:
    material = f"42|{source}|{condition}|0".encode("utf-8")
    return 1 + int(hashlib.sha256(material).hexdigest()[:8], 16) % 2_000_000_000


def finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def location(value: Any, parent: Path) -> str | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        candidates = (parent / path, WORKSPACE / path)
        path = next((p for p in candidates if p.exists()), candidates[0])
    return str(path.resolve())


def normalize(row: dict[str, Any], parent: Path, sources: list[str], legacy: bool,
              default_thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    src = row.get("src", row.get("source_index"))
    source = row.get("source_path")
    if source is None and src is not None and 0 <= int(src) < len(sources):
        source = sources[int(src)]
    source = location(source, parent)
    condition = str(row.get("cond", row.get("condition", row.get("weather", "unknown"))))
    variant = str(row.get("prompt_variant", "released"))
    strategy = str(row.get("strat", row.get("strategy", "released")))
    mode = "iclight" if legacy else str(row.get("mode", ""))
    if not mode:
        mode = "qwen" if "qwen" in str(row.get("model", "")).lower() else "iclight"
    method = f"iclight_legacy_{strategy}" if legacy else str(row.get("method") or (
        f"qwen_{variant}" if mode == "qwen" else f"iclight_{variant}_{strategy}"))
    metrics = row.get("metrics", {})
    geo = finite(row.get("s_geo", metrics.get("s_geo")))
    div = finite(row.get("s_div", metrics.get("s_div")))
    thresholds = row.get("thresholds") or default_thresholds or {}
    tau_geo = finite(thresholds.get("s_geo", thresholds.get("tau_geo", thresholds.get("TAU_GEO"))))
    tau_div = finite(thresholds.get("s_div", thresholds.get("tau_div", thresholds.get("TAU_DIV"))))
    tau_geo = LEGACY_THRESHOLDS["s_geo"] if tau_geo is None else tau_geo
    tau_div = LEGACY_THRESHOLDS["s_div"] if tau_div is None else tau_div
    status = str(row.get("status", "ok"))
    valid = status == "ok" and geo is not None and div is not None
    geo_ok = bool(row.get("geo_ok", geo is not None and geo >= tau_geo))
    div_ok = bool(row.get("div_ok", div is not None and div >= tau_div))
    passed = bool(row.get("passed", geo_ok and div_ok)) if valid else False
    if status != "ok":
        failure = "error"
    elif not valid:
        failure = "invalid_metrics"
    elif passed:
        failure = "pass"
    elif not geo_ok and not div_ok:
        failure = "geometry_and_diversity"
    elif not geo_ok:
        failure = "geometry_only"
    elif not div_ok:
        failure = "diversity_only"
    else:
        failure = "inconsistent_pass_flag"
    seed = row.get("seed")
    if legacy and source:
        seed = deterministic_seed(source, condition)
    output = row.get("output_path", row.get("image_path", row.get("result_path")))
    if legacy:
        output = parent / f"s{src}_{condition}_{strategy}.jpg"
    return {
        "method": method, "mode": mode, "src": src, "source_path": source,
        "source_png_path": location(row.get("source_png_path"), parent),
        "cond": condition, "seed": seed, "s_geo": geo, "s_div": div,
        "tau_geo": tau_geo, "tau_div": tau_div, "geo_ok": geo_ok,
        "div_ok": div_ok, "passed": passed, "valid_scored": valid,
        "failure": failure, "status": status, "error": row.get("error"),
        "output_path": location(output, parent), "legacy_jpeg": legacy,
        "raw_output_path": location(row.get("raw_output_path"), parent),
        "raw_dimensions": row.get("raw_dimensions"), "source_dimensions": row.get("source_dimensions"),
        "resize_applied": row.get("resize_applied"), "resize_method": row.get("resize_method"),
        "origin": str(parent), "config_fingerprint": row.get("config_fingerprint"),
        "threshold_mismatch": bool(valid and (geo_ok != (geo >= tau_geo) or div_ok != (div >= tau_div))),
        "pass_flag_mismatch": bool(valid and passed != (geo_ok and div_ok)),
        "prompt": row.get("prompt"), "negative_prompt": row.get("negative_prompt"),
        "sampling": row.get("sampling"), "elapsed_seconds": row.get("elapsed_seconds"),
    }


def paired_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["source_path"] or f"index:{row['src']}", row["cond"], row["seed"])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in rows if r["valid_scored"]]
    passed = sum(r["passed"] for r in scored)
    return {
        "attempted": len(rows), "scored": len(scored), "passed": passed,
        "pass_rate_scored": passed / len(scored) if scored else None,
        "pass_rate_attempted": passed / len(rows) if rows else None,
        "geo_ok": sum(r["geo_ok"] for r in scored),
        "div_ok": sum(r["div_ok"] for r in scored),
        "mean_geo": statistics.mean(r["s_geo"] for r in scored) if scored else None,
        "median_geo": statistics.median(r["s_geo"] for r in scored) if scored else None,
        "mean_div": statistics.mean(r["s_div"] for r in scored) if scored else None,
        "median_div": statistics.median(r["s_div"] for r in scored) if scored else None,
        **{name: sum(r["failure"] == name for r in rows) for name in (
            "geometry_only", "diversity_only", "geometry_and_diversity", "error",
            "invalid_metrics", "inconsistent_pass_flag")},
    }


def aggregates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["cond"])].append(row)
        grouped[(row["method"], "all")].append(row)
    return [{"method": method, "condition": condition, **summarize(group)}
            for (method, condition), group in sorted(grouped.items())]


def paired_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = defaultdict(dict)
    for row in rows:
        methods[row["method"]][paired_key(row)] = row
    output = []
    for baseline, treatment, intervention in COMPARISONS:
        if not methods[baseline] or not methods[treatment]:
            continue
        all_keys = methods[baseline].keys() & methods[treatment].keys()
        for condition in ("all", *CONDITIONS):
            keys = [k for k in all_keys if condition == "all" or k[1] == condition]
            scored = [(methods[baseline][k], methods[treatment][k]) for k in keys
                      if methods[baseline][k]["valid_scored"] and methods[treatment][k]["valid_scored"]]
            if not keys:
                continue
            count = len(scored)
            changes_geo = [b["s_geo"] - a["s_geo"] for a, b in scored]
            changes_div = [b["s_div"] - a["s_div"] for a, b in scored]
            output.append({
                "baseline": baseline, "treatment": treatment, "intervention": intervention,
                "condition": condition, "matched_attempts": len(keys), "scored_pairs": count,
                "unmatched_baseline": sum(condition == "all" or k[1] == condition
                                          for k in methods[baseline].keys() - methods[treatment].keys()),
                "unmatched_treatment": sum(condition == "all" or k[1] == condition
                                           for k in methods[treatment].keys() - methods[baseline].keys()),
                "mean_delta_geo": statistics.mean(changes_geo) if count else None,
                "median_delta_geo": statistics.median(changes_geo) if count else None,
                "mean_delta_div": statistics.mean(changes_div) if count else None,
                "median_delta_div": statistics.median(changes_div) if count else None,
                "baseline_passed": sum(a["passed"] for a, b in scored),
                "treatment_passed": sum(b["passed"] for a, b in scored),
                "delta_pass_rate": sum(int(b["passed"]) - int(a["passed"]) for a, b in scored) / count if count else None,
                "fail_to_pass": sum(not a["passed"] and b["passed"] for a, b in scored),
                "pass_to_fail": sum(a["passed"] and not b["passed"] for a, b in scored),
            })
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def planned_scopes(run_configs: list[dict[str, Any]], sources: list[str], legacy: bool) -> dict[str, set[tuple[str, str]]]:
    """Read scheduled cells from frozen configs rather than the gallery cohort."""
    scopes: dict[str, set[tuple[str, str]]] = defaultdict(set)
    if legacy:
        for strategy in ("released", "released_cfg3", "sdedit_0.85", "sdedit_0.70", "sdedit_0.55"):
            scopes[f"iclight_legacy_{strategy}"].update((source, condition) for source in sources for condition in CONDITIONS)
    for record in run_configs:
        config = record["config"]
        parent = Path(record["path"]).parent
        planned_sources = [location(row.get("source_path") if isinstance(row, dict) else row, parent)
                           for row in config.get("sources", [])]
        cells = {(source, str(condition)) for source in planned_sources if source
                 for condition in config.get("conditions", [])}
        for variant in config.get("prompt_variants", []):
            if config.get("mode") == "qwen":
                scopes[f"qwen_{variant}"].update(cells)
            else:
                for strategy in config.get("strategies", []):
                    scopes[f"iclight_{variant}_{strategy}"].update(cells)
    return dict(scopes)


def missing_cell_label(method: str, source: str, condition: str,
                       scopes: dict[str, set[tuple[str, str]]]) -> str:
    if method not in scopes:
        return "NO RECORDED PLAN"
    if (source, condition) in scopes[method]:
        return "PENDING / NOT RUN"
    if method.startswith("qwen_") and source not in {s for s, c in scopes[method]}:
        return "Outside Qwen cohort / NOT SCHEDULED"
    return "NOT SCHEDULED"


def coverage(rows: list[dict[str, Any]], scopes: dict[str, set[tuple[str, str]]]) -> list[dict[str, Any]]:
    methods = [method for method, _ in MAIN_METHODS if method != "source"]
    methods += sorted((set(scopes) | {r["method"] for r in rows}) - set(methods))
    output = []
    for method in methods:
        group = [r for r in rows if r["method"] == method]
        observed = {(r["source_path"], r["cond"]) for r in group}
        cells = scopes.get(method)
        output.append({
            "method": method, "expected": len(cells) if cells is not None else None,
            "recorded": len(group), "planned_source_count": len({s for s, c in cells}) if cells is not None else None,
            "planned_conditions": sorted({c for s, c in cells}) if cells is not None else None,
            "pending_cells": len(cells - observed) if cells is not None else None,
            "recorded_outside_plan": len(observed - cells) if cells is not None else None,
        })
    return output


def score_label(row: dict[str, Any] | None, compact: bool = False,
                missing_label: str = "PENDING / NOT RUN") -> str:
    if row is None:
        return missing_label
    if not row["valid_scored"]:
        return f"{row['failure'].upper()}: {str(row['error'] or '')[:55]}"
    failure = row["failure"]
    if compact:
        failure = {"geometry_only": "geo fail", "diversity_only": "div fail",
                   "geometry_and_diversity": "geo+div fail"}.get(failure, failure)
    return f"geo {row['s_geo']:.3f} | div {row['s_div']:.3f} | {failure}"


def sheet_row(lookup: dict[tuple, dict], source: str, condition: str, method: str) -> dict | None:
    """Prefer fresh controls; an explicit footer identifies legacy fallback."""
    row = lookup.get((source, condition, method))
    if row is None and method in {"iclight_released_released", "iclight_released_sdedit_0.85"}:
        fallback = method.replace("iclight_released_", "iclight_legacy_", 1)
        row = lookup.get((source, condition, fallback))
    return row


def make_sheets(output: Path, rows: list[dict[str, Any]], sources: list[str], width: int,
                scopes: dict[str, set[tuple[str, str]]]) -> list[str]:
    lookup = {(r["source_path"], r["cond"], r["method"]): r for r in rows}
    image_height, label_height, header_height = width * 3 // 4, 46, 64
    row_height = image_height + label_height
    files = []
    for condition in CONDITIONS:
        sheet = Image.new("RGB", (len(MAIN_METHODS) * width, header_height + len(sources) * row_height), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        for column, (method, label) in enumerate(MAIN_METHODS):
            x = column * width
            draw.text((x + 7, 5), condition.upper(), font=font(17), fill="#101010")
            # Two lines prevent labels from crossing the next column.
            head = label.replace(": ", ":\n")
            draw.multiline_text((x + 7, 28), head, font=font(12), fill="#202020", spacing=2)
            for index, source in enumerate(sources):
                y = header_height + index * row_height
                row = None if method == "source" else sheet_row(lookup, source, condition, method)
                missing_label = missing_cell_label(method, source, condition, scopes)
                path = source if method == "source" else row["output_path"] if row else None
                if path and Path(path).is_file():
                    with Image.open(path) as original:
                        picture = ImageOps.contain(original.convert("RGB"), (width - 2, image_height), Image.Resampling.LANCZOS)
                    sheet.paste(picture, (x + (width - picture.width) // 2, y + (image_height - picture.height) // 2))
                else:
                    image_label = "Outside Qwen cohort" if missing_label.startswith("Outside Qwen") else "Missing image"
                    draw.text((x + 14, y + image_height // 2), image_label, font=font(15), fill="#777777")
                color = "#245b39" if row and row["passed"] else "#713329" if row and row["valid_scored"] else "#444444"
                draw.rectangle((x, y + image_height, x + width - 1, y + row_height - 1), fill=color)
                first = f"source {index}" if method == "source" else f"source {index} | {condition}"
                if row:
                    first += " | legacy JPEG95" if row["legacy_jpeg"] else " | fresh PNG"
                second = "Original input" if method == "source" else score_label(row, compact=True, missing_label=missing_label)
                draw.text((x + 6, y + image_height + 4), first, font=font(13), fill="white")
                draw.text((x + 6, y + image_height + 23), second[:width // 7], font=font(11), fill="white")
        name = f"comparison_{condition}.png"
        sheet.save(output / name, format="PNG")
        files.append(name)
    return files


def html_gallery(output: Path, rows: list[dict[str, Any]], sources: list[str], summary: dict[str, Any],
                 scopes: dict[str, set[tuple[str, str]]]) -> None:
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    copied: dict[str, str] = {}

    def asset(path: str | None) -> str | None:
        if not path or not Path(path).is_file():
            return None
        if path not in copied:
            source = Path(path)
            target = hashlib.sha256(path.encode()).hexdigest()[:16] + source.suffix.lower()
            shutil.copyfile(source, assets / target)
            copied[path] = f"assets/{target}"
        return copied[path]

    def img(path: str | None, alt: str, missing_label: str = "Missing image") -> str:
        url = asset(path)
        if not url:
            return f'<div class="missing">{html.escape(missing_label)}</div>'
        return f'<a href="{url}" target="_blank"><img loading="lazy" src="{url}" alt="{html.escape(alt)}"></a>'

    methods = [key for key, _ in MAIN_METHODS if key != "source"]
    methods += sorted({r["method"] for r in rows} - set(methods))
    lookup = {(r["source_path"], r["cond"], r["method"]): r for r in rows}
    body = ["<h1>Generation diagnosis evidence</h1>",
            "<p>Measured: geometric matcher inlier ratio and CLIP image distance. "
            "Weather target correctness is <strong>not measured</strong>. Click an image for its original saved file.</p>",
            "<p>" + " · ".join(f'<a href="{c}">{c}</a>' for c in ("summary.json", "aggregates.csv", "paired_deltas.csv", "rows.csv")) + "</p>",
            "<details open><summary>Scope and interpretation</summary><ul>" + "".join(
                f"<li>{html.escape(note)}</li>" for note in summary["limitations"]) + "</ul></details>",
            "<nav>" + " · ".join(f'<a href="#{c}">{c}</a>' for c in CONDITIONS) + "</nav>"]
    all_stats = [r for r in summary["aggregates"] if r["condition"] == "all"]
    plans = {r["method"]: r for r in summary["coverage"]}
    body.append("<h2>All conditions</h2><table><tr><th>Method</th><th>Recorded / planned</th><th>Pending</th><th>Pass / scored</th><th>Median geo</th><th>Median div</th><th>Geo only</th><th>Div only</th><th>Both</th><th>Errors</th></tr>")
    for row in all_stats:
        plan = plans[row["method"]]
        body.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in (
            row["method"], f"{plan['recorded']} / {plan['expected'] if plan['expected'] is not None else 'unknown'}",
            plan["pending_cells"] if plan["pending_cells"] is not None else "unknown", f"{row['passed']} / {row['scored']}",
            f"{row['median_geo']:.3f}" if row["median_geo"] is not None else "—",
            f"{row['median_div']:.3f}" if row["median_div"] is not None else "—",
            row["geometry_only"], row["diversity_only"], row["geometry_and_diversity"], row["error"])) + "</tr>")
    body.append("</table>")
    for condition in CONDITIONS:
        body.append(f'<section id="{condition}"><h2>{condition}</h2>')
        if f"comparison_{condition}.png" in summary["contact_sheets"]:
            body.append(f'<p><a href="comparison_{condition}.png">PNG comparison sheet: eight sources × seven methods</a></p>')
        for index, source in enumerate(sources):
            body.append(f'<h3>Source {index} · {html.escape(Path(source).name)}</h3><div class="grid">')
            body.append(f'<article><h4>Source</h4>{img(source, "source")}<p>Original input</p></article>')
            for method in methods:
                row = lookup.get((source, condition, method))
                if row is None and method not in {m for m, _ in MAIN_METHODS}:
                    continue
                missing_label = missing_cell_label(method, source, condition, scopes)
                body.append(f'<article><h4>{html.escape(method)}</h4>{img(row["output_path"] if row else None, method, missing_label if row is None else "Missing image")}')
                body.append(f'<p class="{"pass" if row and row["passed"] else "fail"}">{html.escape(score_label(row, missing_label=missing_label))}</p>')
                if row:
                    if row["raw_output_path"]:
                        raw_url = asset(row["raw_output_path"])
                        if raw_url:
                            body.append(f'<p><a href="{raw_url}" target="_blank">Raw Qwen output</a> · raw {html.escape(str(row["raw_dimensions"]))} → evaluated {html.escape(str(row["source_dimensions"]))}</p>')
                    details = {k: row[k] for k in ("seed", "legacy_jpeg", "origin", "config_fingerprint", "sampling", "prompt", "negative_prompt", "resize_applied", "resize_method", "error")}
                    body.append(f'<details><summary>Run details</summary><pre>{html.escape(json.dumps(details, ensure_ascii=False, indent=2))}</pre></details>')
                body.append("</article>")
            body.append("</div>")
        body.append("</section>")
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Generation diagnosis</title><style>body{font:15px system-ui,sans-serif;background:#f3f4f6;color:#17202a;margin:24px}a{color:#174e9c}nav{position:sticky;top:0;background:#fff;padding:12px;z-index:1}table{border-collapse:collapse;background:#fff}th,td{border:1px solid #ddd;padding:8px;text-align:left}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}article{background:#fff;padding:12px;border-radius:8px}h4{font-size:13px;overflow-wrap:anywhere}img{width:100%;height:230px;object-fit:contain}.missing{height:230px;display:grid;place-items:center;background:#eee;color:#666}pre{font-size:11px;white-space:pre-wrap;overflow-wrap:anywhere}.pass{color:#205732}.fail{color:#9b3327}section{scroll-margin-top:60px}li{margin-bottom:6px}</style><body>' + "\n".join(body) + "</body></html>"
    (output / "index.html").write_text(document, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-dir", type=Path, default=WORKSPACE / "outputs/gen_diagnosis/grid")
    parser.add_argument("--source-jsonl", type=Path, default=WORKSPACE / "outputs/vpr_guidance_smoke/cand_probe/candidates.jsonl")
    parser.add_argument("--run-dir", type=Path, action="append", default=[], help="Directory recursively containing results.jsonl; may be repeated.")
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE / "outputs/gen_diagnosis/report")
    parser.add_argument("--cell-width", type=int, default=320)
    parser.add_argument("--no-sheets", action="store_true")
    parser.add_argument("--no-gallery", action="store_true")
    args = parser.parse_args(argv)
    if args.cell_width < 200:
        parser.error("--cell-width must be at least 200")
    sources = sorted({r["source_path"] for r in jsonl(args.source_jsonl)})[::5][:8]
    sources = [str(Path(p).resolve()) for p in sources]
    rows = []
    inputs = []
    run_configs = []
    loaded_configs: dict[str, dict[str, Any]] = {}
    old = args.grid_dir / "results.json"
    if old.is_file():
        rows.extend(normalize(r, old.parent.resolve(), sources, True) for r in json.loads(old.read_text()))
        inputs.append(str(old.resolve()))
    for directory in args.run_dir:
        config_paths = [directory.parent / "generation_config.json"] if directory.is_file() else sorted(directory.rglob("generation_config.json"))
        for config_path in config_paths:
            key = str(config_path.resolve())
            if config_path.is_file() and key not in loaded_configs:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                loaded_configs[key] = config
                run_configs.append({"path": key, "config": config})
        files = [directory] if directory.is_file() else sorted(directory.rglob("results.jsonl"))
        for path in files:
            if str(path.resolve()) in inputs:
                continue
            config_path = path.parent / "generation_config.json"
            config = loaded_configs.get(str(config_path.resolve()), {})
            thresholds = config.get("verification", {}).get("thresholds")
            rows.extend(normalize(r, path.parent.resolve(), sources, False, thresholds) for r in jsonl(path))
            inputs.append(str(path.resolve()))
    deduplicated = {}
    duplicates = 0
    for row in rows:
        key = (row["method"], *paired_key(row))
        if key in deduplicated:
            duplicates += 1
        deduplicated[key] = row
    rows = list(deduplicated.values())
    rows.sort(key=lambda r: (r["method"], r["cond"], str(r["src"]), str(r["seed"])))
    scopes = planned_scopes(run_configs, sources, old.is_file())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "inputs": inputs, "source_selection": "sorted unique candidate source_path [::5][:8]",
        "sources": sources, "cities": sorted({Path(p).parent.name for p in sources}),
        "conditions": list(CONDITIONS), "default_global_thresholds": LEGACY_THRESHOLDS,
        "weather_target_measured": False, "duplicate_rows_replaced_by_last": duplicates,
        "threshold_mismatch_count": sum(r["threshold_mismatch"] for r in rows),
        "pass_flag_mismatch_count": sum(r["pass_flag_mismatch"] for r in rows),
        "run_configs": run_configs,
        "planned_scopes": {method: [{"source_path": source, "condition": condition}
                                    for source, condition in sorted(cells)]
                           for method, cells in sorted(scopes.items())},
        "limitations": [
            "Eight Bangkok source images, selected by sorted source_path [::5][:8], are a small diagnostic sample; findings do not establish city-wide or cross-city performance.",
            "One deterministic seed per source/condition, reused across methods. Forty cells share only eight independent sources; report descriptive paired changes, not statistical significance.",
            "Geometry inlier ratio is not a guarantee that every building, vehicle or road feature is unchanged. CLIP image distance does not measure whether the requested weather was achieved.",
            "Legacy JPEG95 results remain distinct from fresh PNG controls. Primary prompt/sampling pairs use fresh controls; legacy-vs-fresh pairs include serialization and rerun differences and do not isolate compression alone.",
            "PNG contact sheets use aspect-preserving display resizing and no additional lossy encode. Evidence assets copy the original bytes; existing JPEG losses cannot be recovered.",
            "Global thresholds default to geo >= 0.78 and div >= 0.15 unless recorded per row or in generation_config.json. Scores and flags are reported from the run; threshold and pass-flag mismatches are counted.",
            "Pairs require identical source path, condition and recorded seed. A shared numeric seed across different model families does not imply shared latent noise or an isolated model effect.",
            "Qwen raw PNGs are retained beside evaluated PNGs normalized by LANCZOS to source dimensions, matching the production wrapper. Aspect-ratio changes or normalization can affect geometry; inspect both versions.",
            "Main sheets prefer fresh released controls and use explicitly labeled legacy JPEG95 fallback only when a fresh control is absent. The gallery retains every recorded method, including no_negations and all legacy strengths.",
            "Coverage uses each method's frozen source/condition plan. Unfinished cells inside the plan are pending; sources outside the selected Qwen cohort are not scheduled. Generation errors are excluded from scored denominators and retained in attempted denominators; duplicate method/source/condition/seed rows use the last supplied record.",
        ],
        "aggregates": aggregates(rows), "paired_deltas": paired_deltas(rows),
        "coverage": coverage(rows, scopes),
        "contact_sheets": [],
    }
    if not args.no_sheets:
        summary["contact_sheets"] = make_sheets(args.output_dir, rows, sources, args.cell_width, scopes)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "aggregates.csv", summary["aggregates"])
    write_csv(args.output_dir / "paired_deltas.csv", summary["paired_deltas"])
    write_csv(args.output_dir / "rows.csv", rows)
    if not args.no_gallery:
        html_gallery(args.output_dir, rows, sources, summary, scopes)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "records": len(rows),
                      "inputs": len(inputs), "contact_sheets": len(summary["contact_sheets"]),
                      "gallery": not args.no_gallery}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
