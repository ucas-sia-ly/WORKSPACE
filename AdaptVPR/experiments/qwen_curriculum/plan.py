"""Freeze one Qwen weather-edit job per mined real training source.

The default night/snow/fog/rain weights are a proposed curriculum prior, not
GIFT's published distribution. Largest-remainder quotas and seeded label
shuffling preserve exact budgets without coupling miner order to domains.
Planning reads image bytes and metadata only; it never loads a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.qwen_curriculum import common
else:
    from . import common

common.use_adaptvpr()

from prompts import rules  # noqa: E402

DOMAINS = frozenset({"night", "snow", "fog", "rain"})
DEFAULT_DOMAIN_WEIGHTS = "night=0.5,snow=0.2,fog=0.2,rain=0.1"


def _fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _weights(value):
    if isinstance(value, str):
        parsed = {}
        for term in value.split(","):
            parts = term.split("=")
            if len(parts) != 2 or not parts[0].strip():
                raise ValueError("Domain weights must be comma-separated domain=positive_weight pairs")
            domain = parts[0].strip()
            if domain in parsed:
                raise ValueError(f"Duplicate domain weight: {domain}")
            try:
                parsed[domain] = float(parts[1])
            except ValueError as exc:
                raise ValueError(f"Invalid domain weight: {term}") from exc
        value = parsed
    if not isinstance(value, dict) or not value:
        raise ValueError("Domain weights must be a nonempty mapping")
    unknown = set(value) - DOMAINS
    if unknown:
        raise ValueError(f"Unsupported domain weights: {sorted(unknown)}; allowed: {sorted(DOMAINS)}")
    for domain, weight in value.items():
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"Domain weight for {domain} must be finite and positive")
    # Scaling first also supports very large finite user-supplied weights.
    scale = max(value.values())
    scaled = {domain: weight / scale for domain, weight in value.items()}
    total = math.fsum(scaled.values())
    return {domain: scaled[domain] / total for domain in sorted(scaled)}


def allocate_domains(n, weights, seed):
    """Return n shuffled labels with exact largest-remainder domain quotas.

    Equal fractional remainders break ties by domain name. Domain labels are
    shuffled independently of the fixed source ordering using ``seed``.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("Domain allocation count must be a nonnegative integer")
    weights = _weights(weights)
    expected = {domain: n * weight for domain, weight in weights.items()}
    quotas = {domain: math.floor(count) for domain, count in expected.items()}
    remainder = n - sum(quotas.values())
    for domain in sorted(weights, key=lambda key: (-(expected[key] - quotas[key]), key))[:remainder]:
        quotas[domain] += 1
    labels = [domain for domain in sorted(quotas) for _ in range(quotas[domain])]
    random.Random(seed).shuffle(labels)
    return labels


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--sources", type=Path, required=True, help="City-balanced hardness-mined real training sources JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=1000, help="One planned variant per distinct source")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--domain-weights", default=None,
                        help=f"Positive weights for night/snow/fog/rain; default proposed prior: {DEFAULT_DOMAIN_WEIGHTS}")
    parser.add_argument("--domain-stats", type=Path,
                        help="Optional JSON with domain_weights and provenance; supplies weights unless explicitly overridden")
    args = parser.parse_args(argv)
    if args.num_images <= 0:
        parser.error("--num-images must be positive")
    for field in ("sources", "output_dir", "domain_stats"):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, value.expanduser().resolve())
    return args


def _load_sources(manifest):
    sources, paths, hashes = [], {}, {}
    for line, original in enumerate(common.read_jsonl(manifest), start=1):
        location = f"{manifest}:row {line}"
        row = dict(original)
        if not isinstance(row.get("source_path"), str) or not row["source_path"].strip():
            raise ValueError(f"{location}: source_path must be a nonempty string")
        source = Path(row["source_path"]).expanduser()
        source = (source if source.is_absolute() else manifest.parent / source).resolve()
        expected_hash = row.get("source_sha256")
        if not isinstance(expected_hash, str) or not re.fullmatch("[0-9a-f]{64}", expected_hash):
            raise ValueError(f"{location}: source_sha256 must be a SHA-256 hex digest")
        if not isinstance(row.get("city"), str) or not row["city"].strip():
            raise ValueError(f"{location}: city must be a nonempty string")
        if type(row.get("place_id")) is not int or row["place_id"] < 0:
            raise ValueError(f"{location}: place_id must be a nonnegative integer")
        if "hardness" in row and (type(row["hardness"]) not in (int, float) or not math.isfinite(row["hardness"])):
            raise ValueError(f"{location}: hardness must be a finite number")
        identity = (row["city"], row["place_id"])
        if source in paths:
            raise ValueError(f"{location}: duplicate source path: {source}")
        actual_hash = common.file_sha256(source)
        if actual_hash != expected_hash or ("image_hash" in row and row["image_hash"] != actual_hash):
            raise ValueError(f"{location}: source image bytes do not match their recorded hash: {source}")
        if actual_hash in hashes:
            previous_path, previous_identity = hashes[actual_hash]
            if previous_identity != identity:
                raise ValueError(f"{location}: identical source image bytes have conflicting place labels: {previous_path}, {source}")
            raise ValueError(f"{location}: duplicate source image content: {previous_path}, {source}")
        with Image.open(source) as image:
            image.load()
            dimensions = list(image.size)
        if "source_dimensions" in row and row["source_dimensions"] != dimensions:
            raise ValueError(f"{location}: source_dimensions differ from the decoded source image")
        paths[source] = identity
        hashes[actual_hash] = (source, identity)
        sources.append({**row, "source_path": str(source), "source_sha256": actual_hash,
                        "image_hash": actual_hash, "source_dimensions": dimensions})
    return sources


def _plan_bytes(rows):
    return "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows).encode("utf-8")


def _select_sources(sources, count):
    # Preserve hardness order within each city while making a smaller requested
    # cohort balanced even when the miner manifest is grouped by city.
    cities = defaultdict(list)
    for source in sources:
        cities[source["city"]].append(source)
    selected, offset = [], 0
    while len(selected) < count:
        for city in sorted(cities):
            if offset < len(cities[city]):
                selected.append(cities[city][offset])
                if len(selected) == count:
                    break
        offset += 1
    return selected


def build_plan(args):
    """Return frozen configuration and sealed rows, without writing files."""
    manifest = Path(args.sources).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if type(args.num_images) is not int or args.num_images <= 0:
        raise ValueError("num_images must be a positive integer")
    stats_path = getattr(args, "domain_stats", None)
    stats, stats_hash = None, None
    if stats_path is not None:
        stats_path = Path(stats_path).expanduser().resolve()
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if not isinstance(stats, dict) or "domain_weights" not in stats:
            raise ValueError("Domain stats JSON must contain a domain_weights mapping")
        _weights(stats["domain_weights"])
        stats_hash = common.file_sha256(stats_path)
    explicit_weights = getattr(args, "domain_weights", None)
    requested_weights = (explicit_weights if explicit_weights is not None else
                         stats["domain_weights"] if stats is not None else DEFAULT_DOMAIN_WEIGHTS)
    weights = _weights(requested_weights)
    sources = _load_sources(manifest)
    if args.num_images > len(sources):
        raise ValueError(f"Requested {args.num_images} images but only {len(sources)} distinct sources are available")
    selected = _select_sources(sources, args.num_images)
    domains = allocate_domains(args.num_images, weights, args.seed)
    rows = []
    for source, domain in zip(selected, domains):
        prompt = rules.build_structured_prompt(route="global", weather=domain, occlusion=None)
        sample_id = "qwen_" + _fingerprint({
            "source_sha256": source["source_sha256"], "condition": domain,
            "prompt": prompt, "seed": args.seed,
        })[:24]
        mining_info = {key: value for key, value in source.items()
                       if key not in {"source_path", "source_sha256", "image_hash", "source_dimensions", "city", "place_id"}}
        row = {
            **source, "sample_id": sample_id, "candidate_index": 0, "route": "global",
            "domain": domain, "condition": domain, "prompt": prompt, "negative_prompt": "",
            "seed": common.candidate_seed(args.seed, sample_id, 0), "mining_info": mining_info,
        }
        row.pop("record_sha256", None)
        row["record_sha256"] = _fingerprint(row)
        rows.append(row)
    code_paths = [Path(__file__).resolve(), Path(common.__file__).resolve(), Path(rules.__file__).resolve()]
    config = {
        "schema_version": 1, "stage": "qwen_curriculum_plan",
        "sources": str(manifest), "sources_sha256": common.file_sha256(manifest),
        "output_dir": str(output_dir), "num_images": args.num_images, "seed": args.seed,
        "domain_weights": weights, "domain_quotas": dict(sorted(Counter(domains).items())),
        "domain_weight_origin": "explicit_user_weights" if explicit_weights is not None else
                                "domain_stats" if stats is not None else "proposed_prior_not_gift_exact",
        "default_prior": _weights(DEFAULT_DOMAIN_WEIGHTS),
        "prior_interpretation": "proposed night/snow/fog/rain prior; not GIFT's exact distribution",
        "domain_stats": str(stats_path) if stats_path is not None else None,
        "domain_stats_sha256": stats_hash, "domain_stats_provenance": stats.get("provenance") if stats else None,
        "source_pool": "mined_real_training_pool_no_evaluation_images",
        "source_selection": "city_round_robin_preserving_within_city_hardness_miner_order",
        "source_deduplication": "reject_duplicate_resolved_paths_or_image_content",
        "variants_per_source": 1, "source_manifest_rows": len(sources),
        "prompt_policy": "released_global_weather_templates", "prompt_policy_version": rules.PROMPT_POLICY_VERSION,
        "negative_prompt": "", "allocation": "largest_remainder_lexicographic_ties_seeded_domain_shuffle",
        "budget_unit": "planned_source_domain_jobs; execution may reuse exact cached jobs",
        "implementation_sha256": {str(path.relative_to(common.ADAPTVPR_ROOT)): common.file_sha256(path)
                                  for path in code_paths},
        "plan_sha256": hashlib.sha256(_plan_bytes(rows)).hexdigest(),
    }
    config["fingerprint"] = _fingerprint(config)
    return config, rows


def main(argv=None):
    args = parse_args(argv)
    config, rows = build_plan(args)
    config_path, plan_path = args.output_dir / "plan_config.json", args.output_dir / "plan.jsonl"
    if config_path.exists():
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(saved, dict) or saved.get("fingerprint") != _fingerprint({k: v for k, v in saved.items() if k != "fingerprint"}):
            raise ValueError("Saved plan configuration fingerprint is invalid")
        if saved != config:
            raise ValueError("Plan configuration changed; use a new --output-dir")
        if plan_path.exists():
            if common.file_sha256(plan_path) != config["plan_sha256"]:
                raise ValueError("Saved plan checksum differs; do not silently replace a frozen plan")
            print(f"[skip] validated frozen Qwen plan: {len(rows)} jobs", flush=True)
            return
    elif plan_path.exists():
        raise ValueError("Cannot reuse an existing plan without its frozen configuration")
    # All validation precedes publication. A missing plan after config creation
    # can be repaired using the same exact request and sealed deterministic rows.
    common.write_json(config_path, config)
    common.write_jsonl(plan_path, rows)
    print(f"[plan] {len(rows)} distinct sources; domain quotas={config['domain_quotas']}", flush=True)


if __name__ == "__main__":
    main()
