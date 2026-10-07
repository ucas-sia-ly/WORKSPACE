"""Extract hard cases from SALAD validation results.

This script parses SALAD evaluation outputs and creates a hard_cases.json
file suitable for generator fine-tuning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def extract_hard_cases_from_salad_eval(
    eval_results_path: Path,
    output_path: Path,
    min_rank: int = 5,
    max_cases: int | None = None,
) -> None:
    """Extract hard cases from SALAD evaluation results.

    Args:
        eval_results_path: Path to SALAD evaluation results (JSON)
        output_path: Path to save hard_cases.json
        min_rank: Minimum rank to consider as a hard case (default: 5)
        max_cases: Maximum number of cases to extract (None = all)
    """
    if not eval_results_path.exists():
        raise FileNotFoundError(f"Evaluation results not found: {eval_results_path}")

    print(f"Loading evaluation results from {eval_results_path}")
    data = json.loads(eval_results_path.read_text(encoding="utf-8"))

    # Expected format from SALAD evaluation:
    # {
    #   "dataset": "SVOX",
    #   "recall": {"R@1": 0.85, "R@5": 0.92, ...},
    #   "error_queries": [
    #     {
    #       "query_id": "...",
    #       "query_path": "...",
    #       "ground_truth": "...",
    #       "predicted": "...",
    #       "rank": 15,
    #       "distance_pred": 0.12,
    #       "distance_gt": 0.45,
    #     },
    #     ...
    #   ]
    # }

    error_queries = data.get("error_queries", [])
    print(f"Found {len(error_queries)} error queries")

    # Filter by rank
    hard_cases = [
        query for query in error_queries
        if query.get("rank", 0) >= min_rank
    ]
    print(f"Filtered to {len(hard_cases)} hard cases (rank >= {min_rank})")

    # Sort by rank (descending) - hardest cases first
    hard_cases.sort(key=lambda x: x.get("rank", 0), reverse=True)

    # Limit number of cases
    if max_cases is not None and len(hard_cases) > max_cases:
        hard_cases = hard_cases[:max_cases]
        print(f"Limited to {max_cases} hardest cases")

    # Convert to output format
    output_cases = []
    for query in hard_cases:
        case = {
            "query_id": query["query_id"],
            "query_path": query["query_path"],
            "correct_match_id": query["ground_truth"],
            "wrong_match_id": query["predicted"],
            "retrieval_rank": query["rank"],
            "distance_to_wrong": query.get("distance_pred", 0.0),
            "distance_to_correct": query.get("distance_gt", 1.0),
        }

        # Benchmark query IDs do not identify GSV-Cities source images. Only
        # preserve a mapping explicitly supplied by the evaluation manifest.
        case["source_id"] = query.get("source_id")

        output_cases.append(case)

    # Save output
    output_data = {
        "source": str(eval_results_path),
        "dataset": data.get("dataset", "unknown"),
        "min_rank_threshold": min_rank,
        "total_error_queries": len(error_queries),
        "hard_cases_extracted": len(output_cases),
        "error_cases": output_cases,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(output_cases)} hard cases to {output_path}")

    # Print statistics
    ranks = [c["retrieval_rank"] for c in output_cases]
    if ranks:
        print(f"\nStatistics:")
        print(f"  Min rank: {min(ranks)}")
        print(f"  Max rank: {max(ranks)}")
        print(f"  Mean rank: {sum(ranks) / len(ranks):.1f}")
        print(f"  Median rank: {sorted(ranks)[len(ranks) // 2]}")


def merge_hard_cases(
    input_paths: list[Path],
    output_path: Path,
    max_cases: int | None = None,
) -> None:
    """Merge hard cases from multiple datasets.

    Args:
        input_paths: List of hard_cases.json files
        output_path: Path to save merged hard_cases.json
        max_cases: Maximum total cases (None = all)
    """
    all_cases = []
    sources = []

    for path in input_paths:
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue

        data = json.loads(path.read_text(encoding="utf-8"))
        cases = data.get("error_cases", [])
        all_cases.extend(cases)
        sources.append({
            "path": str(path),
            "dataset": data.get("dataset", "unknown"),
            "count": len(cases),
        })
        print(f"Loaded {len(cases)} cases from {path}")

    # Remove duplicates by query_id
    seen = set()
    unique_cases = []
    for case in all_cases:
        query_id = case["query_id"]
        if query_id not in seen:
            seen.add(query_id)
            unique_cases.append(case)

    print(f"\nTotal cases: {len(all_cases)}")
    print(f"Unique cases: {len(unique_cases)}")

    # Sort by rank (descending)
    unique_cases.sort(key=lambda x: x.get("retrieval_rank", 0), reverse=True)

    # Limit
    if max_cases is not None and len(unique_cases) > max_cases:
        unique_cases = unique_cases[:max_cases]
        print(f"Limited to {max_cases} hardest cases")

    # Save
    output_data = {
        "sources": sources,
        "total_cases": len(all_cases),
        "unique_cases": len(unique_cases),
        "error_cases": unique_cases,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(unique_cases)} merged hard cases to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract hard cases from SALAD evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract from single evaluation")
    extract_parser.add_argument("eval_results", type=Path, help="SALAD evaluation results JSON")
    extract_parser.add_argument("--output", type=Path, required=True, help="Output hard_cases.json")
    extract_parser.add_argument("--min-rank", type=int, default=5, help="Minimum rank to consider hard")
    extract_parser.add_argument("--max-cases", type=int, help="Maximum cases to extract")

    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge multiple hard_cases.json files")
    merge_parser.add_argument("inputs", type=Path, nargs="+", help="Input hard_cases.json files")
    merge_parser.add_argument("--output", type=Path, required=True, help="Output merged hard_cases.json")
    merge_parser.add_argument("--max-cases", type=int, help="Maximum total cases")

    args = parser.parse_args()

    if args.command == "extract":
        extract_hard_cases_from_salad_eval(
            args.eval_results,
            args.output,
            min_rank=args.min_rank,
            max_cases=args.max_cases,
        )
    elif args.command == "merge":
        merge_hard_cases(
            args.inputs,
            args.output,
            max_cases=args.max_cases,
        )


if __name__ == "__main__":
    main()
