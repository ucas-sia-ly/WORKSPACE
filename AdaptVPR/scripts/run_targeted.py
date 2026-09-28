"""Independent Stage3 targeted input validation and dry-run task export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generation.targeted_inputs import run_targeted


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Bag-of-Queries schema 1 targets.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/stage3_targeted",
                        help="Directory for dry-run tasks.jsonl")
    parser.add_argument("--check-only", action="store_true", help="Validate and decode only; do not write tasks")
    parser.add_argument("--limit", type=int, default=0, help="Select first N records; 0 means all")
    parser.add_argument("--seed", type=int, default=0, help="Task seed; original Stage2 seed is preserved separately")
    parser.add_argument("--resume", action="store_true", help="Resume an identical validated task prefix")
    args = parser.parse_args(argv)
    if args.limit < 0 or args.seed < 0:
        parser.error("--limit and --seed must be nonnegative")
    try:
        result = run_targeted(args.input, args.output, check_only=args.check_only,
                              limit=args.limit, seed=args.seed, resume=args.resume)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Targeted input validation failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
