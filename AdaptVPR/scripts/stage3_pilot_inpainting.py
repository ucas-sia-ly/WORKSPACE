"""Prepare/run 10--20 dev edits using an explicitly local trained inpainting model."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generation.inpainting_pilot import DEV, prepare_pilot, run_prepared


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=ROOT/"outputs/stage3_dev/candidate_masks")
    parser.add_argument("--dev-dir", type=Path, default=DEV)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/stage3_dev/inpainting_pilot")
    parser.add_argument("--count", type=int, default=10, help="Default 10; never silently pad an insufficient cohort")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selection-mode", choices=("frozen", "render_semantics"), default="frozen",
                        help="render_semantics explicitly drops only the weighted-coverage gate for a rendering pilot")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="Save validated inputs without loading any model")
    mode.add_argument("--run-prepared", type=Path, help="Generate exactly the already-saved pilot plan")
    args = parser.parse_args(argv)
    try:
        if args.run_prepared:
            result = run_prepared(args.run_prepared)
        else:
            result = prepare_pilot(candidate_dir=args.candidate_dir, output=args.output, dev_dir=args.dev_dir,
                                   count=args.count, seed=args.seed, selection_mode=args.selection_mode)
            if not args.prepare_only:
                result = run_prepared(args.output)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"Inpainting pilot stopped: {exc}\n")
    print(json.dumps({k:v for k,v in result.items() if k in {
        "status", "requested_count", "selected_count", "available_count", "generated_count", "failed_count"}}, indent=2))


if __name__ == "__main__":
    main()
