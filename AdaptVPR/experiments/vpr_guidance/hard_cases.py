"""Hard case management for LoRA fine-tuning.

This module handles loading SALAD retrieval error cases and preparing them
for generator fine-tuning. A hard case is a query where SALAD made a retrieval
mistake on the validation set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class HardCase:
    """A retrieval error case from SALAD validation.

    Attributes:
        query_id: Identifier of the query image
        query_path: Path to the query image
        source_path: Path to the original source image (if available)
        correct_match_id: ID of the correct match (ground truth)
        wrong_match_id: ID of the incorrectly retrieved match
        retrieval_rank: Where the correct match actually ranked
        distance_to_wrong: SALAD descriptor distance to wrong match
        distance_to_correct: SALAD descriptor distance to correct match
    """
    query_id: str
    query_path: str
    source_path: Optional[str]
    correct_match_id: str
    wrong_match_id: str
    retrieval_rank: int
    distance_to_wrong: float
    distance_to_correct: float


def load_hard_cases(manifest_path: Path, gsv_root: Optional[Path] = None) -> list[HardCase]:
    """Load hard cases from a SALAD validation manifest.

    Args:
        manifest_path: Path to the hard_cases.json file produced by SALAD evaluation
        gsv_root: Root directory of GSV-Cities images (optional, for source lookup)

    Returns:
        List of HardCase instances
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Hard case manifest not found: {manifest_path}")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = []

    for entry in data.get("error_cases", []):
        query_id = entry["query_id"]
        query_path = entry["query_path"]

        # Try to find source image if GSV root is provided
        source_path = None
        if gsv_root and "source_id" in entry:
            source_id = entry["source_id"]
            # GSV-Cities structure: Images/CITY/PANOID_HEADING.jpg
            potential = gsv_root / "Images" / source_id
            if potential.exists():
                source_path = str(potential)

        cases.append(HardCase(
            query_id=query_id,
            query_path=query_path,
            source_path=source_path,
            correct_match_id=entry["correct_match_id"],
            wrong_match_id=entry["wrong_match_id"],
            retrieval_rank=entry.get("rank", -1),
            distance_to_wrong=entry.get("distance_to_wrong", 0.0),
            distance_to_correct=entry.get("distance_to_correct", 1.0),
        ))

    return cases


def filter_cases_with_source(cases: list[HardCase]) -> list[HardCase]:
    """Filter to only cases where source image is available."""
    return [c for c in cases if c.source_path is not None]


def save_hard_cases_summary(cases: list[HardCase], output_path: Path) -> None:
    """Save a summary of hard cases for inspection."""
    summary = {
        "total_cases": len(cases),
        "cases_with_source": sum(1 for c in cases if c.source_path is not None),
        "avg_rank": sum(c.retrieval_rank for c in cases) / len(cases) if cases else 0,
        "cases": [
            {
                "query_id": c.query_id,
                "has_source": c.source_path is not None,
                "rank": c.retrieval_rank,
                "distance_gap": c.distance_to_correct - c.distance_to_wrong,
            }
            for c in cases[:100]  # First 100 for inspection
        ]
    }
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
