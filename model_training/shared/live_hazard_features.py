"""
Shared helpers for Evo training — merge labeled Excel baseline with live hazard feeds.

Used by Evo 1.2+ Colab notebooks. Do not treat hazard_live_seed rows as labeled outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELED_PATH = PROJECT_ROOT / "data" / "processed" / "evacuation_reference.json"
SEED_PATH = PROJECT_ROOT / "data" / "processed" / "hazard_live_seed.json"


def load_labeled_reference() -> list[dict[str, Any]]:
    return json.loads(LABELED_PATH.read_text(encoding="utf-8"))


def load_live_hazard_seed() -> dict[str, Any]:
    if not SEED_PATH.exists():
        return {"rows": [], "note": "Run scripts/build_hazard_training_seed.py first"}
    return json.loads(SEED_PATH.read_text(encoding="utf-8"))


def merge_for_training(
    *,
    include_unlabeled_live_rows: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Return (rows, metadata).

    Default: labeled Excel-derived rows only.
    Set include_unlabeled_live_rows=True to append live hazard feature rows
    (features only — no success/time labels).
    """
    labeled = load_labeled_reference()
    seed = load_live_hazard_seed()
    rows = list(labeled)
    if include_unlabeled_live_rows:
        rows.extend(seed.get("rows", []))
    meta = {
        "labeled_rows": len(labeled),
        "live_seed_rows": len(seed.get("rows", [])),
        "merged_rows": len(rows),
        "live_sources": seed.get("sources_active", []),
    }
    return rows, meta
