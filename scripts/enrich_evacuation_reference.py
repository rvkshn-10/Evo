#!/usr/bin/env python3
"""Merge FCUSD reference rows with public NIST/NFPA/transit evacuation studies."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_PATH = ROOT / "data/processed/evacuation_reference.json"
STUDIES_PATH = ROOT / "data/reference/public_evacuation_studies.json"
DEFAULT_OUT = ROOT / "data/processed/evacuation_reference_enriched.json"


def row_signature(row: dict) -> str:
    parts = [
        str(row.get("Scenario", "")),
        str(row.get("Category", "")),
        str(row.get("Occupancy (#)", "")),
        str(row.get("Density (#)", "")),
        str(row.get("source_sheet", "")),
        str(row.get("data_origin", "fcusd_reference")),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build enriched evacuation reference dataset")
    parser.add_argument("--base", type=Path, default=BASE_PATH)
    parser.add_argument("--studies", type=Path, default=STUDIES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    base_rows = load_rows(args.base)
    study_rows = load_rows(args.studies)
    if not base_rows:
        raise SystemExit(f"Base reference missing or empty: {args.base}")

    seen = {row_signature(row) for row in base_rows}
    added = 0
    merged = list(base_rows)
    for row in study_rows:
        sig = row_signature(row)
        if sig in seen:
            continue
        merged.append(row)
        seen.add(sig)
        added += 1

    payload = {
        "generated_from": {
            "base": str(args.base.relative_to(ROOT)),
            "studies": str(args.studies.relative_to(ROOT)),
        },
        "base_row_count": len(base_rows),
        "public_study_rows_added": added,
        "total_rows": len(merged),
        "rows": merged,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Flat array copy for trainers that expect a list at evacuation_reference_enriched rows only
    flat_path = args.output.with_name("evacuation_reference_enriched_rows.json")
    flat_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(args.output),
                "flat_rows": str(flat_path),
                "base_row_count": len(base_rows),
                "public_study_rows_added": added,
                "total_rows": len(merged),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
