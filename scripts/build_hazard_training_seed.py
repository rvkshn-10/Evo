#!/usr/bin/env python3
"""Refresh live hazard feature rows for Evo training (supplements Excel baseline)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.hazard_feature_builder import write_training_seed


def main() -> None:
    payload = write_training_seed()
    print(f"Wrote {payload['spot_count']} spot rows from {payload['hazard_count']} live hazards")
    print(f"Active sources: {', '.join(payload['sources_active'])}")


if __name__ == "__main__":
    main()
