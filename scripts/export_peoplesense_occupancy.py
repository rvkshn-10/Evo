#!/usr/bin/env python3
"""Export matched PeopleSense GET occupancy records as Evo 1.3 OccupancyXML."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings
from services.peoplesense_client import PeopleSenseClient
from services.peoplesense_xml import build_occupancy_xml


DEFAULT_OUTPUT_DIR = ROOT / "data" / "incoming" / "evo1.3" / "peoplesense"
DEFAULT_LOCATIONS = ROOT / "config" / "monitoring_locations.json"


def parse_timestamp(value: Any) -> datetime:
    if value:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def records_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("data", "items", "records", "places"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def safe_filename(timestamp: datetime) -> str:
    value = timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"peoplesense-{value}.xml"


def load_spots(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [spot for spot in payload.get("spots", []) if spot.get("peoplesense_match")]


def fetch_payload(args: argparse.Namespace, client: PeopleSenseClient) -> dict[str, Any]:
    if args.input_json:
        return json.loads(args.input_json.read_text(encoding="utf-8"))
    if client.is_placeholder:
        raise RuntimeError(
            "PEOPLESENSE_API_KEY is not configured; no live occupancy export was written"
        )
    payload = client.fetch_all_occupancy(filter_value="ALL", force=True)
    if payload.get("mode") in ("placeholder", "error"):
        raise RuntimeError(str(payload.get("error") or "PeopleSense GET returned no live data"))
    return payload


def export_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    client = PeopleSenseClient(cache_ttl_seconds=1)
    payload = fetch_payload(args, client)
    records = records_from_payload(payload)
    if not records:
        raise RuntimeError("PeopleSense response contains no occupancy records")

    generated_at = parse_timestamp(
        payload.get("fetchedAt") or payload.get("fetched_at") or payload.get("timestamp")
    )
    zones = []
    unmatched = []
    for spot in load_spots(args.monitoring_locations):
        reading = client._reading_from_records(spot, records, payload)
        if not reading or reading.get("source") != "peoplesense_get_api":
            unmatched.append(spot["id"])
            continue
        zones.append(
            {
                "id": spot["id"],
                "name": spot.get("name"),
                "lat": spot.get("lat"),
                "lon": spot.get("lon"),
                "radius_m": spot.get("radius_m"),
                "category": spot.get("category"),
                "occupancy_count": reading["occupancy_count"],
                "occupancy_density": reading["occupancy_density"],
                "occupancy_volatility": reading["occupancy_volatility"],
            }
        )
    if not zones:
        raise RuntimeError(
            "No PeopleSense records matched monitoring_locations.json Place/Group/GPS rules"
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / safe_filename(generated_at)
    if output_path.exists() and not args.overwrite:
        raise RuntimeError(f"Snapshot already exists: {output_path}; pass --overwrite to replace")
    output_path.write_text(
        build_occupancy_xml(zones, generated_at=generated_at.isoformat()),
        encoding="utf-8",
    )
    return {
        "status": "written",
        "output": str(output_path),
        "generated_at": generated_at.isoformat(),
        "api_record_count": len(records),
        "matched_spot_ids": [zone["id"] for zone in zones],
        "unmatched_spot_ids": unmatched,
        "phase_b_label_warning": (
            "This is an occupancy feature snapshot, not a drill outcome label. "
            "Use it only when aligned to a measured drill outcome_timestamp."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--monitoring-locations", type=Path, default=DEFAULT_LOCATIONS)
    parser.add_argument(
        "--input-json",
        type=Path,
        help="Optional saved PeopleSense GET response; otherwise call the authenticated live API",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = export_snapshot(args)
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        print(json.dumps({"status": "blocked", "error": re.sub(r"x-api-key[^,]*", "x-api-key: [redacted]", str(exc))}, indent=2))
        raise SystemExit(2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
