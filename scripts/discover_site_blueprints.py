#!/usr/bin/env python3
"""Create offline blueprint feature records from manual sources and OSM footprints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.blueprint_features import (
    CONFIG_PATH,
    FEATURE_DEFAULTS,
    LOCATIONS_PATH,
    REFERENCE_DIR,
    extract_blueprint_features,
    reference_payload,
)


def osm_footprint(lat: float, lon: float) -> dict:
    query = f'[out:json][timeout:20];way(around:120,{lat},{lon})["building"];out tags center 1;'
    response = requests.post("https://overpass-api.de/api/interpreter", data={"data": query}, timeout=30)
    response.raise_for_status()
    elements = response.json().get("elements", [])
    return elements[0] if elements else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-overpass", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {"sites": {}}
    spots = json.loads(LOCATIONS_PATH.read_text(encoding="utf-8")).get("spots", [])
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    for spot in spots:
        site = (config.get("sites") or {}).get(spot["id"], {})
        source = site.get("blueprint_url") or site.get("campus_map_url")
        extraction_error = None
        try:
            features = extract_blueprint_features(source) if source else dict(FEATURE_DEFAULTS)
        except (OSError, ValueError, requests.RequestException) as exc:
            features = dict(FEATURE_DEFAULTS)
            extraction_error = str(exc)
        manual = False
        for key in FEATURE_DEFAULTS:
            value = site.get(f"manual_{key}")
            if value is not None:
                features[key] = float(value)
                manual = True
        osm = {} if args.skip_overpass else osm_footprint(float(spot["lat"]), float(spot["lon"]))
        payload = reference_payload(spot["id"], source, features, "manual_override" if manual else ("source_text" if source else "placeholder"))
        payload["osm_building"] = {"id": osm.get("id"), "tags": osm.get("tags", {})} if osm else None
        payload["extraction_error"] = extraction_error
        (REFERENCE_DIR / f'{spot["id"]}.json').write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(spot["id"], payload["confidence"])


if __name__ == "__main__":
    main()
