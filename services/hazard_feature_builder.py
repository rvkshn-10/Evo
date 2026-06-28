"""Join live hazard feeds to FCUSD monitoring spots for inference and training seeds."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config.settings import settings
from services.fema_ipaws_client import FEMAIPAWSClient
from services.gdacs_client import GDACSClient
from services.nasa_firms_client import NASAFIRMSClient
from services.noaa_client import NOAAClient
from services.usgs_client import USGSClient

LOCATIONS_PATH = settings.PROJECT_ROOT / "config" / "monitoring_locations.json"
SEED_OUTPUT_PATH = settings.PROJECT_ROOT / "data" / "processed" / "hazard_live_seed.json"

NOAA_SEVERITY = {
    "extreme": 1.0,
    "severe": 0.8,
    "moderate": 0.55,
    "minor": 0.3,
    "unknown": 0.2,
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _load_spots() -> list[dict[str, Any]]:
    if not LOCATIONS_PATH.exists():
        return []
    payload = json.loads(LOCATIONS_PATH.read_text(encoding="utf-8"))
    return payload.get("spots", [])


def _hazard_magnitude(hazard: dict[str, Any]) -> float:
    if hazard.get("hazard_magnitude") is not None:
        return float(hazard["hazard_magnitude"])
    if hazard.get("magnitude") is not None:
        return float(hazard["magnitude"])
    if hazard.get("frp") is not None:
        return float(hazard["frp"])
    return 0.0


def _hazard_severity(hazard: dict[str, Any]) -> float:
    if hazard.get("severity_score") is not None:
        return float(hazard["severity_score"])
    severity = str(hazard.get("severity") or "").lower()
    if severity in NOAA_SEVERITY:
        return NOAA_SEVERITY[severity]
    alert = str(hazard.get("alert_level") or "").lower()
    if alert == "red":
        return 1.0
    if alert == "orange":
        return 0.75
    return 0.35


def collect_live_hazards(
    *,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    area: Optional[str] = None,
) -> list[dict[str, Any]]:
    lat = lat if lat is not None else settings.DEFAULT_MAP_LAT
    lon = lon if lon is not None else settings.DEFAULT_MAP_LON
    area = area or settings.DEFAULT_ALERT_AREA

    noaa = NOAAClient()
    usgs = USGSClient()
    gdacs = GDACSClient()
    firms = NASAFIRMSClient()
    fema = FEMAIPAWSClient()

    hazards: list[dict[str, Any]] = []
    hazards.extend(noaa.get_active_alerts(point=(lat, lon), limit=25))
    if not hazards:
        hazards.extend(noaa.get_active_alerts(area=area, limit=25))

    for quake in usgs.get_significant_earthquakes(min_magnitude=3.5, limit=15):
        quake["severity_score"] = min(1.0, (quake.get("magnitude") or 0) / 8.0)
        hazards.append(quake)

    # Historical local events provide real non-neutral features when no active
    # hazard happens to be inside the FCUSD 250 km training join radius.
    for quake in usgs.get_nearby_earthquakes(
        lat=lat, lon=lon, radius_km=250.0, days=90, min_magnitude=1.5, limit=100
    ):
        quake["severity_score"] = min(1.0, (quake.get("magnitude") or 0) / 8.0)
        hazards.append(quake)

    hazards.extend(gdacs.get_california_relevant(lat=lat, lon=lon, limit=20))
    hazards.extend(
        event
        for event in gdacs.search_events(days=90, limit=100)
        if event.get("center_lat") is not None
        and event.get("center_lon") is not None
        and haversine_km(
            lat,
            lon,
            float(event["center_lat"]),
            float(event["center_lon"]),
        )
        <= 250.0
    )
    hazards.extend(firms.get_california_fires(limit=30))
    # IPAWS archive rows currently lack normalized point geometry. Preserve
    # them for feed provenance/audit but never force a spatial match.
    hazards.extend(fema.get_recent_alerts(hours=24 * 90, limit=100))

    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for hazard in hazards:
        unique[(str(hazard.get("source")), str(hazard.get("id")))] = hazard
    return list(unique.values())


def nearest_hazard(
    spot: dict[str, Any],
    hazards: list[dict[str, Any]],
    *,
    max_km: float = 250.0,
) -> Optional[dict[str, Any]]:
    slat, slon = float(spot["lat"]), float(spot["lon"])
    best: Optional[dict[str, Any]] = None
    best_km = max_km
    for hazard in hazards:
        hlat = hazard.get("center_lat")
        hlon = hazard.get("center_lon")
        if hlat is None or hlon is None:
            continue
        dist = haversine_km(slat, slon, float(hlat), float(hlon))
        if dist < best_km:
            best_km = dist
            best = {**hazard, "distance_km": round(dist, 2)}
    return best


def enrich_spot_with_hazards(
    spot: dict[str, Any],
    hazards: list[dict[str, Any]],
) -> dict[str, Any]:
    nearest = nearest_hazard(spot, hazards)
    severity = _hazard_severity(nearest) if nearest else 0.0
    magnitude = _hazard_magnitude(nearest) if nearest else 0.0
    event_type = (nearest or {}).get("event_type", "other")

    return {
        **spot,
        "live_hazard": nearest,
        "hazard_source": (nearest or {}).get("source"),
        "event_type": event_type,
        "severity_score": round(severity, 3),
        "hazard_magnitude": round(magnitude, 3),
        "hazard_distance_km": (nearest or {}).get("distance_km"),
    }


def build_training_seed_rows(
    *,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    area: Optional[str] = None,
) -> dict[str, Any]:
    """
    Build supplemental training rows from live feeds + monitoring spots.

    Excel (`evacuation_reference.json`) remains the labeled evacuation baseline;
    this seed adds real NOAA/USGS/GDACS/FIRMS hazard features per FCUSD site.
    """
    spots = _load_spots()
    hazards = collect_live_hazards(lat=lat, lon=lon, area=area)
    enriched = [enrich_spot_with_hazards(spot, hazards) for spot in spots]

    rows = []
    for item in enriched:
        rows.append(
            {
                "Category": item.get("category"),
                "Scenario": _scenario_from_hazard(item),
                "Occupancy (#)": item.get("default_occupancy"),
                "Density (#)": item.get("default_density"),
                "Evacuation Time (Min)": None,
                "Evacuation Success (%)": None,
                "event_type": item.get("event_type"),
                "severity_score": item.get("severity_score"),
                "hazard_magnitude": item.get("hazard_magnitude"),
                "hazard_source": item.get("hazard_source"),
                "hazard_distance_km": item.get("hazard_distance_km"),
                "spot_id": item.get("id"),
                "spot_name": item.get("name"),
                "data_origin": "live_hazard_seed",
                "labels_available": False,
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Supplemental feature rows from live public feeds. "
            "Merge with data/processed/evacuation_reference.json for Evo training; "
            "do not treat as labeled evacuation outcomes."
        ),
        "hazard_count": len(hazards),
        "spot_count": len(spots),
        "sources_active": _active_sources(hazards),
        "rows": rows,
        "hazards": hazards,
        "raw_hazards_sample": hazards[:10],
    }
    return payload


def write_training_seed(path: Path = SEED_OUTPUT_PATH) -> dict[str, Any]:
    payload = build_training_seed_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _scenario_from_hazard(item: dict[str, Any]) -> str:
    event = item.get("event_type", "other")
    mapping = {
        "fire": "Electrical Fire",
        "flood": "Station Flooding",
        "earthquake": "Panic Chain Reaction",
        "tornado": "Sudden Overcrowding",
        "tsunami": "Station Flooding",
        "other": "Standard Evacuation Drill",
    }
    return mapping.get(event, "Standard Evacuation Drill")


def _active_sources(hazards: list[dict[str, Any]]) -> list[str]:
    return sorted({h.get("source") for h in hazards if h.get("source")})
