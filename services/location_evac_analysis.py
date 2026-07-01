"""Analyze a user-picked map location: occupancy, Evo prediction, and egress routes."""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

from config.settings import settings
from services.evacuation_predictor import EvacuationPredictor
from services.hazard_feature_builder import (
    LOCATIONS_PATH,
    collect_live_hazards,
    enrich_spot_with_hazards,
    haversine_km,
    nearest_hazard,
)
from services.peoplesense_client import PeopleSenseClient
from services.blueprint_features import load_blueprint_for_point
from services.evo_features import encode_features, load_feature_schema
from services.evo_runtime import get_evo14_runtime

logger = logging.getLogger(__name__)

OSRM_BASE = "https://router.project-osrm.org/route/v1/foot"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
PIN_TRAINING_LOG = settings.PROJECT_ROOT / "data" / "incoming" / "evo1.3" / "pin_location_analyses.jsonl"

# Candidate assembly headings (degrees) — research demo, not site-specific CAD
ASSEMBLY_HEADINGS = (0, 90, 180, 270)
DETOUR_HEADINGS = (45, 135, 225, 315)
DEFAULT_ASSEMBLY_DISTANCE_M = 350.0
COMPASS_BY_HEADING = {
    0: "North",
    45: "Northeast",
    90: "East",
    135: "Southeast",
    180: "South",
    225: "Southwest",
    270: "West",
    315: "Northwest",
}


def _load_spots() -> list[dict[str, Any]]:
    if not LOCATIONS_PATH.exists():
        return []
    payload = json.loads(LOCATIONS_PATH.read_text(encoding="utf-8"))
    return payload.get("spots", [])


def _destination_point(lat: float, lon: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
    radius = 6371000.0
    bearing = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(
        math.sin(lat1) * math.cos(distance_m / radius)
        + math.cos(lat1) * math.sin(distance_m / radius) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(distance_m / radius) * math.cos(lat1),
        math.cos(distance_m / radius) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def _reverse_geocode(lat: float, lon: float) -> Optional[str]:
    try:
        response = requests.get(
            NOMINATIM_URL,
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 16},
            headers={"User-Agent": settings.NWS_USER_AGENT or "Evo-Evac-Intelligence/1.0"},
            timeout=12,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("display_name")
    except requests.RequestException as exc:
        logger.warning("Reverse geocode failed: %s", exc)
        return None


def _nearest_monitoring_spot(lat: float, lon: float) -> Optional[dict[str, Any]]:
    best: Optional[dict[str, Any]] = None
    best_km = float("inf")
    for spot in _load_spots():
        dist = haversine_km(lat, lon, float(spot["lat"]), float(spot["lon"]))
        if dist < best_km:
            best_km = dist
            best = {**spot, "distance_km": round(dist, 2)}
    return best


def _fallback_walk_route(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
) -> dict[str, Any]:
    """Straight-line walking estimate when OSRM is unreachable."""
    distance_m = haversine_km(start_lat, start_lon, end_lat, end_lon) * 1000.0 * 1.3
    duration_min = distance_m / 80.0
    return {
        "distance_m": round(distance_m, 1),
        "duration_min": round(duration_min, 2),
        "geometry": None,
    }


def _osrm_walk_route(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
) -> Optional[dict[str, Any]]:
    url = f"{OSRM_BASE}/{start_lon},{start_lat};{end_lon},{end_lat}"
    try:
        response = requests.get(
            url,
            params={"overview": "false", "steps": "false"},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != "Ok" or not payload.get("routes"):
            return _fallback_walk_route(start_lat, start_lon, end_lat, end_lon)
        route = payload["routes"][0]
        return {
            "distance_m": round(float(route.get("distance") or 0), 1),
            "duration_min": round(float(route.get("duration") or 0) / 60.0, 2),
            "geometry": route.get("geometry"),
            "source": "osrm_foot",
        }
    except requests.RequestException as exc:
        logger.warning("OSRM route failed, using haversine fallback: %s", exc)
        result = _fallback_walk_route(start_lat, start_lon, end_lat, end_lon)
        result["source"] = "haversine_fallback"
        return result


def _estimate_egress_defaults(nearest_spot: Optional[dict[str, Any]], occupancy: int) -> dict[str, float]:
    """Proxy egress geometry from nearest FCUSD site or category heuristics."""
    if nearest_spot and float(nearest_spot.get("distance_km") or 99) <= 2.0:
        category = str(nearest_spot.get("category") or "Office Building")
    else:
        category = "Office Building"

    if category == "Stadium":
        return {
            "egress_exit_count": 12.0,
            "egress_usable_width_m": 4.5,
            "egress_blockage_fraction": 0.05,
        }
    if category == "Train Station":
        return {
            "egress_exit_count": 6.0,
            "egress_usable_width_m": 3.0,
            "egress_blockage_fraction": 0.1,
        }
    # School / office default
    exits = max(2, min(8, int(math.ceil(occupancy / 400))))
    return {
        "egress_exit_count": float(exits),
        "egress_usable_width_m": 2.0,
        "egress_blockage_fraction": 0.08,
    }


def _point_near_block(
    lat: float,
    lon: float,
    blocked_points: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    for block in blocked_points:
        blat = block.get("lat")
        blon = block.get("lon")
        if blat is None or blon is None:
            continue
        radius_m = float(block.get("radius_m") or 80.0)
        if haversine_km(lat, lon, float(blat), float(blon)) * 1000.0 <= radius_m:
            return block
    return None


def _rank_evacuation_routes(
    *,
    lat: float,
    lon: float,
    occupancy: int,
    density: float,
    egress: dict[str, float],
    blocked_headings: Optional[list[int]] = None,
    blocked_points: Optional[list[dict[str, Any]]] = None,
    blockage_reason: Optional[str] = None,
) -> list[dict[str, Any]]:
    blocked_headings = set(blocked_headings or [])
    blocked_points = blocked_points or []
    exit_count = max(1.0, egress["egress_exit_count"])
    width_m = max(0.5, egress["egress_usable_width_m"])
    blockage = min(0.95, max(0.0, egress["egress_blockage_fraction"]))

    # Fewer usable exits when operator marks directions blocked (fire, debris, etc.)
    blocked_exit_penalty = min(0.85, len(blocked_headings) * 0.2)
    effective_exits = max(1.0, exit_count * (1.0 - blocked_exit_penalty))
    flow_capacity = effective_exits * width_m * 45.0 * (1.0 - blockage)
    routes: list[dict[str, Any]] = []

    headings = list(ASSEMBLY_HEADINGS)
    if blocked_headings:
        headings.extend(h for h in DETOUR_HEADINGS if h not in headings)

    for heading in headings:
        if heading in blocked_headings:
            continue

        end_lat, end_lon = _destination_point(lat, lon, heading, DEFAULT_ASSEMBLY_DISTANCE_M)
        if _point_near_block(end_lat, end_lon, blocked_points):
            continue

        osrm = _osrm_walk_route(lat, lon, end_lat, end_lon)
        if not osrm:
            continue

        distance_m = float(osrm["distance_m"])
        duration_min = float(osrm["duration_min"])
        crowd_delay = 1.0 + (occupancy / max(flow_capacity, 1.0)) * 0.35 + density * 0.25
        detour_penalty = 1.08 if heading in DETOUR_HEADINGS else 1.0
        estimated_clear_min = round(duration_min * crowd_delay * detour_penalty, 2)
        score = estimated_clear_min / max(flow_capacity / 100.0, 1.0)

        compass = COMPASS_BY_HEADING.get(heading, str(heading))
        routes.append(
            {
                "heading_deg": heading,
                "compass": compass,
                "assembly_lat": round(end_lat, 6),
                "assembly_lon": round(end_lon, 6),
                "walk_distance_m": distance_m,
                "walk_time_min": duration_min,
                "estimated_clear_time_min": estimated_clear_min,
                "crowd_delay_factor": round(crowd_delay, 3),
                "rank_score": round(score, 4),
                "is_detour": heading in DETOUR_HEADINGS,
                "source": osrm.get("source", "osrm_foot"),
            }
        )

    routes.sort(key=lambda item: item["rank_score"])
    for index, route in enumerate(routes, start=1):
        route["rank"] = index
        route["recommended"] = index == 1

    if not routes and (blocked_headings or blocked_points):
        return [
            {
                "heading_deg": None,
                "compass": "No route",
                "blocked": True,
                "blockage_reason": blockage_reason or "user_blocked",
                "note": "All cardinal/detour headings are blocked. Add fewer blockers or widen hazard radius.",
            }
        ]
    return routes


def _append_training_log(row: dict[str, Any]) -> None:
    try:
        PIN_TRAINING_LOG.parent.mkdir(parents=True, exist_ok=True)
        with PIN_TRAINING_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
    except OSError as exc:
        logger.warning("Could not append pin training log: %s", exc)


def generate_rule_based_briefing(analysis: dict[str, Any]) -> str:
    """Credit-free operational summary from model and route outputs."""
    route = analysis.get("recommended_route") or {}
    prediction = analysis.get("prediction") or {}
    compass = route.get("compass") or "No safe route"
    clear = route.get("estimated_clear_time_min")
    risk = prediction.get("risk_level", "unknown")
    source = route.get("source", "route fallback")
    blocked = len(analysis.get("blocked_headings") or []) + len(analysis.get("blocked_points") or [])
    clear_text = f"{float(clear):.1f} minutes" if clear is not None else "unavailable"
    return f"Recommended heading: {compass}. Estimated clear time: {clear_text}. Risk: {risk}. Route source: {source}. Active blocked-exit/hazard markers: {blocked}."


def analyze_map_location(
    *,
    lat: float,
    lon: float,
    name: Optional[str] = None,
    category: str = "Office Building",
    use_evo13: bool = True,
    use_evo14: bool = True,
    log_for_training: bool = True,
    blocked_headings: Optional[list[int]] = None,
    blocked_points: Optional[list[dict[str, Any]]] = None,
    blockage_reason: Optional[str] = None,
    blueprint: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Full analysis for a user-selected point on the map."""
    peoplesense = PeopleSenseClient()
    occupancy_reading = peoplesense.get_zone_occupancy(
        lat=lat,
        lon=lon,
        radius_m=400,
        zone_name=name or "User selected location",
    )

    occupancy = int(occupancy_reading.get("occupancy_count") or 0)
    density = float(occupancy_reading.get("occupancy_density") or 0.35)
    if occupancy <= 0:
        occupancy = 200
        density = 0.3

    nearest_spot = _nearest_monitoring_spot(lat, lon)
    place_name = name or _reverse_geocode(lat, lon) or f"Pin {lat:.4f}, {lon:.4f}"
    egress_defaults = _estimate_egress_defaults(nearest_spot, occupancy)
    structured_blueprint = load_blueprint_for_point(lat, lon)
    if blueprint:
        structured_blueprint.update({key: value for key, value in blueprint.items() if value is not None})
    blueprint = structured_blueprint
    if blueprint:
        if blueprint.get("exit_count"):
            egress_defaults["egress_exit_count"] = float(blueprint["exit_count"])
        if blueprint.get("notes") and "narrow" in str(blueprint["notes"]).lower():
            egress_defaults["egress_usable_width_m"] = max(
                1.0, egress_defaults["egress_usable_width_m"] - 0.5
            )
    if blocked_headings or blocked_points:
        extra_block = min(0.75, 0.15 * len(blocked_headings or []) + 0.1 * len(blocked_points or []))
        egress_defaults["egress_blockage_fraction"] = min(
            0.95,
            egress_defaults["egress_blockage_fraction"] + extra_block,
        )

    hazards = collect_live_hazards(lat=lat, lon=lon)
    pseudo_spot = {
        "id": "user-pin",
        "name": place_name,
        "category": category,
        "lat": lat,
        "lon": lon,
        "default_occupancy": occupancy,
        "default_density": density,
    }
    hazard_ctx = enrich_spot_with_hazards(pseudo_spot, hazards)
    nearest = nearest_hazard(pseudo_spot, hazards)

    routes = _rank_evacuation_routes(
        lat=lat,
        lon=lon,
        occupancy=occupancy,
        density=density,
        egress=egress_defaults,
        blocked_headings=blocked_headings,
        blocked_points=blocked_points,
        blockage_reason=blockage_reason,
    )
    best_route = next((route for route in routes if not route.get("blocked")), None)
    route_length_m = float(best_route.get("walk_distance_m") or 150.0) if best_route else 150.0

    route_model_output = None
    if use_evo14:
        runtime = get_evo14_runtime()
        schema = load_feature_schema(settings.EVO14_MODEL_VERSION)
        if runtime.is_available and schema:
            route_features = encode_features(
                schema=schema, occupancy=occupancy, density=density, category=category,
                scenario="Standard Evacuation Drill",
                egress_exit_count=egress_defaults["egress_exit_count"],
                egress_usable_width_m=egress_defaults["egress_usable_width_m"],
                egress_route_length_m=route_length_m,
                egress_blockage_fraction=egress_defaults["egress_blockage_fraction"],
                blocked_headings=list(blocked_headings or []),
                hazard_point_count=len(blocked_points or []),
                blueprint_exit_count=float(blueprint.get("exit_count") or 0),
                blueprint_floor_count=float(blueprint.get("floor_count") or 0),
                blueprint_corridor_length_m=float(blueprint.get("longest_corridor_m") or 0),
            )
            route_model_output = runtime.predict_route_head(route_features)
            if route_model_output:
                candidate = next(
                    (route for route in routes if route.get("heading_deg") == route_model_output["best_heading_deg"]),
                    None,
                )
                if candidate is None and route_model_output.get("best_heading_deg") is not None:
                    heading = int(route_model_output["best_heading_deg"])
                    end_lat, end_lon = _destination_point(lat, lon, heading, DEFAULT_ASSEMBLY_DISTANCE_M)
                    osrm = _osrm_walk_route(lat, lon, end_lat, end_lon)
                    if osrm:
                        candidate = {
                            "heading_deg": heading,
                            "compass": route_model_output.get("best_compass_heading") or COMPASS_BY_HEADING.get(heading, str(heading)),
                            "assembly_lat": round(end_lat, 6),
                            "assembly_lon": round(end_lon, 6),
                            "walk_distance_m": osrm["distance_m"],
                            "walk_time_min": osrm["duration_min"],
                            "estimated_clear_time_min": round(route_model_output["estimated_clear_time_min"], 2),
                            "source": "evo1.4_route_head",
                            "heading_confidence": round(route_model_output["heading_confidence"], 4),
                            "is_detour": heading in DETOUR_HEADINGS,
                            "rank_score": 0.0,
                        }
                        routes.append(candidate)
                if candidate:
                    candidate = {**candidate, "estimated_clear_time_min": round(route_model_output["estimated_clear_time_min"], 2), "source": "evo1.4_route_head", "heading_confidence": round(route_model_output["heading_confidence"], 4)}
                    routes = [candidate, *[route for route in routes if route.get("heading_deg") != candidate.get("heading_deg")]]
                    for index, route in enumerate(routes, 1): route.update({"rank": index, "recommended": index == 1})
                    best_route = candidate

    predictor = EvacuationPredictor(use_evo14=use_evo14, use_evo13=use_evo13, use_evo=not use_evo13 and not use_evo14)
    capacity_hint = None
    if nearest_spot:
        capacity_hint = int(nearest_spot.get("default_occupancy") or 0) or None

    prediction = predictor.predict_for_spot(
        spot_id="user-pin",
        name=place_name,
        category=category,
        occupancy=occupancy,
        density=density,
        event_type=str(hazard_ctx.get("event_type") or "other"),
        lat=lat,
        lon=lon,
        hazard={
            **hazard_ctx,
            "egress_exit_count": egress_defaults["egress_exit_count"],
            "egress_usable_width_m": egress_defaults["egress_usable_width_m"],
            "egress_route_length_m": route_length_m,
            "egress_blockage_fraction": egress_defaults["egress_blockage_fraction"],
            "blocked_headings": list(blocked_headings or []),
            "blocked_points": blocked_points or [],
            "blueprint_exit_count": blueprint.get("exit_count") or 0,
            "blueprint_floor_count": blueprint.get("floor_count") or 0,
            "blueprint_corridor_length_m": blueprint.get("longest_corridor_m") or 0,
        },
        capacity_hint=capacity_hint,
    )

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "location": {
            "lat": lat,
            "lon": lon,
            "name": place_name,
            "category": category,
        },
        "peoplesense": {
            "occupancy": occupancy,
            "density": density,
            "mode": occupancy_reading.get("mode") or peoplesense.occupancy_api_enabled,
            "source": occupancy_reading.get("source"),
            "zone_name": occupancy_reading.get("zone_name"),
        },
        "nearest_monitoring_spot": nearest_spot,
        "nearest_active_hazard": nearest,
        "egress_assumptions": {
            **egress_defaults,
            "egress_route_length_m": route_length_m,
            "note": "Proxied from nearest FCUSD site category when within 2 km; otherwise school/office defaults.",
        },
        "evacuation_routes": routes,
        "recommended_route": best_route,
        "blocked_headings": list(blocked_headings or []),
        "blocked_points": blocked_points or [],
        "blockage_reason": blockage_reason,
        "blueprint": blueprint,
        "prediction": prediction,
        "route_model_output": route_model_output,
        "research_notice": (
            "Evo 1.4 route-head is used when validated research artifacts are present; OSRM remains the fallback. "
            "Mark blocked exits/hazards on the map to force detour headings. "
            "Indoor floor plans are not used until blueprint features are trained into Evo — see "
            "docs/codex/EVO_ROUTE_BLUEPRINT_TRAINING_PROMPT.md. "
            "Pin rows log to data/incoming/evo1.3/pin_location_analyses.jsonl."
        ),
    }
    result["rule_based_briefing"] = generate_rule_based_briefing(result)

    if log_for_training:
        _append_training_log(
            {
                "recorded_at": result["generated_at"],
                "lat": lat,
                "lon": lon,
                "place_name": place_name,
                "category": category,
                "occupancy": occupancy,
                "density": density,
                "egress_exit_count": egress_defaults["egress_exit_count"],
                "egress_usable_width_m": egress_defaults["egress_usable_width_m"],
                "egress_route_length_m": route_length_m,
                "egress_blockage_fraction": egress_defaults["egress_blockage_fraction"],
                "blocked_headings": list(blocked_headings or []),
                "blocked_points": blocked_points or [],
                "blockage_reason": blockage_reason,
                "blueprint_url": (blueprint or {}).get("url"),
                "blueprint_exit_count": (blueprint or {}).get("exit_count"),
                "recommended_compass": (best_route or {}).get("compass"),
                "is_detour": (best_route or {}).get("is_detour"),
                "estimated_clear_time_min": (best_route or {}).get("estimated_clear_time_min"),
                "predicted_evacuation_success_pct": prediction.get("predicted_evacuation_success_pct"),
                "predicted_evacuation_time_min": prediction.get("predicted_evacuation_time_min"),
                "data_origin": "user_map_pin",
                "labels_available": False,
            }
        )

    return result
