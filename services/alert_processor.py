"""Combine NOAA alerts, PeopleSense occupancy, and evacuation predictions."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config.settings import settings
from services.evacuation_predictor import EvacuationPredictor
from services.gdacs_client import GDACSClient
from services.hazard_feature_builder import enrich_spot_with_hazards
from services.nasa_firms_client import NASAFIRMSClient
from services.noaa_client import NOAAClient
from services.peoplesense_client import PeopleSenseClient
from services.usgs_client import USGSClient

logger = logging.getLogger(__name__)

LOCATIONS_PATH = Path(__file__).resolve().parents[1] / "config" / "monitoring_locations.json"


class AlertProcessor:
    """Orchestrates ingestion and enrichment for the public dashboard."""

    def __init__(self, *, use_evo: bool = False, use_evo13: bool = False):
        self.noaa = NOAAClient()
        self.usgs = USGSClient()
        self.gdacs = GDACSClient()
        self.firms = NASAFIRMSClient()
        self.peoplesense = PeopleSenseClient()
        self.predictor = EvacuationPredictor(use_evo=use_evo, use_evo13=use_evo13)
        self.use_evo13 = use_evo13
        self.locations = self._load_locations()

    def _load_locations(self) -> list[dict[str, Any]]:
        if not LOCATIONS_PATH.exists():
            return []
        with LOCATIONS_PATH.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload.get("spots", [])

    def get_dashboard_snapshot(
        self,
        *,
        area: Optional[str] = None,
        point: Optional[tuple[float, float]] = None,
    ) -> dict[str, Any]:
        area = area or settings.DEFAULT_ALERT_AREA
        point = point or (settings.DEFAULT_MAP_LAT, settings.DEFAULT_MAP_LON)

        # Build the live hazard set once per dashboard request. Re-fetching the
        # same remote feeds for every alert made a single request take minutes
        # and could monopolize a small single-worker VM.
        from services.hazard_feature_builder import collect_live_hazards

        live_hazards = collect_live_hazards(lat=point[0], lon=point[1], area=area)

        # NWS rejects combined area+point filters; prefer point-based alerts near map center.
        alerts = self.noaa.get_active_alerts(point=point, limit=20)
        if not alerts:
            alerts = self.noaa.get_active_alerts(area=area, limit=20)
        enriched_alerts = [
            self.enrich_alert(alert, live_hazards=live_hazards) for alert in alerts[:8]
        ]

        earthquakes = self.usgs.get_significant_earthquakes(min_magnitude=4.0, limit=10)
        eew_candidates = self.usgs.get_earthquake_early_warning_candidates(limit=5)
        enriched_quakes = []
        for quake in earthquakes[:5]:
            enriched = self.enrich_alert(
                _quake_as_alert(quake), live_hazards=live_hazards
            )
            enriched["source"] = "usgs"
            enriched["hazard_category"] = "earthquake"
            enriched["magnitude"] = quake.get("magnitude")
            enriched["title"] = quake.get("title")
            enriched["headline"] = quake.get("title") or enriched.get("headline")
            enriched_quakes.append(enriched)

        for alert in enriched_alerts:
            alert["hazard_category"] = _hazard_category(alert)

        gdacs_events = self.gdacs.get_california_relevant(lat=point[0], lon=point[1], limit=10)
        wildfires = self.firms.get_california_fires(limit=15)

        hazard_enriched_spots = [
            enrich_spot_with_hazards(spot, live_hazards) for spot in self.locations
        ]

        heatmap_points = _build_heatmap_points(
            enriched_alerts, enriched_quakes, eew_candidates, gdacs_events, wildfires
        )

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "area": area,
            "map_center": {"lat": point[0], "lon": point[1]},
            "peoplesense_mode": (
                "live"
                if self.peoplesense.occupancy_api_enabled and not self.peoplesense.is_placeholder
                else "placeholder"
            ),
            "peoplesense_source": (
                "get_api"
                if self.peoplesense.occupancy_api_enabled and not self.peoplesense.is_placeholder
                else "simulated"
            ),
            "monitoring_spots": self.locations,
            "hazard_enriched_spots": hazard_enriched_spots,
            "gdacs_events": gdacs_events,
            "wildfire_hotspots": wildfires,
            "feed_sources": _active_feed_sources(self.firms),
            "alerts": enriched_alerts,
            "earthquakes": enriched_quakes,
            "eew_candidates": eew_candidates,
            "heatmap_points": heatmap_points,
            "summary": {
                "active_alerts": len(alerts),
                "significant_earthquakes": len(earthquakes),
                "high_risk_spots": sum(
                    1
                    for alert in enriched_alerts
                    for spot in alert.get("evacuation_predictions", [])
                    if spot.get("risk_level") == "high"
                ),
            },
            "prediction_policy": (
                "evo1.3_research"
                if self.use_evo13
                else "evo1.2_hybrid"
                if self.predictor.use_evo
                else "knn_reference"
            ),
        }

    def enrich_alert(
        self,
        alert: dict[str, Any],
        *,
        live_hazards: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        occupancy_overlay = self.peoplesense.get_alert_overlay(
            alert_id=str(alert.get("id")),
            geometry=alert.get("geometry"),
            spots=self.locations,
        )

        occupancy_by_spot = {
            reading["zone_name"]: reading for reading in occupancy_overlay["zones"]
        }
        # Match readings back to configured spots by zone name
        mapped: dict[str, dict[str, Any]] = {}
        for spot in self.locations:
            reading = next(
                (
                    r
                    for r in occupancy_overlay["zones"]
                    if r.get("zone_name") == spot["name"]
                ),
                None,
            )
            if reading:
                mapped[spot["id"]] = reading

        from services.hazard_feature_builder import collect_live_hazards, enrich_spot_with_hazards

        if live_hazards is None:
            live_hazards = collect_live_hazards()
        hazard_by_spot = {
            spot["id"]: enrich_spot_with_hazards(spot, live_hazards) for spot in self.locations
        }

        predictions = self.predictor.predict_for_alert(
            alert, self.locations, mapped, hazard_by_spot=hazard_by_spot
        )
        return {
            **alert,
            "peoplesense": occupancy_overlay,
            "evacuation_predictions": predictions,
        }

    def save_snapshot(self, snapshot: dict[str, Any]) -> Path:
        out_dir = Path(settings.OUTPUT_DIR) / "dashboard"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "latest_snapshot.json"
        path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        return path

    def alert_to_emergency_event(self, alert: dict[str, Any]) -> Optional[dict[str, Any]]:
        lat = alert.get("center_lat")
        lon = alert.get("center_lon")
        if lat is None or lon is None:
            return None

        total_occupancy = alert.get("peoplesense", {}).get("summary", {}).get("total_occupancy", 0)
        affected_population = int(total_occupancy or 5000)

        return {
            "status": "start",
            "timestamp": alert.get("sent") or datetime.now(timezone.utc).isoformat(),
            "event_type": alert.get("event_type", "other"),
            "center_lat": lat,
            "center_lon": lon,
            "diameter_miles": 15.0,
            "affected_population": affected_population,
            "title": alert.get("headline") or alert.get("event") or "Government Weather Alert",
            "description": alert.get("description") or alert.get("instruction") or "",
        }


def _quake_as_alert(quake: dict[str, Any]) -> dict[str, Any]:
    """Normalize a USGS earthquake record for dashboard enrichment."""
    return {
        "id": quake.get("id"),
        "event": "Earthquake",
        "headline": quake.get("title"),
        "description": (
            f"Magnitude {quake.get('magnitude')} earthquake at {quake.get('place')}. "
            f"Depth: {quake.get('depth_km')} km."
        ),
        "severity": "Severe" if (quake.get("magnitude") or 0) >= 5.5 else "Moderate",
        "event_type": "earthquake",
        "center_lat": quake.get("center_lat"),
        "center_lon": quake.get("center_lon"),
        "sent": quake.get("time"),
        "source": "usgs",
    }


def _hazard_category(alert: dict[str, Any]) -> str:
    event_type = (alert.get("event_type") or "").lower()
    event_name = (alert.get("event") or alert.get("headline") or "").lower()

    if event_type == "earthquake" or "earthquake" in event_name:
        return "earthquake"
    if event_type in ("fire",) or "fire" in event_name or "red flag" in event_name or "smoke" in event_name:
        return "wildfire"
    if event_type == "flood" or "flood" in event_name:
        return "flood"
    if event_type == "tornado" or "tornado" in event_name:
        return "tornado"
    if event_type == "tsunami" or "tsunami" in event_name:
        return "tsunami"
    return "severe_weather"


def _build_heatmap_points(
    alerts: list[dict[str, Any]],
    earthquakes: list[dict[str, Any]],
    eew_candidates: list[dict[str, Any]],
    gdacs_events: Optional[list[dict[str, Any]]] = None,
    wildfires: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []

    for quake in eew_candidates + earthquakes:
        lat, lon = quake.get("center_lat"), quake.get("center_lon")
        if lat is None or lon is None:
            continue
        mag = float(quake.get("magnitude") or 4.0)
        points.append({
            "lat": lat,
            "lon": lon,
            "intensity": min(1.0, mag / 8.0),
            "hazard_category": "earthquake",
            "label": quake.get("headline") or quake.get("title") or "Earthquake",
            "magnitude": mag,
        })

    for alert in alerts:
        lat, lon = alert.get("center_lat"), alert.get("center_lon")
        if lat is None or lon is None:
            continue
        category = alert.get("hazard_category") or _hazard_category(alert)
        severity = (alert.get("severity") or "").lower()
        intensity = 0.5
        if severity == "extreme":
            intensity = 0.95
        elif severity == "severe":
            intensity = 0.75
        elif severity == "moderate":
            intensity = 0.55
        points.append({
            "lat": lat,
            "lon": lon,
            "intensity": intensity,
            "hazard_category": category,
            "label": alert.get("headline") or alert.get("event") or "Alert",
        })

    for event in gdacs_events or []:
        lat, lon = event.get("center_lat"), event.get("center_lon")
        if lat is None or lon is None:
            continue
        alert_level = str(event.get("alert_level") or "Green").lower()
        intensity = 0.9 if alert_level == "red" else 0.7 if alert_level == "orange" else 0.45
        points.append({
            "lat": lat,
            "lon": lon,
            "intensity": intensity,
            "hazard_category": _hazard_category(event),
            "label": event.get("headline") or event.get("event") or "GDACS event",
        })

    for fire in wildfires or []:
        lat, lon = fire.get("center_lat"), fire.get("center_lon")
        if lat is None or lon is None:
            continue
        points.append({
            "lat": lat,
            "lon": lon,
            "intensity": float(fire.get("severity_score") or 0.5),
            "hazard_category": "wildfire",
            "label": fire.get("headline") or "Wildfire hotspot",
        })

    return points


def _active_feed_sources(firms: NASAFIRMSClient) -> list[str]:
    sources = ["noaa_nws", "usgs", "gdacs"]
    if firms.is_configured:
        sources.append("nasa_firms")
    return sources
