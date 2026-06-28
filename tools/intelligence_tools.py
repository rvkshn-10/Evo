"""LangChain tools for the Evacuation Intelligence Agent."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from langchain.tools import tool

from config.settings import settings
from services.alert_processor import AlertProcessor
from services.evacuation_predictor import EvacuationPredictor
from services.government_feed_sync import GovernmentFeedSync
from services.noaa_client import NOAAClient
from services.peoplesense_client import PeopleSenseClient
from services.peoplesense_deployment import PeopleSenseDeploymentService
from services.usgs_client import USGSClient

logger = logging.getLogger(__name__)

_noaa = NOAAClient()
_peoplesense = PeopleSenseClient()
_predictor = EvacuationPredictor()
_processor = AlertProcessor()
_usgs = USGSClient()
_deployer = PeopleSenseDeploymentService()
_feed_sync = GovernmentFeedSync()


@tool
def fetch_government_alerts(area: str = "CA", limit: int = 10) -> str:
    """
    Fetch active weather and hazard warnings from NOAA/NWS for a US state.

    Args:
        area: Two-letter US state code (e.g. CA, TX)
        limit: Maximum number of alerts to return

    Returns:
        JSON string of normalized government alerts
    """
    alerts = _noaa.get_active_alerts(area=area, limit=limit)
    if not alerts:
        lat, lon = settings.DEFAULT_MAP_LAT, settings.DEFAULT_MAP_LON
        alerts = _noaa.get_active_alerts(point=(lat, lon), limit=limit)
    return json.dumps({"count": len(alerts), "alerts": alerts}, indent=2)


@tool
def get_peoplesense_occupancy(lat: float, lon: float, zone_name: str = "") -> str:
    """
    Get real-time crowd occupancy from PeopleSense for a geographic zone.

    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
        zone_name: Human-readable name for the zone (e.g. school or shelter name)

    Returns:
        JSON string with occupancy count, density, volatility, and confidence
    """
    reading = _peoplesense.get_zone_occupancy(
        lat=lat,
        lon=lon,
        radius_m=400,
        zone_name=zone_name or None,
    )
    return json.dumps(reading, indent=2)


@tool
def predict_evacuation_rate(
    spot_name: str,
    category: str,
    occupancy: int,
    density: float,
    event_type: str = "other",
) -> str:
    """
    Predict evacuation success rate and time for a location using reference datasets.

    Args:
        spot_name: Name of the monitoring location
        category: Venue type — Train Station, Stadium, or Office Building
        occupancy: Current occupancy count from PeopleSense or estimates
        density: Crowd density (0.0–1.0)
        event_type: Hazard type — fire, flood, earthquake, tornado, tsunami, or other

    Returns:
        JSON string with predicted evacuation rate, time, and risk level
    """
    spot_id = spot_name.lower().replace(" ", "-")
    result = _predictor.predict_for_spot(
        spot_id=spot_id,
        name=spot_name,
        category=category,
        occupancy=occupancy,
        density=density,
        event_type=event_type,
    )
    return json.dumps(result, indent=2)


@tool
def list_monitoring_spots() -> str:
    """
    List configured evacuation monitoring spots (schools, shelters, transit hubs).

    Returns:
        JSON string of spot definitions with coordinates and default occupancy
    """
    return json.dumps({"spots": _processor.locations}, indent=2)


@tool
def publish_dashboard_snapshot(output_path: str = "output/dashboard/latest_snapshot.json") -> str:
    """
    Build and save the public dashboard snapshot from current NOAA + PeopleSense + model data.

    Args:
        output_path: Where to write the dashboard JSON consumed by the website

    Returns:
        Confirmation with alert count and high-risk spot count
    """
    snapshot = _processor.get_dashboard_snapshot()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return json.dumps(
        {
            "status": "published",
            "path": str(path),
            "active_alerts": snapshot["summary"]["active_alerts"],
            "high_risk_spots": snapshot["summary"]["high_risk_spots"],
        },
        indent=2,
    )


@tool
def enrich_alert_with_evacuation_analysis(alert_id: str) -> str:
    """
    Combine a NOAA alert with PeopleSense occupancy and evacuation predictions for all spots.

    Args:
        alert_id: NOAA alert identifier

    Returns:
        JSON string with full enriched analysis for the alert
    """
    alert = _noaa.get_alert_by_id(alert_id)
    if not alert:
        return json.dumps({"error": f"Alert not found: {alert_id}"})
    enriched = _processor.enrich_alert(alert)
    return json.dumps(enriched, indent=2)


@tool
def fetch_earthquake_early_warnings(min_magnitude: float = 4.0, limit: int = 10) -> str:
    """
    Fetch USGS earthquake early-warning candidates from public feeds.

    Args:
        min_magnitude: Minimum magnitude threshold for EEW-style action
        limit: Maximum events to return

    Returns:
        JSON string of earthquake EEW candidates with recommended actions
    """
    candidates = _usgs.get_earthquake_early_warning_candidates(
        min_magnitude=min_magnitude, limit=limit
    )
    return json.dumps({"count": len(candidates), "candidates": candidates}, indent=2)


@tool
def sync_government_feeds(area: str = "CA", auto_deploy: bool = True) -> str:
    """
    Ingest USGS, FEMA IPAWS, and NOAA feeds; auto-deploy PeopleSense zones for hazards.

    Args:
        area: US state code for regional filtering
        auto_deploy: Whether to auto-deploy PeopleSense zones for detected hazards

    Returns:
        JSON string with sync results and deployment records
    """
    result = _feed_sync.sync_all(area=area, auto_deploy=auto_deploy)
    return json.dumps(result, indent=2)


@tool
def deploy_peoplesense_zone(
    name: str,
    lat: float,
    lon: float,
    radius_m: float = 500,
    occupancy: int = 0,
) -> str:
    """
    Auto-deploy a PeopleSense monitoring zone at a GPS coordinate.

    Args:
        name: Zone name (school, shelter, etc.)
        lat: Latitude
        lon: Longitude
        radius_m: Monitoring radius in meters
        occupancy: Expected occupancy count

    Returns:
        JSON string with deployment record
    """
    record = _deployer.deploy_zone(
        name=name,
        lat=lat,
        lon=lon,
        radius_m=radius_m,
        occupancy=occupancy or None,
        feed_source="agent",
    )
    return json.dumps(record, indent=2)
