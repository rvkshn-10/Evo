"""USGS earthquake feeds and early-warning style event detection.

ShakeAlert® production XML feeds require a USGS Technical Partnership agreement.
Phase 1 uses the public USGS GeoJSON earthquake feeds and flags significant events
as early-warning candidates for automated downstream action.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import requests

from config.settings import settings

logger = logging.getLogger(__name__)

USGS_FEED_BASE = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary"
USGS_FDSN_BASE = "https://earthquake.usgs.gov/fdsnws/event/1/query"

FeedWindow = Literal["hour", "day", "week", "month"]


class USGSClient:
    """Fetch earthquake data from public USGS feeds."""

    def __init__(self, user_agent: Optional[str] = None):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent or settings.NWS_USER_AGENT,
                "Accept": "application/json",
            }
        )

    def get_recent_earthquakes(
        self,
        *,
        window: FeedWindow = "hour",
        min_magnitude: Optional[float] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        url = f"{USGS_FEED_BASE}/all_{window}.geojson"
        try:
            response = self.session.get(url, timeout=20)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("USGS feed request failed: %s", exc)
            return []

        features = response.json().get("features", [])
        events = [self._normalize_feature(feature, source_feed=window) for feature in features]

        if min_magnitude is not None:
            events = [e for e in events if (e.get("magnitude") or 0) >= min_magnitude]

        return events[:limit]

    def get_significant_earthquakes(
        self,
        *,
        min_magnitude: Optional[float] = None,
        hours: int = 24,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        """Query FDSN event API for significant recent earthquakes."""
        min_mag = min_magnitude if min_magnitude is not None else settings.USGS_EEW_MIN_MAGNITUDE
        params = {
            "format": "geojson",
            "minmagnitude": min_mag,
            "orderby": "time",
            "limit": limit,
        }
        try:
            response = self.session.get(USGS_FDSN_BASE, params=params, timeout=20)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("USGS FDSN query failed: %s", exc)
            return self.get_recent_earthquakes(
                window="day", min_magnitude=min_mag, limit=limit
            )

        features = response.json().get("features", [])
        return [
            self._normalize_feature(feature, source_feed="fdsn")
            for feature in features
        ]

    def get_earthquake_early_warning_candidates(
        self,
        *,
        min_magnitude: Optional[float] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Return earthquakes that meet early-warning thresholds for automated action.

        True ShakeAlert XML messages require a USGS partnership. This method surfaces
        public-feed events that would trigger EEW-style workflows in our system.
        """
        min_mag = min_magnitude if min_magnitude is not None else settings.USGS_EEW_MIN_MAGNITUDE
        candidates = self.get_significant_earthquakes(min_magnitude=min_mag, limit=limit)

        enriched = []
        for event in candidates:
            event["eew_status"] = _classify_eew_status(event, min_mag)
            event["recommended_action"] = _recommended_action(event)
            if event["eew_status"] != "below_threshold":
                enriched.append(event)
        return enriched

    def get_earthquake_by_id(self, usgs_id: str) -> Optional[dict[str, Any]]:
        url = f"https://earthquake.usgs.gov/fdsnws/event/1/query"
        try:
            response = self.session.get(
                url,
                params={"format": "geojson", "eventid": usgs_id},
                timeout=20,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("USGS event lookup failed for %s: %s", usgs_id, exc)
            return None

        features = response.json().get("features", [])
        if not features:
            return None
        return self._normalize_feature(features[0], source_feed="fdsn")

    def _normalize_feature(
        self, feature: dict[str, Any], *, source_feed: str
    ) -> dict[str, Any]:
        props = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        coords = geometry.get("coordinates", [None, None, None])

        magnitude = props.get("mag")
        mmi = props.get("mmi")
        time_ms = props.get("time")
        timestamp = None
        if time_ms:
            timestamp = datetime.fromtimestamp(time_ms / 1000, tz=timezone.utc).isoformat()

        lon = float(coords[0]) if coords[0] is not None else None
        lat = float(coords[1]) if coords[1] is not None else None
        depth_km = float(coords[2]) if len(coords) > 2 and coords[2] is not None else None

        diameter_miles = _estimate_impact_diameter_miles(magnitude)

        return {
            "id": feature.get("id"),
            "event_type": "earthquake",
            "title": f"M{magnitude} Earthquake — {props.get('place', 'Unknown location')}"
            if magnitude is not None
            else props.get("title", "Earthquake"),
            "place": props.get("place"),
            "magnitude": magnitude,
            "mag_type": props.get("magType"),
            "mmi": mmi,
            "alert": props.get("alert"),
            "tsunami": bool(props.get("tsunami")),
            "status": props.get("status"),
            "center_lat": lat,
            "center_lon": lon,
            "depth_km": depth_km,
            "diameter_miles": diameter_miles,
            "time": timestamp,
            "updated": props.get("updated"),
            "url": props.get("url"),
            "sig": props.get("sig"),
            "source": "usgs",
            "source_feed": source_feed,
            "shakealert_partnership_required": True,
        }


def _estimate_impact_diameter_miles(magnitude: Optional[float]) -> float:
    if magnitude is None:
        return 10.0
    if magnitude >= 7.0:
        return 120.0
    if magnitude >= 6.0:
        return 60.0
    if magnitude >= 5.0:
        return 30.0
    if magnitude >= 4.0:
        return 15.0
    return 8.0


def _classify_eew_status(event: dict[str, Any], min_magnitude: float) -> str:
    mag = event.get("magnitude") or 0
    if mag >= min_magnitude + 1.5:
        return "critical"
    if mag >= min_magnitude:
        return "warning"
    return "below_threshold"


def _recommended_action(event: dict[str, Any]) -> str:
    status = event.get("eew_status", "below_threshold")
    if status == "critical":
        return "deploy_peoplesense_zones_and_trigger_pipeline"
    if status == "warning":
        return "deploy_peoplesense_zones_and_monitor"
    return "log_only"
