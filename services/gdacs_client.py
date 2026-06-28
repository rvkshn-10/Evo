"""GDACS multi-hazard disaster alerts (free JSON API, no key required)."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

GDACS_BASE = "https://www.gdacs.org/gdacsapi/api/events/geteventlist"

EVENT_TYPE_MAP = {
    "EQ": "earthquake",
    "FL": "flood",
    "TC": "other",
    "WF": "fire",
    "VO": "other",
    "DR": "other",
    "TS": "tsunami",
}

ALERT_SEVERITY = {
    "red": 1.0,
    "orange": 0.75,
    "green": 0.35,
}


class GDACSClient:
    """Fetch global disaster events from GDACS MHEWS API."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def get_recent_events(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Last ~4 days, up to 100 events (GDACS cap)."""
        try:
            response = self.session.get(f"{GDACS_BASE}/EVENTS4APP", timeout=25)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("GDACS EVENTS4APP request failed: %s", exc)
            return []

        features = response.json().get("features", [])[:limit]
        return [self._normalize(feature) for feature in features]

    def search_events(
        self,
        *,
        event_types: Optional[list[str]] = None,
        alert_levels: Optional[list[str]] = None,
        days: int = 90,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Custom date-range search (max 100 results per GDACS)."""
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=days)
        params: dict[str, str] = {
            "fromdate": start.isoformat(),
            "todate": end.isoformat(),
        }
        if event_types:
            params["eventlist"] = ";".join(event_types)
        if alert_levels:
            params["alertlevel"] = ";".join(alert_levels)

        try:
            response = self.session.get(f"{GDACS_BASE}/SEARCH", params=params, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("GDACS SEARCH request failed: %s", exc)
            return []

        features = response.json().get("features", [])[:limit]
        return [self._normalize(feature) for feature in features]

    def get_california_relevant(
        self,
        *,
        lat: float,
        lon: float,
        radius_km: float = 800.0,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """Filter recent GDACS events within radius of a map center (Sacramento default)."""
        events = self.get_recent_events(limit=100)
        relevant = []
        for event in events:
            elat = event.get("center_lat")
            elon = event.get("center_lon")
            if elat is None or elon is None:
                continue
            if _haversine_km(lat, lon, elat, elon) <= radius_km:
                relevant.append(event)
        return relevant[:limit]

    def _normalize(self, feature: dict[str, Any]) -> dict[str, Any]:
        props = feature.get("properties", {})
        geometry = feature.get("geometry") or {}
        coords = geometry.get("coordinates") or []
        lat = float(coords[1]) if len(coords) >= 2 else None
        lon = float(coords[0]) if len(coords) >= 2 else None

        event_code = str(props.get("eventtype") or "")
        alert_level = str(props.get("alertlevel") or "Green").lower()
        severity_data = props.get("severitydata") or {}

        magnitude = None
        if event_code == "EQ":
            try:
                magnitude = float(severity_data.get("severity"))
            except (TypeError, ValueError):
                magnitude = None

        return {
            "id": f"gdacs-{props.get('eventtype')}-{props.get('eventid')}",
            "event": props.get("name") or props.get("description"),
            "headline": props.get("name"),
            "description": props.get("description"),
            "event_type": EVENT_TYPE_MAP.get(event_code, "other"),
            "gdacs_event_type": event_code,
            "alert_level": props.get("alertlevel"),
            "alert_score": props.get("alertscore"),
            "severity_score": ALERT_SEVERITY.get(alert_level, 0.35),
            "hazard_magnitude": magnitude,
            "severity_text": severity_data.get("severitytext"),
            "severity_unit": severity_data.get("severityunit"),
            "country": props.get("country"),
            "from_date": props.get("fromdate"),
            "to_date": props.get("todate"),
            "center_lat": lat,
            "center_lon": lon,
            "geometry": geometry,
            "source": "gdacs",
            "url": (props.get("url") or {}).get("report") if isinstance(props.get("url"), dict) else None,
        }


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math

    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))
