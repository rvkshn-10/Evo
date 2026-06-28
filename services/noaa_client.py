"""NOAA / NWS active weather alerts client."""

from __future__ import annotations

import logging
from typing import Any, Optional

import requests

from config.settings import settings

logger = logging.getLogger(__name__)

NWS_BASE_URL = "https://api.weather.gov"


class NOAAClient:
    """Fetch government weather warnings from the NWS API."""

    def __init__(self, user_agent: Optional[str] = None):
        self.user_agent = user_agent or settings.NWS_USER_AGENT
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent or settings.NWS_USER_AGENT,
                "Accept": "application/geo+json",
            }
        )

    def get_active_alerts(
        self,
        *,
        area: Optional[str] = None,
        point: Optional[tuple[float, float]] = None,
        zone: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return normalized active alerts for a state, point, or forecast zone."""
        params: dict[str, str] = {}
        if area:
            params["area"] = area
        if point:
            params["point"] = f"{point[0]},{point[1]}"
        if zone:
            params["zone"] = zone

        url = f"{NWS_BASE_URL}/alerts/active"
        try:
            response = self.session.get(url, params=params, timeout=20)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("NWS alerts request failed: %s", exc)
            return []

        payload = response.json()
        features = payload.get("features", [])[:limit]
        return [self._normalize_alert(feature) for feature in features]

    def get_alert_by_id(self, alert_id: str) -> Optional[dict[str, Any]]:
        url = f"{NWS_BASE_URL}/alerts/{alert_id}"
        try:
            response = self.session.get(url, timeout=20)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("NWS alert lookup failed for %s: %s", alert_id, exc)
            return None
        return self._normalize_alert(response.json())

    @staticmethod
    def _normalize_alert(feature: dict[str, Any]) -> dict[str, Any]:
        props = feature.get("properties", {})
        geometry = feature.get("geometry")
        center = _geometry_center(geometry)

        return {
            "id": props.get("id") or feature.get("id"),
            "event": props.get("event"),
            "headline": props.get("headline"),
            "description": props.get("description"),
            "instruction": props.get("instruction"),
            "severity": props.get("severity"),
            "urgency": props.get("urgency"),
            "certainty": props.get("certainty"),
            "area_desc": props.get("areaDesc"),
            "sent": props.get("sent"),
            "effective": props.get("effective"),
            "onset": props.get("onset"),
            "expires": props.get("expires"),
            "status": props.get("status"),
            "message_type": props.get("messageType"),
            "category": props.get("category"),
            "event_type": _map_event_type(props.get("event", "")),
            "geometry": geometry,
            "center_lat": center[0] if center else None,
            "center_lon": center[1] if center else None,
            "source": "noaa_nws",
        }


def _geometry_center(geometry: Optional[dict[str, Any]]) -> Optional[tuple[float, float]]:
    if not geometry:
        return None

    coords = geometry.get("coordinates")
    geom_type = geometry.get("type")
    points: list[tuple[float, float]] = []

    if geom_type == "Point" and isinstance(coords, list) and len(coords) >= 2:
        return float(coords[1]), float(coords[0])

    if geom_type == "Polygon" and isinstance(coords, list):
        ring = coords[0] if coords else []
        points = [(float(c[1]), float(c[0])) for c in ring if len(c) >= 2]

    if geom_type == "MultiPolygon" and isinstance(coords, list):
        for polygon in coords:
            if polygon and polygon[0]:
                points.extend((float(c[1]), float(c[0])) for c in polygon[0] if len(c) >= 2)

    if not points:
        return None

    lat = sum(p[0] for p in points) / len(points)
    lon = sum(p[1] for p in points) / len(points)
    return lat, lon


def _map_event_type(event_name: str) -> str:
    name = (event_name or "").lower()
    if "earthquake" in name or "seismic" in name:
        return "earthquake"
    if "tsunami" in name:
        return "tsunami"
    if "tornado" in name:
        return "tornado"
    if "flood" in name or "flash flood" in name:
        return "flood"
    if "fire" in name or "smoke" in name:
        return "fire"
    return "other"
