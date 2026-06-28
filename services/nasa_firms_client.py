"""NASA FIRMS active fire detections (free MAP_KEY from firms.modaps.eosdis.nasa.gov)."""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import requests

from config.settings import settings

logger = logging.getLogger(__name__)

FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


class NASAFIRMSClient:
    """
    Near-real-time wildfire hotspots for California bounding boxes.

    Register a free MAP_KEY: https://firms.modaps.eosdis.nasa.gov/api/
    Without a key, returns an empty list (dashboard still works on NOAA/USGS/GDACS).
    """

    def __init__(self, map_key: Optional[str] = None):
        self.map_key = (map_key or settings.NASA_FIRMS_MAP_KEY or "").strip()
        self.session = requests.Session()

    @property
    def is_configured(self) -> bool:
        key = self.map_key
        return bool(key) and not key.startswith("your-") and key != "placeholder"

    def get_active_fires(
        self,
        *,
        west: float,
        south: float,
        east: float,
        north: float,
        day_range: int = 2,
        source: str = "VIIRS_SNPP_NRT",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if not self.is_configured:
            return []

        # FIRMS area API: MAP_KEY/SOURCE/west,south,east,north/day_range
        bbox = f"{west},{south},{east},{north}"
        url = f"{FIRMS_AREA_URL}/{self.map_key}/{source}/{bbox}/{day_range}"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("NASA FIRMS request failed: %s", exc)
            return []

        reader = csv.DictReader(io.StringIO(response.text))
        fires = []
        for row in reader:
            try:
                lat = float(row.get("latitude") or row.get("lat") or 0)
                lon = float(row.get("longitude") or row.get("lon") or 0)
            except (TypeError, ValueError):
                continue
            if lat == 0 and lon == 0:
                continue

            frp = _safe_float(row.get("frp"))
            confidence = (row.get("confidence") or row.get("conf") or "").strip()
            severity = _frp_to_severity(frp, confidence)

            fires.append(
                {
                    "id": f"firms-{row.get('acq_date', '')}-{lat:.4f}-{lon:.4f}",
                    "event": "Wildfire hotspot",
                    "headline": f"Active fire detection (FRP {frp or 'n/a'})",
                    "event_type": "fire",
                    "center_lat": lat,
                    "center_lon": lon,
                    "frp": frp,
                    "brightness": _safe_float(row.get("bright_ti4") or row.get("brightness")),
                    "confidence": confidence,
                    "acq_date": row.get("acq_date"),
                    "acq_time": row.get("acq_time"),
                    "satellite": row.get("satellite") or source,
                    "severity_score": severity,
                    "hazard_magnitude": frp,
                    "source": "nasa_firms",
                }
            )

        fires.sort(key=lambda f: f.get("frp") or 0, reverse=True)
        return fires[:limit]

    def get_california_fires(self, *, day_range: int = 2, limit: int = 50) -> list[dict[str, Any]]:
        """Northern/Central California bbox covering Sacramento / FCUSD region."""
        return self.get_active_fires(
            west=-125.0,
            south=35.0,
            east=-114.0,
            north=42.5,
            day_range=day_range,
            limit=limit,
        )


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value in (None, "", "nan"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _frp_to_severity(frp: Optional[float], confidence: str) -> float:
    base = 0.4
    if frp is not None:
        if frp >= 100:
            base = 0.95
        elif frp >= 40:
            base = 0.8
        elif frp >= 15:
            base = 0.65
        elif frp >= 5:
            base = 0.5
    if confidence.lower() in ("high", "h"):
        base = min(1.0, base + 0.1)
    return round(base, 3)
