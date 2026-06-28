"""PeopleSense crowd occupancy API client (placeholder mode supported)."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import requests

from config.settings import settings

logger = logging.getLogger(__name__)


class PeopleSenseClient:
    """
    PeopleSense provides real-time occupancy and crowd volatility for emergency zones.

    When PEOPLESENSE_API_KEY is unset or still the placeholder, the client returns
    deterministic simulated readings so the rest of the pipeline can be developed.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key if api_key is not None else settings.PEOPLESENSE_API_KEY
        self.base_url = (base_url or settings.PEOPLESENSE_BASE_URL).rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    @property
    def is_placeholder(self) -> bool:
        key = (self.api_key or "").strip()
        return not key or key.startswith("your-peoplesense") or key == "placeholder"

    def get_zone_occupancy(
        self,
        *,
        lat: float,
        lon: float,
        radius_m: float = 500,
        zone_name: Optional[str] = None,
    ) -> dict[str, Any]:
        if self.is_placeholder:
            return self._simulate_zone(lat, lon, radius_m, zone_name)

        try:
            response = self.session.get(
                f"{self.base_url}/v1/occupancy",
                params={"lat": lat, "lon": lon, "radius_m": radius_m},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
            return self._normalize_zone(data, lat, lon, radius_m, zone_name)
        except requests.RequestException as exc:
            logger.warning("PeopleSense API unavailable, using placeholder data: %s", exc)
            return self._simulate_zone(lat, lon, radius_m, zone_name)

    def get_alert_overlay(
        self,
        *,
        alert_id: str,
        geometry: Optional[dict[str, Any]] = None,
        spots: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        spots = spots or []
        readings = [
            self.get_zone_occupancy(
                lat=spot["lat"],
                lon=spot["lon"],
                radius_m=spot.get("radius_m", 400),
                zone_name=spot.get("name"),
            )
            for spot in spots
        ]
        return {
            "alert_id": alert_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "placeholder" if self.is_placeholder else "live",
            "zones": readings,
            "summary": {
                "total_occupancy": sum(r["occupancy_count"] for r in readings),
                "avg_density": round(
                    sum(r["occupancy_density"] for r in readings) / max(len(readings), 1),
                    3,
                ),
                "high_volatility_zones": [
                    r["zone_name"] for r in readings if r["occupancy_volatility"] >= 0.65
                ],
            },
        }

    def _normalize_zone(
        self,
        data: dict[str, Any],
        lat: float,
        lon: float,
        radius_m: float,
        zone_name: Optional[str],
    ) -> dict[str, Any]:
        return {
            "zone_name": zone_name or data.get("name") or f"zone_{lat:.4f}_{lon:.4f}",
            "lat": lat,
            "lon": lon,
            "radius_m": radius_m,
            "occupancy_count": int(data.get("occupancy_count", data.get("count", 0))),
            "occupancy_density": float(data.get("occupancy_density", data.get("density", 0))),
            "occupancy_volatility": float(
                data.get("occupancy_volatility", data.get("volatility", 0))
            ),
            "confidence_level": float(data.get("confidence_level", data.get("confidence", 0.8))),
            "source": "peoplesense_live",
            "timestamp": data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        }

    def _simulate_zone(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        zone_name: Optional[str],
    ) -> dict[str, Any]:
        seed = hashlib.sha256(f"{lat:.5f}:{lon:.5f}:{radius_m}".encode()).hexdigest()
        bucket = int(seed[:8], 16)

        occupancy = 120 + (bucket % 1800)
        density = round(0.15 + (bucket % 850) / 1000, 3)
        volatility = round(0.2 + (bucket % 700) / 1000, 3)
        confidence = round(0.72 + (bucket % 200) / 1000, 3)

        return {
            "zone_name": zone_name or f"zone_{lat:.4f}_{lon:.4f}",
            "lat": lat,
            "lon": lon,
            "radius_m": radius_m,
            "occupancy_count": occupancy,
            "occupancy_density": density,
            "occupancy_volatility": volatility,
            "confidence_level": confidence,
            "source": "peoplesense_placeholder",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
