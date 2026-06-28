"""PeopleSense zone auto-deployment for government alert ingestion."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

from config.settings import settings
from services.peoplesense_client import PeopleSenseClient

logger = logging.getLogger(__name__)

DEPLOYMENT_MANIFEST = Path(settings.OUTPUT_DIR) / "peoplesense" / "deployments.json"


class PeopleSenseDeploymentService:
    """
    Auto-deploy monitoring zones to PeopleSense when hazards are detected.

  Mirrors the PeopleSense platform pattern of ingesting USGS + FEMA feeds and
    provisioning occupancy zones for affected geographies.
    """

    def __init__(self, client: Optional[PeopleSenseClient] = None):
        self.client = client or PeopleSenseClient()
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def deploy_zone(
        self,
        *,
        name: str,
        lat: float,
        lon: float,
        radius_m: float = 500,
        occupancy: Optional[int] = None,
        source_alert_id: Optional[str] = None,
        feed_source: str = "manual",
    ) -> dict[str, Any]:
        """Register or update a monitoring zone in PeopleSense."""
        zone_id = f"zone-{uuid.uuid4().hex[:12]}"
        payload = {
            "zone_id": zone_id,
            "name": name,
            "lat": lat,
            "lon": lon,
            "radius_m": radius_m,
            "occupancy": occupancy,
            "source_alert_id": source_alert_id,
            "feed_source": feed_source,
        }

        if self.client.is_placeholder:
            record = self._record_placeholder_deployment(payload)
        else:
            record = self._deploy_live(payload)

        self._append_manifest(record)
        return record

    def deploy_from_event(
        self,
        *,
        center_lat: float,
        center_lon: float,
        diameter_miles: float,
        occupancy: int,
        event_type: str = "earthquake",
        title: Optional[str] = None,
        source_alert_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Auto-deploy a PeopleSense zone from simplified event POST data.

        Larry's Phase 1 structure: GPS coordinates, diameter, occupancy.
        """
        radius_m = max(200, diameter_miles * 1609.34 / 2)
        name = title or f"{event_type.title()} zone ({center_lat:.4f}, {center_lon:.4f})"

        zone = self.deploy_zone(
            name=name,
            lat=center_lat,
            lon=center_lon,
            radius_m=radius_m,
            occupancy=occupancy,
            source_alert_id=source_alert_id,
            feed_source=f"event:{event_type}",
        )

        reading = self.client.get_zone_occupancy(
            lat=center_lat,
            lon=center_lon,
            radius_m=radius_m,
            zone_name=name,
        )

        return {
            "deployment": zone,
            "occupancy_reading": reading,
            "event_summary": {
                "center_lat": center_lat,
                "center_lon": center_lon,
                "diameter_miles": diameter_miles,
                "occupancy": occupancy,
                "event_type": event_type,
            },
        }

    def auto_deploy_monitoring_spots(self) -> list[dict[str, Any]]:
        """Deploy all spots from config/monitoring_locations.json."""
        locations_path = Path(__file__).resolve().parents[1] / "config" / "monitoring_locations.json"
        if not locations_path.exists():
            return []

        with locations_path.open(encoding="utf-8") as handle:
            spots = json.load(handle).get("spots", [])

        deployments = []
        for spot in spots:
            deployments.append(
                self.deploy_zone(
                    name=spot["name"],
                    lat=spot["lat"],
                    lon=spot["lon"],
                    radius_m=spot.get("radius_m", 400),
                    occupancy=spot.get("default_occupancy"),
                    feed_source="monitoring_config",
                )
            )
        return deployments

    def auto_deploy_for_alert(self, alert: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Deploy a zone when a government alert includes coordinates."""
        lat = alert.get("center_lat")
        lon = alert.get("center_lon")
        if lat is None or lon is None:
            return None

        diameter = alert.get("diameter_miles", 15.0)
        occupancy = alert.get("occupancy") or alert.get("affected_population") or 1000

        return self.deploy_from_event(
            center_lat=lat,
            center_lon=lon,
            diameter_miles=diameter,
            occupancy=int(occupancy),
            event_type=alert.get("event_type", "other"),
            title=alert.get("headline") or alert.get("title") or alert.get("event"),
            source_alert_id=str(alert.get("id", "")),
        )

    def get_deployment_status(self) -> dict[str, Any]:
        manifest = self._load_manifest()
        return {
            "mode": "placeholder" if self.client.is_placeholder else "live",
            "total_deployments": len(manifest.get("deployments", [])),
            "last_deployment": manifest.get("deployments", [{}])[-1] if manifest.get("deployments") else None,
            "manifest_path": str(DEPLOYMENT_MANIFEST),
        }

    def _deploy_live(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self.session.post(
                f"{self.client.base_url}/v1/zones/deploy",
                json={
                    "name": payload["name"],
                    "lat": payload["lat"],
                    "lon": payload["lon"],
                    "radius_m": payload["radius_m"],
                    "occupancy": payload.get("occupancy"),
                    "metadata": {
                        "source_alert_id": payload.get("source_alert_id"),
                        "feed_source": payload.get("feed_source"),
                    },
                },
                headers={"Authorization": f"Bearer {self.client.api_key}"},
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
            return {
                **payload,
                "status": "deployed",
                "peoplesense_zone_id": data.get("zone_id", payload["zone_id"]),
                "deployed_at": datetime.now(timezone.utc).isoformat(),
                "mode": "live",
            }
        except requests.RequestException as exc:
            logger.warning("PeopleSense deploy API failed, recording placeholder: %s", exc)
            return self._record_placeholder_deployment(payload)

    def _record_placeholder_deployment(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            **payload,
            "status": "deployed_placeholder",
            "peoplesense_zone_id": payload["zone_id"],
            "deployed_at": datetime.now(timezone.utc).isoformat(),
            "mode": "placeholder",
        }

    def _load_manifest(self) -> dict[str, Any]:
        if not DEPLOYMENT_MANIFEST.exists():
            return {"deployments": []}
        with DEPLOYMENT_MANIFEST.open(encoding="utf-8") as handle:
            return json.load(handle)

    def _append_manifest(self, record: dict[str, Any]) -> None:
        manifest = self._load_manifest()
        manifest.setdefault("deployments", []).append(record)
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        DEPLOYMENT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        DEPLOYMENT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
