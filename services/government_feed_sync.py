"""Sync USGS, FEMA, and NOAA feeds and auto-deploy PeopleSense zones."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config.settings import settings
from services.fema_ipaws_client import FEMAIPAWSClient
from services.noaa_client import NOAAClient
from services.peoplesense_deployment import PeopleSenseDeploymentService
from services.usgs_client import USGSClient

logger = logging.getLogger(__name__)

SYNC_STATE_PATH = Path(settings.OUTPUT_DIR) / "feeds" / "last_sync.json"


class GovernmentFeedSync:
    """
    Orchestrate government feed ingestion and PeopleSense auto-deployment.

    Aligns with the framework's Natural Disaster Agent + Occupancy Agent split:
    - USGS / FEMA / NOAA → hazard detection
    - PeopleSense → zone deployment + occupancy
    """

    def __init__(self):
        self.usgs = USGSClient()
        self.fema = FEMAIPAWSClient()
        self.noaa = NOAAClient()
        self.deployer = PeopleSenseDeploymentService()

    def sync_all(
        self,
        *,
        area: Optional[str] = None,
        auto_deploy: bool = True,
    ) -> dict[str, Any]:
        area = area or settings.DEFAULT_ALERT_AREA
        lat, lon = settings.DEFAULT_MAP_LAT, settings.DEFAULT_MAP_LON

        earthquakes = self.usgs.get_recent_earthquakes(window="hour", limit=20)
        eew_candidates = self.usgs.get_earthquake_early_warning_candidates(limit=10)
        fema_alerts = self.fema.get_recent_alerts(hours=48, limit=15)
        noaa_alerts = self.noaa.get_active_alerts(point=(lat, lon), limit=15)
        if not noaa_alerts:
            noaa_alerts = self.noaa.get_active_alerts(area=area, limit=15)

        deployments = []
        if auto_deploy and settings.PEOPLESENSE_AUTO_DEPLOY:
            for candidate in eew_candidates:
                result = self.deployer.auto_deploy_for_alert(candidate)
                if result:
                    deployments.append(result)

            for alert in noaa_alerts[:5]:
                if alert.get("severity") in ("Extreme", "Severe"):
                    result = self.deployer.auto_deploy_for_alert(alert)
                    if result:
                        deployments.append(result)

        snapshot = {
            "synced_at": datetime.now(timezone.utc).isoformat(),
            "area": area,
            "earthquakes_last_hour": len(earthquakes),
            "eew_candidates": eew_candidates,
            "fema_ipaws_alerts": fema_alerts,
            "noaa_alerts": noaa_alerts,
            "peoplesense_deployments": deployments,
            "peoplesense_status": self.deployer.get_deployment_status(),
        }

        self._save_sync_state(snapshot)
        return snapshot

    def eew_candidates_to_events(self) -> list[dict[str, Any]]:
        """Convert EEW candidates into simplified emergency event payloads."""
        events = []
        for candidate in self.usgs.get_earthquake_early_warning_candidates():
            if candidate.get("center_lat") is None:
                continue
            events.append({
                "status": "start",
                "timestamp": candidate.get("time") or datetime.now(timezone.utc).isoformat(),
                "event_type": "earthquake",
                "center_lat": candidate["center_lat"],
                "center_lon": candidate["center_lon"],
                "diameter_miles": candidate.get("diameter_miles", 15.0),
                "affected_population": 5000,
                "title": candidate.get("title", "USGS Earthquake Alert"),
                "description": (
                    f"USGS detected M{candidate.get('magnitude')} earthquake at "
                    f"{candidate.get('place')}. EEW status: {candidate.get('eew_status')}."
                ),
                "source": "usgs_eew",
                "usgs_id": candidate.get("id"),
            })
        return events

    def _save_sync_state(self, snapshot: dict[str, Any]) -> None:
        SYNC_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        SYNC_STATE_PATH.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
