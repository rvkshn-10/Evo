"""FEMA IPAWS archived alerts via OpenFEMA API."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

FEMA_IPAWS_URL = "https://www.fema.gov/api/open/v1/IpawsArchivedAlerts"


class FEMAIPAWSClient:
    """
    Fetch IPAWS alert records from FEMA OpenFEMA.

    Note: archived alerts may lag ~24h. Live operational alerts are also available
    via NOAA/NWS in our NOAA client. PeopleSense ingests both USGS and FEMA feeds
    natively; this client mirrors that ingestion path for our agent system.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def get_recent_alerts(
        self,
        *,
        hours: int = 48,
        state_fips_prefix: Optional[str] = "06",
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        """
        Fetch recent IPAWS alerts, optionally filtered to a state FIPS prefix (06 = CA).
        """
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%d")
        filters = [f"sent ge '{since}'"]
        if state_fips_prefix:
            filters.append(
                f"startswith(infos/areas/geocode/value,'{state_fips_prefix}')"
            )

        params = {
            "$filter": " and ".join(filters),
            "$top": min(limit, 100),
            "$orderby": "sent desc",
            "$metadata": "false",
        }

        try:
            response = self.session.get(FEMA_IPAWS_URL, params=params, timeout=30)
            if response.status_code == 400 and state_fips_prefix:
                # OpenFEMA currently rejects nested geocode filters on this
                # endpoint. Fall back to the bounded date query and retain the
                # records as non-spatial provenance rather than inventing a
                # California coordinate.
                params["$filter"] = f"sent ge '{since}'"
                response = self.session.get(FEMA_IPAWS_URL, params=params, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("FEMA IPAWS request failed: %s", exc)
            return []

        payload = response.json()
        records = payload.get("IpawsArchivedAlerts") or payload.get("data") or []
        if isinstance(records, dict):
            records = [records]

        return [self._normalize_alert(record) for record in records[:limit]]

    @staticmethod
    def _normalize_alert(record: dict[str, Any]) -> dict[str, Any]:
        infos = record.get("info") or []
        info = infos[0] if infos else {}
        event = info.get("event") or record.get("event")
        headline = info.get("headline") or record.get("headline")

        return {
            "id": record.get("identifier") or record.get("cogId"),
            "event": event,
            "headline": headline,
            "sent": record.get("sent"),
            "msg_type": record.get("msgType"),
            "sender": record.get("sender"),
            "status": record.get("status"),
            "scope": record.get("scope"),
            "event_type": _map_event_type(str(event or "")),
            "source": "fema_ipaws",
            "center_lat": None,
            "center_lon": None,
        }


def _map_event_type(event_name: str) -> str:
    name = event_name.lower()
    if "earthquake" in name:
        return "earthquake"
    if "tsunami" in name:
        return "tsunami"
    if "tornado" in name:
        return "tornado"
    if "flood" in name:
        return "flood"
    if "fire" in name:
        return "fire"
    return "other"
