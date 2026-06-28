"""PeopleSense FCUSD integration — GET occupancy database + POST event API."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from datetime import datetime, timezone
from typing import Any, Optional

import requests

from config.settings import settings
from services.peoplesense_xml import build_occupancy_xml, parse_occupancy_xml

logger = logging.getLogger(__name__)

_OCCUPANCY_CACHE: dict[str, Any] = {"fetched_at": 0.0, "payload": None}


class PeopleSenseClient:
    """
    PeopleSense FCUSD integration.

    Read path (preferred, rate-limited ~1/min):
      GET {PEOPLESENSE_OCCUPANCY_URL}?filter=ALL
      Header: x-api-key: {PEOPLESENSE_API_KEY}

    Write path (Pi edge / hazard events):
      POST {PEOPLESENSE_EVENT_URL}
      Body: {"OccupancyXML": "<xml>...</xml>", "Message": "event text"}
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        event_url: Optional[str] = None,
        occupancy_url: Optional[str] = None,
        cache_ttl_seconds: Optional[int] = None,
    ):
        self.api_key = api_key if api_key is not None else settings.PEOPLESENSE_API_KEY
        self.event_url = (event_url or settings.PEOPLESENSE_EVENT_URL).strip()
        self.occupancy_url = (occupancy_url or settings.PEOPLESENSE_OCCUPANCY_URL).strip()
        self.cache_ttl_seconds = cache_ttl_seconds or settings.PEOPLESENSE_CACHE_TTL_SECONDS
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
            }
        )

    @property
    def is_placeholder(self) -> bool:
        key = (self.api_key or "").strip()
        if not key or key.startswith("your-peoplesense") or key == "placeholder":
            return True
        if self.occupancy_url and "execute-api" in self.occupancy_url:
            return False
        return not self.event_url or self.event_url.startswith("https://api.peoplesense.ai")

    @property
    def occupancy_api_enabled(self) -> bool:
        return bool(self.occupancy_url) and "execute-api" in self.occupancy_url

    def fetch_all_occupancy(self, *, filter_value: str = "ALL", force: bool = False) -> dict[str, Any]:
        """Fetch all place occupancy records; cached for PEOPLESENSE_CACHE_TTL_SECONDS."""
        if self.is_placeholder:
            return {
                "filter": filter_value,
                "count": 0,
                "fetchedAt": datetime.now(timezone.utc).isoformat(),
                "data": [],
                "mode": "placeholder",
            }

        now = time.monotonic()
        cached = _OCCUPANCY_CACHE.get("payload")
        age = now - float(_OCCUPANCY_CACHE.get("fetched_at") or 0)
        if cached and not force and age < self.cache_ttl_seconds:
            return {**cached, "cache_hit": True, "cache_age_seconds": round(age, 1)}

        url = self.occupancy_url.rstrip("/")
        try:
            response = self.session.get(
                url,
                params={"filter": filter_value},
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            logger.warning("PeopleSense occupancy GET failed: %s", exc)
            if cached:
                return {**cached, "cache_hit": True, "stale": True, "error": str(exc)}
            return {
                "filter": filter_value,
                "count": 0,
                "fetchedAt": datetime.now(timezone.utc).isoformat(),
                "data": [],
                "mode": "error",
                "error": str(exc),
            }

        if isinstance(payload, dict):
            payload.setdefault("mode", "live")
            payload["cache_hit"] = False
            _OCCUPANCY_CACHE["payload"] = payload
            _OCCUPANCY_CACHE["fetched_at"] = now
            return payload

        return {
            "filter": filter_value,
            "count": 0,
            "fetchedAt": datetime.now(timezone.utc).isoformat(),
            "data": [],
            "mode": "error",
            "error": "unexpected response shape",
        }

    def post_event(
        self,
        *,
        occupancy_xml: str,
        message: str,
    ) -> dict[str, Any]:
        """POST OccupancyXML + Message to the PeopleSense event endpoint."""
        if self.is_placeholder:
            return {
                "status": "placeholder",
                "message": message,
                "mode": "placeholder",
            }

        payload = {"OccupancyXML": occupancy_xml, "Message": message}
        try:
            response = self.session.post(self.event_url, json=payload, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("PeopleSense event POST failed: %s", exc)
            return {"status": "error", "error": str(exc), "message": message, "mode": "live"}

        return self._parse_event_response(response, message=message)

    def get_zone_occupancy(
        self,
        *,
        lat: float,
        lon: float,
        radius_m: float = 500,
        zone_name: Optional[str] = None,
        spot: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        spot = dict(spot or {})
        spot.setdefault("lat", lat)
        spot.setdefault("lon", lon)
        spot.setdefault("radius_m", radius_m)
        spot.setdefault("name", zone_name)

        if self.occupancy_api_enabled:
            payload = self.fetch_all_occupancy()
            records = payload.get("data") or []
            reading = self._reading_from_records(spot, records, payload)
            if reading:
                return reading
            return self._default_zone_from_spot(spot, source="peoplesense_no_local_match")

        if self.is_placeholder:
            return self._simulate_zone(lat, lon, radius_m, zone_name)

        return self._fallback_event_zone(lat, lon, radius_m, zone_name, spot)

    def get_alert_overlay(
        self,
        *,
        alert_id: str,
        geometry: Optional[dict[str, Any]] = None,
        spots: Optional[list[dict[str, Any]]] = None,
        alert_message: Optional[str] = None,
    ) -> dict[str, Any]:
        spots = spots or []

        if self.occupancy_api_enabled and spots:
            payload = self.fetch_all_occupancy()
            records = payload.get("data") or []
            readings = [
                self._reading_from_records(spot, records, payload)
                or self._default_zone_from_spot(spot, source="peoplesense_no_local_match")
                for spot in spots
            ]
            mode = "live"
        elif not self.is_placeholder and spots:
            xml = build_occupancy_xml(
                [
                    {
                        "id": spot.get("id"),
                        "name": spot.get("name"),
                        "lat": spot.get("lat"),
                        "lon": spot.get("lon"),
                        "radius_m": spot.get("radius_m", 400),
                        "category": spot.get("category"),
                        "occupancy_count": spot.get("default_occupancy"),
                        "occupancy_density": spot.get("default_density"),
                    }
                    for spot in spots
                ]
            )
            message = alert_message or f"FCUSD hazard overlay for alert {alert_id}"
            batch = self.post_event(occupancy_xml=xml, message=message)
            zones_from_response = batch.get("zones") or []
            readings = []
            for spot in spots:
                match = _match_zone(
                    zones_from_response,
                    spot.get("name"),
                    spot.get("lat"),
                    spot.get("lon"),
                )
                if match and match.get("occupancy_count") is not None:
                    readings.append(
                        self._normalize_zone(
                            match,
                            spot["lat"],
                            spot["lon"],
                            spot.get("radius_m", 400),
                            spot.get("name"),
                            live=True,
                        )
                    )
                else:
                    readings.append(
                        self.get_zone_occupancy(
                            lat=spot["lat"],
                            lon=spot["lon"],
                            radius_m=spot.get("radius_m", 400),
                            zone_name=spot.get("name"),
                            spot=spot,
                        )
                    )
            mode = "live"
        else:
            readings = [
                self.get_zone_occupancy(
                    lat=spot["lat"],
                    lon=spot["lon"],
                    radius_m=spot.get("radius_m", 400),
                    zone_name=spot.get("name"),
                    spot=spot,
                )
                for spot in spots
            ]
            mode = "placeholder"

        return {
            "alert_id": alert_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "source": "peoplesense_get_api" if self.occupancy_api_enabled else mode,
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

    def _reading_from_records(
        self,
        spot: dict[str, Any],
        records: list[dict[str, Any]],
        payload: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        matched = _match_records_for_spot(spot, records)
        if not matched:
            return None

        counts = [_occupancy_count(record) for record in matched]
        valid_counts = [value for value in counts if value is not None]
        if not valid_counts:
            return None

        occupancy = int(sum(valid_counts))
        max_caps = [
            int(record.get("MaxOccupancy"))
            for record in matched
            if record.get("MaxOccupancy") not in (None, 0)
        ]
        capacity = sum(max_caps) if max_caps else int(spot.get("default_occupancy") or occupancy or 1)
        density = round(min(1.0, occupancy / max(capacity, 1)), 3)

        live_count = sum(1 for record in matched if (record.get("ScanMode") or "").upper() == "LIVE")
        confidence = round(0.7 + 0.25 * (live_count / max(len(matched), 1)), 3)
        volatility = round(min(1.0, 0.2 + (len(matched) / 20)), 3)

        timestamps = [record.get("Timestamp") for record in matched if record.get("Timestamp")]
        return {
            "zone_name": spot.get("name") or spot.get("id"),
            "lat": spot.get("lat"),
            "lon": spot.get("lon"),
            "radius_m": spot.get("radius_m", 400),
            "occupancy_count": occupancy,
            "occupancy_density": density,
            "occupancy_volatility": volatility,
            "confidence_level": confidence,
            "source": "peoplesense_get_api",
            "timestamp": payload.get("fetchedAt") or datetime.now(timezone.utc).isoformat(),
            "matched_locations": len(matched),
            "scan_modes": sorted({(record.get("ScanMode") or "unknown") for record in matched}),
            "peoplesense_places": sorted({record.get("PlaceID") for record in matched if record.get("PlaceID")}),
            "last_sensor_timestamp": max(timestamps) if timestamps else None,
            "cache_hit": payload.get("cache_hit", False),
        }

    def _fallback_event_zone(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        zone_name: Optional[str],
        spot: dict[str, Any],
    ) -> dict[str, Any]:
        zone_payload = {
            "id": spot.get("id") or f"zone_{lat:.4f}_{lon:.4f}",
            "name": zone_name or spot.get("name") or f"zone_{lat:.4f}_{lon:.4f}",
            "lat": lat,
            "lon": lon,
            "radius_m": radius_m,
            "category": spot.get("category"),
            "occupancy_count": spot.get("default_occupancy"),
            "occupancy_density": spot.get("default_density"),
        }
        xml = build_occupancy_xml([zone_payload])
        message = f"FCUSD occupancy snapshot request: {zone_payload['name']}"
        result = self.post_event(occupancy_xml=xml, message=message)

        if result.get("status") == "error":
            simulated = self._simulate_zone(lat, lon, radius_m, zone_name)
            simulated["source"] = "peoplesense_fallback"
            simulated["peoplesense_error"] = result.get("error")
            return simulated

        zones = result.get("zones") or []
        match = _match_zone(zones, zone_name, lat, lon) or (zones[0] if zones else None)
        if match and match.get("occupancy_count") is not None:
            return self._normalize_zone(match, lat, lon, radius_m, zone_name, live=True)

        return {
            "zone_name": zone_name or zone_payload["name"],
            "lat": lat,
            "lon": lon,
            "radius_m": radius_m,
            "occupancy_count": int(zone_payload.get("occupancy_count") or 0),
            "occupancy_density": float(zone_payload.get("occupancy_density") or 0.4),
            "occupancy_volatility": 0.35,
            "confidence_level": 0.7,
            "source": "peoplesense_live_event",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "peoplesense_status": result.get("status", "accepted"),
            "peoplesense_message": result.get("raw_message") or message,
        }

    def _parse_event_response(self, response: requests.Response, *, message: str) -> dict[str, Any]:
        content_type = response.headers.get("content-type", "")
        body_text = response.text
        parsed: dict[str, Any] = {
            "status": "accepted",
            "http_status": response.status_code,
            "mode": "live",
            "raw_message": message,
        }

        if "json" in content_type:
            try:
                data = response.json()
            except json.JSONDecodeError:
                data = {"raw": body_text}
            parsed.update(data if isinstance(data, dict) else {"body": data})
            xml_blob = _extract_xml_blob(parsed)
            if xml_blob:
                parsed["zones"] = parse_occupancy_xml(xml_blob)
            return parsed

        if body_text.strip().startswith("<"):
            parsed["zones"] = parse_occupancy_xml(body_text)
            parsed["OccupancyXML"] = body_text
            return parsed

        parsed["body"] = body_text
        return parsed

    def _normalize_zone(
        self,
        data: dict[str, Any],
        lat: float,
        lon: float,
        radius_m: float,
        zone_name: Optional[str],
        *,
        live: bool,
    ) -> dict[str, Any]:
        return {
            "zone_name": zone_name or data.get("zone_name") or data.get("name") or f"zone_{lat:.4f}_{lon:.4f}",
            "lat": data.get("lat") if data.get("lat") is not None else lat,
            "lon": data.get("lon") if data.get("lon") is not None else lon,
            "radius_m": radius_m,
            "occupancy_count": int(data.get("occupancy_count") or 0),
            "occupancy_density": float(data.get("occupancy_density") or 0.0),
            "occupancy_volatility": float(data.get("occupancy_volatility") or 0.35),
            "confidence_level": float(data.get("confidence_level") or 0.85),
            "source": "peoplesense_live" if live else "peoplesense_placeholder",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _default_zone_from_spot(
        self,
        spot: dict[str, Any],
        *,
        source: str,
    ) -> dict[str, Any]:
        occupancy = int(spot.get("default_occupancy") or 0)
        density = float(spot.get("default_density") or 0.4)
        return {
            "zone_name": spot.get("name") or spot.get("id"),
            "lat": spot.get("lat"),
            "lon": spot.get("lon"),
            "radius_m": spot.get("radius_m", 400),
            "occupancy_count": occupancy,
            "occupancy_density": density,
            "occupancy_volatility": 0.35,
            "confidence_level": 0.55,
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "matched_locations": 0,
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


def _occupancy_count(record: dict[str, Any]) -> Optional[int]:
    derived = record.get("DerivedCount")
    if derived is not None:
        return int(float(derived))
    occupancy = record.get("Occupancy")
    if occupancy is not None:
        return int(occupancy)
    return None


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _match_records_for_spot(spot: dict[str, Any], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    match_cfg = spot.get("peoplesense_match") or {}
    place_ids = match_cfg.get("place_ids") or []
    if match_cfg.get("place_id"):
        place_ids.append(match_cfg["place_id"])
    place_ids = [value.upper() for value in place_ids if value]

    group_id = match_cfg.get("group_id")
    group_contains = (match_cfg.get("group_contains") or "").strip().lower()
    radius_m = float(spot.get("peoplesense_match_radius_m") or spot.get("radius_m") or 500)
    lat = float(spot["lat"])
    lon = float(spot["lon"])

    matched: list[dict[str, Any]] = []
    for record in records:
        rlat = float(record.get("Latitude") or 0)
        rlon = float(record.get("Longitude") or 0)
        if rlat == 0.0 and rlon == 0.0:
            continue

        pid = (record.get("PlaceID") or "").upper()
        gid = (record.get("GroupID") or "") or ""
        within_radius = _haversine_m(lat, lon, rlat, rlon) <= radius_m
        if not within_radius:
            continue

        if place_ids and pid not in place_ids:
            continue
        if group_id and gid != group_id:
            continue
        if group_contains and group_contains not in gid.lower():
            continue
        matched.append(record)

    if matched:
        return _prefer_live_records(matched)
    return []


def _prefer_live_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    live = [record for record in records if (record.get("ScanMode") or "").upper() == "LIVE"]
    if live:
        return live
    return records


def _extract_xml_blob(data: dict[str, Any]) -> Optional[str]:
    for key in ("OccupancyXML", "occupancy_xml", "occupancyXml", "xml"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _match_zone(
    zones: list[dict[str, Any]],
    name: Optional[str],
    lat: Optional[float],
    lon: Optional[float],
) -> Optional[dict[str, Any]]:
    if not zones:
        return None
    if name:
        for zone in zones:
            if zone.get("zone_name") == name or zone.get("name") == name:
                return zone
    if lat is not None and lon is not None:
        for zone in zones:
            zlat, zlon = zone.get("lat"), zone.get("lon")
            if zlat is None or zlon is None:
                continue
            if abs(float(zlat) - float(lat)) < 0.01 and abs(float(zlon) - float(lon)) < 0.01:
                return zone
    return None
