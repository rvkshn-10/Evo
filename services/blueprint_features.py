"""Offline, structured blueprint metadata for Evo route inference (no LLM)."""

from __future__ import annotations

import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from config.settings import settings
from services.hazard_feature_builder import haversine_km

CONFIG_PATH = settings.PROJECT_ROOT / "config" / "site_blueprints.json"
REFERENCE_DIR = settings.PROJECT_ROOT / "data" / "reference" / "site_blueprints"
LOCATIONS_PATH = settings.PROJECT_ROOT / "config" / "monitoring_locations.json"
FEATURE_DEFAULTS = {
    "exit_count": 0.0,
    "stairwell_count": 0.0,
    "floor_count": 0.0,
    "longest_corridor_m": 0.0,
    "usable_exit_width_m": 0.0,
    "assembly_points_count": 0.0,
}


def _number_near(text: str, label: str) -> float:
    patterns = (
        rf"(?:{label})\s*[:=-]?\s*(\d+(?:\.\d+)?)",
        rf"(\d+(?:\.\d+)?)\s*(?:{label})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return 0.0


def extract_blueprint_features(url_or_path: str) -> dict[str, float]:
    """Extract conservative structured values from PDF text or image metadata."""
    source = str(url_or_path or "").strip()
    if not source:
        return dict(FEATURE_DEFAULTS)
    temporary: tempfile.NamedTemporaryFile | None = None
    path = Path(source).expanduser()
    try:
        if source.startswith(("http://", "https://")):
            response = requests.get(source, timeout=25, headers={"User-Agent": "Evo-Blueprint-Offline/1.0"})
            response.raise_for_status()
            suffix = Path(source.split("?", 1)[0]).suffix or ".bin"
            temporary = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            temporary.write(response.content)
            temporary.close()
            path = Path(temporary.name)
        text = ""
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader

            text = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
        elif path.suffix.lower() in {".json", ".geojson"}:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "extracted_features" in payload:
                return {key: float(payload["extracted_features"].get(key) or 0) for key in FEATURE_DEFAULTS}
            text = json.dumps(payload)
        # PNG/JPEG contains no trustworthy egress semantics without a manual override.
        return {
            "exit_count": _number_near(text, r"exits?"),
            "stairwell_count": _number_near(text, r"stairwells?|stairs?"),
            "floor_count": _number_near(text, r"floors?|stories|storeys"),
            "longest_corridor_m": _number_near(text, r"(?:longest\s+)?corridor(?:\s+length)?(?:\s*\(?m\)?)?"),
            "usable_exit_width_m": _number_near(text, r"(?:usable\s+)?exit\s+width(?:\s*\(?m\)?)?"),
            "assembly_points_count": _number_near(text, r"assembly\s+points?"),
        }
    finally:
        if temporary:
            Path(temporary.name).unlink(missing_ok=True)


def load_blueprint_for_spot(spot_id: str) -> dict[str, Any]:
    result: dict[str, Any] = {"spot_id": spot_id, **FEATURE_DEFAULTS, "confidence": "none"}
    reference = REFERENCE_DIR / f"{spot_id}.json"
    if reference.exists():
        payload = json.loads(reference.read_text(encoding="utf-8"))
        result.update(payload.get("extracted_features") or {})
        result.update({key: payload.get(key) for key in ("source_url", "confidence", "extracted_at")})
    if CONFIG_PATH.exists():
        site = (json.loads(CONFIG_PATH.read_text(encoding="utf-8")).get("sites") or {}).get(spot_id, {})
        manual_map = {
            "exit_count": "manual_exit_count",
            "stairwell_count": "manual_stairwell_count",
            "floor_count": "manual_floor_count",
            "longest_corridor_m": "manual_longest_corridor_m",
            "usable_exit_width_m": "manual_usable_exit_width_m",
            "assembly_points_count": "manual_assembly_points_count",
        }
        for feature, field in manual_map.items():
            if site.get(field) is not None:
                result[feature] = float(site[field])
                result["confidence"] = "manual_override"
        result["source_url"] = site.get("blueprint_url") or site.get("campus_map_url") or result.get("source_url")
    return result


def load_blueprint_for_point(lat: float, lon: float) -> dict[str, Any]:
    if not LOCATIONS_PATH.exists():
        return {**FEATURE_DEFAULTS, "confidence": "none"}
    spots = json.loads(LOCATIONS_PATH.read_text(encoding="utf-8")).get("spots", [])
    if not spots:
        return {**FEATURE_DEFAULTS, "confidence": "none"}
    nearest = min(spots, key=lambda spot: haversine_km(lat, lon, float(spot["lat"]), float(spot["lon"])))
    result = load_blueprint_for_spot(str(nearest["id"]))
    result["distance_km"] = round(haversine_km(lat, lon, float(nearest["lat"]), float(nearest["lon"])), 3)
    return result


def reference_payload(spot_id: str, source_url: str | None, features: dict[str, Any], confidence: str) -> dict[str, Any]:
    return {
        "spot_id": spot_id,
        "source_url": source_url,
        "extracted_features": {key: float(features.get(key) or 0) for key in FEATURE_DEFAULTS},
        "confidence": confidence,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "notice": "Assistive metadata only; validate against FCUSD drills and official egress plans.",
    }
