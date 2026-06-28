"""Encode runtime feature vectors for Evo 1.2+ using exported feature_schema.json."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from config.settings import settings

HAZARD_SOURCES = ["none", "noaa_nws", "usgs", "gdacs", "nasa_firms", "fema_ipaws"]
NOAA_SEVERITY = {
    "extreme": 1.0,
    "severe": 0.8,
    "moderate": 0.55,
    "minor": 0.3,
}


def load_feature_schema(model_version: Optional[str] = None) -> dict[str, Any]:
    version = model_version or settings.EVO_MODEL_VERSION
    path = settings.PROJECT_ROOT / "models" / version / "feature_schema.json"
    if not path.exists():
        path = settings.PROJECT_ROOT / "artifacts" / version / "feature_schema.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def severity_from_alert(alert: dict[str, Any]) -> float:
    if alert.get("severity_score") is not None:
        return float(alert["severity_score"])
    level = str(alert.get("severity") or alert.get("alert_level") or "").lower()
    if level in NOAA_SEVERITY:
        return NOAA_SEVERITY[level]
    if level == "red":
        return 1.0
    if level == "orange":
        return 0.75
    if level == "green":
        return 0.35
    magnitude = alert.get("magnitude") or alert.get("hazard_magnitude")
    if magnitude is not None:
        return min(1.0, float(magnitude) / 8.0)
    return 0.35


def hazard_context_from_alert(alert: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_type": alert.get("event_type") or "other",
        "severity_score": severity_from_alert(alert),
        "hazard_magnitude": float(alert.get("magnitude") or alert.get("hazard_magnitude") or alert.get("frp") or 0.0),
        "hazard_distance_km": float(alert.get("hazard_distance_km") or 250.0),
        "hazard_depth_km": float(alert.get("depth_km") or 0.0),
        "hazard_source": str(alert.get("source") or "none"),
        "real_hazard_join": bool(alert.get("source")),
        "synthetic_augmentation": False,
    }


def encode_features(
    *,
    schema: dict[str, Any],
    occupancy: int,
    density: float,
    category: str,
    scenario: str,
    event_type: str = "other",
    severity_score: float = 0.0,
    hazard_magnitude: float = 0.0,
    hazard_distance_km: float = 250.0,
    hazard_depth_km: float = 0.0,
    hazard_source: str = "none",
    real_hazard_join: bool = False,
    synthetic_augmentation: bool = False,
    peoplesense_count: Optional[float] = None,
    peoplesense_density: Optional[float] = None,
    peoplesense_volatility: float = 0.0,
    peoplesense_sample_age_hours: float = 0.0,
    peoplesense_observed: bool = True,
    egress_exit_count: float = 0.0,
    egress_usable_width_m: float = 0.0,
    egress_route_length_m: float = 0.0,
    egress_blockage_fraction: float = 0.0,
) -> list[float]:
    if not schema:
        return []

    norm = schema.get("normalization") or {}
    means = norm.get("means") or []
    scales = norm.get("scales") or [1.0] * len(means)
    density_default = float(norm.get("density_missing_value") or 0.5)
    capacity_map = norm.get("category_capacity") or {
        "Train Station": 2000.0,
        "Office Building": 1000.0,
        "Stadium": 60000.0,
    }

    occ = max(float(occupancy), 0.0)
    occupancy_log = math.log1p(occ)
    den = max(0.0, min(1.0, float(density if density is not None else density_default)))
    severity = max(0.0, min(1.0, float(severity_score)))
    magnitude_log = math.log1p(max(0.0, float(hazard_magnitude)))
    distance_log = math.log1p(max(0.0, float(hazard_distance_km)))
    depth = max(0.0, float(hazard_depth_km))
    capacity = float(capacity_map.get(category, 2000.0))
    utilization = min(4.0, occ / capacity)
    interaction = occupancy_log * den
    real_flag = 1.0 if real_hazard_join else 0.0
    synthetic_flag = 1.0 if synthetic_augmentation else 0.0

    raw_numeric = [
        occupancy_log,
        den,
        severity,
        magnitude_log,
        distance_log,
        depth,
        interaction,
        utilization,
        real_flag,
        synthetic_flag,
    ]
    requested_numeric = schema.get("numeric_features") or []
    if len(requested_numeric) > len(raw_numeric):
        ps_count = occ if peoplesense_count is None else max(float(peoplesense_count), 0.0)
        ps_density = den if peoplesense_density is None else max(0.0, min(1.0, float(peoplesense_density)))
        raw_numeric.extend(
            [
                math.log1p(ps_count),
                ps_density,
                max(0.0, float(peoplesense_volatility)),
                math.log1p(max(0.0, float(peoplesense_sample_age_hours))),
                1.0 if peoplesense_observed else 0.0,
                max(0.0, float(egress_exit_count)),
                max(0.0, float(egress_usable_width_m)),
                max(0.0, float(egress_route_length_m)),
                max(0.0, min(1.0, float(egress_blockage_fraction))),
            ]
        )

    numeric_scaled = []
    for index, value in enumerate(raw_numeric):
        mean = means[index] if index < len(means) else 0.0
        scale = scales[index] if index < len(scales) and scales[index] else 1.0
        numeric_scaled.append((value - mean) / scale)

    categoricals = schema.get("categorical_features") or {}
    one_hot: list[float] = []
    for column, values in (
        ("category", categoricals.get("category", [])),
        ("scenario", categoricals.get("scenario", [])),
        ("event_type", categoricals.get("event_type", [])),
        ("hazard_source", categoricals.get("hazard_source", HAZARD_SOURCES)),
    ):
        selected = {
            "category": category,
            "scenario": scenario,
            "event_type": event_type,
            "hazard_source": hazard_source if hazard_source in HAZARD_SOURCES else "none",
        }[column]
        one_hot.extend([1.0 if selected == value else 0.0 for value in values])

    return [float(x) for x in numeric_scaled + one_hot]
