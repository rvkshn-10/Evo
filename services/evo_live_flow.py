"""Build live Evo inference trace for the interactive network visualization."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from config.settings import settings
from services.alert_processor import AlertProcessor
from services.evo_features import encode_features, hazard_context_from_alert, load_feature_schema
from services.evo_runtime import get_evo_runtime
from services.evacuation_predictor import CATEGORY_ALIASES, SCENARIO_BY_EVENT

NUMERIC_LABELS = {
    "occupancy_log_scaled": "Occupancy (log-scaled)",
    "density_scaled": "Crowd density",
    "severity_score_scaled": "Hazard severity",
    "hazard_magnitude_log_scaled": "Hazard magnitude (log)",
    "hazard_distance_log_scaled": "Distance to hazard (log km)",
    "hazard_depth_km_scaled": "Earthquake depth",
    "occupancy_density_interaction_scaled": "Occupancy × density",
    "capacity_utilization_scaled": "Venue capacity use",
    "real_hazard_join_flag_scaled": "Live hazard join",
    "synthetic_augmentation_flag_scaled": "Synthetic augment flag",
}

FEED_FOR_FEATURE = {
    "occupancy_log_scaled": "peoplesense",
    "density_scaled": "peoplesense",
    "occupancy_density_interaction_scaled": "peoplesense",
    "capacity_utilization_scaled": "peoplesense",
    "severity_score_scaled": "noaa",
    "hazard_magnitude_log_scaled": "usgs",
    "hazard_distance_log_scaled": "hazard",
    "hazard_depth_km_scaled": "usgs",
    "real_hazard_join_flag_scaled": "hazard",
    "synthetic_augmentation_flag_scaled": "model",
}


def _pick_spot_and_prediction(snapshot: dict[str, Any], spot_id: Optional[str]) -> tuple[dict, dict, dict]:
    spots = {s["id"]: s for s in snapshot.get("monitoring_spots") or []}
    candidates: list[tuple[dict, dict, dict]] = []

    for alert in (snapshot.get("alerts") or []) + (snapshot.get("earthquakes") or []):
        for pred in alert.get("evacuation_predictions") or []:
            sid = str(pred.get("spot_id") or "")
            spot = spots.get(sid) or {"id": sid, "name": pred.get("name"), "category": pred.get("category")}
            candidates.append((spot, alert, pred))

    if spot_id:
        for spot, alert, pred in candidates:
            if spot.get("id") == spot_id:
                return spot, alert, pred

    if candidates:
        return candidates[0]

    spot = next(iter(spots.values()), {"id": "folsom-high", "name": "Monitoring spot", "category": "Office Building"})
    return spot, {}, {}


def build_live_flow(*, use_evo: bool = True, spot_id: Optional[str] = None) -> dict[str, Any]:
    processor = AlertProcessor(use_evo=use_evo)
    snapshot = processor.get_dashboard_snapshot()
    spot, alert, prediction = _pick_spot_and_prediction(snapshot, spot_id)

    event_type = alert.get("event_type") or prediction.get("event_type") or "other"
    category = CATEGORY_ALIASES.get(str(spot.get("category", "")).lower(), spot.get("category", "Office Building"))
    scenario = prediction.get("scenario") or SCENARIO_BY_EVENT.get(event_type, "Standard Evacuation Drill")
    inputs = prediction.get("inputs") or {}
    occupancy = int(inputs.get("occupancy") or spot.get("default_occupancy") or 500)
    density = float(inputs.get("density") or spot.get("default_density") or 0.4)

    hazard = hazard_context_from_alert(alert) if alert else {
        "event_type": event_type,
        "severity_score": 0.0,
        "hazard_magnitude": 0.0,
        "hazard_distance_km": 250.0,
        "hazard_depth_km": 0.0,
        "hazard_source": "none",
        "real_hazard_join": False,
        "synthetic_augmentation": False,
    }

    schema = load_feature_schema()
    numeric_keys = schema.get("numeric_features") or list(NUMERIC_LABELS.keys())
    vector = encode_features(
        schema=schema,
        occupancy=occupancy,
        density=density,
        category=category,
        scenario=scenario,
        event_type=str(hazard.get("event_type") or event_type),
        severity_score=float(hazard.get("severity_score") or 0.0),
        hazard_magnitude=float(hazard.get("hazard_magnitude") or 0.0),
        hazard_distance_km=float(hazard.get("hazard_distance_km") or 250.0),
        hazard_depth_km=float(hazard.get("hazard_depth_km") or 0.0),
        hazard_source=str(hazard.get("hazard_source") or "none"),
        real_hazard_join=bool(hazard.get("real_hazard_join")),
        synthetic_augmentation=bool(hazard.get("synthetic_augmentation")),
    )

    runtime = get_evo_runtime()
    inference_ms = None
    if use_evo and runtime.is_available and vector:
        import time

        start = time.perf_counter()
        runtime.predict(vector)
        inference_ms = round((time.perf_counter() - start) * 1000, 2)

    alert_title = alert.get("headline") or alert.get("title") or alert.get("event") or "No active hazard"
    feeds = [
        {
            "id": "noaa",
            "label": "NOAA / NWS",
            "status": "live" if snapshot.get("alerts") else "idle",
            "detail": f"{len(snapshot.get('alerts') or [])} weather alerts",
        },
        {
            "id": "usgs",
            "label": "USGS",
            "status": "live" if snapshot.get("earthquakes") else "idle",
            "detail": f"{len(snapshot.get('earthquakes') or [])} earthquakes",
        },
        {
            "id": "peoplesense",
            "label": "PeopleSense",
            "status": snapshot.get("peoplesense_mode", "placeholder"),
            "detail": (
                f"{occupancy:,} occupants @ {spot.get('name', 'spot')}"
                if snapshot.get("peoplesense_mode") == "live"
                else "Placeholder occupancy"
            ),
        },
        {
            "id": "gdacs",
            "label": "GDACS / FIRMS",
            "status": "live" if (snapshot.get("gdacs_events") or snapshot.get("wildfire_hotspots")) else "idle",
            "detail": f"{len(snapshot.get('gdacs_events') or [])} GDACS · {len(snapshot.get('wildfire_hotspots') or [])} hotspots",
        },
    ]

    raw_numeric = _raw_numeric_values(occupancy, density, hazard, category)
    features = []
    for index, key in enumerate(numeric_keys):
        scaled = vector[index] if index < len(vector) else 0.0
        raw = raw_numeric.get(key, {})
        features.append(
            {
                "key": key,
                "label": NUMERIC_LABELS.get(key, key),
                "scaled": round(float(scaled), 4),
                "raw_display": raw.get("display", "—"),
                "source": FEED_FOR_FEATURE.get(key, "model"),
            }
        )

    categoricals = []
    for column, value in (
        ("category", category),
        ("scenario", scenario),
        ("event_type", str(hazard.get("event_type") or event_type)),
        ("hazard_source", str(hazard.get("hazard_source") or "none")),
    ):
        categoricals.append({"field": column, "value": value})

    edges = _build_edges(feeds, features, categoricals, alert_title, prediction)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": settings.EVO_MODEL_VERSION,
        "backend": runtime.get_runtime_status().get("backend"),
        "openvino_connected": runtime.get_runtime_status().get("openvino_connected"),
        "hybrid_mode": settings.EVO_HYBRID_MODE,
        "inference_ms": inference_ms,
        "spot": {
            "id": spot.get("id"),
            "name": spot.get("name"),
            "category": category,
        },
        "alert": {
            "title": alert_title,
            "event_type": event_type,
        },
        "feeds": feeds,
        "features": features,
        "categoricals": categoricals,
        "vector_dim": len(vector),
        "prediction": {
            "success_pct": prediction.get("predicted_evacuation_success_pct"),
            "time_min": prediction.get("predicted_evacuation_time_min"),
            "risk_level": prediction.get("risk_level"),
            "model": prediction.get("model"),
            "inference_mode": prediction.get("inference_mode"),
        },
        "edges": edges,
        "pipeline_status": _pipeline_status(snapshot, prediction, runtime.is_available),
    }


def _raw_numeric_values(
    occupancy: int,
    density: float,
    hazard: dict[str, Any],
    category: str,
) -> dict[str, dict[str, Any]]:
    import math

    schema = load_feature_schema()
    capacity_map = (schema.get("normalization") or {}).get("category_capacity") or {}
    capacity = float(capacity_map.get(category, 2000.0))
    return {
        "occupancy_log_scaled": {"display": f"{occupancy:,} people"},
        "density_scaled": {"display": f"{density:.2f} density index"},
        "severity_score_scaled": {"display": f"{float(hazard.get('severity_score', 0)):.2f} severity"},
        "hazard_magnitude_log_scaled": {
            "display": f"Magnitude / intensity {float(hazard.get('hazard_magnitude', 0)):.2f}"
        },
        "hazard_distance_log_scaled": {
            "display": f"{float(hazard.get('hazard_distance_km', 250)):.1f} km to hazard"
        },
        "hazard_depth_km_scaled": {"display": f"{float(hazard.get('hazard_depth_km', 0)):.1f} km depth"},
        "occupancy_density_interaction_scaled": {
            "display": f"log(occ)×density = {math.log1p(occupancy) * density:.2f}"
        },
        "capacity_utilization_scaled": {
            "display": f"{occupancy / capacity * 100:.1f}% of {category} capacity"
        },
        "real_hazard_join_flag_scaled": {
            "display": "Joined to live hazard" if hazard.get("real_hazard_join") else "No nearby hazard join"
        },
        "synthetic_augmentation_flag_scaled": {"display": "Live row (not synthetic)"},
    }


def _build_edges(
    feeds: list[dict],
    features: list[dict],
    categoricals: list[dict],
    alert_title: str,
    prediction: dict,
) -> list[dict[str, Any]]:
    feed_map = {f["id"]: f for f in feeds}
    edges: list[dict[str, Any]] = []

    for feature in features:
        source_id = feature.get("source") or "model"
        if source_id == "hazard":
            source_id = "usgs" if "earthquake" in alert_title.lower() else "noaa"
        feed = feed_map.get(source_id) or feed_map.get("noaa")
        edges.append(
            {
                "from": feed["id"],
                "to": feature["key"],
                "label": feature["label"],
                "detail": f"{feed['label']} → {feature['raw_display']}",
                "value": feature["scaled"],
            }
        )
        edges.append(
            {
                "from": feature["key"],
                "to": "encoder",
                "label": feature["label"],
                "detail": f"Scaled {feature['scaled']:.3f} · {feature['raw_display']}",
                "value": abs(feature["scaled"]),
            }
        )

    cat_text = ", ".join(f"{c['field']}={c['value']}" for c in categoricals)
    edges.append(
        {
            "from": "encoder",
            "to": "mlp",
            "label": "MLP tensor",
            "detail": f"34-dim encoded vector · {cat_text}",
            "value": 1.0,
        }
    )
    edges.append(
        {
            "from": "encoder",
            "to": "lgbm",
            "label": "Tree tensor",
            "detail": f"Same features → gradient boosting heads · {cat_text}",
            "value": 1.0,
        }
    )
    edges.append(
        {
            "from": "mlp",
            "to": "ensemble",
            "label": "MLP heads",
            "detail": "Dual-head MLP: success % + evacuation time",
            "value": 0.5,
        }
    )
    edges.append(
        {
            "from": "lgbm",
            "to": "ensemble",
            "label": "LGBM heads",
            "detail": "LightGBM regression heads (OOF ensemble)",
            "value": 0.5,
        }
    )
    success = prediction.get("predicted_evacuation_success_pct")
    time_min = prediction.get("predicted_evacuation_time_min")
    mode = prediction.get("inference_mode") or "hybrid"
    edges.append(
        {
            "from": "ensemble",
            "to": "success",
            "label": "Success output",
            "detail": f"{mode}: {success}% evac success" if success is not None else "Success head",
            "value": (float(success) / 100.0) if success is not None else 0.5,
        }
    )
    edges.append(
        {
            "from": "ensemble",
            "to": "time",
            "label": "Time output",
            "detail": f"{mode}: {time_min} min evacuation time" if time_min is not None else "Time head",
            "value": min(1.0, float(time_min or 0) / 20.0),
        }
    )
    return edges


def _pipeline_status(snapshot: dict, prediction: dict, model_available: bool) -> list[dict[str, str]]:
    return [
        {"step": "ingest", "label": "Ingest feeds", "status": "complete"},
        {
            "step": "occupancy",
            "label": "PeopleSense overlay",
            "status": "live" if snapshot.get("peoplesense_mode") == "live" else "simulated",
        },
        {"step": "encode", "label": "Feature encode", "status": "complete"},
        {
            "step": "infer",
            "label": "Evo inference",
            "status": "active" if model_available else "knn_fallback",
        },
        {
            "step": "hybrid",
            "label": "Hybrid merge",
            "status": prediction.get("inference_mode") or "knn",
        },
    ]
