"""Evacuation rate prediction using reference evacuation datasets."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

REFERENCE_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "evacuation_reference.json"

SCENARIO_BY_EVENT = {
    "fire": "Electrical Fire",
    "flood": "Station Flooding",
    "earthquake": "Panic Chain Reaction",
    "tsunami": "Station Flooding",
    "tornado": "Sudden Overcrowding",
    "other": "Standard Evacuation Drill",
}

CATEGORY_ALIASES = {
    "train station": "Train Station",
    "stadium": "Stadium",
    "office": "Office Building",
    "office building": "Office Building",
    "school": "Office Building",
}

# Evo 1.2 validation: time head reliable only on Train Station (Office MAE ~19 min).
EVO_TIME_CATEGORIES_DEFAULT = ("Train Station",)


class EvacuationPredictor:
    """Predict evacuation success rate and time from historical reference records."""

    def __init__(self, reference_path: Path = REFERENCE_PATH, *, use_evo: bool = False):
        self.reference_path = reference_path
        self.records = self._load_records()
        self.use_evo = use_evo
        self._evo = None
        self._evo_schema: Optional[dict[str, Any]] = None
        if use_evo:
            from services.evo_runtime import get_evo_runtime
            from services.evo_features import load_feature_schema

            self._evo = get_evo_runtime()
            self._evo_schema = load_feature_schema()

    def _load_records(self) -> list[dict[str, Any]]:
        if not self.reference_path.exists():
            logger.warning("Evacuation reference data not found at %s", self.reference_path)
            return []

        with self.reference_path.open(encoding="utf-8") as handle:
            data = json.load(handle)

        cleaned: list[dict[str, Any]] = []
        for row in data:
            try:
                cleaned.append(
                    {
                        "scenario": row.get("Scenario", ""),
                        "occupancy": float(row.get("Occupancy (#)", 0) or 0),
                        "evacuation_time_min": float(row.get("Evacuation Time (Min)", 0) or 0),
                        "category": row.get("Category", ""),
                        "density": float(row.get("Density (#)", 0) or 0),
                        "evacuation_success_pct": float(row.get("Evacuation Success (%)", 0) or 0),
                    }
                )
            except (TypeError, ValueError):
                continue
        return cleaned

    def predict_for_spot(
        self,
        *,
        spot_id: str,
        name: str,
        category: str,
        occupancy: int,
        density: float,
        event_type: str = "other",
        scenario: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        hazard: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        scenario = scenario or SCENARIO_BY_EVENT.get(event_type, "Standard Evacuation Drill")
        category = CATEGORY_ALIASES.get(category.lower(), category)
        hazard = hazard or {}

        knn = self._predict_knn(
            spot_id=spot_id,
            name=name,
            category=category,
            occupancy=occupancy,
            density=density,
            event_type=event_type,
            scenario=scenario,
            lat=lat,
            lon=lon,
        )

        if not (self.use_evo and self._evo and self._evo.is_available):
            return knn

        evo_out = self._predict_evo(
            occupancy=occupancy,
            density=density,
            category=category,
            scenario=scenario,
            event_type=hazard.get("event_type") or event_type,
            hazard=hazard,
        )
        if not evo_out:
            return knn

        time_categories = self._evo_time_categories()
        use_evo_time = category in time_categories
        hybrid = settings.EVO_HYBRID_MODE

        if hybrid:
            result = {
                **knn,
                "predicted_evacuation_success_pct": knn["predicted_evacuation_success_pct"],
                "predicted_evacuation_rate": knn["predicted_evacuation_rate"],
                "risk_level": knn["risk_level"],
                "predicted_evacuation_time_min": (
                    evo_out["predicted_evacuation_time_min"]
                    if use_evo_time
                    else knn["predicted_evacuation_time_min"]
                ),
                "model": f"{settings.EVO_MODEL_VERSION}_hybrid",
                "inference_mode": "hybrid",
                "production_approved": False,
                "evo_time_source": "evo" if use_evo_time else "knn_category_guard",
                "evo_success_source": "knn",
            }
        else:
            result = {**evo_out, "inference_mode": "evo_full", "production_approved": False}

        result.update({
            "spot_id": spot_id,
            "name": name,
            "category": category,
            "lat": lat,
            "lon": lon,
            "event_type": event_type,
            "scenario": scenario,
            "inputs": {"occupancy": occupancy, "density": density},
        })
        return result

    def _predict_knn(
        self,
        *,
        spot_id: str,
        name: str,
        category: str,
        occupancy: int,
        density: float,
        event_type: str,
        scenario: str,
        lat: Optional[float],
        lon: Optional[float],
    ) -> dict[str, Any]:
        neighbors = self._nearest_neighbors(
            occupancy=occupancy,
            density=density,
            category=category,
            scenario=scenario,
            k=25,
        )

        if not neighbors:
            return self._fallback_prediction(
                spot_id=spot_id,
                name=name,
                category=category,
                occupancy=occupancy,
                density=density,
                event_type=event_type,
                scenario=scenario,
                lat=lat,
                lon=lon,
            )

        weights = [n["weight"] for n in neighbors]
        weight_sum = sum(weights) or 1.0
        success = sum(n["record"]["evacuation_success_pct"] * n["weight"] for n in neighbors) / weight_sum
        evac_time = sum(n["record"]["evacuation_time_min"] * n["weight"] for n in neighbors) / weight_sum
        evac_rate = max(0.0, min(1.0, success / 100.0))

        risk = self._risk_band(evac_rate, density, occupancy)
        return {
            "spot_id": spot_id,
            "name": name,
            "category": category,
            "lat": lat,
            "lon": lon,
            "event_type": event_type,
            "scenario": scenario,
            "inputs": {
                "occupancy": occupancy,
                "density": density,
            },
            "predicted_evacuation_success_pct": round(success, 2),
            "predicted_evacuation_rate": round(evac_rate, 4),
            "predicted_evacuation_time_min": round(evac_time, 2),
            "risk_level": risk,
            "confidence": round(min(0.95, 0.55 + len(neighbors) / 50), 2),
            "reference_samples": len(neighbors),
            "model": "knn_reference_dataset",
        }

    def predict_for_alert(
        self,
        alert: dict[str, Any],
        spots: list[dict[str, Any]],
        occupancy_by_spot: Optional[dict[str, dict[str, Any]]] = None,
        hazard_by_spot: Optional[dict[str, dict[str, Any]]] = None,
    ) -> list[dict[str, Any]]:
        from services.evo_features import hazard_context_from_alert

        occupancy_by_spot = occupancy_by_spot or {}
        hazard_by_spot = hazard_by_spot or {}
        alert_hazard = hazard_context_from_alert(alert)
        event_type = alert.get("event_type", "other")
        predictions = []

        for spot in spots:
            spot_id = spot["id"]
            occ = occupancy_by_spot.get(spot_id, {})
            occupancy = int(occ.get("occupancy_count", spot.get("default_occupancy", 500)))
            density = float(occ.get("occupancy_density", spot.get("default_density", 0.4)))
            hazard = {**alert_hazard, **hazard_by_spot.get(spot_id, {})}

            prediction = self.predict_for_spot(
                spot_id=spot_id,
                name=spot["name"],
                category=spot.get("category", "Office Building"),
                occupancy=occupancy,
                density=density,
                event_type=event_type,
                lat=spot.get("lat"),
                lon=spot.get("lon"),
                hazard=hazard,
            )
            prediction["alert_id"] = alert.get("id")
            prediction["alert_event"] = alert.get("event")
            predictions.append(prediction)

        predictions.sort(key=lambda item: item["predicted_evacuation_rate"])
        return predictions

    def _predict_evo(
        self,
        *,
        occupancy: int,
        density: float,
        category: str,
        scenario: str,
        event_type: str,
        hazard: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        from services.evo_features import encode_features

        hazard = hazard or {}
        if self._evo_schema:
            features = encode_features(
                schema=self._evo_schema,
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
                real_hazard_join=bool(hazard.get("real_hazard_join") or hazard.get("hazard_source")),
                synthetic_augmentation=bool(hazard.get("synthetic_augmentation")),
            )
        else:
            features = [
                math.log1p(max(occupancy, 0)),
                density,
                hash(category) % 100 / 100.0,
                hash(scenario) % 100 / 100.0,
                hash(event_type) % 100 / 100.0,
            ]

        if not features:
            return None

        out = self._evo.predict(features)
        if not out:
            return None
        success = out["evacuation_success_pct"]
        evac_time = out["evacuation_time_min"]
        evac_rate = max(0.0, min(1.0, success / 100.0))
        risk = self._risk_band(evac_rate, density, occupancy)
        return {
            "predicted_evacuation_success_pct": round(success, 2),
            "predicted_evacuation_rate": round(evac_rate, 4),
            "predicted_evacuation_time_min": round(evac_time, 2),
            "risk_level": risk,
            "confidence": 0.88,
            "reference_samples": 0,
            "model": settings.EVO_MODEL_VERSION,
        }

    @staticmethod
    def _evo_time_categories() -> tuple[str, ...]:
        raw = getattr(settings, "EVO_TIME_CATEGORIES", "Train Station")
        values = tuple(part.strip() for part in str(raw).split(",") if part.strip())
        return values or EVO_TIME_CATEGORIES_DEFAULT

    def _nearest_neighbors(
        self,
        *,
        occupancy: float,
        density: float,
        category: str,
        scenario: str,
        k: int,
    ) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for record in self.records:
            distance = self._feature_distance(record, occupancy, density, category, scenario)
            weight = math.exp(-distance)
            scored.append({"record": record, "distance": distance, "weight": weight})

        scored.sort(key=lambda item: item["distance"])
        return scored[:k]

    @staticmethod
    def _feature_distance(
        record: dict[str, Any],
        occupancy: float,
        density: float,
        category: str,
        scenario: str,
    ) -> float:
        occ_delta = abs(record["occupancy"] - occupancy) / max(occupancy, 1.0)
        density_delta = abs(record["density"] - density)
        category_penalty = 0.0 if record["category"] == category else 0.35
        scenario_penalty = 0.0 if record["scenario"] == scenario else 0.2
        return occ_delta + density_delta + category_penalty + scenario_penalty

    @staticmethod
    def _risk_band(evac_rate: float, density: float, occupancy: int) -> str:
        if evac_rate < 0.9 or density > 0.85 or occupancy > 1500:
            return "high"
        if evac_rate < 0.95 or density > 0.6:
            return "medium"
        return "low"

    @staticmethod
    def _fallback_prediction(
        *,
        spot_id: str,
        name: str,
        category: str,
        occupancy: int,
        density: float,
        event_type: str,
        scenario: str,
        lat: Optional[float],
        lon: Optional[float],
    ) -> dict[str, Any]:
        base_success = 96.5 - (density * 8) - (occupancy / 2500)
        evac_time = 5.5 + (density * 2.5) + (occupancy / 900)
        evac_rate = max(0.0, min(1.0, base_success / 100.0))
        return {
            "spot_id": spot_id,
            "name": name,
            "category": category,
            "lat": lat,
            "lon": lon,
            "event_type": event_type,
            "scenario": scenario,
            "inputs": {"occupancy": occupancy, "density": density},
            "predicted_evacuation_success_pct": round(base_success, 2),
            "predicted_evacuation_rate": round(evac_rate, 4),
            "predicted_evacuation_time_min": round(evac_time, 2),
            "risk_level": "medium",
            "confidence": 0.45,
            "reference_samples": 0,
            "model": "heuristic_fallback",
        }
