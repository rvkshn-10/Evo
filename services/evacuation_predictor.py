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


class EvacuationPredictor:
    """Predict evacuation success rate and time from historical reference records."""

    def __init__(self, reference_path: Path = REFERENCE_PATH, *, use_evo: bool = False):
        self.reference_path = reference_path
        self.records = self._load_records()
        self.use_evo = use_evo
        self._evo = None
        if use_evo:
            from services.evo_runtime import get_evo_runtime

            self._evo = get_evo_runtime()

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
    ) -> dict[str, Any]:
        scenario = scenario or SCENARIO_BY_EVENT.get(event_type, "Standard Evacuation Drill")
        category = CATEGORY_ALIASES.get(category.lower(), category)

        if self.use_evo and self._evo and self._evo.is_available:
            evo_out = self._predict_evo(
                occupancy=occupancy,
                density=density,
                category=category,
                scenario=scenario,
                event_type=event_type,
            )
            if evo_out:
                evo_out.update({
                    "spot_id": spot_id,
                    "name": name,
                    "category": category,
                    "lat": lat,
                    "lon": lon,
                    "event_type": event_type,
                    "scenario": scenario,
                    "inputs": {"occupancy": occupancy, "density": density},
                })
                return evo_out

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
    ) -> list[dict[str, Any]]:
        occupancy_by_spot = occupancy_by_spot or {}
        event_type = alert.get("event_type", "other")
        predictions = []

        for spot in spots:
            spot_id = spot["id"]
            occ = occupancy_by_spot.get(spot_id, {})
            occupancy = int(occ.get("occupancy_count", spot.get("default_occupancy", 500)))
            density = float(occ.get("occupancy_density", spot.get("default_density", 0.4)))

            prediction = self.predict_for_spot(
                spot_id=spot_id,
                name=spot["name"],
                category=spot.get("category", "Office Building"),
                occupancy=occupancy,
                density=density,
                event_type=event_type,
                lat=spot.get("lat"),
                lon=spot.get("lon"),
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
    ) -> Optional[dict[str, Any]]:
        import math

        features = [
            math.log1p(max(occupancy, 0)),
            density,
            hash(category) % 100 / 100.0,
            hash(scenario) % 100 / 100.0,
            hash(event_type) % 100 / 100.0,
        ]
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
            "model": settings.EVO_MODEL_VERSION if hasattr(settings, "EVO_MODEL_VERSION") else "evo1.0",
        }

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
