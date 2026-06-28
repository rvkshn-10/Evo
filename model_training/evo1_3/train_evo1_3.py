#!/usr/bin/env python3
"""Data-first Evo 1.3 trainer. Refuses to train without real FCUSD inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_training.evo1_2 import train_evo1_2 as v12
from model_training.shared.live_hazard_features import merge_for_training
from services.peoplesense_xml import parse_occupancy_xml


MODEL_VERSION = "evo1.3"
REQUIRED_PI_SITES = {"vista-del-lago", "folsom-high", "cordova-park"}
PEOPLESENSE_FEATURES = [
    "peoplesense_count_log_scaled",
    "peoplesense_density_scaled",
    "peoplesense_volatility_scaled",
    "peoplesense_sample_age_hours_log_scaled",
    "peoplesense_observed_flag_scaled",
]
EGRESS_FEATURES = [
    "egress_exit_count_scaled",
    "egress_usable_width_m_scaled",
    "egress_route_length_m_scaled",
    "egress_blockage_fraction_scaled",
]

# Version all reused Evo 1.2 components and its Evo 1.1 dependency.
v12.MODEL_VERSION = MODEL_VERSION
v12.v11.MODEL_VERSION = MODEL_VERSION


def finite(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def parse_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_monitoring_sites(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(spot["id"]): spot for spot in payload.get("spots", [])}


def extract_xml_payload(path: Path) -> tuple[str, datetime | None]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        xml_text = str(
            payload.get("OccupancyXML")
            or payload.get("occupancy_xml")
            or payload.get("occupancyXml")
            or ""
        )
        timestamp = parse_timestamp(
            payload.get("timestamp") or payload.get("received_at") or payload.get("generated_at")
        )
        return xml_text, timestamp
    try:
        generated_at = ET.fromstring(text).attrib.get("generated_at")
    except ET.ParseError:
        generated_at = None
    return text, parse_timestamp(generated_at)


def load_peoplesense_samples(directory: Path) -> tuple[pd.DataFrame, list[str]]:
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    if not directory.exists():
        return pd.DataFrame(), [f"PeopleSense sample directory is missing: {directory}"]
    files = sorted(
        path
        for path in [*directory.glob("*.xml"), *directory.glob("*.json")]
        if not path.name.startswith("._")
    )
    if not files:
        return pd.DataFrame(), [f"No .xml or .json OccupancyXML samples found in {directory}"]
    for path in files:
        try:
            xml_text, envelope_timestamp = extract_xml_payload(path)
            zones = parse_occupancy_xml(xml_text)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        if not zones:
            errors.append(f"{path.name}: no valid Occupancy/Zone records")
            continue
        try:
            root_timestamp = parse_timestamp(ET.fromstring(xml_text).attrib.get("generated_at"))
        except ET.ParseError:
            root_timestamp = None
        timestamp = envelope_timestamp or root_timestamp
        if timestamp is None:
            errors.append(f"{path.name}: missing timestamp/generated_at")
            continue
        for zone in zones:
            zone_id = str(zone.get("zone_id") or "").strip()
            count = finite(zone.get("occupancy_count"))
            density = finite(zone.get("occupancy_density"))
            volatility = finite(zone.get("occupancy_volatility"))
            if not zone_id or count is None or density is None or volatility is None:
                errors.append(f"{path.name}: Zone requires id, Count, Density, and Volatility")
                continue
            records.append(
                {
                    "spot_id": zone_id,
                    "peoplesense_timestamp": timestamp,
                    "peoplesense_count": count,
                    "peoplesense_density": density,
                    "peoplesense_volatility": volatility,
                    "peoplesense_source_file": path.name,
                }
            )
    return pd.DataFrame(records), errors


def load_real_outcomes(path: Path) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        return pd.DataFrame(), [f"Real drill outcome file is missing: {path}"]
    if ".template." in path.name or path.name.endswith(".template"):
        return pd.DataFrame(), [
            "The real-outcome template is documentation only; provide measured drill outcomes"
        ]
    try:
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            frame = pd.DataFrame(payload.get("rows", payload) if isinstance(payload, dict) else payload)
    except (OSError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        return pd.DataFrame(), [f"Could not read outcome file: {exc}"]
    aliases = {
        "Scenario": "scenario",
        "Category": "category",
        "Occupancy (#)": "occupancy",
        "Density (#)": "density",
        "Evacuation Success (%)": "evacuation_success_pct",
        "Evacuation Time (Min)": "evacuation_time_min",
        "timestamp": "outcome_timestamp",
    }
    frame = frame.rename(columns=aliases)
    required = {
        "spot_id",
        "outcome_timestamp",
        "scenario",
        "category",
        "evacuation_success_pct",
        "evacuation_time_min",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        return pd.DataFrame(), [f"Real outcome file is missing columns: {', '.join(missing)}"]
    errors = []
    frame["outcome_timestamp"] = frame["outcome_timestamp"].map(parse_timestamp)
    if frame["outcome_timestamp"].isna().any():
        errors.append("Every real outcome requires an ISO-8601 outcome_timestamp")
    for column in ("evacuation_success_pct", "evacuation_time_min"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any():
            errors.append(f"Every real outcome requires numeric {column}")
    non_train = frame[frame["category"].isin(["Office Building", "Stadium"])]
    if non_train.empty:
        errors.append("At least one real Office Building or Stadium outcome is required")
    return frame, errors


def validate_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sites = load_monitoring_sites(args.monitoring_locations)
    people, people_errors = load_peoplesense_samples(args.peoplesense_dir)
    outcomes, outcome_errors = load_real_outcomes(args.real_outcomes)
    errors = [*people_errors, *outcome_errors]
    missing_sites = sorted(REQUIRED_PI_SITES - set(sites))
    if missing_sites:
        errors.append(f"monitoring_locations.json is missing PeopleSense sites: {', '.join(missing_sites)}")
    invalid_coords = sorted(
        site_id
        for site_id in REQUIRED_PI_SITES & set(sites)
        if finite(sites[site_id].get("lat")) is None or finite(sites[site_id].get("lon")) is None
    )
    if invalid_coords:
        errors.append(f"PeopleSense sites have missing coordinates: {', '.join(invalid_coords)}")
    if not args.coords_confirmed:
        errors.append(
            "PeopleSense site coordinates have not been confirmed (--coords-confirmed required)"
        )
    if not people.empty:
        unknown = sorted(set(people["spot_id"]) - set(sites))
        if unknown:
            errors.append(f"PeopleSense samples reference unknown spot_id values: {', '.join(unknown)}")
        missing_sample_sites = sorted(REQUIRED_PI_SITES - set(people["spot_id"]))
        if missing_sample_sites:
            errors.append(
                "PeopleSense samples are required for all provisioned Pi sites; missing: "
                + ", ".join(missing_sample_sites)
            )
    if not outcomes.empty:
        unknown = sorted(set(outcomes["spot_id"].astype(str)) - set(sites))
        if unknown:
            errors.append(f"Outcome rows reference unknown spot_id values: {', '.join(unknown)}")
    report = {
        "status": "ready" if not errors else "blocked_missing_real_data",
        "model_version": MODEL_VERSION,
        "errors": errors,
        "peoplesense_sample_rows": len(people),
        "real_outcome_rows": len(outcomes),
        "required_pi_sites": sorted(REQUIRED_PI_SITES),
        "configured_pi_sites": sorted(REQUIRED_PI_SITES & set(sites)),
        "coords_confirmed": args.coords_confirmed,
    }
    return people, outcomes, report


def merge_people_with_outcomes(
    outcomes: pd.DataFrame,
    people: pd.DataFrame,
    *,
    max_age_hours: float,
) -> tuple[pd.DataFrame, list[str]]:
    merged = []
    errors = []
    for index, outcome in outcomes.iterrows():
        candidates = people[people["spot_id"] == str(outcome["spot_id"])].copy()
        if candidates.empty:
            errors.append(f"Outcome row {index}: no PeopleSense sample for {outcome['spot_id']}")
            continue
        candidates["age_hours"] = candidates["peoplesense_timestamp"].map(
            lambda value: abs((outcome["outcome_timestamp"] - value).total_seconds()) / 3600
        )
        sample = candidates.sort_values("age_hours").iloc[0]
        if float(sample["age_hours"]) > max_age_hours:
            errors.append(
                f"Outcome row {index}: nearest PeopleSense sample is {sample['age_hours']:.1f}h away"
            )
            continue
        item = outcome.to_dict()
        item.update(sample.to_dict())
        merged.append(item)
    return pd.DataFrame(merged), errors


def hazard_fields_for_spot(spot_id: str, seed: dict[str, Any]) -> dict[str, Any]:
    for row in seed.get("rows", []):
        if str(row.get("spot_id")) == spot_id:
            live = row.get("live_hazard") or {}
            return {
                "event_type": str(row.get("event_type") or "other"),
                "severity_score": finite(row.get("severity_score"), 0.0),
                "hazard_magnitude": finite(row.get("hazard_magnitude"), 0.0),
                "hazard_distance_km": finite(row.get("hazard_distance_km"), 250.0),
                "hazard_depth_km": finite(live.get("depth_km"), 0.0),
                "hazard_source": str(row.get("hazard_source") or "none"),
                "real_hazard_join": bool(row.get("hazard_source")),
            }
    return {
        "event_type": "other",
        "severity_score": 0.0,
        "hazard_magnitude": 0.0,
        "hazard_distance_km": 250.0,
        "hazard_depth_km": 0.0,
        "hazard_source": "none",
        "real_hazard_join": False,
    }


def real_outcome_training_rows(merged: pd.DataFrame, seed: dict[str, Any]) -> pd.DataFrame:
    records = []
    for row in merged.to_dict(orient="records"):
        identity = [row.get(key) for key in ("spot_id", "outcome_timestamp", "scenario", "category")]
        record = {
            "row_id": "real-" + hashlib.sha256(json.dumps(identity, default=str).encode()).hexdigest()[:16],
            "scenario": v12.v11.normalize_scenario(row["scenario"]),
            "category": str(row["category"]),
            "occupancy": finite(row.get("occupancy"), finite(row.get("peoplesense_count"), 0.0)),
            "density": finite(row.get("density"), finite(row.get("peoplesense_density"), 0.5)),
            "evacuation_success_pct": finite(row["evacuation_success_pct"]),
            "evacuation_time_min": finite(row["evacuation_time_min"]),
            "spot_id": str(row["spot_id"]),
            "data_origin": "real_fcusd_outcome_with_peoplesense",
            "labels_available": True,
            "metric_eligible": True,
            "synthetic_augmentation": False,
            "peoplesense_count": finite(row.get("peoplesense_count")),
            "peoplesense_density": finite(row.get("peoplesense_density")),
            "peoplesense_volatility": finite(row.get("peoplesense_volatility")),
            "peoplesense_sample_age_hours": finite(row.get("age_hours"), 0.0),
            "peoplesense_observed": True,
            "egress_exit_count": finite(row.get("egress_exit_count"), 0.0),
            "egress_usable_width_m": finite(row.get("egress_usable_width_m"), 0.0),
            "egress_route_length_m": finite(row.get("egress_route_length_m"), 0.0),
            "egress_blockage_fraction": finite(row.get("egress_blockage_fraction"), 0.0),
        }
        record.update(hazard_fields_for_spot(record["spot_id"], seed))
        records.append(record)
    return pd.DataFrame(records)


def ensure_v13_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    defaults = {
        "peoplesense_count": np.nan,
        "peoplesense_density": np.nan,
        "peoplesense_volatility": np.nan,
        "peoplesense_sample_age_hours": np.nan,
        "peoplesense_observed": False,
        "egress_exit_count": 0.0,
        "egress_usable_width_m": 0.0,
        "egress_route_length_m": 0.0,
        "egress_blockage_fraction": 0.0,
    }
    for column, default in defaults.items():
        if column not in result:
            result[column] = default
        else:
            result[column] = result[column].fillna(default) if not pd.isna(default) else result[column]
    return result


@dataclass
class Evo13Preprocessor(v12.HazardPreprocessor):
    @staticmethod
    def numeric(frame: pd.DataFrame, density_imputation: float) -> np.ndarray:
        frame = ensure_v13_features(frame)
        base = v12.HazardPreprocessor.numeric(frame, density_imputation)
        count = np.log1p(frame["peoplesense_count"].fillna(0).clip(lower=0).to_numpy(float))
        density = frame["peoplesense_density"].fillna(density_imputation).clip(0, 1).to_numpy(float)
        volatility = frame["peoplesense_volatility"].fillna(0).clip(lower=0).to_numpy(float)
        age = np.log1p(frame["peoplesense_sample_age_hours"].fillna(0).clip(lower=0).to_numpy(float))
        observed = frame["peoplesense_observed"].fillna(False).astype(float).to_numpy()
        egress = frame[
            ["egress_exit_count", "egress_usable_width_m", "egress_route_length_m", "egress_blockage_fraction"]
        ].fillna(0).to_numpy(float)
        return np.column_stack([base, count, density, volatility, age, observed, egress])

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "Evo13Preprocessor":
        frame = ensure_v13_features(frame)
        density = float(frame["density"].median()) if frame["density"].notna().any() else 0.5
        return cls(StandardScaler().fit(cls.numeric(frame, density)), density)

    @property
    def feature_order(self) -> list[str]:
        return (
            v12.LIVE_NUMERIC_FEATURES
            + PEOPLESENSE_FEATURES
            + EGRESS_FEATURES
            + [f"category={x}" for x in v12.v11.CATEGORIES]
            + [f"scenario={x}" for x in v12.v11.SCENARIOS]
            + [f"event_type={x}" for x in v12.v11.EVENT_TYPES]
            + [f"hazard_source={x}" for x in v12.HAZARD_SOURCES]
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return super().transform(ensure_v13_features(frame))

    def schema(self) -> dict[str, Any]:
        schema = super().schema()
        schema["model_version"] = MODEL_VERSION
        schema["numeric_features"] = v12.LIVE_NUMERIC_FEATURES + PEOPLESENSE_FEATURES + EGRESS_FEATURES
        schema["feature_order"] = self.feature_order
        schema["peoplesense_contract"] = {
            "transport": "POST OccupancyXML with x-api-key",
            "fields": ["Count", "Density", "Volatility", "generated_at"],
            "metrics_policy": "only rows joined to real outcomes are metric eligible",
        }
        return schema


def clean(value: Any) -> Any:
    return v12.clean(value)


def run_training(args: argparse.Namespace, people: pd.DataFrame, outcomes: pd.DataFrame) -> None:
    # Required shared helper proves the reference/live split and label policy.
    _, shared_meta = merge_for_training(include_unlabeled_live_rows=True)
    seed = v12.load_seed(args.hazard_seed)
    base, baseline_audit = v12.v11.load_reference(args.data)
    joined_baseline = ensure_v13_features(v12.assign_real_hazards(base, seed))
    matched, match_errors = merge_people_with_outcomes(
        outcomes, people, max_age_hours=args.max_sample_age_hours
    )
    if match_errors:
        raise SystemExit(json.dumps({"status": "blocked_unmatched_real_data", "errors": match_errors}, indent=2))
    real_rows = ensure_v13_features(real_outcome_training_rows(matched, seed))
    labeled = pd.concat([joined_baseline, real_rows], ignore_index=True)
    augmented = ensure_v13_features(v12.augment_labeled(labeled))
    live = ensure_v13_features(v12.live_feature_frame(seed))
    pseudo = ensure_v13_features(v12.pseudo_label_live(live, labeled))

    # Reused CV/final/export functions now instantiate the Evo 1.3 preprocessor.
    v12.HazardPreprocessor = Evo13Preprocessor
    splitter = StratifiedGroupKFold(n_splits=v12.N_FOLDS, shuffle=True, random_state=v12.SEED)
    splits = list(splitter.split(augmented, augmented["scenario"], augmented["row_id"]))
    mlp = v12.cross_validate_mlp(
        augmented, pseudo, splits, max_epochs=args.max_epochs, patience=args.patience
    )
    lgbm = v12.cross_validate_lightgbm(augmented, pseudo, splits)
    actual, observed = v12.target_arrays(augmented)
    ensemble_prediction = (mlp["oof_prediction"] + lgbm["oof_prediction"]) / 2
    ensemble = {
        "name": "mlp_lightgbm_oof_average",
        "oof_prediction": ensemble_prediction,
        "aggregate_metrics": v12.metrics(actual, ensemble_prediction, observed),
    }
    candidates = [mlp, lgbm, ensemble]
    winner = max(candidates, key=v12.candidate_score)
    baselines = v12.mean_and_knn_baselines(augmented, splits)
    final = v12.train_final_models(
        augmented, pseudo, splits[0], max_epochs=args.max_epochs, patience=args.patience
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export = v12.export_selected(winner["name"], final, args.output_dir, final["val_x"][:32])
    importance = v12.permutation_importance(final, winner["name"])
    selected = winner["aggregate_metrics"]
    mean_metrics = baselines["mean"]["aggregate_metrics"]
    knn_metrics = baselines["knn_k25_production_distance"]["aggregate_metrics"]
    gates = {
        "cv_mae_success_under_2_pct": selected["mae_success_pct"] < 2,
        "cv_r2_success_at_least_0_25": selected["r2_success_pct"] >= 0.25,
        "cv_mae_time_under_1_min": selected["mae_time_min"] < 1,
        "cv_r2_time_at_least_0_75": selected["r2_time_min"] >= 0.75,
        "beats_mean_success": v12.v11.beats(selected, mean_metrics, "success_pct"),
        "beats_mean_time": v12.v11.beats(selected, mean_metrics, "time_min"),
        "beats_knn_success": v12.v11.beats(selected, knn_metrics, "success_pct"),
        "beats_knn_time": v12.v11.beats(selected, knn_metrics, "time_min"),
        "onnx_parity": export["onnx_matches_selected_model_within_1pct"],
        "openvino_parity": export["openvino_matches_selected_model_within_1pct"],
        "openvino_ir_loaded": export["openvino_ir_loaded"],
        "p95_inference_under_10ms": export["single_sample_under_10ms_p95"],
    }
    all_pass = all(gates.values())
    best_success = max(candidate["aggregate_metrics"]["r2_success_pct"] for candidate in candidates)
    best_time = max(candidate["aggregate_metrics"]["r2_time_min"] for candidate in candidates)
    ceiling = None if all_pass else ("DATA_CEILING" if best_success < 0.25 or best_time < 0.75 else "MODEL_CEILING")
    recommendation = (
        "promote_evo1.3"
        if all_pass
        else "keep_evo1.2_hybrid"
    )
    report = {
        "model_version": MODEL_VERSION,
        "evo1.2_baseline": {"success_r2": 0.184, "time_r2": 0.636},
        "evaluation_protocol": {
            "folds": v12.N_FOLDS,
            "splitter": "StratifiedGroupKFold by scenario with row_id groups",
            "synthetic_variants_grouped": True,
            "unlabeled_live_rows_in_metrics": False,
        },
        "selected_model": winner["name"],
        "model_comparison": {candidate["name"]: clean(candidate) for candidate in candidates},
        "cross_validation": {
            "aggregate_metrics": selected,
            "per_category": v12.per_category(augmented, winner["oof_prediction"]),
        },
        "baseline_comparisons": baselines,
        "feature_importance": importance,
        "export_validation": export,
        "quality_gates": gates,
        "all_quality_gates_pass": all_pass,
        "failed_quality_gates": [name for name, passed in gates.items() if not passed],
        "failure_classification": ceiling,
        "promotion_recommendation": recommendation,
        "production_policy_unchanged": not all_pass,
        "honest_assessment": "PeopleSense features are evaluated only where timestamped real outcomes exist.",
    }
    audit = {
        **baseline_audit,
        "shared_live_merge_metadata": shared_meta,
        "real_outcome_rows": len(real_rows),
        "real_outcomes_by_category": real_rows["category"].value_counts().to_dict(),
        "peoplesense_samples": len(people),
        "peoplesense_samples_joined_to_outcomes": len(matched),
        "unlabeled_live_rows_counted_in_metrics": 0,
        "synthetic_rows": int(augmented["synthetic_augmentation"].sum()),
        "synthetic_rows_grouped": True,
        "coords_confirmed": True,
    }
    files = {
        "validation_report.json": report,
        "data_audit.json": audit,
        "feature_schema.json": final["preprocessor"].schema(),
        "architecture.json": v12.architecture(winner["name"]),
        "learning_curves.json": {"model_version": MODEL_VERSION, **mlp["learning_curves"]},
        "dashboard_curves.json": {"model_version": MODEL_VERSION, **mlp["learning_curves"]},
        "metrics.json": {
            "status": "passed" if all_pass else "failed_quality_gate",
            "model_version": MODEL_VERSION,
            "selected_model": winner["name"],
            "promotion_recommendation": recommendation,
            "train_loss": final["mlp_fit"].history["train_loss"],
            "val_loss": final["mlp_fit"].history["val_loss"],
            "val_mae_success_pct": selected["mae_success_pct"],
            "val_r2_success_pct": selected["r2_success_pct"],
            "val_mae_time_min": selected["mae_time_min"],
            "val_r2_time_min": selected["r2_time_min"],
            "all_quality_gates_pass": all_pass,
        },
    }
    for name, payload in files.items():
        (args.output_dir / name).write_text(json.dumps(clean(payload), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "all_quality_gates_pass": all_pass, "promotion_recommendation": recommendation}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data/processed/evacuation_reference.json")
    parser.add_argument("--hazard-seed", type=Path, default=PROJECT_ROOT / "data/processed/hazard_live_seed.json")
    parser.add_argument("--monitoring-locations", type=Path, default=PROJECT_ROOT / "config/monitoring_locations.json")
    parser.add_argument("--peoplesense-dir", type=Path, required=True)
    parser.add_argument("--real-outcomes", type=Path, required=True)
    parser.add_argument("--coords-confirmed", action="store_true")
    parser.add_argument("--max-sample-age-hours", type=float, default=24.0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/evo1.3")
    parser.add_argument("--max-epochs", type=int, default=220)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    people, outcomes, preflight = validate_inputs(args)
    print(json.dumps(preflight, indent=2))
    if preflight["errors"]:
        raise SystemExit(2)
    if args.preflight_only:
        return
    run_training(args, people, outcomes)


if __name__ == "__main__":
    main()
