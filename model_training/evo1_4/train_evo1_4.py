#!/usr/bin/env python3
"""Train Evo 1.4 research multi-task route and evacuation heads."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_training.evo1_3 import train_evo1_3 as v13
from services.blueprint_features import load_blueprint_for_point, load_blueprint_for_spot
from services import location_evac_analysis as route_teacher

MODEL_VERSION = "evo1.4"
SEED = 42
HEADINGS = [0, 45, 90, 135, 180, 225, 270, 315]
COMPASS = [route_teacher.COMPASS_BY_HEADING[value] for value in HEADINGS]
CATEGORIES = ["Train Station", "Office Building", "Stadium"]
NUMERIC_FEATURES = [
    "occupancy_log_scaled", "density_scaled", "egress_exit_count_scaled",
    "egress_usable_width_m_scaled", "egress_route_length_m_scaled",
    "egress_blockage_fraction_scaled", "blocked_exit_fraction_scaled",
    "hazard_point_count_scaled", "blueprint_exit_count_scaled",
    "blueprint_floor_count_scaled", "blueprint_corridor_length_scaled",
    "detour_required_flag_scaled", "route_duration_proxy_scaled", "crowd_flow_pressure_scaled",
] + [f"blocked_heading_{heading}_flag_scaled" for heading in HEADINGS]


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def load_pin_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], ["Pin analysis log is absent; teacher-derived synthetic route rows will be used."]
    rows, warnings = [], []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            warnings.append(f"Skipped malformed pin JSONL line {number}")
    return rows, warnings


def _sites(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")).get("spots", [])


def _bearing(start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> float:
    lat1, lat2 = math.radians(start_lat), math.radians(end_lat)
    delta = math.radians(end_lon - start_lon)
    value = math.degrees(math.atan2(math.sin(delta) * math.cos(lat2), math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta)))
    return value % 360


def deterministic_osrm_teacher(start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> dict[str, Any]:
    """Offline approximation injected into the existing OSRM ranking function."""
    bearing = _bearing(start_lat, start_lon, end_lat, end_lon)
    site_bias = math.sin(math.radians(bearing + start_lat * 7 + start_lon * 3))
    distance = route_teacher.DEFAULT_ASSEMBLY_DISTANCE_M * (1.12 + 0.22 * site_bias)
    return {"distance_m": round(distance, 1), "duration_min": round(distance / 72.0, 2), "geometry": None}


def _teacher_route(row: dict[str, Any]) -> dict[str, Any] | None:
    original = route_teacher._osrm_walk_route
    route_teacher._osrm_walk_route = deterministic_osrm_teacher
    try:
        ranked = route_teacher._rank_evacuation_routes(
            lat=float(row["lat"]), lon=float(row["lon"]), occupancy=int(row["occupancy"]),
            density=float(row["density"]),
            egress={
                "egress_exit_count": float(row["egress_exit_count"]),
                "egress_usable_width_m": float(row["egress_usable_width_m"]),
                "egress_blockage_fraction": float(row["egress_blockage_fraction"]),
            },
            blocked_headings=list(row["blocked_headings"]),
            blocked_points=list(row.get("blocked_points") or []),
            blockage_reason=row.get("blockage_reason"),
        )
        return ranked[0] if ranked and ranked[0].get("heading_deg") is not None else None
    finally:
        route_teacher._osrm_walk_route = original


def _egress(category: str, occupancy: float) -> tuple[float, float]:
    if category == "Stadium":
        return 12.0, 4.5
    if category == "Train Station":
        return 6.0, 3.0
    return float(max(2, min(8, math.ceil(occupancy / 400)))), 2.0


def build_route_rows(args: argparse.Namespace, pins: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = random.Random(SEED)
    sites = _sites(args.monitoring_locations)
    reference = json.loads(args.data.read_text(encoding="utf-8"))
    if isinstance(reference, dict):
        reference = reference.get("rows", [])
    usable_reference = [row for row in reference if _finite(row.get("Occupancy (#)")) > 0]
    rows: list[dict[str, Any]] = []

    def enrich(base: dict[str, Any], *, origin: str) -> None:
        blocked = sorted({int(value) % 360 for value in base.get("blocked_headings") or [] if int(value) % 45 == 0})
        blueprint = base.get("blueprint") or load_blueprint_for_point(float(base["lat"]), float(base["lon"]))
        record = {
            **base,
            "blocked_headings": blocked,
            "blocked_points": base.get("blocked_points") or [],
            "blueprint_exit_count": _finite(blueprint.get("exit_count") or base.get("blueprint_exit_count")),
            "blueprint_floor_count": _finite(blueprint.get("floor_count")),
            "blueprint_corridor_length": _finite(blueprint.get("longest_corridor_m")),
            "data_origin": origin,
        }
        teacher = _teacher_route(record)
        if not teacher:
            return
        compass = str(base.get("best_compass_heading") or base.get("recommended_compass") or teacher["compass"])
        record["heading_index"] = COMPASS.index(compass) if compass in COMPASS else HEADINGS.index(int(teacher["heading_deg"]))
        record["estimated_clear_time_min"] = _finite(base.get("estimated_clear_time_min"), float(teacher["estimated_clear_time_min"]))
        record["egress_route_length_m"] = _finite(base.get("egress_route_length_m"), float(teacher["walk_distance_m"]))
        record["success_pct"] = _finite(base.get("predicted_evacuation_success_pct"), 96.0 - float(record["egress_blockage_fraction"]) * 4.0)
        record["evacuation_time_min"] = _finite(base.get("predicted_evacuation_time_min"), float(record["estimated_clear_time_min"]) + 1.5)
        rows.append(record)

    for pin in pins:
        nearest = min(sites, key=lambda site: abs(float(site["lat"]) - _finite(pin.get("lat"))) + abs(float(site["lon"]) - _finite(pin.get("lon"))))
        occupancy = max(1, int(_finite(pin.get("occupancy"), nearest.get("default_occupancy", 200))))
        exits, width = _egress(str(pin.get("category") or nearest.get("category")), occupancy)
        enrich({
            **pin, "lat": _finite(pin.get("lat"), nearest["lat"]), "lon": _finite(pin.get("lon"), nearest["lon"]),
            "category": str(pin.get("category") or nearest.get("category") or "Office Building"),
            "occupancy": occupancy, "density": _finite(pin.get("density"), 0.35),
            "egress_exit_count": _finite(pin.get("egress_exit_count"), exits),
            "egress_usable_width_m": _finite(pin.get("egress_usable_width_m"), width),
            "egress_blockage_fraction": _finite(pin.get("egress_blockage_fraction"), 0.1),
        }, origin="observed_pin_teacher_label")

    for index in range(args.synthetic_route_rows):
        site = sites[index % len(sites)]
        ref = usable_reference[rng.randrange(len(usable_reference))]
        category = str(site.get("category") or ref.get("Category") or "Office Building")
        occupancy = max(20, int(_finite(ref.get("Occupancy (#)"), site.get("default_occupancy", 200)) * rng.uniform(0.75, 1.25)))
        density = min(0.98, max(0.05, _finite(ref.get("Density (#)"), site.get("default_density", 0.35)) + rng.uniform(-0.08, 0.08)))
        exits, width = _egress(category, occupancy)
        blocked_count = rng.choices([0, 1, 2, 3], weights=[1, 4, 3, 1])[0]
        blocked = rng.sample(HEADINGS, blocked_count)
        blockage = rng.uniform(0.1, 0.9)
        success = _finite(ref.get("Evacuation Success (%)"), 96.0) - blockage * 3.0
        evac_time = _finite(ref.get("Evacuation Time (Min)"), 8.0) * (1.0 + blockage * 0.18)
        enrich({
            "lat": float(site["lat"]), "lon": float(site["lon"]), "spot_id": site["id"],
            "category": category, "occupancy": occupancy, "density": density,
            "egress_exit_count": exits, "egress_usable_width_m": width,
            "egress_blockage_fraction": blockage, "blocked_headings": blocked,
            "blocked_points": ([{"lat": site["lat"], "lon": site["lon"], "radius_m": 30}] if rng.random() < 0.3 else []),
            "blockage_reason": rng.choice(["fire", "debris", "flood", "police_cordon"]),
            "predicted_evacuation_success_pct": success,
            "predicted_evacuation_time_min": evac_time,
            "blueprint": load_blueprint_for_spot(str(site["id"])),
        }, origin="synthetic_osrm_teacher")
    frame = pd.DataFrame(rows)
    return frame, {
        "pin_rows_loaded": len(pins), "teacher_labeled_pin_rows": sum(frame["data_origin"] == "observed_pin_teacher_label"),
        "synthetic_teacher_rows": sum(frame["data_origin"] == "synthetic_osrm_teacher"),
        "teacher": "location_evac_analysis._rank_evacuation_routes with deterministic offline OSRM geometry",
    }


def raw_features(frame: pd.DataFrame) -> np.ndarray:
    values = []
    for row in frame.to_dict(orient="records"):
        blocked = set(row["blocked_headings"])
        numeric = [
            math.log1p(max(0, row["occupancy"])), row["density"], row["egress_exit_count"],
            row["egress_usable_width_m"], row["egress_route_length_m"], row["egress_blockage_fraction"],
            len(blocked) / 8.0, len(row["blocked_points"]), row["blueprint_exit_count"],
            row["blueprint_floor_count"], row["blueprint_corridor_length"], float(bool(blocked)),
            row["egress_route_length_m"] / 72.0,
            row["occupancy"] / max(row["egress_exit_count"] * row["egress_usable_width_m"] * 45.0 * (1.0 - row["egress_blockage_fraction"]), 1.0),
            *[float(heading in blocked) for heading in HEADINGS],
        ]
        numeric.extend(float(row["category"] == category) for category in CATEGORIES)
        values.append(numeric)
    return np.asarray(values, dtype=np.float32)


class Evo14Model(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.12), nn.Linear(128, 64), nn.ReLU())
        self.success = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.evac_time = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.heading = nn.Sequential(nn.Linear(64, 48), nn.ReLU(), nn.Linear(48, 8))
        self.clear_time = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.shared(features)
        return self.success(hidden), self.evac_time(hidden), self.heading(hidden), self.clear_time(hidden)


class ExportModel(nn.Module):
    def __init__(self, model: Evo14Model, target_mean: np.ndarray, target_scale: np.ndarray):
        super().__init__()
        self.model = model
        self.register_buffer("target_mean", torch.tensor(target_mean, dtype=torch.float32))
        self.register_buffer("target_scale", torch.tensor(target_scale, dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        success, evac_time, heading, clear = self.model(features)
        regression = torch.cat((success, evac_time, clear), 1) * self.target_scale + self.target_mean
        clear_minutes = (torch.exp(regression[:, 2:3]) - 1.0).clamp_min(0.0)
        return torch.cat((regression[:, :2], heading, clear_minutes), 1)


def train(args: argparse.Namespace, frame: pd.DataFrame, audit: dict[str, Any]) -> None:
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    x_raw = raw_features(frame)
    train_idx, val_idx = train_test_split(np.arange(len(frame)), test_size=0.2, random_state=SEED, stratify=frame["heading_index"])
    scaler = StandardScaler().fit(x_raw[train_idx, :len(NUMERIC_FEATURES)])
    x = x_raw.copy(); x[:, :len(NUMERIC_FEATURES)] = scaler.transform(x[:, :len(NUMERIC_FEATURES)])
    y_reg = frame[["success_pct", "evacuation_time_min", "estimated_clear_time_min"]].to_numpy(np.float32)
    transformed_targets = y_reg.copy(); transformed_targets[:, 2] = np.log1p(transformed_targets[:, 2])
    target_scaler = StandardScaler().fit(transformed_targets[train_idx]); y_scaled = target_scaler.transform(transformed_targets).astype(np.float32)
    y_heading = frame["heading_index"].to_numpy(np.int64)
    model = Evo14Model(x.shape[1]); optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-5)
    best_state, best_loss, stale = None, float("inf"), 0
    history = {"train_loss": [], "val_loss": []}
    xt, yt, ht = torch.from_numpy(x[train_idx]), torch.from_numpy(y_scaled[train_idx]), torch.from_numpy(y_heading[train_idx])
    xv, yv, hv = torch.from_numpy(x[val_idx]), torch.from_numpy(y_scaled[val_idx]), torch.from_numpy(y_heading[val_idx])
    for _epoch in range(args.max_epochs):
        model.train(); optimizer.zero_grad(); s, t, h, c = model(xt)
        loss = nn.functional.mse_loss(s[:, 0], yt[:, 0]) + nn.functional.mse_loss(t[:, 0], yt[:, 1]) + 1.4 * nn.functional.cross_entropy(h, ht) + nn.functional.mse_loss(c[:, 0], yt[:, 2])
        loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            s, t, h, c = model(xv)
            val_loss = nn.functional.mse_loss(s[:, 0], yv[:, 0]) + nn.functional.mse_loss(t[:, 0], yv[:, 1]) + 1.4 * nn.functional.cross_entropy(h, hv) + nn.functional.mse_loss(c[:, 0], yv[:, 2])
        history["train_loss"].append(float(loss)); history["val_loss"].append(float(val_loss))
        if float(val_loss) < best_loss - 1e-4:
            best_loss, best_state, stale = float(val_loss), {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= args.patience: break
    model.load_state_dict(best_state or model.state_dict()); model.eval()
    export_model = ExportModel(model, target_scaler.mean_, target_scaler.scale_).eval()
    with torch.no_grad(): predictions = export_model(xv).numpy()
    heading_pred = predictions[:, 2:10].argmax(1)
    metrics = {
        "mae_success_pct": float(mean_absolute_error(y_reg[val_idx, 0], predictions[:, 0])),
        "r2_success_pct": float(r2_score(y_reg[val_idx, 0], predictions[:, 0])),
        "mae_time_min": float(mean_absolute_error(y_reg[val_idx, 1], predictions[:, 1])),
        "r2_time_min": float(r2_score(y_reg[val_idx, 1], predictions[:, 1])),
        "route_heading_accuracy": float(accuracy_score(y_heading[val_idx], heading_pred)),
        "clear_time_mae_min": float(mean_absolute_error(y_reg[val_idx, 2], predictions[:, 10])),
        "held_out_rows": len(val_idx),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "input_dim": x.shape[1]}, args.output_dir / "evo1.4.pt")
    onnx_path = args.output_dir / "evo1.4.onnx"
    torch.onnx.export(export_model, (xv[:4],), onnx_path, input_names=["features"], output_names=["predictions"], dynamic_axes={"features": {0: "batch"}, "predictions": {0: "batch"}}, opset_version=17, dynamo=False)
    import onnxruntime as ort
    import openvino as ov
    ort_out = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).run(None, {"features": x[val_idx[:32]].astype(np.float32)})[0]
    ov_dir = args.output_dir / "openvino"; ov_dir.mkdir(exist_ok=True)
    ov.save_model(ov.convert_model(str(onnx_path)), str(ov_dir / "evo1.4.xml"))
    compiled = ov.Core().compile_model(str(ov_dir / "evo1.4.xml"), "CPU")
    ov_out = compiled(x[val_idx[:32]].astype(np.float32))[compiled.output(0)]
    sample = x[val_idx[:1]].astype(np.float32); timings = []
    for _ in range(20): compiled(sample)
    for _ in range(200):
        started = time.perf_counter_ns(); compiled(sample); timings.append((time.perf_counter_ns() - started) / 1e6)
    export = {"onnx_parity": bool(np.allclose(predictions[:32], ort_out, rtol=.01, atol=.01)), "openvino_parity": bool(np.allclose(ort_out, ov_out, rtol=.01, atol=.01)), "openvino_ir_loaded": True, "p95_inference_ms": float(np.percentile(timings, 95))}
    gates = {
        "route_heading_accuracy_at_least_0_55": metrics["route_heading_accuracy"] >= .55,
        "clear_time_mae_at_most_2_min": metrics["clear_time_mae_min"] <= 2,
        "success_mae_under_2_pct": metrics["mae_success_pct"] < 2,
        "success_r2_at_least_0_25": metrics["r2_success_pct"] >= .25,
        "time_mae_under_1_min": metrics["mae_time_min"] < 1,
        "time_r2_at_least_0_75": metrics["r2_time_min"] >= .75,
        "onnx_parity": export["onnx_parity"], "openvino_parity": export["openvino_parity"],
        "openvino_ir_loaded": True, "p95_inference_under_10ms": export["p95_inference_ms"] < 10,
    }
    all_pass = all(gates.values())
    schema = {
        "model_version": MODEL_VERSION, "input_name": "features", "numeric_features": NUMERIC_FEATURES,
        "categorical_features": {"category": CATEGORIES},
        "feature_order": NUMERIC_FEATURES + [f"category={value}" for value in CATEGORIES],
        "normalization": {"means": scaler.mean_.tolist(), "scales": scaler.scale_.tolist()},
        "outputs": ["evacuation_success_pct", "evacuation_time_min", *[f"heading_logit_{value}" for value in COMPASS], "estimated_clear_time_min"],
        "route_head": {"headings_deg": HEADINGS, "compass_labels": COMPASS},
    }
    report = {
        "model_version": MODEL_VERSION, "status": "complete_research_demo", "selected_model": "evo1.4_multitask_mlp",
        "metrics": metrics, "quality_gates": gates, "failed_quality_gates": [k for k, v in gates.items() if not v],
        "all_quality_gates_pass": all_pass, "failure_classification": None if all_pass else "DATA_CEILING",
        "production_approved": False, "production_promotion_allowed": False,
        "promotion_recommendation": "keep_evo1.2_hybrid",
        "teacher_label_notice": "Route labels are synthetic/teacher-derived and are not FCUSD drill validation.",
        "export_validation": export, "data_audit": audit,
    }
    architecture = {"model_version": MODEL_VERSION, "name": "Multi-task MLP", "heads": ["success regression", "evacuation time regression", "8-way route heading classification", "clear time regression"], "production_approved": False}
    payloads = {"feature_schema.json": schema, "validation_report.json": report, "metrics.json": {**metrics, **history, "production_approved": False}, "learning_curves.json": history, "dashboard_curves.json": history, "architecture.json": architecture, "data_audit.json": audit}
    for name, payload in payloads.items(): (args.output_dir / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete_research_demo", "all_quality_gates_pass": all_pass, "production_approved": False, "promotion_recommendation": "keep_evo1.2_hybrid", "metrics": metrics}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data/processed/evacuation_reference.json")
    parser.add_argument("--monitoring-locations", type=Path, default=PROJECT_ROOT / "config/monitoring_locations.json")
    parser.add_argument("--peoplesense-dir", type=Path, required=True)
    parser.add_argument("--real-outcomes", type=Path, required=True)
    parser.add_argument("--pin-analyses", type=Path, required=True)
    parser.add_argument("--coords-confirmed", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/evo1.4")
    parser.add_argument("--synthetic-route-rows", type=int, default=2400)
    parser.add_argument("--max-epochs", type=int, default=180)
    parser.add_argument("--patience", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v13_args = argparse.Namespace(**vars(args), hazard_seed=PROJECT_ROOT / "data/processed/hazard_live_seed.json", max_sample_age_hours=24.0)
    people, outcomes, report = v13.validate_inputs(v13_args)
    pins, warnings = load_pin_rows(args.pin_analyses)
    report.update({"model_version": MODEL_VERSION, "pin_rows": len(pins), "warnings": warnings, "route_training_mode": "observed_plus_teacher_synthetic" if pins else "teacher_synthetic_only", "production_approved": False})
    print(json.dumps(report, indent=2))
    if report["errors"]: raise SystemExit(2)
    if args.preflight_only: return
    frame, audit = build_route_rows(args, pins)
    audit.update({"preflight": report, "real_outcomes_present": len(outcomes), "peoplesense_samples_present": len(people), "production_eligible_data": False})
    train(args, frame, audit)


if __name__ == "__main__":
    main()
