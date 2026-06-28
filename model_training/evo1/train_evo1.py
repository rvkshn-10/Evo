#!/usr/bin/env python3
"""Train, export, and validate the Evo 1.0 evacuation regression model.

The exported model accepts one preprocessed float32 tensor named ``features``
and returns ``[evacuation_success_pct, evacuation_time_min]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


MODEL_VERSION = "evo1.0"
SEED = 42

CATEGORIES = ["Train Station", "Office Building", "Stadium"]
SCENARIOS = [
    "Electrical Fire",
    "Tunnel Smoke",
    "Sudden Overcrowding",
    "Panic Chain Reaction",
    "Platform Fire",
    "Station Flooding",
    "Standard Evacuation Drill",
    "Blockage",
]
EVENT_TYPES = ["fire", "flood", "earthquake", "tornado", "tsunami", "other"]

SCENARIO_BY_EVENT = {
    "fire": "Electrical Fire",
    "flood": "Station Flooding",
    "earthquake": "Panic Chain Reaction",
    "tsunami": "Station Flooding",
    "tornado": "Sudden Overcrowding",
    "other": "Standard Evacuation Drill",
}

# Source-only scenarios are normalized into the deployment vocabulary. Rows for
# Station Flooding are expanded across flood and tsunami so both event one-hot
# columns are represented during training.
EVENTS_BY_SCENARIO = {
    "Electrical Fire": ["fire"],
    "Tunnel Smoke": ["fire"],
    "Sudden Overcrowding": ["tornado"],
    "Panic Chain Reaction": ["earthquake"],
    "Platform Fire": ["fire"],
    "Station Flooding": ["flood", "tsunami"],
    "Standard Evacuation Drill": ["other"],
    "Blockage": ["other"],
}

NUMERIC_FEATURES = ["occupancy_log", "density", "severity_score", "hazard_magnitude"]
OUTPUTS = ["evacuation_success_pct", "evacuation_time_min"]


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def normalize_scenario(value: str) -> str:
    value = str(value).strip()
    if value == "Overcrowding":
        return "Sudden Overcrowding"
    if value.startswith("Fire Drill"):
        return "Standard Evacuation Drill"
    if value not in SCENARIOS:
        raise ValueError(f"Unsupported scenario after normalization: {value!r}")
    return value


def stable_row_id(row: dict[str, Any]) -> str:
    fields = [
        row.get("Scenario"),
        row.get("Occupancy (#)"),
        row.get("Evacuation Time (Min)"),
        row.get("Category"),
        row.get("Density (#)"),
        row.get("Evacuation Success (%)"),
    ]
    payload = json.dumps(fields, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def load_and_audit(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_rows = json.loads(path.read_text(encoding="utf-8"))
    deduplicated: dict[str, dict[str, Any]] = {}
    for row in raw_rows:
        deduplicated.setdefault(stable_row_id(row), row)

    records: list[dict[str, Any]] = []
    for row_id, row in deduplicated.items():
        category = str(row.get("Category", "")).strip()
        if category not in CATEGORIES:
            raise ValueError(f"Unsupported category: {category!r}")
        records.append(
            {
                "row_id": row_id,
                "scenario": normalize_scenario(row.get("Scenario", "")),
                "category": category,
                "occupancy": finite_or_none(row.get("Occupancy (#)")),
                "density": finite_or_none(row.get("Density (#)")),
                "evacuation_success_pct": finite_or_none(row.get("Evacuation Success (%)")),
                "evacuation_time_min": finite_or_none(row.get("Evacuation Time (Min)")),
            }
        )

    frame = pd.DataFrame.from_records(records)
    if frame["occupancy"].isna().any() or frame["evacuation_time_min"].isna().any():
        raise ValueError("Occupancy and evacuation time must be present on every unique row")

    audit = {
        "raw_rows": len(raw_rows),
        "unique_rows": len(frame),
        "exact_duplicate_rows_removed": len(raw_rows) - len(frame),
        "missing_density_rows": int(frame["density"].isna().sum()),
        "missing_success_target_rows": int(frame["evacuation_success_pct"].isna().sum()),
        "category_counts_unique": frame["category"].value_counts().sort_index().to_dict(),
        "scenario_counts_unique_normalized": frame["scenario"].value_counts().sort_index().to_dict(),
        "limitations": [
            "Office Building rows have no density or success target; density is imputed and only time loss is used.",
            "Stadium rows have no success target; only time loss is used.",
            "Severity and hazard magnitude are absent from the reference data, so Evo 1.0 is trained to treat them as neutral inputs.",
        ],
    }
    return frame, audit


def expand_event_types(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in frame.to_dict(orient="records"):
        for event_type in EVENTS_BY_SCENARIO[record["scenario"]]:
            expanded = dict(record)
            expanded["event_type"] = event_type
            expanded["severity_score"] = 0.0
            expanded["hazard_magnitude"] = 0.0
            rows.append(expanded)
    return pd.DataFrame.from_records(rows)


@dataclass
class Preprocessor:
    occupancy_scaler: StandardScaler
    density_imputation: float

    @classmethod
    def fit(cls, train: pd.DataFrame) -> "Preprocessor":
        occupancy_log = np.log1p(train[["occupancy"]].to_numpy(dtype=np.float64))
        density_imputation = float(train["density"].median())
        return cls(StandardScaler().fit(occupancy_log), density_imputation)

    @property
    def feature_order(self) -> list[str]:
        return (
            NUMERIC_FEATURES
            + [f"category={item}" for item in CATEGORIES]
            + [f"scenario={item}" for item in SCENARIOS]
            + [f"event_type={item}" for item in EVENT_TYPES]
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        occupancy_log = np.log1p(frame[["occupancy"]].to_numpy(dtype=np.float64))
        occupancy_scaled = self.occupancy_scaler.transform(occupancy_log).reshape(-1)
        density = frame["density"].fillna(self.density_imputation).clip(0.0, 1.0).to_numpy()
        numeric = np.column_stack(
            [
                occupancy_scaled,
                density,
                frame["severity_score"].clip(0.0, 1.0).to_numpy(),
                frame["hazard_magnitude"].clip(lower=0.0).to_numpy(),
            ]
        )
        categorical_parts = []
        for column, vocabulary in (
            ("category", CATEGORIES),
            ("scenario", SCENARIOS),
            ("event_type", EVENT_TYPES),
        ):
            values = frame[column].to_numpy()
            categorical_parts.append(np.column_stack([values == item for item in vocabulary]))
        return np.column_stack([numeric, *categorical_parts]).astype(np.float32)

    def schema(self) -> dict[str, Any]:
        return {
            "model_version": MODEL_VERSION,
            "input_name": "features",
            "input_dtype": "float32",
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": {
                "category": CATEGORIES,
                "scenario": SCENARIOS,
                "event_type": EVENT_TYPES,
            },
            "feature_order": self.feature_order,
            "normalization": {
                "occupancy_log": {
                    "transform": "log1p_then_standardize",
                    "mean": float(self.occupancy_scaler.mean_[0]),
                    "scale": float(self.occupancy_scaler.scale_[0]),
                },
                "density": {
                    "transform": "clip_0_1",
                    "missing_value": self.density_imputation,
                },
                "severity_score": {"transform": "clip_0_1", "default": 0.0},
                "hazard_magnitude": {"transform": "clip_min_0", "default": 0.0},
            },
            "scenario_by_event": SCENARIO_BY_EVENT,
            "outputs": OUTPUTS,
            "derived_outputs": {
                "evacuation_rate": "clip(evacuation_success_pct / 100, 0, 1)",
                "risk_level": {
                    "high": "evacuation_rate < 0.85 or density > 0.7",
                    "medium": "evacuation_rate < 0.92",
                    "low": "otherwise",
                },
            },
        }


class EvoMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        # The reference set has no severity or magnitude labels. These columns
        # are deliberately neutral in Evo 1.0 instead of retaining arbitrary
        # random initialization that could affect deployment predictions.
        with torch.no_grad():
            first_layer = self.network[0]
            assert isinstance(first_layer, nn.Linear)
            first_layer.weight[:, 2:4] = 0.0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class ExportModel(nn.Module):
    """Wrap a standardized-target model so deployment receives physical units."""

    def __init__(self, model: EvoMLP, target_mean: np.ndarray, target_scale: np.ndarray):
        super().__init__()
        self.model = model
        self.register_buffer("target_mean", torch.tensor(target_mean, dtype=torch.float32))
        self.register_buffer("target_scale", torch.tensor(target_scale, dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        physical = self.model(features) * self.target_scale + self.target_mean
        success = physical[:, 0].clamp(0.0, 100.0)
        evacuation_time = physical[:, 1].clamp_min(0.01)
        return torch.stack((success, evacuation_time), dim=1)


def make_targets(frame: pd.DataFrame, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    targets = frame[OUTPUTS].to_numpy(dtype=np.float32)
    observed = np.isfinite(targets)
    target_mean = np.zeros(2, dtype=np.float32)
    target_scale = np.ones(2, dtype=np.float32)
    for index in range(2):
        values = targets[train_mask & observed[:, index], index]
        target_mean[index] = values.mean()
        target_scale[index] = max(values.std(), 1e-6)
    standardized = np.where(observed, (targets - target_mean) / target_scale, 0.0)
    return standardized.astype(np.float32), observed.astype(np.float32), target_mean, target_scale


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    squared = (prediction - target).pow(2) * mask
    per_output = squared.sum(dim=0) / mask.sum(dim=0).clamp_min(1.0)
    return per_output.mean()


def evaluate(prediction: np.ndarray, target: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for index, short_name in enumerate(("success_pct", "time_min")):
        mask = observed[:, index].astype(bool)
        metrics[f"mae_{short_name}"] = float(mean_absolute_error(target[mask, index], prediction[mask, index]))
        metrics[f"r2_{short_name}"] = float(r2_score(target[mask, index], prediction[mask, index]))
        metrics[f"samples_{short_name}"] = int(mask.sum())
    return metrics


def train_model(
    features: np.ndarray,
    targets: np.ndarray,
    observed: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
) -> EvoMLP:
    model = EvoMLP(features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_data = TensorDataset(
        torch.from_numpy(features[train_mask]),
        torch.from_numpy(targets[train_mask]),
        torch.from_numpy(observed[train_mask]),
    )
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator)
    val_x = torch.from_numpy(features[val_mask])
    val_y = torch.from_numpy(targets[val_mask])
    val_observed = torch.from_numpy(observed[val_mask])

    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    patience = 50
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y, batch_observed in loader:
            augmented = batch_x.clone()
            augmented[:, 0] += torch.randn_like(augmented[:, 0]) * 0.02
            augmented[:, 1] = (augmented[:, 1] + torch.randn_like(augmented[:, 1]) * 0.01).clamp(0.0, 1.0)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mse(model(augmented), batch_y, batch_observed)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(masked_mse(model(val_x), val_y, val_observed))
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"epoch={epoch:03d} val_standardized_mse={val_loss:.6f}")
        if stale_epochs >= patience:
            print(f"early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model


def export_and_validate(
    model: ExportModel,
    sample: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    onnx_path = output_dir / f"{MODEL_VERSION}.onnx"
    xml_path = output_dir / f"{MODEL_VERSION}.xml"
    sample_tensor = torch.from_numpy(sample.astype(np.float32))
    with torch.no_grad():
        torch_output = model(sample_tensor).numpy()

    torch.onnx.export(
        model,
        (sample_tensor,),
        onnx_path,
        input_names=["features"],
        output_names=["predictions"],
        dynamic_axes={"features": {0: "batch"}, "predictions": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    import onnx
    import onnxruntime as ort
    import openvino as ov

    onnx.checker.check_model(onnx.load(str(onnx_path)))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(["predictions"], {"features": sample.astype(np.float32)})[0]
    onnx_close = bool(np.allclose(torch_output, onnx_output, rtol=0.01, atol=0.01))

    ov_model = ov.convert_model(str(onnx_path))
    ov.save_model(ov_model, str(xml_path))
    core = ov.Core()
    loaded = core.read_model(str(xml_path))
    compiled = core.compile_model(loaded, "CPU")
    infer = compiled.create_infer_request()
    ov_output = infer.infer({"features": sample.astype(np.float32)})["predictions"]
    openvino_close = bool(np.allclose(torch_output, ov_output, rtol=0.01, atol=0.01))

    single = sample[:1].astype(np.float32)
    for _ in range(20):
        infer.infer({"features": single})
    timings = []
    for _ in range(200):
        started = time.perf_counter_ns()
        infer.infer({"features": single})
        timings.append((time.perf_counter_ns() - started) / 1_000_000)

    return {
        "onnx_matches_pytorch_within_1pct": onnx_close,
        "onnx_max_abs_difference": float(np.max(np.abs(torch_output - onnx_output))),
        "openvino_matches_pytorch_within_1pct": openvino_close,
        "openvino_max_abs_difference": float(np.max(np.abs(torch_output - ov_output))),
        "openvino_ir_loaded": True,
        "single_sample_latency_ms_p50": float(np.percentile(timings, 50)),
        "single_sample_latency_ms_p95": float(np.percentile(timings, 95)),
        "single_sample_under_10ms_p50": bool(np.percentile(timings, 50) < 10.0),
        "single_sample_under_10ms_p95": bool(np.percentile(timings, 95) < 10.0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evo1.0"))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when a quality gate fails")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base, audit = load_and_audit(args.data)
    train_ids, val_ids = train_test_split(
        base["row_id"].to_numpy(),
        test_size=0.2,
        random_state=SEED,
        stratify=base["scenario"].to_numpy(),
    )
    train_id_set, val_id_set = set(train_ids), set(val_ids)
    expanded = expand_event_types(base)
    train_mask = expanded["row_id"].isin(train_id_set).to_numpy()
    val_mask = expanded["row_id"].isin(val_id_set).to_numpy()
    if set(expanded.loc[train_mask, "row_id"]) & set(expanded.loc[val_mask, "row_id"]):
        raise AssertionError("Duplicate group leaked across the train/validation boundary")

    preprocessor = Preprocessor.fit(expanded.loc[train_mask])
    features = preprocessor.transform(expanded)
    targets, observed, target_mean, target_scale = make_targets(expanded, train_mask)
    core_model = train_model(
        features,
        targets,
        observed,
        train_mask,
        val_mask,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    export_model = ExportModel(core_model, target_mean, target_scale).eval()

    with torch.no_grad():
        val_prediction = export_model(torch.from_numpy(features[val_mask])).numpy()
    raw_targets = expanded[OUTPUTS].to_numpy(dtype=np.float32)[val_mask]
    metrics = evaluate(val_prediction, raw_targets, observed[val_mask])

    checkpoint = {
        "model_version": MODEL_VERSION,
        "input_dim": features.shape[1],
        "state_dict": core_model.state_dict(),
        "target_mean": target_mean,
        "target_scale": target_scale,
    }
    torch.save(checkpoint, args.output_dir / f"{MODEL_VERSION}.pt")
    (args.output_dir / "feature_schema.json").write_text(
        json.dumps(preprocessor.schema(), indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    sample = features[val_mask][: min(32, int(val_mask.sum()))]
    export_metrics = export_and_validate(export_model, sample, args.output_dir)
    quality_gates = {
        "val_mae_time_under_1_5_min": metrics["mae_time_min"] < 1.5,
        "val_mae_success_under_3_pct": metrics["mae_success_pct"] < 3.0,
        "onnx_matches_pytorch_within_1pct": export_metrics["onnx_matches_pytorch_within_1pct"],
        "openvino_matches_pytorch_within_1pct": export_metrics["openvino_matches_pytorch_within_1pct"],
        "openvino_ir_loaded": export_metrics["openvino_ir_loaded"],
        "single_sample_under_10ms_p95": export_metrics["single_sample_under_10ms_p95"],
    }
    report = {
        "model_version": MODEL_VERSION,
        "split": {
            "strategy": "80/20 unique-row split stratified by normalized scenario; event expansions remain grouped",
            "train_unique_rows": len(train_id_set),
            "validation_unique_rows": len(val_id_set),
            "train_expanded_rows": int(train_mask.sum()),
            "validation_expanded_rows": int(val_mask.sum()),
        },
        "validation_metrics": metrics,
        "export_validation": export_metrics,
        "quality_gates": quality_gates,
        "all_quality_gates_pass": all(quality_gates.values()),
    }
    (args.output_dir / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Artifacts saved to {args.output_dir.resolve()}")
    if args.strict and not report["all_quality_gates_pass"]:
        raise SystemExit("One or more Evo 1.0 quality gates failed")


if __name__ == "__main__":
    main()
