#!/usr/bin/env python3
"""Train and honestly validate the Evo 1.1 dual-head evacuation model."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


MODEL_VERSION = "evo1.1"
SEED = 42
N_FOLDS = 5
SEVERITY_VARIANTS = (0.0, 0.3, 0.6, 0.9)

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
CATEGORY_CAPACITY = {"Train Station": 2_000.0, "Office Building": 1_000.0, "Stadium": 60_000.0}

SCENARIO_BY_EVENT = {
    "fire": "Electrical Fire",
    "flood": "Station Flooding",
    "earthquake": "Panic Chain Reaction",
    "tsunami": "Station Flooding",
    "tornado": "Sudden Overcrowding",
    "other": "Standard Evacuation Drill",
}
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

NUMERIC_FEATURES = [
    "occupancy_log_scaled",
    "density",
    "severity_score",
    "hazard_magnitude",
    "occupancy_density_interaction_scaled",
    "capacity_utilization_scaled",
]
OUTPUTS = ["evacuation_success_pct", "evacuation_time_min"]


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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
    return hashlib.sha256(json.dumps(fields, default=str).encode()).hexdigest()[:16]


def load_reference(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    unique: dict[str, dict[str, Any]] = {}
    for row in raw:
        unique.setdefault(stable_row_id(row), row)

    records = []
    for row_id, row in unique.items():
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
    frame = pd.DataFrame(records)
    if frame["occupancy"].isna().any() or frame["evacuation_time_min"].isna().any():
        raise ValueError("Every unique row must have occupancy and evacuation time")

    train_station = frame[frame["category"] == "Train Station"]
    correlations = {}
    for feature in ("occupancy", "density", "evacuation_time_min"):
        correlations[feature] = float(
            train_station["evacuation_success_pct"].corr(train_station[feature])
        )
    scenario_counts = frame["scenario"].value_counts().sort_index().to_dict()
    audit = {
        "raw_rows": len(raw),
        "unique_rows": len(frame),
        "exact_duplicate_rows_removed": len(raw) - len(frame),
        "missing_density_rows": int(frame["density"].isna().sum()),
        "missing_success_target_rows": int(frame["evacuation_success_pct"].isna().sum()),
        "category_counts_unique": frame["category"].value_counts().sort_index().to_dict(),
        "scenario_counts_unique_normalized": scenario_counts,
        "raw_train_station_success": {
            "mean": float(train_station["evacuation_success_pct"].mean()),
            "std": float(train_station["evacuation_success_pct"].std()),
            "min": float(train_station["evacuation_success_pct"].min()),
            "max": float(train_station["evacuation_success_pct"].max()),
            "feature_correlations": correlations,
        },
        "cv_warning": (
            "Blockage has only three unique groups, so five-fold grouped stratification cannot "
            "place that scenario in every fold. Synthetic variants remain grouped to prevent leakage."
        ),
    }
    return frame, audit


def augment_reference(base: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in base.to_dict(orient="records"):
        for event_type in EVENTS_BY_SCENARIO[record["scenario"]]:
            for severity in SEVERITY_VARIANTS:
                item = dict(record)
                item["event_type"] = event_type
                item["severity_score"] = severity
                item["hazard_magnitude"] = 0.0
                success = item["evacuation_success_pct"]
                if success is not None and math.isfinite(success):
                    item["evacuation_success_pct"] = float(np.clip(success - severity * 3.0, 0.0, 100.0))
                rows.append(item)
    return pd.DataFrame(rows)


@dataclass
class Preprocessor:
    engineered_scaler: StandardScaler
    density_imputation: float

    @staticmethod
    def engineered(frame: pd.DataFrame, density_imputation: float) -> np.ndarray:
        occupancy_log = np.log1p(frame["occupancy"].to_numpy(dtype=np.float64))
        density = frame["density"].fillna(density_imputation).clip(0.0, 1.0).to_numpy(dtype=np.float64)
        capacity = frame["category"].map(CATEGORY_CAPACITY).to_numpy(dtype=np.float64)
        utilization = np.clip(frame["occupancy"].to_numpy(dtype=np.float64) / capacity, 0.0, 2.0)
        interaction = occupancy_log * density
        return np.column_stack([occupancy_log, interaction, utilization])

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "Preprocessor":
        density_imputation = float(frame["density"].median())
        scaler = StandardScaler().fit(cls.engineered(frame, density_imputation))
        return cls(scaler, density_imputation)

    @property
    def feature_order(self) -> list[str]:
        return (
            NUMERIC_FEATURES
            + [f"category={value}" for value in CATEGORIES]
            + [f"scenario={value}" for value in SCENARIOS]
            + [f"event_type={value}" for value in EVENT_TYPES]
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        engineered = self.engineered(frame, self.density_imputation)
        engineered_scaled = self.engineered_scaler.transform(engineered)
        density = frame["density"].fillna(self.density_imputation).clip(0.0, 1.0).to_numpy()
        numeric = np.column_stack(
            [
                engineered_scaled[:, 0],
                density,
                frame["severity_score"].clip(0.0, 1.0).to_numpy(),
                frame["hazard_magnitude"].clip(lower=0.0).to_numpy(),
                engineered_scaled[:, 1],
                engineered_scaled[:, 2],
            ]
        )
        one_hot = []
        for column, values in (
            ("category", CATEGORIES),
            ("scenario", SCENARIOS),
            ("event_type", EVENT_TYPES),
        ):
            source = frame[column].to_numpy()
            one_hot.append(np.column_stack([source == value for value in values]))
        return np.column_stack([numeric, *one_hot]).astype(np.float32)

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
                "engineered_features": ["occupancy_log", "occupancy_log_x_density", "capacity_utilization"],
                "means": self.engineered_scaler.mean_.tolist(),
                "scales": self.engineered_scaler.scale_.tolist(),
                "density_missing_value": self.density_imputation,
                "category_capacity": CATEGORY_CAPACITY,
            },
            "synthetic_hazard_rule": {
                "severity_variants": list(SEVERITY_VARIANTS),
                "success_adjustment": "clip(original_success_pct - severity_score * 3, 0, 100)",
                "variants_grouped_during_cv": True,
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


@dataclass(frozen=True)
class Hyperparameters:
    hidden_dim: int
    dropout: float
    learning_rate: float

    @property
    def key(self) -> str:
        return f"h{self.hidden_dim}_d{self.dropout}_lr{self.learning_rate}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
        }


class DualHeadMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        tower_dim = max(16, hidden_dim // 2)

        def tower() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, tower_dim),
                nn.ReLU(),
                nn.Linear(tower_dim, 1),
            )

        self.success_head = tower()
        self.time_head = tower()
        # Magnitude is retained in the contract but has no labels. Keep its
        # first-layer influence exactly neutral rather than randomly initialized.
        with torch.no_grad():
            self.success_head[0].weight[:, 3] = 0.0
            self.time_head[0].weight[:, 3] = 0.0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.success_head(features), self.time_head(features)), dim=1)


@dataclass
class TargetTransform:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "TargetTransform":
        success = 100.0 - frame["evacuation_success_pct"].to_numpy(dtype=np.float32)
        time_log = np.log1p(frame["evacuation_time_min"].to_numpy(dtype=np.float32))
        success_observed = np.isfinite(success)
        mean = np.array([success[success_observed].mean(), time_log.mean()], dtype=np.float32)
        scale = np.array([success[success_observed].std(), time_log.std()], dtype=np.float32)
        return cls(mean, np.maximum(scale, 1e-6))

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        success = 100.0 - frame["evacuation_success_pct"].to_numpy(dtype=np.float32)
        time_log = np.log1p(frame["evacuation_time_min"].to_numpy(dtype=np.float32))
        raw = np.column_stack([success, time_log])
        observed = np.isfinite(raw)
        standardized = np.where(observed, (raw - self.mean) / self.scale, 0.0)
        return standardized.astype(np.float32), observed.astype(np.float32)

    def inverse(self, standardized: np.ndarray) -> np.ndarray:
        transformed = standardized * self.scale + self.mean
        success = np.clip(100.0 - transformed[:, 0], 0.0, 100.0)
        evacuation_time = np.maximum(np.expm1(transformed[:, 1]), 0.01)
        return np.column_stack([success, evacuation_time]).astype(np.float32)


class ExportModel(nn.Module):
    def __init__(self, model: DualHeadMLP, transform: TargetTransform):
        super().__init__()
        self.model = model
        self.register_buffer("target_mean", torch.tensor(transform.mean, dtype=torch.float32))
        self.register_buffer("target_scale", torch.tensor(transform.scale, dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        transformed = self.model(features) * self.target_scale + self.target_mean
        success = (100.0 - transformed[:, 0]).clamp(0.0, 100.0)
        # ONNX opset 17 does not expose aten::expm1 in the legacy exporter.
        # exp(x) - 1 is mathematically equivalent and portable to OpenVINO.
        evacuation_time = (torch.exp(transformed[:, 1]) - 1.0).clamp_min(0.01)
        return torch.stack((success, evacuation_time), dim=1)


def masked_component_losses(
    prediction: torch.Tensor, target: torch.Tensor, observed: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    squared = (prediction - target).pow(2)
    success = (squared[:, 0] * observed[:, 0]).sum() / observed[:, 0].sum().clamp_min(1.0)
    time_loss = squared[:, 1].mean()
    return success, time_loss


def composite_loss(
    prediction: torch.Tensor, target: torch.Tensor, observed: torch.Tensor
) -> torch.Tensor:
    success, time_loss = masked_component_losses(prediction, target, observed)
    return success * 0.5 + time_loss * 0.5


@dataclass
class FitResult:
    model: DualHeadMLP
    history: dict[str, list[float]]
    best_epoch: int


def fit_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_observed: np.ndarray,
    val_x: np.ndarray | None,
    val_y: np.ndarray | None,
    val_observed: np.ndarray | None,
    params: Hyperparameters,
    *,
    max_epochs: int,
    patience: int,
) -> FitResult:
    seed_everything()
    model = DualHeadMLP(train_x.shape[1], params.hidden_dim, params.dropout)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5
    )
    dataset = TensorDataset(
        torch.from_numpy(train_x),
        torch.from_numpy(train_y),
        torch.from_numpy(train_observed),
    )
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_tensors = None
    if val_x is not None and val_y is not None and val_observed is not None:
        val_tensors = (
            torch.from_numpy(val_x),
            torch.from_numpy(val_y),
            torch.from_numpy(val_observed),
        )

    history = {"train_loss": [], "val_loss": [], "learning_rate": []}
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        batch_losses = []
        for batch_x, batch_y, batch_observed in loader:
            augmented = batch_x.clone()
            augmented[:, 0] += torch.randn_like(augmented[:, 0]) * 0.015
            augmented[:, 1] = (augmented[:, 1] + torch.randn_like(augmented[:, 1]) * 0.01).clamp(0, 1)
            optimizer.zero_grad(set_to_none=True)
            loss = composite_loss(model(augmented), batch_y, batch_observed)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        train_loss = float(np.mean(batch_losses))

        model.eval()
        with torch.no_grad():
            if val_tensors is None:
                val_loss = train_loss
            else:
                val_loss = float(composite_loss(model(val_tensors[0]), val_tensors[1], val_tensors[2]))
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if val_tensors is not None and stale >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return FitResult(model, history, best_epoch)


def safe_r2(actual: np.ndarray, prediction: np.ndarray) -> float | None:
    if len(actual) < 2 or float(np.var(actual)) <= 1e-12:
        return None
    return float(r2_score(actual, prediction))


def regression_metrics(
    actual: np.ndarray, prediction: np.ndarray, observed: np.ndarray
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, name in enumerate(("success_pct", "time_min")):
        mask = observed[:, index].astype(bool)
        result[f"samples_{name}"] = int(mask.sum())
        if not mask.any():
            result[f"mae_{name}"] = None
            result[f"r2_{name}"] = None
            continue
        y, p = actual[mask, index], prediction[mask, index]
        result[f"mae_{name}"] = float(mean_absolute_error(y, p))
        result[f"r2_{name}"] = safe_r2(y, p)
    return result


def physical_targets(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = frame[OUTPUTS].to_numpy(dtype=np.float32)
    observed = np.isfinite(values)
    return values, observed


def mean_baseline(train: pd.DataFrame, validation: pd.DataFrame) -> np.ndarray:
    success_mean = float(train["evacuation_success_pct"].mean())
    time_mean = float(train["evacuation_time_min"].mean())
    return np.column_stack(
        [
            np.full(len(validation), success_mean, dtype=np.float32),
            np.full(len(validation), time_mean, dtype=np.float32),
        ]
    )


def production_knn_baseline(train: pd.DataFrame, validation: pd.DataFrame, k: int = 25) -> np.ndarray:
    candidates = (
        train[
            train["density"].notna()
            & train["evacuation_success_pct"].notna()
            & (train["severity_score"] == 0.0)
        ]
        .drop_duplicates("row_id")
        .reset_index(drop=True)
    )
    occ = candidates["occupancy"].to_numpy(dtype=np.float64)
    density = candidates["density"].to_numpy(dtype=np.float64)
    category = candidates["category"].to_numpy()
    scenario = candidates["scenario"].to_numpy()
    success = candidates["evacuation_success_pct"].to_numpy(dtype=np.float64)
    evacuation_time = candidates["evacuation_time_min"].to_numpy(dtype=np.float64)
    default_density = float(candidates["density"].median())
    predictions = []
    for row in validation.itertuples(index=False):
        row_density = default_density if pd.isna(row.density) else float(row.density)
        distance = (
            np.abs(occ - float(row.occupancy)) / max(float(row.occupancy), 1.0)
            + np.abs(density - row_density)
            + np.where(category == row.category, 0.0, 0.35)
            + np.where(scenario == row.scenario, 0.0, 0.2)
        )
        nearest = np.argpartition(distance, min(k, len(distance)) - 1)[:k]
        weights = np.exp(-distance[nearest])
        weight_sum = weights.sum() or 1.0
        predictions.append(
            [
                float(np.sum(success[nearest] * weights) / weight_sum),
                float(np.sum(evacuation_time[nearest] * weights) / weight_sum),
            ]
        )
    return np.asarray(predictions, dtype=np.float32)


def per_category_metrics(
    frame: pd.DataFrame, actual: np.ndarray, prediction: np.ndarray, observed: np.ndarray
) -> dict[str, Any]:
    output = {}
    categories = frame["category"].to_numpy()
    for category in CATEGORIES:
        mask = categories == category
        output[category] = regression_metrics(actual[mask], prediction[mask], observed[mask])
    return output


def metric_mean_std(folds: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["mae_success_pct", "r2_success_pct", "mae_time_min", "r2_time_min"]
    result = {}
    for key in keys:
        values = [float(fold[key]) for fold in folds if fold.get(key) is not None]
        result[key] = {
            "mean": float(statistics.mean(values)),
            "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        }
    return result


def average_histories(histories: list[dict[str, list[float]]]) -> dict[str, list[float]]:
    max_length = max(len(item["train_loss"]) for item in histories)
    output = {}
    for key in ("train_loss", "val_loss", "learning_rate"):
        values = []
        for epoch in range(max_length):
            epoch_values = [history[key][epoch] for history in histories if epoch < len(history[key])]
            values.append(float(np.mean(epoch_values)))
        output[key] = values
    return output


def cross_validate(
    augmented: pd.DataFrame,
    params: Hyperparameters,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    max_epochs: int,
    patience: int,
) -> dict[str, Any]:
    model_oof = np.full((len(augmented), 2), np.nan, dtype=np.float32)
    fold_metrics = []
    histories = []
    best_epochs = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        train, val = augmented.iloc[train_idx], augmented.iloc[val_idx]
        preprocessor = Preprocessor.fit(train)
        transform = TargetTransform.fit(train)
        train_x, val_x = preprocessor.transform(train), preprocessor.transform(val)
        train_y, train_observed = transform.transform(train)
        val_y, val_observed = transform.transform(val)
        fit = fit_model(
            train_x,
            train_y,
            train_observed,
            val_x,
            val_y,
            val_observed,
            params,
            max_epochs=max_epochs,
            patience=patience,
        )
        with torch.no_grad():
            standardized = fit.model(torch.from_numpy(val_x)).numpy()
        prediction = transform.inverse(standardized)
        model_oof[val_idx] = prediction
        actual, observed = physical_targets(val)
        fold_metrics.append(
            {
                "fold": fold + 1,
                "best_epoch": fit.best_epoch,
                **regression_metrics(actual, prediction, observed),
            }
        )
        histories.append(fit.history)
        best_epochs.append(fit.best_epoch)

    actual, observed = physical_targets(augmented)
    aggregate = regression_metrics(actual, model_oof, observed)
    selection_score = float((aggregate["r2_success_pct"] or -10.0) + (aggregate["r2_time_min"] or -10.0))
    return {
        "hyperparameters": params.as_dict(),
        "selection_score": selection_score,
        "aggregate_metrics": aggregate,
        "fold_metrics": fold_metrics,
        "fold_mean_std": metric_mean_std(fold_metrics),
        "mean_best_epoch": float(np.mean(best_epochs)),
        "learning_curves": average_histories(histories),
        "oof_prediction": model_oof,
    }


def compute_baselines(
    augmented: pd.DataFrame, splits: list[tuple[np.ndarray, np.ndarray]]
) -> dict[str, Any]:
    mean_oof = np.full((len(augmented), 2), np.nan, dtype=np.float32)
    knn_oof = np.full((len(augmented), 2), np.nan, dtype=np.float32)
    for train_idx, val_idx in splits:
        train, val = augmented.iloc[train_idx], augmented.iloc[val_idx]
        mean_oof[val_idx] = mean_baseline(train, val)
        knn_oof[val_idx] = production_knn_baseline(train, val)
    actual, observed = physical_targets(augmented)
    return {
        "mean": {
            "aggregate_metrics": regression_metrics(actual, mean_oof, observed),
            "per_category": per_category_metrics(augmented, actual, mean_oof, observed),
        },
        "knn_k25_production_distance": {
            "aggregate_metrics": regression_metrics(actual, knn_oof, observed),
            "per_category": per_category_metrics(augmented, actual, knn_oof, observed),
        },
    }


def beats(model: dict[str, Any], baseline: dict[str, Any], target: str) -> bool:
    return bool(
        model[f"mae_{target}"] < baseline[f"mae_{target}"]
        and model[f"r2_{target}"] > baseline[f"r2_{target}"]
    )


def export_and_validate(
    model: ExportModel, sample: np.ndarray, output_dir: Path
) -> dict[str, Any]:
    onnx_path = output_dir / f"{MODEL_VERSION}.onnx"
    openvino_dir = output_dir / "openvino"
    openvino_dir.mkdir(parents=True, exist_ok=True)
    xml_path = openvino_dir / f"{MODEL_VERSION}.xml"
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
    ov_model = ov.convert_model(str(onnx_path))
    ov.save_model(ov_model, str(xml_path))
    core = ov.Core()
    compiled = core.compile_model(core.read_model(str(xml_path)), "CPU")
    request = compiled.create_infer_request()
    openvino_output = request.infer({"features": sample.astype(np.float32)})["predictions"]

    single = sample[:1].astype(np.float32)
    for _ in range(20):
        request.infer({"features": single})
    timings = []
    for _ in range(200):
        started = time.perf_counter_ns()
        request.infer({"features": single})
        timings.append((time.perf_counter_ns() - started) / 1_000_000)
    return {
        "onnx_matches_pytorch_within_1pct": bool(np.allclose(torch_output, onnx_output, rtol=0.01, atol=0.01)),
        "onnx_max_abs_difference": float(np.max(np.abs(torch_output - onnx_output))),
        "openvino_matches_pytorch_within_1pct": bool(
            np.allclose(torch_output, openvino_output, rtol=0.01, atol=0.01)
        ),
        "openvino_max_abs_difference": float(np.max(np.abs(torch_output - openvino_output))),
        "openvino_ir_loaded": True,
        "single_sample_latency_ms_p50": float(np.percentile(timings, 50)),
        "single_sample_latency_ms_p95": float(np.percentile(timings, 95)),
        "single_sample_under_10ms_p95": bool(np.percentile(timings, 95) < 10.0),
    }


def architecture_json(best: Hyperparameters) -> dict[str, Any]:
    half = max(16, best.hidden_dim // 2)
    return {
        "name": "Evo 1.1 dual-head MLP",
        "model_version": MODEL_VERSION,
        "layers": [
            {"id": "in", "type": "input", "label": "Engineered inputs", "column": 0, "lane": 0.5},
            {"id": "success1", "type": "dense", "label": f"Success {best.hidden_dim}\nReLU + Dropout", "column": 1, "lane": 0.25},
            {"id": "success2", "type": "dense", "label": f"Success {half}\nshortfall", "column": 2, "lane": 0.25},
            {"id": "success_out", "type": "output", "label": "success %", "column": 3, "lane": 0.25},
            {"id": "time1", "type": "dense", "label": f"Time {best.hidden_dim}\nReLU + Dropout", "column": 1, "lane": 0.75},
            {"id": "time2", "type": "dense", "label": f"Time {half}\nlog-time", "column": 2, "lane": 0.75},
            {"id": "time_out", "type": "output", "label": "time min", "column": 3, "lane": 0.75},
        ],
        "edges": [
            ["in", "success1"], ["success1", "success2"], ["success2", "success_out"],
            ["in", "time1"], ["time1", "time2"], ["time2", "time_out"],
        ],
    }


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items() if key != "oof_prediction"}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evo1.1"))
    parser.add_argument("--max-epochs", type=int, default=220)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base, audit = load_reference(args.data)
    augmented = augment_reference(base)
    audit["augmented_rows"] = len(augmented)
    audit["severity_variants"] = list(SEVERITY_VARIANTS)

    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(
        splitter.split(
            augmented,
            y=augmented["scenario"].to_numpy(),
            groups=augmented["row_id"].to_numpy(),
        )
    )
    baselines = compute_baselines(augmented, splits)

    grid = [
        Hyperparameters(hidden, dropout, learning_rate)
        for hidden, dropout, learning_rate in itertools.product(
            (64, 128), (0.1, 0.2), (1e-3, 3e-4)
        )
    ]
    sweep = []
    for index, params in enumerate(grid, start=1):
        print(f"sweep {index}/{len(grid)}: {params.key}", flush=True)
        result = cross_validate(
            augmented,
            params,
            splits,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
        sweep.append(result)
        print(json.dumps(result["aggregate_metrics"], sort_keys=True), flush=True)
    best_result = max(sweep, key=lambda item: item["selection_score"])
    best_params = Hyperparameters(**best_result["hyperparameters"])
    print(f"selected {best_params.key}", flush=True)

    # Train the deployable checkpoint on four grouped folds, retaining the fifth
    # for early stopping. CV metrics above remain the acceptance evidence.
    final_train_idx, final_val_idx = splits[0]
    final_train, final_val = augmented.iloc[final_train_idx], augmented.iloc[final_val_idx]
    preprocessor = Preprocessor.fit(final_train)
    transform = TargetTransform.fit(final_train)
    final_train_x, final_val_x = preprocessor.transform(final_train), preprocessor.transform(final_val)
    final_train_y, final_train_observed = transform.transform(final_train)
    final_val_y, final_val_observed = transform.transform(final_val)
    final_fit = fit_model(
        final_train_x,
        final_train_y,
        final_train_observed,
        final_val_x,
        final_val_y,
        final_val_observed,
        best_params,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    export_model = ExportModel(final_fit.model, transform).eval()
    export_metrics = export_and_validate(export_model, final_val_x[:32], args.output_dir)

    checkpoint = {
        "model_version": MODEL_VERSION,
        "input_dim": final_train_x.shape[1],
        "hyperparameters": best_params.as_dict(),
        "state_dict": final_fit.model.state_dict(),
        "target_mean": transform.mean,
        "target_scale": transform.scale,
        "target_transforms": ["success_shortfall_100_minus_pct", "log1p_time_min"],
    }
    torch.save(checkpoint, args.output_dir / f"{MODEL_VERSION}.pt")

    model_metrics = best_result["aggregate_metrics"]
    mean_metrics = baselines["mean"]["aggregate_metrics"]
    knn_metrics = baselines["knn_k25_production_distance"]["aggregate_metrics"]
    quality_gates = {
        "cv_mae_time_under_1_0_min": model_metrics["mae_time_min"] < 1.0,
        "cv_train_station_mae_success_under_2_pct": model_metrics["mae_success_pct"] < 2.0,
        "cv_r2_success_at_least_0_25": model_metrics["r2_success_pct"] >= 0.25,
        "cv_r2_time_at_least_0_75": model_metrics["r2_time_min"] >= 0.75,
        "beats_mean_success": beats(model_metrics, mean_metrics, "success_pct"),
        "beats_mean_time": beats(model_metrics, mean_metrics, "time_min"),
        "beats_knn_success": beats(model_metrics, knn_metrics, "success_pct"),
        "beats_knn_time": beats(model_metrics, knn_metrics, "time_min"),
        "onnx_matches_pytorch_within_1pct": export_metrics["onnx_matches_pytorch_within_1pct"],
        "openvino_matches_pytorch_within_1pct": export_metrics[
            "openvino_matches_pytorch_within_1pct"
        ],
        "openvino_ir_loaded": export_metrics["openvino_ir_loaded"],
        "single_sample_under_10ms_p95": export_metrics["single_sample_under_10ms_p95"],
    }
    all_pass = all(quality_gates.values())
    actual, observed = physical_targets(augmented)
    model_oof = best_result["oof_prediction"]
    per_category = per_category_metrics(augmented, actual, model_oof, observed)
    failure_reasons = [name for name, passed in quality_gates.items() if not passed]
    recommendation = (
        "promote_evo1.1"
        if all_pass
        else (
            "use_evo_for_time_only_and_keep_knn_for_success_until_live_peoplesense_noaa_features"
            if quality_gates["cv_mae_time_under_1_0_min"]
            else "do_not_promote_evo1.1"
        )
    )
    report = {
        "model_version": MODEL_VERSION,
        "evaluation_protocol": {
            "folds": N_FOLDS,
            "splitter": "StratifiedGroupKFold by normalized scenario and source row_id",
            "synthetic_variants_grouped": True,
            "hyperparameter_grid_size": len(grid),
            "selection_metric": "OOF r2_success_pct + OOF r2_time_min",
        },
        "selected_hyperparameters": best_params.as_dict(),
        "cross_validation": {
            "aggregate_metrics": model_metrics,
            "fold_metrics": best_result["fold_metrics"],
            "fold_mean_std": best_result["fold_mean_std"],
            "per_category": per_category,
        },
        "baseline_comparisons": baselines,
        "hyperparameter_sweep": [clean_json(item) for item in sweep],
        "export_validation": export_metrics,
        "quality_gates": quality_gates,
        "all_quality_gates_pass": all_pass,
        "failed_quality_gates": failure_reasons,
        "promotion_recommendation": recommendation,
        "honest_assessment": (
            "Success is only labeled for Train Station rows. Raw success has near-zero correlation "
            "with measured features; synthetic severity adds a controlled relationship but cannot "
            "create missing real-world signal."
        ),
    }
    curves = {
        "model_version": MODEL_VERSION,
        "source": "final grouped holdout fit",
        **final_fit.history,
    }
    metrics_dashboard = {
        "status": "passed" if all_pass else "failed_quality_gate",
        "model_version": MODEL_VERSION,
        "message": recommendation,
        "train_loss": final_fit.history["train_loss"],
        "val_loss": final_fit.history["val_loss"],
        "val_mae_success_pct": model_metrics["mae_success_pct"],
        "val_r2_success_pct": model_metrics["r2_success_pct"],
        "val_mae_time_min": model_metrics["mae_time_min"],
        "val_r2_time_min": model_metrics["r2_time_min"],
        "all_quality_gates_pass": all_pass,
    }
    files = {
        "feature_schema.json": preprocessor.schema(),
        "validation_report.json": report,
        "learning_curves.json": curves,
        "data_audit.json": audit,
        "architecture.json": architecture_json(best_params),
        "metrics.json": metrics_dashboard,
    }
    for filename, content in files.items():
        (args.output_dir / filename).write_text(
            json.dumps(clean_json(content), indent=2) + "\n", encoding="utf-8"
        )
    summary = {
        "model_version": MODEL_VERSION,
        "selected_hyperparameters": best_params.as_dict(),
        "cross_validation_metrics": model_metrics,
        "baseline_metrics": {
            name: value["aggregate_metrics"] for name, value in baselines.items()
        },
        "quality_gates": quality_gates,
        "all_quality_gates_pass": all_pass,
        "promotion_recommendation": recommendation,
        "artifacts": str(args.output_dir.resolve()),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.strict and not all_pass:
        raise SystemExit("Evo 1.1 failed one or more strict quality gates")


if __name__ == "__main__":
    main()
