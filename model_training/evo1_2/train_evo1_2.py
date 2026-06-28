#!/usr/bin/env python3
"""Train and honestly validate Evo 1.2 with real FCUSD hazard features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_training.evo1_1 import train_evo1_1 as v11


MODEL_VERSION = "evo1.2"
SEED = 42
N_FOLDS = 5
SYNTHETIC_SEVERITIES = (0.3, 0.6, 0.9)
HAZARD_SOURCES = ["none", "noaa_nws", "usgs", "gdacs", "nasa_firms", "fema_ipaws"]
LIVE_NUMERIC_FEATURES = [
    "occupancy_log_scaled",
    "density_scaled",
    "severity_score_scaled",
    "hazard_magnitude_log_scaled",
    "hazard_distance_log_scaled",
    "hazard_depth_km_scaled",
    "occupancy_density_interaction_scaled",
    "capacity_utilization_scaled",
    "real_hazard_join_flag_scaled",
    "synthetic_augmentation_flag_scaled",
]

# Reused Evo 1.1 model/loss/export classes read these globals at runtime.
v11.MODEL_VERSION = MODEL_VERSION


def finite(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def load_seed(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"rows": [], "hazards": [], "sources_active": []}
    return json.loads(path.read_text(encoding="utf-8"))


def assign_real_hazards(base: pd.DataFrame, seed: dict[str, Any]) -> pd.DataFrame:
    """Assign each reference row to a category-compatible FCUSD spot and its nearest hazard."""
    live_rows = seed.get("rows", [])
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in live_rows:
        by_category.setdefault(str(row.get("Category")), []).append(row)
    all_rows = list(live_rows)
    joined = []
    for record in base.to_dict(orient="records"):
        candidates = by_category.get(record["category"]) or all_rows
        if candidates:
            slot = int(record["row_id"], 16) % len(candidates)
            hazard = candidates[slot]
        else:
            hazard = {}
        item = dict(record)
        item.update(
            {
                "event_type": str(hazard.get("event_type") or v11.EVENTS_BY_SCENARIO[item["scenario"]][0]),
                "severity_score": finite(hazard.get("severity_score"), 0.0),
                "hazard_magnitude": finite(hazard.get("hazard_magnitude"), 0.0),
                "hazard_distance_km": finite(hazard.get("hazard_distance_km"), 250.0),
                "hazard_depth_km": finite((hazard.get("live_hazard") or {}).get("depth_km"), 0.0),
                "hazard_source": str(hazard.get("hazard_source") or "none"),
                "spot_id": hazard.get("spot_id"),
                "data_origin": "labeled_reference_real_hazard_join",
                "labels_available": True,
                "metric_eligible": True,
                "real_hazard_join": bool(hazard.get("hazard_source")),
                "synthetic_augmentation": False,
            }
        )
        joined.append(item)
    return pd.DataFrame(joined)


def augment_labeled(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in joined.to_dict(orient="records"):
        rows.append(dict(record))
        for severity in SYNTHETIC_SEVERITIES:
            item = dict(record)
            item["severity_score"] = severity
            item["synthetic_augmentation"] = True
            item["data_origin"] = "synthetic_hazard_augmentation"
            success = finite(item.get("evacuation_success_pct"))
            if success is not None:
                item["evacuation_success_pct"] = float(np.clip(success - severity * 3.0, 0.0, 100.0))
            rows.append(item)
    return pd.DataFrame(rows)


def live_feature_frame(seed: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for index, source in enumerate(seed.get("rows", [])):
        category = str(source.get("Category") or "Office Building")
        scenario = v11.normalize_scenario(source.get("Scenario") or "Standard Evacuation Drill")
        rows.append(
            {
                "row_id": f"live-{source.get('spot_id') or index}",
                "scenario": scenario,
                "category": category,
                "occupancy": finite(source.get("Occupancy (#)"), 0.0),
                "density": finite(source.get("Density (#)")),
                "evacuation_success_pct": np.nan,
                "evacuation_time_min": np.nan,
                "event_type": str(source.get("event_type") or "other"),
                "severity_score": finite(source.get("severity_score"), 0.0),
                "hazard_magnitude": finite(source.get("hazard_magnitude"), 0.0),
                "hazard_distance_km": finite(source.get("hazard_distance_km"), 250.0),
                "hazard_depth_km": finite((source.get("live_hazard") or {}).get("depth_km"), 0.0),
                "hazard_source": str(source.get("hazard_source") or "none"),
                "spot_id": source.get("spot_id"),
                "data_origin": "live_hazard_seed_pseudo_label",
                "labels_available": False,
                "metric_eligible": False,
                "real_hazard_join": bool(source.get("hazard_source")),
                "synthetic_augmentation": False,
            }
        )
    return pd.DataFrame(rows)


@dataclass
class HazardPreprocessor:
    scaler: StandardScaler
    density_imputation: float

    @staticmethod
    def numeric(frame: pd.DataFrame, density_imputation: float) -> np.ndarray:
        occupancy = frame["occupancy"].fillna(0).to_numpy(dtype=np.float64)
        occupancy_log = np.log1p(np.maximum(occupancy, 0.0))
        density = frame["density"].fillna(density_imputation).clip(0, 1).to_numpy(dtype=np.float64)
        severity = frame["severity_score"].fillna(0).clip(0, 1).to_numpy(dtype=np.float64)
        magnitude = np.log1p(frame["hazard_magnitude"].fillna(0).clip(lower=0).to_numpy(dtype=np.float64))
        distance = np.log1p(frame["hazard_distance_km"].fillna(250).clip(lower=0).to_numpy(dtype=np.float64))
        depth = frame["hazard_depth_km"].fillna(0).clip(lower=0).to_numpy(dtype=np.float64)
        capacity = frame["category"].map(v11.CATEGORY_CAPACITY).fillna(2000).to_numpy(dtype=np.float64)
        utilization = np.clip(occupancy / capacity, 0, 4)
        interaction = occupancy_log * density
        real_join = frame["real_hazard_join"].fillna(False).astype(float).to_numpy()
        synthetic = frame["synthetic_augmentation"].fillna(False).astype(float).to_numpy()
        return np.column_stack(
            [occupancy_log, density, severity, magnitude, distance, depth, interaction, utilization, real_join, synthetic]
        )

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "HazardPreprocessor":
        density = float(frame["density"].median()) if frame["density"].notna().any() else 0.5
        return cls(StandardScaler().fit(cls.numeric(frame, density)), density)

    @property
    def feature_order(self) -> list[str]:
        return (
            LIVE_NUMERIC_FEATURES
            + [f"category={x}" for x in v11.CATEGORIES]
            + [f"scenario={x}" for x in v11.SCENARIOS]
            + [f"event_type={x}" for x in v11.EVENT_TYPES]
            + [f"hazard_source={x}" for x in HAZARD_SOURCES]
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        numeric = self.scaler.transform(self.numeric(frame, self.density_imputation))
        one_hot = []
        for column, values in (
            ("category", v11.CATEGORIES),
            ("scenario", v11.SCENARIOS),
            ("event_type", v11.EVENT_TYPES),
            ("hazard_source", HAZARD_SOURCES),
        ):
            source = frame[column].fillna("none").to_numpy()
            one_hot.append(np.column_stack([source == value for value in values]))
        return np.column_stack([numeric, *one_hot]).astype(np.float32)

    def schema(self) -> dict[str, Any]:
        return {
            "model_version": MODEL_VERSION,
            "input_name": "features",
            "input_dtype": "float32",
            "numeric_features": LIVE_NUMERIC_FEATURES,
            "categorical_features": {
                "category": v11.CATEGORIES,
                "scenario": v11.SCENARIOS,
                "event_type": v11.EVENT_TYPES,
                "hazard_source": HAZARD_SOURCES,
            },
            "feature_order": self.feature_order,
            "normalization": {
                "means": self.scaler.mean_.tolist(),
                "scales": self.scaler.scale_.tolist(),
                "density_missing_value": self.density_imputation,
                "category_capacity": v11.CATEGORY_CAPACITY,
            },
            "live_join": {
                "radius_km": 250,
                "labeled_row_location_proxy": "deterministic category-compatible FCUSD monitoring spot",
                "unlabeled_seed_used_for": "k-NN pseudo-label training only; excluded from metrics",
            },
            "synthetic_hazard_rule": {
                "severity_variants": list(SYNTHETIC_SEVERITIES),
                "success_adjustment": "clip(original_success_pct - severity_score * 3, 0, 100)",
                "explicit_flag": "synthetic_augmentation_flag_scaled",
                "variants_grouped_during_cv": True,
            },
            "outputs": v11.OUTPUTS,
        }


def pseudo_label_live(live: pd.DataFrame, labeled: pd.DataFrame) -> pd.DataFrame:
    if live.empty:
        return live
    reference = labeled.drop_duplicates("row_id").copy()
    # Reuse the production-distance k-NN contract while avoiding the Evo 1.1
    # zero-severity candidate restriction.
    candidates = reference[reference["evacuation_success_pct"].notna() & reference["density"].notna()]
    result = live.copy()
    predictions = []
    for row in result.itertuples(index=False):
        occ = candidates["occupancy"].to_numpy(float)
        den = candidates["density"].to_numpy(float)
        distance = (
            np.abs(occ - float(row.occupancy)) / max(float(row.occupancy), 1.0)
            + np.abs(den - float(row.density if pd.notna(row.density) else candidates["density"].median()))
            + np.where(candidates["category"].to_numpy() == row.category, 0.0, 0.35)
            + np.where(candidates["scenario"].to_numpy() == row.scenario, 0.0, 0.2)
        )
        k = min(25, len(candidates))
        nearest = np.argpartition(distance, k - 1)[:k]
        weights = np.exp(-distance[nearest])
        predictions.append(
            [
                float(np.average(candidates.iloc[nearest]["evacuation_success_pct"], weights=weights)),
                float(np.average(candidates.iloc[nearest]["evacuation_time_min"], weights=weights)),
            ]
        )
    result[["evacuation_success_pct", "evacuation_time_min"]] = np.asarray(predictions)
    return result


def target_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = frame[v11.OUTPUTS].to_numpy(dtype=np.float32)
    observed = np.isfinite(values) & frame["metric_eligible"].to_numpy()[:, None]
    return values, observed


def metrics(actual: np.ndarray, prediction: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    return v11.regression_metrics(actual, prediction, observed)


def add_pseudo(train: pd.DataFrame, pseudo: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([train, pseudo], ignore_index=True) if not pseudo.empty else train.copy()


def cross_validate_mlp(
    frame: pd.DataFrame,
    pseudo: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    max_epochs: int,
    patience: int,
) -> dict[str, Any]:
    params = v11.Hyperparameters(128, 0.2, 1e-3)
    oof = np.full((len(frame), 2), np.nan, dtype=np.float32)
    fold_results, histories, best_epochs = [], [], []
    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        train, val = frame.iloc[train_idx], frame.iloc[val_idx]
        fit_train = add_pseudo(train, pseudo)
        pre = HazardPreprocessor.fit(fit_train)
        transform = v11.TargetTransform.fit(fit_train)
        train_x, val_x = pre.transform(fit_train), pre.transform(val)
        train_y, train_seen = transform.transform(fit_train)
        val_y, val_seen = transform.transform(val)
        fit = v11.fit_model(
            train_x, train_y, train_seen, val_x, val_y, val_seen, params,
            max_epochs=max_epochs, patience=patience,
        )
        with torch.no_grad():
            oof[val_idx] = transform.inverse(fit.model(torch.from_numpy(val_x)).numpy())
        actual, observed = target_arrays(val)
        fold_results.append({"fold": fold, "best_epoch": fit.best_epoch, **metrics(actual, oof[val_idx], observed)})
        histories.append(fit.history)
        best_epochs.append(fit.best_epoch)
    actual, observed = target_arrays(frame)
    return {
        "name": "dual_head_mlp",
        "oof_prediction": oof,
        "aggregate_metrics": metrics(actual, oof, observed),
        "fold_metrics": fold_results,
        "fold_mean_std": v11.metric_mean_std(fold_results),
        "learning_curves": v11.average_histories(histories),
        "mean_best_epoch": float(np.mean(best_epochs)),
    }


def make_lgbm(seed_offset: int = 0):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=450,
        learning_rate=0.025,
        num_leaves=15,
        max_depth=6,
        min_child_samples=30,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=0.2,
        random_state=SEED + seed_offset,
        verbosity=-1,
        n_jobs=1,
    )


def cross_validate_lightgbm(
    frame: pd.DataFrame,
    pseudo: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    oof = np.full((len(frame), 2), np.nan, dtype=np.float32)
    fold_results, importances = [], []
    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        train, val = frame.iloc[train_idx], frame.iloc[val_idx]
        fit_train = add_pseudo(train, pseudo)
        pre = HazardPreprocessor.fit(fit_train)
        train_x, val_x = pre.transform(fit_train), pre.transform(val)
        fold_importance = {}
        for head, column in enumerate(v11.OUTPUTS):
            mask = fit_train[column].notna().to_numpy()
            model = make_lgbm(fold * 10 + head)
            model.fit(train_x[mask], fit_train.loc[mask, column].to_numpy(float))
            oof[val_idx, head] = model.predict(val_x)
            fold_importance[column] = dict(zip(pre.feature_order, model.feature_importances_.astype(float)))
        actual, observed = target_arrays(val)
        fold_results.append({"fold": fold, **metrics(actual, oof[val_idx], observed)})
        importances.append(fold_importance)
    actual, observed = target_arrays(frame)
    return {
        "name": "lightgbm_dual_regressors",
        "oof_prediction": oof,
        "aggregate_metrics": metrics(actual, oof, observed),
        "fold_metrics": fold_results,
        "fold_mean_std": v11.metric_mean_std(fold_results),
        "gain_importance_by_fold": importances,
    }


def candidate_score(result: dict[str, Any]) -> float:
    m = result["aggregate_metrics"]
    return float((m.get("r2_success_pct") or -10) + (m.get("r2_time_min") or -10))


def mean_and_knn_baselines(
    frame: pd.DataFrame, splits: list[tuple[np.ndarray, np.ndarray]]
) -> dict[str, Any]:
    mean_oof = np.full((len(frame), 2), np.nan, dtype=np.float32)
    knn_oof = np.full((len(frame), 2), np.nan, dtype=np.float32)
    for train_idx, val_idx in splits:
        train, val = frame.iloc[train_idx], frame.iloc[val_idx]
        originals = train[~train["synthetic_augmentation"]].drop_duplicates("row_id")
        mean_oof[val_idx] = v11.mean_baseline(originals, val)
        candidates = originals[originals["density"].notna() & originals["evacuation_success_pct"].notna()]
        predictions = pseudo_label_live(val, candidates)
        knn_oof[val_idx] = predictions[v11.OUTPUTS].to_numpy(np.float32)
    actual, observed = target_arrays(frame)
    return {
        "mean": {"aggregate_metrics": metrics(actual, mean_oof, observed)},
        "knn_k25_production_distance": {"aggregate_metrics": metrics(actual, knn_oof, observed)},
    }


def per_category(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, Any]:
    actual, observed = target_arrays(frame)
    output = {}
    for category in v11.CATEGORIES:
        mask = frame["category"].to_numpy() == category
        output[category] = metrics(actual[mask], prediction[mask], observed[mask])
    return output


def train_final_models(
    frame: pd.DataFrame,
    pseudo: pd.DataFrame,
    split: tuple[np.ndarray, np.ndarray],
    *,
    max_epochs: int,
    patience: int,
) -> dict[str, Any]:
    train_idx, val_idx = split
    train, val = frame.iloc[train_idx], frame.iloc[val_idx]
    fit_train = add_pseudo(train, pseudo)
    pre = HazardPreprocessor.fit(fit_train)
    transform = v11.TargetTransform.fit(fit_train)
    train_x, val_x = pre.transform(fit_train), pre.transform(val)
    train_y, train_seen = transform.transform(fit_train)
    val_y, val_seen = transform.transform(val)
    mlp_fit = v11.fit_model(
        train_x, train_y, train_seen, val_x, val_y, val_seen,
        v11.Hyperparameters(128, 0.2, 1e-3), max_epochs=max_epochs, patience=patience,
    )
    lgbm_models = []
    for head, column in enumerate(v11.OUTPUTS):
        mask = fit_train[column].notna().to_numpy()
        model = make_lgbm(900 + head)
        model.fit(train_x[mask], fit_train.loc[mask, column].to_numpy(float))
        lgbm_models.append(model)
    return {
        "preprocessor": pre,
        "target_transform": transform,
        "mlp_fit": mlp_fit,
        "lgbm_models": lgbm_models,
        "train_x": train_x,
        "val_x": val_x,
        "val": val,
    }


def mlp_predict(final: dict[str, Any], x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        standardized = final["mlp_fit"].model(torch.from_numpy(x.astype(np.float32))).numpy()
    return final["target_transform"].inverse(standardized)


def lgbm_predict(final: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return np.column_stack([model.predict(x) for model in final["lgbm_models"]]).astype(np.float32)


def export_mlp_onnx(final: dict[str, Any], sample: np.ndarray, path: Path) -> None:
    export_model = v11.ExportModel(final["mlp_fit"].model, final["target_transform"]).eval()
    torch.onnx.export(
        export_model,
        (torch.from_numpy(sample.astype(np.float32)),),
        path,
        input_names=["features"], output_names=["predictions"],
        dynamic_axes={"features": {0: "batch"}, "predictions": {0: "batch"}},
        opset_version=17, dynamo=False,
    )


def export_lightgbm_onnx(final: dict[str, Any], sample: np.ndarray, path: Path) -> None:
    """Export LightGBM heads as tensor operators OpenVINO can compile.

    The conventional ONNX converter emits ai.onnx.ml TreeEnsembleRegressor,
    which current OpenVINO releases do not support. Hummingbird lowers each
    tree ensemble to ordinary PyTorch/ONNX tensor operations instead.
    """
    from hummingbird.ml import convert

    class DualLightGBMTensorModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            converter_sample = sample[:1].astype(np.float32)
            self.success_head = convert(final["lgbm_models"][0], "torch", converter_sample).model
            self.time_head = convert(final["lgbm_models"][1], "torch", converter_sample).model

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return torch.cat((self.success_head(features), self.time_head(features)), dim=1)

    export_model = DualLightGBMTensorModel().eval()
    torch.onnx.export(
        export_model,
        (torch.from_numpy(sample[:2].astype(np.float32)),),
        path,
        input_names=["features"], output_names=["predictions"],
        dynamic_axes={"features": {0: "batch"}, "predictions": {0: "batch"}},
        opset_version=17, dynamo=False,
    )


def combine_onnx_average(first: Path, second: Path, output: Path, input_dim: int) -> None:
    import onnx
    from onnx import TensorProto, helper

    models = [onnx.compose.add_prefix(onnx.load(str(first)), "a_"), onnx.compose.add_prefix(onnx.load(str(second)), "b_")]
    nodes, initializers, sparse, value_info, outputs = [], [], [], [], []
    opsets: dict[str, int] = {"": 17}
    for prefix, model in zip(("a_", "b_"), models):
        for node in model.graph.node:
            for index, name in enumerate(node.input):
                if name == f"{prefix}features":
                    node.input[index] = "features"
            nodes.append(node)
        initializers.extend(model.graph.initializer)
        sparse.extend(model.graph.sparse_initializer)
        value_info.extend(model.graph.value_info)
        outputs.append(model.graph.output[0].name)
        for opset in model.opset_import:
            opsets[opset.domain] = max(opsets.get(opset.domain, 0), opset.version)
    initializers.append(helper.make_tensor("ensemble_half", TensorProto.FLOAT, [1], [0.5]))
    nodes.extend(
        [
            helper.make_node("Add", outputs, ["ensemble_sum"]),
            helper.make_node("Mul", ["ensemble_sum", "ensemble_half"], ["predictions"]),
        ]
    )
    graph = helper.make_graph(
        nodes, "evo1_2_ensemble",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, input_dim])],
        [helper.make_tensor_value_info("predictions", TensorProto.FLOAT, [None, 2])],
        initializer=initializers, sparse_initializer=sparse, value_info=value_info,
    )
    model = helper.make_model(
        graph, producer_name="evo1.2",
        opset_imports=[helper.make_opsetid(domain, version) for domain, version in opsets.items()],
    )
    model.ir_version = min(model.ir_version, 10)
    onnx.checker.check_model(model)
    onnx.save(model, output)


def export_selected(
    winner: str,
    final: dict[str, Any],
    output_dir: Path,
    sample: np.ndarray,
) -> dict[str, Any]:
    import onnx
    import onnxruntime as ort
    import openvino as ov

    output = output_dir / f"{MODEL_VERSION}.onnx"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mlp_path, lgbm_path = tmp / "mlp.onnx", tmp / "lgbm.onnx"
        export_mlp_onnx(final, sample, mlp_path)
        if winner == "dual_head_mlp":
            shutil.copy2(mlp_path, output)
            expected = mlp_predict(final, sample)
        elif winner == "lightgbm_dual_regressors":
            export_lightgbm_onnx(final, sample, lgbm_path)
            shutil.copy2(lgbm_path, output)
            expected = lgbm_predict(final, sample)
        else:
            export_lightgbm_onnx(final, sample, lgbm_path)
            combine_onnx_average(mlp_path, lgbm_path, output, sample.shape[1])
            expected = (mlp_predict(final, sample) + lgbm_predict(final, sample)) / 2.0

    onnx.checker.check_model(onnx.load(str(output)))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    onnx_output = session.run(["predictions"], {"features": sample.astype(np.float32)})[0]
    ov_dir = output_dir / "openvino"
    ov_dir.mkdir(parents=True, exist_ok=True)
    xml = ov_dir / f"{MODEL_VERSION}.xml"
    ov.save_model(ov.convert_model(str(output)), str(xml))
    compiled = ov.Core().compile_model(str(xml), "CPU")
    request = compiled.create_infer_request()
    ov_output = request.infer({"features": sample.astype(np.float32)})["predictions"]
    single = sample[:1].astype(np.float32)
    for _ in range(20):
        request.infer({"features": single})
    timings = []
    for _ in range(200):
        started = time.perf_counter_ns()
        request.infer({"features": single})
        timings.append((time.perf_counter_ns() - started) / 1e6)
    return {
        "onnx_matches_selected_model_within_1pct": bool(np.allclose(expected, onnx_output, rtol=0.01, atol=0.01)),
        "onnx_max_abs_difference": float(np.max(np.abs(expected - onnx_output))),
        "openvino_matches_selected_model_within_1pct": bool(np.allclose(expected, ov_output, rtol=0.01, atol=0.01)),
        "openvino_max_abs_difference": float(np.max(np.abs(expected - ov_output))),
        "openvino_ir_loaded": True,
        "single_sample_latency_ms_p50": float(np.percentile(timings, 50)),
        "single_sample_latency_ms_p95": float(np.percentile(timings, 95)),
        "single_sample_under_10ms_p95": bool(np.percentile(timings, 95) < 10),
    }


def permutation_importance(
    final: dict[str, Any], winner: str, seed: int = SEED
) -> dict[str, Any]:
    x = final["val_x"]
    actual, observed = target_arrays(final["val"])
    predictors: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "dual_head_mlp": lambda value: mlp_predict(final, value),
        "lightgbm_dual_regressors": lambda value: lgbm_predict(final, value),
        "mlp_lightgbm_oof_average": lambda value: (mlp_predict(final, value) + lgbm_predict(final, value)) / 2,
    }
    predict = predictors[winner]
    baseline = metrics(actual, predict(x), observed)
    rng = np.random.default_rng(seed)
    rows = []
    for index, name in enumerate(final["preprocessor"].feature_order):
        shuffled = x.copy()
        shuffled[:, index] = shuffled[rng.permutation(len(shuffled)), index]
        changed = metrics(actual, predict(shuffled), observed)
        rows.append(
            {
                "feature": name,
                "success_r2_drop": None if baseline["r2_success_pct"] is None else float(baseline["r2_success_pct"] - changed["r2_success_pct"]),
                "time_r2_drop": None if baseline["r2_time_min"] is None else float(baseline["r2_time_min"] - changed["r2_time_min"]),
                "is_live_hazard_feature": name.startswith(("severity_", "hazard_", "real_hazard_")) or name.startswith("hazard_source="),
            }
        )
    return {
        "method": "grouped-holdout permutation importance; positive R2 drop means useful",
        "baseline_holdout_metrics": baseline,
        "features": sorted(rows, key=lambda row: max(row["success_r2_drop"] or 0, row["time_r2_drop"] or 0), reverse=True),
        "live_feature_summary": [row for row in rows if row["is_live_hazard_feature"]],
        "warning": "Synthetic severity has an explicit target adjustment; its importance is not evidence of a real-world causal relationship.",
    }


def architecture(winner: str) -> dict[str, Any]:
    return {
        "name": f"Evo 1.2 {winner}",
        "model_version": MODEL_VERSION,
        "selected_model": winner,
        "layers": [
            {"id": "in", "type": "input", "label": "FCUSD + live hazards", "column": 0, "lane": 0.5},
            {"id": "mlp", "type": "dense", "label": "Dual-head MLP", "column": 1, "lane": 0.3},
            {"id": "lgbm", "type": "tree", "label": "LightGBM heads", "column": 1, "lane": 0.7},
            {"id": "select", "type": "dense", "label": winner, "column": 2, "lane": 0.5},
            {"id": "success", "type": "output", "label": "success %", "column": 3, "lane": 0.3},
            {"id": "time", "type": "output", "label": "time min", "column": 3, "lane": 0.7},
        ],
        "edges": [["in", "mlp"], ["in", "lgbm"], ["mlp", "select"], ["lgbm", "select"], ["select", "success"], ["select", "time"]],
    }


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items() if key != "oof_prediction"}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
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
    parser.add_argument("--hazard-seed", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evo1.2"))
    parser.add_argument("--max-epochs", type=int, default=220)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v11.seed_everything(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base, reference_audit = v11.load_reference(args.data)
    seed = load_seed(args.hazard_seed)
    joined = assign_real_hazards(base, seed)
    augmented = augment_labeled(joined)
    live = live_feature_frame(seed)
    pseudo = pseudo_label_live(live, joined)

    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(augmented, augmented["scenario"], augmented["row_id"]))
    print("training dual-head MLP", flush=True)
    mlp = cross_validate_mlp(
        augmented, pseudo, splits, max_epochs=args.max_epochs, patience=args.patience
    )
    print(json.dumps(mlp["aggregate_metrics"], sort_keys=True), flush=True)
    print("training LightGBM heads", flush=True)
    lgbm = cross_validate_lightgbm(augmented, pseudo, splits)
    print(json.dumps(lgbm["aggregate_metrics"], sort_keys=True), flush=True)
    ensemble_prediction = (mlp["oof_prediction"] + lgbm["oof_prediction"]) / 2
    actual, observed = target_arrays(augmented)
    ensemble = {
        "name": "mlp_lightgbm_oof_average",
        "oof_prediction": ensemble_prediction,
        "aggregate_metrics": metrics(actual, ensemble_prediction, observed),
    }
    candidates = [mlp, lgbm, ensemble]
    winner = max(candidates, key=candidate_score)
    print(f"selected {winner['name']}", flush=True)

    baselines = mean_and_knn_baselines(augmented, splits)
    final = train_final_models(
        augmented, pseudo, splits[0], max_epochs=args.max_epochs, patience=args.patience
    )
    export_metrics = export_selected(
        winner["name"], final, args.output_dir, final["val_x"][:32]
    )
    importance = permutation_importance(final, winner["name"])
    selected_metrics = winner["aggregate_metrics"]
    mean_metrics = baselines["mean"]["aggregate_metrics"]
    knn_metrics = baselines["knn_k25_production_distance"]["aggregate_metrics"]
    gates = {
        "cv_mae_success_under_2_pct": selected_metrics["mae_success_pct"] < 2,
        "cv_r2_success_at_least_0_25": selected_metrics["r2_success_pct"] >= 0.25,
        "cv_mae_time_under_1_min": selected_metrics["mae_time_min"] < 1,
        "cv_r2_time_at_least_0_75": selected_metrics["r2_time_min"] >= 0.75,
        "beats_mean_success": v11.beats(selected_metrics, mean_metrics, "success_pct"),
        "beats_mean_time": v11.beats(selected_metrics, mean_metrics, "time_min"),
        "beats_knn_success": v11.beats(selected_metrics, knn_metrics, "success_pct"),
        "beats_knn_time": v11.beats(selected_metrics, knn_metrics, "time_min"),
        "onnx_parity": export_metrics["onnx_matches_selected_model_within_1pct"],
        "openvino_parity": export_metrics["openvino_matches_selected_model_within_1pct"],
        "openvino_ir_loaded": export_metrics["openvino_ir_loaded"],
        "p95_inference_under_10ms": export_metrics["single_sample_under_10ms_p95"],
    }
    all_pass = all(gates.values())
    failed = [name for name, passed in gates.items() if not passed]
    ceiling = None
    if not all_pass:
        best_success = max(candidate["aggregate_metrics"]["r2_success_pct"] for candidate in candidates)
        best_time = max(candidate["aggregate_metrics"]["r2_time_min"] for candidate in candidates)
        ceiling = "DATA_CEILING" if best_success < 0.25 or best_time < 0.75 else "MODEL_CEILING"
    recommendation = (
        "promote_evo1.2"
        if all_pass
        else "evo_time_only_knn_success_and_risk_until_real_outcomes_expand"
    )
    audit = {
        **reference_audit,
        "hazard_seed_generated_at": seed.get("generated_at"),
        "hazard_feed_count": seed.get("hazard_count", len(seed.get("hazards", []))),
        "hazard_sources_active": seed.get("sources_active", []),
        "labeled_rows_with_real_hazard_join": int(joined["real_hazard_join"].sum()),
        "labeled_rows_without_nearby_hazard": int((~joined["real_hazard_join"]).sum()),
        "real_join_severity_nonzero_rows": int((joined["severity_score"] > 0).sum()),
        "real_join_magnitude_nonzero_rows": int((joined["hazard_magnitude"] > 0).sum()),
        "synthetic_variants_per_labeled_row": len(SYNTHETIC_SEVERITIES),
        "synthetic_rows": int(augmented["synthetic_augmentation"].sum()),
        "synthetic_rows_explicitly_flagged": True,
        "live_seed_rows": len(live),
        "live_seed_labels_available": 0,
        "live_seed_pseudo_labeled_for_training": len(pseudo),
        "live_seed_rows_counted_in_metrics": 0,
        "location_proxy_limitation": "Reference rows have no coordinates; deterministic category-compatible FCUSD monitoring spots were used for the spatial join.",
    }
    report = {
        "model_version": MODEL_VERSION,
        "evaluation_protocol": {
            "folds": N_FOLDS,
            "splitter": "StratifiedGroupKFold by scenario with row_id groups",
            "synthetic_variants_grouped": True,
            "unlabeled_live_rows_in_metrics": False,
        },
        "evo1.1_baseline_to_beat": {"success_r2": 0.182, "time_r2": 0.613, "openvino_p95_ms": 0.215},
        "model_comparison": {candidate["name"]: clean(candidate) for candidate in candidates},
        "selected_model": winner["name"],
        "cross_validation": {
            "aggregate_metrics": selected_metrics,
            "per_category": per_category(augmented, winner["oof_prediction"]),
        },
        "baseline_comparisons": baselines,
        "feature_importance": importance,
        "export_validation": export_metrics,
        "quality_gates": gates,
        "all_quality_gates_pass": all_pass,
        "failed_quality_gates": failed,
        "failure_classification": ceiling,
        "promotion_recommendation": recommendation,
        "minimum_new_data": [
            "PeopleSense live occupancy and density with timestamps for FCUSD spots",
            "real evacuation success labels for Office Building and Stadium categories",
            "egress geometry: exits, usable width, route length, and blockage state",
            "hazard-linked evacuation outcomes rather than synthetic severity adjustments",
        ],
        "honest_assessment": "Live public feeds add real hazard covariates but no evacuation outcomes. They cannot by themselves create supervised signal for success or time.",
    }
    curves = {
        "model_version": MODEL_VERSION,
        "source": "dual-head MLP grouped cross-validation average",
        **mlp["learning_curves"],
    }
    dashboard = {
        "status": "passed" if all_pass else "failed_quality_gate",
        "model_version": MODEL_VERSION,
        "message": recommendation,
        "selected_model": winner["name"],
        "train_loss": final["mlp_fit"].history["train_loss"],
        "val_loss": final["mlp_fit"].history["val_loss"],
        "val_mae_success_pct": selected_metrics["mae_success_pct"],
        "val_r2_success_pct": selected_metrics["r2_success_pct"],
        "val_mae_time_min": selected_metrics["mae_time_min"],
        "val_r2_time_min": selected_metrics["r2_time_min"],
        "all_quality_gates_pass": all_pass,
    }
    files = {
        "feature_schema.json": final["preprocessor"].schema(),
        "validation_report.json": report,
        "learning_curves.json": curves,
        "data_audit.json": audit,
        "architecture.json": architecture(winner["name"]),
        "metrics.json": dashboard,
    }
    for filename, payload in files.items():
        (args.output_dir / filename).write_text(json.dumps(clean(payload), indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "model_version": MODEL_VERSION,
                "selected_model": winner["name"],
                "cross_validation_metrics": selected_metrics,
                "all_quality_gates_pass": all_pass,
                "failed_quality_gates": failed,
                "failure_classification": ceiling,
                "promotion_recommendation": recommendation,
                "artifacts": str(args.output_dir.resolve()),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.strict and not all_pass:
        raise SystemExit("Evo 1.2 failed one or more strict quality gates")


if __name__ == "__main__":
    main()
