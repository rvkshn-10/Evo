"""Evo 1.0 model runtime — OpenVINO / ONNX with visualization metadata."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

MODEL_DIR = settings.PROJECT_ROOT / "models" / "evo1.0"


class EvoRuntime:
    """Load Evo 1.0 artifacts when present; otherwise report unavailable."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self._compiled = None
        self._backend: Optional[str] = None

    @property
    def is_available(self) -> bool:
        return (self.model_dir / "evo1.0.onnx").exists() or (
            self.model_dir / "openvino" / "evo1.0.xml"
        ).exists()

    def _load_openvino(self) -> bool:
        xml_path = self.model_dir / "openvino" / "evo1.0.xml"
        if not xml_path.exists():
            return False
        try:
            from openvino.runtime import Core

            core = Core()
            self._compiled = core.compile_model(str(xml_path), "CPU")
            self._backend = "openvino"
            logger.info("Evo 1.0 loaded via OpenVINO")
            return True
        except Exception as exc:
            logger.warning("OpenVINO load failed: %s", exc)
            return False

    def _load_onnx(self) -> bool:
        onnx_path = self.model_dir / "evo1.0.onnx"
        if not onnx_path.exists():
            return False
        try:
            import onnxruntime as ort

            self._compiled = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
            self._backend = "onnxruntime"
            logger.info("Evo 1.0 loaded via ONNX Runtime")
            return True
        except Exception as exc:
            logger.warning("ONNX Runtime load failed: %s", exc)
            return False

    def ensure_loaded(self) -> bool:
        if self._compiled is not None:
            return True
        if settings.EVO_PREFER_OPENVINO and self._load_openvino():
            return True
        return self._load_onnx()

    def predict(self, features: list[float]) -> Optional[dict[str, float]]:
        if not self.ensure_loaded():
            return None
        try:
            if self._backend == "openvino":
                result = self._compiled([features])[0]
                outputs = list(result.values())[0][0]
            else:
                input_name = self._compiled.get_inputs()[0].name
                outputs = self._compiled.run(None, {input_name: [features]})[0][0]
            return {
                "evacuation_success_pct": float(outputs[0]),
                "evacuation_time_min": float(outputs[1]),
            }
        except Exception as exc:
            logger.error("Evo inference failed: %s", exc)
            return None

    def get_visualization(self) -> dict[str, Any]:
        arch_path = self.model_dir / "architecture.json"
        metrics_path = self.model_dir / "metrics.json"
        schema_path = self.model_dir / "feature_schema.json"

        architecture = _read_json(arch_path) or _default_architecture()
        metrics = _read_json(metrics_path) or _placeholder_metrics()
        schema = _read_json(schema_path) or {}

        return {
            "model_version": settings.EVO_MODEL_VERSION,
            "available": self.is_available,
            "backend": self._backend,
            "openvino_connected": self._backend == "openvino",
            "architecture": architecture,
            "metrics": metrics,
            "feature_schema": schema,
            "repo": settings.EVO_MODEL_REPO,
        }


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _default_architecture() -> dict[str, Any]:
    return {
        "name": "Evo 1.0",
        "layers": [
            {"id": "in", "type": "input", "label": "Features\n(occupancy, density, hazard)"},
            {"id": "h1", "type": "dense", "label": "Dense 64\nReLU + Dropout"},
            {"id": "h2", "type": "dense", "label": "Dense 32\nReLU"},
            {"id": "out", "type": "output", "label": "Outputs\nsuccess % · time min"},
        ],
        "edges": [["in", "h1"], ["h1", "h2"], ["h2", "out"]],
    }


def _placeholder_metrics() -> dict[str, Any]:
    return {
        "status": "awaiting_training",
        "message": "Train Evo 1.0 in Colab and commit models/evo1.0/metrics.json",
        "train_loss": [],
        "val_loss": [],
        "val_mae_success_pct": None,
        "val_mae_time_min": None,
    }


_evo_runtime: Optional[EvoRuntime] = None


def get_evo_runtime() -> EvoRuntime:
    global _evo_runtime
    if _evo_runtime is None:
        _evo_runtime = EvoRuntime()
    return _evo_runtime
