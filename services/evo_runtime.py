"""Version-aware Evo model runtime with OpenVINO/ONNX and NCS accelerator support."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from config.settings import settings
from services.evo_accelerator import (
    accelerator_status_message,
    get_requested_accelerator,
    probe_accelerator_status,
    resolve_openvino_device,
    set_runtime_accelerator,
)

logger = logging.getLogger(__name__)

MODEL_VERSION = settings.EVO_MODEL_VERSION
MODEL_DIR = settings.PROJECT_ROOT / "models" / MODEL_VERSION


class EvoRuntime:
    """Load the configured Evo artifacts when present; otherwise report unavailable."""

    def __init__(self, model_dir: Path = MODEL_DIR, model_version: str = MODEL_VERSION):
        self.model_dir = model_dir
        self.model_version = model_version
        self._compiled = None
        self._backend: Optional[str] = None
        self._active_device: Optional[str] = None
        self._resolved_accelerator: str = "cpu"
        self._last_load_error: Optional[str] = None

    @property
    def is_available(self) -> bool:
        return (self.model_dir / f"{self.model_version}.onnx").exists() or (
            self.model_dir / "openvino" / f"{self.model_version}.xml"
        ).exists()

    def reset_runtime(self) -> None:
        self._compiled = None
        self._backend = None
        self._active_device = None
        self._last_load_error = None

    def set_accelerator(self, value: str) -> dict[str, Any]:
        set_runtime_accelerator(value)
        self.reset_runtime()
        self.ensure_loaded()
        return self.get_runtime_status()

    def _load_openvino(self, device: str) -> bool:
        xml_path = self.model_dir / "openvino" / f"{self.model_version}.xml"
        if not xml_path.exists():
            return False
        try:
            try:
                from openvino import Core
            except ImportError:
                from openvino.runtime import Core

            core = Core()
            self._compiled = core.compile_model(str(xml_path), device)
            self._backend = "openvino"
            self._active_device = device
            logger.info("%s loaded via OpenVINO on %s", self.model_version, device)
            return True
        except Exception as exc:
            self._last_load_error = str(exc)
            logger.warning("OpenVINO load failed on %s: %s", device, exc)
            return False

    def _load_onnx(self) -> bool:
        onnx_path = self.model_dir / f"{self.model_version}.onnx"
        if not onnx_path.exists():
            return False
        try:
            import onnxruntime as ort

            self._compiled = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
            self._backend = "onnxruntime"
            self._active_device = "CPU"
            self._resolved_accelerator = "cpu"
            logger.info("%s loaded via ONNX Runtime", self.model_version)
            return True
        except Exception as exc:
            self._last_load_error = str(exc)
            logger.warning("ONNX Runtime load failed: %s", exc)
            return False

    def _openvino_parity_passed(self) -> bool:
        """Refuse an IR artifact that its own validation report marks unsafe."""
        report = _read_json(self.model_dir / "validation_report.json")
        gates = (report or {}).get("quality_gates", {})
        return gates.get("openvino_parity") is not False

    def ensure_loaded(self) -> bool:
        if self._compiled is not None:
            return True

        probe = probe_accelerator_status(settings.EVO_ACCELERATOR)
        requested = get_requested_accelerator(settings.EVO_ACCELERATOR)
        device, resolved = resolve_openvino_device(requested, probe["myriad_devices"])
        self._resolved_accelerator = resolved

        use_openvino = settings.EVO_PREFER_OPENVINO or requested in {"ncs1", "ncs2", "auto"}
        if use_openvino and device and self._openvino_parity_passed():
            if self._load_openvino(device):
                return True
        elif use_openvino and device:
            self._last_load_error = (
                "OpenVINO parity validation failed; using verified ONNX Runtime artifact."
            )

        if requested in {"ncs1", "ncs2"}:
            logger.warning(
                "Requested %s but MYRIAD device unavailable; falling back to ONNX CPU",
                requested,
            )

        return self._load_onnx()

    def predict(self, features: list[float]) -> Optional[dict[str, float]]:
        if not self.ensure_loaded():
            return None
        try:
            import numpy as np

            vector = np.asarray(features, dtype=np.float32).reshape(1, -1)
            if self._backend == "openvino":
                result = self._compiled(vector)
                outputs = np.asarray(list(result.values())[0])[0]
            else:
                input_name = self._compiled.get_inputs()[0].name
                outputs = self._compiled.run(None, {input_name: vector})[0][0]
            return {
                "evacuation_success_pct": float(outputs[0]),
                "evacuation_time_min": float(outputs[1]),
            }
        except Exception as exc:
            logger.error("Evo inference failed: %s", exc)
            return None

    def get_runtime_status(self) -> dict[str, Any]:
        """Probe load and report which inference backend is active."""
        loaded = self.ensure_loaded()
        probe = probe_accelerator_status(settings.EVO_ACCELERATOR)
        requested = probe["accelerator_requested"]
        status = {
            "model_version": self.model_version,
            "available": self.is_available,
            "loaded": loaded,
            "backend": self._backend,
            "device": self._active_device,
            "accelerator": self._resolved_accelerator,
            "accelerator_requested": requested,
            "openvino_connected": self._backend == "openvino",
            "openvino_ir_present": (
                self.model_dir / "openvino" / f"{self.model_version}.xml"
            ).exists(),
            "onnx_present": (self.model_dir / f"{self.model_version}.onnx").exists(),
            "prefer_openvino": settings.EVO_PREFER_OPENVINO,
            "last_load_error": self._last_load_error,
            "status_message": accelerator_status_message(
                requested=requested,
                resolved=self._resolved_accelerator,
                active_device=self._active_device,
                loaded=loaded,
                backend=self._backend,
                myriad_devices=probe["myriad_devices"],
            ),
        }
        status.update(probe)
        return status

    def get_visualization(self) -> dict[str, Any]:
        arch_path = self.model_dir / "architecture.json"
        metrics_path = self.model_dir / "metrics.json"
        schema_path = self.model_dir / "feature_schema.json"

        architecture = _read_json(arch_path) or _default_architecture()
        metrics = _read_json(metrics_path) or _placeholder_metrics()
        schema = _read_json(schema_path) or {}

        return {
            "model_version": self.model_version,
            "available": self.is_available,
            "backend": self._backend,
            "accelerator": self._resolved_accelerator,
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
        "name": MODEL_VERSION,
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
        "message": f"Train {MODEL_VERSION} in Colab and add models/{MODEL_VERSION}/metrics.json",
        "train_loss": [],
        "val_loss": [],
        "val_mae_success_pct": None,
        "val_mae_time_min": None,
    }


_evo_runtime: Optional[EvoRuntime] = None
_evo13_runtime: Optional[EvoRuntime] = None


def get_evo_runtime() -> EvoRuntime:
    global _evo_runtime
    if _evo_runtime is None:
        _evo_runtime = EvoRuntime()
    return _evo_runtime


def get_evo13_runtime() -> EvoRuntime:
    global _evo13_runtime
    if _evo13_runtime is None:
        version = settings.EVO13_MODEL_VERSION
        _evo13_runtime = EvoRuntime(
            model_dir=settings.PROJECT_ROOT / "models" / version,
            model_version=version,
        )
    return _evo13_runtime
