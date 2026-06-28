"""Intel Neural Compute Stick (NCS1/NCS2) and CPU accelerator probing for Evo."""

from __future__ import annotations

import platform
from typing import Any, Literal, Optional

AcceleratorChoice = Literal["auto", "cpu", "ncs1", "ncs2"]

ACCELERATOR_OPTIONS: list[dict[str, str]] = [
    {"id": "auto", "label": "Auto (detect USB stick)"},
    {"id": "cpu", "label": "CPU only (ONNX / OpenVINO)"},
    {"id": "ncs2", "label": "Neural Compute Stick 2"},
    {"id": "ncs1", "label": "Neural Compute Stick 1"},
]

_runtime_override: Optional[AcceleratorChoice] = None


def normalize_accelerator(value: Optional[str]) -> AcceleratorChoice:
    key = (value or "auto").strip().lower()
    if key in {"auto", "cpu", "ncs1", "ncs2"}:
        return key  # type: ignore[return-value]
    return "auto"


def get_requested_accelerator(default: str = "auto") -> AcceleratorChoice:
    if _runtime_override is not None:
        return _runtime_override
    return normalize_accelerator(default)


def set_runtime_accelerator(value: str) -> AcceleratorChoice:
    global _runtime_override
    choice = normalize_accelerator(value)
    _runtime_override = choice
    return choice


def _classify_myriad_name(full_name: str) -> str:
    lower = full_name.lower()
    if any(token in lower for token in ("myriadx", "ma2485", "ncs2", "movidius myriad x")):
        return "ncs2"
    if any(token in lower for token in ("myriad2", "ma2450", "ncs1", "movidius myriad 2")):
        return "ncs1"
    if "myriad" in lower:
        return "myriad"
    return "unknown"


def probe_myriad_devices() -> list[dict[str, str]]:
    """Return MYRIAD devices visible to OpenVINO (NCS sticks over USB)."""
    try:
        try:
            from openvino import Core
        except ImportError:
            from openvino.runtime import Core

        core = Core()
        devices: list[dict[str, str]] = []
        for device_id in core.available_devices:
            if not str(device_id).upper().startswith("MYRIAD"):
                continue
            full_name = ""
            for prop in ("FULL_DEVICE_NAME", "DEVICE_ID", "DEVICE_GAV_VERSION"):
                try:
                    full_name = str(core.get_property(device_id, prop))
                    if full_name:
                        break
                except Exception:
                    continue
            variant = _classify_myriad_name(full_name or str(device_id))
            devices.append(
                {
                    "id": str(device_id),
                    "variant": variant,
                    "name": full_name or str(device_id),
                }
            )
        return devices
    except Exception:
        return []


def probe_accelerator_status(default_choice: str = "auto") -> dict[str, Any]:
    myriad_devices = probe_myriad_devices()
    ncs1_detected = any(d["variant"] == "ncs1" for d in myriad_devices)
    ncs2_detected = any(d["variant"] == "ncs2" for d in myriad_devices)
    any_myriad = bool(myriad_devices)

    return {
        "accelerator_requested": get_requested_accelerator(default_choice),
        "accelerator_options": [opt["id"] for opt in ACCELERATOR_OPTIONS],
        "accelerator_labels": {opt["id"]: opt["label"] for opt in ACCELERATOR_OPTIONS},
        "myriad_devices": myriad_devices,
        "ncs1_detected": ncs1_detected,
        "ncs2_detected": ncs2_detected,
        "myriad_detected": any_myriad,
        "host_platform": platform.system().lower(),
        "host_machine": platform.machine().lower(),
        "openvino_importable": _openvino_importable(),
    }


def _openvino_importable() -> bool:
    try:
        try:
            from openvino import Core  # noqa: F401
        except ImportError:
            from openvino.runtime import Core  # noqa: F401
        return True
    except Exception:
        return False


def resolve_openvino_device(requested: AcceleratorChoice, myriad_devices: list[dict[str, str]]) -> tuple[Optional[str], str]:
    """
    Pick an OpenVINO device string.

    Returns (device_name, resolved_accelerator_label).
    device_name is None when OpenVINO should not be used for this choice.
    """
    if requested == "cpu":
        return "CPU", "cpu"

    if requested == "ncs2":
        for device in myriad_devices:
            if device["variant"] == "ncs2":
                return device["id"], "ncs2"
        if myriad_devices:
            return myriad_devices[0]["id"], "ncs2"
        return None, "ncs2"

    if requested == "ncs1":
        for device in myriad_devices:
            if device["variant"] == "ncs1":
                return device["id"], "ncs1"
        if myriad_devices:
            return myriad_devices[0]["id"], "ncs1"
        return None, "ncs1"

    # auto
    for variant in ("ncs2", "ncs1"):
        for device in myriad_devices:
            if device["variant"] == variant:
                return device["id"], variant
    if myriad_devices:
        return myriad_devices[0]["id"], myriad_devices[0]["variant"]
    return "CPU", "cpu"


def accelerator_status_message(
    *,
    requested: AcceleratorChoice,
    resolved: str,
    active_device: Optional[str],
    loaded: bool,
    backend: Optional[str],
    myriad_devices: list[dict[str, str]],
) -> str:
    if loaded and backend == "openvino" and active_device and active_device.upper().startswith("MYRIAD"):
        stick = "NCS2" if resolved == "ncs2" else "NCS1" if resolved == "ncs1" else "Neural Compute Stick"
        return f"Evo running on {stick} via {active_device}."
    if loaded and backend == "openvino" and active_device == "CPU":
        return "Evo running on CPU via OpenVINO."
    if loaded and backend == "onnxruntime":
        if requested in {"ncs1", "ncs2"} and not myriad_devices:
            return (
                f"No Neural Compute Stick detected for {requested.upper()}. "
                "Plug the USB stick into this Mac and restart the API, or choose CPU."
            )
        return "Evo running on CPU via ONNX Runtime."
    if requested in {"ncs1", "ncs2"} and not myriad_devices:
        return "Stick not detected — plug NCS into USB and click Check again."
    return "Model not loaded — confirm models/evo1.2/ artifacts and restart the API."
