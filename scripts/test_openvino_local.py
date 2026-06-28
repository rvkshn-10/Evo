#!/usr/bin/env python3
"""Verify Evo OpenVINO inference locally (run after: pip install openvino)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings
from services.evo_features import encode_features, load_feature_schema
from services.evo_runtime import EvoRuntime


def main() -> None:
    version = settings.EVO_MODEL_VERSION
    print(f"Model: {version}")
    print(f"EVO_PREFER_OPENVINO={settings.EVO_PREFER_OPENVINO}")

    runtime = EvoRuntime()
    if not runtime.is_available:
        print("FAIL: model files missing in models/", version)
        sys.exit(1)

    if not runtime.ensure_loaded():
        print("FAIL: could not load ONNX or OpenVINO")
        sys.exit(1)

    print(f"Backend: {runtime._backend}")
    schema = load_feature_schema(version)
    features = encode_features(
        schema=schema,
        occupancy=950,
        density=0.58,
        category="Train Station",
        scenario="Electrical Fire",
        event_type="fire",
        severity_score=0.6,
        hazard_magnitude=5.0,
        hazard_distance_km=120.0,
        hazard_source="usgs",
        real_hazard_join=True,
    )
    if not features:
        print("FAIL: feature encoding returned empty vector")
        sys.exit(1)

    start = time.perf_counter()
    for _ in range(100):
        out = runtime.predict(features)
    elapsed_ms = (time.perf_counter() - start) / 100 * 1000

    print(f"Sample output: {out}")
    print(f"Latency (100 runs avg): {elapsed_ms:.3f} ms")
    if runtime._backend == "openvino":
        print("PASS: OpenVINO loaded and inference works")
    else:
        print("NOTE: OpenVINO not used — install with: pip install openvino")
        print(f"      Currently using: {runtime._backend}")


if __name__ == "__main__":
    main()
