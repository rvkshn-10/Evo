# Evo 1.4 Phase B status

## Colab research run

- Preflight: passed (exit 0)
- Training: completed (exit 0)
- Route heading accuracy: 0.9979
- Estimated clear-time MAE: 1.4295 minutes
- ONNX parity: passed
- OpenVINO parity/load: passed
- p95 single-sample inference: under 10 ms

The route-head feasibility gates passed. The evacuation outcome gates did not: success R² was 0.0951, evacuation-time MAE was 1.1150 minutes, and time R² was -0.9387.

This run used 2,400 explicitly synthetic, teacher-labeled route rows because `pin_location_analyses.jsonl` did not yet exist. It is not FCUSD drill evidence. `production_approved` remains `false`; keep Evo 1.2 hybrid in production.

Before promotion, collect real pin selections, blocked-route decisions, measured clear times, timestamp-aligned PeopleSense readings, and validated structured blueprint counts across Folsom High, Vista del Lago, and Cordova Park.
