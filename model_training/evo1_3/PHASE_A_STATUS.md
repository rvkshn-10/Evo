# Evo Phase A status — 2026-06-28

## Completed

- Refreshed `hazard_live_seed.json`: 253 records from NOAA/NWS, USGS, GDACS,
  NASA FIRMS, and FEMA IPAWS; six monitoring-spot rows have nonzero severity and
  magnitude.
- Exported one current PeopleSense GET snapshot from 364 API records. All six
  configured monitoring spots matched their Place/Group/GPS rules.
- Re-ran the Evo 1.2 grouped baseline in Colab and verified ONNX/OpenVINO parity
  and CPU latency.
- Added the PeopleSense export helper, FCUSD data checklist, and guarded Evo 1.3
  Colab notebook.

## Refreshed Evo 1.2 result

| Metric | Result | Gate |
|---|---:|---:|
| Success MAE | 1.647% | < 2% (pass) |
| Success R² | 0.184 | >= 0.25 (fail) |
| Time MAE | 0.830 min | < 1 min (pass) |
| Time R² | 0.636 | >= 0.75 (fail) |
| OpenVINO p95 | 0.341 ms | < 10 ms (pass) |

Classification remains `DATA_CEILING`. Results are effectively unchanged from
the prior Evo 1.2 run because refreshed hazard and occupancy features do not add
measured evacuation outcomes.

## Phase B — synthetic demo trained (mentor preview)

Synthetic drill outcomes and aligned PeopleSense XML are in
`data/incoming/evo1.3/` (see `DEMO_SYNTHETIC_DATA.md`).

**Training completed 2026-06-28:** preflight exit 0, training exit 0,
status `complete_research_demo`. Artifacts copied to `models/evo1.3/`.
Metrics: Success R² 0.185, Time R² 0.596 → `DATA_CEILING`;
`keep_evo1.2_hybrid`. Full report: `model_training/evo1_3/PHASE_B_STATUS.md`.

**Still required for real promotion:** measured drill outcomes (not synthetic),
verified GPS/Group IDs, enough labeled rows to clear R² gates. Evo 1.2 hybrid
remains production policy.
