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

## Phase B remains blocked

The current PeopleSense file is a live feature snapshot, not a drill-aligned
label. FCUSD still needs to provide:

- measured `real_outcomes.json` rows;
- drill-timestamp PeopleSense exports for Vista del Lago, Folsom High, and
  Cordova Park;
- GGV2 deployment/Group ID confirmation for each site;
- GPS confirmation for all three sites;
- egress geometry where available.

The Evo 1.3 preflight returns code 2. Evo 1.3 was not trained or promoted, and
the Evo 1.2 hybrid production policy remains unchanged.
