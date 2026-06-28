# Evo 1.2 Colab run summary

## Decision

**Failed mandatory R² gates; do not promote Evo 1.2 to production.**

The dual-head MLP beat LightGBM, the MLP/LightGBM OOF average, the mean
baseline, and production-distance k-NN. It passed MAE, ONNX/OpenVINO parity,
IR loading, and latency checks, but failed both predictive-signal gates.

| Metric | Evo 1.2 | Gate | Evo 1.1 | Status |
|---|---:|---:|---:|---|
| Success MAE | 1.647% | < 2.0% | 1.655% | Pass |
| Success R² | 0.184 | >= 0.25 | 0.182 | **Fail** |
| Time MAE | 0.830 min | < 1.0 min | 0.797 min | Pass |
| Time R² | 0.636 | >= 0.75 | 0.613 | **Fail** |

Failure classification: **DATA_CEILING**.

## Data and evaluation

- 4,082 reference rows reduced to 2,041 exact unique labeled examples.
- Refreshed Phase A feed snapshot: 253 NOAA/NWS, USGS, GDACS, NASA FIRMS, and FEMA
  IPAWS records across six monitoring-spot rows.
- All 2,041 labeled rows received a real, nonzero nearest-hazard join through a
  deterministic category-compatible FCUSD monitoring-spot proxy.
- 6,123 synthetic rows were explicitly flagged and kept in their source-row CV
  groups.
- Six unlabeled live rows received k-NN pseudo-labels for training only. Zero
  unlabeled rows were counted in outcome metrics.
- Five-fold `StratifiedGroupKFold` used scenario strata and source `row_id`
  groups.

The reference rows have no real coordinates or hazard timestamps, so the
monitoring-spot assignment is a documented proxy rather than an observed
hazard/outcome linkage.

## Model comparison

| Candidate | Success R² | Time R² |
|---|---:|---:|
| Dual-head MLP | **0.184** | **0.636** |
| LightGBM dual regressors | 0.136 | 0.498 |
| OOF average ensemble | 0.173 | 0.609 |

Because the ensemble did not win OOF, it was not exported. The selected MLP
was exported as the single Evo 1.2 ONNX/OpenVINO model.

## Live-feature importance

Grouped-holdout permutation importance showed:

- Severity success R² drop: 0.173. This is largely induced by the explicitly
  synthetic `success -= severity * 3` rule and is not causal evidence.
- Hazard magnitude time R² drop: 0.244.
- Hazard distance time R² drop: 0.109.
- Hazard source and real-join flags: no measurable R² movement.

Real public feeds add useful covariates but no evacuation outcomes; they cannot
create missing supervised signal by themselves.

## Export validation

- ONNX maximum absolute difference: 9.54e-7
- OpenVINO maximum absolute difference: 0.00166
- OpenVINO IR load: pass
- OpenVINO latency: p50 0.298 ms, p95 0.341 ms

## Production policy

The report records the requested hybrid fallback: Evo for evacuation time only,
with k-NN retained for success percentage and risk level. Because category-level
time performance remains poor—Office Building MAE is 18.99 minutes and Stadium
time R² is strongly negative—time-only use must be guarded and category-scoped.

Minimum new data: timestamped PeopleSense occupancy/density, real Office and
Stadium success outcomes, egress geometry, and real hazard-linked evacuation
outcomes.
