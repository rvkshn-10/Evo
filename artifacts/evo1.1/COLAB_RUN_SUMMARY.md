# Evo 1.1 Colab run summary

## Decision

**Failed mandatory quality gates; do not promote this artifact bundle to production.**

The selected dual-head model improved on both validation baselines and passed the
MAE, export-parity, load, and latency checks. It did not reach either mandatory
R² threshold:

| Metric | Result | Gate | Status |
|---|---:|---:|---|
| Train Station success MAE | 1.655% | < 2.0% | Pass |
| Success R² | 0.182 | >= 0.25 | **Fail** |
| Time MAE | 0.797 min | < 1.0 min | Pass |
| Time R² | 0.613 | >= 0.75 | **Fail** |

## Run protocol

- Five-fold `StratifiedGroupKFold`, stratified by scenario and grouped by source
  row so synthetic variants cannot leak between train and validation.
- Full eight-configuration sweep: hidden dimensions 64/128, dropout 0.1/0.2,
  and learning rates 1e-3/3e-4.
- Dual independent success/time towers, masked success loss, success-shortfall
  target, engineered interaction/capacity features, severity augmentation,
  early stopping, and `ReduceLROnPlateau`.
- Selected configuration: hidden dimension 128, dropout 0.2, learning rate
  1e-3.

Five-fold mean +/- standard deviation:

- Success MAE: 1.655 +/- 0.078 percentage points
- Success R²: 0.182 +/- 0.018
- Time MAE: 0.799 +/- 0.169 minutes
- Time R²: 0.575 +/- 0.086

## Baselines and category warning

The model beat the mean and production-distance k-NN (k=25) baselines on both
aggregate targets. This is not sufficient for promotion because the absolute R²
gates failed.

The aggregate time MAE is dominated by Train Station rows. Office Building time
MAE was 20.208 minutes, and category-level time R² was negative for Train
Station, Office Building, and Stadium. Success labels exist only for Train
Station. The requested fallback recommendation is to use Evo for time only and
retain k-NN for success/risk until live PeopleSense and NOAA features exist, but
time-only deployment should still be category-scoped and independently guarded.

## Export validation

- ONNX/PyTorch max absolute difference: 4.77e-7 (pass)
- OpenVINO/PyTorch max absolute difference: 3.43e-4 (pass)
- OpenVINO IR load: pass
- OpenVINO single-sample latency: p50 0.191 ms, p95 0.215 ms (pass)

See `validation_report.json` for all fold results, baseline comparisons,
per-category metrics, quality-gate booleans, and the honest assessment.
