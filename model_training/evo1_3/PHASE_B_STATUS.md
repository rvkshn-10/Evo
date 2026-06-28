# Evo Phase B status — synthetic mentor demo (2026-06-28)

## Training completed locally

| Step | Result |
|------|--------|
| Preflight | Exit `0` |
| PeopleSense rows | `12` (6 drill XML + 6-zone snapshot) |
| Synthetic outcomes | `6` |
| Training | Exit `0` · status `complete_research_demo` |
| Selected model | `mlp_lightgbm_oof_average` (MLP + LightGBM OOF ensemble) |

## Cross-validation metrics

| Metric | Result | Gate | Pass |
|--------|--------|------|------|
| Success MAE | 1.657% | < 2% | ✅ |
| Success R² | 0.185 | ≥ 0.25 | ❌ |
| Time MAE | 0.873 min | < 1 min | ✅ |
| Time R² | 0.596 | ≥ 0.75 | ❌ |
| ONNX parity | pass | within 1% | ✅ |
| OpenVINO IR loaded | yes | — | ✅ |
| OpenVINO parity | fail (max Δ 62.7) | within 1% | ❌ |
| p95 inference | 21.4 ms | < 10 ms | ❌ |

## Classification & policy

- **Failure class:** `DATA_CEILING`
- **Promotion:** `keep_evo1.2_hybrid`
- **`synthetic_demo`:** `true` → `production_promotion_allowed: false`
- **Runtime:** Evo 1.3 research loads via **ONNX** (OpenVINO skipped when parity gate fails)
- **Production:** Evo 1.2 hybrid unchanged

## Artifacts shipped

```
models/evo1.3/
  evo1.3.onnx
  openvino/evo1.3.xml + evo1.3.bin
  metrics.json
  validation_report.json
  feature_schema.json
  architecture.json
```

Inputs: `data/incoming/evo1.3/real_outcomes.json` + `peoplesense/drill-*.xml`

**For mentors:** pipeline proved end-to-end. Metrics are illustrative until measured drill data replaces synthetic rows.
