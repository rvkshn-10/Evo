# Evo — full technical reference (for slide authors)

Plain-language speaker notes are in **`SLIDES.md`**. This file is the deep reference.

---

## URLs & repos

| Item | Value |
|------|-------|
| Live dashboard | https://evac-evo.vercel.app |
| GitHub | https://github.com/rvkshn-10/Evo |
| Local UI | http://localhost:5173 |
| Local API | http://localhost:8092 |
| API docs | http://localhost:8092/docs |

---

## Run modes (`services/run_modes.py`)

| Mode | API | Predictor | LLM | Saves history? |
|------|-----|-----------|-----|--------------|
| `sync` | `POST /api/alerts/sync?mode=sync` | k-NN k=25 | No | Yes |
| `evo` | `?mode=evo` | Evo 1.2 hybrid | No | Yes |
| `evo13` | `?mode=evo13` | Evo 1.3 research | No | Yes |
| `external_ai` | `?mode=external_ai` | k-NN + summary | Gemini→OpenAI | Yes |
| `broadcast` | `?mode=broadcast` | k-NN (in agent path) | 7× GPT-4o | No |

### Sync
- `GovernmentFeedSync.sync_all()` → `AlertProcessor.get_dashboard_snapshot()`
- `EvacuationPredictor(use_evo=False)`
- `save_dashboard_snapshot(run_mode=sync)`

### Evo 1.2 hybrid
- `EvacuationPredictor(use_evo=True)` → `get_evo_runtime()` from `models/evo1.2/`
- Hybrid: k-NN for `predicted_evacuation_success_pct`, `risk_level`; neural for `predicted_evacuation_time_min` only when category ∈ `EVO_TIME_CATEGORIES` (default: Train Station)
- `inference_mode: hybrid`, model tag `evo1.2_hybrid`

### Evo 1.3 research
- `EvacuationPredictor(use_evo13=True)`
- Enriched reference: `data/processed/evacuation_reference_enriched_rows.json`
- Optional artifact blend from `models/evo1.3/` via `get_evo13_runtime()`
- `research_preview: true`, `production_promotion_allowed: false` when `synthetic_demo` in metrics
- Runtime: ONNX if OpenVINO parity gate failed (`services/evo_runtime.py::_openvino_parity_passed`)

### External AI
- After dashboard: `generate_intelligence_summary()` in `services/llm_router.py`
- Context: alert count, quake count, high-risk spots from snapshot JSON
- Not persisted to disaster DB

### Broadcast
- `run_full_agent_cycle()` in `services/agent_runner.py`
- Steps (`services/pipeline.py`): Coordinator → Evacuation Intelligence → Researcher → Panel (Bob + Fire Chief) → Writer → Producer → Script Writer
- Outputs under `output/{timestamp}_{event_type}/`

---

## Data pipeline

```
NOAA, USGS, GDACS, NASA FIRMS, FEMA IPAWS, PeopleSense GET
    → services/government_feed_sync.py
    → services/alert_processor.py
        → collect_live_hazards()
        → enrich_alert() per NOAA alert
        → enrich_spot_with_hazards() per monitoring spot
        → services/evacuation_predictor.py
    → output/dashboard/latest_snapshot.json
    → GET /api/dashboard
    → web/src/app.js renderSnapshot()
    → services/disaster_history.py (non-broadcast modes)
```

### Key files
- `services/government_feed_sync.py` — feed polling
- `services/alert_processor.py` — dashboard snapshot builder
- `services/hazard_feature_builder.py` — hazard covariates
- `services/evacuation_predictor.py` — k-NN / hybrid / evo13
- `config/monitoring_locations.json` — six FCUSD spots

### Risk band (`_risk_band`)
High if: evac_rate < 0.9 OR density > 0.85 OR occupancy > 1500

---

## Map (`web/src/app.js`)

- Library: Leaflet + heatmap plugin
- Layers: OSM tiles, `L.heatLayer` (earthquakes), `L.layerGroup` markers
- Filters: `HAZARD_FILTERS` — earthquake, wildfire, flood, tornado, tsunami, severe_weather, monitoring
- Heatmap source: `snapshot.heatmap_points` or fallback from earthquake list
- Center: first filtered quake or `map_center` (38.58, -121.30)

---

## Evo 1.2 training

- Script: `model_training/evo1_2/train_evo1_2.py`
- Colab deps: `model_training/evo1_2/requirements-colab.txt`
- Architecture: dual-head MLP + LightGBM, OOF ensemble average
- CV: 5-fold `StratifiedGroupKFold` by scenario, groups `row_id`
- Features: 10 numeric live hazard features + categoricals (`services/evo_features.py`)
- Export: `models/evo1.2/evo1.2.onnx`, `openvino/evo1.2.xml`, `metrics.json`, `validation_report.json`

### Phase A metrics (Evo 1.2)
| Metric | Value | Gate |
|--------|-------|------|
| Success MAE | 1.647% | < 2% ✓ |
| Success R² | 0.184 | ≥ 0.25 ✗ |
| Time MAE | 0.830 min | < 1 min ✓ |
| Time R² | 0.636 | ≥ 0.75 ✗ |
| OpenVINO p95 | 0.341 ms | < 10 ms ✓ |

Classification: `DATA_CEILING`  
Promotion: `evo_time_only_knn_success_and_risk_until_real_outcomes_expand`

---

## Evo 1.3 training (synthetic demo completed 2026-06-28)

- Script: `model_training/evo1_3/train_evo1_3.py`
- Inputs: `data/incoming/evo1.3/peoplesense/*.xml`, `real_outcomes.json`, `--coords-confirmed`
- Status: `complete_research_demo`
- Selected: `mlp_lightgbm_oof_average`
- Artifacts: `models/evo1.3/`

### Phase B metrics (synthetic)
| Metric | Value | Gate |
|--------|-------|------|
| Success MAE | 1.657% | ✓ |
| Success R² | 0.185 | ✗ |
| Time MAE | 0.873 min | ✓ |
| Time R² | 0.596 | ✗ |
| ONNX parity | pass | ✓ |
| OpenVINO parity | fail (Δ 62.7) | ✗ |

`synthetic_demo: true` → `production_promotion_allowed: false`  
`promotion_recommendation: keep_evo1.2_hybrid`

See `model_training/evo1_3/PHASE_B_STATUS.md`

---

## Model modal (`web/src/evo-viz.js`)

| UI element | Source | Meaning |
|------------|--------|---------|
| Live inference canvas | `GET /api/evo/live-flow` (2.8s poll) | DAG: Ingest → Features → Encode → MLP/LGBM → Merge → Outputs |
| Train/val loss chart | `models/evo1.2/metrics.json` | MLP training curves from grouped CV |
| Metrics panel | `GET /api/evo/visualization` | MAE, R², backend |
| Badge | `all_quality_gates_pass` | Gates passed vs Research preview |
| Edge tooltips | live-flow edge metadata | Raw → scaled feature values |

### Quality gates (both 1.2 and 1.3)
- cv_mae_success_under_2_pct
- cv_r2_success_at_least_0_25
- cv_mae_time_under_1_min
- cv_r2_time_at_least_0_75
- beats_mean_success/time, beats_knn_success/time
- onnx_parity, openvino_parity, openvino_ir_loaded
- p95_inference_under_10ms

---

## Disaster history (`services/disaster_history.py`)

**Tables:** `disaster_snapshots`, `hazard_events`, `evacuation_predictions`  
**Storage:** SQLite `data/disaster_history.db` or Neon via `DATABASE_URL`

| API | Purpose |
|-----|---------|
| `GET /api/history` | Recent snapshots |
| `GET /api/history/timeseries` | Chart data |
| `GET /api/history/high-risk` | High-risk prediction rows |
| `GET /api/history/export?format=json\|csv\|sqlite` | Download |

UI: `web/src/history.js`

---

## Deployment

### Vercel (`vercel.json`)
- Build: `cd web && npm run build` → `web/dist`
- Rewrites: `/api/*`, `/emergency/*`, `/backend/*` → Oracle `163.192.61.140:8092`

### Oracle
- Ubuntu ARM VM, port 8092, `systemd` service `emergency-api`
- Deploy: `git pull` + `sudo systemctl restart emergency-api`
- Docs: `docs/ORACLE_SETUP.md`

### CORS (`main.py`)
- localhost:5173, localhost:8092
- `CORS_ORIGINS` env + `https://.*\.vercel\.app`

---

## Neural Compute Stick

- **Hardware:** Intel NCS1/NCS2 USB
- **Software:** OpenVINO (`MYRIAD` device plugin)
- **Code:** `services/evo_accelerator.py`, `services/evo_runtime.py`
- **UI:** `web/src/accelerator.js`, Inference device dropdown
- **Cloud:** No USB — CPU only; banner on Vercel when Evo 1.2 selected
- **Setup guide:** `docs/NCS_SETUP.md`

Load order:
1. If NCS requested and OpenVINO parity passed → OpenVINO on MYRIAD
2. Else → ONNX Runtime on CPU

---

## Key API endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/dashboard` | Full snapshot |
| `POST /api/alerts/sync?mode=` | Run agent cycle |
| `GET /api/evo/runtime` | Model availability, backend, device |
| `GET /api/evo/live-flow` | Live inference trace for modal |
| `GET /api/evo/visualization` | Architecture + metrics for modal |
| `POST /api/evo/accelerator` | Switch CPU/NCS |
| `GET /health` | API status |

---

## File index

| Topic | Path |
|-------|------|
| Run modes | `services/run_modes.py` |
| Predictions | `services/evacuation_predictor.py` |
| Evo runtime | `services/evo_runtime.py` |
| Live viz | `services/evo_live_flow.py` |
| Dashboard UI | `web/src/app.js` |
| Model modal | `web/src/evo-viz.js` |
| History UI | `web/src/history.js` |
| Evo 1.2 train | `model_training/evo1_2/train_evo1_2.py` |
| Evo 1.3 train | `model_training/evo1_3/train_evo1_3.py` |
| NCS setup | `docs/NCS_SETUP.md` |
