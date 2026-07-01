# Codex prompt — Train Evo 1.4 for map-pin egress & blocked-route detours (no LLM at inference)

Copy everything below the line into Codex / ChatGPT Codex / Cursor Agent.

---

## Task

Extend the Evo evacuation ML pipeline so **map-picked locations**, **user-blocked exits/hazards**, and **optional building blueprint metadata** are handled entirely by **trained Evo models** — **not** Gemini or OpenAI at inference time.

Repo: `Emergency Management Office AI` (Evo / FCUSD evacuation intelligence).

## Context (read first)

- Dashboard: user clicks map → `POST /api/location/analyze` → `services/location_evac_analysis.py`
- Logs training rows: `data/incoming/evo1.3/pin_location_analyses.jsonl`
- Current Evo trainer: `model_training/evo1_3/train_evo1_3.py`
- Feature encoder: `services/evo_features.py` (already has `egress_*` scaled features)
- PeopleSense: `services/peoplesense_client.py`
- **Do not** add Gemini/OpenAI calls to the hot path for route ranking or blueprint parsing.

## Goals

1. **Route ranking head** — Given occupancy, density, egress geometry, blocked headings, hazard points, and blueprint features → predict:
   - `best_compass_heading` (classification: N, NE, E, …)
   - `estimated_clear_time_min` (regression)
   - Keep existing `predicted_evacuation_success_pct` and `predicted_evacuation_time_min` heads.

2. **Blueprint feature ingestion (offline only)** — One-time or batch script (not LLM):
   - Input: public URL or local PDF/PNG path in `config/site_blueprints.json` (you create)
   - Extract **structured features only**: `exit_count`, `stairwell_count`, `floor_count`, `longest_corridor_m`, `usable_exit_width_m`, `assembly_points_count`
   - Methods allowed: PDF text extraction (`pypdf`), image metadata, manual JSON overrides per FCUSD site — **no** paid vision API required for training pipeline
   - Optional: use open campus map GeoJSON if found; store in `data/reference/site_blueprints/`

3. **Blocked egress training data** — Merge:
   - `data/incoming/evo1.3/pin_location_analyses.jsonl` (has `blocked_headings`, `blocked_points`, `blockage_reason`)
   - `data/incoming/evo1.3/real_outcomes.json`
   - `data/processed/evacuation_reference.json`
   - Synthetic augment rows where `egress_blockage_fraction` ∈ [0.1, 0.9] and random blocked cardinal directions — label = alternate detour heading with lowest clear time (use existing OSRM logic from `location_evac_analysis.py` as teacher)

4. **Model version** — Ship as `evo1.4` (or `evo1.3.1` if you prefer patch):
   - Output dir: `models/evo1.4/` and `artifacts/evo1.4/`
   - Update `config/settings.py` `EVO13_MODEL_VERSION` only if promoting; add `EVO14_MODEL_VERSION`
   - Wire `EvacuationPredictor(use_evo14=True)` and `/api/location/analyze` to use route head when available

## New features to add to `feature_schema.json`

Add numeric (with normalization in trainer):

| Feature | Description |
|---------|-------------|
| `blocked_exit_fraction_scaled` | len(blocked_headings) / 8 |
| `hazard_point_count_scaled` | count of blocked map hazards |
| `blueprint_exit_count_scaled` | from blueprint JSON |
| `blueprint_floor_count_scaled` | floors in building |
| `blueprint_corridor_length_scaled` | longest corridor m |
| `detour_required_flag_scaled` | 1 if any cardinal heading blocked |

## Training script requirements

Create `model_training/evo1_4/train_evo1_4.py` that:

1. Reuses Evo 1.3 preflight checks (PeopleSense dirs, real outcomes, coords confirmed).
2. Loads pin JSONL + enriches rows with blueprint features via `spot_id` or lat/lon nearest match to `config/monitoring_locations.json`.
3. Uses **teacher labels** from `location_evac_analysis._rank_evacuation_routes()` for rows missing `best_compass_heading` label.
4. Multi-task loss: success %, evac time, **route heading CE**, **clear time MSE**.
5. Exports ONNX like Evo 1.3; update `services/evo_runtime.py` to load `evo1.4` when present.
6. Writes `validation_report.json` with gates:
   - Route heading accuracy ≥ 0.55 on held-out pin rows (research demo bar)
   - Clear time MAE ≤ 2.0 min on pin rows
   - Existing success/time gates from Evo 1.3 or document `DATA_CEILING` if only synthetic labels

## Runtime changes

1. `services/location_evac_analysis.py`:
   - After Evo 1.4 available, call `predict_route_head()` instead of pure OSRM heuristic rank when `use_evo14=True`.
   - Keep OSRM as fallback if model missing.

2. `services/blueprint_features.py` (new):
   - `load_blueprint_for_spot(spot_id)` → dict
   - `load_blueprint_for_point(lat, lon)` → nearest site blueprint
   - `extract_blueprint_features(url_or_path)` → structured dict (offline script)

3. **Remove dependency on LLM** for location analyze default (`use_llm=False` already default). Add `generate_rule_based_briefing()` using template strings from model outputs — no API credits.

## Blueprint discovery (offline script)

Create `scripts/discover_site_blueprints.py`:

- For each spot in `config/monitoring_locations.json`, search hints:
  - School district PDF egress plans (manual URL list in `config/site_blueprints.json` first)
  - OpenStreetMap building footprints via Overpass API (`building=*` at lat/lon)
- Do **not** scrape behind logins.
- Output: `data/reference/site_blueprints/{spot_id}.json` with `source_url`, `extracted_features`, `confidence`, `extracted_at`.

## Files you will likely touch

```
model_training/evo1_4/train_evo1_4.py          (new)
model_training/evo1_4/DATA_COLLECTION.md     (update)
services/blueprint_features.py               (new)
services/location_evac_analysis.py           (evo14 route head)
services/evo_features.py                     (new feature columns)
services/evo_runtime.py
services/evacuation_predictor.py
config/site_blueprints.json                  (new, manual URLs ok)
data/reference/site_blueprints/              (new)
scripts/discover_site_blueprints.py          (new)
routes/intelligence.py                       (use_evo14 flag)
```

## Commands to verify

```bash
# Preflight
python model_training/evo1_4/train_evo1_4.py --preflight-only \
  --peoplesense-dir data/incoming/evo1.3/peoplesense \
  --real-outcomes data/incoming/evo1.3/real_outcomes.json \
  --pin-analyses data/incoming/evo1.3/pin_location_analyses.jsonl \
  --coords-confirmed

# Train (research demo with synthetic pin rows ok)
python model_training/evo1_4/train_evo1_4.py \
  --peoplesense-dir data/incoming/evo1.3/peoplesense \
  --real-outcomes data/incoming/evo1.3/real_outcomes.json \
  --pin-analyses data/incoming/evo1.3/pin_location_analyses.jsonl \
  --output-dir models/evo1.4

# API smoke test
python3 -c "
from services.location_evac_analysis import analyze_map_location
r = analyze_map_location(lat=38.5616, lon=-121.4246, blocked_headings=[0,90], blockage_reason='fire')
print(r['recommended_route'])
"
```

## Constraints

- **No Gemini/OpenAI in inference path** for route or blueprint analysis.
- Label research outputs `production_approved: false` until FCUSD drill validation.
- Indoor CAD accuracy is not claimed — blueprint features are **assistive**, validated against drills.
- Match existing code style in `train_evo1_3.py` and `location_evac_analysis.py`.
- Minimize diff scope; do not break Evo 1.2 hybrid production policy.

## Deliverables

1. Working `train_evo1_4.py` + artifacts
2. `PHASE_B_STATUS.md` update for route-head metrics
3. Dashboard uses Evo route head when model present; blocked exits + detours work without LLM
4. `config/site_blueprints.json` template with Sac State / Folsom High placeholder URLs for FCUSD to fill in

Start by reading `model_training/evo1_3/train_evo1_3.py`, `services/location_evac_analysis.py`, and `data/incoming/evo1.3/pin_location_analyses.jsonl`, then implement.
