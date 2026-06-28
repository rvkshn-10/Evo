# Project Notes

## Goal

Build an **AI agent** that autonomously coordinates emergency evacuation intelligence:

1. Ingest government hazard warnings (NOAA/NWS)
2. Enrich with PeopleSense crowd occupancy
3. Predict evacuation rates using reference datasets (FCUSD / Kaggle-style data)
4. Publish findings to the public dashboard and event reports

## Core agent

**Evacuation Intelligence Agent** (`agents/evacuation_intelligence/agent.py`)

A LangChain agent (GPT-4o) with tools for NOAA alerts, PeopleSense occupancy,
evacuation prediction, and dashboard publishing. It runs:

- **Autonomously** via `POST /api/agent/run` — no manual event required
- **In the pipeline** after the Emergency Coordinator when an event is submitted

### Agent tools

| Tool | Purpose |
|------|---------|
| `fetch_government_alerts` | NOAA/NWS active warnings |
| `get_peoplesense_occupancy` | Real-time crowd data per zone |
| `predict_evacuation_rate` | ML-style prediction from reference data |
| `list_monitoring_spots` | Schools, shelters, transit hubs |
| `enrich_alert_with_evacuation_analysis` | Full alert + occupancy + predictions |
| `publish_dashboard_snapshot` | Push results to the website |

## Data flow

```
Evacuation Intelligence Agent (GPT-4o)
    │
    ├── fetch_government_alerts (NOAA)
    ├── get_peoplesense_occupancy (PeopleSense API)
    ├── predict_evacuation_rate (reference dataset model)
    │
    └── publish_dashboard_snapshot → website
```

## Reference materials

- `docs/AI Project FCUSD.pdf` — FCUSD project brief
- `data/reference/model-step1-data.xlsx` — evacuation modeling source data (4,082 records)
- `data/processed/evacuation_reference.json` — normalized training/reference records used by the predictor

## Data flow

```
Evacuation Intelligence Agent
    ├── NOAA alerts
    ├── PeopleSense occupancy
    └── Evacuation model → dashboard + reports
```

Legacy multi-agent broadcast pipeline (Researcher, Writer, Producer, etc.) still runs for full emergency events.

## PeopleSense API key

`PEOPLESENSE_API_KEY` defaults to `placeholder`. While placeholder mode is active, occupancy readings are deterministic simulated values so development can continue without credentials.

Replace the placeholder in `.env` when the real key is available.
