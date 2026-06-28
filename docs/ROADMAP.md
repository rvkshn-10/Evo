# Development Roadmap

## Meeting summary (team sync)

The team is building an **AI agent-based emergency management system** for natural disaster response. Key decisions:

- **Framework agents:** Natural Disaster Agent + Occupancy Agent (PeopleSense already ingests USGS + FEMA feeds natively)
- **Phase 1 (Larry):** Simplified POST intake with GPS coordinates, diameter, and occupancy — defer complex processing to later phases
- **Tomorrow's focus:** PeopleSense auto-deployment + earthquake early warning APIs

---

## Completed (ready for tomorrow morning)

### Earthquake early warning APIs
- `GET /api/earthquakes` — USGS GeoJSON feed (hour/day/week/month)
- `GET /api/earthquakes/eew` — EEW-style candidates with recommended actions
- `services/usgs_client.py` — public USGS feeds + magnitude/MMI normalization

> **Note:** Production ShakeAlert® XML requires a [USGS Technical Partnership](https://www.shakealert.org/). Phase 1 uses public USGS feeds and flags significant events for automated action.

### PeopleSense auto-deployment
- `POST /api/peoplesense/deploy` — deploy a single zone
- `POST /api/peoplesense/auto-deploy` — deploy all monitoring spots
- `GET /api/peoplesense/deployments` — deployment manifest status
- `services/peoplesense_deployment.py` — records deployments to `output/peoplesense/deployments.json`

### Government feed sync
- `POST /api/feeds/sync` — ingest USGS + FEMA IPAWS + NOAA, auto-deploy zones
- `services/government_feed_sync.py` — orchestrates all three feeds
- `services/fema_ipaws_client.py` — FEMA OpenFEMA IPAWS archived alerts

### Simplified Phase 1 event POST
- `POST /api/event` — body: `{ center_lat, center_lon, diameter_miles, occupancy, event_type }`
- Auto-deploys PeopleSense zone + triggers agent pipeline

### Agent tools updated
The Evacuation Intelligence Agent now has tools for:
- `sync_government_feeds`
- `fetch_earthquake_early_warnings`
- `deploy_peoplesense_zone`

---

## Tomorrow morning checklist

```bash
# 1. Sync all government feeds + auto-deploy PeopleSense zones
curl -X POST "http://localhost:8092/api/feeds/sync?sync=true"

# 2. Check earthquake early-warning candidates
curl http://localhost:8092/api/earthquakes/eew

# 3. Submit a simplified Phase 1 event
curl -X POST http://localhost:8092/api/event \
  -H "Content-Type: application/json" \
  -d '{
    "center_lat": 38.5616,
    "center_lon": -121.4246,
    "diameter_miles": 10,
    "occupancy": 85000,
    "event_type": "earthquake"
  }'

# 4. Run the full autonomous agent
curl -X POST "http://localhost:8092/api/agent/run?sync=true"
```

---

## Phase 2 (deferred)

- [ ] USGS ShakeAlert Technical Partnership for live XML EEW messages
- [ ] Replace PeopleSense placeholder with production API key
- [ ] Kaggle dataset integration for evacuation model retraining
- [ ] Scheduled background feed polling (cron / APScheduler)
- [ ] CAP/OIP extension for occupancy parameters in alert messages
- [ ] Physics-based reward system (Larry's lunar lander approach)
