# Emergency Management Office AI

An AI-powered emergency event coordination and broadcast system built with **LangChain**, **FastAPI**, and **OpenAI GPT-4o**. When an emergency event is reported via the REST API, a pipeline of specialized agents automatically logs the event, researches it, downloads media, generates charts, writes news content, assembles a broadcast panel, and produces a HeyGen-ready script — all in the background.

---

## System Architecture

```
POST /emergency/event
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│  Emergency Coordinator                                       │
│  • Logs event_data.json + event_log.txt                     │
│  • Deterministically writes population_log.json (always)    │
└────────────────────────┬─────────────────────────────────────┘
                         │ coordinator briefing
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  Researcher                                                  │
│  • Tavily web search + image search                         │
│  • Writes media_manifest.json                               │
│  • Calls download_media_images → images/ folder             │
│  • Calls generate_population_chart → population_chart.png   │
│  • Writes sitrep.md with chart + images embedded            │
└──────────┬──────────────────────────────────────────────────-┘
           │ SITREP ready
    ┌──────┴──────────────────────────────────────┐
    │  (run in parallel based on event_type)      │
    ▼                                             ▼
┌──────────────────────┐             ┌──────────────────────────┐
│  Bob (Seismologist)  │ earthquake/ │  Fire Chief              │ fire/flood/
│                      │ tsunami     │                          │ tornado/
│  Writes:             │             │  Writes:                 │ tsunami/other
│  panel_bob_          │             │  panel_fire_chief_       │
│  commentary.md       │             │  commentary.md           │
└──────────┬───────────┘             └────────────┬─────────────┘
           └──────────────────┬───────────────────┘
                              │ panel commentaries available
                              ▼
                  ┌───────────────────────┐
                  │  Writer               │
                  │  • Writes article.md  │  (overwritten on each update)
                  │  • Appends to         │
                  │    blog_posts.json    │  (one entry per notification)
                  └───────────┬───────────┘
                              │ article
                              ▼
                  ┌───────────────────────┐
                  │  Producer             │
                  │  • Determines panel   │
                  │    lineup by type     │
                  │  • Writes             │
                  │    production_brief.md│
                  └───────────┬───────────┘
                              │ production brief
                              ▼
                  ┌───────────────────────┐
                  │  Script Writer        │
                  │  • Writes             │
                  │    broadcast_script.md│
                  │  (HeyGen-compatible)  │
                  └───────────────────────┘
```

---

## Agents

### Emergency Coordinator (`agents/emergency_coordinator/`)

The pipeline orchestrator. Accepts the raw event payload and:
- Creates the timestamped output folder (`output/{timestamp}_{type}/`)
- Writes `event_data.json` with the full event payload (overwritten on each notification)
- Writes/appends `event_log.txt` — a running timeline log of all start/update/end entries
- **Deterministically** appends the `{timestamp, affected_population, status}` tuple to `population_log.json` in Python before the agent runs — this data always gets written regardless of LLM decisions
- Emits a structured coordinator briefing for downstream agents

### Researcher (`agents/researcher/`)

Takes the coordinator briefing and performs full situational research:
- Reads `event_data.json` for full context
- **Tavily web search** — current news, historical data, infrastructure details, expert resources
- **Tavily image search** — finds images from the affected area
- Writes `media_manifest.json` — JSON array of `{url, title, description}` for all found images
- **Downloads all images** via `download_media_images` tool → saved to `images/` subfolder; manifest updated with `local_path` fields
- **Generates population chart** via `generate_population_chart` tool → reads `population_log.json`, plots a time-series line chart, saves as `population_chart.png`
- Writes `sitrep.md` — comprehensive Situation Report with the population chart and all downloaded images embedded using relative markdown paths

A Python fallback (`_ensure_chart_and_downloads`) runs after the agent to guarantee the chart and images are generated even if the LLM skipped those tool calls.

### Writer (`agents/writer/`)

Reads the SITREP and produces journalistic content:
- `article.md` — full inverted-pyramid news article (overwritten on each update)
- `blog_posts.json` — JSON array; one new entry appended per notification (start/update/end)

Each blog post entry includes:
```json
{
  "post_number": 1,
  "timestamp": "ISO-8601",
  "status": "start | update | end",
  "headline": "...",
  "body": "...",
  "image_url": "...",
  "tags": ["earthquake", "sacramento", "..."]
}
```

### Producer (`agents/producer/`)

Reads the article and SITREP, then:
- Determines which panel experts are needed based on `event_type`
- Writes `production_brief.md` — panel lineup table, segment order, key messages, tone guidance

**Panel selection logic:**

| Event Type | Panel Members Called |
|------------|----------------------|
| earthquake | Bob (Seismologist) |
| tsunami | Bob + Fire Chief |
| fire | Fire Chief |
| tornado | Fire Chief |
| flood | Fire Chief |
| other | Fire Chief |

### Script Writer (`agents/script_writer/`)

Reads the production brief, article, and SITREP, then writes `broadcast_script.md` — a HeyGen-compatible broadcast script. See [HeyGen Script Format](#heygen-script-format) for conventions. Target runtime: under 4 minutes (~500–600 words).

### Panel: Bob the Seismologist (`agents/panel/bob_seismologist/`)

**Dr. Robert "Bob" Hendricks, Ph.D.**
- Senior Research Seismologist, 27 years experience studying California fault systems
- MIT Ph.D. on P-wave propagation in the Sacramento Valley; personally recorded 12,000+ seismic events
- **Personality:** Extremely nerdy, gets visibly excited about fault data, tangents into seismic theory unprompted, zero interest in anything unrelated to earthquakes
- **Coverage:** Magnitude analysis, energy release, fault system ID, aftershock probability, liquefaction zones, seismic history
- **Output:** `panel_bob_commentary.md`

### Panel: Fire Chief (`agents/panel/fire_chief/`)

**Chief Patricia "Pat" Vasquez**
- Fire Chief, Sacramento Regional Emergency Services, 31 years experience
- Led responses to the 2017 Cascade Fire, 2019 Delta Complex, and multiple earthquake response operations
- **Personality:** Calm, serious, matter-of-fact; every sentence serves a purpose; reports only confirmed information
- **Coverage:** Fire status, containment percentage, evacuation orders and zones, shelter locations, road closures, air quality, resource deployment
- **Output:** `panel_fire_chief_commentary.md`

---

## Output Structure

Each event creates a timestamped folder under `output/`. The folder name is derived from the first event's timestamp and type:

```
output/
└── 2026-06-18T14-32-00_earthquake/
    │
    │  ── Coordinator outputs ──────────────────────────────
    ├── event_data.json             ← Full event payload (updated on each notification)
    ├── event_log.txt               ← Timeline of all start/update/end entries
    ├── population_log.json         ← [{timestamp, affected_population, status}, ...]
    │
    │  ── Researcher outputs ───────────────────────────────
    ├── sitrep.md                   ← Situation Report with chart + images embedded
    ├── media_manifest.json         ← [{url, title, description, local_path}, ...]
    ├── population_chart.png        ← Time-series chart of affected population
    ├── images/
    │   ├── image_001_*.jpg         ← Downloaded media images
    │   ├── image_002_*.jpg
    │   └── ...
    │
    │  ── Panel expert outputs ─────────────────────────────
    ├── panel_bob_commentary.md     ← Bob's seismological analysis (earthquake/tsunami)
    ├── panel_fire_chief_commentary.md  ← Fire Chief's operational status
    │
    │  ── Writer outputs ───────────────────────────────────
    ├── article.md                  ← News article (overwritten on each update)
    ├── blog_posts.json             ← Array of blog posts (appended each notification)
    │
    │  ── Broadcast outputs ────────────────────────────────
    ├── production_brief.md         ← Panel lineup + segment plan
    └── broadcast_script.md         ← HeyGen-ready broadcast script
```

### Population Chart

`population_chart.png` is a dark-themed time-series line chart generated with matplotlib showing how the estimated affected population changes from `start` through `update` notifications to `end`. Each data point is annotated with its status label and the exact population figure. The chart is embedded at the top of `sitrep.md`.

### Blog Posts JSON

`blog_posts.json` grows with every notification. Example with two entries:
```json
[
  {
    "post_number": 1,
    "timestamp": "2026-06-18T14:32:00Z",
    "status": "start",
    "headline": "Major Earthquake Strikes Sacramento State University",
    "body": "...",
    "image_url": "https://...",
    "tags": ["earthquake", "sacramento", "emergency"]
  },
  {
    "post_number": 2,
    "timestamp": "2026-06-18T16:15:00Z",
    "status": "update",
    "headline": "M4.2 Aftershock Recorded; Search and Rescue Ongoing",
    "body": "...",
    "image_url": "https://...",
    "tags": ["earthquake", "aftershock", "search-and-rescue"]
  }
]
```

---

## API

**Base URL:** `http://localhost:8092`

### `POST /emergency/event`

Submit an emergency event notification. Returns `202 Accepted` immediately; the full pipeline runs in the background.

**Request body:**
```json
{
  "status": "start",
  "timestamp": "2026-06-18T14:32:00Z",
  "event_type": "earthquake",
  "center_lat": 38.5616,
  "center_lon": -121.4246,
  "diameter_miles": 10.0,
  "affected_population": 85000,
  "title": "M6.5 Earthquake — Sacramento State University",
  "description": "A magnitude 6.5 earthquake struck beneath the Sacramento State University campus..."
}
```

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| `status` | string | `start`, `update`, `end` | Event lifecycle stage |
| `timestamp` | string | ISO-8601 | Time of this notification |
| `event_type` | string | `earthquake`, `fire`, `tsunami`, `tornado`, `flood`, `other` | Hazard category |
| `center_lat` | float | decimal degrees | Latitude of event epicenter/center |
| `center_lon` | float | decimal degrees | Longitude of event epicenter/center |
| `diameter_miles` | float | miles | Affected area diameter from center |
| `affected_population` | integer | count | Estimated people in the affected area |
| `title` | string | — | Short descriptive title |
| `description` | string | — | Detailed narrative of the current situation |

**Response `202 Accepted`:**
```json
{
  "status": "accepted",
  "event_folder": "output/2026-06-18T14-32-00_earthquake",
  "message": "Event 'M6.5 Earthquake — Sacramento State University' accepted. Pipeline running in background.",
  "pipeline_triggered": true
}
```

### `GET /health`
```json
{ "status": "ok", "agency": "Emergency Management Office AI" }
```

### `GET /docs`
FastAPI interactive Swagger UI — try any endpoint directly in your browser at `http://localhost:8092/docs`.

---

## Postman Collection

Import `postman_collection.json` into Postman for four ready-to-run requests simulating the full lifecycle of a **M6.5 earthquake at Sacramento State University** (38.5616°N, 121.4246°W):

| # | Request | Timestamp | Key Details |
|---|---------|-----------|-------------|
| 1 | **START** | 2026-06-18 14:32 UTC | Initial M6.5 strike, campus evacuating, 85,000 affected |
| 2 | **UPDATE** | 2026-06-18 16:15 UTC | M4.2 aftershock, 14 fatalities confirmed, ARCO Arena shelter open |
| 3 | **END** | 2026-06-19 08:00 UTC | 19 fatalities final, recovery phase, FEMA activated |
| 4 | Health Check | — | `GET /health` |

---

## Setup & Running

### 1. Clone the repository
```bash
git clone https://github.com/lenger06/emergency_management_office_ai.git
cd emergency_management_office_ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

**Dependencies include:**
- `langchain`, `langchain-openai`, `langchain-core`, `langchain-community` — agent framework
- `fastapi`, `uvicorn` — REST API server
- `openai` — GPT-4o LLM
- `requests` — HTTP for Tavily search and image downloads
- `matplotlib` — population chart generation
- `python-dotenv`, `pydantic` — config and validation

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env and add your API keys
```

Required keys:
```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

### 4. Start the server
```bash
python main.py
```

The API is available at `http://localhost:8092`.  
Swagger UI is available at `http://localhost:8092/docs`.

### 5. Send your first event
```bash
curl -X POST http://localhost:8092/emergency/event \
  -H "Content-Type: application/json" \
  -d '{
    "status": "start",
    "timestamp": "2026-06-18T14:32:00Z",
    "event_type": "earthquake",
    "center_lat": 38.5616,
    "center_lon": -121.4246,
    "diameter_miles": 10.0,
    "affected_population": 85000,
    "title": "M6.5 Earthquake — Sacramento State University",
    "description": "A magnitude 6.5 earthquake struck beneath the Sacramento State campus..."
  }'
```

Watch the console for pipeline progress. Output files appear in `output/2026-06-18T14-32-00_earthquake/` as each agent completes.

---

## Event Lifecycle

```
start  →  New output folder created
          population_log.json initialized
          Full pipeline runs: research → panel → write → produce → script

update →  event_log.txt appended
          population_log.json appended (new data point on chart)
          Full pipeline re-runs:
            • sitrep.md overwritten (chart regenerated with new data point)
            • images re-downloaded if new ones found
            • article.md overwritten
            • blog_posts.json gets a NEW entry appended
            • production_brief.md overwritten
            • broadcast_script.md overwritten

end    →  Final append to event_log.txt with closure note
          population_log.json gets final entry
          Full pipeline runs one last time producing final artifacts
```

Multiple `update` events can be sent between `start` and `end`. Each one triggers a full pipeline re-run, adds a new data point to the population chart, and appends a new blog post.

---

## HeyGen Script Format

`broadcast_script.md` follows these conventions for HeyGen avatar video generation:

| Convention | Example |
|-----------|---------|
| Speaker change | `** SPEAKER: Bob (Seismologist) **` |
| Pause marker | `[PAUSE]` |
| Numbers spoken out | `"six point five"` not `"6.5"` |
| Units spoken out | `"ten miles"` not `"10 miles"` |
| No symbols | `"degrees"` not `"°"`, `"percent"` not `"%"` |
| No markdown in spoken text | plain prose only |
| Target length | under 4 minutes (~500–600 words) |

**Segment structure:**
1. Anchor intro (30 sec) — sets the scene
2. Panel expert segment(s) (60–90 sec each) — expert speaks in character
3. Anchor outro (30 sec) — summarizes action items

---

## Configuration Reference

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `OPENAI_API_KEY` | — | Yes | OpenAI API key (GPT-4o) |
| `TAVILY_API_KEY` | — | Yes | Tavily search API key |
| `AGENCY_NAME` | `Emergency Management Office AI` | No | Display name |
| `HOST` | `0.0.0.0` | No | Server bind address |
| `PORT` | `8092` | No | Server port |
| `DEBUG` | `True` | No | Enable hot-reload |
| `LOG_LEVEL` | `INFO` | No | Logging verbosity |
| `OUTPUT_DIR` | `./output` | No | Root output directory |

---

## Project Structure

```
emergency_management_office_ai/
├── main.py                          ← FastAPI app + pipeline orchestration
├── requirements.txt
├── .env.example
├── postman_collection.json          ← Ready-to-import Postman requests
│
├── config/
│   └── settings.py                  ← Centralized config from .env
│
├── agents/
│   ├── registry.py                  ← Lazy-init singleton getters
│   ├── emergency_coordinator/
│   │   └── agent.py                 ← Coordinator + population_log writer
│   ├── researcher/
│   │   └── agent.py                 ← Research + chart + image download
│   ├── writer/
│   │   └── agent.py                 ← Article + blog posts
│   ├── producer/
│   │   └── agent.py                 ← Panel lineup + production brief
│   ├── script_writer/
│   │   └── agent.py                 ← HeyGen broadcast script
│   └── panel/
│       ├── bob_seismologist/
│       │   └── agent.py             ← Dr. Bob Hendricks (earthquake expert)
│       └── fire_chief/
│           └── agent.py             ← Chief Patricia Vasquez
│
├── tools/
│   ├── filesystem_tool.py           ← write_file, read_file, append_json_array, list_files
│   ├── web_research_tool.py         ← Tavily web search (@tool)
│   ├── image_search_tool.py         ← Tavily image search (@tool)
│   ├── chart_tool.py                ← matplotlib population chart (@tool)
│   └── image_downloader_tool.py     ← HTTP image downloader (@tool)
│
└── output/                          ← Generated per-event folders (git-ignored)
```

---

## Extending the System

### Adding a new panel expert

1. Create `agents/panel/your_expert/agent.py` following the Bob or Fire Chief pattern — define a profile string, a system prompt (with character voice), and a `get_your_expert()` singleton
2. Import `get_your_expert` in `agents/registry.py`
3. Add the parallel invocation in `main.py`'s `run_pipeline()` under the appropriate `event_type` condition
4. Update the Producer's `PRODUCER_PROMPT` to include the new expert's name, role, and domain topics

### Adding new event types

1. Add the type to the `Literal` in `EmergencyEvent.event_type` in `main.py`
2. Add the panel routing in `run_pipeline()` 
3. Create any domain-specific panel agents as needed
4. Update the Producer prompt's panel selection table

### Customizing the chart

Edit `tools/chart_tool.py`. The chart uses matplotlib with a dark theme (`#1a1a2e` / `#16213e`). Adjust colors, figure size, annotation style, or add secondary axes for magnitude or wind speed alongside population.

### Swapping the LLM

All agents initialize `ChatOpenAI(model="gpt-4o", ...)`. To use a different model, update the `model=` parameter in each agent's `__init__`. Claude models can be used via the `langchain-anthropic` package as a drop-in replacement.

---

## Evacuation Intelligence Agent

The centerpiece of this project is an **autonomous AI agent** that monitors government warnings, queries PeopleSense, predicts evacuation rates, and publishes to the website.

**Agent:** `agents/evacuation_intelligence/agent.py`  
**Trigger:** `POST /api/agent/run` or `POST /api/alerts/sync`

```
Evacuation Intelligence Agent (GPT-4o)
    │
    ├── NOAA/NWS government alerts
    ├── PeopleSense crowd occupancy (placeholder until API key issued)
    ├── Evacuation rate model (k-NN over FCUSD reference data)
    │
    └── Publishes: dashboard JSON + evacuation_intelligence_report.md
```

The agent also runs automatically in the main pipeline (Step 2) after the Emergency Coordinator logs an event.

### Run the agent

```bash
# Autonomous cycle — fetches NOAA, predicts, publishes dashboard
curl -X POST "http://localhost:8092/api/agent/run?sync=true"

# Or fire-and-forget (runs in background)
curl -X POST http://localhost:8092/api/alerts/sync
```

Requires `OPENAI_API_KEY` in `.env`. PeopleSense uses placeholder mode until a real key is configured.

### Dashboard

After starting the server, open:

- **Dashboard UI:** `http://localhost:8092/`
- **Snapshot API:** `GET /api/dashboard`
- **Sync NOAA data:** `POST /api/alerts/sync`

### Key endpoints

| Endpoint | Description |
| -------- | ----------- |
| `GET /api/dashboard` | Combined NOAA alerts, PeopleSense zones, evacuation predictions |
| `GET /api/alerts` | Raw NOAA active alerts |
| `GET /api/alerts/{id}/analysis` | Single alert with occupancy + evacuation model output |
| `GET /api/peoplesense/zones` | Occupancy for configured monitoring spots |
| `POST /api/evacuation/predict` | Run evacuation model for a custom spot |

### Configuration

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `PEOPLESENSE_API_KEY` | `placeholder` | PeopleSense API key (placeholder returns simulated occupancy) |
| `DEFAULT_ALERT_AREA` | `CA` | Default NOAA state filter |
| `DEFAULT_MAP_LAT` / `DEFAULT_MAP_LON` | Sacramento region | Dashboard map center |

Monitoring locations are configured in `config/monitoring_locations.json`.

See `docs/PROJECT.md` for architecture notes and data references.

