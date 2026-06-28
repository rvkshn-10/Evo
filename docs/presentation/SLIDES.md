# Evo — slide outline + speaker notes

Each slide below has:
- **Title** — put on the slide
- **On slide** — short bullets (keep slides readable)
- **Image** — file in `images/` (optional)
- **Speaker notes** — what to say out loud, plain English

---

## Slide 1 — Title

**Title:** Evo  
**Subtitle:** Evacuation intelligence for schools and campuses

**On slide:**
- Live dashboard: evac-evo.vercel.app
- GitHub: github.com/rvkshn-10/Evo

**Image:** none (or `04-dashboard-desktop.png` as background, dimmed)

**Speaker notes:**  
"I'm presenting Evo — a system that pulls live weather, earthquake, and crowd data together and estimates how well an evacuation might go at each school site. The demo is live online, and the code is open on GitHub."

---

## Slide 2 — Why this matters

**Title:** The problem

**On slide:**
- Emergencies use many separate data sources
- Occupancy at schools changes constantly
- Leaders need one view: hazards + people + estimated evac time

**Image:** none

**Speaker notes:**  
"When something happens, safety staff jump between weather apps, earthquake feeds, and attendance systems. Evo puts hazards, how many people are on site, and a predicted evacuation success rate on one map so decisions are faster."

---

## Slide 3 — What Evo does (one sentence)

**Title:** What Evo does

**On slide:**
- Ingests live government hazard feeds
- Reads PeopleSense occupancy per campus
- Predicts evacuation success % and time per site
- Shows everything on an interactive map dashboard

**Image:** `04-dashboard-desktop.png`

**Speaker notes:**  
"Evo is not just a chatbot. It's a data pipeline: fetch alerts, match them to schools, run a machine learning model, and publish results to a dashboard you can refresh with one button."

---

## Slide 4 — System architecture

**Title:** How it's hosted

**On slide:**
- Browser → Vercel (website)
- Vercel → Oracle cloud (Python API)
- API talks to NOAA, USGS, PeopleSense, etc.

**Image:** `diagram-architecture.png`

**Speaker notes:**  
"The website is static files on Vercel — fast and free to deploy. All the heavy work — fetching feeds and running the model — happens on an Oracle virtual machine. Your browser never talks to fifteen APIs directly; the backend does."

---

## Slide 5 — Data pipeline (backend)

**Title:** How data moves through the backend

**On slide:**
1. Sync hazard feeds  
2. Enrich each alert with occupancy  
3. Build feature vector per site  
4. Run prediction model  
5. Save dashboard JSON + history  

**Image:** `diagram-data-pipeline.png`

**Speaker notes:**  
"Every time you click Run Agent, the server pulls fresh hazard data, overlays PeopleSense counts for each monitoring spot, turns that into numbers the model understands, runs inference, and writes a snapshot. That snapshot powers the map and the predictions table."

---

## Slide 6 — Live dashboard tour

**Title:** The dashboard

**On slide:**
- Map with hazard layers
- Stats: alerts, high-risk spots, earthquakes
- Predictions table per school site
- Run mode picker + Run Agent button

**Image:** `04-dashboard-desktop.png`

**Speaker notes:**  
"This is the production UI. Left side you toggle earthquakes, fires, floods, and monitoring spots. The table shows each site — how full it is, predicted evac rate, minutes to clear, and risk level. Run Agent refreshes everything."

---

## Slide 7 — Data sources (inputs)

**Title:** What data we take in

**On slide:**

| Source | What it gives us |
|--------|------------------|
| NOAA / NWS | Weather and hazard alerts |
| USGS | Earthquakes + early warning |
| GDACS | Global disaster context |
| NASA FIRMS | Wildfire hotspots |
| FEMA IPAWS | Public alert feed |
| PeopleSense | Live occupancy, density, volatility |
| FCUSD reference | 4,082 historical evacuation records |

**Image:** none or small icons

**Speaker notes:**  
"We combine public government feeds with PeopleSense, which tracks how many people are at each campus. The historical spreadsheet gives the model examples of past evacuation drills and scenarios. None of these alone tells you evac success — we merge them."

---

## Slide 8 — PeopleSense & monitoring spots

**Title:** Campus monitoring

**On slide:**
- Raspberry Pis → PeopleSense database → our API (GET)
- Six FCUSD spots: Vista del Lago, Folsom High, Cordova Park, Sac State, ARCO, Folsom Dam
- Each spot has GPS, category (school, stadium, train station), match rules

**Image:** `01-dashboard-main.png` (shows layer toggles)

**Speaker notes:**  
"We don't read the Pi directly. Devices push into PeopleSense, and we pull occupancy through their API. Each school is configured in a JSON file with coordinates and rules so the right sensor data maps to the right site."

---

## Slide 9 — Alerts & enrichment

**Title:** How alerts become predictions

**On slide:**
- Fetch active NOAA + earthquake + fire alerts
- For each alert near a school: attach occupancy
- Run evacuation model for that spot
- Flag **high risk** if evac rate low, density high, or crowd very large

**Image:** none

**Speaker notes:**  
"An alert by itself doesn't know how many students are in the building. We join alert geometry to the nearest monitoring spot, pull live occupancy, and ask the model: given this hazard and this crowd, what success rate and time do we expect?"

---

## Slide 10 — The map

**Title:** Map layers (Leaflet)

**On slide:**
- Base map: OpenStreetMap
- Heatmap: earthquake intensity (toggle)
- Markers: schools (cyan), quakes (yellow), hazard zones (red/orange)
- Filters: earthquake, wildfire, flood, tornado, tsunami, severe weather, monitoring

**Image:** `04-dashboard-desktop.png` (map portion)

**Speaker notes:**  
"The map is standard Leaflet. You can stack layers — for example earthquakes plus monitoring spots. The heatmap shows where seismic activity clusters. Everything filters client-side from one JSON snapshot the API returns."

---

## Slide 11 — Run modes overview

**Title:** Five run modes

**On slide:**

| Mode | What happens | Cost |
|------|----------------|------|
| **Sync** (default) | Refresh feeds + basic model | Free |
| **Evo 1.2** | Same + production ML hybrid | Free |
| **Evo 1.3** | Research model + enriched data | Free |
| **External AI** | Adds Gemini/OpenAI summary | Paid API |
| **Broadcast** | Full 7-agent content pipeline | Paid API |

**Image:** none

**Speaker notes:**  
"Sync is the everyday mode — no AI API charges. Evo 1.2 turns on our trained model. Evo 1.3 is experimental. External AI writes a short briefing paragraph. Broadcast runs seven GPT agents to produce reports and a video script — that's for demos, not the default."

---

## Slide 12 — Run mode: Sync

**Title:** Sync only (default)

**On slide:**
- Refreshes all hazard feeds
- Predictions via k-nearest neighbors (k=25) on reference data
- Saves run to disaster history database
- No OpenAI / Gemini calls

**Speaker notes:**  
"Sync compares each live situation to the closest historical examples in our dataset — classic k-NN. It's fast, free, and good enough for monitoring. Every run is logged so you can chart trends over time."

---

## Slide 13 — Run mode: Evo 1.2 hybrid (production)

**Title:** Evo 1.2 — production model

**On slide:**
- Neural network + gradient boosting ensemble
- **Hybrid rule:** k-NN for success & risk; neural net for evac time at train stations
- Trained in Google Colab, exported to ONNX
- Runs on Oracle CPU in production

**Image:** `diagram-evo-hybrid.png`

**Speaker notes:**  
"Evo 1.2 is our production policy. We trained two model types and averaged them. Honestly, the data doesn't support trusting the neural net for success rate at every building type yet, so we keep a proven k-NN for that and use the neural net mainly for evacuation time at transit-style sites like Cordova Park."

---

## Slide 14 — Evo 1.2 validation (Phase A)

**Title:** Evo 1.2 — honest metrics

**On slide:**

| Metric | Result | Pass? |
|--------|--------|-------|
| Success error (MAE) | 1.65% | Yes |
| Success R² | 0.18 | No |
| Time error (MAE) | 0.83 min | Yes |
| Time R² | 0.64 | No |

- Label: **DATA_CEILING** — live feeds add features, not outcome labels

**Speaker notes:**  
"We're transparent about metrics. Error on success percentage is small, but R-squared is low — the model can't explain much variance because we don't have enough real post-drill outcomes in the training set. That's why we call it a data ceiling, not a model failure."

---

## Slide 15 — Run mode: Evo 1.3 (research)

**Title:** Evo 1.3 — research path

**On slide:**
- Needs drill timestamps + measured outcomes + PeopleSense snapshots
- Extra features: egress exits, route length, blockage
- **Synthetic demo trained** June 2026 — pipeline proof only
- **Not promoted** to production

**Speaker notes:**  
"Evo 1.3 is the next version when we have real drill data joined to sensor timestamps. We ran a synthetic demo with fake drill rows to prove the trainer works end-to-end. The metrics look similar to 1.2 — still a data ceiling — so production stays on 1.2 hybrid."

---

## Slide 16 — Evo 1.3 training results (synthetic demo)

**Title:** Evo 1.3 synthetic training — completed

**On slide:**
- Preflight: passed (12 sensor rows, 6 outcomes)
- Model: MLP + LightGBM ensemble
- Success MAE 1.66% ✓ · Success R² 0.19 ✗
- Time MAE 0.87 min ✓ · Time R² 0.60 ✗
- Result: **keep Evo 1.2 hybrid**

**Image:** none (optional: training curve from repo `models/evo1.3/learning_curves.json`)

**Speaker notes:**  
"We deliberately used fake drill data labeled as synthetic. Training finished successfully and artifacts are on the server, but promotion gates failed on R-squared, as expected. The point for mentors: the pipeline works; better data is what unlocks better models."

---

## Slide 17 — Run mode: External AI

**Title:** External AI (RAG-style briefing)

**On slide:**
- After dashboard build, send **structured snapshot** to Gemini
- Fallback: OpenAI if Gemini fails
- Returns 4–6 sentence summary (not stored in history)
- Like RAG: retrieve facts first, then generate text

**Speaker notes:**  
"This mode doesn't change predictions. It reads the live dashboard numbers — how many alerts, which sites are high risk — and asks an LLM to write a short situation briefing. That's retrieval-augmented generation in practice: facts from our API, language from the model."

---

## Slide 18 — Run mode: Full broadcast

**Title:** Full broadcast pipeline

**On slide:**
7 GPT-4o agents in order:
1. Coordinator → 2. Evacuation intel → 3. Researcher → 4. Panel experts → 5. Writer → 6. Producer → 7. Script writer

**Output:** sitrep, charts, articles, HeyGen-ready script

**Image:** `diagram-broadcast.png`

**Speaker notes:**  
"Broadcast is the research demo of autonomous agents. One event triggers a chain that logs the emergency, researches it, writes expert commentary, and outputs a broadcast script. It's separate from the daily dashboard and costs API credits."

---

## Slide 19 — Training in Google Colab

**Title:** How models are trained

**On slide:**
- Script: `model_training/evo1_2/train_evo1_2.py`
- 5-fold cross-validation, grouped by scenario
- Export: ONNX + OpenVINO files → `models/evo1.2/`
- Quality gates must pass before promotion
- Evo 1.3 trainer: `train_evo1_3.py` with preflight checks

**Speaker notes:**  
"Training happens in Colab or locally with Python. We use cross-validation so we're not fooling ourselves with one lucky split. The script exports files the API loads at runtime and writes a validation report that says whether promotion is allowed."

---

## Slide 20 — Model panel (UI)

**Title:** Model button — what you see

**On slide:**
- Live inference diagram (animated data flow)
- Train vs validation loss chart
- Metrics: MAE, R², runtime backend
- Badge: Gates passed vs Research preview

**Image:** `02-model-modal.png`

**Speaker notes:**  
"Click the diamond Model button. You get a live diagram showing data moving from feeds through the encoder into the neural net and boosting models. The loss chart is from training — blue is training error, gray is validation. Hover connections to see actual feature values from the current dashboard."

---

## Slide 21 — Understanding the loss chart

**Title:** Train / val loss — plain meaning

**On slide:**
- **Train loss (blue):** model fitting the training data — should go down
- **Val loss (gray):** error on held-out folds — shows generalization
- Flat val loss = we've learned what the data allows (data ceiling)

**Image:** `02-model-modal.png` (crop loss chart area)

**Speaker notes:**  
"If validation loss stops improving while training keeps dropping, that's overfitting. If both flatten high, we often don't have enough labeled evacuation outcomes — that's our data ceiling. We don't hide that; it's on the dashboard."

---

## Slide 22 — Metrics glossary (on the modal)

**Title:** What the numbers mean

**On slide:**
- **MAE:** average prediction error (%, or minutes)
- **R²:** how much variance explained (1.0 = perfect, ~0 = weak)
- **Gates passed:** all promotion checks OK
- **Research preview:** failed a gate — don't use for official ops
- **Hybrid policy:** which part of the model made this prediction

**Speaker notes:**  
"MAE is intuitive — off by about 1.6 percentage points on success rate. R-squared is harder: 0.18 means we're only slightly better than guessing the average. Gates passed is our checklist before we'd call a model production-ready."

---

## Slide 23 — Disaster history

**Title:** History & export

**On slide:**
- Every sync/evo run saved to SQLite or Neon Postgres
- Charts: high-risk spots over time, alert counts
- Export: JSON, CSV zip, SQLite file
- Open History button (📊) on dashboard

**Image:** `05-history-modal.png`

**Speaker notes:**  
"History is for after-action review. Each run stores the full snapshot plus a breakdown of hazards and predictions. You can export to Excel or pipe the SQLite file into your own analytics. Broadcast mode doesn't write history — only the dashboard modes do."

---

## Slide 24 — Neural Compute Stick

**Title:** Intel Neural Compute Stick (optional hardware)

**On slide:**
- USB stick accelerates neural inference on a **local Mac**
- **OpenVINO** = Intel driver software (not a separate product)
- **Vercel / Oracle:** CPU only — no USB in the cloud
- Local setup: `docs/NCS_SETUP.md`

**Image:** `diagram-ncs-stack.png`

**Speaker notes:**  
"The Neural Compute Stick is optional USB hardware for edge demos. OpenVINO is just how Intel talks to the stick or the CPU. The public website runs on a cloud server with no USB port, so production uses CPU. For stick demos, run the API on your laptop and open localhost."

---

## Slide 25 — Vercel + Oracle connection

**Title:** Deployment details

**On slide:**
- `vercel.json` rewrites `/api/*` → Oracle IP port 8092
- Frontend env: `VITE_API_BASE` optional
- Oracle: `systemd` service, `git pull` to deploy API
- CORS allows `*.vercel.app`

**Speaker notes:**  
"Vercel serves the React-style frontend and proxies API calls so the browser stays on HTTPS. Oracle runs Python 24/7 on a free ARM VM. Push to GitHub and Vercel redeploys the UI automatically; we SSH to Oracle to pull API changes."

---

## Slide 26 — How Evo helps people

**Title:** Real-world value

**On slide:**
- **Faster awareness:** one screen instead of many tabs
- **Prioritization:** high-risk sites surface first
- **Planning:** estimated evac time supports staging resources
- **Audit trail:** history for drills and research
- **Extensible:** research model path when better labels arrive

**Image:** `04-dashboard-desktop.png`

**Speaker notes:**  
"The goal isn't to replace human judgment. It's to compress situational awareness — which sites are crowded, which hazards are active, which predictions look worst — so safety staff spend seconds, not minutes, orienting during an incident or drill."

---

## Slide 27 — Current status & next steps

**Title:** Where we are today

**On slide:**
- Live: evac-evo.vercel.app (Evo 1.2 production)
- Evo 1.3: trained on synthetic demo, research mode only
- Presentation pack: `docs/presentation/`
- Colab notebooks: add to `docs/colab/`

**Speaker notes:**  
"We're in research preview — honest about metrics and data limits. The system is deployed and usable for demos and pilot monitoring. Evo 1.3 proves the training pipeline; replacing synthetic data with real drill measurements is the path to stronger models."

---

## Slide 28 — Thank you / Q&A

**Title:** Thank you

**On slide:**
- evac-evo.vercel.app
- github.com/rvkshn-10/Evo
- Questions?

**Image:** none

**Speaker notes:**  
"Happy to walk through the live site or the repo. The full technical reference is in REFERENCE.md in this same folder if you want deeper detail on any pipeline."
