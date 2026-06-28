import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config.settings import settings
from routes.intelligence import router as intelligence_router
from services.pipeline import run_pipeline
from services.pipeline_status import get_pipeline_status

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Emergency Management Office AI",
    description="AI-powered emergency event coordination, research, and broadcast system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins(),
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIST = Path(__file__).resolve().parent / "web" / "dist"
WEB_STATIC = Path(__file__).resolve().parent / "web" / "static"
WEB_DIR = WEB_DIST if WEB_DIST.exists() else WEB_STATIC

if WEB_DIST.exists() and (WEB_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")
else:
    app.mount("/static", StaticFiles(directory=WEB_STATIC), name="static")
app.include_router(intelligence_router)


# ── Request schema ─────────────────────────────────────────────────────────────
class EmergencyEvent(BaseModel):
    status: Literal["start", "update", "end"] = Field(
        ..., description="Event lifecycle status"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 timestamp of this notification",
    )
    event_type: Literal["earthquake", "fire", "tsunami", "tornado", "flood", "other"] = Field(
        ..., description="Category of emergency event"
    )
    center_lat: float = Field(..., description="Latitude of event center (decimal degrees)")
    center_lon: float = Field(..., description="Longitude of event center (decimal degrees)")
    diameter_miles: float = Field(..., description="Affected area diameter in miles from center")
    affected_population: int = Field(..., description="Estimated number of people in affected area")
    title: str = Field(..., description="Short descriptive title of the event")
    description: str = Field(..., description="Detailed description of the event")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "start",
                "timestamp": "2026-06-18T14:32:00Z",
                "event_type": "earthquake",
                "center_lat": 38.5616,
                "center_lon": -121.4246,
                "diameter_miles": 10.0,
                "affected_population": 85000,
                "title": "Earthquake at Sacramento State University",
                "description": (
                    "A magnitude 6.5 earthquake struck beneath the Sacramento State University campus "
                    "at a depth of 8 km. Significant ground shaking reported across the greater "
                    "Sacramento area. Multiple building collapses reported on campus. Power outages "
                    "affecting approximately 40,000 customers in Sacramento County."
                ),
            }
        }


# ── Response schema ────────────────────────────────────────────────────────────
class EventResponse(BaseModel):
    status: str
    event_folder: str
    message: str
    pipeline_triggered: bool


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.post("/emergency/event", response_model=EventResponse, status_code=202)
async def receive_event(event: EmergencyEvent, background_tasks: BackgroundTasks):
    """
    Receive an emergency event notification and trigger the full agent pipeline.
    Returns immediately with 202 Accepted; processing runs in the background.
    """
    event_dict = event.model_dump()
    logger.info(f"[API] Received {event.status} event: {event.title}")

    # Derive the event folder name so we can return it to the caller immediately
    safe_ts = event.timestamp.replace(":", "-").replace(".", "-")[:19]
    event_folder = f"output/{safe_ts}_{event.event_type}"

    tracker = get_pipeline_status()
    if not tracker.is_running():
        tracker.start(event.title)
        tracker.set_step("coordinator", event_folder=event_folder)

    background_tasks.add_task(run_pipeline, event_dict)

    return EventResponse(
        status="accepted",
        event_folder=event_folder,
        message=f"Event '{event.title}' accepted. Pipeline running in background.",
        pipeline_triggered=True,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "agency": settings.AGENCY_NAME}


@app.get("/")
async def dashboard():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api")
async def api_index():
    return {
        "service": settings.AGENCY_NAME,
        "version": "1.0.0",
        "endpoints": {
            "GET /": "Evacuation intelligence dashboard",
            "POST /api/agent/run": "Full emergency pipeline (all agents + broadcast files)",
            "GET /api/dashboard": "NOAA + PeopleSense + evacuation snapshot",
            "GET /api/pipeline/status": "Pipeline progress for loading bar",
            "POST /api/alerts/sync": "Same as Run Agent — full emergency pipeline",
            "POST /api/event": "Phase 1 simplified event POST (GPS, diameter, occupancy)",
            "POST /api/feeds/sync": "Sync USGS + FEMA + NOAA and auto-deploy PeopleSense",
            "GET /api/earthquakes/eew": "Earthquake early-warning candidates",
            "POST /api/peoplesense/auto-deploy": "Deploy all monitoring spots to PeopleSense",
            "POST /emergency/event": "Submit an emergency event notification",
            "GET /health": "Health check",
            "GET /docs": "Interactive API documentation",
        },
    }


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        settings.validate(require_agent_keys=False)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise SystemExit(1)

    logger.info(f"Starting {settings.AGENCY_NAME} on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
