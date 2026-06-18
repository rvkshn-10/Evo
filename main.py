import logging
import asyncio
from datetime import datetime, timezone
from typing import Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config.settings import settings
from agents.registry import (
    get_coordinator,
    get_researcher,
    get_writer,
    get_producer,
    get_script_writer,
    get_bob,
    get_fire_chief,
)

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


# ── Pipeline orchestration ─────────────────────────────────────────────────────
async def run_pipeline(event_dict: dict):
    """
    Full agent pipeline run in a background thread pool.
    Order: coordinator → researcher → [panel experts] → writer → producer → script_writer
    """
    loop = asyncio.get_event_loop()

    try:
        # 1. Coordinator — log the event, produce briefing
        logger.info("[pipeline] Step 1: Emergency Coordinator")
        coordinator_result = await loop.run_in_executor(
            None, get_coordinator().process_event, event_dict
        )
        event_folder = coordinator_result["event_folder"]
        logger.info(f"[pipeline] Event folder: {event_folder}")

        # 2. Researcher — internet research + SITREP
        logger.info("[pipeline] Step 2: Researcher")
        researcher_result = await loop.run_in_executor(
            None, get_researcher().research_event, coordinator_result
        )

        # 3. Panel experts — run relevant experts based on event type
        event_type = event_dict.get("event_type", "other")
        logger.info(f"[pipeline] Step 3: Panel experts for event_type={event_type}")

        panel_tasks = []
        if event_type in ("earthquake", "tsunami"):
            panel_tasks.append(
                loop.run_in_executor(
                    None,
                    get_bob().provide_commentary,
                    event_folder,
                    event_dict,
                )
            )
        if event_type in ("fire", "tsunami", "tornado", "flood", "other"):
            panel_tasks.append(
                loop.run_in_executor(
                    None,
                    get_fire_chief().provide_commentary,
                    event_folder,
                    event_dict,
                )
            )

        if panel_tasks:
            await asyncio.gather(*panel_tasks)

        # 4. Writer — article + blog post
        logger.info("[pipeline] Step 4: Writer")
        writer_result = await loop.run_in_executor(
            None, get_writer().write_content, researcher_result
        )

        # 5. Producer — determine panel lineup + production brief
        logger.info("[pipeline] Step 5: Producer")
        producer_result = await loop.run_in_executor(
            None, get_producer().produce, writer_result
        )

        # 6. Script Writer — HeyGen broadcast script
        logger.info("[pipeline] Step 6: Script Writer")
        await loop.run_in_executor(
            None, get_script_writer().write_script, producer_result
        )

        logger.info(f"[pipeline] Complete for {event_folder}")

    except Exception as e:
        logger.error(f"[pipeline] Error: {e}", exc_info=True)


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
async def root():
    return {
        "service": settings.AGENCY_NAME,
        "version": "1.0.0",
        "endpoints": {
            "POST /emergency/event": "Submit an emergency event notification",
            "GET /health": "Health check",
            "GET /docs": "Interactive API documentation",
        },
    }


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        settings.validate()
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
