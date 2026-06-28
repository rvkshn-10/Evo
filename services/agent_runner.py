"""Orchestrate the full agent cycle: feed sync, dashboard, and emergency pipeline."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from config.settings import settings
from services.alert_processor import AlertProcessor
from services.government_feed_sync import GovernmentFeedSync
from services.pipeline import run_pipeline
from services.pipeline_status import get_pipeline_status

logger = logging.getLogger(__name__)


async def run_full_agent_cycle(
    area: Optional[str] = None,
    *,
    max_pipeline_events: int = 1,
) -> dict[str, Any]:
    area = area or settings.DEFAULT_ALERT_AREA
    tracker = get_pipeline_status()

    if tracker.get()["status"] == "idle":
        tracker.start("Starting emergency pipeline…")

    feed_sync = GovernmentFeedSync()
    processor = AlertProcessor()

    try:
        tracker.set_step("sync_feeds")
        feed_snapshot = feed_sync.sync_all(area=area, auto_deploy=True)

        events = _collect_pipeline_events(feed_sync, processor, feed_snapshot, area)
        events = events[:max_pipeline_events]

        tracker.set_step("dashboard")
        dashboard_snapshot = processor.get_dashboard_snapshot(area=area)
        dashboard_path = processor.save_snapshot(dashboard_snapshot)

        pipeline_results = []
        for event in events:
            logger.info("[agent_runner] Starting full pipeline for: %s", event.get("title"))
            await run_pipeline(event)
            safe_ts = event["timestamp"].replace(":", "-").replace(".", "-")[:19]
            event_folder = f"output/{safe_ts}_{event['event_type']}"
            pipeline_results.append({
                "title": event.get("title"),
                "event_folder": event_folder,
                "event_type": event.get("event_type"),
            })

        return {
            "status": "completed",
            "job_id": tracker.get().get("job_id"),
            "area": area,
            "dashboard_path": str(dashboard_path),
            "active_alerts": dashboard_snapshot["summary"]["active_alerts"],
            "pipeline_events_triggered": len(pipeline_results),
            "pipelines": pipeline_results,
        }
    except Exception as exc:
        tracker.fail(str(exc))
        raise


def _collect_pipeline_events(
    feed_sync: GovernmentFeedSync,
    processor: AlertProcessor,
    feed_snapshot: dict[str, Any],
    area: str,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    seen_titles: set[str] = set()

    def add_event(event: Optional[dict[str, Any]]) -> None:
        if not event:
            return
        title = event.get("title", "")
        if title in seen_titles:
            return
        seen_titles.add(title)
        events.append(_sanitize_event(event))

    for raw in feed_sync.eew_candidates_to_events():
        add_event(raw)

    for alert in feed_snapshot.get("noaa_alerts", []):
        if alert.get("severity") in ("Extreme", "Severe"):
            add_event(processor.alert_to_emergency_event(alert))

    for alert in feed_snapshot.get("noaa_alerts", []):
        add_event(processor.alert_to_emergency_event(alert))

    if not events:
        add_event(_fallback_regional_event(area, feed_snapshot))

    return events


def _fallback_regional_event(area: str, feed_snapshot: dict[str, Any]) -> dict[str, Any]:
    alerts = feed_snapshot.get("noaa_alerts") or []
    top = alerts[0] if alerts else {}

    return {
        "status": "start",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": top.get("event_type", "other"),
        "center_lat": settings.DEFAULT_MAP_LAT,
        "center_lon": settings.DEFAULT_MAP_LON,
        "diameter_miles": 15.0,
        "affected_population": 5000,
        "title": top.get("headline") or top.get("event") or f"Regional hazard monitoring — {area}",
        "description": (
            top.get("description")
            or top.get("instruction")
            or f"Autonomous agent cycle for active hazards in {area}."
        ),
    }


def _sanitize_event(event: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "status", "timestamp", "event_type", "center_lat", "center_lon",
        "diameter_miles", "affected_population", "title", "description",
    }
    return {key: value for key, value in event.items() if key in allowed}


def run_full_agent_cycle_sync(area: Optional[str] = None, **kwargs: Any) -> dict[str, Any]:
    return asyncio.run(run_full_agent_cycle(area, **kwargs))
