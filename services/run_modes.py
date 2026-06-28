"""Agent run modes: sync-only, Evo, external AI, full broadcast."""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional

from config.settings import settings
from services.agent_runner import run_full_agent_cycle
from services.alert_processor import AlertProcessor
from services.disaster_history import save_dashboard_snapshot
from services.government_feed_sync import GovernmentFeedSync
from services.pipeline_status import get_pipeline_status

logger = logging.getLogger(__name__)

RunMode = Literal["sync", "external_ai", "evo", "evo13", "broadcast"]


async def execute_run_mode(
    mode: RunMode,
    *,
    area: Optional[str] = None,
) -> dict[str, Any]:
    area = area or settings.DEFAULT_ALERT_AREA
    tracker = get_pipeline_status()

    feed_sync = GovernmentFeedSync()
    processor = AlertProcessor(use_evo=(mode == "evo"), use_evo13=(mode == "evo13"))

    if mode == "broadcast":
        return await run_full_agent_cycle(area)

    try:
        tracker.set_step("sync_feeds")
        feed_snapshot = feed_sync.sync_all(area=area, auto_deploy=True)

        tracker.set_step("dashboard")
        dashboard_snapshot = processor.get_dashboard_snapshot(area=area)
        dashboard_path = processor.save_snapshot(dashboard_snapshot)
        neon_id = save_dashboard_snapshot(dashboard_snapshot, run_mode=mode)

        result: dict[str, Any] = {
            "status": "completed",
            "run_mode": mode,
            "area": area,
            "dashboard_path": str(dashboard_path),
            "neon_snapshot_id": neon_id,
            "active_alerts": dashboard_snapshot["summary"]["active_alerts"],
            "significant_earthquakes": dashboard_snapshot["summary"]["significant_earthquakes"],
            "pipeline_events_triggered": 0,
        }

        if mode == "external_ai":
            tracker.set_step("external_ai")
            from services.llm_router import generate_intelligence_summary

            summary = await generate_intelligence_summary(dashboard_snapshot)
            result["ai_summary"] = summary

        tracker.complete()
        return result
    except Exception as exc:
        tracker.fail(str(exc))
        raise


def execute_run_mode_sync(mode: RunMode, area: Optional[str] = None) -> dict[str, Any]:
    import asyncio

    return asyncio.run(execute_run_mode(mode, area=area))
