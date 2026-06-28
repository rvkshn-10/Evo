"""Background agent pipeline orchestration."""

from __future__ import annotations

import asyncio
import logging

from agents.registry import (
    get_bob,
    get_coordinator,
    get_evacuation_intelligence,
    get_fire_chief,
    get_producer,
    get_researcher,
    get_script_writer,
    get_writer,
)
from services.pipeline_status import get_pipeline_status

logger = logging.getLogger(__name__)


async def run_pipeline(event_dict: dict):
    """
    Full agent pipeline run in a background thread pool.
    Order: coordinator → evacuation intelligence → researcher → panel → writer → producer → script_writer
    """
    loop = asyncio.get_event_loop()
    tracker = get_pipeline_status()

    try:
        tracker.set_step("coordinator")
        logger.info("[pipeline] Step 1: Emergency Coordinator")
        coordinator_result = await loop.run_in_executor(
            None, get_coordinator().process_event, event_dict
        )
        event_folder = coordinator_result["event_folder"]
        tracker.set_step("evacuation_intelligence", event_folder=event_folder)
        logger.info("[pipeline] Event folder: %s", event_folder)

        logger.info("[pipeline] Step 2: Evacuation Intelligence Agent")
        intelligence_result = await loop.run_in_executor(
            None, get_evacuation_intelligence().analyze_event, coordinator_result
        )
        tracker.set_step("researcher", event_folder=event_folder)
        logger.info(
            "[pipeline] Evacuation report: %s",
            intelligence_result.get("report_path"),
        )

        logger.info("[pipeline] Step 3: Researcher")
        researcher_result = await loop.run_in_executor(
            None, get_researcher().research_event, coordinator_result
        )

        event_type = event_dict.get("event_type", "other")
        tracker.set_step("panel", event_folder=event_folder)
        logger.info("[pipeline] Step 4: Panel experts for event_type=%s", event_type)

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

        tracker.set_step("writer", event_folder=event_folder)
        logger.info("[pipeline] Step 5: Writer")
        writer_result = await loop.run_in_executor(
            None, get_writer().write_content, researcher_result
        )

        tracker.set_step("producer", event_folder=event_folder)
        logger.info("[pipeline] Step 6: Producer")
        producer_result = await loop.run_in_executor(
            None, get_producer().produce, writer_result
        )

        tracker.set_step("script_writer", event_folder=event_folder)
        logger.info("[pipeline] Step 7: Script Writer")
        await loop.run_in_executor(
            None, get_script_writer().write_script, producer_result
        )

        tracker.complete(event_folder=event_folder)
        logger.info("[pipeline] Complete for %s", event_folder)

    except Exception as exc:
        tracker.fail(str(exc))
        logger.error("[pipeline] Error: %s", exc, exc_info=True)
        raise
