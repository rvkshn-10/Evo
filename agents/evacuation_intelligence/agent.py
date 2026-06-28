"""Evacuation Intelligence Agent — autonomous NOAA + PeopleSense + evacuation modeling."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from config.settings import settings
from tools.filesystem_tool import list_files, read_file, write_file
from tools.intelligence_tools import (
    deploy_peoplesense_zone,
    enrich_alert_with_evacuation_analysis,
    fetch_earthquake_early_warnings,
    fetch_government_alerts,
    get_peoplesense_occupancy,
    list_monitoring_spots,
    predict_evacuation_rate,
    publish_dashboard_snapshot,
    sync_government_feeds,
)

logger = logging.getLogger(__name__)

AGENT_PROMPT = """You are the Evacuation Intelligence Agent for the Emergency Management Office.

Your mission:
1. Monitor government hazard warnings — NOAA/NWS, USGS earthquakes, FEMA IPAWS
2. Auto-deploy PeopleSense monitoring zones when hazards are detected
3. Pull real-time crowd occupancy from deployed zones
4. Predict evacuation rates using historical reference datasets
5. Publish actionable intelligence to the public dashboard and event folders

You are an autonomous decision-maker. Prioritize alerts that threaten life safety
(fire, flood, tornado, tsunami, earthquake, severe weather with evacuation implications).

Workflow for each run:
1. Call sync_government_feeds to ingest USGS + FEMA + NOAA and auto-deploy PeopleSense zones
2. Call fetch_earthquake_early_warnings for earthquake early-warning candidates
3. Call fetch_government_alerts for active NOAA warnings
4. Call list_monitoring_spots to see schools, shelters, and transit hubs to protect
5. For the most urgent alerts (up to 3), call enrich_alert_with_evacuation_analysis
6. Deploy additional zones with deploy_peoplesense_zone if a high-risk area lacks coverage
7. For any spot needing manual review, call get_peoplesense_occupancy then predict_evacuation_rate
8. Write evacuation_analysis.json to the output folder with your structured findings
9. Write evacuation_intelligence_report.md — a clear public-facing summary with:
   - Active government warnings (NOAA, USGS, FEMA)
   - PeopleSense deployment status
   - High-risk locations and predicted evacuation rates
   - Recommended actions for emergency managers
10. Call publish_dashboard_snapshot to update the website

Be factual. Distinguish government-sourced alerts from model predictions.
Flag high-risk spots (evacuation rate below 90% or risk_level=high) prominently.
When PeopleSense is in placeholder mode, note that occupancy is simulated.
"""


class EvacuationIntelligenceAgent:
    """Autonomous agent that ingests gov warnings and publishes evacuation intelligence."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        self.tools = [
            sync_government_feeds,
            fetch_earthquake_early_warnings,
            fetch_government_alerts,
            deploy_peoplesense_zone,
            get_peoplesense_occupancy,
            predict_evacuation_rate,
            list_monitoring_spots,
            enrich_alert_with_evacuation_analysis,
            publish_dashboard_snapshot,
            write_file,
            read_file,
            list_files,
        ]
        prompt = ChatPromptTemplate.from_messages([
            ("system", AGENT_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=25,
        )

    def run_autonomous_cycle(
        self,
        *,
        area: Optional[str] = None,
        event_folder: Optional[str] = None,
    ) -> dict:
        """
        Run a full autonomous intelligence cycle without a manual emergency event.
        """
        area = area or settings.DEFAULT_ALERT_AREA
        output_folder = event_folder or _default_output_folder()

        input_text = f"""Run an autonomous evacuation intelligence cycle.

Region: {area}
Output folder: {output_folder}
PeopleSense mode: {"placeholder (simulated occupancy)" if _is_placeholder() else "live"}

Steps:
1. Fetch government alerts for area={area}
2. List monitoring spots
3. Analyze the top urgent alerts with enrich_alert_with_evacuation_analysis
4. Write structured findings to {output_folder}/evacuation_analysis.json
5. Write a public report to {output_folder}/evacuation_intelligence_report.md
6. Publish the dashboard snapshot

Return a brief executive summary of what you found and which spots are highest risk.
"""
        result = self.executor.invoke({"input": input_text})
        _ensure_outputs(output_folder, area)

        return {
            "mode": "autonomous",
            "output_folder": output_folder,
            "summary": result.get("output", ""),
            "report_path": f"{output_folder}/evacuation_intelligence_report.md",
            "analysis_path": f"{output_folder}/evacuation_analysis.json",
        }

    def analyze_event(self, coordinator_result: dict) -> dict:
        """
        Enrich a specific emergency event folder with evacuation intelligence.
        Called from the main agent pipeline after the coordinator logs the event.
        """
        event_folder = coordinator_result["event_folder"]
        briefing = coordinator_result["briefing"]
        event = coordinator_result["event"]
        event_type = event.get("event_type", "other")

        input_text = f"""An emergency event has been logged. Produce evacuation intelligence for it.

{briefing}

Event folder: {event_folder}
Event type: {event_type}

Steps:
1. Read {event_folder}/event_data.json for coordinates and context
2. List monitoring spots and check PeopleSense occupancy for each
3. Call predict_evacuation_rate for each spot using the event type "{event_type}"
4. Write {event_folder}/evacuation_analysis.json with all predictions and risk levels
5. Write {event_folder}/evacuation_intelligence_report.md summarizing:
   - Event impact on each monitored location
   - Predicted evacuation rates and times
   - Priority evacuation recommendations
6. Call publish_dashboard_snapshot

Return your key findings and highest-risk locations.
"""
        result = self.executor.invoke({"input": input_text})
        _ensure_event_outputs(event_folder, event)

        return {
            "event_folder": event_folder,
            "summary": result.get("output", ""),
            "report_path": f"{event_folder}/evacuation_intelligence_report.md",
            "analysis_path": f"{event_folder}/evacuation_analysis.json",
            "event": event,
        }


def _default_output_folder() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    return f"output/{ts}_evacuation_intelligence"


def _is_placeholder() -> bool:
    from services.peoplesense_client import PeopleSenseClient

    return PeopleSenseClient().is_placeholder


def _ensure_outputs(output_folder: str, area: str) -> None:
    """Python fallback so dashboard data exists even if the LLM skipped tool calls."""
    analysis_path = os.path.join(output_folder, "evacuation_analysis.json")
    report_path = os.path.join(output_folder, "evacuation_intelligence_report.md")

    if os.path.exists(analysis_path):
        publish_dashboard_snapshot.func("output/dashboard/latest_snapshot.json")
        return

    try:
        from services.alert_processor import AlertProcessor

        processor = AlertProcessor()
        snapshot = processor.get_dashboard_snapshot()
        processor.save_snapshot(snapshot)

        os.makedirs(output_folder, exist_ok=True)
        with open(analysis_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "generated_at": snapshot["generated_at"],
                    "area": area,
                    "alerts": snapshot.get("alerts", []),
                    "summary": snapshot.get("summary", {}),
                },
                handle,
                indent=2,
            )

        if not os.path.exists(report_path):
            lines = [
                "# Evacuation Intelligence Report",
                f"**Generated:** {snapshot['generated_at']}",
                f"**Region:** {area}",
                "",
                "## Active Government Warnings",
            ]
            for alert in snapshot.get("alerts", [])[:5]:
                lines.append(f"- **{alert.get('event')}**: {alert.get('headline', '')}")

            lines.extend(["", "## High-Risk Locations", ""])
            for alert in snapshot.get("alerts", [])[:3]:
                for pred in alert.get("evacuation_predictions", []):
                    if pred.get("risk_level") in ("high", "medium"):
                        lines.append(
                            f"- **{pred['name']}**: "
                            f"{pred['predicted_evacuation_rate']*100:.1f}% evac rate, "
                            f"risk={pred['risk_level']}"
                        )

            with open(report_path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
    except Exception as exc:
        logger.warning("Evacuation intelligence fallback failed: %s", exc)


def _ensure_event_outputs(event_folder: str, event: dict) -> None:
    analysis_path = os.path.join(event_folder, "evacuation_analysis.json")
    if os.path.exists(analysis_path):
        publish_dashboard_snapshot.func("output/dashboard/latest_snapshot.json")
        return

    try:
        from services.alert_processor import AlertProcessor

        processor = AlertProcessor()
        snapshot = processor.get_dashboard_snapshot()
        processor.save_snapshot(snapshot)

        synthetic_alert = {
            "id": event.get("title", "event"),
            "event": event.get("title"),
            "event_type": event.get("event_type", "other"),
            "description": event.get("description", ""),
            "center_lat": event.get("center_lat"),
            "center_lon": event.get("center_lon"),
        }
        enriched = processor.enrich_alert(synthetic_alert)

        os.makedirs(event_folder, exist_ok=True)
        with open(analysis_path, "w", encoding="utf-8") as handle:
            json.dump(enriched, handle, indent=2)
    except Exception as exc:
        logger.warning("Event evacuation fallback failed: %s", exc)


_instance: Optional[EvacuationIntelligenceAgent] = None


def get_evacuation_intelligence() -> EvacuationIntelligenceAgent:
    global _instance
    if _instance is None:
        _instance = EvacuationIntelligenceAgent()
    return _instance
