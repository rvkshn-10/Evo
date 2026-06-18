import logging
import json
import os
from typing import Optional

from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from config.settings import settings
from tools.web_research_tool import web_research_tool
from tools.image_search_tool import image_search_tool
from tools.filesystem_tool import write_file, read_file, list_files
from tools.chart_tool import generate_population_chart
from tools.image_downloader_tool import download_media_images

logger = logging.getLogger(__name__)

RESEARCHER_PROMPT = """You are the Emergency Researcher for the Emergency Management Office.

Your role is to:
1. Receive a coordinator briefing about an active emergency event
2. Read the event_data.json from the event folder for full context
3. Search the internet for supplemental information relevant to the event:
   - Current news coverage of the event
   - Historical data about similar events in the area
   - Infrastructure details (hospitals, shelters, roads, utilities)
   - Weather conditions and forecasts if relevant
   - Images from the affected area
   - Expert resources and guidance documents
4. Write a media manifest JSON array to {{event_folder}}/media_manifest.json
   Each entry: {{"url": "...", "title": "...", "description": "..."}}
5. Call download_media_images with the event folder to download all images locally
6. Call generate_population_chart with the event folder to create the population chart
7. Write a comprehensive SITREP to {{event_folder}}/sitrep.md — embed the chart and
   downloaded images using relative markdown image syntax

SITREP format:
# SITUATION REPORT — {{event title}}
**Generated:** {{timestamp}}
**Status:** {{status}}

## Affected Population Over Time
![Population Over Time](population_chart.png)

## Event Summary
[1-2 paragraph summary]

## Location & Impact
- Coordinates: lat, lon
- Affected Radius: X miles
- Estimated Population Affected: N

## Current Situation
[Detailed narrative from event data + research]

## Background & Context
[Historical context, area details, infrastructure]

## Key Resources & Response Assets
[Hospitals, shelters, emergency services identified through research]

## Media & Imagery
For each downloaded image, embed it:
![Image Title](images/filename.jpg)
*Caption: description*

## Sources
[All URLs referenced]

Always be factual. Clearly label any information as "Reported" (from event data) vs "Researched" (from internet).
After download_media_images runs, use the returned local_path values to embed images with relative paths.
"""


class ResearcherAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        self.tools = [
            web_research_tool,
            image_search_tool,
            write_file,
            read_file,
            list_files,
            generate_population_chart,
            download_media_images,
        ]
        prompt = ChatPromptTemplate.from_messages([
            ("system", RESEARCHER_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=20,
        )

    def research_event(self, coordinator_result: dict) -> dict:
        """
        Take a coordinator result and produce a SITREP with chart + downloaded images.
        """
        event_folder = coordinator_result["event_folder"]
        briefing = coordinator_result["briefing"]
        event = coordinator_result["event"]

        input_text = f"""You have received a coordinator briefing for an emergency event.

{briefing}

Event Folder: {event_folder}

Steps:
1. Read {event_folder}/event_data.json for full event details
2. Search the web for current news, background, and resources about this event
3. Search for relevant images of the affected area and event type
4. Write the media manifest JSON array to {event_folder}/media_manifest.json
   Each entry: {{"url": "...", "title": "...", "description": "..."}}
5. Call download_media_images("{event_folder}") to download all images locally
6. Call generate_population_chart("{event_folder}") to generate the population chart
7. Write the comprehensive SITREP to {event_folder}/sitrep.md
   - Embed the population chart as: ![Population Over Time](population_chart.png)
   - For each downloaded image, use its local_path (relative) returned by download_media_images
     to embed it: ![title](images/filename.jpg)
8. Return a summary of your key findings.
"""
        result = self.executor.invoke({"input": input_text})

        # Also attempt chart + download directly in Python as fallback if agent skipped them
        _ensure_chart_and_downloads(event_folder)

        return {
            "event_folder": event_folder,
            "research_summary": result.get("output", ""),
            "sitrep_path": f"{event_folder}/sitrep.md",
            "media_manifest_path": f"{event_folder}/media_manifest.json",
            "event": event,
        }


def _ensure_chart_and_downloads(event_folder: str) -> None:
    """
    Fallback: if the agent didn't call the tools, call them directly in Python.
    Safe to call multiple times — generate_population_chart just overwrites the PNG.
    """
    try:
        chart_path = os.path.join(event_folder, "population_chart.png")
        if not os.path.exists(chart_path):
            generate_population_chart.func(event_folder)
    except Exception as e:
        logger.warning(f"[_ensure_chart_and_downloads] chart fallback failed: {e}")

    try:
        manifest_path = os.path.join(event_folder, "media_manifest.json")
        images_dir = os.path.join(event_folder, "images")
        if os.path.exists(manifest_path) and not os.path.isdir(images_dir):
            download_media_images.func(event_folder)
    except Exception as e:
        logger.warning(f"[_ensure_chart_and_downloads] download fallback failed: {e}")


_instance: Optional[ResearcherAgent] = None


def get_researcher() -> ResearcherAgent:
    global _instance
    if _instance is None:
        _instance = ResearcherAgent()
    return _instance
