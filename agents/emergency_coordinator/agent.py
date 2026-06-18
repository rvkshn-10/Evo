import logging
import os
import json
from datetime import datetime, timezone
from typing import Optional

from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from config.settings import settings
from tools.filesystem_tool import write_file, read_file, list_files

logger = logging.getLogger(__name__)

COORDINATOR_PROMPT = """You are the Emergency Coordinator for the Emergency Management Office.

Your role is to:
1. Receive incoming emergency event data (start, update, or end notifications)
2. Log all event data to the appropriate output folder: output/{{event_folder}}/
3. Create and maintain an event_data.json file with the raw event payload
4. Create an event_log.txt that records the timeline of all updates
5. Coordinate downstream agents by preparing a structured briefing

When you receive event data:
- For STATUS=start: Create the output folder, write event_data.json, write event_log.txt with initial entry
- For STATUS=update: Read existing event_log.txt, append the update, overwrite event_data.json with latest data
- For STATUS=end: Append the final entry to event_log.txt, update event_data.json, write a closure note

Always log data to: output/{{event_folder}}/
The event_folder is provided in your input.

After logging, output a structured briefing summary for downstream agents in this format:
--- COORDINATOR BRIEFING ---
Event Folder: <path>
Status: <start|update|end>
Event Type: <type>
Title: <title>
Timestamp: <timestamp>
Location: <lat>, <lon>
Diameter: <miles> miles
Affected Population: <count>
Description: <description>
--- END BRIEFING ---
"""


class EmergencyCoordinatorAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        self.tools = [write_file, read_file, list_files]
        prompt = ChatPromptTemplate.from_messages([
            ("system", COORDINATOR_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
        )

    def process_event(self, event: dict) -> dict:
        """
        Process an incoming emergency event, log it, and return a briefing.
        """
        timestamp_str = event.get("timestamp", datetime.now(timezone.utc).isoformat())
        # Build a filesystem-safe folder name from timestamp + event type
        safe_ts = timestamp_str.replace(":", "-").replace(".", "-")[:19]
        event_type = event.get("event_type", "other").lower()
        event_folder = f"output/{safe_ts}_{event_type}"

        # Deterministically track population over time — don't rely on the LLM for this
        os.makedirs(event_folder, exist_ok=True)
        pop_log_path = os.path.join(event_folder, "population_log.json")
        pop_entry = {
            "timestamp": timestamp_str,
            "affected_population": event.get("affected_population", 0),
            "status": event.get("status", "start"),
        }
        pop_data = []
        if os.path.exists(pop_log_path):
            try:
                with open(pop_log_path, "r", encoding="utf-8") as f:
                    pop_data = json.load(f)
            except Exception:
                pop_data = []
        pop_data.append(pop_entry)
        with open(pop_log_path, "w", encoding="utf-8") as f:
            json.dump(pop_data, f, indent=2)

        event_json = json.dumps(event, indent=2)

        input_text = f"""Process this emergency event and log all data.

Event Folder: {event_folder}

Raw Event Data:
{event_json}

Steps:
1. Write the raw event JSON to: {event_folder}/event_data.json
2. If status is "start", write a new event_log.txt with header and first entry.
   If status is "update" or "end", first read {event_folder}/event_log.txt, then append a new timestamped entry.
3. Output the coordinator briefing.
"""
        result = self.executor.invoke({"input": input_text})
        return {
            "event_folder": event_folder,
            "briefing": result.get("output", ""),
            "event": event,
        }


_instance: Optional[EmergencyCoordinatorAgent] = None


def get_coordinator() -> EmergencyCoordinatorAgent:
    global _instance
    if _instance is None:
        _instance = EmergencyCoordinatorAgent()
    return _instance
