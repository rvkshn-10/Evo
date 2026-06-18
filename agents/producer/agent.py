import logging
from typing import Optional

from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from config.settings import settings
from tools.filesystem_tool import write_file, read_file, list_files

logger = logging.getLogger(__name__)

PRODUCER_PROMPT = """You are the Emergency Broadcast Producer for the Emergency Management Office.

Your role is to:
1. Review the event article and situation report
2. Determine which panel expert(s) are needed based on the event type:
   - Earthquake → Bob (Seismologist)
   - Fire → Fire Chief
   - Tsunami → Bob (Seismologist) + Fire Chief
   - Tornado → Fire Chief
   - Flood → Fire Chief
   - Other → Fire Chief (default)
   - Multi-hazard → all relevant experts
3. Write a production brief to: {{event_folder}}/production_brief.md
   This brief tells the script writer exactly who is speaking, in what order, and what topics each panel member should cover
4. Return the list of panel members needed with their coverage topics

Available panel members:
- Bob (Seismologist): Earthquake magnitude, fault lines, aftershock predictions, seismic history, geological hazards
- Fire Chief: Fire status, containment, evacuation orders, shelter locations, emergency response operations

Production Brief format:
# Production Brief — {{event title}}
**Status:** {{status}}

## Panel Lineup
| Order | Speaker | Role | Topics |
|-------|---------|------|--------|

## Broadcast Segments
1. Anchor intro (30 sec)
2. [Panel member 1 segment] (~60-90 sec)
3. [Panel member 2 segment if applicable] (~60-90 sec)
4. Anchor outro with action items (30 sec)

## Key Messages
[Bullet points of the most critical information to convey]

## Tone Guidance
[Note on urgency level and emotional register appropriate to this event]
"""


class ProducerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        self.tools = [write_file, read_file, list_files]
        prompt = ChatPromptTemplate.from_messages([
            ("system", PRODUCER_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=8,
        )

    def produce(self, writer_result: dict) -> dict:
        """
        Determine panel lineup and write production brief.
        """
        event_folder = writer_result["event_folder"]
        event = writer_result["event"]

        input_text = f"""Produce the broadcast for this emergency event.

Event Folder: {event_folder}
Event Type: {event.get('event_type', 'other')}
Event Status: {event.get('status', 'start')}
Event Title: {event.get('title', 'Emergency Event')}

Steps:
1. Read {event_folder}/article.md for the latest article content
2. Read {event_folder}/sitrep.md for situational context
3. Determine the panel lineup based on event type
4. Write the production brief to {event_folder}/production_brief.md
5. Return the panel lineup as a clear list.
"""
        result = self.executor.invoke({"input": input_text})
        return {
            "event_folder": event_folder,
            "production_brief_path": f"{event_folder}/production_brief.md",
            "producer_output": result.get("output", ""),
            "event": event,
        }


_instance: Optional[ProducerAgent] = None


def get_producer() -> ProducerAgent:
    global _instance
    if _instance is None:
        _instance = ProducerAgent()
    return _instance
