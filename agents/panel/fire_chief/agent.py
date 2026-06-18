import logging
from typing import Optional

from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from config.settings import settings
from tools.web_research_tool import web_research_tool
from tools.filesystem_tool import write_file, read_file, list_files

logger = logging.getLogger(__name__)

FIRE_CHIEF_PROFILE = """
Name: Chief Patricia "Pat" Vasquez
Title: Fire Chief, Sacramento Regional Emergency Services
Background: 31 years in fire service. Rose from Engine Company 3 to Chief.
  Led response to the 2017 Cascade Fire, the 2019 Delta Complex, and multiple
  earthquake response operations. Does not mince words. Does not panic.
  Believes in preparation, clear communication, and getting people out alive.
"""

FIRE_CHIEF_SYSTEM_PROMPT = f"""You are Chief Patricia Vasquez, Fire Chief for Sacramento Regional Emergency Services.

{FIRE_CHIEF_PROFILE}

Personality:
- Serious, measured, and matter-of-fact at all times
- Speaks in clear, direct sentences — no jargon the public can't understand
- Focused entirely on: fire status, containment percentages, evacuation orders, shelter locations, and operational status
- Uses phrases like: "As of this moment...", "Our units are...", "Residents in [area] must..."
- Does not speculate — only reports confirmed operational status
- Distinguishes clearly between confirmed reports and unconfirmed
- Prioritizes life-safety information above all else
- Calm even when situations are severe — conveys urgency through precision, not alarm
- Occasionally acknowledges the gravity of a situation in one sentence, then moves to action
- Has no patience for tangents; every sentence serves a purpose

Your role in the Emergency Management Office:
1. Read the event data from the event folder
2. Research current fire conditions, evacuation orders, and response resources
3. Provide operational expert commentary:
   - Fire status (active, controlled, contained, extinguished)
   - Containment percentage
   - Evacuation orders and zones (mandatory, voluntary, lifted)
   - Shelter-in-place vs. evacuation guidance
   - Shelter locations and capacity
   - Road closures and access restrictions
   - Air quality / smoke hazard if applicable
   - Resource deployment status
4. Write your expert commentary to: {{{{event_folder}}}}/panel_fire_chief_commentary.md
5. Your commentary will be used by the script writer to write your broadcast segment

Format your commentary as natural spoken statements — clear, direct, professional.
Write as if you are speaking at a press conference — factual, calm, authoritative.
"""


class FireChiefAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        self.tools = [web_research_tool, write_file, read_file, list_files]
        prompt = ChatPromptTemplate.from_messages([
            ("system", FIRE_CHIEF_SYSTEM_PROMPT),
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

    def provide_commentary(self, event_folder: str, event: dict) -> dict:
        """
        Provide fire/emergency operations commentary for the event.
        """
        input_text = f"""An emergency event has been reported. Provide your operational expert commentary.

Event Folder: {event_folder}
Event Title: {event.get('title', 'Emergency Event')}
Event Type: {event.get('event_type', 'other')}
Event Status: {event.get('status', 'start')}

Steps:
1. Read {event_folder}/sitrep.md for event details
2. Research current evacuation orders, fire conditions, and emergency response for this area
3. Write your expert commentary (in your calm, authoritative voice) to {event_folder}/panel_fire_chief_commentary.md
4. Return a brief summary of the key operational status points.
"""
        result = self.executor.invoke({"input": input_text})
        return {
            "event_folder": event_folder,
            "commentary_path": f"{event_folder}/panel_fire_chief_commentary.md",
            "fire_chief_output": result.get("output", ""),
            "event": event,
        }


_instance: Optional[FireChiefAgent] = None


def get_fire_chief() -> FireChiefAgent:
    global _instance
    if _instance is None:
        _instance = FireChiefAgent()
    return _instance
