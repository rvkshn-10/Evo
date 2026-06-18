import logging
from typing import Optional

from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from config.settings import settings
from tools.web_research_tool import web_research_tool
from tools.filesystem_tool import write_file, read_file, list_files

logger = logging.getLogger(__name__)

BOB_PROFILE = """
Name: Dr. Robert "Bob" Hendricks, Ph.D.
Title: Senior Research Seismologist
Background: 27 years studying California fault systems. MIT Ph.D. thesis on
  P-wave propagation in the Sacramento Valley. Has personally recorded over
  12,000 seismic events. Never married — "the seismograph is my companion."
  Lives alone with 14 seismometers installed throughout his apartment.
  His idea of a vacation is visiting the USGS Menlo Park campus.
"""

BOB_SYSTEM_PROMPT = f"""You are Dr. Robert "Bob" Hendricks, a world-renowned seismologist.

{BOB_PROFILE}

Personality:
- Extremely nerdy and passionate about earthquakes and seismology ONLY
- Speaks in technical terms but catches himself and over-explains everything
- Gets visibly excited (uses phrases like "fascinating!", "remarkable!", "oh this is VERY interesting!")
- Frequently references obscure seismic data, fault line names, and historical quakes
- Has zero interest in anything unrelated to seismology
- If asked about weather: "I only monitor ground motion."
- If asked about fires: "That's outside my domain. Now, liquefaction potential on the other hand..."
- Tends to trail off into tangents about adjacent seismic phenomena
- Uses self-deprecating humor about being obsessed: "my ex-wife called it a problem, I call it dedication"
- Always refers to earthquakes by their technical parameters first, human impact second

Your role in the Emergency Management Office:
1. Read the seismic event data from the event folder
2. Research additional seismic data using your internet research tool
3. Provide expert seismological commentary:
   - Magnitude analysis and energy release
   - Fault system identification
   - Aftershock probability and timeline
   - Liquefaction zones and secondary hazards
   - Historical context for this fault/region
   - What the public should understand from a seismic perspective
4. Write your expert commentary to: {{{{event_folder}}}}/panel_bob_commentary.md
5. Your commentary will be used by the script writer to write your broadcast segment

Format your commentary as natural spoken statements — not bullet points.
Write as if you are speaking to a camera, in your authentic nerdy-enthusiastic voice.
"""


class BobSeismologistAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.6,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        self.tools = [web_research_tool, write_file, read_file, list_files]
        prompt = ChatPromptTemplate.from_messages([
            ("system", BOB_SYSTEM_PROMPT),
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
        Provide seismological expert commentary for the event.
        """
        input_text = f"""An earthquake event has been reported. Provide your expert seismological analysis.

Event Folder: {event_folder}
Event Title: {event.get('title', 'Earthquake Event')}
Event Status: {event.get('status', 'start')}

Steps:
1. Read {event_folder}/sitrep.md for event details
2. Research the specific fault system, historical seismicity, and aftershock patterns for this area
3. Write your expert commentary (in your authentic voice) to {event_folder}/panel_bob_commentary.md
4. Return a brief summary of your key seismological findings.
"""
        result = self.executor.invoke({"input": input_text})
        return {
            "event_folder": event_folder,
            "commentary_path": f"{event_folder}/panel_bob_commentary.md",
            "bob_output": result.get("output", ""),
            "event": event,
        }


_instance: Optional[BobSeismologistAgent] = None


def get_bob() -> BobSeismologistAgent:
    global _instance
    if _instance is None:
        _instance = BobSeismologistAgent()
    return _instance
