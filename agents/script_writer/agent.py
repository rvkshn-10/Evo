import logging
from typing import Optional

from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from config.settings import settings
from tools.filesystem_tool import write_file, read_file, list_files

logger = logging.getLogger(__name__)

SCRIPT_WRITER_PROMPT = """You are the Emergency Broadcast Script Writer for the Emergency Management Office.

Your role is to write broadcast scripts suitable for HeyGen avatar video generation.

HeyGen script requirements:
- Plain prose, no markdown formatting in the spoken portions
- Natural spoken language — short sentences, clear phrasing
- Pauses indicated with [PAUSE] markers
- Speaker changes indicated with: ** SPEAKER: [Name/Role] **
- Emphasis indicated with CAPS (sparingly)
- Numbers spelled out: "six point five" not "6.5", "ten miles" not "10 miles"
- Avoid symbols: use "degrees" not "°", "percent" not "%"
- Each segment clearly delimited

Script structure (follow the production brief):
1. ANCHOR INTRO — sets the scene, introduces the situation
2. PANEL SEGMENT(S) — each expert speaks in their authentic voice
3. ANCHOR OUTRO — summarizes action items, closes the broadcast

Script file: {{event_folder}}/broadcast_script.md

On updates, overwrite the script with the revised version and note at the top:
"UPDATE #{{n}} — {{timestamp}}"

Always write as if being read aloud by a professional broadcaster avatar.
Do not include stage directions about camera angles or visuals — HeyGen handles that.
Keep total broadcast runtime under 4 minutes (approximately 500-600 words of spoken text).
"""


class ScriptWriterAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        self.tools = [write_file, read_file, list_files]
        prompt = ChatPromptTemplate.from_messages([
            ("system", SCRIPT_WRITER_PROMPT),
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

    def write_script(self, producer_result: dict) -> dict:
        """
        Write the HeyGen broadcast script based on the article and production brief.
        """
        event_folder = producer_result["event_folder"]
        event = producer_result["event"]

        input_text = f"""Write the broadcast script for this emergency event.

Event Folder: {event_folder}
Event Status: {event.get('status', 'start')}
Event Title: {event.get('title', 'Emergency Event')}
Event Type: {event.get('event_type', 'other')}

Steps:
1. Read {event_folder}/production_brief.md for the panel lineup and segment plan
2. Read {event_folder}/article.md for the full article content
3. Read {event_folder}/sitrep.md for additional technical details
4. Write the HeyGen-ready broadcast script to {event_folder}/broadcast_script.md
5. Return the script title and approximate word count.
"""
        result = self.executor.invoke({"input": input_text})
        return {
            "event_folder": event_folder,
            "script_path": f"{event_folder}/broadcast_script.md",
            "script_writer_output": result.get("output", ""),
            "event": event,
        }


_instance: Optional[ScriptWriterAgent] = None


def get_script_writer() -> ScriptWriterAgent:
    global _instance
    if _instance is None:
        _instance = ScriptWriterAgent()
    return _instance
