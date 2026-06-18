import logging
import json
from datetime import datetime, timezone
from typing import Optional

from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from config.settings import settings
from tools.filesystem_tool import write_file, read_file, append_json_array, list_files

logger = logging.getLogger(__name__)

WRITER_PROMPT = """You are the Emergency Journalist for the Emergency Management Office.

Your role is to:
1. Read the SITREP and media manifest from the event folder
2. Write a clear, factual, professional news article about the emergency event
3. Write a concise blog post suitable for social media / web publishing
4. On updates, revise the article with new information and append a new blog post

Article guidelines:
- Journalistic inverted-pyramid style (most important facts first)
- Include who, what, when, where, why, how
- Reference images from the media manifest where relevant
- Keep tone neutral, professional, and informative
- Clearly note if this is an initial report, update, or final report

Blog post guidelines:
- 150-300 words
- Engaging opener
- Key facts in bullet points
- Call to action (e.g., "Follow updates at...", "Residents should...")
- Include a timestamp and update number

Output files:
- Article: {{event_folder}}/article.md  (overwrite on updates)
- Blog posts: {{event_folder}}/blog_posts.json  (append new entry on each call, never overwrite)

Blog post JSON entry format:
{{
  "post_number": 1,
  "timestamp": "ISO-8601",
  "status": "start|update|end",
  "headline": "...",
  "body": "...",
  "image_url": "... (first image from media manifest if available)",
  "tags": ["earthquake", "sacramento", ...]
}}
"""


class WriterAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.4,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        self.tools = [write_file, read_file, append_json_array, list_files]
        prompt = ChatPromptTemplate.from_messages([
            ("system", WRITER_PROMPT),
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

    def write_content(self, researcher_result: dict) -> dict:
        """
        Produce or update the article and blog post for the event.
        """
        event_folder = researcher_result["event_folder"]
        research_summary = researcher_result["research_summary"]
        event = researcher_result["event"]
        status = event.get("status", "start")

        # Determine blog post number by checking existing posts
        blog_path = f"{event_folder}/blog_posts.json"

        input_text = f"""You are writing {'an initial article' if status == 'start' else 'an updated article'} about this emergency event.

Event Folder: {event_folder}
Event Status: {status}

Researcher Summary:
{research_summary}

Steps:
1. Read {event_folder}/sitrep.md for full situational detail
2. Read {event_folder}/media_manifest.json for available images (use try-read, it may not exist yet)
3. Write/overwrite the news article to {event_folder}/article.md
4. Build a blog post JSON entry (use post_number=1 for start; for update/end, first read {blog_path} to find the current count then increment)
5. Append the blog post JSON to {blog_path} using the append_json_array tool
6. Return a brief summary of the article headline and key points.
"""
        result = self.executor.invoke({"input": input_text})
        return {
            "event_folder": event_folder,
            "article_path": f"{event_folder}/article.md",
            "blog_posts_path": blog_path,
            "writer_summary": result.get("output", ""),
            "event": event,
        }


_instance: Optional[WriterAgent] = None


def get_writer() -> WriterAgent:
    global _instance
    if _instance is None:
        _instance = WriterAgent()
    return _instance
