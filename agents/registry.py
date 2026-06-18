"""
Agent registry — lazy-initializes and caches all agent singletons.
Import get_* functions from here to avoid circular imports.
"""
from agents.emergency_coordinator.agent import get_coordinator
from agents.researcher.agent import get_researcher
from agents.writer.agent import get_writer
from agents.producer.agent import get_producer
from agents.script_writer.agent import get_script_writer
from agents.panel.bob_seismologist.agent import get_bob
from agents.panel.fire_chief.agent import get_fire_chief

__all__ = [
    "get_coordinator",
    "get_researcher",
    "get_writer",
    "get_producer",
    "get_script_writer",
    "get_bob",
    "get_fire_chief",
]
