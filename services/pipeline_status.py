"""Track pipeline progress for the dashboard loading bar."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config.settings import settings

STATUS_PATH = Path(settings.OUTPUT_DIR) / "pipeline_status.json"

STEP_WEIGHTS: dict[str, int] = {
    "sync_feeds": 8,
    "dashboard": 5,
    "external_ai": 10,
    "coordinator": 7,
    "evacuation_intelligence": 12,
    "researcher": 25,
    "panel": 10,
    "writer": 10,
    "producer": 8,
    "script_writer": 15,
}

STEP_LABELS: dict[str, str] = {
    "sync_feeds": "Syncing government feeds (NOAA, USGS, FEMA)",
    "dashboard": "Refreshing dashboard",
    "external_ai": "External AI — Gemini/OpenAI summary",
    "coordinator": "Emergency Coordinator — logging event",
    "evacuation_intelligence": "Evacuation Intelligence Agent",
    "researcher": "Researcher — building SITREP",
    "panel": "Panel experts — expert commentary",
    "writer": "Writer — news article & blog post",
    "producer": "Producer — production brief",
    "script_writer": "Script Writer — broadcast script",
}

_TOTAL_WEIGHT = sum(STEP_WEIGHTS.values())


class PipelineStatusTracker:
    """Thread-safe pipeline progress persisted for frontend polling."""

    def __init__(self, path: Path = STATUS_PATH):
        self.path = path
        self._lock = threading.Lock()
        self._state: dict[str, Any] = self._idle_state()

    def start(self, event_title: str = "Emergency pipeline") -> str:
        with self._lock:
            job_id = uuid.uuid4().hex[:12]
            now = _now_iso()
            self._state = {
                "job_id": job_id,
                "status": "running",
                "event_title": event_title,
                "event_folder": None,
                "current_step": "sync_feeds",
                "current_step_label": STEP_LABELS["sync_feeds"],
                "completed_steps": [],
                "steps_total": len(STEP_WEIGHTS),
                "progress_percent": 0,
                "estimated_seconds_remaining": None,
                "started_at": now,
                "updated_at": now,
                "completed_at": None,
                "error": None,
            }
            self._write_unlocked()
            return job_id

    def set_step(self, step_id: str, *, event_folder: Optional[str] = None) -> None:
        with self._lock:
            if self._state.get("status") != "running":
                return

            current = self._state.get("current_step")
            completed = self._state.get("completed_steps", [])
            if current and current not in completed:
                completed.append(current)

            self._state["completed_steps"] = completed
            self._state["current_step"] = step_id
            self._state["current_step_label"] = STEP_LABELS.get(step_id, step_id)
            if event_folder:
                self._state["event_folder"] = event_folder
            self._state["progress_percent"] = _calc_progress(completed, step_id)
            self._state["estimated_seconds_remaining"] = _estimate_remaining(
                self._state.get("started_at"),
                self._state["progress_percent"],
            )
            self._state["updated_at"] = _now_iso()
            self._write_unlocked()

    def complete(self, *, event_folder: Optional[str] = None) -> None:
        with self._lock:
            if self._state.get("status") != "running":
                return

            current = self._state.get("current_step")
            completed = self._state.get("completed_steps", [])
            if current and current not in completed:
                completed.append(current)

            self._state["status"] = "completed"
            self._state["completed_steps"] = completed
            self._state["current_step"] = None
            self._state["current_step_label"] = "Complete"
            self._state["progress_percent"] = 100
            self._state["estimated_seconds_remaining"] = 0
            if event_folder:
                self._state["event_folder"] = event_folder
            self._state["completed_at"] = _now_iso()
            self._state["updated_at"] = self._state["completed_at"]
            self._write_unlocked()

    def fail(self, error: str) -> None:
        with self._lock:
            self._state["status"] = "failed"
            self._state["error"] = error
            self._state["updated_at"] = _now_iso()
            self._write_unlocked()

    def get(self) -> dict[str, Any]:
        with self._lock:
            if self.path.exists():
                try:
                    data = json.loads(self.path.read_text(encoding="utf-8"))
                    self._state = data
                    return dict(data)
                except (json.JSONDecodeError, OSError):
                    pass
            return dict(self._state)

    def is_running(self) -> bool:
        return self.get().get("status") == "running"

    def _idle_state(self) -> dict[str, Any]:
        return {
            "job_id": None,
            "status": "idle",
            "event_title": None,
            "event_folder": None,
            "current_step": None,
            "current_step_label": None,
            "completed_steps": [],
            "steps_total": len(STEP_WEIGHTS),
            "progress_percent": 0,
            "estimated_seconds_remaining": None,
            "started_at": None,
            "updated_at": None,
            "completed_at": None,
            "error": None,
        }

    def _write_unlocked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")


_tracker: Optional[PipelineStatusTracker] = None


def get_pipeline_status() -> PipelineStatusTracker:
    global _tracker
    if _tracker is None:
        _tracker = PipelineStatusTracker()
    return _tracker


def _calc_progress(completed_steps: list[str], current_step: Optional[str]) -> int:
    done_weight = sum(STEP_WEIGHTS.get(step, 0) for step in completed_steps)
    if current_step:
        done_weight += STEP_WEIGHTS.get(current_step, 0) * 0.35
    return min(99, int(done_weight / _TOTAL_WEIGHT * 100))


def _estimate_remaining(started_at: Optional[str], progress_percent: int) -> Optional[int]:
    if not started_at or progress_percent <= 2:
        return None
    try:
        started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    if elapsed <= 0:
        return None
    total_estimated = elapsed / (progress_percent / 100)
    return max(0, int(total_estimated - elapsed))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
