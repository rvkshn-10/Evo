"""External AI with automatic provider failover (Gemini → OpenAI)."""

from __future__ import annotations

import logging
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)


async def generate_intelligence_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    """One short situation summary — tries Gemini first, then OpenAI."""
    prompt = _build_prompt(snapshot)
    errors: list[str] = []

    if settings.GOOGLE_API_KEY:
        try:
            text = _call_gemini(prompt)
            return {"provider": "gemini", "text": text}
        except Exception as exc:
            errors.append(f"gemini: {exc}")
            logger.warning("Gemini failed, failing over: %s", exc)

    if settings.OPENAI_API_KEY:
        try:
            text = _call_openai(prompt, model="gpt-4o-mini")
            return {"provider": "openai", "model": "gpt-4o-mini", "text": text}
        except Exception as exc:
            errors.append(f"openai-mini: {exc}")
            logger.warning("OpenAI mini failed, failing over: %s", exc)
        try:
            text = _call_openai(prompt, model="gpt-4o")
            return {"provider": "openai", "model": "gpt-4o", "text": text}
        except Exception as exc:
            errors.append(f"openai: {exc}")

    return {
        "provider": None,
        "text": "AI summary unavailable. Dashboard data synced successfully.",
        "errors": errors,
    }


def _build_prompt(snapshot: dict[str, Any]) -> str:
    summary = snapshot.get("summary") or {}
    quakes = snapshot.get("earthquakes") or []
    alerts = snapshot.get("alerts") or []
    lines = [
        "Write a 4-6 sentence emergency manager briefing.",
        f"Active alerts: {summary.get('active_alerts', 0)}",
        f"Significant earthquakes: {summary.get('significant_earthquakes', 0)}",
        f"High-risk spots: {summary.get('high_risk_spots', 0)}",
    ]
    for q in quakes[:3]:
        lines.append(f"Earthquake: {q.get('headline') or q.get('title')}")
    for a in alerts[:3]:
        lines.append(f"Alert: {a.get('headline') or a.get('event')}")
    return "\n".join(lines)


def _call_gemini(prompt: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=settings.GOOGLE_API_KEY)
    model = genai.GenerativeModel(settings.GEMINI_MODEL)
    response = model.generate_content(prompt)
    return (response.text or "").strip()


def _call_openai(prompt: str, *, model: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an emergency management analyst."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return (response.choices[0].message.content or "").strip()
