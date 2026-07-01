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


async def generate_location_evac_briefing(analysis: dict[str, Any]) -> dict[str, Any]:
    """LLM narrative for a user-picked map location analysis."""
    prompt = _build_location_prompt(analysis)
    errors: list[str] = []

    if settings.GOOGLE_API_KEY:
        try:
            text = _call_gemini(prompt)
            return {"provider": "gemini", "text": text}
        except Exception as exc:
            errors.append(f"gemini: {exc}")

    if settings.OPENAI_API_KEY:
        try:
            text = _call_openai(prompt, model="gpt-4o-mini")
            return {"provider": "openai", "model": "gpt-4o-mini", "text": text}
        except Exception as exc:
            errors.append(f"openai-mini: {exc}")

    return {
        "provider": None,
        "text": _fallback_location_briefing(analysis),
        "errors": errors,
    }


def _build_location_prompt(analysis: dict[str, Any]) -> str:
    loc = analysis.get("location") or {}
    ps = analysis.get("peoplesense") or {}
    pred = analysis.get("prediction") or {}
    best = analysis.get("recommended_route") or {}
    routes = analysis.get("evacuation_routes") or []
    lines = [
        "You are an emergency evacuation planner. Write a concise 5-7 sentence briefing for a site commander.",
        "Distinguish model estimates from facts. Do not claim to predict earthquakes.",
        f"Location: {loc.get('name')} ({loc.get('lat')}, {loc.get('lon')})",
        f"Category: {loc.get('category')}",
        f"PeopleSense occupancy: {ps.get('occupancy')} people, density {ps.get('density')}",
        f"Predicted evac success: {pred.get('predicted_evacuation_success_pct')}%",
        f"Predicted evac time (model): {pred.get('predicted_evacuation_time_min')} min",
        f"Recommended route: {best.get('compass')} — est. clear time {best.get('estimated_clear_time_min')} min",
        "Alternate routes:",
    ]
    for route in routes[1:4]:
        lines.append(
            f"- {route.get('compass')}: walk {route.get('walk_distance_m')} m, "
            f"est. clear {route.get('estimated_clear_time_min')} min"
        )
    nearest = analysis.get("nearest_active_hazard")
    if nearest:
        lines.append(f"Nearest hazard: {nearest.get('headline') or nearest.get('title')} ({nearest.get('distance_km')} km)")
    lines.append("Recommend primary egress direction and crowd management steps.")
    return "\n".join(lines)


def _fallback_location_briefing(analysis: dict[str, Any]) -> str:
    loc = analysis.get("location") or {}
    ps = analysis.get("peoplesense") or {}
    pred = analysis.get("prediction") or {}
    best = analysis.get("recommended_route") or {}
    if not best:
        return (
            f"Analysis for {loc.get('name')}: ~{ps.get('occupancy')} people on site. "
            f"Model estimates {pred.get('predicted_evacuation_success_pct')}% evac success. "
            "Walking routes could not be computed — check network or try again."
        )
    return (
        f"For {loc.get('name')} with ~{ps.get('occupancy')} people (PeopleSense), "
        f"prefer the {best.get('compass')} assembly route "
        f"({best.get('walk_distance_m')} m walk, ~{best.get('estimated_clear_time_min')} min estimated clear time "
        f"with current crowd density). "
        f"Evo model estimates {pred.get('predicted_evacuation_success_pct')}% evacuation success "
        f"and {pred.get('predicted_evacuation_time_min')} min overall evac time. "
        "Use official site plans and live incident commands — this is a research assistant only."
    )
