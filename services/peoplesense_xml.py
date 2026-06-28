"""Build and parse PeopleSense OccupancyXML payloads."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Optional
from xml.dom import minidom


def build_occupancy_xml(
    zones: list[dict[str, Any]],
    *,
    generated_at: Optional[str] = None,
) -> str:
    """Build OccupancyXML for one or more FCUSD monitoring zones."""
    root = ET.Element(
        "Occupancy",
        {
            "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
            "source": "fcusd_emergency_management_ai",
        },
    )
    for zone in zones:
        attrs = {
            "id": str(zone.get("id", "")),
            "name": str(zone.get("name", "")),
            "lat": str(zone.get("lat", "")),
            "lon": str(zone.get("lon", "")),
        }
        if zone.get("radius_m") is not None:
            attrs["radius_m"] = str(zone.get("radius_m"))
        if zone.get("category"):
            attrs["category"] = str(zone.get("category"))
        element = ET.SubElement(root, "Zone", attrs)
        if zone.get("occupancy_count") is not None:
            ET.SubElement(element, "Count").text = str(int(zone["occupancy_count"]))
        if zone.get("occupancy_density") is not None:
            ET.SubElement(element, "Density").text = str(float(zone["occupancy_density"]))
        if zone.get("occupancy_volatility") is not None:
            ET.SubElement(element, "Volatility").text = str(float(zone["occupancy_volatility"]))
    rough = ET.tostring(root, encoding="unicode")
    return minidom.parseString(rough).toprettyxml(indent="  ")


def parse_occupancy_xml(xml_text: str) -> list[dict[str, Any]]:
    """Parse OccupancyXML into normalized zone dicts."""
    if not xml_text or not str(xml_text).strip():
        return []

    try:
        root = ET.fromstring(str(xml_text))
    except ET.ParseError:
        return []

    tag = root.tag.split("}")[-1]
    if tag != "Occupancy":
        return []

    zones: list[dict[str, Any]] = []
    for node in root:
        if node.tag.split("}")[-1] != "Zone":
            continue
        zones.append(_parse_zone_node(node))
    return zones


def _parse_zone_node(node: ET.Element) -> dict[str, Any]:
    attrs = node.attrib
    count = _child_text(node, "Count", "Occupancy", "occupancy")
    density = _child_text(node, "Density", "density")
    volatility = _child_text(node, "Volatility", "volatility")

    count = count or attrs.get("count") or attrs.get("occupancy")
    density = density or attrs.get("density")
    volatility = volatility or attrs.get("volatility")

    return {
        "zone_id": attrs.get("id"),
        "zone_name": attrs.get("name"),
        "lat": _float_or_none(attrs.get("lat")),
        "lon": _float_or_none(attrs.get("lon")),
        "radius_m": _float_or_none(attrs.get("radius_m")),
        "occupancy_count": _int_or_none(count),
        "occupancy_density": _float_or_none(density),
        "occupancy_volatility": _float_or_none(volatility),
    }


def _child_text(node: ET.Element, *names: str) -> Optional[str]:
    for child in node:
        local = child.tag.split("}")[-1]
        if local in names and child.text:
            return child.text.strip()
    return None


def _int_or_none(value: Any) -> Optional[int]:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
