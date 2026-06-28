"""Persist dashboard snapshots and predictions to Neon (Postgres)."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

_SCHEMA_READY = False


def is_configured() -> bool:
    return bool(settings.DATABASE_URL)


def _connect():
    import psycopg2

    return psycopg2.connect(settings.DATABASE_URL)


def ensure_schema() -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY or not is_configured():
        return
    schema_path = settings.PROJECT_ROOT / "db" / "schema.sql"
    if not schema_path.exists():
        logger.warning("Schema file missing: %s", schema_path)
        return
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(schema_path.read_text(encoding="utf-8"))
            conn.commit()
        _SCHEMA_READY = True
        logger.info("Neon schema ready")
    except Exception as exc:
        logger.error("Neon schema setup failed: %s", exc)


def save_dashboard_snapshot(
    snapshot: dict[str, Any],
    *,
    run_mode: str = "sync",
) -> Optional[int]:
    """Store a dashboard snapshot and normalized hazard/prediction rows."""
    if not is_configured():
        return None

    ensure_schema()
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                summary = snapshot.get("summary") or {}
                cur.execute(
                    """
                    INSERT INTO disaster_snapshots
                        (area, run_mode, active_alerts, significant_earthquakes, high_risk_spots, snapshot)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    RETURNING id
                    """,
                    (
                        snapshot.get("area"),
                        run_mode,
                        summary.get("active_alerts", 0),
                        summary.get("significant_earthquakes", 0),
                        summary.get("high_risk_spots", 0),
                        json.dumps(snapshot),
                    ),
                )
                snapshot_id = cur.fetchone()[0]

                for quake in snapshot.get("earthquakes") or []:
                    cur.execute(
                        """
                        INSERT INTO hazard_events
                            (snapshot_id, hazard_category, source, title, severity, magnitude,
                             center_lat, center_lon, raw)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                        """,
                        (
                            snapshot_id,
                            "earthquake",
                            quake.get("source", "usgs"),
                            quake.get("headline") or quake.get("title"),
                            quake.get("severity"),
                            quake.get("magnitude"),
                            quake.get("center_lat"),
                            quake.get("center_lon"),
                            json.dumps(quake),
                        ),
                    )

                for alert in snapshot.get("alerts") or []:
                    cur.execute(
                        """
                        INSERT INTO hazard_events
                            (snapshot_id, hazard_category, source, title, severity, magnitude,
                             center_lat, center_lon, raw)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                        """,
                        (
                            snapshot_id,
                            alert.get("hazard_category", "severe_weather"),
                            "noaa",
                            alert.get("headline") or alert.get("event"),
                            alert.get("severity"),
                            None,
                            alert.get("center_lat"),
                            alert.get("center_lon"),
                            json.dumps(alert),
                        ),
                    )

                model_name = settings.EVO_MODEL_VERSION if run_mode == "evo" else "knn_reference_dataset"
                for alert in (snapshot.get("alerts") or []) + (snapshot.get("earthquakes") or []):
                    for pred in alert.get("evacuation_predictions") or []:
                        cur.execute(
                            """
                            INSERT INTO evacuation_predictions
                                (snapshot_id, spot_id, spot_name, model_name, event_type, occupancy, density,
                                 predicted_evacuation_rate, predicted_evacuation_time_min, risk_level, raw)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                            """,
                            (
                                snapshot_id,
                                pred.get("spot_id"),
                                pred.get("name"),
                                pred.get("model", model_name),
                                pred.get("event_type"),
                                (pred.get("inputs") or {}).get("occupancy"),
                                (pred.get("inputs") or {}).get("density"),
                                pred.get("predicted_evacuation_rate"),
                                pred.get("predicted_evacuation_time_min"),
                                pred.get("risk_level"),
                                json.dumps(pred),
                            ),
                        )

            conn.commit()
        return snapshot_id
    except Exception as exc:
        logger.error("Failed to save disaster history: %s", exc)
        return None


def get_recent_history(limit: int = 20) -> list[dict[str, Any]]:
    if not is_configured():
        return []
    ensure_schema()
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, captured_at, area, run_mode, active_alerts,
                       significant_earthquakes, high_risk_spots
                FROM disaster_snapshots
                ORDER BY captured_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
        return [
            {
                "id": row[0],
                "captured_at": row[1].isoformat() if row[1] else None,
                "area": row[2],
                "run_mode": row[3],
                "active_alerts": row[4],
                "significant_earthquakes": row[5],
                "high_risk_spots": row[6],
            }
            for row in rows
        ]
    except Exception as exc:
        logger.error("Failed to read disaster history: %s", exc)
        return []
