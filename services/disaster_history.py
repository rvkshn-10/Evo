"""Persist dashboard snapshots and predictions (Postgres or local SQLite)."""

from __future__ import annotations

import csv
import io
import json
import logging
import shutil
import sqlite3
import tempfile
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

_SCHEMA_READY = False

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS disaster_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at TEXT NOT NULL,
    area TEXT,
    run_mode TEXT NOT NULL DEFAULT 'sync',
    active_alerts INTEGER DEFAULT 0,
    significant_earthquakes INTEGER DEFAULT 0,
    high_risk_spots INTEGER DEFAULT 0,
    snapshot TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hazard_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL,
    hazard_category TEXT NOT NULL,
    source TEXT,
    title TEXT,
    severity TEXT,
    magnitude REAL,
    center_lat REAL,
    center_lon REAL,
    recorded_at TEXT NOT NULL,
    raw TEXT,
    FOREIGN KEY (snapshot_id) REFERENCES disaster_snapshots(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evacuation_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL,
    spot_id TEXT,
    spot_name TEXT,
    model_name TEXT NOT NULL DEFAULT 'knn_reference_dataset',
    event_type TEXT,
    occupancy INTEGER,
    density REAL,
    predicted_evacuation_rate REAL,
    predicted_evacuation_time_min REAL,
    risk_level TEXT,
    recorded_at TEXT NOT NULL,
    raw TEXT,
    FOREIGN KEY (snapshot_id) REFERENCES disaster_snapshots(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_disaster_snapshots_captured_at
    ON disaster_snapshots (captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_hazard_events_category
    ON hazard_events (hazard_category, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_evac_predictions_spot
    ON evacuation_predictions (spot_id, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_evac_predictions_risk
    ON evacuation_predictions (risk_level, recorded_at DESC);
"""


def sqlite_db_path() -> Path:
    override = (getattr(settings, "DISASTER_HISTORY_DB", None) or "").strip()
    if override:
        return Path(override)
    return settings.PROJECT_ROOT / "data" / "disaster_history.db"


def storage_backend() -> str:
    return "postgres" if settings.DATABASE_URL else "sqlite"


def is_configured() -> bool:
    """History storage is always available (SQLite fallback)."""
    return True


def postgres_configured() -> bool:
    return bool(settings.DATABASE_URL)


def get_storage_info() -> dict[str, Any]:
    backend = storage_backend()
    info: dict[str, Any] = {
        "backend": backend,
        "postgres_configured": postgres_configured(),
        "sqlite_path": str(sqlite_db_path()) if backend == "sqlite" else None,
    }
    if backend == "sqlite" and sqlite_db_path().exists():
        info["sqlite_size_bytes"] = sqlite_db_path().stat().st_size
    return info


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


@contextmanager
def _sqlite_conn() -> Iterator[sqlite3.Connection]:
    path = sqlite_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _postgres_connect():
    import psycopg2

    return psycopg2.connect(settings.DATABASE_URL)


def ensure_schema() -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return

    if postgres_configured():
        schema_path = settings.PROJECT_ROOT / "db" / "schema.sql"
        if not schema_path.exists():
            logger.warning("Schema file missing: %s", schema_path)
            return
        try:
            with _postgres_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(schema_path.read_text(encoding="utf-8"))
                conn.commit()
            _SCHEMA_READY = True
            logger.info("Postgres disaster history schema ready")
        except Exception as exc:
            logger.error("Postgres schema setup failed: %s", exc)
        return

    try:
        with _sqlite_conn() as conn:
            conn.executescript(SQLITE_SCHEMA)
        _SCHEMA_READY = True
        logger.info("SQLite disaster history ready at %s", sqlite_db_path())
    except Exception as exc:
        logger.error("SQLite schema setup failed: %s", exc)


def _insert_related_rows(
    *,
    snapshot_id: int,
    snapshot: dict[str, Any],
    run_mode: str,
    recorded_at: str,
    execute,
) -> None:
    model_name = settings.EVO_MODEL_VERSION if run_mode == "evo" else "knn_reference_dataset"
    if run_mode == "evo13":
        model_name = "evo1.3_research"

    for quake in snapshot.get("earthquakes") or []:
        execute(
            """
            INSERT INTO hazard_events
                (snapshot_id, hazard_category, source, title, severity, magnitude,
                 center_lat, center_lon, recorded_at, raw)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                recorded_at,
                json.dumps(quake),
            ),
        )

    for alert in snapshot.get("alerts") or []:
        execute(
            """
            INSERT INTO hazard_events
                (snapshot_id, hazard_category, source, title, severity, magnitude,
                 center_lat, center_lon, recorded_at, raw)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                recorded_at,
                json.dumps(alert),
            ),
        )

    def insert_prediction(pred: dict[str, Any]) -> None:
        execute(
            """
            INSERT INTO evacuation_predictions
                (snapshot_id, spot_id, spot_name, model_name, event_type, occupancy, density,
                 predicted_evacuation_rate, predicted_evacuation_time_min, risk_level, recorded_at, raw)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                recorded_at,
                json.dumps(pred),
            ),
        )

    spot_preds = snapshot.get("spot_predictions") or []
    if spot_preds:
        for pred in spot_preds:
            insert_prediction(pred)
    else:
        seen_spots: set[str] = set()
        for alert in (snapshot.get("alerts") or []) + (snapshot.get("earthquakes") or []):
            for pred in alert.get("evacuation_predictions") or []:
                spot_key = str(pred.get("spot_id") or "")
                if not spot_key or spot_key in seen_spots:
                    continue
                seen_spots.add(spot_key)
                insert_prediction(pred)


def _save_postgres(snapshot: dict[str, Any], *, run_mode: str) -> Optional[int]:
    ensure_schema()
    summary = snapshot.get("summary") or {}
    try:
        with _postgres_connect() as conn:
            with conn.cursor() as cur:
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
                recorded_at = _iso(_utc_now())

                def execute(sql: str, params: tuple) -> None:
                    pg_sql = sql.replace("?", "%s")
                    cur.execute(pg_sql, params)

                _insert_related_rows(
                    snapshot_id=snapshot_id,
                    snapshot=snapshot,
                    run_mode=run_mode,
                    recorded_at=recorded_at,
                    execute=execute,
                )
            conn.commit()
        return snapshot_id
    except Exception as exc:
        logger.error("Failed to save disaster history (postgres): %s", exc)
        return None


def _save_sqlite(snapshot: dict[str, Any], *, run_mode: str) -> Optional[int]:
    ensure_schema()
    summary = snapshot.get("summary") or {}
    recorded_at = _iso(_utc_now())
    try:
        with _sqlite_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO disaster_snapshots
                    (captured_at, area, run_mode, active_alerts, significant_earthquakes,
                     high_risk_spots, snapshot)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    recorded_at,
                    snapshot.get("area"),
                    run_mode,
                    summary.get("active_alerts", 0),
                    summary.get("significant_earthquakes", 0),
                    summary.get("high_risk_spots", 0),
                    json.dumps(snapshot),
                ),
            )
            snapshot_id = int(cur.lastrowid)

            def execute(sql: str, params: tuple) -> None:
                cur.execute(sql, params)

            _insert_related_rows(
                snapshot_id=snapshot_id,
                snapshot=snapshot,
                run_mode=run_mode,
                recorded_at=recorded_at,
                execute=execute,
            )
        return snapshot_id
    except Exception as exc:
        logger.error("Failed to save disaster history (sqlite): %s", exc)
        return None


def save_dashboard_snapshot(
    snapshot: dict[str, Any],
    *,
    run_mode: str = "sync",
) -> Optional[int]:
    if postgres_configured():
        return _save_postgres(snapshot, run_mode=run_mode)
    return _save_sqlite(snapshot, run_mode=run_mode)


def _range_clause(
    since: Optional[datetime],
    until: Optional[datetime],
    column: str,
    *,
    postgres: bool,
) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if since:
        clauses.append(f"{column} >= {'%s' if postgres else '?'}")
        params.append(_iso(since) if not postgres else since)
    if until:
        clauses.append(f"{column} <= {'%s' if postgres else '?'}")
        params.append(_iso(until) if not postgres else until)
    if not clauses:
        return "", []
    return " WHERE " + " AND ".join(clauses), params


def get_recent_history(limit: int = 20) -> list[dict[str, Any]]:
    ensure_schema()
    if postgres_configured():
        return _postgres_recent(limit)
    return _sqlite_recent(limit)


def _row_to_snapshot_summary(row: Any) -> dict[str, Any]:
    captured = row["captured_at"] if isinstance(row, dict) else row[1]
    if hasattr(captured, "isoformat"):
        captured = captured.isoformat()
    if isinstance(row, dict):
        return {
            "id": row["id"],
            "captured_at": captured,
            "area": row["area"],
            "run_mode": row["run_mode"],
            "active_alerts": row["active_alerts"],
            "significant_earthquakes": row["significant_earthquakes"],
            "high_risk_spots": row["high_risk_spots"],
        }
    return {
        "id": row[0],
        "captured_at": captured,
        "area": row[2],
        "run_mode": row[3],
        "active_alerts": row[4],
        "significant_earthquakes": row[5],
        "high_risk_spots": row[6],
    }


def _postgres_recent(limit: int) -> list[dict[str, Any]]:
    try:
        with _postgres_connect() as conn, conn.cursor() as cur:
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
        logger.error("Failed to read disaster history (postgres): %s", exc)
        return []


def _sqlite_recent(limit: int) -> list[dict[str, Any]]:
    try:
        with _sqlite_conn() as conn:
            rows = conn.execute(
                """
                SELECT id, captured_at, area, run_mode, active_alerts,
                       significant_earthquakes, high_risk_spots
                FROM disaster_snapshots
                ORDER BY captured_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [_row_to_snapshot_summary(dict(row)) for row in rows]
    except Exception as exc:
        logger.error("Failed to read disaster history (sqlite): %s", exc)
        return []


def get_history_timeseries(
    *,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    ensure_schema()
    if postgres_configured():
        return _postgres_timeseries(since=since, until=until)
    return _sqlite_timeseries(since=since, until=until)


def _postgres_timeseries(
    *,
    since: Optional[datetime],
    until: Optional[datetime],
) -> list[dict[str, Any]]:
    where, params = _range_clause(since, until, "captured_at", postgres=True)
    sql = f"""
        SELECT captured_at, active_alerts, significant_earthquakes, high_risk_spots, run_mode
        FROM disaster_snapshots
        {where}
        ORDER BY captured_at ASC
    """
    try:
        with _postgres_connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [
            {
                "captured_at": row[0].isoformat() if row[0] else None,
                "active_alerts": row[1],
                "significant_earthquakes": row[2],
                "high_risk_spots": row[3],
                "run_mode": row[4],
            }
            for row in rows
        ]
    except Exception as exc:
        logger.error("Failed timeseries (postgres): %s", exc)
        return []


def _sqlite_timeseries(
    *,
    since: Optional[datetime],
    until: Optional[datetime],
) -> list[dict[str, Any]]:
    where, params = _range_clause(since, until, "captured_at", postgres=False)
    sql = f"""
        SELECT captured_at, active_alerts, significant_earthquakes, high_risk_spots, run_mode
        FROM disaster_snapshots
        {where}
        ORDER BY captured_at ASC
    """
    try:
        with _sqlite_conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "captured_at": row["captured_at"],
                "active_alerts": row["active_alerts"],
                "significant_earthquakes": row["significant_earthquakes"],
                "high_risk_spots": row["high_risk_spots"],
                "run_mode": row["run_mode"],
            }
            for row in rows
        ]
    except Exception as exc:
        logger.error("Failed timeseries (sqlite): %s", exc)
        return []


def query_high_risk_predictions(
    *,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    risk_level: Optional[str] = "high",
    spot_id: Optional[str] = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    ensure_schema()
    if postgres_configured():
        return _postgres_high_risk(
            since=since, until=until, risk_level=risk_level, spot_id=spot_id, limit=limit
        )
    return _sqlite_high_risk(
        since=since, until=until, risk_level=risk_level, spot_id=spot_id, limit=limit
    )


def _enrich_prediction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        rate = item.get("predicted_evacuation_rate")
        if rate is not None:
            try:
                item["predicted_evacuation_success_pct"] = round(float(rate) * 100, 2)
            except (TypeError, ValueError):
                pass
        enriched.append(item)
    return enriched


def _sqlite_high_risk(
    *,
    since: Optional[datetime],
    until: Optional[datetime],
    risk_level: Optional[str],
    spot_id: Optional[str],
    limit: int,
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if since:
        clauses.append("recorded_at >= ?")
        params.append(_iso(since))
    if until:
        clauses.append("recorded_at <= ?")
        params.append(_iso(until))
    if spot_id and spot_id.lower() not in {"all", "any"}:
        clauses.append("spot_id = ?")
        params.append(spot_id)
    if risk_level and risk_level.lower() not in {"all", "any"}:
        levels = [part.strip().lower() for part in risk_level.split(",") if part.strip()]
        if levels:
            placeholders = ", ".join("?" for _ in levels)
            clauses.append(f"LOWER(risk_level) IN ({placeholders})")
            params.extend(levels)
    where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"""
        SELECT spot_id, spot_name, risk_level, predicted_evacuation_rate,
               predicted_evacuation_time_min, event_type, occupancy, density,
               model_name, recorded_at, snapshot_id
        FROM evacuation_predictions
        {where_sql}
        ORDER BY recorded_at DESC
        LIMIT ?
    """
    try:
        with _sqlite_conn() as conn:
            rows = conn.execute(sql, [*params, limit]).fetchall()
        return _enrich_prediction_rows([dict(row) for row in rows])
    except Exception as exc:
        logger.error("Failed high-risk query (sqlite): %s", exc)
        return []


def _postgres_high_risk(
    *,
    since: Optional[datetime],
    until: Optional[datetime],
    risk_level: Optional[str],
    spot_id: Optional[str],
    limit: int,
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    all_params: list[Any] = []
    if since:
        clauses.append("recorded_at >= %s")
        all_params.append(since)
    if until:
        clauses.append("recorded_at <= %s")
        all_params.append(until)
    if spot_id and spot_id.lower() not in {"all", "any"}:
        clauses.append("spot_id = %s")
        all_params.append(spot_id)
    if risk_level and risk_level.lower() not in {"all", "any"}:
        levels = [part.strip().lower() for part in risk_level.split(",") if part.strip()]
        if levels:
            clauses.append("LOWER(risk_level) = ANY(%s)")
            all_params.append(levels)
    where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"""
        SELECT spot_id, spot_name, risk_level, predicted_evacuation_rate,
               predicted_evacuation_time_min, event_type, occupancy, density,
               model_name, recorded_at, snapshot_id
        FROM evacuation_predictions
        {where_sql}
        ORDER BY recorded_at DESC
        LIMIT %s
    """
    all_params.append(limit)
    try:
        with _postgres_connect() as conn, conn.cursor() as cur:
            cur.execute(sql, all_params)
            rows = cur.fetchall()
        keys = [
            "spot_id", "spot_name", "risk_level", "predicted_evacuation_rate",
            "predicted_evacuation_time_min", "event_type", "occupancy", "density",
            "model_name", "recorded_at", "snapshot_id",
        ]
        return _enrich_prediction_rows(
            [
                {
                    key: (value.isoformat() if hasattr(value, "isoformat") else value)
                    for key, value in zip(keys, row)
                }
                for row in rows
            ]
        )
    except Exception as exc:
        logger.error("Failed high-risk query (postgres): %s", exc)
        return []


def count_snapshots(
    *,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> int:
    ensure_schema()
    if postgres_configured():
        where, params = _range_clause(since, until, "captured_at", postgres=True)
        sql = f"SELECT COUNT(*) FROM disaster_snapshots{where}"
        try:
            with _postgres_connect() as conn, conn.cursor() as cur:
                cur.execute(sql, params)
                return int(cur.fetchone()[0])
        except Exception:
            return 0
    where, params = _range_clause(since, until, "captured_at", postgres=False)
    sql = f"SELECT COUNT(*) FROM disaster_snapshots{where}"
    try:
        with _sqlite_conn() as conn:
            row = conn.execute(sql, params).fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def _collect_export_rows(
    *,
    since: Optional[datetime],
    until: Optional[datetime],
    spot_id: Optional[str] = None,
) -> dict[str, list[dict[str, Any]]]:
    snapshots = get_history_timeseries(since=since, until=until)
    predictions = query_high_risk_predictions(
        since=since, until=until, risk_level="all", spot_id=spot_id, limit=5000
    )
    hazards: list[dict[str, Any]] = []
    ensure_schema()
    if postgres_configured():
        where, params = _range_clause(since, until, "recorded_at", postgres=True)
        sql = f"""
            SELECT hazard_category, source, title, severity, magnitude,
                   center_lat, center_lon, recorded_at, snapshot_id
            FROM hazard_events
            {where}
            ORDER BY recorded_at DESC
            LIMIT 5000
        """
        try:
            with _postgres_connect() as conn, conn.cursor() as cur:
                cur.execute(sql, params)
                keys = [
                    "hazard_category", "source", "title", "severity", "magnitude",
                    "center_lat", "center_lon", "recorded_at", "snapshot_id",
                ]
                hazards = [
                    {
                        k: (v.isoformat() if hasattr(v, "isoformat") else v)
                        for k, v in zip(keys, row)
                    }
                    for row in cur.fetchall()
                ]
        except Exception as exc:
            logger.error("Export hazards (postgres): %s", exc)
    else:
        where, params = _range_clause(since, until, "recorded_at", postgres=False)
        sql = f"""
            SELECT hazard_category, source, title, severity, magnitude,
                   center_lat, center_lon, recorded_at, snapshot_id
            FROM hazard_events
            {where}
            ORDER BY recorded_at DESC
            LIMIT 5000
        """
        try:
            with _sqlite_conn() as conn:
                hazards = [dict(row) for row in conn.execute(sql, params).fetchall()]
        except Exception as exc:
            logger.error("Export hazards (sqlite): %s", exc)

    return {
        "snapshots": snapshots,
        "predictions": predictions,
        "hazards": hazards,
    }


def build_json_export(
    *,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    spot_id: Optional[str] = None,
) -> bytes:
    payload = {
        "exported_at": _iso(_utc_now()),
        "storage": get_storage_info(),
        "range": {
            "since": _iso(since) if since else None,
            "until": _iso(until) if until else None,
            "spot_id": spot_id,
        },
        **_collect_export_rows(since=since, until=until, spot_id=spot_id),
    }
    return json.dumps(payload, indent=2).encode("utf-8")


def _dicts_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def build_csv_zip_export(
    *,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    spot_id: Optional[str] = None,
) -> bytes:
    data = _collect_export_rows(since=since, until=until, spot_id=spot_id)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, rows in data.items():
            archive.writestr(f"{name}.csv", _dicts_to_csv(rows))
        archive.writestr(
            "README.txt",
            "FCUSD evacuation intelligence export\n"
            "snapshots.csv — one row per Run Agent\n"
            "predictions.csv — evacuation predictions per spot (predicted_evacuation_success_pct is 0–100)\n"
            "hazards.csv — NOAA/USGS hazard events\n",
        )
    return buffer.getvalue()


def sqlite_export_path() -> Path:
    return sqlite_db_path()


def copy_sqlite_export() -> Path:
    """Return path to a temp copy of the SQLite DB for safe download."""
    ensure_schema()
    src = sqlite_db_path()
    if not src.exists():
        with _sqlite_conn():
            pass
    tmp = Path(tempfile.mkstemp(suffix=".db", prefix="disaster_history_")[1])
    shutil.copy2(src, tmp)
    return tmp


def parse_range_params(
    since: Optional[str],
    until: Optional[str],
) -> tuple[Optional[datetime], Optional[datetime]]:
    return _parse_dt(since), _parse_dt(until)
