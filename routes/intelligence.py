"""Public dashboard and intelligence API routes."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from config.settings import settings
from services.agent_runner import run_full_agent_cycle, run_full_agent_cycle_sync
from services.alert_processor import AlertProcessor
from services.disaster_history import (
    build_csv_zip_export,
    build_json_export,
    copy_sqlite_export,
    count_snapshots,
    get_history_timeseries,
    get_recent_history,
    get_storage_info,
    parse_range_params,
    postgres_configured,
    query_high_risk_predictions,
    storage_backend,
)
from services.evo_live_flow import build_live_flow
from services.evo_runtime import get_evo_runtime
from services.pipeline_status import get_pipeline_status
from services.run_modes import RunMode, execute_run_mode_sync
from services.evacuation_predictor import EvacuationPredictor
from services.fema_ipaws_client import FEMAIPAWSClient
from services.government_feed_sync import GovernmentFeedSync
from services.hazard_feature_builder import build_training_seed_rows, write_training_seed
from services.noaa_client import NOAAClient
from services.peoplesense_client import PeopleSenseClient
from services.peoplesense_deployment import PeopleSenseDeploymentService
from services.usgs_client import USGSClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["intelligence"])

_processor = AlertProcessor()
_noaa = NOAAClient()
_peoplesense = PeopleSenseClient()
_predictor = EvacuationPredictor()
_usgs = USGSClient()
_fema = FEMAIPAWSClient()
_deployer = PeopleSenseDeploymentService()
_feed_sync = GovernmentFeedSync()


class SimpleEventRequest(BaseModel):
    """Larry's Phase 1 simplified event POST: GPS, diameter, occupancy."""

    center_lat: float = Field(..., description="Event center latitude")
    center_lon: float = Field(..., description="Event center longitude")
    diameter_miles: float = Field(default=10.0, ge=0.1, description="Affected area diameter")
    occupancy: int = Field(..., ge=0, description="Estimated occupancy in affected area")
    event_type: str = Field(default="earthquake", description="Hazard type")
    title: Optional[str] = Field(default=None, description="Optional event title")
    description: Optional[str] = Field(default=None, description="Optional description")
    deploy_peoplesense: bool = Field(default=True, description="Auto-deploy PeopleSense zone")
    trigger_agent: bool = Field(default=True, description="Trigger evacuation intelligence agent")


class SpotPredictionRequest(BaseModel):
    spot_id: str
    name: str
    category: str = "Office Building"
    occupancy: int = Field(..., ge=0)
    density: float = Field(..., ge=0)
    event_type: str = "other"
    lat: Optional[float] = None
    lon: Optional[float] = None


class EvoAcceleratorRequest(BaseModel):
    accelerator: str = Field(
        ...,
        description="auto | cpu | ncs1 | ncs2 — Neural Compute Stick or CPU inference",
    )


@router.post("/event")
async def post_simple_event(request: SimpleEventRequest, background_tasks: BackgroundTasks):
    """
    Simplified Phase 1 event intake: GPS coordinates, diameter, and occupancy.

    Auto-deploys a PeopleSense zone and optionally triggers the agent pipeline.
    """
    from services.pipeline import run_pipeline

    deployment = None
    if request.deploy_peoplesense:
        deployment = _deployer.deploy_from_event(
            center_lat=request.center_lat,
            center_lon=request.center_lon,
            diameter_miles=request.diameter_miles,
            occupancy=request.occupancy,
            event_type=request.event_type,
            title=request.title,
        )

    event_dict = {
        "status": "start",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": request.event_type if request.event_type in (
            "earthquake", "fire", "tsunami", "tornado", "flood", "other"
        ) else "other",
        "center_lat": request.center_lat,
        "center_lon": request.center_lon,
        "diameter_miles": request.diameter_miles,
        "affected_population": request.occupancy,
        "title": request.title or f"{request.event_type.title()} at ({request.center_lat}, {request.center_lon})",
        "description": request.description or (
            f"Phase 1 event: {request.occupancy} occupants within "
            f"{request.diameter_miles} mile diameter."
        ),
    }

    if request.trigger_agent:
        background_tasks.add_task(run_pipeline, event_dict)

    return {
        "status": "accepted",
        "event": event_dict,
        "peoplesense_deployment": deployment,
        "pipeline_triggered": request.trigger_agent,
    }


@router.get("/earthquakes")
async def get_earthquakes(
    window: str = Query(default="hour", description="hour, day, week, or month"),
    min_magnitude: Optional[float] = Query(default=None),
    limit: int = Query(default=25, ge=1, le=100),
):
    quakes = _usgs.get_recent_earthquakes(
        window=window if window in ("hour", "day", "week", "month") else "hour",
        min_magnitude=min_magnitude,
        limit=limit,
    )
    return {"count": len(quakes), "earthquakes": quakes}


@router.get("/earthquakes/eew")
async def get_earthquake_early_warnings(
    min_magnitude: Optional[float] = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
):
    """
    Earthquake early-warning candidates from USGS public feeds.

    Production ShakeAlert XML requires a USGS Technical Partnership.
    """
    candidates = _usgs.get_earthquake_early_warning_candidates(
        min_magnitude=min_magnitude,
        limit=limit,
    )
    return {
        "count": len(candidates),
        "shakealert_partnership_required_for_live_xml": True,
        "min_magnitude_threshold": min_magnitude or settings.USGS_EEW_MIN_MAGNITUDE,
        "candidates": candidates,
    }


@router.get("/fema/ipaws")
async def get_fema_ipaws_alerts(
    hours: int = Query(default=48, ge=1, le=168),
    limit: int = Query(default=25, ge=1, le=100),
):
    alerts = _fema.get_recent_alerts(hours=hours, limit=limit)
    return {"count": len(alerts), "alerts": alerts, "note": "Archived IPAWS may lag ~24h; use NOAA for live alerts"}


@router.post("/peoplesense/deploy")
async def deploy_peoplesense_zone(
    name: str = Query(...),
    lat: float = Query(...),
    lon: float = Query(...),
    radius_m: float = Query(default=500),
    occupancy: Optional[int] = Query(default=None),
):
    return _deployer.deploy_zone(
        name=name, lat=lat, lon=lon, radius_m=radius_m, occupancy=occupancy
    )


@router.post("/peoplesense/auto-deploy")
async def auto_deploy_peoplesense():
    """Deploy all configured monitoring spots to PeopleSense."""
    deployments = _deployer.auto_deploy_monitoring_spots()
    return {
        "status": "deployed",
        "count": len(deployments),
        "deployments": deployments,
        "peoplesense_status": _deployer.get_deployment_status(),
    }


@router.get("/peoplesense/deployments")
async def get_peoplesense_deployments():
    return _deployer.get_deployment_status()


@router.post("/feeds/sync")
async def sync_government_feeds(
    background_tasks: BackgroundTasks,
    area: Optional[str] = Query(default=None),
    auto_deploy: bool = Query(default=True),
    sync: bool = Query(default=False, description="Wait for sync to complete"),
):
    """
    Ingest USGS + FEMA + NOAA feeds and auto-deploy PeopleSense zones for hazards.
    """
    area = area or settings.DEFAULT_ALERT_AREA

    if sync:
        return _feed_sync.sync_all(area=area, auto_deploy=auto_deploy)

    background_tasks.add_task(_feed_sync.sync_all, area=area, auto_deploy=auto_deploy)
    return {
        "status": "accepted",
        "message": "Government feed sync started (NOAA, USGS, FEMA, GDACS, NASA FIRMS)",
        "area": area,
        "auto_deploy": auto_deploy,
    }


@router.get("/feeds/sources")
async def get_hazard_feed_sources():
    """Catalog of integrated public hazard data sources."""
    from pathlib import Path
    import json

    path = settings.PROJECT_ROOT / "data" / "reference" / "hazard_feed_sources.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"sources": [], "note": "hazard_feed_sources.json missing"}


@router.post("/training/hazard-seed")
async def refresh_hazard_training_seed(
    persist: bool = Query(default=True, description="Write data/processed/hazard_live_seed.json"),
):
    """Pull live NOAA/USGS/GDACS/FIRMS features for FCUSD monitoring spots."""
    if persist:
        return write_training_seed()
    return build_training_seed_rows()


@router.get("/dashboard")
def get_dashboard(
    area: Optional[str] = Query(default=None, description="US state code, e.g. CA"),
    lat: Optional[float] = Query(default=None),
    lon: Optional[float] = Query(default=None),
    use_evo: bool = Query(default=False),
    use_evo13: bool = Query(default=False),
):
    point = (lat, lon) if lat is not None and lon is not None else None
    if use_evo13:
        processor = AlertProcessor(use_evo13=True)
    elif use_evo:
        processor = AlertProcessor(use_evo=True)
    else:
        processor = _processor
    return processor.get_dashboard_snapshot(area=area, point=point)


@router.get("/history")
async def get_disaster_history(limit: int = Query(default=20, ge=1, le=100)):
    storage = get_storage_info()
    return {
        "storage": storage,
        "neon_configured": postgres_configured(),
        "total_snapshots": count_snapshots(),
        "snapshots": get_recent_history(limit=limit),
    }


@router.get("/history/timeseries")
async def get_history_timeseries_route(
    since: Optional[str] = Query(default=None),
    until: Optional[str] = Query(default=None),
):
    since_dt, until_dt = parse_range_params(since, until)
    return {
        "storage": get_storage_info(),
        "range": {"since": since, "until": until},
        "points": get_history_timeseries(since=since_dt, until=until_dt),
        "count": count_snapshots(since=since_dt, until=until_dt),
    }


@router.get("/history/high-risk")
async def get_high_risk_history(
    since: Optional[str] = Query(default=None),
    until: Optional[str] = Query(default=None),
    risk_level: str = Query(default="high", description="high, medium, high,medium, or all"),
    limit: int = Query(default=200, ge=1, le=5000),
):
    since_dt, until_dt = parse_range_params(since, until)
    rows = query_high_risk_predictions(
        since=since_dt,
        until=until_dt,
        risk_level=risk_level,
        limit=limit,
    )
    return {
        "storage": get_storage_info(),
        "range": {"since": since, "until": until},
        "risk_level": risk_level,
        "count": len(rows),
        "predictions": rows,
    }


@router.get("/history/export")
async def export_disaster_history(
    format: str = Query(default="json", pattern="^(json|csv|sqlite)$"),
    since: Optional[str] = Query(default=None),
    until: Optional[str] = Query(default=None),
):
    since_dt, until_dt = parse_range_params(since, until)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if format == "json":
        payload = build_json_export(since=since_dt, until=until_dt)
        return Response(
            content=payload,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="fcusd_history_{stamp}.json"'},
        )

    if format == "csv":
        payload = build_csv_zip_export(since=since_dt, until=until_dt)
        return Response(
            content=payload,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="fcusd_history_{stamp}.zip"'},
        )

    if storage_backend() != "sqlite":
        raise HTTPException(
            status_code=400,
            detail="SQLite file export is only available with local SQLite storage. Use format=json or format=csv, or export from Neon.",
        )

    db_copy = copy_sqlite_export()
    try:
        data = db_copy.read_bytes()
    finally:
        db_copy.unlink(missing_ok=True)

    return Response(
        content=data,
        media_type="application/x-sqlite3",
        headers={"Content-Disposition": f'attachment; filename="fcusd_disaster_history_{stamp}.db"'},
    )


@router.get("/evo/visualization")
async def get_evo_visualization():
    return get_evo_runtime().get_visualization()


@router.get("/evo/live-flow")
async def get_evo_live_flow(
    use_evo: bool = Query(default=True),
    spot_id: Optional[str] = Query(default=None),
):
    """Live inference trace for the interactive Evo network visualization."""
    return build_live_flow(use_evo=use_evo, spot_id=spot_id)


@router.get("/evo/runtime")
async def get_evo_runtime_status():
    """Active Evo inference backend (CPU, NCS1/NCS2, or ONNX fallback)."""
    return get_evo_runtime().get_runtime_status()


@router.post("/evo/accelerator")
async def set_evo_accelerator(request: EvoAcceleratorRequest):
    """Select inference device: auto, cpu, ncs1, or ncs2 (Neural Compute Stick)."""
    choice = request.accelerator.strip().lower()
    if choice not in {"auto", "cpu", "ncs1", "ncs2"}:
        raise HTTPException(
            status_code=400,
            detail="accelerator must be one of: auto, cpu, ncs1, ncs2",
        )
    return get_evo_runtime().set_accelerator(choice)


@router.get("/alerts")
async def get_alerts(
    area: Optional[str] = Query(default=None),
    lat: Optional[float] = Query(default=None),
    lon: Optional[float] = Query(default=None),
    limit: int = Query(default=25, ge=1, le=100),
):
    point = (lat, lon) if lat is not None and lon is not None else None
    alerts = _noaa.get_active_alerts(area=area or settings.DEFAULT_ALERT_AREA, point=point, limit=limit)
    return {"count": len(alerts), "alerts": alerts}


@router.get("/alerts/{alert_id}/analysis")
async def analyze_alert(alert_id: str):
    alert = _noaa.get_alert_by_id(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return _processor.enrich_alert(alert)


@router.get("/peoplesense/zones")
def get_peoplesense_zones():
    readings = [
        _peoplesense.get_zone_occupancy(
            lat=spot["lat"],
            lon=spot["lon"],
            radius_m=spot.get("radius_m", 400),
            zone_name=spot["name"],
            spot=spot,
        )
        for spot in _processor.locations
    ]
    return {
        "mode": "placeholder" if _peoplesense.is_placeholder else "live",
        "source": "get_api" if _peoplesense.occupancy_api_enabled else "event_api",
        "occupancy_url": settings.PEOPLESENSE_OCCUPANCY_URL,
        "event_url": settings.PEOPLESENSE_EVENT_URL,
        "cache_ttl_seconds": settings.PEOPLESENSE_CACHE_TTL_SECONDS,
        "zones": readings,
    }


@router.get("/peoplesense/occupancy")
def get_peoplesense_occupancy(filter: str = "ALL"):
    """Fetch cached PeopleSense occupancy database (respects 60s rate limit)."""
    payload = _peoplesense.fetch_all_occupancy(filter_value=filter)
    return {
        "filter": payload.get("filter", filter),
        "count": payload.get("count", len(payload.get("data") or [])),
        "fetched_at": payload.get("fetchedAt"),
        "cache_hit": payload.get("cache_hit", False),
        "cache_age_seconds": payload.get("cache_age_seconds"),
        "mode": payload.get("mode", "live" if not _peoplesense.is_placeholder else "placeholder"),
        "data": payload.get("data") or [],
    }


class PeopleSenseEventRequest(BaseModel):
    occupancy_xml: str = Field(alias="OccupancyXML")
    message: str = Field(alias="Message")

    model_config = {"populate_by_name": True}


@router.post("/peoplesense/event")
async def post_peoplesense_event(request: PeopleSenseEventRequest):
    """Forward OccupancyXML + Message to the FCUSD PeopleSense event API."""
    return _peoplesense.post_event(
        occupancy_xml=request.occupancy_xml,
        message=request.message,
    )


@router.post("/evacuation/predict")
async def predict_evacuation(request: SpotPredictionRequest):
    return _predictor.predict_for_spot(
        spot_id=request.spot_id,
        name=request.name,
        category=request.category,
        occupancy=request.occupancy,
        density=request.density,
        event_type=request.event_type,
        lat=request.lat,
        lon=request.lon,
    )


@router.get("/pipeline/status")
async def get_pipeline_status_endpoint():
    """Poll pipeline progress for the dashboard loading bar."""
    return get_pipeline_status().get()


def _begin_pipeline_run() -> str:
    tracker = get_pipeline_status()
    if tracker.is_running():
        raise HTTPException(status_code=409, detail="Pipeline is already running")
    return tracker.start("Starting emergency pipeline…")


@router.post("/agent/run")
async def run_evacuation_agent(
    background_tasks: BackgroundTasks,
    area: Optional[str] = Query(default=None, description="US state code, e.g. CA"),
    sync: bool = Query(default=False, description="Wait for full pipeline to finish before responding"),
):
    """
    Run the full emergency agent cycle: sync government feeds, refresh dashboard,
    and execute the complete pipeline (sitrep, article, broadcast_script.md, etc.).
    """
    area = area or settings.DEFAULT_ALERT_AREA

    if sync:
        try:
            _begin_pipeline_run()
            result = await run_full_agent_cycle(area)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Full agent cycle failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return result

    try:
        job_id = _begin_pipeline_run()
    except HTTPException as exc:
        raise exc

    background_tasks.add_task(run_full_agent_cycle_sync, area)
    return {
        "status": "accepted",
        "job_id": job_id,
        "message": (
            f"Full emergency pipeline started for area={area}. "
            "Poll /api/pipeline/status for progress."
        ),
        "pipeline": "full",
    }


@router.post("/alerts/sync")
async def sync_alerts(
    background_tasks: BackgroundTasks,
    mode: RunMode = Query(default="sync", description="sync | external_ai | evo | evo13 | broadcast"),
):
    """Sync feeds and optionally run AI/broadcast depending on mode."""
    area = settings.DEFAULT_ALERT_AREA
    try:
        job_id = _begin_pipeline_run()
    except HTTPException as exc:
        if exc.status_code == 409:
            return {
                "status": "already_running",
                "job_id": get_pipeline_status().get().get("job_id"),
                "message": "Pipeline is already in progress",
            }
        raise

    background_tasks.add_task(execute_run_mode_sync, mode, area)
    messages = {
        "sync": "Sync only — NOAA/USGS/FEMA + predictions (no LLM)",
        "external_ai": "Sync + Gemini/OpenAI summary with auto-failover",
        "evo": "Sync + Evo 1.2 hybrid predictions (ONNX/OpenVINO) — production",
        "evo13": "Sync + Evo 1.3 research (internet + enriched reference) — not production",
        "broadcast": "Full 7-step broadcast pipeline",
    }
    return {
        "status": "accepted",
        "job_id": job_id,
        "run_mode": mode,
        "message": messages.get(mode, "Pipeline started"),
        "area": area,
    }
