-- Neon (Postgres) schema for disaster history and prediction analysis.
-- Run once in Neon SQL editor or via: psql $DATABASE_URL -f db/schema.sql

CREATE TABLE IF NOT EXISTS disaster_snapshots (
    id BIGSERIAL PRIMARY KEY,
    captured_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    area TEXT,
    run_mode TEXT NOT NULL DEFAULT 'sync',
    active_alerts INT DEFAULT 0,
    significant_earthquakes INT DEFAULT 0,
    high_risk_spots INT DEFAULT 0,
    snapshot JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS hazard_events (
    id BIGSERIAL PRIMARY KEY,
    snapshot_id BIGINT REFERENCES disaster_snapshots(id) ON DELETE CASCADE,
    hazard_category TEXT NOT NULL,
    source TEXT,
    title TEXT,
    severity TEXT,
    magnitude DOUBLE PRECISION,
    center_lat DOUBLE PRECISION,
    center_lon DOUBLE PRECISION,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    raw JSONB
);

CREATE TABLE IF NOT EXISTS evacuation_predictions (
    id BIGSERIAL PRIMARY KEY,
    snapshot_id BIGINT REFERENCES disaster_snapshots(id) ON DELETE CASCADE,
    spot_id TEXT,
    spot_name TEXT,
    model_name TEXT NOT NULL DEFAULT 'knn_reference_dataset',
    event_type TEXT,
    occupancy INT,
    density DOUBLE PRECISION,
    predicted_evacuation_rate DOUBLE PRECISION,
    predicted_evacuation_time_min DOUBLE PRECISION,
    risk_level TEXT,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    raw JSONB
);

CREATE INDEX IF NOT EXISTS idx_disaster_snapshots_captured_at ON disaster_snapshots (captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_hazard_events_category ON hazard_events (hazard_category, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_evac_predictions_spot ON evacuation_predictions (spot_id, recorded_at DESC);
