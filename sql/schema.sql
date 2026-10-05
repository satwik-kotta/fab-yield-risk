-- Fab line schema: one row per wafer run, long-format sensor readings,
-- inspection outcome arriving separately, model scores written back.
CREATE TABLE IF NOT EXISTS wafer_runs (
    run_id      INTEGER PRIMARY KEY,
    run_ts      TIMESTAMP NOT NULL,
    ingested_at TIMESTAMP NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_wafer_runs_ts ON wafer_runs (run_ts);

CREATE TABLE IF NOT EXISTS sensor_readings (
    run_id    INTEGER  NOT NULL REFERENCES wafer_runs (run_id) ON DELETE CASCADE,
    sensor_id SMALLINT NOT NULL,
    value     DOUBLE PRECISION,           -- NULL = sensor did not report
    PRIMARY KEY (run_id, sensor_id)
);
CREATE INDEX IF NOT EXISTS ix_readings_sensor ON sensor_readings (sensor_id);

CREATE TABLE IF NOT EXISTS inspection_results (
    run_id       INTEGER PRIMARY KEY REFERENCES wafer_runs (run_id) ON DELETE CASCADE,
    failed       BOOLEAN NOT NULL,
    inspected_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS risk_scores (
    run_id        INTEGER PRIMARY KEY REFERENCES wafer_runs (run_id) ON DELETE CASCADE,
    fail_prob     DOUBLE PRECISION NOT NULL,
    risk_score    DOUBLE PRECISION NOT NULL,   -- 0-100, rank vs the previous 200 scored runs
    risk_score_fixed DOUBLE PRECISION,         -- 0-100, rank vs training runs (kept for comparison)
    risk_tier     TEXT NOT NULL,
    top_sensors   TEXT,
    model_version TEXT NOT NULL,
    scored_at     TIMESTAMP NOT NULL DEFAULT now()
);
