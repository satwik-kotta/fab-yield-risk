-- Daily yield with a 7-day rolling fail rate.
CREATE OR REPLACE VIEW v_daily_yield AS
WITH daily AS (
    SELECT r.run_ts::date AS day,
           count(*)                                   AS runs,
           count(*) FILTER (WHERE i.failed)           AS fails
    FROM wafer_runs r
    JOIN inspection_results i USING (run_id)
    GROUP BY 1
)
SELECT day, runs, fails,
       round(100.0 * (runs - fails) / runs, 2) AS yield_pct,
       round(100.0 * sum(fails) OVER w / NULLIF(sum(runs) OVER w, 0), 2) AS fail_rate_7d_pct
FROM daily
WINDOW w AS (ORDER BY day RANGE BETWEEN INTERVAL '6 days' PRECEDING AND CURRENT ROW)
ORDER BY day;

-- Per-sensor health: missing rate and spread, used for data-quality checks.
CREATE OR REPLACE VIEW v_sensor_health AS
SELECT sensor_id,
       count(*)                                         AS runs,
       round(100.0 * count(*) FILTER (WHERE value IS NULL) / count(*), 2) AS missing_pct,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY value) AS median_value,
       stddev_samp(value)                               AS std_value
FROM sensor_readings
GROUP BY sensor_id;

-- Runs waiting for a score: what the scoring service polls.
CREATE OR REPLACE VIEW v_unscored_runs AS
SELECT r.run_id, r.run_ts
FROM wafer_runs r
LEFT JOIN risk_scores s USING (run_id)
WHERE s.run_id IS NULL
ORDER BY r.run_ts;

-- Scored runs joined to their eventual inspection outcome.
CREATE OR REPLACE VIEW v_scored_runs AS
SELECT r.run_id, r.run_ts, s.fail_prob, s.risk_score, s.risk_tier, s.top_sensors,
       i.failed, s.model_version
FROM wafer_runs r
JOIN risk_scores s USING (run_id)
LEFT JOIN inspection_results i USING (run_id)
ORDER BY r.run_ts;

-- How much of the scrap each risk tier catches.
CREATE OR REPLACE VIEW v_tier_performance AS
SELECT risk_tier,
       count(*)                              AS runs,
       count(*) FILTER (WHERE failed)        AS fails,
       round(100.0 * count(*) FILTER (WHERE failed) / count(*), 2) AS fail_rate_pct,
       round(100.0 * count(*) FILTER (WHERE failed)
             / NULLIF(sum(count(*) FILTER (WHERE failed)) OVER (), 0), 2) AS share_of_fails_pct
FROM v_scored_runs
WHERE failed IS NOT NULL
GROUP BY risk_tier;
