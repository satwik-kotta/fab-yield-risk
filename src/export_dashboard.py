"""Query PostgreSQL and write docs/data.js for the static dashboard.

    python -m src.export_dashboard
"""
import json

import pandas as pd
from sqlalchemy import text

from . import config
from .db import get_engine


def q(conn, sql):
    return pd.read_sql(text(sql), conn)


def main():
    engine = get_engine()
    with engine.connect() as conn:
        runs = q(conn, """SELECT run_id, to_char(run_ts, 'YYYY-MM-DD HH24:MI') AS ts,
                                 round(risk_score::numeric, 1)::float AS risk, risk_tier AS tier,
                                 round(fail_prob::numeric, 4)::float AS prob,
                                 top_sensors, failed FROM v_scored_runs""")
        tiers = q(conn, "SELECT * FROM v_tier_performance")
        daily = q(conn, """SELECT to_char(day, 'YYYY-MM-DD') AS day, runs, fails,
                                  yield_pct::float, fail_rate_7d_pct::float FROM v_daily_yield""")
        totals = q(conn, """SELECT count(*) AS runs, count(*) FILTER (WHERE failed) AS fails
                            FROM inspection_results""").iloc[0]
        readings = int(q(conn, "SELECT count(*) AS n FROM sensor_readings").iloc[0, 0])

    metrics = json.loads((config.ARTIFACTS / "metrics.json").read_text())
    causal_summary = json.loads((config.ARTIFACTS / "causal_summary.json").read_text())
    effects = pd.read_csv(config.ARTIFACTS / "causal_effects.csv")
    shown = effects[effects.naive_flag | effects.causal_flag].sort_values("naive_diff", ascending=False)
    cols = ["sensor", "excursion_runs", "naive_diff", "naive_q", "identified", "ate",
            "ci_low", "ci_high", "p", "q", "causal_flag", "overlap"]
    order = {"High": 0, "Elevated": 1, "Normal": 2}
    tiers = tiers.sort_values("risk_tier", key=lambda s: s.map(order))

    data = {
        "totals": {"runs": int(totals.runs), "fails": int(totals.fails), "readings": readings},
        "metrics": metrics, "causal_summary": causal_summary,
        "runs": json.loads(runs.to_json(orient="records")),
        "tiers": json.loads(tiers.to_json(orient="records")),
        "daily": json.loads(daily.to_json(orient="records")),
        "effects": json.loads(shown[cols].to_json(orient="records")),
    }
    config.DOCS.mkdir(exist_ok=True)
    (config.DOCS / "data.js").write_text("window.FAB_DATA = " + json.dumps(data) + ";\n")
    print(f"wrote docs/data.js: {len(data['runs'])} scored runs, {len(data['effects'])} sensors")


if __name__ == "__main__":
    main()
