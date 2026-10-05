"""Scoring service: pick up runs that have no score yet and write a Yield Risk Score.

    python -m src.score            # score everything waiting, then exit
    python -m src.score --watch 2  # keep polling every 2 seconds (use with live replay)

Only runs after the training window are scored, so every score in the
database is out-of-sample.
"""
import argparse
import time

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text

from . import config
from .db import get_engine
from .features import load_wide


def tier(score):
    if score >= config.HIGH_TIER:
        return "High"
    if score >= config.ELEVATED_TIER:
        return "Elevated"
    return "Normal"


def rolling_risk(probs, history, window=config.RISK_WINDOW):
    """Rank each run against the `window` runs before it (0-100).

    The line drifts, so a fixed cut-off learned on the training months goes
    stale. Ranking against recent runs needs no labels, so it works live.
    `history` holds the fail probabilities of the runs before this batch.
    """
    hist = list(history)[-window:]
    out = []
    for p in probs:
        ref = np.asarray(hist[-window:])
        out.append(100.0 * float((ref <= p).mean()) if len(ref) else 50.0)
        hist.append(p)
    return np.array(out)


def score_frame(bundle, X, history):
    """X must be in time order."""
    Z = bundle["pre"].transform(X)
    prob = bundle["model"].predict_proba(Z)[:, 1]
    risk = rolling_risk(prob, history)
    fixed = 100.0 * np.searchsorted(bundle["ref_probs"], prob, side="right") / len(bundle["ref_probs"])
    watch = [c for c in bundle["top_sensors"] if c in Z.columns]
    tops = []
    for _, row in Z[watch].iterrows():
        worst = row.abs().sort_values(ascending=False).head(3)
        tops.append(", ".join(f"S{c} ({row[c]:+.1f}σ)" for c in worst.index))
    return pd.DataFrame({"run_id": X.index, "fail_prob": prob, "risk_score": risk,
                         "risk_score_fixed": fixed,
                         "risk_tier": [tier(r) for r in risk], "top_sensors": tops,
                         "model_version": bundle["version"]})


def score_pending(engine, bundle):
    with engine.connect() as conn:
        ids = [r[0] for r in conn.execute(text(
            "SELECT run_id FROM v_unscored_runs WHERE run_ts > :t"),
            {"t": bundle["train_end"]})]
        if not ids:
            return 0
        prev = [r[0] for r in conn.execute(text(
            """SELECT fail_prob FROM (
                   SELECT s.fail_prob, r.run_ts, r.run_id FROM risk_scores s JOIN wafer_runs r USING (run_id)
                   ORDER BY r.run_ts DESC, r.run_id DESC LIMIT :n) t
               ORDER BY run_ts, run_id"""), {"n": config.RISK_WINDOW})]
    history = list(bundle["warm_probs"]) + prev      # training out-of-bag scores warm the window
    X, _ = load_wide(engine, ids)
    out = score_frame(bundle, X, history)
    with engine.begin() as conn:
        conn.execute(text(
            """INSERT INTO risk_scores (run_id, fail_prob, risk_score, risk_score_fixed, risk_tier, top_sensors, model_version)
               VALUES (:run_id, :fail_prob, :risk_score, :risk_score_fixed, :risk_tier, :top_sensors, :model_version)
               ON CONFLICT (run_id) DO NOTHING"""), out.to_dict("records"))
    return len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=float, default=None)
    args = ap.parse_args()
    engine = get_engine()
    bundle = joblib.load(config.ARTIFACTS / "model.joblib")
    while True:
        n = score_pending(engine, bundle)
        if n:
            print(f"scored {n} runs", flush=True)
        if args.watch is None:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
