"""Train the yield-risk model on the earliest runs and test it on the latest.

    python -m src.model
"""
import json

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from . import config
from .db import get_engine
from .features import SensorPreprocessor, load_wide, time_split

MODEL_VERSION = "yield-risk-v1"


def candidates():
    out = {}
    for c in (0.01, 0.03, 0.1):
        out[f"logit_l1_C{c}"] = LogisticRegression(
            penalty="l1", C=c, solver="liblinear", class_weight="balanced", max_iter=2000)
    out["random_forest"] = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=5, class_weight="balanced_subsample",
        oob_score=True, n_jobs=-1, random_state=config.SEED)
    out["hist_gbm"] = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, class_weight="balanced",
        random_state=config.SEED)
    return out


def cv_scores(model, X_raw, y, n_splits=4):
    """Expanding-window CV. Preprocessing is refit inside every fold."""
    oof = np.full(len(y), np.nan)
    for tr, va in TimeSeriesSplit(n_splits=n_splits).split(X_raw):
        if y[tr].sum() < 5 or y[va].sum() < 2:
            continue
        pre = SensorPreprocessor().fit(X_raw.iloc[tr])
        m = model.fit(pre.transform(X_raw.iloc[tr]), y[tr])
        oof[va] = m.predict_proba(pre.transform(X_raw.iloc[va]))[:, 1]
    ok = ~np.isnan(oof)
    return {"roc_auc": roc_auc_score(y[ok], oof[ok]),
            "pr_auc": average_precision_score(y[ok], oof[ok])}


def capture_at(y, score, top_frac):
    k = max(1, int(round(len(y) * top_frac)))
    idx = np.argsort(-score)[:k]
    return {"runs_flagged": k, "fails_caught": int(y[idx].sum()),
            "recall": float(y[idx].sum() / max(y.sum(), 1)),
            "precision": float(y[idx].mean()),
            "lift": float(y[idx].mean() / y.mean())}


def bootstrap_auc(y, score, n=1000, seed=config.SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            vals.append(roc_auc_score(y[i], score[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    engine = get_engine()
    X, meta = load_wide(engine)
    y = meta["failed"].astype(int).to_numpy()
    train_ids, test_ids = time_split(meta)
    n_tr = len(train_ids)
    X_tr, X_te, y_tr, y_te = X.iloc[:n_tr], X.iloc[n_tr:], y[:n_tr], y[n_tr:]

    board = {name: cv_scores(m, X_tr, y_tr) for name, m in candidates().items()}
    best = max(board, key=lambda k: board[k]["roc_auc"])
    for name, s in sorted(board.items(), key=lambda kv: -kv[1]["roc_auc"]):
        print(f"  cv {name:16s} roc_auc={s['roc_auc']:.3f} pr_auc={s['pr_auc']:.3f}")
    print(f"selected: {best}")

    pre = SensorPreprocessor().fit(X_tr)
    model = candidates()[best].fit(pre.transform(X_tr), y_tr)
    p_tr = model.predict_proba(pre.transform(X_tr))[:, 1]
    p_te = model.predict_proba(pre.transform(X_te))[:, 1]

    metrics = {
        "model_version": MODEL_VERSION, "selected_model": best, "cv_leaderboard": board,
        "preprocessing": pre.report_,
        "train": {"runs": int(n_tr), "fails": int(y_tr.sum()),
                  "start": str(meta["run_ts"].iloc[0]), "end": str(meta["run_ts"].iloc[n_tr - 1])},
        "test": {"runs": int(len(y_te)), "fails": int(y_te.sum()),
                 "start": str(meta["run_ts"].iloc[n_tr]), "end": str(meta["run_ts"].iloc[-1]),
                 "base_fail_rate": float(y_te.mean()),
                 "roc_auc": float(roc_auc_score(y_te, p_te)),
                 "roc_auc_ci95": bootstrap_auc(y_te, p_te),
                 "pr_auc": float(average_precision_score(y_te, p_te)),
                 "top_10pct": capture_at(y_te, p_te, 0.10),
                 "top_20pct": capture_at(y_te, p_te, 0.20),
                 "top_30pct": capture_at(y_te, p_te, 0.30)},
    }
    config.ARTIFACTS.mkdir(exist_ok=True)
    # Reference distribution for the 0-100 score. In-sample forest probabilities are
    # overfit, so use out-of-bag predictions on the training runs when available.
    ref = model.oob_decision_function_[:, 1] if hasattr(model, "oob_decision_function_") else p_tr
    importance = getattr(model, "feature_importances_", None)
    if importance is None:
        importance = np.abs(model.coef_).ravel()
    top = [int(pre.columns_[i]) for i in np.argsort(-importance)[:30]]
    metrics["top_model_sensors"] = top[:15]
    joblib.dump({"pre": pre, "model": model, "ref_probs": np.sort(ref), "top_sensors": top,
                 "warm_probs": np.asarray(ref)[-config.RISK_WINDOW:],
                 "train_end": meta["run_ts"].iloc[n_tr - 1], "version": MODEL_VERSION},
                config.ARTIFACTS / "model.joblib")
    (config.ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    t = metrics["test"]
    print(f"held-out test: roc_auc={t['roc_auc']:.3f} (95% CI {t['roc_auc_ci95'][0]:.3f}-"
          f"{t['roc_auc_ci95'][1]:.3f}) pr_auc={t['pr_auc']:.3f} base={t['base_fail_rate']:.3f}")
    for k in ("top_10pct", "top_20pct", "top_30pct"):
        print(f"  {k}: {t[k]}")


if __name__ == "__main__":
    main()
