"""Read runs out of PostgreSQL and turn raw sensors into model features."""
import numpy as np
import pandas as pd
from sqlalchemy import text

from . import config


def load_wide(engine, run_ids=None):
    """Return (X, meta): X is runs x sensors, meta has run_ts and failed."""
    where = ""
    params = {}
    if run_ids is not None:
        where = "WHERE run_id = ANY(:ids)"
        params = {"ids": list(map(int, run_ids))}
    with engine.connect() as conn:
        long = pd.read_sql(text(
            f"SELECT run_id, sensor_id, value FROM sensor_readings {where}"), conn, params=params)
        meta = pd.read_sql(text(
            f"""SELECT r.run_id, r.run_ts, i.failed
                FROM wafer_runs r LEFT JOIN inspection_results i USING (run_id)
                {where.replace('run_id', 'r.run_id')} ORDER BY r.run_ts, r.run_id"""),
            conn, params=params)
    X = long.pivot(index="run_id", columns="sensor_id", values="value")
    meta = meta.set_index("run_id")
    X = X.loc[meta.index]
    X.columns = [int(c) for c in X.columns]
    return X, meta


def time_split(meta, train_fraction=config.TRAIN_FRACTION):
    """Earliest runs train, latest runs test. No shuffling: the line drifts."""
    n_train = int(len(meta) * train_fraction)
    return meta.index[:n_train], meta.index[n_train:]


class SensorPreprocessor:
    """Fit on training runs only, then applied unchanged to live runs.

    1. drop sensors that are mostly missing or never move
    2. drop one of each pair of near-duplicate sensors
    3. fill gaps with the training median
    4. scale by training median and MAD (robust to the spikes we care about)
    """

    def __init__(self, max_missing=config.MAX_MISSING, max_corr=config.MAX_CORR):
        self.max_missing = max_missing
        self.max_corr = max_corr

    def fit(self, X):
        miss = X.isna().mean()
        keep = miss[miss <= self.max_missing].index
        spread = X[keep].nunique(dropna=True)
        keep = [c for c in keep if spread[c] > 1]
        self.report_ = {"raw": X.shape[1],
                        "dropped_missing": int((miss > self.max_missing).sum()),
                        "dropped_constant": int(X.shape[1] - (miss > self.max_missing).sum() - len(keep))}

        med = X[keep].median()
        filled = X[keep].fillna(med)
        corr = np.nan_to_num(filled.corr().abs().to_numpy(copy=True))
        np.fill_diagonal(corr, 0.0)
        dropped = set()
        for i in range(len(keep)):
            if i in dropped:
                continue
            for j in np.where(corr[i, i + 1:] > self.max_corr)[0] + i + 1:
                dropped.add(int(j))
        self.columns_ = [c for k, c in enumerate(keep) if k not in dropped]
        self.report_["dropped_duplicate"] = len(dropped)
        self.report_["kept"] = len(self.columns_)

        self.median_ = med[self.columns_]
        mad = (filled[self.columns_] - self.median_).abs().median() * 1.4826
        std = filled[self.columns_].std()
        self.scale_ = mad.where(mad > 1e-9, std).where(lambda s: s > 1e-9, 1.0)
        return self

    def transform(self, X):
        Z = (X[self.columns_].fillna(self.median_) - self.median_) / self.scale_
        return Z.clip(-10, 10)       # cap extreme spikes so one glitch cannot dominate

    def fit_transform(self, X):
        return self.fit(X).transform(X)
