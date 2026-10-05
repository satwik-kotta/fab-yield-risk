import numpy as np
import pandas as pd

from src.causal import propensity, stratified_effect
from src.features import SensorPreprocessor
from src.score import rolling_risk, tier
from src.stream_simulator import fix_timestamp, load_raw


def test_timestamp_swap_is_undone():
    assert fix_timestamp(pd.Timestamp("2008-01-08 02:02")) == pd.Timestamp("2008-08-01 02:02")
    assert fix_timestamp(pd.Timestamp("2008-07-19 11:55")) == pd.Timestamp("2008-07-19 11:55")


def test_raw_file_matches_published_secom():
    df = load_raw()
    assert len(df) == 1567 and sum(c.isdigit() for c in df.columns) == 590
    assert (df["Pass_Fail"] == 1).sum() == 104
    assert df["Time"].is_monotonic_increasing
    assert df["Time"].min() >= pd.Timestamp("2008-07-01") and df["Time"].max() <= pd.Timestamp("2008-10-31")


def test_preprocessor_drops_bad_sensors_and_uses_train_stats_only():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    X = pd.DataFrame({1: a, 2: a * 2 + 1e-6 * rng.normal(size=200), 3: 5.0,
                      4: np.where(rng.random(200) < 0.6, np.nan, 1.0) * rng.normal(size=200),
                      5: rng.normal(size=200)})
    pre = SensorPreprocessor().fit(X.iloc[:100])
    assert pre.columns_ == [1, 5]                     # duplicate, constant and mostly-missing dropped
    med_before = pre.median_.copy()
    pre.transform(X.iloc[100:] + 100)                 # live data must not change fitted stats
    assert pre.median_.equals(med_before)
    assert pre.transform(X).abs().to_numpy().max() <= 10


def test_rolling_risk_ranks_against_recent_runs():
    r = rolling_risk([0.9, 0.0, 0.5], history=np.linspace(0.1, 0.8, 50))
    assert r[0] == 100.0 and r[1] == 0.0 and 40 < r[2] < 70
    assert tier(95) == "High" and tier(75) == "Elevated" and tier(10) == "Normal"


def _simulate(effect, n=3000, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    t = (rng.random(n) < 1 / (1 + np.exp(-(x[:, 0] - 1.2)))).astype(float)
    y = (rng.random(n) < np.clip(0.05 + 0.06 * (x[:, 0] > 0.5) + effect * t, 0, 1)).astype(float)
    return y, t, x


def test_stratified_effect_recovers_a_real_effect_through_confounding():
    y, t, x = _simulate(effect=0.15)
    est = stratified_effect(y, t, propensity(t.astype(int), x), np.random.default_rng(0))
    assert est["p"] < 0.01 and 0.07 < est["ate"] < 0.23


def test_stratified_effect_is_quiet_when_only_confounding_exists():
    hits = 0
    for seed in range(8):
        y, t, x = _simulate(effect=0.0, seed=seed)
        assert y[t == 1].mean() > y[t == 0].mean()      # the raw comparison is fooled
        est = stratified_effect(y, t, propensity(t.astype(int), x), np.random.default_rng(seed))
        hits += est["p"] < 0.05
    assert hits <= 2
