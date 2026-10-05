"""Which sensor excursions move yield, after adjusting for the rest of the line?

For every sensor we define a treatment: "this run had an excursion on sensor j"
(|robust z| above a threshold). Two estimates of its effect on the fail rate:

  naive     difference in fail rates, excursion vs no excursion (what a
            correlation screen or a feature-importance chart reports)
  adjusted  propensity-stratified risk difference on the common-support
            region, controlling for time drift and the state of the rest of
            the line (principal components of the other sensors), with a
            within-stratum permutation p-value

Both get Benjamini-Hochberg correction across all sensors tested. Placebo
runs with shuffled outcomes check the procedure does not manufacture findings.

    python -m src.causal
"""
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.multitest import multipletests

from . import config
from .db import get_engine
from .features import SensorPreprocessor, load_wide

Z_THRESHOLD = 2.5       # excursion = more than 2.5 robust sigma from the sensor's median
MIN_TREATED = 40        # need enough excursion runs to estimate anything
MAX_TREATED_SHARE = 0.30
RELATED_CORR = 0.5      # sensors this correlated with j are left out of j's controls
N_COMPONENTS = 15
N_FOLDS = 5
CLIP = 0.05             # common support: keep runs with propensity in [0.05, 0.95]
MIN_TREATED_SUPPORT = 25
N_STRATA = 5
N_PERM = 5000
N_BOOT = 500
N_PLACEBO = 5
FDR = 0.10


def build_controls(Z, j, time_feats):
    """Time drift + principal components of sensors not closely tied to j."""
    corr = Z.corrwith(Z[j]).abs()
    others = [c for c in Z.columns if c != j and not corr[c] > RELATED_CORR]
    pcs = PCA(n_components=N_COMPONENTS, random_state=config.SEED).fit_transform(Z[others])
    pcs = pcs / pcs.std(axis=0)
    return np.hstack([time_feats, pcs])


def propensity(t, X, seed=config.SEED):
    """Cross-fitted P(excursion | time, state of the rest of the line)."""
    e = np.zeros(len(t))
    for tr, te in StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, t):
        e[te] = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr], t[tr]).predict_proba(X[te])[:, 1]
    return e


def stratified_effect(y, t, e, rng):
    """Propensity-stratified risk difference with a within-stratum permutation test.

    Runs outside the common-support region are trimmed, the rest are cut into
    propensity strata, and excursion vs normal runs are compared inside each
    stratum. The p-value comes from shuffling the excursion flag *within*
    strata, so it stays valid with rare fails and empty cells, where
    large-sample standard errors do not (see README: the first version of this
    analysis used AIPW standard errors and failed its own placebo test).
    """
    keep = (e > CLIP) & (e < 1 - CLIP)
    overlap = float(keep.mean())
    y, t, e = y[keep], t[keep], e[keep]
    if t.sum() < MIN_TREATED_SUPPORT or (1 - t).sum() < MIN_TREATED_SUPPORT:
        return None
    edges = np.unique(np.quantile(e, np.linspace(0, 1, N_STRATA + 1)[1:-1]))
    s = np.searchsorted(edges, e, side="right")
    ok = np.zeros(len(y), bool)
    for k in np.unique(s):
        m = s == k
        if t[m].sum() >= 3 and (1 - t[m]).sum() >= 3:
            ok |= m
    y, t, s = y[ok], t[ok], s[ok]
    if t.sum() < MIN_TREATED_SUPPORT:
        return None
    _, s = np.unique(s, return_inverse=True)
    S = np.eye(s.max() + 1)[s]                       # n x strata indicator
    n_s, nt_s, f_s = S.sum(0), t @ S, y @ S
    nc_s = n_s - nt_s
    w = n_s / n_s.sum()

    def stat(tf):                                    # tf: treated fails per stratum (.. x strata)
        return ((tf / nt_s) - ((f_s - tf) / nc_s)) @ w

    obs = float(stat((t * y) @ S))
    keys = rng.random((N_PERM, len(y))) + s          # sort within strata = permute within strata
    order = np.argsort(keys, axis=1)
    t_sorted = t[np.argsort(s, kind="stable")]       # treated flags grouped by stratum
    t_perm = np.empty((N_PERM, len(y)), dtype=np.float32)
    np.put_along_axis(t_perm, order, np.broadcast_to(t_sorted, t_perm.shape).astype(np.float32), axis=1)
    null = stat((t_perm * y) @ S)
    p = float((1 + (np.abs(null) >= abs(obs) - 1e-12).sum()) / (N_PERM + 1))

    boot = []                                        # stratified bootstrap for the interval
    idx_by = [np.where(s == k)[0] for k in range(S.shape[1])]
    for _ in range(N_BOOT):
        i = np.concatenate([rng.choice(ix, len(ix)) for ix in idx_by])
        tb, yb, Sb = t[i], y[i], S[i]
        ntb = tb @ Sb
        ncb = Sb.sum(0) - ntb
        if (ntb == 0).any() or (ncb == 0).any():
            continue
        tfb = (tb * yb) @ Sb
        boot.append(float(((tfb / ntb) - ((yb @ Sb - tfb) / ncb)) @ w))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"ate": obs, "ci_low": float(lo), "ci_high": float(hi), "p": p, "overlap": overlap,
            "runs_on_support": int(len(y)), "excursions_on_support": int(t.sum()),
            "strata": int(S.shape[1])}


def naive(y, t):
    a, b = y[t == 1], y[t == 0]
    diff = a.mean() - b.mean()
    _, p = stats.fisher_exact([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])
    return diff, p


def analyse(Z, y, time_feats, cache, label="real"):
    rng = np.random.default_rng(config.SEED)
    rows = []
    for j in Z.columns:
        t = (Z[j].abs() > Z_THRESHOLD).astype(int).to_numpy()
        if t.sum() < MIN_TREATED or t.mean() > MAX_TREATED_SHARE:
            continue
        if j not in cache:                           # propensity does not depend on the outcome
            cache[j] = propensity(t, build_controls(Z, j, time_feats))
        d, p_naive = naive(y, t)
        row = {"sensor": int(j), "excursion_runs": int(t.sum()),
               "fail_rate_excursion": float(y[t == 1].mean()),
               "fail_rate_normal": float(y[t == 0].mean()),
               "naive_diff": float(d), "naive_p": float(p_naive)}
        est = stratified_effect(y.astype(float), t.astype(float), cache[j], rng)
        row["identified"] = est is not None
        row.update(est or {})
        rows.append(row)
    out = pd.DataFrame(rows)
    out["naive_q"] = multipletests(out["naive_p"], method="fdr_bh")[1]
    out["naive_flag"] = out["naive_q"] < FDR
    idn = out["identified"]
    out.loc[idn, "q"] = multipletests(out.loc[idn, "p"], method="fdr_bh")[1]
    out["causal_flag"] = (out["q"] < FDR).fillna(False)
    print(f"[{label}] sensors screened={len(out)} with enough overlap={int(idn.sum())} "
          f"naive-significant={int(out.naive_flag.sum())} "
          f"adjusted-significant={int(out.causal_flag.sum())} (BH FDR {FDR:.0%})")
    return out


def main():
    engine = get_engine()
    X, meta = load_wide(engine)
    y = meta["failed"].astype(int).to_numpy()
    Z = SensorPreprocessor().fit_transform(X)

    order = np.arange(len(y)) / len(y)
    hour = meta["run_ts"].dt.hour.to_numpy() / 24.0
    time_feats = np.column_stack([order, order ** 2, order ** 3,
                                  np.sin(2 * np.pi * hour), np.cos(2 * np.pi * hour)])
    time_feats = (time_feats - time_feats.mean(0)) / time_feats.std(0)

    cache = {}
    real = analyse(Z, y, time_feats, cache).sort_values("p")
    rng = np.random.default_rng(config.SEED)
    placebo_hits = []
    for k in range(N_PLACEBO):
        pl = analyse(Z, rng.permutation(y), time_feats, cache, label=f"placebo {k + 1}: shuffled outcomes")
        placebo_hits.append(int(pl.causal_flag.sum()))

    config.ARTIFACTS.mkdir(exist_ok=True)
    real.to_csv(config.ARTIFACTS / "causal_effects.csv", index=False)
    summary = {
        "runs": int(len(y)), "fails": int(y.sum()), "base_fail_rate": float(y.mean()),
        "sensors_after_cleaning": int(Z.shape[1]), "sensors_screened": int(len(real)),
        "sensors_identified": int(real.identified.sum()),
        "z_threshold": Z_THRESHOLD, "fdr": FDR,
        "naive_significant": int(real.naive_flag.sum()),
        "adjusted_significant": int(real.causal_flag.sum()),
        "naive_only": int((real.naive_flag & ~real.causal_flag).sum()),
        "both": int((real.naive_flag & real.causal_flag).sum()),
        "adjusted_only": int((~real.naive_flag & real.causal_flag).sum()),
        "median_overlap": float(real.overlap.median()),
        "placebo_runs": N_PLACEBO, "placebo_adjusted_significant": placebo_hits,
    }
    (config.ARTIFACTS / "causal_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    cols = ["sensor", "excursion_runs", "fail_rate_excursion", "fail_rate_normal",
            "naive_diff", "naive_q", "ate", "ci_low", "ci_high", "p", "q", "overlap",
            "excursions_on_support"]
    print(real[cols].head(12).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
