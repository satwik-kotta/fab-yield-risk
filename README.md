# Fab Yield Risk

Scores every wafer run on a semiconductor line for scrap risk from its sensor readings, and separates the sensor excursions that plausibly move yield from the ones that only correlate with it.

**Live dashboard:** https://satwik-kotta.github.io/fab-yield-risk/

Built on the public [UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom) dataset: 1,567 wafer runs, 590 sensors per run, 104 inspection fails (6.6%).

## What it found

| Question | Answer |
|---|---|
| Can sensor data rank runs by scrap risk before inspection? | Weakly. ROC AUC 0.64 on held-out later runs (95% CI 0.54 to 0.74). |
| Is the High tier worth reviewing first? | High-tier runs failed at 11.1%, twice the 5.5% line average. High + Elevated caught 16 of 26 fails while flagging 40% of runs. |
| Which sensors does a correlation screen flag? | 16 of 183 screened sensors (10% false-discovery rate). |
| Which hold up after adjusting for drift and the rest of the line? | 1 of the 16: sensor 348, +13 points on fail rate, with a wide interval (95% CI 1 to 28 points). |
| Does the method invent findings? | No. Five placebo runs with shuffled outcomes flagged 0 sensors. |

The headline is the gap between 16 and 1. Excursions on this line arrive together, so most "top sensors" from a correlation or feature-importance chart cannot be told apart from their neighbours. Sending an engineer to all 16 would mostly be wasted effort; the data supports one lead, and only tentatively.

## Architecture

```
uci-secom.csv
     │  stream_simulator.py   replays runs in time order (backfill or paced "live" mode)
     ▼
PostgreSQL ── wafer_runs · sensor_readings (924,530 rows, long format) · inspection_results
     │
     ├─ model.py     train on the earliest 70% of runs, test on the latest 30%
     ├─ score.py     polls v_unscored_runs, writes risk_scores (can run as a watcher)
     ├─ causal.py    per-sensor excursion effects, naive vs adjusted, with placebo
     │
     ▼
SQL views (v_daily_yield, v_tier_performance, v_scored_runs, v_sensor_health)
     │  export_dashboard.py
     ▼
docs/  static dashboard (ECharts), served by GitHub Pages
```

## Method

**Ingestion.** Each run lands as a header row, 590 sensor readings and an inspection result, the shape a fab historian produces. `--mode live` paces inserts by the gaps between real timestamps; `score.py --watch` picks up new runs as they arrive.

**Data fix.** This copy of SECOM has day and month swapped on every date with day ≤ 12, which scatters three months of runs across a whole year. `fix_timestamp` undoes it; afterwards the file is strictly ordered from 19 Jul to 17 Oct 2008. A time-based split is meaningless without this.

**Features.** Fitted on training runs only: drop sensors missing in over 40% of runs (28), constant sensors (122), and one of each pair correlated above 0.95 (169). 271 of 590 remain. Median fill, robust scaling.

**Risk model.** Five candidates compared with expanding-window cross-validation inside the training period; a random forest was selected (CV AUC 0.68). It was then tested once on the final 471 runs.

**Risk score.** A run's score is its rank against the 200 runs before it. The first version ranked against the training period and failed: the line drifts, scores shifted down, and a fixed cut-off flagged 22 runs and caught none of the 26 fails. Ranking against recent runs needs no labels, so it works in real time. This change was made after seeing test-period behaviour, so the tier numbers above are slightly optimistic.

**Causal analysis.** For each sensor, the treatment is "this run had an excursion" (more than 2.5 robust σ from the median). Controls are time drift and 15 principal components of the sensors not closely tied to the one under test. Estimate: propensity-stratified difference in fail rate on the common-support region. P-values come from permuting the excursion flag within strata, then Benjamini-Hochberg across sensors.

The first version used AIPW with large-sample standard errors. It flagged 18 sensors on the real data and 17 on shuffled outcomes, so it was discarded: with rare fails and excursion groups containing zero fails, those standard errors collapse. The permutation version flags 0 on five placebos. `tests/` also checks it on simulated data with a known effect and with pure confounding.

## Limits

- 104 fails in total and 26 in the test period. Every number here has a wide interval.
- Sensors are anonymised and their order in the process is unknown. A control could sit downstream of the sensor being tested, which would bias its estimate. "Causal" here means "survives adjustment for everything else measured", not a proven mechanism.
- 36 of 183 sensors had too little overlap to estimate at all: their excursions almost never happen alone.
- One fab, three months, 2008.

## Run it

```bash
pip install -r requirements.txt
docker compose up -d --wait        # PostgreSQL 16 on localhost:5432
make all                           # load → train → score → causal → dashboard (about 8 minutes)
make test
open docs/index.html
```

Set `DATABASE_URL` to use an existing PostgreSQL instance. Results are seeded and reproduce exactly.

Live replay in two terminals, after `make train`:

```bash
python -m src.score --watch 2
python -m src.stream_simulator --mode live --speedup 5000
```

## Layout

```
sql/        schema and views
src/        stream_simulator, features, model, score, causal, export_dashboard
tests/      data checks and simulated-data checks of the causal estimator
artifacts/  metrics.json, causal_effects.csv, causal_summary.json
docs/       dashboard
data/       SECOM CSV and its licence note (CC BY 4.0)
```
