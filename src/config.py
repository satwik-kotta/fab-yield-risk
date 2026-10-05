"""Shared settings. Everything is overridable from the environment."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "uci-secom.csv"
ARTIFACTS = ROOT / "artifacts"
DOCS = ROOT / "docs"
SQL_DIR = ROOT / "sql"

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/fab"
)

TRAIN_FRACTION = 0.70      # earliest 70% of runs train the model, the rest replay as "live"
MAX_MISSING = 0.40         # drop sensors missing in more than 40% of training runs
MAX_CORR = 0.95            # drop one of each pair of near-duplicate sensors
RISK_WINDOW = 200          # a run is ranked against the previous 200 scored runs
HIGH_TIER = 90.0           # risk score >= 90 -> High
ELEVATED_TIER = 70.0       # risk score >= 70 -> Elevated
SEED = 7
