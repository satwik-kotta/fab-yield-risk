"""Database helpers."""
from sqlalchemy import create_engine, text

from . import config


def get_engine():
    return create_engine(config.DATABASE_URL, future=True)


def run_sql_file(engine, path):
    with engine.begin() as conn:
        conn.execute(text(path.read_text()))


def init_db(engine, reset=False):
    if reset:
        with engine.begin() as conn:
            conn.execute(text(
                "DROP TABLE IF EXISTS risk_scores, inspection_results, "
                "sensor_readings, wafer_runs CASCADE"))
    run_sql_file(engine, config.SQL_DIR / "schema.sql")
    run_sql_file(engine, config.SQL_DIR / "views.sql")
