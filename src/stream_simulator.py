"""Replay the SECOM line into PostgreSQL in timestamp order.

Each wafer run is written the way a fab historian would see it: the run
header, then ~590 sensor readings, then (later) the inspection result.

    python -m src.stream_simulator --mode backfill            # load everything now
    python -m src.stream_simulator --mode live --speedup 5000 # replay with pacing
"""
import argparse
import io
import time

import numpy as np
import pandas as pd

from . import config
from .db import get_engine, init_db


def fix_timestamp(ts):
    """Undo the day/month swap in this copy of SECOM.

    The source timestamps are dd/mm/yyyy. This CSV was exported after a
    month-first parse, so every date whose day is <= 12 has day and month
    swapped (1 Aug shows as 8 Jan). Dates with day > 12 were parsed correctly.
    After the fix the file is strictly ordered: 19 Jul 2008 -> 17 Oct 2008.
    """
    return ts.replace(month=ts.day, day=ts.month) if ts.day <= 12 else ts


def load_raw():
    df = pd.read_csv(config.RAW_CSV)
    df["Time"] = pd.to_datetime(df["Time"]).map(fix_timestamp)
    df = df.sort_values("Time", kind="mergesort").reset_index(drop=True)
    df = pd.concat([pd.Series(np.arange(1, len(df) + 1), name="run_id"), df], axis=1)
    return df


def _copy(cur, table, columns, frame):
    buf = io.StringIO()
    frame.to_csv(buf, index=False, header=False, na_rep="\\N")
    buf.seek(0)
    cur.copy_expert(
        f"COPY {table} ({', '.join(columns)}) FROM STDIN WITH (FORMAT csv, NULL '\\N')", buf)


def write_batch(engine, batch, sensor_cols):
    runs = batch[["run_id", "Time"]]
    long = batch.melt(id_vars="run_id", value_vars=sensor_cols,
                      var_name="sensor_id", value_name="value")
    long["sensor_id"] = long["sensor_id"].astype(int)
    labels = batch[["run_id"]].assign(failed=(batch["Pass_Fail"] == 1).map({True: "t", False: "f"}))
    raw = engine.raw_connection()
    try:
        cur = raw.cursor()
        _copy(cur, "wafer_runs", ["run_id", "run_ts"], runs)
        _copy(cur, "sensor_readings", ["run_id", "sensor_id", "value"], long)
        _copy(cur, "inspection_results", ["run_id", "failed"], labels)
        raw.commit()
    finally:
        raw.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["backfill", "live"], default="backfill")
    ap.add_argument("--speedup", type=float, default=5000.0,
                    help="live mode: real seconds between runs are divided by this")
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-reset", action="store_true")
    args = ap.parse_args()

    engine = get_engine()
    init_db(engine, reset=not args.no_reset)
    df = load_raw()
    if args.limit:
        df = df.head(args.limit)
    sensor_cols = [c for c in df.columns if c.isdigit()]

    t0 = time.time()
    if args.mode == "backfill":
        for start in range(0, len(df), args.batch):
            write_batch(engine, df.iloc[start:start + args.batch], sensor_cols)
    else:
        prev = None
        for i in range(len(df)):
            row = df.iloc[i:i + 1]
            ts = row["Time"].iloc[0]
            if prev is not None:
                time.sleep(min((ts - prev).total_seconds() / args.speedup, 2.0))
            prev = ts
            write_batch(engine, row, sensor_cols)
            if (i + 1) % 100 == 0:
                print(f"streamed {i + 1} runs (line time {ts})", flush=True)
    print(f"loaded {len(df)} runs x {len(sensor_cols)} sensors "
          f"= {len(df) * len(sensor_cols):,} readings in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
