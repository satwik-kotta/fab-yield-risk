# Data

`raw/uci-secom.csv` is the SECOM dataset (McCann & Johnston, 2008), UCI Machine Learning
Repository, https://archive.ics.uci.edu/dataset/179/secom, licensed CC BY 4.0.

1,567 wafer runs, 590 anonymised sensor readings per run, a timestamp and a pass/fail
inspection label (104 fails). This copy is the single-CSV export distributed on Kaggle.
Its timestamps have day and month swapped for days 1-12; `src/stream_simulator.py`
corrects that on load (see `fix_timestamp`).
