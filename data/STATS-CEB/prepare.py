import os
import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from data.datasets import (
    scan_min_max_count, calculate_ndv, calculate_skew_kurt,
    build_frequency_and_sample, freedman_diaconis_bins, DOMAIN_MAX
)


def process_table(name, csv_file, workload_file):
    print(f"Processing {name}...")
    out_dir = Path("data/stats-ceb") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read workload (low,high csv)
    wl = pd.read_csv(workload_file, sep=",")
    count = len(wl)
    workload_path = out_dir / f"workload_driven_{count}.csv"
    wl.to_csv(workload_path, index=False, header=True, sep=",")
    print(f"  Saved {count} workload queries to {workload_path}")

    # 2. Process data csv → data.csv (headerless, clean integers)
    data_path = out_dir / "data.csv"
    df = pd.read_csv(csv_file, header=None, names=["value"], dtype=str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna()
    df["value"] = df["value"].astype(int)
    df["value"].to_csv(data_path, index=False, header=False)
    print(f"  Clean data: {len(df):,} rows")

    # 3. Build meta.pkl
    mn, mx, N = scan_min_max_count(data_path)
    ndv = calculate_ndv(data_path)
    skew, kurt = calculate_skew_kurt(data_path)

    TARGET_SAMPLE_SIZE = 100_000
    freq, sample = build_frequency_and_sample(data_path, mn, mx, N, TARGET_SAMPLE_SIZE, 42)
    k = freedman_diaconis_bins(sample, mn, mx, N)

    meta_path = out_dir / "meta.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump((mn, mx, N, freq, sample, k, skew, kurt), f)

    print(f"  meta.pkl: N={N}, mn={mn}, mx={mx}, k={k}, ndv={ndv}")


process_table(
    "PostHistoryLengthText",
    "data/stats-ceb/PostHistoryLengthText.csv",
    "data/stats-ceb/PostHistoryLengthText_workload.csv",  # your buckets.csv renamed
)

print("Done processing STATS-CEB dataset.")