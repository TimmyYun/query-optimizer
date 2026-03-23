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

def process_table(name, csv_file, sql_file):
    print(f"Processing {name}...")
    out_dir = Path("data/dsb") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Read sql and create workload csv
    queries = []
    with open(sql_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # format: where ws_item_sk between 4236 and 4236 + 400;
            m = re.search(r"between\s+(\d+)\s+and\s+(\d+)\s*\+\s*(\d+)", line, re.IGNORECASE)
            if m:
                low = int(m.group(1))
                # High is low + width OR parsed as literal?
                # Sometimes it might just be 'between X and Y;'
                high = int(m.group(2)) + int(m.group(3))
                queries.append((low, high))
            else:
                m2 = re.search(r"between\s+(\d+)\s+and\s+(\d+)", line, re.IGNORECASE)
                if m2:
                    queries.append((int(m2.group(1)), int(m2.group(2))))

    assert len(queries) > 0, f"No queries found in {sql_file}"
    count = len(queries)
    workload_path = out_dir / f"workload_driven_{count}.csv"
    with open(workload_path, "w") as f:
        f.write("low,high\n")
        for q in queries:
            f.write(f"{q[0]},{q[1]}\n")
            
    print(f"Saved {count} workload queries to {workload_path}")
    
    # 2. Process data.csv to create meta.pkl
    data_path = out_dir / "data.csv"
    # To use existing functions like scan_min_max_count without header issues, 
    # we should copy the data to expected format if needed. 
    # Actually `scan_min_max_count` expects no header. Let's read the csv and write it out without header just in case.
    df = pd.read_csv(csv_file)
    col_name = df.columns[0]
    df[col_name].to_csv(data_path, index=False, header=False)
    
    mn, mx, N = scan_min_max_count(data_path)
    ndv = calculate_ndv(data_path)
    skew, kurt = calculate_skew_kurt(data_path)
    
    TARGET_SAMPLE_SIZE = 100_000
    freq, sample = build_frequency_and_sample(data_path, mn, mx, N, TARGET_SAMPLE_SIZE, 42)
    k = freedman_diaconis_bins(sample, mn, mx, N)
    
    meta_path = out_dir / "meta.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump((mn, mx, N, freq, sample, k, skew, kurt), f)
        
    print(f"Created meta.pkl for {name}. N={N}, mn={mn}, mx={mx}, k={k}")

process_table("ws_item_sk", "data/dsb/ws_item_sk.csv", "data/dsb/custom_ws_item_sk.sql")
process_table("cr_item_sk", "data/dsb/cr_item_sk.csv", "data/dsb/custom_cr_item_sk.sql")
process_table("cr_returned_time_sk", "data/dsb/cr_returned_time_sk.csv", "data/dsb/custom_cr_returned_time_sk.sql")

print("Done processing DSB dataset.")
