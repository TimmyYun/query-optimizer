from datasets import plot_data_distribution
import json
import pandas as pd
from pathlib import Path
import numpy as np

def refresh_60m_datasets():
    rows = 60000000
    dataset_root = Path("data/generated") / str(rows)
    distributions = ["uniform", "normal", "zipf", "sparse_cluster", "anti_zipf"]

    for dist in distributions:
        ds_dir = dataset_root / dist
        stats_path = ds_dir / "stats.json"
        data_path = ds_dir / "data.csv"

        if not stats_path.exists() or not data_path.exists():
            print(f"Skipping {dist} - missing stats or data")
            continue

        print(f"Refreshing {dist}...")
        with open(stats_path, "r") as f:
            stats = json.load(f)
        
        k = stats["Bin Count (k)"]
        
        # Load the data to re-plot
        # Since it's 60M rows, we follow the same logic as datasets.py: random sample of 1M
        vals = pd.read_csv(data_path, header=None, names=["v"], dtype="int64", chunksize=1_000_000, engine="c")
        all_vals = []
        for chunk in vals:
             all_vals.append(chunk["v"].to_numpy())
        
        vals_array = np.concatenate(all_vals)
        
        plot_data_distribution(vals_array, dist, ds_dir / "hist.png", n_bins=k)
        print(f"Refreshed {dist} with k={k}")

if __name__ == "__main__":
    refresh_60m_datasets()
