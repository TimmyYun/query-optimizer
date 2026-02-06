import numpy as np
import pandas as pd
import json
import shutil
import pickle
from pathlib import Path
from .data_utils import (
    gen_values, save_csv_column, scan_min_max_count, 
    build_frequency_and_sample, load_imdb_lengths, load_census_age
)
from .stats_utils import (
    freedman_diaconis_bins, calculate_skew_kurt, calculate_ndv, save_stats
)
from .workload_utils import generate_workload, save_workload, load_workload, RangeQuery
from .plot_utils import plot_data_distribution

class DatasetManager:
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)

    def get_dataset_dir(self, rows: int, dist: str) -> Path:
        return self.base_path / "generated" / str(rows) / dist

    def prepare_dataset(self, rows: int, dist: str, force_regeneration: bool = False):
        ds_dir = self.get_dataset_dir(rows, dist)
        ds_dir.mkdir(parents=True, exist_ok=True)
        
        data_path = ds_dir / "data.csv"
        stats_path = ds_dir / "stats.json"
        
        if not data_path.exists() or force_regeneration:
            print(f"Generating dataset: {rows} rows, {dist}...")
            rng = np.random.default_rng(42)
            if dist.lower() == "imdb":
                vals = load_imdb_lengths(Path("datasets/files/imdb/IMDB Dataset.csv"))
            elif dist.lower() == "census":
                vals = load_census_age(Path("datasets/files/census/USCensus1990.data.txt.csv"))
            else:
                vals = gen_values(rng, dist, rows, 0, 200_000)
            
            save_csv_column(vals, data_path)
            
            # Compute Stats
            mn, mx, N = scan_min_max_count(data_path)
            ndv = calculate_ndv(data_path)
            skew, kurt = calculate_skew_kurt(data_path)
            
            # For FD bins, we need a frequency map and sample
            freq, sample = build_frequency_and_sample(data_path, mn, mx, N, 100_000, 42)
            k = freedman_diaconis_bins(sample, mn, mx, N)
            
            stats = {
                "Rows": int(N),
                "Min": int(mn),
                "Max": int(mx),
                "NDV": int(ndv),
                "Skewness": float(skew),
                "Kurtosis": float(kurt),
                "Bin Count (k)": int(k),
                "Bin Width (h)": float((mx - mn) / k if k > 0 else 0)
            }
            save_stats(stats, stats_path)
            
            # Save a binary meta file for fast loading in main.py (legacy compatibility)
            meta_path = ds_dir / "meta.pkl"
            with open(meta_path, "wb") as f:
                pickle.dump((mn, mx, N, freq, sample, k, skew, kurt), f)
                
            # Plot distribution
            plot_data_distribution(vals, dist, ds_dir / "hist.png")
            
        print(f"Dataset ready at {ds_dir}")
        return ds_dir

    def prepare_workload(self, rows: int, dist: str, n_queries: int, force_regeneration: bool = False):
        ds_dir = self.get_dataset_dir(rows, dist)
        workload_path = ds_dir / "workload.json"
        
        if not workload_path.exists() or force_regeneration:
            # Load stats to get domain
            with open(ds_dir / "stats.json", "r") as f:
                stats = json.load(f)
            
            print(f"Generating workload: {n_queries} queries for {dist}...")
            queries = generate_workload(n_queries, stats["Min"], stats["Max"])
            save_workload(queries, workload_path)
            
        return workload_path
