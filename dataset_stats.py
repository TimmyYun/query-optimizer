
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import scan_min_max_count, build_frequency_and_sample, freedman_diaconis_bins

def calculate_stats():
    distributions = ['uniform', 'normal', 'zipf', 'sparse_cluster', 'anti_zipf']
    cache_dir = Path("datasets/files/generated")
    
    stats_list = []
    
    for dist in distributions:
        print(f"Processing {dist}...")
        ds_path = cache_dir / f"bench_{dist}.csv"
        
        if not ds_path.exists():
            print(f"Warning: {ds_path} not found. Skipping.")
            continue
            
        # 1. Load Data for Pandas Stats
        # Use chunking if too large, but 1M ints is small enough for pandas (~8MB)
        df = pd.read_csv(ds_path, header=None, names=['v'])
        vals = df['v']
        
        # Calculate Pandas stats (Fisher's definitions by default)
        skew = vals.skew()
        kurt = vals.kurt()
        
        # 2. Dataset Metadata for Bins
        mn, mx, N = scan_min_max_count(ds_path)
        freq, sample = build_frequency_and_sample(ds_path, mn, mx, N, 100_000, 42)
        
        # 3. Bin Metrics (Freedman-Diaconis)
        k = freedman_diaconis_bins(sample, mn, mx, N, 2000)
        h = (mx - mn) / k if k > 0 else 0
        
        stats_list.append({
            "Distribution": dist,
            "Skewness": skew,
            "Kurtosis": kurt,
            "Bin Count (k)": k,
            "Bin Width (h)": h,
            "Min": mn,
            "Max": mx,
            "Count (N)": N
        })
        
    stats_df = pd.DataFrame(stats_list)
    print("\nDataset Statistics:")
    print(stats_df)
    
    out_file = "datasets/dataset_statistics.xlsx"
    stats_df.to_excel(out_file, index=False)
    print(f"\nSaved stats to {out_file}")

if __name__ == "__main__":
    calculate_stats()
