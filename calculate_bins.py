import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())

from datasets import gen_values, freedman_diaconis_bins

def calculate_bin_stats(name, dist_key, rows=1_000_000):
    rng = np.random.default_rng(42)
    lo, hi = 0, 200_000
    
    # Generate data
    data = gen_values(rng, dist_key, rows, lo, hi)
    mn, mx = int(data.min()), int(data.max())
    N = len(data)
    
    # Calculate sampling for FD (match main.py logic)
    sample_size = 2000 # as used in main.py
    if N > sample_size:
        sample = rng.choice(data, size=sample_size, replace=False)
    else:
        sample = data
        
    # Calculate FD Bins
    n_bins = freedman_diaconis_bins(sample, mn, mx, N, 2000)
    
    # Calculate Bucket Width (Equi-Width)
    total_range = mx - mn + 1
    width = max(1, int(np.ceil(total_range / n_bins)))
    
    return [name, n_bins, width, mn, mx]

def main():
    rows = 1_000_000
    
    distributions = [
        ("Uniform (Baseline)", "uniform"),
        ("Normal", "normal"),
        ("Zipfian (alpha=2.0)", "zipf"),
        ("Sparse Cluster", "sparse_cluster"),
        ("Anti-Zipf", "anti_zipf")
    ]
    
    results = []
    print(f"Calculating stats for {rows} rows per dataset...\n")
    
    for name, dist_key in distributions:
        stats = calculate_bin_stats(name, dist_key, rows)
        results.append(stats)
        
    # Create DataFrame for clean printing
    df = pd.DataFrame(results, columns=["Dataset", "Bins (FD)", "Bucket Width", "Min Val", "Max Val"])
    
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
