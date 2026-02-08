import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import shutil
import pickle
from pathlib import Path
from typing import Tuple, List

"""
Datasets Module
===============

This module handles the generation, storage, and statistical analysis of datasets used for benchmarking
query optimizer cardinality estimation strategies.

Key Responsibilities:
1. Data Generation: Generates synthetic data (Uniform, Normal, Zipf, etc.) or loads real-world data (IMDB, Census).
2. Statistical Analysis: Computes metadata like Min, Max, Count, NDV, Skewness, Kurtosis.
3. Visualization: Generates histograms and boxplots for data distributions and error metrics.
4. Dataset Management: Manages directory structures and file persistence/caching.
"""

# ==========================================
# Data Utils
# ==========================================

def clamp_int(x, lo, hi):
    """Clamps an integer x between lo and hi (inclusive)."""
    return int(min(max(int(round(x)), lo), hi))

def gen_values(rng: np.random.Generator, dist: str, n: int, lo: int, hi: int, shift: int = 0) -> np.ndarray:
    """
    Generates an array of integer values according to a specified distribution.

    Args:
        rng (np.random.Generator): Random number generator instance.
        dist (str): Distribution type ("uniform", "normal", "zipf", "sparse_cluster", "anti_zipf").
        n (int): Number of values to generate.
        lo (int): Lower bound of the value domain.
        hi (int): Upper bound of the value domain.
        shift (int): Shift applied to the distribution (used for drift simulation).

    Returns:
        np.ndarray: Array of generated integer values.
    """
    mid = 0.5 * (lo + hi) + shift
    span = max(1, hi - lo)

    if dist == "uniform":
        v = rng.integers(lo + shift, hi + 1 + shift, size=n)
    elif dist == "normal":
        v = rng.normal(loc=mid, scale=span / 6.0, size=n)
    elif dist == "zipf":
        v = lo + shift + rng.zipf(2.0, size=n)
    elif dist == "sparse_cluster":
        # Create 10 dense clusters
        centers = rng.integers(lo, hi, size=10) + shift
        v = []
        for c in centers:
            c_lo = max(lo + shift, c - 50)
            c_hi = min(hi + shift + 200_000, c + 50)
            if c_lo < c_hi:
                v.append(rng.integers(c_lo, c_hi, size=n // 10))
            else:
                v.append(rng.integers(lo+shift, hi+shift+1, size=n//10))
        v = np.concatenate(v)
    elif dist == "anti_zipf":
        # Uniform distribution over a small subset of the domain
        v = rng.integers(lo + shift, lo + shift + 20000, size=n)
    else:
        # Default fallback to uniform
        v = rng.integers(lo, hi + 1, size=n)
        
    if n == 0: return np.array([], dtype=np.int64)
    
    # Ensure all values are strictly within the global domain limits [0, 200_000]
    v = np.vectorize(lambda x: clamp_int(x, 0, 200_000))(v)
    return v.astype(np.int64)

def save_csv_column(values: np.ndarray, path: Path, mode='w'):
    """
    Saves a numpy array as a single-column CSV file without a header.
    
    Args:
        values (np.ndarray): Data to save.
        path (Path): Destination file path.
        mode (str): File open mode ('w' for write/overwrite, 'a' for append).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.Series(values)
    df.to_csv(path, index=False, header=False, mode=mode)

def scan_min_max_count(csv_path: Path, chunksize: int = 1_000_000) -> Tuple[int, int, int]:
    """
    Efficiently scans a large CSV file to find the minimum value, maximum value, and total count.
    
    Args:
        csv_path (Path): Path to the CSV file.
        chunksize (int): Number of rows to process at a time.

    Returns:
        Tuple[int, int, int]: (min_val, max_val, count)
    """
    mn, mx, n = None, None, 0
    for ch in pd.read_csv(csv_path, header=None, names=["v"], dtype="int64", chunksize=chunksize, engine="c"):
        v = ch["v"].to_numpy()
        n += v.size
        if v.size > 0:
            cmin, cmax = int(v.min()), int(v.max())
            mn = cmin if mn is None else min(mn, cmin)
            mx = cmax if mx is None else max(mx, cmax)
    if mn is None: return 0, 0, 0 
    return mn, mx, n

def build_frequency_and_sample(csv_path: Path, mn: int, mx: int, n_rows: int, sample_size: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scans the dataset to build an exact frequency histogram and collect a reservoir sample.

    Args:
        csv_path (Path): Path to the CSV dataset.
        mn (int): Minimum value in the dataset (for offset calculation).
        mx (int): Maximum value in the dataset.
        n_rows (int): Total expected number of rows (used for sampling probability).
        sample_size (int): Target size for the reservoir sample.
        seed (int): Random seed for sampling.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - freq: Exact frequency counts for each value in the range [mn, mx].
            - sample: A random sample of values from the dataset.
    """
    width = mx - mn + 1
    if width <= 0: return np.array([]), np.array([])
    freq = np.zeros(width, dtype=np.int64)
    rng = np.random.default_rng(seed)
    sampled = []
    p = min(1.0, float(sample_size * 2) / float(max(n_rows, 1)))

    for ch in pd.read_csv(csv_path, header=None, names=["v"], dtype="int64", chunksize=1_000_000, engine="c"):
        vals = ch["v"].to_numpy()
        idx = vals - mn
        
        # Update exact frequencies
        m = (idx >= 0) & (idx < width)
        valid_idx = idx[m]
        if valid_idx.size:
            freq += np.bincount(valid_idx, minlength=width)
            
        # Reservoir sampling step
        if p > 0 and len(sampled) < sample_size:
            mask = rng.random(vals.size) < p
            s = vals[mask]
            if s.size > 0: sampled.append(s)

    if sampled:
        sample = np.concatenate(sampled)
        if sample.size > sample_size:
            sample = rng.choice(sample, size=sample_size, replace=False)
    else:
        sample = np.array([], dtype=np.int64)
    return freq, sample

def load_imdb_lengths(csv_path: Path) -> np.ndarray:
    """Loads review lengths from the IMDB dataset."""
    try:
        df = pd.read_csv(csv_path)
        col = "review" if "review" in df.columns else df.columns[0]
        vals = df[col].astype(str).str.len().to_numpy()
        return vals.astype(np.int64)
    except Exception as e:
        print(f"Error loading IMDB: {e}")
        return np.array([], dtype=np.int64)

def load_census_age(csv_path: Path) -> np.ndarray:
    """Loads age data ('dAge') from the US Census dataset."""
    try:
        df = pd.read_csv(csv_path, usecols=['dAge'])
        return df['dAge'].to_numpy().astype(np.int64)
    except Exception as e:
        print(f"Error loading Census: {e}")
        return np.array([], dtype=np.int64)

# ==========================================
# Stats Utils
# ==========================================

# freedman_diaconis_bins REMOVED

def calculate_skew_kurt(csv_path: Path) -> Tuple[float, float]:
    """Calculates skewness and kurtosis of the dataset."""
    try:
        # Note: Reading simple CSV for skew/kurt. For huge files, this might be slow.
        df = pd.read_csv(csv_path, header=None, names=["v"], dtype="int64")
        return float(df["v"].skew()), float(df["v"].kurt())
    except Exception as e:
        print(f"Error calculating skew/kurt: {e}")
        return 0.0, 0.0

def calculate_ndv(csv_path: Path):
    """Calculates the Number of Distinct Values (NDV) in the dataset."""
    seen = np.zeros(200_001, dtype=bool)
    try:
        for chunk in pd.read_csv(csv_path, header=None, names=["v"], dtype="int64", chunksize=1_000_000, engine="c"):
            vals = chunk["v"].to_numpy()
            vals = vals[(vals >= 0) & (vals <= 200_000)]
            seen[vals] = True
        return np.count_nonzero(seen)
    except Exception as e:
        print(f"Error calculating NDV for {csv_path}: {e}")
        return 0

def save_stats(stats: dict, output_path: Path):
    """Saves a dictionary of statistics to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=4)

# ==========================================
# Plot Utils
# ==========================================

def plot_data_distribution(vals, dist_name, output_path):
    """
    Plots a histogram of the data distribution.
    
    Args:
        vals (np.ndarray): The full dataset or a large sample.
        dist_name (str): Name of the distribution (e.g., "Zipf").
        output_path (Path): Path to save the PNG plot.
    """
    print(f"Plotting distribution for {dist_name} to {output_path}...")
    plot_vals = np.random.choice(vals, min(len(vals), 1_000_000), replace=False)
    stats_text = (f"Count (N): {len(vals)}\nMin: {np.min(vals)}\nMax: {np.max(vals)}\n"
                  f"Mean: {np.mean(vals):.2f}\nStd Dev: {np.std(vals):.2f}")

    plt.figure(figsize=(10, 6))
    use_log = (dist_name.lower() == 'zipf')
    plt.hist(plot_vals, bins=100, color='skyblue', edgecolor='black', alpha=0.7, log=use_log)
    plt.title(f"Distribution: {dist_name}")
    plt.xlabel("Value")
    plt.ylabel("Frequency" + (" (Log Scale)" if use_log else ""))
    plt.grid(axis='y', alpha=0.3)
    if dist_name.lower() not in ['imdb', 'census']: plt.xlim(0, 200_000)
    plt.gcf().text(0.78, 0.6, stats_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

def generate_boxplots(csv_path, output_path, title="Q-Error Distribution"):
    """
    Generates faceted boxplots for Q-Error distributions across multiple models and data distributions.
    Expects a DataFrame with columns: 'Model', 'QErr', 'Distribution'.
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    if df.empty: return

    plt.figure(figsize=(12, 6))
    g = sns.catplot(data=df, x="Model", y="QErr", col="Distribution", kind="box", 
                    showfliers=False, palette="Set2", sharey=False, col_wrap=3, height=4, aspect=1.2)
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(title)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Faceted boxplot saved to {output_path}")

def plot_model_comparison(csv_path: str, output_path: str, title: str):
    """
    Generates a single boxplot comparing models for a specific experiment scenario.
    """
    df = pd.read_csv(csv_path)
    if df.empty: return
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Model", y="QErr", showfliers=False, palette="Set2")
    plt.title(title)
    plt.ylabel("Q-Error")
    plt.yscale("log")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Single distribution boxplot saved to {output_path}")


# ==========================================
# Dataset Manager
# ==========================================

class DatasetManager:
    """
    Manages the lifecycle of datasets, including generation, caching, and retrieval.
    """
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)

    def get_dataset_dir(self, rows: int, dist: str) -> Path:
        """Returns the structured directory path for a specific dataset configuration."""
        return self.base_path / "generated" / str(rows) / dist

    def prepare_dataset(self, rows: int, dist: str, force_regeneration: bool = False):
        """
        Orchestrates the creation of a dataset.
        
        Steps:
        1. Checks if dataset exists. If so, returns early (unless force_regeneration=True).
        2. Generates values based on distribution.
        3. Saves data to CSV.
        4. Calculates statistics (Min, Max, Skew, Kurtosis, NDV).
        5. Saves statistics to JSON.
        6. Saves a pickle metadata file for fast loading.
        7. Generates a distribution plot.

        Args:
            rows (int): Number of rows to generate.
            dist (str): Distribution type.
            force_regeneration (bool): If True, overwrites existing data.

        Returns:
            Path: The directory containing the generated dataset and metadata.
        """
        ds_dir = self.get_dataset_dir(rows, dist)
        ds_dir.mkdir(parents=True, exist_ok=True)
        
        data_path = ds_dir / "data.csv"
        stats_path = ds_dir / "stats.json"
        
        if not data_path.exists() or force_regeneration:
            print(f"Generating dataset: {rows} rows, {dist}...")
            rng = np.random.default_rng(42)
            if dist.lower() == "imdb":
                vals = load_imdb_lengths(Path("data/imdb/IMDB Dataset.csv"))
            elif dist.lower() == "census":
                vals = load_census_age(Path("data/census/USCensus1990.data.txt.csv"))
            else:
                vals = gen_values(rng, dist, rows, 0, 200_000)
            
            save_csv_column(vals, data_path)
            
            # Compute Stats
            mn, mx, N = scan_min_max_count(data_path)
            ndv = calculate_ndv(data_path)
            skew, kurt = calculate_skew_kurt(data_path)
            
            # For FD bins - REMOVED (User request: bins is a hyperparam now)
            # freq, sample = build_frequency_and_sample(data_path, mn, mx, N, 100_000, 42)
            # k = freedman_diaconis_bins(sample, mn, mx, N)
            
            # We still need freq for some stats perhaps? No, freq is expensive to build here.
            # If we remove freq build, we save time.
            
            # Wait, meta.pkl requires freq and sample.
            # "freq" and "sample" are used in main.py for Hybrid estimator?
            # Yes, HybridEstimator uses valid buckets built from "freq".
            # So we MUST build freq.
            freq, sample = build_frequency_and_sample(data_path, mn, mx, N, 100_000, 42)
            
            # k is now irrelevant, set to 0 or None
            k = 0 
            
            stats = {
                "Rows": int(N),
                "Min": int(mn),
                "Max": int(mx),
                "NDV": int(ndv),
                "Skewness": float(skew),
                "Kurtosis": float(kurt),
                "Bin Count (k)": int(k),
                "Bin Width (h)": 0.0
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

# ==========================================
# Main Execution (Generation)
# ==========================================
import time

def main():
    """
    Batch generation entry point.
    Generates datasets for predefined distributions (uniform, normal, zipf, etc.)
    and row counts (1M, 10M, 60M).
    """
    rows_list = [1_000_000, 10_000_000, 60_000_000]
    distributions = ["uniform", "normal", "zipf", "sparse_cluster", "anti_zipf"]
    
    dm = DatasetManager()
    
    total_start = time.time()
    
    for rows in rows_list:
        for dist in distributions:
            print(f"\n>>> Generating Dataset: {rows} rows, {dist} <<<")
            start = time.time()
            try:
                # 1. Generate Data
                ds_dir = dm.prepare_dataset(rows, dist)
                
            except Exception as e:
                print(f"Failed to generate {rows} / {dist}: {e}")
                import traceback
                traceback.print_exc()
            
            elapsed = time.time() - start
            print(f"Done in {elapsed:.2f}s")

    total_elapsed = time.time() - total_start
    print(f"\nAll operations completed in {total_elapsed:.2f}s")

if __name__ == "__main__":
    main()
