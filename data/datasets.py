import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from pathlib import Path
from typing import Tuple

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

DOMAIN_MAX = 1_000_000

def clamp_int(x, lo, hi):
    """Clamps an integer x between lo and hi (inclusive)."""
    return int(min(max(int(round(x)), lo), hi))


def gen_values(rng: np.random.Generator, dist: str, n: int, lo: int, hi: int, shift: int = 0, use_micro_dist: bool = True) -> np.ndarray:
    """
    Generates an array of integer values according to a specified distribution.
    By default, uses 'True Fractal' generation where the micro-distribution inside each
    1000-width bucket perfectly matches the shape of the global macro-distribution.
    If use_micro_dist is False, uses a usual simple generation approach.
    """
    mid = 0.5 * (lo + hi) + shift
    span = max(1, hi - lo)

    # We use a 1000-width bucket as the "zoom level" for our fractal patterns.
    bucket_size = 1000

    if dist == "uniform":
        if use_micro_dist:
            # Macro: Flat line across all buckets
            macro_bins = rng.integers(0, span // bucket_size, size=n)
            # Micro: Flat line inside the 1000-width bucket
            micro_noise = rng.integers(0, bucket_size, size=n)
            v = lo + shift + (macro_bins * bucket_size) + micro_noise
        else:
            v = rng.integers(lo + shift, hi + shift + 1, size=n)

    elif dist == "normal":
        if use_micro_dist:
            # Macro: Global bell curve
            num_chunks = span // bucket_size
            global_norm = rng.normal(loc=num_chunks / 2.0, scale=num_chunks / 6.0, size=n)
            macro_bins = np.clip(np.round(global_norm), 0, num_chunks - 1).astype(np.int64)

            # Micro: A perfect mini bell-curve centered exactly in the middle of the bucket (500)
            micro_noise = rng.normal(loc=bucket_size / 2.0, scale=bucket_size / 6.0, size=n)
            micro_noise = np.clip(np.round(micro_noise), 0, bucket_size - 1)

            v = lo + shift + (macro_bins * bucket_size) + micro_noise
        else:
            v_float = rng.normal(loc=mid, scale=span / 6.0, size=n)
            v = np.clip(np.round(v_float), lo + shift, hi + shift)

    elif dist == "zipf":
        ndv_target = min(300_000, span)
        if use_micro_dist:
            # Macro: Global Zipfian cliff/tail
            macro_ranks = np.arange(1, ndv_target + 1)
            macro_probs = 1.0 / (macro_ranks ** 1.0)
            macro_probs /= macro_probs.sum()
            chosen_macro_indices = rng.choice(ndv_target, size=n, p=macro_probs)
            macro_bins = chosen_macro_indices // bucket_size

            # Micro: A mini Zipf cliff inside EVERY bucket
            # We use a standard skew (1.5) so it looks like a clean Zipf curve inside the bucket
            micro_ranks = np.arange(1, bucket_size + 1)
            micro_probs = 1.0 / (micro_ranks ** 1.5)
            micro_probs /= micro_probs.sum()
            micro_noise = rng.choice(bucket_size, size=n, p=micro_probs)

            v = lo + shift + (macro_bins * bucket_size) + micro_noise
        else:
            # 1. Generate ranks according to Zipf power law
            ranks = np.arange(1, ndv_target + 1)
            probs = 1.0 / (ranks ** 1.5)
            probs /= probs.sum()

            # 2. Pick the 'base' values (macro-scale Zipf)
            base_vals = rng.choice(ndv_target, size=n, p=probs)

            # 3. Add a uniform random offset to each value to "smear" the micro-dist
            # This ensures that vals % 1000 is uniform across the [0, 999] range.
            offsets = rng.integers(0, 1000, size=n)
            v = lo + shift + base_vals + offsets

    elif dist == "sparse_cluster":
        # Create 50 dense clusters
        centers = rng.integers(lo, hi, size=50) + shift
        v = []
        for c in centers:
            c_lo = max(lo + shift, c - 50)
            c_hi = min(hi + shift + DOMAIN_MAX, c + 50)
            if c_lo < c_hi:
                v.append(rng.integers(c_lo, c_hi, size=n // 50))
            else:
                v.append(rng.integers(lo + shift, hi + shift + 1, size=n // 50))
        v = np.concatenate(v)

    elif dist == "anti_zipf":
        # Uniform distribution over a small subset of the domain (10%)
        v = rng.integers(lo + shift, lo + shift + (DOMAIN_MAX // 10), size=n)

    else:
        # Default fallback to uniform
        v = rng.integers(lo, hi + 1, size=n)

    if n == 0: return np.array([], dtype=np.int64)

    # Ensure all values are strictly within the global domain limits [0, DOMAIN_MAX]
    v = np.vectorize(lambda x: clamp_int(x, 0, DOMAIN_MAX))(v)
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
# Statistics and Binning
# ==========================================

def freedman_diaconis_bins(sample: np.ndarray, mn: int, mx: int, n_rows: int, bins_max: int = 2000) -> int:
    """
    Calculates the optimal number of bins using the Freedman-Diaconis rule.
    """
    if sample.size < 10: return 10
    q25, q75 = np.quantile(sample, [0.25, 0.75])
    iqr = q75 - q25
    if iqr <= 0: return 10
    bin_width = 2 * iqr / (n_rows ** (1/3))
    total_width = mx - mn
    if bin_width <= 0: return 10
    return max(1, min(int(total_width / bin_width), bins_max))

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
    seen = np.zeros(DOMAIN_MAX + 1, dtype=bool)
    try:
        for chunk in pd.read_csv(csv_path, header=None, names=["v"], dtype="int64", chunksize=1_000_000, engine="c"):
            vals = chunk["v"].to_numpy()
            vals = vals[(vals >= 0) & (vals <= DOMAIN_MAX)]
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

def plot_data_distribution(vals, dist_name, output_path, n_bins=None):
    """
    Plots a histogram of the data distribution and saves bucket info.

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

    # ---------------------------------------------------------
    # 1. VISUALIZATION (Always 1000 to prevent aliasing/combing)
    # ---------------------------------------------------------
    plot_bins = 1000
    plt.hist(plot_vals, bins=plot_bins, color='skyblue', edgecolor='black', alpha=0.7, log=use_log)

    # ---------------------------------------------------------
    # 2. CSV EXPORT (Strictly using FD bins for accurate modeling)
    # ---------------------------------------------------------
    try:
        # Use passed in n_bins (FD from main loop) or calculate it
        if n_bins is None:
            fd_n_bins = freedman_diaconis_bins(plot_vals, int(plot_vals.min()), int(plot_vals.max()), len(plot_vals),
                                               bins_max=2000)
        else:
            fd_n_bins = n_bins

        # Re-calculate the actual bucket math quietly (no plotting)
        counts, bin_edges = np.histogram(plot_vals, bins=fd_n_bins)

        bucket_data = []
        for i, count in enumerate(counts):
            bucket_data.append({
                "bin_id": i,
                "bin_start": bin_edges[i],
                "bin_end": bin_edges[i + 1],
                "count": int(count)
            })

        hist_csv_path = Path(output_path).parent / "histogram_buckets.csv"
        pd.DataFrame(bucket_data).to_csv(hist_csv_path, index=False)
        print(f"Saved histogram bucket details to {hist_csv_path} (Using {fd_n_bins} FD Bins)")
    except Exception as e:
        print(f"Failed to save histogram buckets: {e}")

    # Finalize Plot
    plt.title(f"Distribution: {dist_name} (FD Bins in CSV: {fd_n_bins} | Visual Bins: {plot_bins})")
    plt.xlabel("Value")
    plt.ylabel("Frequency" + (" (Log Scale)" if use_log else ""))
    plt.grid(axis='y', alpha=0.3)
    if dist_name.lower() not in ['imdb', 'census']: plt.xlim(0, DOMAIN_MAX)
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


def plot_workload_distribution(queries: list, bucket_csv_path: Path, output_path: Path, title: str = "Workload Distribution"):
    """
    Plots a histogram of query counts per dataset bucket.
    Only counts queries that are passed in (caller should filter for >0 selectivity).
    
    Args:
        queries: List of objects with .low and .high attributes (or dicts).
        bucket_csv_path: Path to histogram_buckets.csv.
        output_path: Path to save the plot.
        title: Plot title.
    """
    if not bucket_csv_path.exists():
        print(f"Bucket CSV not found at {bucket_csv_path}. Skipping workload plot.")
        return

    try:
        df_buckets = pd.read_csv(bucket_csv_path)
        if df_buckets.empty:
            print("Bucket CSV is empty.")
            return

        # Prepare bin edges
        # Assuming contiguous bins from sorted start
        starts = df_buckets['bin_start'].values
        ends = df_buckets['bin_end'].values
        
        # robust edges: use starts and the last end
        # But bins might have gaps? FD bins covers min to max.
        # Let's assume contiguous.
        edges = np.concatenate([starts, [ends[-1]]])
        
        # Vectorize queries
        # Handle objects or dicts
        target_queries = queries
        if not target_queries:
             print("No queries to plot.")
             return
             
        if isinstance(target_queries[0], dict):
             ls = np.array([q['low'] for q in target_queries])
             rs = np.array([q['high'] for q in target_queries])
        else:
             ls = np.array([q.low for q in target_queries])
             rs = np.array([q.high for q in target_queries])

        # Find start and end bucket indices for each query
        # searchsorted returns index where value would be inserted to maintain order.
        # side='right' ensures that if value equals edge, it goes to next bucket (consistent with [a, b))?
        # Actually standard hist is [a, b). 
        # If L = edge[i], it belongs to bucket i. index -> i+1. so -1 gives i.
        # If L = edge[i] + eps, it belongs to bucket i. index -> i+1. so -1 gives i.
        
        idx_start = np.searchsorted(edges, ls, side='right') - 1
        idx_end = np.searchsorted(edges, rs, side='right') - 1
        
        # Clamp indices to valid buckets [0, len(buckets)-1]
        # If query is outside domain, clamp to nearest.
        idx_start = np.clip(idx_start, 0, len(df_buckets) - 1)
        idx_end = np.clip(idx_end, 0, len(df_buckets) - 1)
        
        # Use difference array to compute counts
        # counts[i] increments if query covers bucket i.
        # Query covers [idx_start, idx_end] inclusive.
        # diff[idx_start] += 1
        # diff[idx_end + 1] -= 1
        
        diff = np.zeros(len(df_buckets) + 1, dtype=int)
        np.add.at(diff, idx_start, 1)
        np.add.at(diff, idx_end + 1, -1)
        
        counts = np.cumsum(diff)[:-1] # drop last logic element
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        # Use simple bar plot
        # x-axis is bucket index
        x = np.arange(len(counts))
        plt.bar(x, counts, width=1.0, color='orange', edgecolor='black', alpha=0.7)
        
        plt.title(f"{title} (Total Queries: {len(target_queries)})")
        plt.xlabel("Bucket Index (FD Bins)")
        plt.ylabel("Workload Count (Queries Intersecting)")
        plt.grid(axis='y', alpha=0.3)
        
        # Add a text annotation for total bins
        plt.text(0.98, 0.95, f"Bins: {len(counts)}", transform=plt.gca().transAxes, 
                 ha='right', va='top', bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Workload distribution plot saved to {output_path}")

    except Exception as e:
        print(f"Error plotting workload distribution: {e}")
        import traceback
        traceback.print_exc()

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


def plot_micro_distribution(vals: np.ndarray, dist_name: str, output_path: Path):
    """
    Plots a histogram isolating the 'micro-noise' inside the 1000-width buckets.
    Uses the modulo operator to aggregate all micro-buckets into one view.
    """
    bucket_size = 1000

    # Isolate the micro-distribution by taking modulo 1000
    micro_vals = vals % bucket_size

    plt.figure(figsize=(8, 5))
    use_log = (dist_name.lower() == 'zipf')

    # Plot with 100 bins to see the shape clearly
    plt.hist(micro_vals, bins=100, color='coral', edgecolor='black', alpha=0.7, log=use_log)

    plt.title(f"Inner Micro-Bucket Distribution: {dist_name.capitalize()}")
    plt.xlabel(f"Offset inside 1000-width bucket [0, {bucket_size - 1}]")
    plt.ylabel("Frequency" + (" (Log Scale)" if use_log else ""))
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved micro-distribution plot to {output_path}")

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

    def ensure_dataset_exists(self, rows: int, dist: str) -> Path:
        """
        Checks if a dataset exists. Raises FileNotFoundError if not.
        Returns the dataset directory.
        """
        ds_dir = self.get_dataset_dir(rows, dist)
        if not (ds_dir / "data.csv").exists() or not (ds_dir / "meta.pkl").exists():
            raise FileNotFoundError(f"Dataset {dist} ({rows} rows) not found at {ds_dir}. Please run datasets.py first.")
        return ds_dir

    def prepare_dataset(self, rows: int, dist: str, force_regeneration: bool = False, use_micro_dist: bool = True):
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
                vals = gen_values(rng, dist, rows, 0, DOMAIN_MAX, use_micro_dist=use_micro_dist)
            
            save_csv_column(vals, data_path)
            
            # Compute Stats
            mn, mx, N = scan_min_max_count(data_path)
            ndv = calculate_ndv(data_path)
            skew, kurt = calculate_skew_kurt(data_path)
            
            # Automatically calculate optimal bin count using Freedman-Diaconis rule
            freq, sample = build_frequency_and_sample(data_path, mn, mx, N, 100_000, 42)
            k = freedman_diaconis_bins(sample, mn, mx, N)
            h = (mx - mn) / k if k > 0 else 0.0
            
            stats = {
                "Rows": int(N),
                "Min": int(mn),
                "Max": int(mx),
                "NDV": int(ndv),
                "Skewness": float(skew),
                "Kurtosis": float(kurt),
                "Bin Count (k)": int(k),
                "Bin Width (h)": float(h)
            }
            save_stats(stats, stats_path)
            
            # Save a binary meta file for fast loading in main.py (legacy compatibility)
            meta_path = ds_dir / "meta.pkl"
            with open(meta_path, "wb") as f:
                pickle.dump((mn, mx, N, freq, sample, k, skew, kurt), f)
                
            # Plot distribution
            plot_data_distribution(vals, dist, ds_dir / "hist.png", n_bins=k)
            plot_micro_distribution(vals, dist, ds_dir / "hist_micro.png")

        print(f"Dataset ready at {ds_dir}")
        return ds_dir

# ==========================================
# Main Execution (Generation)
# ==========================================
import time

import argparse

def main():
    """
    Batch generation entry point.
    Generates datasets for predefined distributions (uniform, normal, zipf, etc.)
    and row counts (1M, 10M, 60M).
    """
    parser = argparse.ArgumentParser(description="Generate benchmark datasets.")
    parser.add_argument("--rows", type=int, nargs='+', help="List of row counts to generate (e.g. 10000 1000000).")
    parser.add_argument("--dist", type=str, help="Distribution to generate (uniform, normal, zipf, etc).")
    parser.add_argument("--all", action="store_true", help="Generate all default datasets (1M, 10M, 60M).")
    parser.add_argument("--simple", action="store_true", help="Use traditional simple approach without micro-distributions.")
    
    args = parser.parse_args()
    
    dm = DatasetManager()
    total_start = time.time()

    # Determine rows to generate
    rows_list = []
    if args.rows:
        rows_list = args.rows
    elif args.all:
        rows_list = [1_000_000, 10_000_000, 60_000_000]

    # Determine distributions
    if args.dist:
        distributions = [args.dist]
    elif args.all:
        distributions = ["uniform", "normal", "zipf", "sparse_cluster", "anti_zipf"]
    else:
        distributions = []

    if not rows_list or not distributions:
        parser.print_help()
        print("\nError: --rows and --dist are required (or --all).")
        return

    for rows in rows_list:
        for dist in distributions:
            print(f"\n>>> Generating Dataset: {rows} rows, {dist} <<<")
            start = time.time()
            try:
                # 1. Generate Data
                ds_dir = dm.prepare_dataset(rows, dist, use_micro_dist=not args.simple)
                
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
