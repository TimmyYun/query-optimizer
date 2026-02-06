import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import shutil
import pickle
from pathlib import Path
from typing import Tuple, List

# ==========================================
# Data Utils
# ==========================================

def clamp_int(x, lo, hi):
    return int(min(max(int(round(x)), lo), hi))

def gen_values(rng: np.random.Generator, dist: str, n: int, lo: int, hi: int, shift: int = 0) -> np.ndarray:
    mid = 0.5 * (lo + hi) + shift
    span = max(1, hi - lo)

    if dist == "uniform":
        v = rng.integers(lo + shift, hi + 1 + shift, size=n)
    elif dist == "normal":
        v = rng.normal(loc=mid, scale=span / 6.0, size=n)
    elif dist == "zipf":
        v = lo + shift + rng.zipf(2.0, size=n)
    elif dist == "sparse_cluster":
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
        v = rng.integers(lo + shift, lo + shift + 20000, size=n)
    else:
        v = rng.integers(lo, hi + 1, size=n)
        
    if n == 0: return np.array([], dtype=np.int64)
    v = np.vectorize(lambda x: clamp_int(x, 0, 200_000))(v)
    return v.astype(np.int64)

def save_csv_column(values: np.ndarray, path: Path, mode='w'):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.Series(values)
    df.to_csv(path, index=False, header=False, mode=mode)

def scan_min_max_count(csv_path: Path, chunksize: int = 1_000_000) -> Tuple[int, int, int]:
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
    width = mx - mn + 1
    if width <= 0: return np.array([]), np.array([])
    freq = np.zeros(width, dtype=np.int64)
    rng = np.random.default_rng(seed)
    sampled = []
    p = min(1.0, float(sample_size * 2) / float(max(n_rows, 1)))

    for ch in pd.read_csv(csv_path, header=None, names=["v"], dtype="int64", chunksize=1_000_000, engine="c"):
        vals = ch["v"].to_numpy()
        idx = vals - mn
        m = (idx >= 0) & (idx < width)
        valid_idx = idx[m]
        if valid_idx.size:
            freq += np.bincount(valid_idx, minlength=width)
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
    try:
        df = pd.read_csv(csv_path)
        col = "review" if "review" in df.columns else df.columns[0]
        vals = df[col].astype(str).str.len().to_numpy()
        return vals.astype(np.int64)
    except Exception as e:
        print(f"Error loading IMDB: {e}")
        return np.array([], dtype=np.int64)

def load_census_age(csv_path: Path) -> np.ndarray:
    try:
        df = pd.read_csv(csv_path, usecols=['dAge'])
        return df['dAge'].to_numpy().astype(np.int64)
    except Exception as e:
        print(f"Error loading Census: {e}")
        return np.array([], dtype=np.int64)

# ==========================================
# Stats Utils
# ==========================================

def freedman_diaconis_bins(sample: np.ndarray, mn: int, mx: int, n_rows: int, bins_max: int = 2000) -> int:
    if sample.size < 10: return 10
    q25, q75 = np.quantile(sample, [0.25, 0.75])
    iqr = q75 - q25
    if iqr <= 0: return 10
    bin_width = 2 * iqr / (n_rows ** (1/3))
    total_width = mx - mn
    if bin_width <= 0: return 10
    return max(1, min(int(total_width / bin_width), bins_max))

def calculate_skew_kurt(csv_path: Path) -> Tuple[float, float]:
    try:
        # For performance with large files, we should probably chunk this or use sampling, 
        # but maintaining compatibility for now.
        df = pd.read_csv(csv_path, header=None, names=["v"], dtype="int64")
        return float(df["v"].skew()), float(df["v"].kurt())
    except Exception as e:
        print(f"Error calculating skew/kurt: {e}")
        return 0.0, 0.0

def calculate_ndv(csv_path: Path):
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
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=4)

# ==========================================
# Plot Utils
# ==========================================

def plot_data_distribution(vals, dist_name, output_path):
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
