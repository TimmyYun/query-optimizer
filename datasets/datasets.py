import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

def clamp_int(x, lo, hi):
    """
    Clamps a value `x` to the range [lo, hi].

    Args:
        x: The value to clamp.
        lo: The lower bound.
        hi: The upper bound.

    Returns:
        The clamped integer value.
    """
    return int(min(max(int(round(x)), lo), hi))

def gen_values(rng: np.random.Generator, dist: str, n: int, lo: int, hi: int, shift: int = 0) -> np.ndarray:
    """
    Generates synthetic data based on a specified distribution.

    Args:
        rng: The random number generator.
        dist: The distribution type ("uniform", "normal", "zipf", "sparse_cluster", "anti_zipf").
        n: The number of values to generate.
        lo: The lower bound of the range.
        hi: The upper bound of the range.
        shift: An optional shift to apply to the generated values.

    Returns:
        A numpy array of generated integer values.
    """
    mid = 0.5 * (lo + hi) + shift
    span = max(1, hi - lo)

    if dist == "uniform":
        v = rng.integers(lo + shift, hi + 1 + shift, size=n)
    elif dist == "normal":
        v = rng.normal(loc=mid, scale=span / 6.0, size=n)
    elif dist == "zipf":
        a = 2.0
        z = rng.zipf(a, size=n)
        v = lo + shift + z
    elif dist == "sparse_cluster":
        # Create clusters of values with empty gaps
        # Centers shifted
        centers = rng.integers(lo, hi, size=10) + shift
        v = []
        for c in centers:
            # cluster width 100
            # Ensure valid range even after shift
            c_lo = max(lo + shift, c - 50)
            c_hi = min(hi + shift + 200_000, c + 50) # Allow drift to go higher
            if c_lo < c_hi:
                cluster_vals = rng.integers(c_lo, c_hi, size=n // 10)
                v.append(cluster_vals)
            else:
                 # Fallback if cluster is out of bounds
                 v.append(rng.integers(lo+shift, hi+shift+1, size=n//10))
        v = np.concatenate(v)
    elif dist == "anti_zipf":
        # Destructive distribution: Uniform injected into the heavy-hitter region of Zipf
        # Respect lo and shift. Default Zipf pushes to 'lo', so Anti-Zipf should also start at 'lo'.
        start = lo + shift
        # Width of 20,000 matches the original hardcoded range [0, 20000]
        v = rng.integers(start, start + 20000, size=n)
    else:
        v = rng.integers(lo, hi + 1, size=n)
        
    if n == 0: return np.array([], dtype=np.int64)
    v = np.vectorize(lambda x: clamp_int(x, 0, 200_000))(v)
    return v.astype(np.int64)

def save_csv_column(values: np.ndarray, path: Path, mode='w'):
    """
    Saves a numpy array of values to a CSV file.

    Args:
        values: The numpy array of values to save.
        path: The path to the CSV file.
        mode: The file opening mode ('w' for write, 'a' for append).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.Series(values)
    if mode == 'w':
        df.to_csv(path, index=False, header=False)
    else:
        df.to_csv(path, index=False, header=False, mode='a')

def scan_min_max_count(csv_path: Path, chunksize: int = 1_000_000) -> Tuple[int, int, int]:
    """
    Scans a CSV file to find the minimum value, maximum value, and total count.

    Args:
        csv_path: The path to the CSV file.
        chunksize: The number of rows to read per chunk.

    Returns:
        A tuple containing (min_val, max_val, count).
    """
    mn, mx, n = None, None, 0
    for ch in pd.read_csv(csv_path, header=None, names=["v"], dtype="int64", chunksize=chunksize, engine="c"):
        v = ch["v"].to_numpy()
        n += v.size
        # Fix for empty chunk or all NaNs? dtype int64 should appear as ints.
        if v.size > 0:
            cmin, cmax = int(v.min()), int(v.max())
            mn = cmin if mn is None else min(mn, cmin)
            mx = cmax if mx is None else max(mx, cmax)
    if mn is None: return 0, 0, 0 
    return mn, mx, n

def build_frequency_and_sample(csv_path: Path, mn: int, mx: int, n_rows: int, sample_size: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds a frequency histogram and collects a random sample from a CSV file.

    Args:
        csv_path: The path to the CSV file.
        mn: The minimum value in the dataset (used for indexing).
        mx: The maximum value in the dataset.
        n_rows: The total number of rows (estimated or exact).
        sample_size: The desired size of the reservoir sample.
        seed: Random seed for sampling.

    Returns:
        A tuple containing:
        - freq: A numpy array representing the frequency of each value in the range [mn, mx].
        - sample: A numpy array containing the random sample.
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
        m = (idx >= 0) & (idx < width)
        
        valid_idx = idx[m]
        if valid_idx.size:
            freq += np.bincount(valid_idx, minlength=width)
        
        # Sampling
        if p > 0 and len(sampled) < sample_size:
            mask = rng.random(vals.size) < p
            s = vals[mask]
            if s.size > 0:
                sampled.append(s)

    if sampled:
        sample = np.concatenate(sampled)
        if sample.size > sample_size:
            sample = rng.choice(sample, size=sample_size, replace=False)
    else:
        sample = np.array([], dtype=np.int64)
        
    return freq, sample
def load_imdb_lengths(csv_path: Path) -> np.ndarray:
    """Reads IMDB Dataset.csv and returns review lengths as numpy array."""
    try:
        # IMDB Dataset has explicit header "review,sentiment"
        imdb_path = Path("datasets/files/imdb/IMDB Dataset.csv")
        if not imdb_path.exists(): imdb_path = Path("files/imdb/IMDB Dataset.csv")
        # Fallback to current dir if needed, but prefer organized structure
        if not imdb_path.exists(): imdb_path = csv_path
        
        df = pd.read_csv(imdb_path)
        if "review" not in df.columns:
            # Fallback if no header or different name
            df = pd.read_csv(csv_path, header=None)
            vals = df.iloc[:, 0].astype(str).str.len().to_numpy()
        else:
            vals = df["review"].astype(str).str.len().to_numpy()
            
        print(f"Loaded IMDB: {len(vals)} rows. Max len: {vals.max()}, Min len: {vals.min()}")
        return vals.astype(np.int64)
    except Exception as e:
        print(f"Error loading IMDB: {e}")
        return np.array([], dtype=np.int64)

def load_census_age(csv_path: Path) -> np.ndarray:
    """Reads USCensus1990.data.txt.csv and returns dAge column."""
    try:
        # Check if file exists
        if not csv_path.exists():
            print(f"Census file not found: {csv_path}")
            return np.array([], dtype=np.int64)
            
        # Read 'dAge' column. It's the 2nd column (index 1) in the header.
        # Read 'dAge' column. It's the 2nd column (index 1) in the header.
        # Format: caseid,dAge,dAncstry1...
        census_path_default = Path("datasets/files/census/USCensus1990.data.txt.csv")
        if census_path_default.exists():
            csv_path = census_path_default
            
        df = pd.read_csv(csv_path, usecols=['dAge'])
        vals = df['dAge'].to_numpy()
        
        print(f"Loaded Census: {len(vals)} rows. Max Age: {vals.max()}, Min Age: {vals.min()}")
        return vals.astype(np.int64)
    except Exception as e:
        print(f"Error loading Census: {e}")
        return np.array([], dtype=np.int64)

def freedman_diaconis_bins(sample: np.ndarray, mn: int, mx: int, n_rows: int, bins_max: int) -> int:
    """
    Calculates the optimal number of bins using the Freedman-Diaconis rule.
    
    The FD rule uses the Interquartile Range (IQR) to estimate the optimal bin width,
    which is robust to outliers.
    
    Formula: Bin Width = 2 * IQR * n^(-1/3)
    
    Args:
        sample: A sample of values from the dataset.
        mn: Minimum value in the dataset.
        mx: Maximum value in the dataset.
        n_rows: Total number of rows in the dataset.
        bins_max: Maximum allowed number of bins.
    
    Returns:
        The calculated number of bins, clamped between 1 and bins_max.
    """
    if sample.size < 10: return 10
    q25, q75 = np.quantile(sample, [0.25, 0.75])
    iqr = q75 - q25
    if iqr <= 0: return 10
    
    bin_width = 2 * iqr / (n_rows ** (1/3))
    total_width = mx - mn
    if bin_width <= 0: return 10
    
    bins = int(total_width / bin_width)
    return max(1, min(bins, bins_max))
def calculate_skew_kurt(csv_path: Path, chunksize: int = 1_000_000) -> Tuple[float, float]:
    """
    Calculates the skewness and kurtosis of a dataset from a CSV file.
    Uses pandas for calculation on the full series (loads all values into memory).
    """
    try:
        # For simplicity and given the 60M limit (which fits in memory for a single column),
        # we load the values. If memory becomes an issue, this would need an online algorithm.
        df = pd.read_csv(csv_path, header=None, names=["v"], dtype="int64")
        skew = float(df["v"].skew())
        kurt = float(df["v"].kurt())
        return skew, kurt
    except Exception as e:
        print(f"Error calculating skew/kurt: {e}")
        return 0.0, 0.0
