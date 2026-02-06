import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

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
