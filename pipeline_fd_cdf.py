#!/usr/bin/env python3
# pipeline_fd_cdf.py
#
# Hybrid Selectivity Estimation:
# 1. Equi-Width Histograms (Fast build)
# 2. Freedman-Diaconis (FD) Binning (Auto bin count)
# 3. CDF-based Learning in each Bucket (Consistency & Accuracy)
#
# Usage:
#   python pipeline_fd_cdf.py --dist zipf --rows 1000000 --eval-n 1000

training = tail + CDF (true value)
inference = tail = CDF

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from joblib import Parallel, delayed

import numpy as np
import pandas as pd

# -------------------------
# Dataset generation
# -------------------------

def clamp_int(x, lo, hi):
    return int(min(max(int(round(x)), lo), hi))

def gen_values(rng: np.random.Generator, dist: str, n: int, lo: int, hi: int) -> np.ndarray:
    mid = 0.5 * (lo + hi)
    span = max(1, hi - lo)

    if dist == "uniform":
        v = rng.integers(lo, hi + 1, size=n)
    elif dist == "normal":
        v = rng.normal(loc=mid, scale=span / 6.0, size=n)
    elif dist == "zipf":
        a = 2.0
        z = rng.zipf(a, size=n)
        v = lo + z
    else:
        v = rng.integers(lo, hi + 1, size=n)

    v = np.vectorize(lambda x: clamp_int(x, lo, hi))(v)
    return v.astype(np.int64)

def save_csv_column(values: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(values).to_csv(path, index=False, header=False)

# -------------------------
# Histogram Helpers
# -------------------------

@dataclass
class Bucket:
    lo: int
    hi: int
    count: int = 0  # To be filled after creation

def scan_min_max_count(csv_path: Path, chunksize: int = 1_000_000) -> Tuple[int, int, int]:
    mn, mx, n = None, None, 0
    for ch in pd.read_csv(csv_path, header=None, names=["v"], dtype="int64", chunksize=chunksize, engine="c"):
        v = ch["v"].to_numpy()
        n += v.size
        if v.size > 0:
            cmin, cmax = int(v.min()), int(v.max())
            mn = cmin if mn is None else min(mn, cmin)
            mx = cmax if mx is None else max(mx, cmax)
    if mn is None: raise ValueError("Empty CSV")
    return mn, mx, n

def build_frequency_and_sample(csv_path: Path, mn: int, mx: int, n_rows: int, sample_size: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    width = mx - mn + 1
    freq = np.zeros(width, dtype=np.int64)
    rng = np.random.default_rng(seed)
    sampled = []
    
    # Simple reservoir-like sampling or chunk sampling
    # Since we know N, we can just pick random rows to keep? 
    # Or simpler: keep first N. Or random prob p.
    p = min(1.0, float(sample_size * 2) / float(max(n_rows, 1)))

    for ch in pd.read_csv(csv_path, header=None, names=["v"], dtype="int64", chunksize=1_000_000, engine="c"):
        vals = ch["v"].to_numpy()
        idx = vals - mn
        m = (idx >= 0) & (idx < width)
        if not np.all(m): idx = idx[m]
        if idx.size:
            freq += np.bincount(idx, minlength=width)
        
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

def make_equiwidth_buckets(mn: int, mx: int, bins: int, freq: np.ndarray) -> List[Bucket]:
    width = mx - mn + 1
    bw = max(1, int(math.ceil(width / bins)))
    buckets = []
    ps = np.cumsum(freq)
    
    cur = mn
    for _ in range(bins):
        lo = cur
        hi = min(mx, lo + bw - 1)
        li = lo - mn
        ri = hi - mn
        if li < 0: li=0 # safety
        if ri >= len(freq): ri = len(freq)-1
        
        cnt = int(ps[ri] - (ps[li-1] if li > 0 else 0))
        buckets.append(Bucket(lo, hi, count=cnt))
        
        cur = hi + 1
        if cur > mx: break
    return buckets

def freedman_diaconis_bins(sample: np.ndarray, mn: int, mx: int, n_rows: int, bins_max: int) -> int:
    if sample.size < 10: return 10
    q25, q75 = np.quantile(sample, [0.25, 0.75])
    iqr = q75 - q25
    if iqr <= 0: return 10
    
    bin_width = 2 * iqr / (n_rows ** (1/3))
    total_width = mx - mn
    if bin_width <= 0: return 10
    
    bins = int(total_width / bin_width)
    return max(1, min(bins, bins_max))

# -------------------------
# CDF Training and Model
# -------------------------

@dataclass
class CDFTrainRow:
    x_norm: float
    y_cdf: float

def collect_cdf_training_rows(buckets: List[Bucket], freq: np.ndarray, mn: int, points_per_bucket: int, rng) -> Dict[int, List[CDFTrainRow]]:
    ps = np.cumsum(freq)
    rows = {i: [] for i in range(len(buckets))}
    
    for i, b in enumerate(buckets):
        if b.count == 0: continue
        
        # Pick random points to probe CDF
        # Ranging from lo to hi
        xs = rng.integers(b.lo, b.hi + 1, size=points_per_bucket)
        xs = np.sort(xs)
        
        width = b.hi - b.lo + 1
        b_lo_idx = b.lo - mn
        base_cnt = ps[b_lo_idx - 1] if b_lo_idx > 0 else 0
        
        for x in xs:
            x_idx = x - mn
            # Calculate local CDF value: P(X <= x | X in Bucket)
            # = (Cumulative(x) - Cumulative(bucket_start-1)) / BucketCount
            curr_cnt = ps[x_idx]
            local_cnt = curr_cnt - base_cnt
            y_cdf = local_cnt / b.count
            
            x_norm = (x - b.lo) / width
            rows[i].append(CDFTrainRow(x_norm, y_cdf))
            
    return rows

def train_cdf_models(rows: Dict[int, List[CDFTrainRow]]):
    from sklearn.linear_model import Ridge
    models = {}
    train_time = 0.0
    
    # Sequential training for simplicity, or parallel
    for i, rlist in rows.items():
        if len(rlist) < 2:
            models[i] = None
            continue
            
        X = np.array([[r.x_norm] for r in rlist])
        y = np.array([r.y_cdf for r in rlist])
        
        t0 = time.perf_counter()
        mdl = Ridge(alpha=1e-4)
        mdl.fit(X, y)
        train_time += (time.perf_counter() - t0)
        models[i] = mdl
        
    return models, train_time

def predict_local_cdf(model, x_norm) -> float:
    if model is None:
        # Uniform assumption: CDF(x) = x (linear growth)
        return max(0.0, min(1.0, x_norm))
    val = model.predict([[x_norm]])[0]
    return max(0.0, min(1.0, val))

# -------------------------
# Inference
# -------------------------

@dataclass
class RangeQuery:
    low: int
    high: int

def predict_range_hybrid_cdf(q: RangeQuery, buckets: List[Bucket], models) -> float:
    # Estimate total count = Prob(<= high) - Prob(<= low - 1)
    
    def estimate_cum_count(val: int) -> float:
        # Find bucket causing val
        # Since buckets are ordered and contiguous (mostly), valid assumption for equi-width
        if val < buckets[0].lo: return 0.0
        if val > buckets[-1].hi: return sum(b.count for b in buckets)
        
        # Find bucket index
        # For equi-width, we can calculate index directly!
        # b_idx = (val - mn) // bucket_width
        # But let's be robust and use search as general case
        
        # Simple linear search optimized for common case? No, binary search.
        import bisect
        # create list of hi bounds
        # But simpler: just scan or math.
        # Let's trust buckets[i] covers range.
        
        # Fast Equi-Width Math:
        width_domain = buckets[-1].hi - buckets[0].lo + 1 # approx
        # Assuming true equi-width:
        # i = (val - buckets[0].lo) / width_per_bucket ?
        # But width varies by +/- 1 due to int div.
        
        # Linear scan for now (safe)
        b_idx = -1
        cum_pre = 0
        for i, b in enumerate(buckets):
            if val < b.lo:
                break
            if val <= b.hi:
                b_idx = i
                break
            cum_pre += b.count
            
        if b_idx != -1:
            b = buckets[b_idx]
            w = b.hi - b.lo + 1
            x_norm = (val - b.lo) / w
            local_cdf = predict_local_cdf(models.get(b_idx), x_norm)
            return cum_pre + local_cdf * b.count
        else:
            return cum_pre # Should match total if val > max
            
    c_high = estimate_cum_count(q.high)
    c_low_minus = estimate_cum_count(q.low - 1)
    
    return max(0.0, c_high - c_low_minus)


def predict_range_histogram_uniform(q: RangeQuery, buckets: List[Bucket]) -> float:
    # Standard histogram inference
    total = 0.0
    for b in buckets:
        ov_lo = max(q.low, b.lo)
        ov_hi = min(q.high, b.hi)
        if ov_lo <= ov_hi:
            w = b.hi - b.lo + 1
            frac = (ov_hi - ov_lo + 1) / w
            total += frac * b.count
    return total

# -------------------------
# Main
# -------------------------
def q_error_vec(y_true, y_pred, eps=1e-9):
    yt = np.maximum(y_true, eps)
    yp = np.maximum(y_pred, eps)
    return np.maximum(yt/yp, yp/yt)

def summarize(y_true, y_pred):
    qe = q_error_vec(y_true, y_pred)
    return {
        "QErr_median": float(np.median(qe)),
        "QErr_p95": float(np.percentile(qe, 95)),
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--dist", default="zipf")
    ap.add_argument("--bins", type=int, default=100) # Fallback if FD fails or manual
    ap.add_argument("--fd", action="store_true", help="Use Freedman-Diaconis binning")
    ap.add_argument("--eval-n", type=int, default=500)
    ap.add_argument("--out-dir", default="artifacts_fd_cdf")
    ap.add_argument("--input-csv", help="Use input csv")
    
    args = ap.parse_args()
    rng = np.random.default_rng(42)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Dataset
    if args.input_csv:
        ds_path = Path(args.input_csv)
    else:
        ds_path = Path(args.out_dir) / f"data_{args.dist}.csv"
        vals = gen_values(rng, args.dist, args.rows, 0, 100_000)
        save_csv_column(vals, ds_path)
        
    print(f"Dataset: {ds_path}")
    mn, mx, N = scan_min_max_count(ds_path)
    freq, sample = build_frequency_and_sample(ds_path, mn, mx, N, 100_000, 42)
    
    # 2. Binning (Histogram "Training")
    t0_hist = time.perf_counter()
    n_bins = args.bins
    if args.fd:
        n_bins = freedman_diaconis_bins(sample, mn, mx, N, 1000)
        print(f"FD Suggested Bins: {n_bins}")
        
    buckets = make_equiwidth_buckets(mn, mx, n_bins, freq)
    t_hist_build = time.perf_counter() - t0_hist
    
    # 3. Model Training (CDF - Hybrid "Training")
    print("Training CDF models...")
    rows = collect_cdf_training_rows(buckets, freq, mn, 20, rng)
    models, t_ml_train = train_cdf_models(rows)
    print(f"Hist Build: {t_hist_build:.4f}s, ML Train: {t_ml_train:.4f}s")
    
    # Total hybrid training is Hist + ML
    t_hybrid_train = t_hist_build + t_ml_train
    
    # 4. Evaluation
    print("Generating Evaluation Workload...")
    queries = []
    width = mx - mn
    ps = np.cumsum(freq)
    
    for _ in range(args.eval_n):
        l = rng.integers(mn, mx)
        w = rng.integers(1, width // 10)
        r = min(mx, l + w)
        queries.append(RangeQuery(l, r))
        
    y_true = []
    y_hist = []
    y_hybrid = []
    
    # Measure Baseline Inference
    t0_base = time.perf_counter()
    for q in queries:
        h = predict_range_histogram_uniform(q, buckets)
        y_hist.append(h / N)
    t_base_inf = time.perf_counter() - t0_base
    
    # Measure Hybrid Inference
    t0_hyb = time.perf_counter()
    for q in queries:
        c = predict_range_hybrid_cdf(q, buckets, models)
        y_hybrid.append(c / N)
    t_hyb_inf = time.perf_counter() - t0_hyb

    # Truth (outside timing)
    for q in queries:
        li, ri = q.low - mn, q.high - mn
        if li < 0: li=0
        if ri >= len(ps): ri = len(ps)-1
        truth = int(ps[ri] - (ps[li-1] if li > 0 else 0))
        y_true.append(truth / N)
        
    y_true = np.array(y_true)
    y_hist = np.array(y_hist)
    y_hybrid = np.array(y_hybrid)
    
    m_hist = summarize(y_true, y_hist)
    m_hyb = summarize(y_true, y_hybrid)
    
    print("\n--- Results ---")
    print(f"Standard Histogram: Median QErr={m_hist['QErr_median']:.4f}, MAE={m_hist['MAE']:.6f}, InfTime={t_base_inf:.4f}s")
    print(f"Hybrid CDF:         Median QErr={m_hyb['QErr_median']:.4f}, MAE={m_hyb['MAE']:.6f}, InfTime={t_hyb_inf:.4f}s")
    
    summary = {
        "bins": n_bins,
        
        "hist_build_time": t_hist_build,
        "hybrid_train_time_total": t_hybrid_train,
        
        "hist_inference_time": t_base_inf,
        "hybrid_inference_time": t_hyb_inf,
        
        "histogram_baseline": m_hist,
        "hybrid_cdf": m_hyb
    }
    with open(Path(args.out_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
