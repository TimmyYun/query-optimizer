#!/usr/bin/env python3
# pipeline_fd_cdf.py
#
# Hybrid Selectivity Estimation:
# 1. Equi-Width Histograms (Fast build)
# 2. Freedman-Diaconis (FD) Binning (Auto bin count)
# 3. CDF-based Learning (Isotonic Regression) in each Bucket
# 4. NDV-Awareness: Use Exact Values (MCV) for sparse buckets
# 5. Adaptive Maintenance: Handle Data Drift & Retrain Bad Buckets
#
# Usage:
#   python pipeline_fd_cdf.py --dist zipf --rows 10000000 --drift-rows 2000000

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

# -------------------------
# Dataset generation
# -------------------------

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
        a = 2.0
        z = rng.zipf(a, size=n)
        v = lo + shift + z
    elif dist == "sparse_cluster":
        # Create clusters of values with empty gaps
        centers = rng.integers(lo, hi, size=10) + shift
        v = []
        for c in centers:
            # cluster width 100
            cluster_vals = rng.integers(max(lo, c-50), min(hi, c+50), size=n // 10)
            v.append(cluster_vals)
        v = np.concatenate(v)
        # Ensure we match N exactly if needed, but this is approx
    elif dist == "anti_zipf":
        # Destructive distribution: Uniform injected into the heavy-hitter region of Zipf
        # Zipf(a=2) has massive mass at low values.
        # We inject Uniform [0, 20000] (10% domain) to linearize the CDF in that region.
        # This conflicts with the "convex" shape of Zipf CDF.
        v = rng.integers(0, 20000, size=n)
    else:
        v = rng.integers(lo, hi + 1, size=n)
        
    v = np.vectorize(lambda x: clamp_int(x, 0, 200_000))(v)
    return v.astype(np.int64)



def save_csv_column(values: np.ndarray, path: Path, mode='w'):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Append or write
    df = pd.Series(values)
    if mode == 'w':
        df.to_csv(path, index=False, header=False)
    else:
        df.to_csv(path, index=False, header=False, mode='a')

# -------------------------
# Histogram Helpers
# -------------------------

@dataclass
class Bucket:
    lo: int
    hi: int
    count: int = 0  # To be filled after creation
    ndv: int = 0
    exact_values: Optional[List[Tuple[int, int]]] = None # For sparse buckets
    # Store model here? or separate dict? Separate is fine for pickling usually.

@dataclass
class RangeQuery:
    low: int
    high: int


def scan_min_max_count(csv_path: Path, chunksize: int = 1_000_000) -> Tuple[int, int, int]:
    mn, mx, n = None, None, 0
    for ch in pd.read_csv(csv_path, header=None, names=["v"], dtype="int64", chunksize=chunksize, engine="c"):
        v = ch["v"].to_numpy()
        n += v.size
        if v.size > 0:
            cmin, cmax = int(v.min()), int(v.max())
            mn = cmin if mn is None else min(mn, cmin)
            mx = cmax if mx is None else max(mx, cmax)
    if mn is None: return 0, 0, 0 # Handle empty
    return mn, mx, n

def build_frequency_and_sample(csv_path: Path, mn: int, mx: int, n_rows: int, sample_size: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    width = mx - mn + 1
    if width <= 0: return np.array([]), np.array([])
    
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
        
        # Use valid indices
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

def make_equiwidth_buckets(mn: int, mx: int, bins: int, freq: np.ndarray) -> List[Bucket]:
    if len(freq) == 0: return []
    width = mx - mn + 1
    bw = max(1, int(math.ceil(width / bins)))
    buckets = []
    ps = np.cumsum(freq)
    
    # Heuristic for storage: if NDV is small enough, store exact values.
    # Say, up to 200 distinct values per bucket.
    EXACT_STORAGE_THRESHOLD = 200
    
    cur = mn
    for _ in range(bins):
        lo = cur
        hi = min(mx, lo + bw - 1)
        li = lo - mn
        ri = hi - mn
        if li < 0: li=0 # safety
        if ri >= len(freq): ri = len(freq)-1
        
        # Count
        cnt = int(ps[ri] - (ps[li-1] if li > 0 else 0))
        
        # Calculate NDV
        # slice freq array
        freq_slice = freq[li : ri+1]
        ndv = np.count_nonzero(freq_slice)
        
        b = Bucket(lo, hi, count=cnt, ndv=ndv)
        
        # If sparse/low-NDV, store exact values
        if ndv > 0 and ndv <= EXACT_STORAGE_THRESHOLD:
            # Reconstruct values from freq slice
            # indices where freq > 0
            rel_indices = np.nonzero(freq_slice)[0]
            # Map back to absolute values and store (value, count) pairs
            vals_with_counts = []
            for idx in rel_indices:
                 vals_with_counts.append((int(idx + lo), int(freq_slice[idx])))
            b.exact_values = vals_with_counts
            
        buckets.append(b)
        
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
# EquiHist (Self-Tuning) Baseline
# -------------------------

class EquiHistLearner:
    """
    Implements a self-tuning histogram baseline (conceptually based on König & Weikum).
    Starts with a set of buckets (e.g. initial equi-width) and updates bucket counts 
    based on query feedback (prediction error).
    """
    def __init__(self, buckets: List[Bucket], learning_rate: float = 0.5):
        # working copy of buckets (only counts matter for update)
        self.buckets = [Bucket(b.lo, b.hi, count=b.count) for b in buckets]
        self.lr = learning_rate
        
    def predict(self, q: RangeQuery) -> float:
        # Standard uniform assumption
        total = 0.0
        for b in self.buckets:
            ov_lo = max(q.low, b.lo)
            ov_hi = min(q.high, b.hi)
            if ov_lo <= ov_hi:
                w = b.hi - b.lo + 1
                frac = (ov_hi - ov_lo + 1) / w
                total += frac * b.count
        return total
        
    def update(self, q: RangeQuery, actual: float):
        # Update rule: f_new = f_old * (actual / predicted) ^ (learning_rate * overlap_fraction)
        # Simplified Multiplicative Weights
        pred = self.predict(q)
        if pred == 0: return # Avoid div zero
        
        ratio = actual / pred
        # Clamp ratio to avoid explosion
        ratio = max(0.1, min(10.0, ratio))
        
        for b in self.buckets:
            ov_lo = max(q.low, b.lo)
            ov_hi = min(q.high, b.hi)
            if ov_lo <= ov_hi:
                # Fraction of bucket involved in query
                w = b.hi - b.lo + 1
                overlap = (ov_hi - ov_lo + 1) / w
                
                # We update the bucket count
                # If ratio > 1 (underestimate), we boost count
                # If ratio < 1 (overestimate), we shrink count
                # Weighted by overlap: only update part that was touched? 
                # EquiHist updates the whole bucket density usually.
                
                # update factor: ratio ^ (lr * overlap)
                factor = ratio ** (self.lr * overlap)
                b.count = int(b.count * factor)

# -------------------------
# CDF Training (Isotonic)
# -------------------------

@dataclass
class CDFTrainRow:
    x_norm: float
    y_cdf: float

def collect_cdf_training_rows(buckets: List[Bucket], freq: np.ndarray, mn: int, points_per_bucket: int, rng, bucket_indices: List[int] = None) -> Dict[int, List[CDFTrainRow]]:
    """
    Collects training data. 
    If bucket_indices is provided, ONLY collects for those buckets (Adaptive Retraining).
    """
    ps = np.cumsum(freq)
    
    # If partial update, we only process specific indices
    target_indices = bucket_indices if bucket_indices is not None else range(len(buckets))
    rows = {}
    
    for i in target_indices:
        b = buckets[i]
        rows[i] = [] # Reset or init
        if b.count == 0: continue
        if b.exact_values is not None: continue # Skip training for exact buckets
        
        # We need enough points to learn the curve. 
        # For Isotonic, more points = better steps. let's use 50-100?
        # User requested 10M rows, so we can afford more samples.
        n_samples = max(points_per_bucket, 50) 
        
        xs = rng.integers(b.lo, b.hi + 1, size=n_samples)
        xs = np.sort(xs)
        
        width = b.hi - b.lo + 1
        b_lo_idx = b.lo - mn
        base_cnt = ps[b_lo_idx - 1] if b_lo_idx > 0 else 0
        
        for x in xs:
            x_idx = x - mn
            if x_idx < 0 or x_idx >= len(ps): continue
            
            curr_cnt = ps[x_idx]
            local_cnt = curr_cnt - base_cnt
            y_cdf = local_cnt / b.count
            
            x_norm = (x - b.lo) / width
            rows[i].append(CDFTrainRow(x_norm, y_cdf))
            
    return rows

def train_cdf_models(rows: Dict[int, List[CDFTrainRow]]) -> Tuple[Dict[int, Any], float]:
    models = {}
    train_time = 0.0
    
    for i, rlist in rows.items():
        if not rlist: # No training data for this bucket (e.g., exact_values or empty)
            models[i] = None
            continue
            
        # Isotonic expects 1D arrays
        X = np.array([r.x_norm for r in rlist])
        y = np.array([r.y_cdf for r in rlist])
        
        t0 = time.perf_counter()
        # y_min=0, y_max=1 enforces CDF bounds. increasing=True enforces monotonicity.
        mdl = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds='clip')
        mdl.fit(X, y)
        train_time += (time.perf_counter() - t0)
        models[i] = mdl
        
    return models, train_time

def predict_local_cdf(model, x_norm) -> float:
    if model is None:
        # Uniform assumption: CDF(x) = x (linear growth)
        return max(0.0, min(1.0, x_norm))
    # Isotonic transform returns array
    val = model.transform([x_norm])[0]
    return float(val)

# -------------------------
# Inference & Evaluation
# -------------------------

@dataclass
class RangeQuery:
    low: int
    high: int

def identify_bad_buckets(queries: List[RangeQuery], y_true: np.ndarray, y_pred: np.ndarray, buckets: List[Bucket], threshold_mae: float = 0.05) -> List[int]:
    """
    Identify which buckets are responsible for errors.
    """
    bad_buckets = set()
    errors = np.abs(y_true - y_pred)
    
    print(f"DEBUG: Max Error: {np.max(errors):.6f}, Mean Error: {np.mean(errors):.6f}, Threshold: {threshold_mae}")
    n_violations = np.sum(errors > threshold_mae)
    print(f"DEBUG: Queries exceeding threshold: {n_violations}/{len(queries)}")

    # Map high error queries to buckets
    # If error > threshold, mark all buckets in that query range as "suspect"
    for i, err in enumerate(errors):
        if err > threshold_mae:
            q = queries[i]
            # Find buckets touching this query
            for b_idx, b in enumerate(buckets):
                 if b.hi < q.low: continue
                 if b.lo > q.high: break
                 # Check if this bucket is "dense" (ML), sparse usually robust or needs rebuild
                 if b.exact_values is None: 
                     bad_buckets.add(b_idx)
    
    return list(bad_buckets)

def predict_range_hybrid_cdf(q: RangeQuery, buckets: List[Bucket], models: Dict[int, Any]) -> float:
    
    def get_bucket_overlap_count(b_idx: int, q_lo: int, q_hi: int) -> float:
        b = buckets[b_idx]
        
        # Intersect query range with bucket range
        lo = max(q_lo, b.lo)
        hi = min(q_hi, b.hi)
        if lo > hi: return 0.0
        
        # If Exact Values available ("NDV Strategy")
        if b.exact_values is not None:
             # Sum counts of values in range [lo, hi]
             # This is precise!
             c_sum = 0
             for v, cnt in b.exact_values:
                 if lo <= v <= hi:
                     c_sum += cnt
             return float(c_sum)
        
        # Else use ML/CDF
        w = b.hi - b.lo + 1
        
        # We need F(hi) - F(lo-1) local to bucket
        # local_x for hi:
        x_hi_norm = (hi - b.lo) / w
        cdf_hi = predict_local_cdf(models.get(b_idx), x_hi_norm)
        
        # local_x for lo-1:
        prev = lo - 1
        if prev < b.lo: # If lo-1 is before the bucket start, its cumulative count is 0
            cdf_lo = 0.0
        else:
             x_lo_norm = (prev - b.lo) / w
             cdf_lo = predict_local_cdf(models.get(b_idx), x_lo_norm)
             
        return max(0.0, cdf_hi - cdf_lo) * b.count

    # Iterate through buckets and sum up overlap counts
    total = 0.0
    for i, b in enumerate(buckets):
        # Optimization: skip buckets completely before or after query range
        if b.hi < q.low: continue
        if b.lo > q.high: break
        
        total += get_bucket_overlap_count(i, q.low, q.high)
        
    return total


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

def q_error_vec(y_true, y_pred, eps=1e-9):
    yt = np.maximum(y_true, eps)
    yp = np.maximum(y_pred, eps)
    return np.maximum(yt/yp, yp/yt)

def summarize(y_true, y_pred, name="Model"):
    qe = q_error_vec(y_true, y_pred)
    med = float(np.median(qe))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    print(f"[{name}] Median QErr={med:.4f}, MAE={mae:.6f}")
    return {"QErr_median": med, "MAE": mae}

# -------------------------
# Drift & Adaptive Repair
# -------------------------

def identify_bad_buckets(queries: List[RangeQuery], y_true: np.ndarray, y_pred: np.ndarray, buckets: List[Bucket], threshold_mae: float = 0.05) -> List[int]:
    """
    Identify which buckets are responsible for errors.
    This is heuristic: if a query hits a bucket and has high error, we blame the bucket.
    Simpler: Just finding buckets where local distribution shifted? 
    In real system, we might use "feedback loop". 
    Here: we check individual bucket stats if we could. 
    Let's use the provided queries to blame buckets.
    """
    bad_buckets = set()
    errors = np.abs(y_true - y_pred)
    
    # Map high error queries to buckets
    # If error > threshold, mark all buckets in that query range as "suspect"
    for i, err in enumerate(errors):
        if err > threshold_mae:
            q = queries[i]
            # Find buckets touching this query
            for b_idx, b in enumerate(buckets):
                 if b.hi < q.low: continue
                 if b.lo > q.high: break
                 # Check if this bucket is "dense" (ML), sparse usually robust or needs rebuild
                 if b.exact_values is None: 
                     bad_buckets.add(b_idx)
    
    return list(bad_buckets)

def evaluate_workload(name: str, queries: List[RangeQuery], buckets: List[Bucket], models: Dict[int, Any], ps: np.ndarray, N: int, mn: int):
    y_true, y_hyb = [], []
    for q in queries:
        li, ri = q.low - mn, q.high - mn
        if li < 0: li=0
        ri_bound = len(ps)-1
        if ri > ri_bound: ri = ri_bound
        
        truth = int(ps[ri] - (ps[li-1] if li > 0 else 0))
        y_true.append(truth / N)
        
        est = predict_range_hybrid_cdf(q, buckets, models)
        y_hyb.append(est / N)
        
    return np.array(y_true), np.array(y_hyb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=10_000_000)
    ap.add_argument("--dist", default="zipf")
    ap.add_argument("--drift-rows", type=int, default=2_000_000, help="Rows to insert for drift")
    ap.add_argument("--drift-dist", default="normal", help="Distribution of inserted data")
    ap.add_argument("--eval-n", type=int, default=1000)
    ap.add_argument("--out-dir", default="artifacts_fd_cdf")
    
    args = ap.parse_args()
    rng = np.random.default_rng(42)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    # -----------------------------------------------------
    # Phase 1: Initial Build (10M rows)
    # -----------------------------------------------------
    print(f"\n=== Phase 1: Initial Build ({args.rows} rows, {args.dist}) ===")
    ds_path = Path(args.out_dir) / f"data_initial.csv"
    vals = gen_values(rng, args.dist, args.rows, 0, 200_000)
    save_csv_column(vals, ds_path)
    
    mn, mx, N = scan_min_max_count(ds_path)
    freq, sample = build_frequency_and_sample(ds_path, mn, mx, N, 100_000, 42)
    
    n_bins = freedman_diaconis_bins(sample, mn, mx, N, 2000)
    print(f"FD Suggested Bins: {n_bins}")
    
    
    # 2a. Equi-Width (Standard Baseline)
    t0_hist = time.perf_counter()
    buckets_eq_width = make_equiwidth_buckets(mn, mx, n_bins, freq)
    t_hist_build = time.perf_counter() - t0_hist
    
    # 3. Model Training (CDF - Hybrid "Training")
    print("Training CDF models (Skipping buckets with low NDV)...")
    rows = collect_cdf_training_rows(buckets_eq_width, freq, mn, 20, rng)
    models, t_ml_train = train_cdf_models(rows)
    print(f"Hist Build (Width): {t_hist_build:.4f}s, ML Train: {t_ml_train:.4f}s")
    
    # Count how many buckets used Exact vs ML
    n_exact = sum(1 for b in buckets_eq_width if b.exact_values is not None)
    n_ml = sum(1 for b in buckets_eq_width if b.exact_values is None)
    print(f"Bucket Strategy: {n_exact} Exact (Sparse), {n_ml} ML (Dense)")
    
    # Total hybrid training is Hist + ML
    t_hybrid_train = t_hist_build + t_ml_train
    
    # 4. Evaluation
    print("Generating Evaluation Workload...")
    queries = []
    width = mx - mn
    ps = np.cumsum(freq)
    
    for _ in range(args.eval_n):
        l = rng.integers(mn, mx)
        w = rng.integers(1, max(10, width // 20)) 
        r = min(mx, l + w)
        queries.append(RangeQuery(l, r))
        
    y_true = []
    y_hist_width = []
    y_hybrid = []
    
    # Measure Baseline Inference (Equi-Width)
    t0_base = time.perf_counter()
    for q in queries:
        h = predict_range_histogram_uniform(q, buckets_eq_width)
        y_hist_width.append(h / N)
    t_base_inf = time.perf_counter() - t0_base

    # Measure Hybrid Inference
    t0_hyb = time.perf_counter()
    for q in queries:
        c = predict_range_hybrid_cdf(q, buckets_eq_width, models)
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
    y_hist_width = np.array(y_hist_width)
    y_hybrid = np.array(y_hybrid)
    
    m_hist_w = summarize(y_true, y_hist_width, "Equi-Width (Standard)")
    m_hyb = summarize(y_true, y_hybrid, "Hybrid + NDV Smart")
    
    print("\n--- Results ---")
    print(f"Equi-Width:  Median QErr={m_hist_w['QErr_median']:.4f}, MAE={m_hist_w['MAE']:.6f}, Time={t_base_inf:.4f}s")
    print(f"Hybrid(FD):  Median QErr={m_hyb['QErr_median']:.4f}, MAE={m_hyb['MAE']:.6f}, Time={t_hyb_inf:.4f}s")
    
    summary = {
        "bins": n_bins,
        "hist_width_metrics": m_hist_w,
        "hybrid_metrics": m_hyb
    }
    with open(Path(args.out_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    # -----------------------------------------------------
    # Phase 2: Data Drift (Insert Data)
    # -----------------------------------------------------
    print(f"\n=== Phase 2: Data Drift (Inserting {args.drift_rows} rows of {args.drift_dist}) ===")
    drift_vals = gen_values(rng, args.drift_dist, args.drift_rows, 0, 200_000, shift=50_000)
    save_csv_column(drift_vals, ds_path, mode='a')
    
    # Update Ground Truth Frequencies (for evaluation)
    N_new = N + args.drift_rows
    print("Updating Global Frequency (Ground Truth)...")
    mn_new, mx_new, N_real = scan_min_max_count(ds_path) 
    freq_new, _ = build_frequency_and_sample(ds_path, mn_new, mx_new, N_real, 1000, 42)
    ps_new = np.cumsum(freq_new)
    
    # --- Compare Approaches under Drift ---
    
    # 1. Static Equi-Width (STALE)
    buckets_static = [Bucket(b.lo, b.hi, count=b.count, ndv=b.ndv, exact_values=b.exact_values) for b in buckets_eq_width]
    
    # 2. EquiHist (Online Learning)
    # Starts from initial buckets (Phase 1 state)
    eh_learner = EquiHistLearner(buckets_eq_width, learning_rate=0.5)
    
    # 3. Hybrid (Stale initially, then repair)
    
    print("Evaluating Drift Sequence...")
    y_true_seq = []
    y_static = []
    y_eh = []
    y_hybrid_stale = []
    
    for q in queries:
        li, ri = q.low - mn_new, q.high - mn_new
        if li < 0: li=0
        if ri >= len(ps_new): ri = len(ps_new)-1
        truth = int(ps_new[ri] - (ps_new[li-1] if li > 0 else 0))
        actual_sel = truth / N_real
        y_true_seq.append(actual_sel)
        
        # Static
        est_static = predict_range_histogram_uniform(q, buckets_static) # Uses old counts!
        y_static.append(est_static / N) 
        
        # EquiHist
        est_eh = eh_learner.predict(q)
        y_eh.append(est_eh / N) 
        
        # Update EquiHist
        eh_learner.update(q, float(truth)) 
        
        # Hybrid (Stale buckets + Old Models)
        est_hyb = predict_range_hybrid_cdf(q, buckets_eq_width, models)
        y_hybrid_stale.append(est_hyb / N)
        
    y_true_arr = np.array(y_true_seq)
    
    m_static = summarize(y_true_arr, np.array(y_static), "Static Equi-Width (Stale)")
    m_eh = summarize(y_true_arr, np.array(y_eh), "EquiHist (Online Adaptive)")
    m_hyb = summarize(y_true_arr, np.array(y_hybrid_stale), "Hybrid (Stale)")
    
    # -----------------------------------------------------
    # Phase 3: Adaptive Repair (Hybrid)
    # -----------------------------------------------------
    print(f"\n=== Phase 3: Hybrid Adaptive Repair ===")
    
    # Hybrid updates counts (Cheap)
    for b in buckets_eq_width:
        li = b.lo - mn_new
        ri = b.hi - mn_new
        if li < 0: li=0
        if ri >= len(freq_new): ri = len(freq_new)-1
        b.count = int(ps_new[ri] - (ps_new[li-1] if li > 0 else 0))
        
    # Check error again with updated counts
    y_hyb_counts_only = []
    for q in queries:
        y_hyb_counts_only.append(predict_range_hybrid_cdf(q, buckets_eq_width, models) / N_real)
        
    # Check bad buckets
    bad_indices = identify_bad_buckets(queries, y_true_arr, np.array(y_hyb_counts_only), buckets_eq_width, threshold_mae=0.0001)
    print(f"Identified {len(bad_indices)}/{len(buckets_eq_width)} buckets needing repair.")
    
    t0_repair = time.perf_counter()
    if bad_indices:
        print("Retraining specific buckets...")
        rows_repair = collect_cdf_training_rows(buckets_eq_width, freq_new, mn_new, 50, rng, bucket_indices=bad_indices)
        models_repair, _ = train_cdf_models(rows_repair)
        for k, v in models_repair.items():
            models[k] = v
            
    t_repair = time.perf_counter() - t0_repair
    print(f"Repair Time: {t_repair:.4f}s")
    
    # Final Hybrid Eval
    y_hyb_final = []
    for q in queries:
        y_hyb_final.append(predict_range_hybrid_cdf(q, buckets_eq_width, models) / N_real)
        
    m_hyb_final = summarize(y_true_arr, np.array(y_hyb_final), "Hybrid (Repaired)")
    
    final_summary = {
        "metrics": {
            "static_stale": m_static,
            "equihist_online": m_eh,
            "hybrid_repaired": m_hyb_final
        },
        "timings": {
            "repair": t_repair
        }
    }
    
    with open(Path(args.out_dir) / "drift_summary.json", "w") as f:
        json.dump(final_summary, f, indent=2)

if __name__ == "__main__":
    main()
