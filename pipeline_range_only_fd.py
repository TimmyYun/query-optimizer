#!/usr/bin/env python3
# pipeline_range_only_fd.py
#
# End-to-end RANGE-ONLY pipeline:
# 1) generate one-column dataset (salary) OR use input CSV
# 2) build equi-width histogram
#    - either fixed number of bins (default)
#    - OR Freedman–Diaconis (FD) bin-width to choose bin count
# 3) generate bucket-aware training workload (range queries with partial bucket overlaps)
# 4) compute true selectivities (exact via prefix sums over value frequencies)
# 5) train per-bucket tiny model (Ridge) to predict fraction within bucket (hybrid)
# 6) evaluate:
#       - baseline: plain equi-width histogram with uniform assumption
#       - hybrid: histogram + per-bucket ML
# 7) write artifacts + metrics
#
# Notes:
# - Designed for 1D integer domain CSV: one value per line, no header.
# - Range-only (no equality queries).
#
import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from shutil import copyfile

from joblib import Parallel, delayed


# -------------------------
# Dataset generation
# -------------------------

def clamp_int(x, lo, hi):
    return int(min(max(int(round(x)), lo), hi))


def gen_values(rng: np.random.Generator, dist: str, n: int, lo: int, hi: int) -> np.ndarray:
    """Generate integer values in [lo, hi] with different shapes (incl. kurtosis/heavy tails)."""
    mid = 0.5 * (lo + hi)
    span = max(1, hi - lo)

    if dist == "uniform":
        v = rng.integers(lo, hi + 1, size=n)

    elif dist == "normal":
        # ~99.7% in +/- 3 std => std ~ span/6
        v = rng.normal(loc=mid, scale=span / 6.0, size=n)

    elif dist == "laplace":
        # heavier tails than normal
        v = rng.laplace(loc=mid, scale=span / 10.0, size=n)

    elif dist == "student_t":
        # heavy tails (kurtosis depends on df)
        v = mid + (span / 10.0) * rng.standard_t(df=3.0, size=n)

    elif dist == "lognormal":
        # right-skew + high kurtosis; then clamp
        sigma = 0.8
        mu = math.log(max(mid, 1.0))
        v = np.exp(rng.normal(loc=mu, scale=sigma, size=n))

    elif dist == "pareto":
        # heavy right tail
        alpha = 2.0
        xm = max(lo, 1)
        v = xm * (1.0 + rng.pareto(alpha, size=n))

    elif dist == "zipf":
        # very skewed discrete; then shift into [lo, hi]
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
# Histogram + truth helpers
# -------------------------

@dataclass
class Bucket:
    lo: int
    hi: int


def scan_min_max_count(csv_path: Path, chunksize: int = 1_000_000) -> Tuple[int, int, int]:
    mn, mx, n = None, None, 0
    for ch in pd.read_csv(
        csv_path, header=None, names=["v"], dtype="int64", chunksize=chunksize, engine="c"
    ):
        v = ch["v"].to_numpy()
        n += v.size
        cmin, cmax = int(v.min()), int(v.max())
        mn = cmin if mn is None else min(mn, cmin)
        mx = cmax if mx is None else max(mx, cmax)

    if mn is None or mx is None:
        raise ValueError(f"Empty CSV: {csv_path}")
    return mn, mx, n


def build_frequency_and_sample(
    csv_path: Path,
    mn: int,
    mx: int,
    n_rows: int,
    sample_size: int,
    seed: int,
    chunksize: int = 1_000_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build exact frequency array over [mn, mx] AND collect an (approximately) uniform sample
    for quantile/IQR estimation (used by FD bin width).
    """
    width = mx - mn + 1
    freq = np.zeros(width, dtype=np.int64)

    p = min(1.0, float(sample_size) / float(max(n_rows, 1)))
    rng = np.random.default_rng(seed)

    sampled: List[np.ndarray] = []
    sampled_n = 0
    soft_cap = max(sample_size * 2, 1000)

    for ch in pd.read_csv(
        csv_path, header=None, names=["v"], dtype="int64", chunksize=chunksize, engine="c"
    ):
        vals = ch["v"].to_numpy()
        idx = vals - mn
        m = (idx >= 0) & (idx < width)
        if not np.all(m):
            idx = idx[m]
            vals = vals[m]

        if idx.size:
            freq += np.bincount(idx, minlength=width)

        if p > 0.0 and vals.size:
            keep = rng.random(vals.size) < p
            if np.any(keep):
                s = vals[keep]
                sampled.append(s)
                sampled_n += s.size

                if sampled_n > soft_cap:
                    big = np.concatenate(sampled) if sampled else np.array([], dtype=np.int64)
                    if big.size > sample_size:
                        take = rng.choice(big.size, size=sample_size, replace=False)
                        big = big[take]
                    sampled = [big]
                    sampled_n = big.size

    sample = np.concatenate(sampled) if sampled else np.array([], dtype=np.int64)
    if sample.size > sample_size:
        take = rng.choice(sample.size, size=sample_size, replace=False)
        sample = sample[take]
    return freq, sample


def count_range(prefix_sum: np.ndarray, mn: int, lo: int, hi: int) -> int:
    lo_i = max(0, lo - mn)
    hi_i = min(prefix_sum.size - 1, hi - mn)
    if hi_i < 0 or lo_i > prefix_sum.size - 1:
        return 0
    if lo_i == 0:
        return int(prefix_sum[hi_i])
    return int(prefix_sum[hi_i] - prefix_sum[lo_i - 1])


def make_equiwidth_buckets(mn: int, mx: int, bins: int) -> List[Bucket]:
    width = mx - mn + 1
    bw = max(1, int(math.ceil(width / bins)))
    buckets: List[Bucket] = []
    cur = mn
    for _ in range(bins):
        lo = cur
        hi = min(mx, lo + bw - 1)
        buckets.append(Bucket(lo, hi))
        cur = hi + 1
        if cur > mx:
            break
    return buckets


def freedman_diaconis_bins(
    sample: np.ndarray,
    mn: int,
    mx: int,
    n_rows: int,
    bins_max: int,
    bins_min: int = 1,
) -> Tuple[int, float, float, int]:
    """
    Compute FD bin width and suggested number of bins.
    Returns: (bins_used, iqr, bw_int, bins_suggested_unclamped)
    """
    if sample.size < 10:
        return 1, 0.0, 0.0, 1

    q25, q75 = np.quantile(sample.astype(float), [0.25, 0.75])
    iqr = float(q75 - q25)

    n = float(max(n_rows, 1))
    h = 2.0 * iqr / (n ** (1.0 / 3.0))

    if not np.isfinite(h) or h <= 0.0:
        return 1, iqr, float(h), 1

    h_int = max(1, int(math.ceil(h)))
    domain_w = (mx - mn + 1)
    bins_suggested = int(math.ceil(domain_w / h_int))
    bins_used = int(min(max(bins_suggested, bins_min), bins_max))
    return bins_used, iqr, float(h_int), bins_suggested


def bucket_index_for_value(buckets: List[Bucket], v: int) -> int:
    for i, b in enumerate(buckets):
        if b.lo <= v <= b.hi:
            return i
    return -1


def precompute_bucket_counts(buckets: List[Bucket], freq: np.ndarray, mn: int) -> np.ndarray:
    bc = np.zeros(len(buckets), dtype=np.int64)
    ps = np.cumsum(freq)
    for i, b in enumerate(buckets):
        lo_idx = b.lo - mn
        hi_idx = b.hi - mn
        bc[i] = int(ps[hi_idx] - (ps[lo_idx - 1] if lo_idx > 0 else 0))
    return bc


# -------------------------
# Workload (range-only) – bucket-aware
# -------------------------

@dataclass
class RangeQuery:
    low: int
    high: int


def gen_partial_overlaps_for_bucket(
    rng: np.random.Generator, b: Bucket, mn: int, mx: int, k_each_side: int = 3
) -> List[RangeQuery]:
    """
    Generate partial-overlap queries for a given bucket (left/right tails).
    These are used to train local models inside buckets.
    """
    out: List[RangeQuery] = []
    w = max(1, b.hi - b.lo + 1)

    # Left-side partial overlaps
    for _ in range(k_each_side):
        frac = rng.beta(2.0, 5.0)
        ov_w = max(1, int(frac * w))
        start = b.lo + rng.integers(0, max(1, w // 4))
        end = min(b.hi, start + ov_w - 1)
        left_pad = rng.integers(1, max(2, w // 4))
        low = max(mn, start - left_pad)
        high = end
        if low <= high and not (low <= b.lo and high >= b.hi):
            out.append(RangeQuery(low, high))

    # Right-side partial overlaps
    for _ in range(k_each_side):
        frac = rng.beta(2.0, 5.0)
        ov_w = max(1, int(frac * w))
        end = b.hi - rng.integers(0, max(1, w // 4))
        start = max(b.lo, end - ov_w + 1)
        right_pad = rng.integers(1, max(2, w // 4))
        low = start
        high = min(mx, end + right_pad)
        if low <= high and not (low <= b.lo and high >= b.hi):
            out.append(RangeQuery(low, high))

    return out


def gen_eval_random_ranges(rng: np.random.Generator, n_eval: int, mn: int, mx: int) -> List[RangeQuery]:
    """
    Evaluation workload: random ranges with varying widths around random centers.
    """
    out: List[RangeQuery] = []
    span = mx - mn
    for _ in range(n_eval):
        c = int(rng.integers(mn, mx + 1))
        half_frac = float(rng.beta(2.0, 5.0))
        half_w = max(1, int(half_frac * (span / 4)))
        a = max(mn, c - half_w)
        b = min(mx, c + half_w)
        if a > b:
            a, b = b, a
        if a == b and b < mx:
            b = a + 1
        out.append(RangeQuery(a, b))
    return out


# -------------------------
# Bucket training rows
# -------------------------

@dataclass
class BucketTrainRow:
    cov_ratio: float
    start_norm: float
    end_norm: float
    center_norm: float
    y_frac: float


def collect_bucket_training_rows(
    buckets: List[Bucket], freq: np.ndarray, mn: int, queries: List[RangeQuery]
) -> Dict[int, List[BucketTrainRow]]:
    """
    For each bucket, collect training rows (only for partially-overlapping ranges).
    """
    ps = np.cumsum(freq)
    rows: Dict[int, List[BucketTrainRow]] = {i: [] for i in range(len(buckets))}

    for q in queries:
        lo, hi = q.low, q.high
        li = bucket_index_for_value(buckets, lo)
        ri = bucket_index_for_value(buckets, hi)

        if li >= 0:
            b = buckets[li]
            ov_lo = max(lo, b.lo)
            ov_hi = min(hi, b.hi)
            if ov_lo <= ov_hi and not (lo <= b.lo and hi >= b.hi):
                w = max(1, b.hi - b.lo + 1)
                cov = (ov_hi - ov_lo + 1) / w
                start = (ov_lo - b.lo) / w
                end = (ov_hi - b.lo) / w
                center = ((ov_lo + ov_hi) / 2 - b.lo) / w
                b_lo_i = b.lo - mn
                b_hi_i = b.hi - mn
                b_cnt = int(ps[b_hi_i] - (ps[b_lo_i - 1] if b_lo_i > 0 else 0))
                if b_cnt > 0:
                    ycnt = count_range(ps, mn, ov_lo, ov_hi)
                    rows[li].append(BucketTrainRow(cov, start, end, center, ycnt / b_cnt))

        if ri >= 0 and ri != li:
            b = buckets[ri]
            ov_lo = max(lo, b.lo)
            ov_hi = min(hi, b.hi)
            if ov_lo <= ov_hi and not (lo <= b.lo and hi >= b.hi):
                w = max(1, b.hi - b.lo + 1)
                cov = (ov_hi - ov_lo + 1) / w
                start = (ov_lo - b.lo) / w
                end = (ov_hi - b.lo) / w
                center = ((ov_lo + ov_hi) / 2 - b.lo) / w
                b_lo_i = b.lo - mn
                b_hi_i = b.hi - mn
                b_cnt = int(ps[b_hi_i] - (ps[b_lo_i - 1] if b_lo_i > 0 else 0))
                if b_cnt > 0:
                    ycnt = count_range(ps, mn, ov_lo, ov_hi)
                    rows[ri].append(BucketTrainRow(cov, start, end, center, ycnt / b_cnt))

    return rows


# -------------------------
# Per-bucket model (Ridge) + hybrid inference
# -------------------------

def _train_single_bucket(bi: int, rlist: List[BucketTrainRow], min_samples: int):
    """
    Helper for parallel training of a single bucket model.
    Returns (bucket_index, (kind, model), train_time_sec)
    kind: "uniform" or "ridge"
    """
    from sklearn.linear_model import Ridge

    if len(rlist) < min_samples:
        return bi, ("uniform", None), 0.0

    X = np.array([[r.cov_ratio, r.start_norm, r.end_norm, r.center_norm] for r in rlist], dtype=float)
    y = np.array([r.y_frac for r in rlist], dtype=float)

    mdl = Ridge(alpha=1e-4, random_state=42)
    t0 = time.perf_counter()
    mdl.fit(X, y)
    t1 = time.perf_counter()

    return bi, ("ridge", mdl), (t1 - t0)


def train_bucket_models(rows: Dict[int, List[BucketTrainRow]], min_samples: int = 3):
    """
    Train a tiny regression model per bucket in parallel (joblib).
    Buckets with insufficient data fall back to uniform assumption.
    """
    results = Parallel(n_jobs=-1)(
        delayed(_train_single_bucket)(bi, rlist, min_samples)
        for bi, rlist in rows.items()
    )

    models: Dict[int, Tuple[str, object]] = {}
    times: Dict[int, float] = {}
    for bi, model_info, t in results:
        models[bi] = model_info
        times[bi] = t

    return models, times


def bucket_predict(models: Dict[int, Tuple[str, object]], bi: int,
                   cov_ratio: float, start_norm: float, end_norm: float, center_norm: float) -> float:
    """
    Predict fraction of bucket mass covered by overlap using bucket model (if present),
    otherwise uniform assumption.
    """
    kind, mdl = models.get(bi, ("uniform", None))
    if kind == "uniform" or mdl is None:
        # Plain uniform heuristic
        return float(cov_ratio)
    X = np.array([[cov_ratio, start_norm, end_norm, center_norm]], dtype=float)
    y = float(mdl.predict(X)[0])
    return max(0.0, min(1.0, y))


def predict_range_hybrid(q: RangeQuery, buckets: List[Bucket], models, bc: np.ndarray) -> float:
    """
    Hybrid: equal-width histogram + per-bucket ML (for partially-covered edge buckets).
    """
    total = 0.0
    li = bucket_index_for_value(buckets, q.low)
    ri = bucket_index_for_value(buckets, q.high)

    # Left bucket
    if li >= 0:
        b = buckets[li]
        w = max(1, b.hi - b.lo + 1)
        ov_lo = max(q.low, b.lo)
        ov_hi = min(q.high, b.hi)
        if ov_lo <= ov_hi:
            if not (q.low <= b.lo and q.high >= b.hi):
                cov = (ov_hi - ov_lo + 1) / w
                start = (ov_lo - b.lo) / w
                end = (ov_hi - b.lo) / w
                center = ((ov_lo + ov_hi) / 2 - b.lo) / w
                frac = bucket_predict(models, li, cov, start, end, center)
                total += frac * bc[li]
            else:
                total += bc[li]

    # Full middle buckets
    if li >= 0 and ri >= 0:
        a = li + 1
        b = ri - 1
        if b >= a:
            total += float(bc[a: b + 1].sum())

    # Right bucket
    if ri >= 0 and ri != li:
        b = buckets[ri]
        w = max(1, b.hi - b.lo + 1)
        ov_lo = max(q.low, b.lo)
        ov_hi = min(q.high, b.hi)
        if ov_lo <= ov_hi:
            if not (q.low <= b.lo and q.high >= b.hi):
                cov = (ov_hi - ov_lo + 1) / w
                start = (ov_lo - b.lo) / w
                end = (ov_hi - b.lo) / w
                center = ((ov_lo + ov_hi) / 2 - b.lo) / w
                frac = bucket_predict(models, ri, cov, start, end, center)
                total += frac * bc[ri]
            else:
                total += bc[ri]

    return total


def predict_range_histogram_uniform(q: RangeQuery, buckets: List[Bucket], bc: np.ndarray) -> float:
    """
    Baseline: plain equi-width histogram with uniform assumption inside buckets.
    Same structure as hybrid but with frac = coverage ratio (no ML).
    """
    total = 0.0
    li = bucket_index_for_value(buckets, q.low)
    ri = bucket_index_for_value(buckets, q.high)

    # Left bucket
    if li >= 0:
        b = buckets[li]
        w = max(1, b.hi - b.lo + 1)
        ov_lo = max(q.low, b.lo)
        ov_hi = min(q.high, b.hi)
        if ov_lo <= ov_hi:
            cov = (ov_hi - ov_lo + 1) / w
            total += cov * bc[li]

    # Full middle buckets
    if li >= 0 and ri >= 0:
        a = li + 1
        b = ri - 1
        if b >= a:
            total += float(bc[a: b + 1].sum())

    # Right bucket
    if ri >= 0 and ri != li:
        b = buckets[ri]
        w = max(1, b.hi - b.lo + 1)
        ov_lo = max(q.low, b.lo)
        ov_hi = min(q.high, b.hi)
        if ov_lo <= ov_hi:
            cov = (ov_hi - ov_lo + 1) / w
            total += cov * bc[ri]

    return total


# -------------------------
# Metrics
# -------------------------

def q_error_vec(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    yt = np.maximum(y_true, eps)
    yp = np.maximum(y_pred, eps)
    return np.maximum(yt / yp, yp / yt)


def summarize(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    qe = q_error_vec(y_true, y_pred)
    return {
        "QErr_median": float(np.median(qe)),
        "QErr_p95": float(np.percentile(qe, 95)),
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
    }


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Range-only pipeline with FD/fixed bins, histogram baseline + hybrid")
    ap.add_argument("--rows", type=int, default=5_000_000)
    ap.add_argument(
        "--dist",
        default="normal",
        choices=["uniform", "normal", "laplace", "student_t", "lognormal", "pareto", "zipf"],
    )
    ap.add_argument("--vmin", type=int, default=30_000)
    ap.add_argument("--vmax", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--binning", choices=["fixed", "fd"], default="fixed")
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--bins-max", type=int, default=1000)
    ap.add_argument("--bins-min", type=int, default=10)
    ap.add_argument("--fd-sample", type=int, default=1_000_000)

    ap.add_argument("--samples-per-bucket", type=int, default=3)
    ap.add_argument("--max-train-queries", type=int, default=50000)
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--out-root", default="artifacts_range_only_fd")

    ap.add_argument("--input-csv", default=None,
                    help="Path to 1-col int CSV (no header). If set, skip synthetic generation.")

    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Dataset
    if args.input_csv:
        ds_path = Path(args.input_csv).resolve()
        ds_name = ds_path.name
        print(f"[1/7] using input dataset -> {ds_path}")
    else:
        ds_name = f"salary_{args.dist}_{args.rows}.csv"
        ds_path = out_root / "data" / ds_name
        print(f"[1/7] dataset -> {ds_path}")
        values = gen_values(rng, args.dist, args.rows, args.vmin, args.vmax)
        save_csv_column(values, ds_path)

    # 2) Histogram & freq (+ FD sample)
    print("[2/7] histogram & freq (+ FD sample)")
    t0 = time.perf_counter()
    vmin, vmax, N = scan_min_max_count(ds_path)
    freq, sample = build_frequency_and_sample(
        ds_path, vmin, vmax, N, sample_size=args.fd_sample, seed=args.seed
    )

    fd_info = {
        "fd_iqr": None,
        "fd_bin_width": None,
        "fd_bins_suggested": None,
        "fd_bins_used": None,
        "fd_sample_used": int(sample.size),
    }

    if args.binning == "fd":
        bins_used, iqr, bw, bins_suggested = freedman_diaconis_bins(
            sample=sample, mn=vmin, mx=vmax, n_rows=N, bins_max=args.bins_max, bins_min=args.bins_min
        )
        bins = bins_used
        fd_info.update({
            "fd_iqr": float(iqr),
            "fd_bin_width": float(bw),
            "fd_bins_suggested": int(bins_suggested),
            "fd_bins_used": int(bins_used),
        })
        print(f"  FD: IQR={fd_info['fd_iqr']:.3f}, bin_width≈{fd_info['fd_bin_width']}, "
              f"bins_suggested={fd_info['fd_bins_suggested']}, bins_used={bins}")
    else:
        bins = int(args.bins)

    buckets = make_equiwidth_buckets(vmin, vmax, bins)
    bc = precompute_bucket_counts(buckets, freq, vmin)
    hist_sec = time.perf_counter() - t0

    # 3) Training workload (bucket-aware)
    print("[3/7] bucket-aware training workload")
    train_ranges: List[RangeQuery] = []
    for b in buckets:
        train_ranges += gen_partial_overlaps_for_bucket(
            rng, b, vmin, vmax, k_each_side=args.samples_per_bucket
        )
    if len(train_ranges) > args.max_train_queries:
        idx = rng.choice(len(train_ranges), size=args.max_train_queries, replace=False)
        train_ranges = [train_ranges[i] for i in idx]

    # 4) Evaluation workload
    print("[4/7] eval workload random")
    eval_ranges = gen_eval_random_ranges(rng, args.eval_n, vmin, vmax)

    # 5) True selectivities
    print("[5/7] truth selectivities")
    ps = np.cumsum(freq)

    def true_sel(q: RangeQuery) -> float:
        return count_range(ps, vmin, q.low, q.high) / float(N if N > 0 else 1.0)

    y_train = np.array([true_sel(q) for q in train_ranges], dtype=float)
    y_eval = np.array([true_sel(q) for q in eval_ranges], dtype=float)

    # 6) Train per-bucket models (parallel)
    print("[6/7] train per-bucket models (parallel)")
    bucket_rows = collect_bucket_training_rows(buckets, freq, vmin, train_ranges)
    models, train_times = train_bucket_models(bucket_rows, min_samples=3)
    bucket_total_train_sec = float(sum(train_times.values()))
    bucket_max_train_sec = float(max(train_times.values()) if train_times else 0.0)
    bucket_models_trained = int(sum(1 for _, t in train_times.items() if t > 0))
    bucket_models_uniform = len(buckets) - bucket_models_trained

    # 7) Evaluate baseline (histogram) and hybrid
    print("[7/7] eval baseline histogram + hybrid")

    # baseline: pure equi-width histogram with uniform assumption
    t0 = time.perf_counter()
    baseline_counts = [predict_range_histogram_uniform(q, buckets, bc) for q in eval_ranges]
    baseline_infer_sec = time.perf_counter() - t0
    y_base = np.array(baseline_counts, dtype=float) / float(N if N > 0 else 1.0)

    # hybrid: histogram + per-bucket models on edge buckets
    t0 = time.perf_counter()
    hybrid_counts = [predict_range_hybrid(q, buckets, models, bc) for q in eval_ranges]
    hybrid_infer_sec = time.perf_counter() - t0
    y_hybrid = np.array(hybrid_counts, dtype=float) / float(N if N > 0 else 1.0)

    met_h = summarize(y_eval, y_hybrid)
    met_b = summarize(y_eval, y_base)

    # Save artifacts
    res_dir = out_root / "results" / Path(ds_name).stem
    res_dir.mkdir(parents=True, exist_ok=True)

    perq = pd.DataFrame(
        {
            "low": [q.low for q in eval_ranges],
            "high": [q.high for q in eval_ranges],
            "true_selectivity": y_eval,
            "baseline_hist_pred": y_base,
            "baseline_hist_qerror": q_error_vec(y_eval, y_base),
            "hybrid_pred": y_hybrid,
            "hybrid_qerror": q_error_vec(y_eval, y_hybrid),
        }
    )
    perq.to_csv(res_dir / "per_query_eval.csv", index=False)

    cov_stats = []
    for i, b in enumerate(buckets):
        rs = bucket_rows.get(i, [])
        cov_stats.append(
            {
                "bucket": i,
                "lo": b.lo,
                "hi": b.hi,
                "width": b.hi - b.lo + 1,
                "n_rows": len(rs),
                "used_model": "ridge" if (i in models and models[i][0] == "ridge") else "uniform",
            }
        )
    pd.DataFrame(cov_stats).to_csv(res_dir / "bucket_training_coverage.csv", index=False)

    summary = {
        "dataset": ds_name,
        "rows": int(N),
        "dist": args.dist,
        "binning": args.binning,
        "bins": len(buckets),
        "hist_build_sec": hist_sec,
        "train_queries": int(len(train_ranges)),
        "eval_queries": int(len(eval_ranges)),

        "bucket_models_trained": bucket_models_trained,
        "bucket_models_fallback_uniform": bucket_models_uniform,
        "bucket_total_train_sec": bucket_total_train_sec,
        "bucket_max_train_sec": bucket_max_train_sec,

        # baseline = plain equi-width histogram (uniform)
        "baseline_model": "EquiWidthUniform",
        "baseline_infer_sec_total": baseline_infer_sec,
        "baseline_infer_ms_per_query": float(1000.0 * baseline_infer_sec / max(len(eval_ranges), 1)),
        "baseline_QErr_median": met_b["QErr_median"],
        "baseline_QErr_p95": met_b["QErr_p95"],
        "baseline_MAE": met_b["MAE"],
        "baseline_RMSE": met_b["RMSE"],

        # hybrid = histogram + per-bucket ML
        "hybrid_infer_sec_total": hybrid_infer_sec,
        "hybrid_infer_ms_per_query": float(1000.0 * hybrid_infer_sec / max(len(eval_ranges), 1)),
        "hybrid_QErr_median": met_h["QErr_median"],
        "hybrid_QErr_p95": met_h["QErr_p95"],
        "hybrid_MAE": met_h["MAE"],
        "hybrid_RMSE": met_h["RMSE"],

        **fd_info,
    }
    with open(res_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts written to: {res_dir}")


if __name__ == "__main__":
    main()
