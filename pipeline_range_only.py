#!/usr/bin/env python3
# This is a single end-to-end pipeline (range-only).
import argparse, json, math, time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

# -------------------------
# Dataset generation
# -------------------------

def clamp_int(x, lo, hi):
    return int(min(max(int(round(x)), lo), hi))

def gen_values(rng, dist: str, n: int, lo: int, hi: int) -> np.ndarray:
    mid = 0.5 * (lo + hi)
    span = max(1, hi - lo)
    if dist == "uniform":
        v = rng.integers(lo, hi + 1, size=n)
    elif dist == "normal":
        v = rng.normal(loc=mid, scale=span / 6.0, size=n)
    elif dist == "laplace":
        v = rng.laplace(loc=mid, scale=span / 10.0, size=n)
    elif dist == "student_t":
        v = mid + (span / 10.0) * rng.standard_t(df=3.0, size=n)
    elif dist == "lognormal":
        sigma = 0.8
        mu = math.log(max(mid, 1.0))
        v = np.exp(rng.normal(loc=mu, scale=sigma, size=n))
    elif dist == "pareto":
        alpha = 2.0
        xm = max(lo, 1)
        v = xm * (1.0 + rng.pareto(alpha, size=n))
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
# Histogram + truth helpers
# -------------------------

@dataclass
class Bucket:
    lo: int
    hi: int

def scan_min_max_count(csv_path: Path, chunksize=1_000_000) -> Tuple[int,int,int]:
    """
    Scans a CSV file to compute the minimum value, maximum value, and total count of
    values. It processes the file in chunks to handle large datasets efficiently.

    :param csv_path: Path to the CSV file to be scanned.
    :type csv_path: Path
    :param chunksize: Number of rows per chunk to process in memory. Defaults to 1,000,000.
    :type chunksize: int, optional
    :return: A tuple containing the minimum value, maximum value, and the total count
             of elements across the entire file.
    :rtype: Tuple[int, int, int]
    :raises ValueError: If the CSV file is empty or does not contain valid data.
    """
    mn, mx, n = None, None, 0
    for ch in pd.read_csv(csv_path, header=None, names=["v"], dtype="int64",
                          chunksize=chunksize, engine="c"):
        v = ch["v"].to_numpy()
        n += v.size
        cmin, cmax = int(v.min()), int(v.max())
        mn = cmin if mn is None else min(mn, cmin)
        mx = cmax if mx is None else max(mx, cmax)
    if mn is None or mx is None:
        raise ValueError(f"Empty CSV: {csv_path}")
    return mn, mx, n

def build_frequency(csv_path: Path, mn: int, mx: int, chunksize=1_000_000) -> np.ndarray:
    """
    Builds a frequency array from a CSV file where numerical values are stored.

    This function reads data from a CSV file in chunks to efficiently calculate the frequency
    distribution of integers within the given range. It ensures values are mapped correctly based on
    the minimum (`mn`) and maximum (`mx`) range, and it will ignore any values outside this range.
    The function operates on large datasets by processing the input file incrementally using chunks.

    :param csv_path: Path to the CSV file containing the numerical data. Each value should be
        represented in a single column.
    :type csv_path: Path
    :param mn: The minimum integer value (inclusive) for the frequency computation range.
    :type mn: int
    :param mx: The maximum integer value (inclusive) for the frequency computation range.
    :type mx: int
    :param chunksize: The size of each chunk to process from the CSV file. Defaults to 1,000,000.
    :type chunksize: int, optional
    :return: A NumPy array containing the frequency distribution of integers in the range [mn, mx].
        Each index corresponds to a value in this range, and the array value represents the frequency
        of that integer.
    :rtype: np.ndarray
    """
    width = mx - mn + 1
    freq = np.zeros(width, dtype=np.int64)
    for ch in pd.read_csv(csv_path, header=None, names=["v"], dtype="int64",
                          chunksize=chunksize, engine="c"):
        vals = ch["v"].to_numpy()
        idx = vals - mn
        m = (idx>=0) & (idx<width)
        if not np.all(m):
            idx = idx[m]
        if idx.size:
            bc = np.bincount(idx, minlength=width)
            freq += bc
    return freq

def count_range(psum: np.ndarray, mn: int, lo: int, hi: int) -> int:
    """
    Calculate the count of elements in a specified range within a prefix sum array.

    This function determines the count of elements that fall within the range
    [lo, hi] based on a provided prefix sum array `psum` and an offset value `mn`.
    It ensures the range boundaries are adjusted correctly and returns the result.

    :param psum: A ndarray representing the prefix sum array.
    :param mn: An offset value to adjust the boundaries of the range.
    :param lo: The lower bound of the range.
    :param hi: The upper bound of the range.
    :return: The count of elements within the specified range.
    :rtype: int
    """
    lo_i = max(0, lo - mn)
    hi_i = min(psum.size - 1, hi - mn)
    if hi_i < 0 or lo_i > psum.size - 1:
        return 0
    if lo_i == 0:
        return int(psum[hi_i])
    return int(psum[hi_i] - psum[lo_i - 1])

def make_equiwidth_buckets(mn: int, mx: int, bins: int) -> List['Bucket']:
    """
    Creates a list of equi-width buckets between a specified minimum and maximum range.

    The function takes in a specified minimum value (`mn`), maximum value (`mx`),
    and number of desired buckets (`bins`). It computes the bucket width and
    creates buckets of the calculated width, ensuring that the ranges do not exceed
    the maximum value provided. Each bucket is represented by a `Bucket` object.

    :param mn: The minimum value of the range used to create buckets.
    :type mn: int
    :param mx: The maximum value of the range used to create buckets.
    :type mx: int
    :param bins: The number of buckets to be created.
    :type bins: int
    :return: A list of `Bucket` objects representing the equi-width buckets.
    :rtype: List[Bucket]
    """
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

def bucket_index_for_value(buckets: List['Bucket'], v: int) -> int:
    """
    Determine the index of a bucket that contains the specified value.

    This function iterates over a list of `Bucket` objects to find
    the index of the first bucket whose range includes the given
    value. If no such bucket is found, the function returns -1.

    :param buckets: List of `Bucket` objects to search. Each bucket
        has a range defined by a lower bound (`lo`) and an upper
        bound (`hi`).
    :type buckets: List[Bucket]
    :param v: The value for which the bucket index is determined.
    :type v: int
    :return: The index of the bucket containing the value `v`,
        or -1 if no such bucket exists.
    :rtype: int
    """
    for i,b in enumerate(buckets):
        if b.lo <= v <= b.hi: return i
    return -1

def precompute_bucket_counts(buckets: List['Bucket'], freq: np.ndarray, mn: int) -> np.ndarray:
    bc = np.zeros(len(buckets), dtype=np.int64)
    ps = np.cumsum(freq)
    for i,b in enumerate(buckets):
        lo_idx = b.lo - mn
        hi_idx = b.hi - mn
        bc[i] = int(ps[hi_idx] - (ps[lo_idx-1] if lo_idx>0 else 0))
    return bc

# -------------------------
# Workload (range-only) – bucket-aware
# -------------------------

@dataclass
class RangeQuery:
    low: int
    high: int

def gen_partial_overlaps_for_bucket(rng, b: Bucket, mn: int, mx: int, k_each_side: int=3) -> List[RangeQuery]:
    out = []
    w = max(1, b.hi - b.lo + 1)
    for _ in range(k_each_side):
        frac = rng.beta(2.0, 5.0)
        ov_w = max(1, int(frac * w))
        start = b.lo + rng.integers(0, max(1, w//4))
        end = min(b.hi, start + ov_w - 1)
        left_pad = rng.integers(1, max(2, w//4))
        low = max(mn, start - left_pad)
        high = end
        if low <= high and not (low <= b.lo and high >= b.hi):
            out.append(RangeQuery(low, high))
    for _ in range(k_each_side):
        frac = rng.beta(2.0, 5.0)
        ov_w = max(1, int(frac * w))
        end = b.hi - rng.integers(0, max(1, w//4))
        start = max(b.lo, end - ov_w + 1)
        right_pad = rng.integers(1, max(2, w//4))
        low = start
        high = min(mx, end + right_pad)
        if low <= high and not (low <= b.lo and high >= b.hi):
            out.append(RangeQuery(low, high))
    return out

def gen_eval_random_ranges(rng, n_eval: int, mn: int, mx: int) -> List[RangeQuery]:
    out = []
    span = mx - mn
    for _ in range(n_eval):
        c = int(rng.integers(mn, mx + 1))
        half_frac = float(rng.beta(2.0, 5.0))
        half_w = max(1, int(half_frac * (span/4)))
        a = max(mn, c - half_w)
        b = min(mx, c + half_w)
        if a>b: a,b=b,a
        if a==b and b<mx: b=a+1
        out.append(RangeQuery(a,b))
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

def collect_bucket_training_rows(buckets: List[Bucket], freq: np.ndarray, mn: int,
                                 queries: List[RangeQuery]) -> Dict[int, List[BucketTrainRow]]:
    ps = np.cumsum(freq)
    rows: Dict[int, List[BucketTrainRow]] = {i: [] for i in range(len(buckets))}

    for q in queries:
        lo, hi = q.low, q.high
        li = bucket_index_for_value(buckets, lo)
        ri = bucket_index_for_value(buckets, hi)
        if li >= 0:
            b = buckets[li]
            ov_lo = max(lo, b.lo); ov_hi = min(hi, b.hi)
            if ov_lo <= ov_hi and not (lo <= b.lo and hi >= b.hi):
                w = max(1, b.hi - b.lo + 1)
                cov = (ov_hi - ov_lo + 1)/w
                start = (ov_lo - b.lo)/w
                end = (ov_hi - b.lo)/w
                center = ((ov_lo+ov_hi)/2 - b.lo)/w
                b_lo_i = b.lo - mn; b_hi_i = b.hi - mn
                b_cnt = int(ps[b_hi_i] - (ps[b_lo_i-1] if b_lo_i>0 else 0))
                if b_cnt > 0:
                    ycnt = count_range(ps, mn, ov_lo, ov_hi)
                    yfrac = ycnt / b_cnt
                    rows[li].append(BucketTrainRow(cov, start, end, center, yfrac))
        if ri >= 0 and ri != li:
            b = buckets[ri]
            ov_lo = max(lo, b.lo); ov_hi = min(hi, b.hi)
            if ov_lo <= ov_hi and not (lo <= b.lo and hi >= b.hi):
                w = max(1, b.hi - b.lo + 1)
                cov = (ov_hi - ov_lo + 1)/w
                start = (ov_lo - b.lo)/w
                end = (ov_hi - b.lo)/w
                center = ((ov_lo+ov_hi)/2 - b.lo)/w
                b_lo_i = b.lo - mn; b_hi_i = b.hi - mn
                b_cnt = int(ps[b_hi_i] - (ps[b_lo_i-1] if b_lo_i>0 else 0))
                if b_cnt > 0:
                    ycnt = count_range(ps, mn, ov_lo, ov_hi)
                    yfrac = ycnt / b_cnt
                    rows[ri].append(BucketTrainRow(cov, start, end, center, yfrac))
    return rows

# -------------------------
# Per-bucket model (Ridge)
# -------------------------

def train_bucket_models(rows: Dict[int, List[BucketTrainRow]], min_samples=3):
    from sklearn.linear_model import Ridge
    models = {}
    times = {}
    for bi, rlist in rows.items():
        if len(rlist) < min_samples:
            models[bi] = ("uniform", None)
            times[bi] = 0.0
            continue
        X = np.array([[r.cov_ratio, r.start_norm, r.end_norm, r.center_norm] for r in rlist], dtype=float)
        y = np.array([r.y_frac for r in rlist], dtype=float)
        mdl = Ridge(alpha=1e-4, random_state=42)
        t0 = time.perf_counter(); mdl.fit(X, y); t1 = time.perf_counter()
        models[bi] = ("ridge", mdl)
        times[bi] = t1 - t0
    return models, times

def bucket_predict(models, bi: int, cov_ratio, start_norm, end_norm, center_norm):
    kind, mdl = models.get(bi, ("uniform", None))
    if kind == "uniform" or mdl is None:
        return float(cov_ratio)
    X = np.array([[cov_ratio, start_norm, end_norm, center_norm]], dtype=float)
    y = float(mdl.predict(X)[0])
    return max(0.0, min(1.0, y))

# -------------------------
# Hybrid prediction (range only)
# -------------------------

def predict_range_hybrid(q, buckets, models, bc, mn):
    total = 0.0
    li = bucket_index_for_value(buckets, q.low)
    ri = bucket_index_for_value(buckets, q.high)

    if li>=0:
        b = buckets[li]
        w = max(1, b.hi - b.lo + 1)
        ov_lo = max(q.low, b.lo); ov_hi = min(q.high, b.hi)
        if ov_lo <= ov_hi:
            if not (q.low <= b.lo and q.high >= b.hi):
                cov = (ov_hi - ov_lo + 1)/w
                start = (ov_lo - b.lo)/w
                end = (ov_hi - b.lo)/w
                center = ((ov_lo+ov_hi)/2 - b.lo)/w
                frac = bucket_predict(models, li, cov, start, end, center)
                total += frac * bc[li]
            else:
                total += bc[li]

    if li>=0 and ri>=0:
        a = li+1; b = ri-1
        if b >= a:
            total += float(bc[a:b+1].sum())

    if ri>=0 and ri!=li:
        b = buckets[ri]
        w = max(1, b.hi - b.lo + 1)
        ov_lo = max(q.low, b.lo); ov_hi = min(q.high, b.hi)
        if ov_lo <= ov_hi:
            if not (q.low <= b.lo and q.high >= b.hi):
                cov = (ov_hi - ov_lo + 1)/w
                start = (ov_lo - b.lo)/w
                end = (ov_hi - b.lo)/w
                center = ((ov_lo+ov_hi)/2 - b.lo)/w
                frac = bucket_predict(models, ri, cov, start, end, center)
                total += frac * bc[ri]
            else:
                total += bc[ri]
    return total

# -------------------------
# Baseline model (ExtraTrees; range-only features)
# -------------------------

def build_baseline_features(ranges, vmin, vmax):
    dom = max(vmax - vmin, 1)
    lo = np.array([r.low for r in ranges], dtype=float)
    hi = np.array([r.high for r in ranges], dtype=float)
    width = (hi - lo).clip(min=0)
    center = (lo + hi) / 2.0
    df = pd.DataFrame({
        "low_norm": (lo - vmin)/dom,
        "high_norm": (hi - vmin)/dom,
        "width_norm": width/dom,
        "center_norm": (center - vmin)/dom,
        "width_log1p": np.log1p(width),
    })
    return df

def train_baseline_model(X_train, y_train):
    from sklearn.ensemble import ExtraTreesRegressor
    mdl = ExtraTreesRegressor(n_estimators=200, n_jobs=-1, random_state=42)
    t0 = time.perf_counter(); mdl.fit(X_train, y_train); t1 = time.perf_counter()
    return mdl, (t1 - t0)

# -------------------------
# Metrics
# -------------------------

def q_error_vec(y_true, y_pred, eps=1e-12):
    yt = np.maximum(y_true, eps)
    yp = np.maximum(y_pred, eps)
    return np.maximum(yt/yp, yp/yt)

def summarize(y_true, y_pred):
    qe = q_error_vec(y_true, y_pred)
    return {
        "QErr_median": float(np.median(qe)),
        "QErr_p95": float(np.percentile(qe, 95)),
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred)**2))),
    }

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Range-only pipeline with bucket-aware training")
    ap.add_argument("--rows", type=int, default=5_000_000)
    ap.add_argument("--dist", default="normal",
                    choices=["uniform","normal","laplace","student_t","lognormal","pareto","zipf"])
    ap.add_argument("--vmin", type=int, default=30_000)
    ap.add_argument("--vmax", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--samples-per-bucket", type=int, default=3)
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--out-root", default="artifacts_range_only")
    ap.add_argument("--input-csv", default=None, help="Path to input CSV (1 col int) to use instead of generating.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) dataset
    if args.input_csv:
        ds_path = Path(args.input_csv)
        ds_name = ds_path.name
        print(f"[1/7] using input dataset -> {ds_path}")
        # When using input CSV, we assume it's already there, so we skip generation
        # We also need to set N, vmin, vmax from the file later if not consistent, 
        # but the next step (scan_min_max_count) handles vmin/vmax detection.
    else:
        ds_name = f"salary_{args.dist}_{args.rows}.csv"
        ds_path = out_root / "data" / ds_name
        print(f"[1/7] dataset -> {ds_path}")
        values = gen_values(rng, args.dist, args.rows, args.vmin, args.vmax)
        save_csv_column(values, ds_path)

    # 2) histogram
    print("[2/7] histogram & freq")
    t0 = time.perf_counter()
    vmin, vmax, N = scan_min_max_count(ds_path)
    freq = build_frequency(ds_path, vmin, vmax)
    buckets = make_equiwidth_buckets(vmin, vmax, args.bins)
    bc = precompute_bucket_counts(buckets, freq, vmin)
    hist_sec = time.perf_counter() - t0

    # 3) train workload
    print("[3/7] bucket-aware training workload")
    train_ranges = []
    for b in buckets:
        train_ranges += gen_partial_overlaps_for_bucket(rng, b, vmin, vmax, k_each_side=args.samples_per_bucket)

    # 4) eval workload
    print("[4/7] eval workload random")
    eval_ranges = gen_eval_random_ranges(rng, args.eval_n, vmin, vmax)

    # 5) truth
    print("[5/7] truth selectivities")
    ps = np.cumsum(freq)
    def true_sel(q):
        return count_range(ps, vmin, q.low, q.high) / float(N if N>0 else 1.0)
    y_train = np.array([true_sel(q) for q in train_ranges], dtype=float)
    y_eval  = np.array([true_sel(q) for q in eval_ranges], dtype=float)

    # 6) per-bucket models
    print("[6/7] train bucket models")
    t0 = time.perf_counter()
    bucket_rows = collect_bucket_training_rows(buckets, freq, vmin, train_ranges)
    models, train_times = train_bucket_models(bucket_rows, min_samples=3)
    bucket_total_train_sec = float(sum(train_times.values()))
    bucket_max_train_sec = float(max(train_times.values()) if train_times else 0.0)
    bucket_models_trained = int(sum(1 for _,t in train_times.items() if t>0))
    bucket_models_uniform  = len(buckets) - bucket_models_trained

    # 7) eval hybrid + baseline
    print("[7/7] eval hybrid + baseline")
    t0 = time.perf_counter()
    pred_counts = [predict_range_hybrid(q, buckets, models, bc, vmin) for q in eval_ranges]
    hybrid_infer_sec = time.perf_counter() - t0
    y_hybrid = np.array(pred_counts, dtype=float) / float(N if N>0 else 1.0)

    from sklearn.preprocessing import StandardScaler
    X_train_b = build_baseline_features(train_ranges, vmin, vmax)
    X_eval_b  = build_baseline_features(eval_ranges,  vmin, vmax)
    scaler = StandardScaler()
    cols = ["low_norm","high_norm","width_norm","center_norm","width_log1p"]
    X_train_b[cols] = scaler.fit_transform(X_train_b[cols])
    X_eval_b[cols]  = scaler.transform(X_eval_b[cols])
    baseline_model, baseline_train_sec = train_baseline_model(X_train_b, y_train)
    t0 = time.perf_counter()
    y_base = baseline_model.predict(X_eval_b).reshape(-1)
    baseline_infer_sec = time.perf_counter() - t0

    met_h = summarize(y_eval, y_hybrid)
    met_b = summarize(y_eval, y_base)

    res_dir = out_root / "results" / Path(ds_name).stem
    res_dir.mkdir(parents=True, exist_ok=True)

    perq = pd.DataFrame({
        "low": [q.low for q in eval_ranges],
        "high": [q.high for q in eval_ranges],
        "true_selectivity": y_eval,
        "hybrid_pred": y_hybrid,
        "hybrid_qerror": q_error_vec(y_eval, y_hybrid),
        "baseline_pred": y_base,
        "baseline_qerror": q_error_vec(y_eval, y_base),
    })
    perq.to_csv(res_dir / "per_query_eval.csv", index=False)

    summary = {
        "dataset": ds_name,
        "rows": int(N),
        "dist": args.dist,
        "bins": len(buckets),
        "hist_build_sec": hist_sec,
        "bucket_models_trained": bucket_models_trained,
        "bucket_models_fallback_uniform": bucket_models_uniform,
        "bucket_total_train_sec": bucket_total_train_sec,
        "bucket_max_train_sec": bucket_max_train_sec,
        "hybrid_infer_sec_total": hybrid_infer_sec,
        "hybrid_infer_ms_per_query": float(1000.0 * hybrid_infer_sec / max(len(eval_ranges),1)),
        "hybrid_QErr_median": met_h["QErr_median"],
        "hybrid_QErr_p95": met_h["QErr_p95"],
        "hybrid_MAE": met_h["MAE"],
        "hybrid_RMSE": met_h["RMSE"],
        "baseline_model": "ExtraTrees200",
        "baseline_train_sec": baseline_train_sec,
        "baseline_infer_sec_total": baseline_infer_sec,
        "baseline_infer_ms_per_query": float(1000.0 * baseline_infer_sec / max(len(eval_ranges),1)),
        "baseline_QErr_median": met_b["QErr_median"],
        "baseline_QErr_p95": met_b["QErr_p95"],
        "baseline_MAE": met_b["MAE"],
        "baseline_RMSE": met_b["RMSE"],
    }
    with open(res_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    cov_stats = []
    for i,b in enumerate(buckets):
        rs = bucket_rows.get(i, [])
        cov_stats.append({
            "bucket": i,
            "lo": b.lo, "hi": b.hi, "width": b.hi - b.lo + 1,
            "n_rows": len(rs),
            "used_model": "ridge" if (i in models and models[i][0] == "ridge") else "uniform"
        })
    pd.DataFrame(cov_stats).to_csv(res_dir / "bucket_training_coverage.csv", index=False)

    print("\n=== SUMMARY ===")
    import json as _json
    print(_json.dumps(summary, indent=2))
    print(f"\nArtifacts written to: {res_dir}")

if __name__ == "__main__":
    main()
