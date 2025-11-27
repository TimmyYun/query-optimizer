#!/usr/bin/env python3
"""
Bucket-hybrid selectivity estimator:
- Equi-width histogram (default 100 bins)
- One tiny ML model per bucket to predict the fraction of rows selected in that bucket for partial overlaps
- For each query: sum full buckets exactly + predict partial edge buckets via their models
- Compare to true selectivity and to your best global model (loaded from results/<dataset>/leaderboard.csv)

Inputs:
  data/*.csv                      # one integer per line (e.g., salary_normal_500mb.csv)
  workload/workload_100.sql       # 50 equality + 50 ranges, as earlier
  truth/truth_mapping.csv         # optional; if missing, truth computed internally
  results/<dataset>               # from your previous trainer, to load best global model (optional)

Outputs (per dataset):
  hybrid_results/<stem>_per_query.csv     # true, baseline, hybrid predictions + errors
  hybrid_results/<stem>_summary.json      # metrics summary
  hybrid_results/hybrid_eval_all.csv      # appended across datasets
"""

# python train_eval_bucket_hybrid.py \
#   --data-glob "data/salary_*_500mb.csv" \
#   --workload "workload/workload_100.sql" \
#   --bins 100 \
#   --outdir hybrid_results \
#   --min-samples-bucket 3

# [ coverage_ratio, start_norm, end_norm, center_norm, is_eq ]

# 0.0015419580013258383 - hybrid
# 0.014345957999466918 - baseline
# inference time in sec
import argparse, json, math, time, re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from dataclasses import dataclass

# ---------------------------
# Workload parsing
# ---------------------------

EQ_RE = re.compile(
    r"select\s+count\(\*\)\s+from\s+([\w\.]+)\s+where\s+([\w\.]+)\s*=\s*(\d+)\s*;",
    re.IGNORECASE,
)
RANGE_RE = re.compile(
    r"select\s+count\(\*\)\s+from\s+([\w\.]+)\s+where\s+([\w\.]+)\s+between\s+(\d+)\s+and\s+(\d+)\s*;",
    re.IGNORECASE,
)

def parse_workload(sql_path: Path) -> List[Dict[str, Any]]:
    queries = []
    last_comment: Optional[str] = None
    for raw in sql_path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("--"):
            last_comment = s[2:].strip()
            continue
        m = EQ_RE.match(s)
        if m:
            table, col, v = m.groups()
            queries.append({
                "kind":"equality","table":table,"column":col,
                "value":int(v),"low":None,"high":None,"sql":s,"comment":last_comment
            })
            last_comment=None; continue
        m = RANGE_RE.match(s)
        if m:
            table, col, lo, hi = m.groups()
            lo_i, hi_i = int(lo), int(hi)
            if lo_i>hi_i: lo_i, hi_i = hi_i, lo_i
            queries.append({
                "kind":"range","table":table,"column":col,
                "value":None,"low":lo_i,"high":hi_i,"sql":s,"comment":last_comment
            })
            last_comment=None; continue
    return queries

# ---------------------------
# Exact truth via frequency array
# ---------------------------

def pass_min_max_count(csv_path: Path, chunksize=1_000_000) -> Tuple[int,int,int]:
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
    width = mx - mn + 1
    freq = np.zeros(width, dtype=np.int64)
    for ch in pd.read_csv(csv_path, header=None, names=["v"], dtype="int64",
                          chunksize=chunksize, engine="c"):
        vals = ch["v"].to_numpy()
        idx = vals - mn
        m = (idx>=0)&(idx<width)
        if not np.all(m): idx = idx[m]
        if idx.size:
            bc = np.bincount(idx, minlength=width)
            freq += bc
    return freq

def count_eq(freq: np.ndarray, mn: int, val: int) -> int:
    i = val - mn
    if 0 <= i < freq.size:
        return int(freq[i])
    return 0

def count_range(psum: np.ndarray, mn: int, lo: int, hi: int) -> int:
    lo_i = max(0, lo - mn)
    hi_i = min(psum.size - 1, hi - mn)
    if hi_i < 0 or lo_i > psum.size - 1:
        return 0
    if lo_i == 0:
        return int(psum[hi_i])
    return int(psum[hi_i] - psum[lo_i - 1])

# ---------------------------
# Q-error + basic metrics
# ---------------------------

def q_error_vec(y_true: np.ndarray, y_pred: np.ndarray, eps=1e-12) -> np.ndarray:
    yt = np.maximum(y_true, eps)
    yp = np.maximum(y_pred, eps)
    return np.maximum(yt/yp, yp/yt)

def metrics_summary(y_true, y_pred):
    qe = q_error_vec(y_true, y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    return {
        "QErr_median": float(np.median(qe)),
        "QErr_p95": float(np.percentile(qe, 95)),
        "MAE": mae,
        "RMSE": rmse,
    }

# ---------------------------
# Global baseline loader (your best model)
# ---------------------------

def build_global_features(df: pd.DataFrame, vmin: int, vmax: int) -> pd.DataFrame:
    dom = max(vmax - vmin, 1)
    out = pd.DataFrame(index=df.index)
    out["is_eq"] = (df["kind"]=="equality").astype(int)
    v = df["value"].fillna((vmin+vmax)//2).astype(float)
    out["value_norm"] = (v - vmin)/dom
    lo = df["low"].fillna(vmin).astype(float)
    hi = df["high"].fillna(vmin).astype(float)
    width = (hi - lo).clip(lower=0)
    center = (lo + hi)/2
    out["low_norm"] = (lo - vmin)/dom
    out["high_norm"] = (hi - vmin)/dom
    out["width_norm"] = width/dom
    out["center_norm"] = (center - vmin)/dom
    out["width_log1p"] = np.log1p(width)
    out["is_eq_x_value"] = out["is_eq"] * out["value_norm"]
    return out

def try_load_global_best(dataset_stem: str, data_dir: Path) -> Tuple[Optional[Any], Optional[Any], Optional[str]]:
    import joblib
    ds_dir = Path("results")/dataset_stem
    lb = ds_dir/"leaderboard.csv"
    if not lb.exists():
        return None, None, None
    df = pd.read_csv(lb)
    if df.empty or "model" not in df.columns:
        return None, None, None
    best_name = str(df.iloc[0]["model"])
    mdl_path = ds_dir/f"{best_name}.joblib"
    scaler_path = ds_dir/"scaler.joblib"
    if not mdl_path.exists() or not scaler_path.exists():
        return None, None, None
    model = joblib.load(mdl_path)
    scaler = joblib.load(scaler_path)
    return model, scaler, best_name

# ---------------------------
# Equi-width histogram + bucket splits
# ---------------------------

@dataclass
class BucketMeta:
    lo: int
    hi: int
    count: int

def make_buckets(mn: int, mx: int, bins: int) -> List[BucketMeta]:
    """Build integer-aligned equi-width buckets covering [mn, mx]."""
    width = mx - mn + 1
    bw = int(math.ceil(width / bins))
    buckets = []
    cur = mn
    for k in range(bins):
        lo = cur
        hi = min(mx, lo + bw - 1)
        buckets.append(BucketMeta(lo=lo, hi=hi, count=0))
        cur = hi + 1
        if cur > mx: break
    return buckets

def bucket_index_for_value(buckets: List[BucketMeta], v: int) -> int:
    # linear scan ok for 100; can binary search if needed
    for i,b in enumerate(buckets):
        if b.lo <= v <= b.hi:
            return i
    return -1

# ---------------------------
# Train tiny per-bucket models
# ---------------------------

@dataclass
class BucketTrainRow:
    cov_ratio: float
    start_norm: float
    end_norm: float
    center_norm: float
    is_eq: int
    y_frac: float

def collect_bucket_training_data(buckets: List[BucketMeta],
                                 freq: np.ndarray, mn: int,
                                 queries: List[Dict[str,Any]]) -> Dict[int, List[BucketTrainRow]]:
    """Use the workload queries to create training rows per bucket for *partial overlaps*."""
    psum = np.cumsum(freq)
    data: Dict[int, List[BucketTrainRow]] = {i: [] for i in range(len(buckets))}

    for q in queries:
        if q["kind"]=="equality":
            v = q["value"]
            bi = bucket_index_for_value(buckets, v)
            if bi<0: continue
            b = buckets[bi]
            # treat as a very narrow range [v,v]
            lo, hi = v, v
            width = max(1, b.hi - b.lo + 1)
            cov = 1/width
            start = (lo - b.lo)/width
            end = (hi - b.lo)/width
            center = ( (lo+hi)/2 - b.lo ) / width
            # true fraction inside bucket:
            b_lo_idx = b.lo - mn
            b_hi_idx = b.hi - mn
            b_count = int(psum[b_hi_idx] - (psum[b_lo_idx-1] if b_lo_idx>0 else 0))
            if b_count == 0: continue
            # exact count at v:
            ycnt = count_eq(freq, mn, v)
            yfrac = ycnt / b_count
            data[bi].append(BucketTrainRow(cov, start, end, center, 1, yfrac))
        else:
            lo, hi = q["low"], q["high"]
            # left edge
            li = bucket_index_for_value(buckets, lo)
            ri = bucket_index_for_value(buckets, hi)
            if li<0 and ri<0:
                continue
            if li>=0:
                b = buckets[li]
                # is it fully covered? (if hi >= b.hi and lo <= b.lo -> full)
                if not (lo <= b.lo and hi >= b.hi):
                    width = max(1, b.hi - b.lo + 1)
                    ov_lo = max(lo, b.lo); ov_hi = min(hi, b.hi)
                    if ov_lo <= ov_hi:
                        cov = (ov_hi - ov_lo + 1)/width
                        start = (ov_lo - b.lo)/width
                        end = (ov_hi - b.lo)/width
                        center = ((ov_lo+ov_hi)/2 - b.lo)/width
                        # label
                        b_lo_idx = b.lo - mn; b_hi_idx = b.hi - mn
                        b_count = int(psum[b_hi_idx] - (psum[b_lo_idx-1] if b_lo_idx>0 else 0))
                        if b_count>0:
                            ycnt = count_range(psum, mn, ov_lo, ov_hi)
                            yfrac = ycnt / b_count
                            data[li].append(BucketTrainRow(cov, start, end, center, 0, yfrac))
            # right edge
            if ri>=0 and ri!=li:
                b = buckets[ri]
                if not (lo <= b.lo and hi >= b.hi):
                    width = max(1, b.hi - b.lo + 1)
                    ov_lo = max(lo, b.lo); ov_hi = min(hi, b.hi)
                    if ov_lo <= ov_hi:
                        cov = (ov_hi - ov_lo + 1)/width
                        start = (ov_lo - b.lo)/width
                        end = (ov_hi - b.lo)/width
                        center = ((ov_lo+ov_hi)/2 - b.lo)/width
                        b_lo_idx = b.lo - mn; b_hi_idx = b.hi - mn
                        b_count = int(psum[b_hi_idx] - (psum[b_lo_idx-1] if b_lo_idx>0 else 0))
                        if b_count>0:
                            ycnt = count_range(psum, mn, ov_lo, ov_hi)
                            yfrac = ycnt / b_count
                            data[ri].append(BucketTrainRow(cov, start, end, center, 0, yfrac))
    return data

def train_bucket_models(bucket_rows: Dict[int, List[BucketTrainRow]],
                        min_samples: int = 3):
    """
    Train a tiny model per bucket (Ridge regression).
    Fallback: if < min_samples, use 'uniform' predictor y=cov_ratio.
    """
    from sklearn.linear_model import Ridge
    models = {}
    train_times = {}
    for bi, rows in bucket_rows.items():
        if len(rows) < min_samples:
            models[bi] = ("uniform", None)  # fallback
            train_times[bi] = 0.0
            continue
        X = np.array([[r.cov_ratio, r.start_norm, r.end_norm, r.center_norm, r.is_eq] for r in rows], dtype=float)
        y = np.array([r.y_frac for r in rows], dtype=float)
        mdl = Ridge(alpha=1e-4, random_state=42)
        t0 = time.perf_counter()
        mdl.fit(X, y)
        t1 = time.perf_counter()
        models[bi] = ("ridge", mdl)
        train_times[bi] = t1 - t0
    return models, train_times

def bucket_predict(models, bi: int, cov_ratio, start_norm, end_norm, center_norm, is_eq):
    kind, mdl = models.get(bi, ("uniform", None))
    if kind == "uniform" or mdl is None:
        return float(cov_ratio)  # default: fraction ~= covered length ratio
    X = np.array([[cov_ratio, start_norm, end_norm, center_norm, is_eq]], dtype=float)
    y = float(mdl.predict(X)[0])
    # guard into [0,1]
    return max(0.0, min(1.0, y))

# ---------------------------
# Hybrid prediction
# ---------------------------

def precompute_bucket_counts(buckets: List[BucketMeta], freq: np.ndarray, mn: int) -> Tuple[np.ndarray, np.ndarray]:
    bc = np.zeros(len(buckets), dtype=np.int64)
    ps = np.cumsum(freq)
    for i,b in enumerate(buckets):
        lo_idx = b.lo - mn
        hi_idx = b.hi - mn
        bc[i] = int(ps[hi_idx] - (ps[lo_idx-1] if lo_idx>0 else 0))
    return bc, np.cumsum(bc)

def predict_query_hybrid(q, buckets: List[BucketMeta], models, bc: np.ndarray,
                         mn: int, psum_vals: np.ndarray) -> int:
    """
    Return predicted COUNT (not fraction). We'll divide by total later.
    """
    total_pred = 0.0
    if q["kind"]=="equality":
        v = q["value"]
        bi = bucket_index_for_value(buckets, v)
        if bi<0: return 0
        b = buckets[bi]
        width = max(1, b.hi - b.lo + 1)
        cov = 1/width
        start = (v - b.lo)/width
        end = start
        center = start
        frac = bucket_predict(models, bi, cov, start, end, center, 1)
        total_pred += frac * bc[bi]
        return int(round(total_pred))

    lo, hi = q["low"], q["high"]
    li = bucket_index_for_value(buckets, lo)
    ri = bucket_index_for_value(buckets, hi)

    if li<0 and ri<0:
        return 0

    # partial left
    if li>=0:
        b = buckets[li]
        width = max(1, b.hi - b.lo + 1)
        ov_lo = max(lo, b.lo); ov_hi = min(hi, b.hi)
        if ov_lo <= ov_hi:
            if not (lo <= b.lo and hi >= b.hi):  # partial
                cov = (ov_hi - ov_lo + 1)/width
                start = (ov_lo - b.lo)/width
                end = (ov_hi - b.lo)/width
                center = ((ov_lo+ov_hi)/2 - b.lo)/width
                frac = bucket_predict(models, li, cov, start, end, center, 0)
                total_pred += frac * bc[li]
            else:
                total_pred += bc[li]

    # middle full buckets
    if li>=0 and ri>=0:
        a = li+1
        b = ri-1
        if b>=a:
            total_pred += float(bc[a:b+1].sum())

    # right partial
    if ri>=0 and ri!=li:
        b = buckets[ri]
        width = max(1, b.hi - b.lo + 1)
        ov_lo = max(lo, b.lo); ov_hi = min(hi, b.hi)
        if ov_lo <= ov_hi:
            if not (lo <= b.lo and hi >= b.hi):  # partial
                cov = (ov_hi - ov_lo + 1)/width
                start = (ov_lo - b.lo)/width
                end = (ov_hi - b.lo)/width
                center = ((ov_lo+ov_hi)/2 - b.lo)/width
                frac = bucket_predict(models, ri, cov, start, end, center, 0)
                total_pred += frac * bc[ri]
            else:
                total_pred += bc[ri]

    return int(round(total_pred))

# ---------------------------
# Driver
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Train/evaluate bucket-hybrid selectivity models.")
    ap.add_argument("--data-glob", default="data/salary_*_500mb.csv")
    ap.add_argument("--workload", default="workload/workload_100.sql")
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--outdir", default="hybrid_results")
    ap.add_argument("--truth", default="", help="Optional precomputed truth CSV (truth/truth_mapping.csv)")
    ap.add_argument("--min-samples-bucket", type=int, default=1)
    args = ap.parse_args()

    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    queries = parse_workload(Path(args.workload))
    if not queries:
        raise SystemExit(f"No queries parsed from {args.workload}")

    # Optional precomputed truth
    truth_df = None
    if args.truth and Path(args.truth).exists():
        truth_df = pd.read_csv(args.truth)

    all_ds_rows = []

    for csv_path in sorted(Path(".").glob(args.data_glob)):
        if csv_path.suffix.lower() != ".csv":
            continue
        stem = csv_path.stem
        print(f"\n=== Dataset: {stem} ===")

        # 1) Build exact frequency + histogram
        t0 = time.perf_counter()
        vmin, vmax, N = pass_min_max_count(csv_path)
        freq = build_frequency(csv_path, vmin, vmax)
        assert N == int(freq.sum())
        psum_vals = np.cumsum(freq)
        buckets = make_buckets(vmin, vmax, args.bins)

        # bucket counts
        bc, bc_psum = precompute_bucket_counts(buckets, freq, vmin)
        build_sec = time.perf_counter() - t0

        # 2) True selectivity for this dataset
        if truth_df is not None and "dataset" in truth_df.columns and stem+".csv" in truth_df["dataset"].values:
            df_truth = truth_df[truth_df["dataset"]==stem+".csv"].copy()
            y_true = df_truth["true_selectivity"].to_numpy()
        else:
            # compute truth for this dataset specifically
            y_true = []
            for q in queries:
                if q["kind"]=="equality":
                    cnt = count_eq(freq, vmin, q["value"])
                else:
                    cnt = count_range(psum_vals, vmin, q["low"], q["high"])
                y_true.append(cnt / N if N>0 else 0.0)
            y_true = np.array(y_true, dtype=float)
        # pack a working frame for features/preds
        qdf = pd.DataFrame(queries)
        qdf["true_selectivity"] = y_true

        # 3) Train per-bucket models (using workload-derived partials)
        bucket_rows = collect_bucket_training_data(buckets, freq, vmin, queries)
        models, train_times = train_bucket_models(bucket_rows, min_samples=args.min_samples_bucket)
        max_bucket_train_sec = float(max(train_times.values()) if train_times else 0.0)
        total_train_sec = float(sum(train_times.values()))

        # 4) Hybrid predictions (count then normalize)
        t2 = time.perf_counter()
        pred_counts = []
        for q in queries:
            cnt = predict_query_hybrid(q, buckets, models, bc, vmin, psum_vals)
            pred_counts.append(cnt)
        hybrid_infer_sec = time.perf_counter() - t2
        y_hybrid = np.array(pred_counts, dtype=float) / float(N if N>0 else 1.0)

        # 5) Global baseline predictions (if available)
        baseline_name = None
        try:
            mdl, scaler, baseline_name = try_load_global_best(stem, Path("data"))
        except Exception:
            mdl, scaler, baseline_name = None, None, None

        if mdl is not None and scaler is not None and baseline_name is not None:
            Xg = build_global_features(qdf, vmin, vmax)
            Xg_scaled = Xg.copy()
            # scale only non-binary columns used by previous script
            cont_cols = [c for c in Xg.columns if c != "is_eq"]
            Xg_scaled[cont_cols] = scaler.transform(Xg[cont_cols])
            t3 = time.perf_counter()
            y_base = mdl.predict(Xg_scaled).reshape(-1)
            baseline_infer_sec = time.perf_counter() - t3
        else:
            y_base = np.full_like(y_hybrid, np.nan)
            baseline_infer_sec = 0.0
            baseline_name = None

        # 6) Metrics
        sum_h = metrics_summary(y_true, y_hybrid)
        sum_b = metrics_summary(y_true, y_base[~np.isnan(y_base)]) if not np.isnan(y_base).all() else None

        # 7) Save per-query results
        per_query = pd.DataFrame({
            "dataset": stem+".csv",
            "query_index": np.arange(1, len(queries)+1),
            "kind": qdf["kind"],
            "value": qdf["value"],
            "low": qdf["low"],
            "high": qdf["high"],
            "true_selectivity": y_true,
            "baseline_model": baseline_name,
            "baseline_pred": y_base,
            "baseline_qerr": q_error_vec(y_true, np.nan_to_num(y_base, nan=0.0)),
            "hybrid_pred": y_hybrid,
            "hybrid_qerr": q_error_vec(y_true, y_hybrid),
        })
        out_dir = Path(args.outdir)
        out_dir.mkdir(parents=True, exist_ok=True)
        per_query_path = out_dir / f"{stem}_per_query.csv"
        per_query.to_csv(per_query_path, index=False)

        # 8) Save summary
        summary = {
            "dataset": stem+".csv",
            "rows": int(N),
            "bins": len(buckets),
            "hist_build_sec": build_sec,
            "bucket_models_trained": int(sum(1 for k,(t) in train_times.items() if t>0)),
            "bucket_models_fallback_uniform": int(sum(1 for k,(t) in train_times.items() if t==0)),
            "bucket_total_train_sec": total_train_sec,
            "bucket_max_train_sec": max_bucket_train_sec,
            "hybrid_infer_sec_total": hybrid_infer_sec,
            "hybrid_infer_ms_per_query": 1000.0*hybrid_infer_sec/len(queries),
            "hybrid_QErr_median": sum_h["QErr_median"],
            "hybrid_QErr_p95": sum_h["QErr_p95"],
            "hybrid_MAE": sum_h["MAE"],
            "hybrid_RMSE": sum_h["RMSE"],
            "baseline_model": baseline_name,
            "baseline_infer_sec_total": baseline_infer_sec,
            "baseline_infer_ms_per_query": (1000.0*baseline_infer_sec/len(queries)) if baseline_name else None,
            "baseline_QErr_median": sum_b["QErr_median"] if sum_b else None,
            "baseline_QErr_p95": sum_b["QErr_p95"] if sum_b else None,
            "baseline_MAE": sum_b["MAE"] if sum_b else None,
            "baseline_RMSE": sum_b["RMSE"] if sum_b else None,
        }
        with open(out_dir / f"{stem}_summary.json","w") as f:
            json.dump(summary, f, indent=2)

        # append to global
        row = summary.copy()
        all_ds_rows.append(row)

        print(f"[{stem}] hybrid median QErr={summary['hybrid_QErr_median']:.3f}, p95={summary['hybrid_QErr_p95']:.3f} | "
              f"hist_build={build_sec:.2f}s, max_bucket_train={max_bucket_train_sec:.3f}s, "
              f"infer={summary['hybrid_infer_ms_per_query']:.3f} ms/query")

    # save global table
    if all_ds_rows:
        pd.DataFrame(all_ds_rows).to_csv(Path(args.outdir)/"hybrid_eval_all.csv", index=False)

if __name__ == "__main__":
    main()
