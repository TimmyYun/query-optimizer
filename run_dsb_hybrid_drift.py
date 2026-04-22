#!/usr/bin/env python3
"""Hybrid (Shock → Finetune → Rebuild) drift experiment on DSB (CSV-based) data."""
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from models import EquiWidthHistogram, HybridEstimator, summarize
from benchmark_utils import load_and_filter_workload, setup_out_dir

# Column → table mapping
COLUMN_TABLE = {
    "cr_item_sk": "catalog_returns",
    "cr_returned_time_sk": "catalog_returns",
    "ws_item_sk": "web_sales",
}

CHURN_STEPS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
SAMPLE_SIZE = 100_000


def get_sampled_freq(sample, mn, total_n, target_len):
    clipped_sample = np.clip(sample - mn, 0, target_len - 1)
    counts = np.bincount(clipped_sample, minlength=target_len)
    scaling_factor = total_n / len(sample)
    return counts * scaling_factor


def load_csv_data(csv_path):
    """Load a DSB CSV file (has header), return raw values as int64 array."""
    df = pd.read_csv(csv_path)
    return df.iloc[:, 0].to_numpy(dtype=np.int64)


def build_meta_from_values(values, mn, mx):
    """Build freq array and sample from raw values."""
    N = len(values)
    width = mx - mn + 1
    freq = np.zeros(width, dtype=np.int64)
    idx = values - mn
    mask = (idx >= 0) & (idx < width)
    valid_idx = idx[mask]
    if valid_idx.size:
        freq += np.bincount(valid_idx, minlength=width)

    rng = np.random.default_rng(42)
    sample_size = min(SAMPLE_SIZE, N)
    sample = rng.choice(values, size=sample_size, replace=False)
    return N, freq, sample


def detect_bad_buckets(hybrid_est, queries, y_true_counts, y_pred_counts, error_threshold=1.3):
    bucket_errors = {i: [] for i in range(len(hybrid_est.buckets))}

    b_lo_vals = np.array([b.lo for b in hybrid_est.buckets])
    b_hi_vals = np.array([b.hi for b in hybrid_est.buckets])

    for idx, q in enumerate(queries):
        err = max(y_true_counts[idx] / (y_pred_counts[idx] + 1e-9),
                  y_pred_counts[idx] / (y_true_counts[idx] + 1e-9))

        s_idx = np.searchsorted(b_hi_vals, q.low)
        e_idx = np.searchsorted(b_lo_vals, q.high, side="right")

        for i in range(s_idx, e_idx):
            bucket_errors[i].append(err)

    bad_buckets = []
    for i, errors in bucket_errors.items():
        if len(errors) > 5:
            if np.median(errors) > error_threshold:
                bad_buckets.append(i)

    return bad_buckets


def main():
    parser = argparse.ArgumentParser(description="Hybrid Drift (Shock→FT→RB) on DSB data")
    parser.add_argument("--column", type=str, required=True,
                        choices=list(COLUMN_TABLE.keys()),
                        help="Column name (e.g. cr_item_sk)")
    parser.add_argument("--data-dir", type=str, default="data/dsb",
                        help="Base DSB data directory")
    parser.add_argument("--points", type=int, default=200)
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)
    args = parser.parse_args()

    table = COLUMN_TABLE[args.column]
    data_dir = Path(args.data_dir)
    drift_dir = data_dir / "drift"
    col_dir = data_dir / args.column

    # Load initial meta.pkl
    meta_path = col_dir / "meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.pkl not found at {meta_path}. Run prepare_dsb.py first.")

    with open(meta_path, "rb") as f:
        i_mn, i_mx, i_N, i_freq, i_sample, i_k, *_ = pickle.load(f)

    # Find workload
    workload_files = list(col_dir.glob("workload_driven_*.csv"))
    if not workload_files:
        raise FileNotFoundError(f"No workload file found in {col_dir}")
    workload_path = workload_files[0]
    print(f"Using workload: {workload_path}")

    # Setup output
    ds_out_name = f"dsb_{args.column}_Hybrid"
    out_dir = setup_out_dir(args, ds_out_name)
    rng = np.random.default_rng(42)

    reports_dir = out_dir / "model_reports"
    reports_dir.mkdir(exist_ok=True)

    # Load workload once
    queries, _ = load_and_filter_workload(str(workload_path), i_mn, i_mx, i_freq)
    print(f"Loaded {len(queries)} queries")

    # Build initial model from initial data
    print(f"\n>>> Building initial Hybrid estimator for {args.column} <<<")
    i_est_freq = get_sampled_freq(i_sample, i_mn, i_N, len(i_freq))
    t0_build = time.perf_counter()
    init_hist = EquiWidthHistogram.build_from_sample(i_mn, i_mx, i_k, i_sample, i_N)
    hybrid_est = HybridEstimator(init_hist.buckets)
    hybrid_est.train(i_est_freq, i_mn, args.points, rng)
    build_time = time.perf_counter() - t0_build

    hybrid_est.report_models(file_path=reports_dir / "models_initial.txt")

    summary_records = []

    all_steps = ["initial"] + [f"churn_{s}pct" for s in CHURN_STEPS]

    for step_idx, step_name in enumerate(all_steps):
        shift_pct = 0.0 if step_idx == 0 else CHURN_STEPS[step_idx - 1] / 100.0

        csv_file = drift_dir / f"{table}__{args.column}__{step_name}.csv"
        if not csv_file.exists():
            print(f"Warning: {csv_file} not found, skipping.")
            continue

        values = load_csv_data(csv_file)
        current_N, current_freq, current_sample = build_meta_from_values(values, i_mn, i_mx)

        # Compute ground truth
        ps = np.cumsum(current_freq)
        y_true_counts = np.array([
            max(1.0, ps[min(len(current_freq) - 1, max(0, q.high - i_mn))] -
                (ps[max(0, q.low - 1 - i_mn)] if q.low > i_mn else 0))
            for q in queries
        ])
        y_true_counts = np.maximum(y_true_counts, 1.0)
        y_true_sel = y_true_counts / current_N

        current_est_freq = get_sampled_freq(current_sample, i_mn, current_N, len(current_freq))

        # Update bucket counts (simulates ANALYZE)
        ps_est = np.cumsum(current_est_freq)
        for b in hybrid_est.buckets:
            b_lo_idx = max(0, min(len(ps_est) - 1, b.lo - i_mn))
            b_hi_idx = max(0, min(len(ps_est) - 1, b.hi - i_mn))
            new_count = ps_est[b_hi_idx] - (ps_est[b_lo_idx - 1] if b_lo_idx > 0 else 0)
            b.count = max(0.0, new_count)
        hybrid_est._bake_vectorized_data()

        # STAGE 0: SHOCK
        t0_eval = time.perf_counter()
        y_pred_shock = np.maximum(hybrid_est.predict_batch(queries), 1.0)
        eval_time = time.perf_counter() - t0_eval
        m_shock = summarize(y_true_sel, y_pred_shock / current_N, f"Shock_{step_name}")

        m_ft, m_rb = m_shock, m_shock
        t_ft, t_rb, n_rebuilt, n_finetuned = 0.0, 0.0, 0, 0

        if step_idx > 0:
            # STAGE 1: FINETUNE
            t0 = time.perf_counter()
            n_finetuned = hybrid_est.feedback_update(queries, y_true_counts, y_pred_shock, current_N)
            t_ft = time.perf_counter() - t0

            y_pred_ft = np.maximum(hybrid_est.predict_batch(queries), 1.0)
            m_ft = summarize(y_true_sel, y_pred_ft / current_N, f"FT_{step_name}")

            # STAGE 2: REBUILD
            bad_indices = detect_bad_buckets(
                hybrid_est, queries, y_true_counts, y_pred_ft
            )
            n_rebuilt = len(bad_indices)

            if n_rebuilt > 0:
                t1 = time.perf_counter()
                hybrid_est.train(
                    current_est_freq, i_mn, args.points, rng,
                    bucket_indices=bad_indices,
                )
                t_rb = time.perf_counter() - t1
                y_pred_rb = np.maximum(hybrid_est.predict_batch(queries), 1.0)
                m_rb = summarize(y_true_sel, y_pred_rb / current_N, f"RB_{step_name}")

            hybrid_est.report_models(
                file_path=reports_dir / f"models_{step_name}.txt"
            )

        print(
            f"Step {step_name:>15s} ({shift_pct:>4.0%}): "
            f"[Shock: {m_shock['QErr_median']:.2f}] -> "
            f"[FT: {m_ft['QErr_median']:.2f}] -> "
            f"[RB: {m_rb['QErr_median']:.2f}] | FT: {n_finetuned} | Rebuilt: {n_rebuilt}"
        )

        summary_records.append({
            "Shift %": f"{shift_pct:.0%}",
            "Step": step_name,
            "Shock_Med": m_shock["QErr_median"],
            "Shock_P95": m_shock["QErr_p95"],
            "Shock_Avg": m_shock["QErr_avg"],
            "Evaluation Time": eval_time,
            "FT_Med": m_ft["QErr_median"],
            "FT_P95": m_ft["QErr_p95"],
            "FT_Avg": m_ft["QErr_avg"],
            "Finetuned_Count": n_finetuned,
            "RB_Med": m_rb["QErr_median"],
            "RB_P95": m_rb["QErr_p95"],
            "RB_Avg": m_rb["QErr_avg"],
            "Rebuilt_Count": n_rebuilt,
            "FT_Time": t_ft,
            "RB_Time": t_rb,
            "Build_Time": build_time if step_idx == 0 else 0.0,
        })

    pd.DataFrame(summary_records).to_csv(out_dir / "adaptation_ft_rb.csv", index=False)
    print(f"\nExperiment completed. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
