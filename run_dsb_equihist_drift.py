#!/usr/bin/env python3
"""EquiHist drift experiment on DSB (CSV-based) data."""
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from models import EquiWidthHistogram, EquiHistLearner, summarize
from benchmark_utils import load_and_filter_workload, setup_out_dir, exact_counts, run_queries, timing_stats

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


def main():
    parser = argparse.ArgumentParser(description="EquiHist Drift on DSB data")
    parser.add_argument("--column", type=str, required=True,
                        choices=list(COLUMN_TABLE.keys()),
                        help="Column name (e.g. cr_item_sk)")
    parser.add_argument("--data-dir", type=str, default="data/dsb",
                        help="Base DSB data directory")
    parser.add_argument("--lr", type=float, default=0.5,
                        help="Learning rate for EquiHist")
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
    ds_out_name = f"dsb_{args.column}_EquiHist"
    out_dir = setup_out_dir(args, ds_out_name)

    # Build initial histogram and learner
    print(f"\n>>> Building initial EquiHist for {args.column} <<<")
    ew_hist = EquiWidthHistogram.build_from_sample(i_mn, i_mx, i_k, i_sample, i_N)
    eh_learner = EquiHistLearner(ew_hist.buckets, learning_rate=args.lr)

    # Load workload once
    queries, _ = load_and_filter_workload(str(workload_path), i_mn, i_mx, i_freq)
    print(f"Loaded {len(queries)} queries")

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
        y_true_counts = np.maximum(exact_counts(queries, current_freq, i_mn), 1.0)
        y_true_sel = y_true_counts / current_N

        # Update bucket counts from sample (simulates ANALYZE)
        current_est_freq = get_sampled_freq(current_sample, i_mn, current_N, len(current_freq))
        ps_est = np.cumsum(current_est_freq)
        for b in eh_learner.buckets:
            b_lo_idx = max(0, min(len(ps_est) - 1, b.lo - i_mn))
            b_hi_idx = max(0, min(len(ps_est) - 1, b.hi - i_mn))
            new_count = ps_est[b_hi_idx] - (ps_est[b_lo_idx - 1] if b_lo_idx > 0 else 0)
            b.count = max(0.0, new_count)
        # predict читает закэшированные b_count / prefix_counts — без пересборки первый запрос шага идёт по старым счётчикам
        eh_learner._bake_vectorized_data()

        # Online learning on workload
        def feedback(q, true_count, pred_count):
            # Немедленное обновление (Feedback Loop)
            eh_learner.update(q, true_count, pred=pred_count)

        y_pred_counts, infer_times, update_time = run_queries(
            eh_learner.predict, queries, y_true_counts, feedback
        )
        y_pred = np.maximum(y_pred_counts, 1.0) / current_N
        t = timing_stats(infer_times)
        total_time = t["infer_time"] + update_time
        median_infer_time, avg_infer_time, p95_infer_time = (
            t["median_infer_time"], t["avg_infer_time"], t["p95_infer_time"]
        )

        m = summarize(y_true_sel, y_pred, f"EquiHist_{step_name}")
        print(f"Step {step_name:>15s} ({shift_pct:>4.0%}): Median QErr = {m['QErr_median']:.2f}")

        summary_records.append({
            "Shift %": f"{shift_pct:.0%}",
            "Step": step_name,
            "Median Q-Error": m["QErr_median"],
            "P95 Q-Error": m["QErr_p95"],
            "Avg Q-Error": m["QErr_avg"],
            "Time": total_time,
            "Inference Time": t["infer_time"],
            "Update Time": update_time,
            "Median Inference Time": median_infer_time,
            "Avg Inference Time": avg_infer_time,
            "P95 Inference Time": p95_infer_time,
        })

    pd.DataFrame(summary_records).to_csv(out_dir / "equihist_drift_analysis.csv", index=False)
    print(f"\nExperiment completed. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
