#!/usr/bin/env python3
import time
import numpy as np
from pathlib import Path
from models import EquiWidthHistogram, HybridEstimator, summarize
from benchmark_utils import (
    get_common_parser,
    run_benchmark_suite,
    load_and_filter_workload,
    save_benchmark_results,
)
import matplotlib.pyplot as plt


def plot_bucket_debug(hybrid_est, queries, y_true, y_pred, output_path):
    """
    Visualizes the performance of each bucket after initial training.
    """
    n_buckets = len(hybrid_est.buckets)
    bucket_counts = np.array([b.count for b in hybrid_est.buckets])
    q_errors = np.maximum(y_true / (y_pred + 1e-9), y_pred / (y_true + 1e-9))

    bucket_err_lists = [[] for _ in range(n_buckets)]
    for idx, q in enumerate(queries):
        err = q_errors[idx]
        for b_idx in range(n_buckets):
            b = hybrid_est.buckets[b_idx]
            if q.low <= b.hi and q.high >= b.lo:
                bucket_err_lists[b_idx].append(err)

    median_errs = np.array(
        [np.median(errs) if len(errs) > 0 else 1.0 for errs in bucket_err_lists]
    )

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.bar(range(n_buckets), bucket_counts, color="lightgray", alpha=0.5, label="Bucket Row Count")
    ax1.set_xlabel("Bucket Index")
    ax1.set_ylabel("Estimated Row Count", color="gray")

    ax2 = ax1.twinx()
    ax2.plot(range(n_buckets), median_errs, color="red", linewidth=1.5, label="Median Q-Error")
    ax2.set_ylabel("Median Q-Error (Log Scale)", color="red")
    ax2.set_yscale("log")

    plt.title(f"Initial Training Baseline: Error vs. Density")
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Debug plot saved to {output_path}")


def run_logic(args, out_dir, metadata):
    mn, mx, N, freq, sample, n_bins_data, skew, kurt = metadata
    n_bins = args.buckets if args.buckets is not None else n_bins_data
    rng = np.random.default_rng(42)

    # --- STEP 1: INITIAL TRAINING DATA PREP ---
    obs_freq = np.bincount(sample - mn, minlength=len(freq))
    scaling_factor = N / len(sample)
    estimated_freq = obs_freq * scaling_factor

    # Build Buckets
    ew_hist = EquiWidthHistogram.build_from_sample(
        mn=mn, mx=mx, bins=n_bins, sample=sample, total_rows=N
    )

    # Initialize Model
    hybrid_est = HybridEstimator(
        ew_hist.buckets,
        identity_threshold=args.ident,
        mlp_penalty=args.penalty,
        fourier_penalty=args.penalty,
    )

    # --- MEASURE INITIAL TRAINING ---
    t_start = time.perf_counter()
    hybrid_est.train(estimated_freq, mn, args.points, rng)
    t_train = time.perf_counter() - t_start

    # Load Workload
    queries, y_true = load_and_filter_workload(args.workload, mn, mx, freq)

    # --- MEASURE PURE INFERENCE ---
    t_inf_start = time.perf_counter()
    y_pred_counts = hybrid_est.predict_batch(queries)
    y_pred = y_pred_counts / N
    infer_time = time.perf_counter() - t_inf_start

    # --- VISUALIZATION & SUMMARY ---
    debug_plot_path = out_dir / "initial_baseline_debug.png"
    plot_bucket_debug(hybrid_est, queries, y_true, y_pred, debug_plot_path)

    m = summarize(y_true, y_pred, "Hybrid-Baseline")

    metrics = {
        "train_time": t_train,
        "infer_time": infer_time,
        "median_q_error": m["QErr_median"],
        "p25_q_error": m["QErr_p25"],
        "p75_q_error": m["QErr_p75"],
        "p95_q_error": m["QErr_p95"],
        "avg_q_error": m["QErr_avg"],
    }

    print("-" * 30)
    print(f"Hybrid Baseline Results (No Fine-Tune):")
    print(f"Median QErr: {metrics['median_q_error']:.4f}")
    print(f"p95 QErr:    {metrics['p95_q_error']:.4f}")
    print(f"Train Time:  {t_train:.4f}s")
    print("-" * 30)

    save_benchmark_results(
        out_dir, Path(args.workload).stem, "Hybrid", queries, y_true, y_pred, metrics
    )


def main():
    parser = get_common_parser("Run Hybrid Model Baseline")
    parser.add_argument("--points", type=int, default=200, help="Points per bucket for training")
    parser.add_argument("--ident", type=float, default=1e-4, help="Identity threshold")
    parser.add_argument("--penalty", type=float, default=1.5, help="Model selection penalty")
    args = parser.parse_args()
    run_benchmark_suite(args, run_logic)


if __name__ == "__main__":
    main()