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
    ax1.bar(
        range(n_buckets),
        bucket_counts,
        color="lightgray",
        alpha=0.5,
        label="Bucket Row Count",
    )
    ax1.set_xlabel("Bucket Index")
    ax1.set_ylabel("Estimated Row Count", color="gray")

    ax2 = ax1.twinx()
    ax2.plot(
        range(n_buckets),
        median_errs,
        color="red",
        linewidth=1.5,
        label="Median Q-Error",
    )
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

    # --- STEP 1: USE PRE-GENERATED RESERVOIR SAMPLE ---
    obs_freq = np.bincount(sample - mn, minlength=len(freq))

    # Scale the sample back up to N (Crucial for Cardinality Estimation)
    scaling_factor = N / len(sample)
    estimated_freq = obs_freq * scaling_factor
    # --------------------------------------------------

    # Build Initial Buckets using the ESTIMATED frequency
    ew_hist = EquiWidthHistogram.build_from_sample(
        mn=mn, mx=mx, bins=n_bins, sample=sample, total_rows=N
    )

    # Train Hybrid using the ESTIMATED frequency
    print(f"Training Hybrid Model on 100000 Sample (Bins: {n_bins})...")
    hybrid_est = HybridEstimator(
        ew_hist.buckets,
        identity_threshold=args.ident,
        mlp_penalty=args.penalty,
        fourier_penalty=args.penalty,
    )

    t_train = hybrid_est.train(estimated_freq, mn, args.points, rng)

    # Report Model Selections
    model_report_path = out_dir / "model_selection_Hybrid.txt"
    hybrid_est.report_models(file_path=model_report_path)

    # Load Workload
    # Здесь y_true_sel - это селективность (доля от общего числа строк)
    queries, y_true_sel = load_and_filter_workload(args.workload, mn, mx, freq)

    # Evaluate
    t0 = time.perf_counter()
    y_pred_counts = hybrid_est.predict_batch(queries)
    infer_time = time.perf_counter() - t0

    # --- FIX: SANITY FLOOR (Защита от нулей) ---
    y_true_counts = (
        y_true_sel * N
    )  # переводим селективность в абсолютное количество строк

    y_pred_counts = np.maximum(y_pred_counts, 1.0)
    y_true_counts = np.maximum(y_true_counts, 1.0)

    # Возвращаем обратно в селективность для совместимости с функциями метрик
    y_pred_safe = y_pred_counts / N
    y_true_safe = y_true_counts / N
    # -------------------------------------------

    # --- VISUALIZATION & SUMMARY ---
    debug_plot_path = out_dir / "initial_baseline_debug.png"
    # Передаем безопасные значения
    plot_bucket_debug(hybrid_est, queries, y_true_safe, y_pred_safe, debug_plot_path)

    # Summarize
    m = summarize(y_true_safe, y_pred_safe, "Hybrid")
    metrics = {
        "train_time": t_train,
        "infer_time": infer_time,
        "median_q_error": m["QErr_median"],
        "p25_q_error": m["QErr_p25"],
        "p75_q_error": m["QErr_p75"],
        "p95_q_error": m["QErr_p95"],
        "avg_q_error": m["QErr_avg"],
    }

    print(
        f"Hybrid: Median QErr={metrics['median_q_error']:.4f}, Train={t_train:.4f}s, Inf={infer_time:.4f}s"
    )

    save_benchmark_results(
        out_dir,
        Path(args.workload).stem,
        "Hybrid",
        queries,
        y_true_safe,
        y_pred_safe,
        metrics,
    )


def main():
    parser = get_common_parser("Run Hybrid Model Benchmark")
    parser.add_argument(
        "--points", type=int, default=200, help="Points per bucket for training"
    )
    parser.add_argument("--ident", type=float, default=1e-4, help="Identity threshold")
    parser.add_argument(
        "--penalty", type=float, default=1.5, help="Model selection penalty"
    )
    args = parser.parse_args()
    run_benchmark_suite(args, run_logic)


if __name__ == "__main__":
    main()
