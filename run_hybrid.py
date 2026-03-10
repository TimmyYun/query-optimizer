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
    Visualizes the performance of each bucket.
    X-axis: Bucket Index
    Y-axis (Left): Row Count in Bucket
    Y-axis (Right): Median Q-Error of queries intersecting that bucket
    """
    n_buckets = len(hybrid_est.buckets)
    bucket_counts = np.array([b.count for b in hybrid_est.buckets])

    # 1. Calculate Q-Error for every individual query
    # We use y_true/y_pred selectivity, so we multiply by N to get counts if needed,
    # but the ratio remains the same.
    q_errors = np.maximum(y_true / (y_pred + 1e-9), y_pred / (y_true + 1e-9))

    # 2. Map Query Errors to Buckets
    bucket_err_lists = [[] for _ in range(n_buckets)]
    for idx, q in enumerate(queries):
        err = q_errors[idx]
        # Find all buckets this query intersects
        for b_idx in range(n_buckets):
            b = hybrid_est.buckets[b_idx]
            if q.low <= b.hi and q.high >= b.lo:
                bucket_err_lists[b_idx].append(err)

    # 3. Calculate Median per Bucket
    median_errs = np.array(
        [np.median(errs) if len(errs) > 0 else 1.0 for errs in bucket_err_lists]
    )

    # 4. Create the Plot
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Bar plot for bucket sizes (The Data Distribution)
    ax1.bar(
        range(n_buckets),
        bucket_counts,
        color="lightgray",
        alpha=0.5,
        label="Bucket Row Count",
    )
    ax1.set_xlabel("Bucket Index")
    ax1.set_ylabel("Row Count (Initial Estimate)", color="gray")
    ax1.tick_params(axis="y", labelcolor="gray")

    # Line plot for Q-Error (The Performance)
    ax2 = ax1.twinx()
    ax2.plot(
        range(n_buckets),
        median_errs,
        color="red",
        linewidth=1.5,
        label="Median Q-Error",
    )
    ax2.set_ylabel("Median Q-Error (Lower is Better)", color="red")
    ax2.set_yscale("log")  # Q-Error is often best viewed in Log scale
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title(
        f"Bucket-Level Debugging: Error vs. Density\n(Total Buckets: {n_buckets})"
    )
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
    scaling_factor = N / len(sample)
    estimated_freq = obs_freq * scaling_factor

    # Build Initial Buckets
    ew_hist = EquiWidthHistogram.build_from_sample(
        mn=mn, mx=mx, bins=n_bins, sample=sample, total_rows=N
    )

    # Train Hybrid initial state
    print(f"Training Hybrid Model on 100000 Sample (Bins: {n_bins})...")
    hybrid_est = HybridEstimator(
        ew_hist.buckets,
        identity_threshold=args.ident,
        mlp_penalty=args.penalty,
        fourier_penalty=args.penalty,
    )

    # Capture initial training time
    t_initial_train_start = time.perf_counter()
    hybrid_est.train(estimated_freq, mn, args.points, rng)
    t_initial_train = time.perf_counter() - t_initial_train_start

    # Load Workload
    queries, y_true = load_and_filter_workload(args.workload, mn, mx, freq)
    n_queries = len(queries)

    # --- BATCHING & TIMING SETUP ---
    batch_size = max(1, n_queries // 10)  # 10% chunks
    y_pred_all = np.zeros(n_queries)

    total_infer_time = 0.0
    total_fine_tune_time = 0.0

    print(f"\n>>> Starting Incremental Learning (Batches of {batch_size}) <<<")

    for i in range(0, n_queries, batch_size):
        end = min(i + batch_size, n_queries)
        q_batch = queries[i:end]
        y_true_batch = y_true[i:end]

        # 1. MEASURE INFERENCE
        t_inf_start = time.perf_counter()
        y_pred_counts = hybrid_est.predict_batch(q_batch)
        total_infer_time += time.perf_counter() - t_inf_start

        y_pred_all[i:end] = y_pred_counts / N

        # 2. MEASURE FINE-TUNING (Online Training)
        t_ft_start = time.perf_counter()
        hybrid_est.feedback_update(
            queries=q_batch,
            y_true_counts=y_true_batch * N,  # Signature Match
            y_pred_counts=y_pred_counts,  # Signature Match
            N=N,
            error_threshold=1.5,
        )
        total_fine_tune_time += time.perf_counter() - t_ft_start

        print(f"Processed queries {i} to {end}... model refined.")

    debug_plot_path = out_dir / "bucket_error_debug.png"
    plot_bucket_debug(hybrid_est, queries, y_true, y_pred_all, debug_plot_path)

    # --- SUMMARIZATION ---
    m = summarize(y_true, y_pred_all, "Hybrid-Incremental")

    # In research, 'train_time' is usually Initial + Total Fine-Tuning
    final_train_time = t_initial_train + total_fine_tune_time

    metrics = {
        "train_time": final_train_time,
        "initial_train_time": t_initial_train,
        "online_fine_tune_time": total_fine_tune_time,
        "infer_time": total_infer_time,
        "median_q_error": m["QErr_median"],
        "p25_q_error": m["QErr_p25"],
        "p75_q_error": m["QErr_p75"],
        "p95_q_error": m["QErr_p95"],
        "avg_q_error": m["QErr_avg"],
    }

    print("-" * 30)
    print(f"Hybrid Final Results:")
    print(f"Median QErr: {metrics['median_q_error']:.4f}")
    print(f"Total Inference: {total_infer_time:.4f}s")
    print(f"Total Training (Initial + Online): {final_train_time:.4f}s")
    print("-" * 30)

    save_benchmark_results(
        out_dir,
        Path(args.workload).stem,
        "Hybrid",
        queries,
        y_true,
        y_pred_all,
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
