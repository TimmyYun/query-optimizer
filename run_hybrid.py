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
            y_true=y_true_batch * N,
            y_pred=y_pred_counts,
            N=N,
            error_threshold=1.5,
        )
        total_fine_tune_time += time.perf_counter() - t_ft_start

        print(f"Processed queries {i} to {end}... model refined.")

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
