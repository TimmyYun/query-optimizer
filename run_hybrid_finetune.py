#!/usr/bin/env python3
import time
import numpy as np
import pandas as pd
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
    scaling_factor = N / len(sample)
    estimated_freq = obs_freq * scaling_factor
    # --------------------------------------------------

    # Build Initial Buckets
    ew_hist = EquiWidthHistogram.build_from_sample(
        mn=mn, mx=mx, bins=n_bins, sample=sample, total_rows=N
    )

    # Train Hybrid
    print(f"Training Initial Hybrid Model on 100000 Sample (Bins: {n_bins})...")
    hybrid_est = HybridEstimator(
        ew_hist.buckets,
        identity_threshold=args.ident,
        mlp_penalty=args.penalty,
        fourier_penalty=args.penalty,
    )

    t_train = hybrid_est.train(estimated_freq, mn, args.points, rng)

    model_report_path = out_dir / "model_selection_Hybrid.txt"
    hybrid_est.report_models(file_path=model_report_path)

    # Load Workload
    queries, y_true_sel = load_and_filter_workload(args.workload, mn, mx, freq)
    y_true_counts = y_true_sel * N  # Convert selectivity to absolute row counts

    # ==========================================
    # ONLINE FINETUNING LOOP (Batch processing)
    # ==========================================
    n_queries = len(queries)
    batch_size = args.batch_size

    start_msg = f"\nStarting Online Workload Execution ({n_queries} queries, batch size: {batch_size})..."
    print(start_msg)

    all_pred_counts = []
    all_infer_times = []
    total_infer_time = 0.0
    total_ft_time = 0.0

    # Списки для сохранения данных в CSV
    batch_metrics = []

    ft_log_path = out_dir / "finetune_log.txt"
    with open(ft_log_path, "w") as ft_log:
        ft_log.write(start_msg + "\n\n")

        for start_idx in range(0, n_queries, batch_size):
            end_idx = min(start_idx + batch_size, n_queries)
            batch_q = queries[start_idx:end_idx]
            batch_y_true = y_true_counts[start_idx:end_idx]

            # 1. Predict (What the DB sees before execution)
            t0 = time.perf_counter()
            batch_pred = []
            for q in batch_q:
                st = time.perf_counter()
                pred = hybrid_est.predict(q)
                all_infer_times.append(time.perf_counter() - st)
                batch_pred.append(pred)
            batch_pred = np.array(batch_pred)
            total_infer_time += time.perf_counter() - t0

            # --- FIX: SANITY FLOOR (Никогда не предсказываем 0 строк) ---
            batch_pred = np.maximum(batch_pred, 1.0)
            batch_y_true = np.maximum(batch_y_true, 1.0)
            # ------------------------------------------------------------

            all_pred_counts.extend(batch_pred)

            # Print batch metrics to see improvement in real-time
            batch_qerrs = np.maximum(
                batch_y_true / (batch_pred + 1e-9), batch_pred / (batch_y_true + 1e-9)
            )

            med_err = np.median(batch_qerrs)
            p95_err = np.percentile(batch_qerrs, 95)
            batch_num = start_idx // batch_size + 1

            log_line = f"Batch {batch_num:02d} | Queries: {start_idx}-{end_idx} | Median Q-Error: {med_err:.2f} | P95 Q-Error: {p95_err:.2f}"

            print(log_line)
            ft_log.write(log_line + "\n")

            # Сохраняем для CSV
            batch_metrics.append(
                {
                    "Batch": batch_num,
                    "Start_Idx": start_idx,
                    "End_Idx": end_idx,
                    "Median_QErr": med_err,
                    "P95_QErr": p95_err,
                }
            )

            # 2. Execute & Learn (Feedback Update)
            t_ft0 = time.perf_counter()
            hybrid_est.feedback_update(
                batch_q,
                batch_y_true,
                batch_pred,
                N=N,
                error_threshold=args.ft_threshold,
                alpha=args.ft_alpha,
            )
            total_ft_time += time.perf_counter() - t_ft0

        summary_msg = f"\nTotal Inference Time: {total_infer_time:.4f}s\nTotal Finetuning Time: {total_ft_time:.4f}s"
        print(summary_msg)
        ft_log.write(summary_msg + "\n")

    # Сохраняем CSV для красивых графиков в будущем
    pd.DataFrame(batch_metrics).to_csv(out_dir / "finetune_metrics.csv", index=False)
    print(f"Finetune logs saved to {ft_log_path.name} and finetune_metrics.csv")

    # Convert overall predictions back to selectivity for standard evaluation
    y_pred_sel = np.array(all_pred_counts) / N

    # --- VISUALIZATION & SUMMARY ---
    debug_plot_path = out_dir / "finetuned_baseline_debug.png"
    plot_bucket_debug(hybrid_est, queries, y_true_sel, y_pred_sel, debug_plot_path)

    all_infer_times = np.array(all_infer_times)
    median_infer_time = np.median(all_infer_times)
    avg_infer_time = np.mean(all_infer_times)
    p95_infer_time = np.percentile(all_infer_times, 95)

    m = summarize(y_true_sel, y_pred_sel, "Hybrid_Online")
    metrics = {
        "train_time": t_train,
        "infer_time": total_infer_time,
        "median_infer_time": median_infer_time,
        "avg_infer_time": avg_infer_time,
        "p95_infer_time": p95_infer_time,
        "ft_time": total_ft_time,
        "median_q_error": m["QErr_median"],
        "p25_q_error": m["QErr_p25"],
        "p75_q_error": m["QErr_p75"],
        "p95_q_error": m["QErr_p95"],
        "avg_q_error": m["QErr_avg"],
    }

    print(f"\nOverall Online Experience: Median QErr={metrics['median_q_error']:.4f}")

    save_benchmark_results(
        out_dir,
        Path(args.workload).stem,
        "Hybrid",
        queries,
        y_true_sel,
        y_pred_sel,
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

    # --- NEW: Finetuning parameters ---
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of queries per evaluation batch",
    )
    parser.add_argument(
        "--ft-threshold",
        type=float,
        default=1.2,
        help="Q-Error threshold to trigger patch application",
    )
    parser.add_argument(
        "--ft-alpha",
        type=float,
        default=0.3,
        help="Learning rate (weight) for sigmoid patches",
    )

    args = parser.parse_args()
    run_benchmark_suite(args, run_logic)


if __name__ == "__main__":
    main()
