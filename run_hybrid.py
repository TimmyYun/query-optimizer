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


def plot_bucket_debug(hybrid_est, queries, y_true_safe, y_pred_safe, output_path):
    """
    Визуализирует производительность каждого бакета.
    Использует уже очищенные (safe) данные предикта и реальности.
    """
    n_buckets = len(hybrid_est.buckets)
    bucket_counts = np.array([b.count for b in hybrid_est.buckets])

    # --- FIX: Теперь считаем ошибку по очищенным данным без 1e-9 ---
    # Так как y_true_safe и y_pred_safe уже >= 1.0/N, деления на 0 не будет
    q_errors = np.maximum(y_true_safe / y_pred_safe, y_pred_safe / y_true_safe)

    bucket_err_lists = [[] for _ in range(n_buckets)]
    for idx, q in enumerate(queries):
        err = q_errors[idx]
        for b_idx in range(n_buckets):
            b = hybrid_est.buckets[b_idx]
            # Проверяем пересечение запроса с бакетом
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
    ax1.set_ylabel("Estimated Row Count (Mass)", color="gray")

    ax2 = ax1.twinx()
    ax2.plot(
        range(n_buckets),
        median_errs,
        color="red",
        linewidth=1.5,
        label="Median Q-Error",
    )
    ax2.set_ylabel("Median Q-Error (Log Scale)", color="red")
    ax2.set_yscale("log")  # Логарифмическая шкала важна для Zipf

    plt.title(f"Bucket Analysis: Error vs. Density (Safe Mode)")
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Debug plot saved to {output_path}")

def run_logic(args, out_dir, metadata):
    mn, mx, N, freq, sample, n_bins_data, skew, kurt = metadata
    n_bins = args.buckets if args.buckets is not None else n_bins_data
    rng = np.random.default_rng(42)

    # --- NEW: Автоматическое определение пути к ворклоаду ---
    workload_path = Path(args.workload)
    if not workload_path.exists():
        potential_path = Path(args.dataset) / f"workload_{args.workload}.csv"
        if potential_path.exists():
            workload_path = potential_path
        else:
            raise FileNotFoundError(
                f"Workload not found at {args.workload} or {potential_path}"
            )
    # -------------------------------------------------------

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

    queries, y_true_sel = load_and_filter_workload(str(workload_path), mn, mx, freq)

    t0 = time.perf_counter()
    y_pred_counts = []
    infer_times = []
    for q in queries:
        st = time.perf_counter()
        pred = hybrid_est.predict(q)
        infer_times.append(time.perf_counter() - st)
        y_pred_counts.append(pred)
        
    y_pred_counts = np.maximum(np.array(y_pred_counts), 1.0)
    infer_time = time.perf_counter() - t0
    
    infer_times = np.array(infer_times)
    median_infer_time = np.median(infer_times)
    avg_infer_time = np.mean(infer_times)
    p95_infer_time = np.percentile(infer_times, 95)

    y_true_counts = np.maximum(y_true_sel * N, 1.0)
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
        "median_infer_time": median_infer_time,
        "avg_infer_time": avg_infer_time,
        "p95_infer_time": p95_infer_time,
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
        workload_path.stem,
        "Hybrid",
        queries,
        y_true_safe,
        y_pred_safe,
        metrics,
    )


def main():
    parser = get_common_parser("Run Hybrid Model Benchmark")
    parser.add_argument("--points", type=int, default=200, help="Points per bucket")
    parser.add_argument("--ident", type=float, default=1e-4, help="Identity threshold")
    parser.add_argument("--penalty", type=float, default=1.5, help="Model penalty")
    args = parser.parse_args()
    run_benchmark_suite(args, run_logic)


if __name__ == "__main__":
    main()
