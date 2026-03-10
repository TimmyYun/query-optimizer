#!/usr/bin/env python3
import time
from pathlib import Path

import numpy as np
from models import EquiWidthHistogram, EquiHistLearner, summarize
from benchmark_utils import (
    get_common_parser,
    run_benchmark_suite,
    load_and_filter_workload,
    save_benchmark_results,
)


def run_logic(args, out_dir, metadata):
    mn, mx, N, freq, sample, n_bins_data, skew, kurt = metadata
    n_bins = args.buckets if args.buckets is not None else n_bins_data

    # Build Initial Histogram (Equi-Width)
    t0 = time.perf_counter()
    ew_hist = EquiWidthHistogram.build_from_sample(
        mn=mn, mx=mx, bins=n_bins, sample=sample, total_rows=N
    )
    initial_build_time = time.perf_counter() - t0

    eh_learner = EquiHistLearner(ew_hist.buckets, learning_rate=args.lr)

    # Load Workload
    queries, y_true_raw = load_and_filter_workload(args.workload, mn, mx, freq)

    # --- FIX: SANITY FLOOR для реальности ---
    # Реальных строк не может быть 0 (если запрос валидный)
    true_cardinalities = np.maximum(y_true_raw * N, 1.0)
    y_true_safe = true_cardinalities / N
    # ----------------------------------------

    # Evaluate sequentially (No Batching)
    y_pred = []
    inf_time_total = 0.0
    update_time_total = 0.0

    for q, truth in zip(queries, true_cardinalities):
        t0 = time.perf_counter()
        pred_val = eh_learner.predict(q)

        # --- FIX: SANITY FLOOR для предикта ---
        # База данных никогда не оценивает запрос в 0 строк
        pred_val = max(pred_val, 1.0)
        # ----------------------------------------

        inf_time_total += time.perf_counter() - t0

        y_pred.append(pred_val / N)

        t0 = time.perf_counter()
        eh_learner.update(q, float(truth), pred=float(pred_val))
        update_time_total += time.perf_counter() - t0

    y_pred = np.array(y_pred)

    # Summarize (Используем y_true_safe!)
    m = summarize(y_true_safe, y_pred, "EquiHist")
    metrics = {
        "initial_build_time": initial_build_time,
        "update_time_total": update_time_total,
        "total_train_time": initial_build_time + update_time_total,
        "infer_time": inf_time_total,
        "median_q_error": m["QErr_median"],
        "p25_q_error": m["QErr_p25"],
        "p75_q_error": m["QErr_p75"],
        "p95_q_error": m["QErr_p95"],
        "avg_q_error": m["QErr_avg"],
    }

    print(
        f"EquiHist: Median QErr={metrics['median_q_error']:.4f}, Train={metrics['total_train_time']:.4f}s, Inf={inf_time_total:.4f}s"
    )

    save_benchmark_results(
        out_dir, Path(args.workload).stem, "EquiHist", queries, y_true_safe, y_pred, metrics
    )


def main():
    parser = get_common_parser("Run EquiHist Benchmark")
    parser.add_argument("--lr", type=float, default=0.5, help="Learning Rate")
    args = parser.parse_args()
    run_benchmark_suite(args, run_logic)


if __name__ == "__main__":
    main()