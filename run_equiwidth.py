#!/usr/bin/env python3
import time
import numpy as np
from pathlib import Path
from models import EquiWidthHistogram, summarize
from benchmark_utils import (
    get_common_parser,
    run_benchmark_suite,
    load_and_filter_workload,
    save_benchmark_results,
)


def run_logic(args, out_dir, metadata):
    mn, mx, N, freq, sample, n_bins_data, skew, kurt = metadata
    n_bins = args.buckets if args.buckets is not None else n_bins_data

    # --- NEW: Автоматическое определение пути к ворклоаду ---
    workload_path = Path(args.workload)
    if not workload_path.exists():
        # Если передан не путь, а число (count), ищем внутри папки датасета
        potential_path = Path(args.dataset) / f"workload_driven_{args.workload}.csv"
        if potential_path.exists():
            workload_path = potential_path
        else:
            raise FileNotFoundError(
                f"Workload not found at {args.workload} or {potential_path}"
            )
    # -------------------------------------------------------

    # Build Histogram
    t0 = time.perf_counter()
    ew_hist = EquiWidthHistogram.build_from_sample(
        mn=mn, mx=mx, bins=n_bins, sample=sample, total_rows=N
    )
    build_time = time.perf_counter() - t0

    # Load Workload
    queries, y_true_sel = load_and_filter_workload(str(workload_path), mn, mx, freq)
    y_true_counts = y_true_sel * N

    # Evaluate
    t0 = time.perf_counter()
    y_pred_counts = np.array([ew_hist.predict(q) for q in queries])
    infer_time = time.perf_counter() - t0

    y_pred_counts = np.maximum(y_pred_counts, 1.0)
    y_true_counts = np.maximum(y_true_counts, 1.0)

    y_pred = y_pred_counts / N
    y_true_safe = y_true_counts / N

    m = summarize(y_true_safe, y_pred, "Equi-Width")
    metrics = {
        "build_time": build_time,
        "infer_time": infer_time,
        "median_q_error": m["QErr_median"],
        "p25_q_error": m["QErr_p25"],
        "p75_q_error": m["QErr_p75"],
        "p95_q_error": m["QErr_p95"],
        "avg_q_error": m["QErr_avg"],
    }

    print(
        f"Equi-Width: Median QErr={metrics['median_q_error']:.4f}, Build={build_time:.4f}s"
    )

    save_benchmark_results(
        out_dir, workload_path.stem, "EquiWidth", queries, y_true_safe, y_pred, metrics
    )


def main():
    parser = get_common_parser("Run EquiWidth Baseline Benchmark")
    args = parser.parse_args()
    run_benchmark_suite(args, run_logic)


if __name__ == "__main__":
    main()
