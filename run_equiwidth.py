#!/usr/bin/env python3
import time
import numpy as np
from models import EquiWidthHistogram, summarize
from benchmark_utils import (
    get_common_parser, run_benchmark_suite,
    load_and_filter_workload, save_benchmark_results
)

def run_logic(args, out_dir, metadata):
    mn, mx, N, freq, sample, n_bins, skew, kurt = metadata
    
    # Build Histogram
    t0 = time.perf_counter()
    ew_hist = EquiWidthHistogram.build(mn, mx, n_bins, freq)
    build_time = time.perf_counter() - t0
    
    # Load Workload
    queries, y_true = load_and_filter_workload(args.eval_n, mn, mx, freq)
    
    # Evaluate
    t0 = time.perf_counter()
    y_pred_counts = ew_hist.predict_batch(queries)
    y_pred = y_pred_counts / N
    infer_time = time.perf_counter() - t0
    
    # Summarize
    m = summarize(y_true, y_pred, "Equi-Width")
    metrics = {
        "build_time": build_time,
        "infer_time": infer_time,
        "median_q_error": m['QErr_median'],
        "p25_q_error": m['QErr_p25'],
        "p75_q_error": m['QErr_p75'],
        "p95_q_error": m['QErr_p95'],
        "avg_q_error": m['QErr_avg']
    }
    
    print(f"Equi-Width: Median QErr={metrics['median_q_error']:.4f}, Build={build_time:.4f}s, Inf={infer_time:.4f}s")
    
    save_benchmark_results(out_dir, args.eval_n, "EquiWidth", queries, y_true, y_pred, metrics)

def main():
    parser = get_common_parser("Run EquiWidth Baseline Benchmark")
    args = parser.parse_args()
    run_benchmark_suite(args, run_logic)

if __name__ == "__main__":
    main()
