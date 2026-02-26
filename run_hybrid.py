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

    # Build Initial Buckets
    ew_hist = EquiWidthHistogram.build(mn, mx, n_bins, freq)

    # Train Hybrid
    print(f"Training Hybrid Model (Bins: {n_bins})...")
    hybrid_est = HybridEstimator(
        ew_hist.buckets,
        identity_threshold=args.ident,
        mlp_penalty=args.penalty,
        fourier_penalty=args.penalty,
    )
    t_train = hybrid_est.train(freq, mn, args.points, rng)

    # Report Model Selections
    model_report_path = out_dir / f"model_selection_Hybrid.txt"
    hybrid_est.report_models(file_path=model_report_path)

    # Load Workload
    queries, y_true = load_and_filter_workload(args.workload, mn, mx, freq)

    # Evaluate
    t0 = time.perf_counter()
    y_pred_counts = hybrid_est.predict_batch(queries)
    y_pred = y_pred_counts / N
    infer_time = time.perf_counter() - t0

    # Summarize
    m = summarize(y_true, y_pred, "Hybrid")
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
        out_dir, Path(args.workload).stem, "Hybrid", queries, y_true, y_pred, metrics
    )


def main():
    parser = get_common_parser("Run Hybrid Model Benchmark")
    parser.add_argument(
        "--points", type=int, default=20, help="Points per bucket for training"
    )
    parser.add_argument("--ident", type=float, default=1e-4, help="Identity threshold")
    parser.add_argument(
        "--penalty", type=float, default=1.5, help="Model selection penalty"
    )
    args = parser.parse_args()
    run_benchmark_suite(args, run_logic)


if __name__ == "__main__":
    main()
