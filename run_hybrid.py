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
    mn, mx, N, freq, _, n_bins_data, _, _ = metadata
    n_bins = args.buckets if args.buckets is not None else n_bins_data
    rng = np.random.default_rng(42)
    # --- STEP 1: SIMULATE REALISTIC SAMPLING ---
    # In a real DB, this is: SELECT * FROM table TABLESAMPLE BERNOULLI (1)
    sample_rate = 0.50  # 1% Sample
    sample_size = int(N * sample_rate)

    # Create a sampled frequency array (What a DB actually sees)
    indices = np.arange(len(freq))
    # Draw indices based on actual distribution to simulate a random row sample
    sampled_indices = rng.choice(indices, size=sample_size, p=(freq / N))

    # Build the 'observed' frequency array from the sample
    obs_freq = np.zeros_like(freq)
    unique, counts = np.unique(sampled_indices, return_counts=True)
    obs_freq[unique] = counts

    # Scale the sample back up to N (Crucial for Cardinality Estimation)
    scaling_factor = 1.0 / sample_rate
    estimated_freq = obs_freq * scaling_factor
    # --------------------------------------------

    # Build Initial Buckets using the ESTIMATED frequency
    # (Realistic: Histograms are almost always built from samples)
    ew_hist = EquiWidthHistogram.build(mn, mx, n_bins, estimated_freq)

    # Train Hybrid using the ESTIMATED frequency
    print(f"Training Hybrid Model on 1% Sample (Bins: {n_bins})...")
    hybrid_est = HybridEstimator(
        ew_hist.buckets,
        identity_threshold=args.ident,
        mlp_penalty=args.penalty,
        fourier_penalty=args.penalty,
    )

    # The models now learn from the 'estimated_freq'
    t_train = hybrid_est.train(estimated_freq, mn, args.points, rng)

    # Report Model Selections
    model_report_path = out_dir / "model_selection_Hybrid.txt"
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
        "--points", type=int, default=40, help="Points per bucket for training"
    )
    parser.add_argument("--ident", type=float, default=1e-4, help="Identity threshold")
    parser.add_argument(
        "--penalty", type=float, default=1.5, help="Model selection penalty"
    )
    args = parser.parse_args()
    run_benchmark_suite(args, run_logic)


if __name__ == "__main__":
    main()
