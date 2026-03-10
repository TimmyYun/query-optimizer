#!/usr/bin/env python3
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from models import EquiWidthHistogram, HybridEstimator, summarize
from benchmark_utils import load_and_filter_workload, save_benchmark_results


def load_dataset_meta(dataset_dir: str, dist: str):
    meta_path = Path(dataset_dir) / dist / "meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="Run Gradual Hybrid Model Shift Benchmark")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--workload", type=str, required=True)
    parser.add_argument("--init-dist", type=str, default="normal")
    parser.add_argument("--target-dist", type=str, default="zipf")
    parser.add_argument("--points", type=int, default=200)
    args = parser.parse_args()

    out_dir = Path(f"results/gradual_shift_{args.init_dist}_to_{args.target_dist}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    # 1. Load both datasets
    print(f"Loading {args.init_dist} (Source) and {args.target_dist} (Target)...")
    i_mn, i_mx, i_N, i_freq, _, i_k, _, _ = load_dataset_meta(args.dataset, args.init_dist)
    t_mn, t_mx, t_N, t_freq, _, _, _, _ = load_dataset_meta(args.dataset, args.target_dist)

    # Ensure domains match for mixing (pad if necessary)
    # Note: DOMAIN_MAX is usually 1,000,000 in your code, so freq arrays should be same size.

    # 2. Initial Training Phase
    print(f"\n>>> PHASE 1: Training on pure {args.init_dist} <<<")
    init_hist = EquiWidthHistogram.build(i_mn, i_mx, i_k, i_freq)
    hybrid_est = HybridEstimator(init_hist.buckets)
    hybrid_est.train(i_freq, i_mn, args.points, rng)

    # Load combined workload
    queries, y_true_init = load_and_filter_workload(args.workload, i_mn, i_mx, i_freq)

    summary_records = []

    # 3. Gradual Shift Loop (10% to 100%)
    for step in range(1, 11):
        shift_pct = step * 0.1
        print(f"\n>>> ITERATION {step}: Shift = {shift_pct:.0%} <<<")

        # Mix Frequencies: Simulate 10% replacement
        # Mixed Freq = (Initial * 0.9) + (Target * 0.1) ... and so on
        current_freq = (1.0 - shift_pct) * i_freq + (shift_pct) * t_freq
        current_N = (1.0 - shift_pct) * i_N + (shift_pct) * t_N

        # IMPORTANT: Selectivity on mixed data requires a fresh ground truth for the workload
        # We re-filter the workload to get the 'True' selectivity of the mixed data
        _, y_true_mixed = load_and_filter_workload(args.workload, i_mn, i_mx, current_freq)

        # A. Measurement (Before any fine-tuning in this step)
        t0 = time.perf_counter()
        y_pred_raw = hybrid_est.predict_batch(queries)
        inf_time = time.perf_counter() - t0

        y_pred_sel = y_pred_raw / current_N
        m = summarize(y_true_mixed, y_pred_sel, f"Shift_{shift_pct:.1f}")

        # B. Feedback Fine-Tuning (The model learns from the shifted data)
        t_ft_start = time.perf_counter()
        hybrid_est.feedback_update(
            queries=queries,
            y_true_counts=y_true_mixed * current_N,
            y_pred_counts=y_pred_raw,
            N=current_N,
            error_threshold=1.5
        )
        ft_time = time.perf_counter() - t_ft_start

        print(f"Step {step} Results: Median QErr = {m['QErr_median']:.4f}, FT Time = {ft_time:.4f}s")

        summary_records.append({
            "Shift %": f"{shift_pct:.0%}",
            "Median QErr": m["QErr_median"],
            "95% QErr": m["QErr_p95"],
            "Avg QErr": m["QErr_avg"],
            "Inference Time": inf_time,
            "FineTune Time": ft_time
        })

    # 4. Save and Display Results
    df_summary = pd.DataFrame(summary_records)
    print("\n" + "=" * 80)
    print(f"GRADUAL SHIFT SUMMARY: {args.init_dist.upper()} -> {args.target_dist.upper()}")
    print("=" * 80)
    print(df_summary.to_string(index=False))

    summary_csv = out_dir / "gradual_shift_metrics.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"\nSummary saved to: {summary_csv}")


if __name__ == "__main__":
    main()