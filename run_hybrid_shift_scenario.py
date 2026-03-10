#!/usr/bin/env python3
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from models import EquiWidthHistogram, HybridEstimator, summarize
from benchmark_utils import load_and_filter_workload, save_benchmark_results, setup_out_dir


def load_dataset_meta(dataset_dir: str, dist: str):
    meta_path = Path(dataset_dir) / dist / "meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def get_sampled_freq(sample, mn, total_n, target_len):
    counts = np.bincount(sample - mn, minlength=target_len)
    scaling_factor = total_n / len(sample)
    return counts * scaling_factor


def plot_shift_state(true_freq, est_freq, shift_pct, out_dir, init_name, target_name):
    """Generates a histogram comparison of the current data state."""
    plt.figure(figsize=(12, 5))

    # We downsample the plot x-axis for performance (plotting 1M bars is slow)
    # Aggregating into 500 visual bins
    vis_bins = 500
    chunk_size = len(true_freq) // vis_bins

    true_vis = np.array([np.sum(true_freq[i:i + chunk_size]) for i in range(0, len(true_freq), chunk_size)])
    est_vis = np.array([np.sum(est_freq[i:i + chunk_size]) for i in range(0, len(est_freq), chunk_size)])
    x_axis = np.linspace(0, len(true_freq), len(true_vis))

    plt.fill_between(x_axis, true_vis, color='blue', alpha=0.3, label='True Distribution (60M)')
    plt.step(x_axis, est_vis, color='red', alpha=0.7, label='Estimated from Sample (100k)', where='mid')

    plt.title(f"Distribution Shift: {init_name} -> {target_name} ({shift_pct:.0%})")
    plt.xlabel("Domain Value")
    plt.ylabel("Frequency")
    plt.yscale('log')  # Log scale is essential for seeing Zipf tails
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    plot_path = out_dir / "plots"
    plot_path.mkdir(exist_ok=True)
    plt.savefig(plot_path / f"shift_{shift_pct:.1f}.png", dpi=150)
    plt.close()


def detect_bad_buckets(hybrid_est, queries, y_true_counts, y_pred_counts, error_threshold=1.5):
    bad_buckets = set()
    b_lo_vals = np.array([b.lo for b in hybrid_est.buckets])
    b_hi_vals = np.array([b.hi for b in hybrid_est.buckets])

    for idx, q in enumerate(queries):
        q_err = max(
            y_true_counts[idx] / (y_pred_counts[idx] + 1e-9),
            y_pred_counts[idx] / (y_true_counts[idx] + 1e-9),
        )
        if q_err > error_threshold:
            start_idx = np.searchsorted(b_hi_vals, q.low)
            end_idx = np.searchsorted(b_lo_vals, q.high, side="right")
            for i in range(start_idx, end_idx):
                bad_buckets.add(i)
    return list(bad_buckets)


def main():
    parser = argparse.ArgumentParser(description="Detailed Hybrid Adaptation Analysis")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--workload", type=str, required=True)
    parser.add_argument("--init-dist", type=str, default="normal")
    parser.add_argument("--target-dist", type=str, default="zipf")
    parser.add_argument("--points", type=int, default=500)
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)
    args = parser.parse_args()

    ds_out_name = f"gradual_{args.init_dist}_to_{args.target_dist}"
    out_dir = setup_out_dir(args, ds_out_name)
    rng = np.random.default_rng(42)

    i_mn, i_mx, i_N, i_freq, i_sample, i_k, _, _ = load_dataset_meta(args.dataset, args.init_dist)
    t_mn, t_mx, t_N, t_freq, t_sample, _, _, _ = load_dataset_meta(args.dataset, args.target_dist)
    GLOBAL_MIN, GLOBAL_MAX = 0, 1_000_000

    i_est_freq = get_sampled_freq(i_sample, i_mn, i_N, len(i_freq))
    print(f"\n>>> PHASE 1: Training on {args.init_dist} (Sample Only) <<<")
    init_hist = EquiWidthHistogram.build_from_sample(i_mn, i_mx, i_k, i_sample, i_N)
    hybrid_est = HybridEstimator(init_hist.buckets)
    hybrid_est.train(i_est_freq, i_mn, args.points, rng)

    queries, _ = load_and_filter_workload(args.workload, GLOBAL_MIN, GLOBAL_MAX, i_freq)
    summary_records = []

    for step in range(0, 11):
        shift_pct = step * 0.1
        current_N = (1.0 - shift_pct) * i_N + (shift_pct) * t_N
        current_freq = (1.0 - shift_pct) * i_freq + (shift_pct) * t_freq

        # Engine Side
        ps = np.cumsum(current_freq)
        y_true_counts = np.array(
            [max(1.0, ps[min(len(current_freq) - 1, q.high)] - (ps[q.low - 1] if q.low > 0 else 0)) for q in queries])
        y_true_sel = y_true_counts / current_N

        # Optimizer Side
        n_target = int(len(i_sample) * shift_pct)
        mixed_sample = np.concatenate([
            rng.choice(i_sample, len(i_sample) - n_target, replace=False),
            rng.choice(t_sample, n_target, replace=False),
        ])
        current_est_freq = get_sampled_freq(mixed_sample, i_mn, current_N, len(i_freq))

        # --- NEW: VISUALIZATION ---
        plot_shift_state(current_freq, current_est_freq, shift_pct, out_dir, args.init_dist, args.target_dist)

        y_pred_shock = hybrid_est.predict_batch(queries)
        m_shock = summarize(y_true_sel, y_pred_shock / current_N, f"Shock_{shift_pct:.1f}")

        m_rb, m_ft = m_shock, m_shock
        t_rb, t_ft, n_bad = 0.0, 0.0, 0

        if step > 0:
            bad_indices = detect_bad_buckets(hybrid_est, queries, y_true_counts, y_pred_shock)
            n_bad = len(bad_indices)
            if n_bad > 0:
                t0 = time.perf_counter()
                hybrid_est.train(current_est_freq, i_mn, args.points, rng, bucket_indices=bad_indices)
                t_rb = time.perf_counter() - t0
                m_rb = summarize(y_true_sel, hybrid_est.predict_batch(queries) / current_N)

            t1 = time.perf_counter()
            hybrid_est.feedback_update(queries, y_true_counts, hybrid_est.predict_batch(queries), current_N)
            t_ft = time.perf_counter() - t1
            m_ft = summarize(y_true_sel, hybrid_est.predict_batch(queries) / current_N)

        print(
            f"Shift {shift_pct:>4.0%}: [Shock: {m_shock['QErr_median']:.2f}] -> [RB: {m_rb['QErr_median']:.2f}] -> [FT: {m_ft['QErr_median']:.2f}] | RB: {n_bad}")

        summary_records.append(
            {
                "Shift %": f"{shift_pct:.0%}",
                "Shock_Med": m_shock["QErr_median"],
                "Shock_P95": m_shock["QErr_p95"],
                "RB_Med": m_rb["QErr_median"],
                "RB_P95": m_rb["QErr_p95"],
                "FT_Med": m_ft["QErr_median"],
                "FT_P95": m_ft["QErr_p95"],
                "RB_Time": t_rb,
                "FT_Time": t_ft,
                "Rebuilt_Count": n_bad,
            }
        )

    df = pd.DataFrame(summary_records)
    df.to_csv(out_dir / f"adaptation_metrics.csv", index=False)
    print(f"\nSaved plots and breakdown to {out_dir}")


if __name__ == "__main__":
    main()