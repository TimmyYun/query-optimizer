#!/usr/bin/env python3
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from models import EquiWidthHistogram, summarize
from benchmark_utils import (
    load_and_filter_workload,
    setup_out_dir,
)

def load_dataset_meta(dataset_dir: str, dist: str):
    meta_path = Path(dataset_dir) / dist / "meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    with open(meta_path, "rb") as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser(description="Equi-Width Adaptation Analysis")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--workload", type=str, required=True)
    parser.add_argument("--init-dist", type=str, default="normal")
    parser.add_argument("--target-dist", type=str, default="uniform")
    parser.add_argument("--buckets", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)
    args = parser.parse_args()

    ds_out_name = f"gradual_{args.init_dist}_to_{args.target_dist}_EW"
    out_dir = setup_out_dir(args, ds_out_name)
    rng = np.random.default_rng(42)

    # 1. Load Data
    i_mn, i_mx, i_N, i_freq, i_sample, i_k, _, _ = load_dataset_meta(args.dataset, args.init_dist)
    t_mn, t_mx, t_N, t_freq, t_sample, _, _, _ = load_dataset_meta(args.dataset, args.target_dist)
    GLOBAL_MIN, GLOBAL_MAX = 0, 1_000_000
    
    n_bins = args.buckets if args.buckets is not None else i_k

    print(f"\n>>> PHASE 1: Initial Training on {args.init_dist} <<<")
    queries, _ = load_and_filter_workload(args.workload, GLOBAL_MIN, GLOBAL_MAX, i_freq)
    
    summary_records = []

    for step in range(0, 11):
        shift_pct = step * 0.1
        current_N = (1.0 - shift_pct) * i_N + (shift_pct) * t_N
        current_freq = (1.0 - shift_pct) * i_freq + (shift_pct) * t_freq

        # Engine Truth
        ps = np.cumsum(current_freq)
        y_true_counts = np.array(
            [
                max(
                    1.0,
                    ps[min(len(current_freq) - 1, q.high)]
                    - (ps[q.low - 1] if q.low > 0 else 0),
                )
                for q in queries
            ]
        )
        y_true_sel = y_true_counts / current_N

        # Optimizer Sample (Drifted)
        n_target = int(len(i_sample) * shift_pct)
        mixed_sample = np.concatenate(
            [
                rng.choice(i_sample, len(i_sample) - n_target, replace=False),
                rng.choice(t_sample, n_target, replace=False),
            ]
        )

        # REBUILD at every drift step
        t0 = time.perf_counter()
        ew_hist = EquiWidthHistogram.build_from_sample(
            mn=i_mn, mx=i_mx, bins=n_bins, sample=mixed_sample, total_rows=current_N
        )
        t_rb = time.perf_counter() - t0

        y_pred_counts = ew_hist.predict_batch(queries)
        m_rb = summarize(y_true_sel, y_pred_counts / current_N, f"EW_Rebuild_{shift_pct:.1f}")

        print(
            f"Shift {shift_pct:>4.0%}: [RB: {m_rb['QErr_median']:.2f}] (Rebuild Time: {t_rb:.4f}s)"
        )

        summary_records.append(
            {
                "Shift %": f"{shift_pct:.0%}",
                "RB_Med": m_rb["QErr_median"],
                "RB_P95": m_rb["QErr_p95"],
                "RB_Time": t_rb,
            }
        )

    pd.DataFrame(summary_records).to_csv(out_dir / f"adaptation_equiwidth.csv", index=False)


if __name__ == "__main__":
    main()
