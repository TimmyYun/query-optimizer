#!/usr/bin/env python3
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from models import EquiWidthHistogram, EquiHistLearner, summarize
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
    parser = argparse.ArgumentParser(description="Equi-Hist Adaptation Analysis")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--workload", type=str, required=True)
    parser.add_argument("--init-dist", type=str, default="normal")
    parser.add_argument("--target-dist", type=str, default="zipf")
    parser.add_argument("--buckets", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)
    args = parser.parse_args()

    ds_out_name = f"gradual_{args.init_dist}_to_{args.target_dist}_EH"
    out_dir = setup_out_dir(args, ds_out_name)

    # 1. Load Data
    i_mn, i_mx, i_N, i_freq, i_sample, i_k, _, _ = load_dataset_meta(
        args.dataset, args.init_dist
    )
    t_mn, t_mx, t_N, t_freq, t_sample, _, _, _ = load_dataset_meta(
        args.dataset, args.target_dist
    )
    GLOBAL_MIN, GLOBAL_MAX = 0, 1_000_000

    n_bins = args.buckets if args.buckets is not None else i_k

    print(f"\n>>> PHASE 1: Initial Training on {args.init_dist} <<<")
    # Base histogram from original sample
    init_hist = EquiWidthHistogram.build_from_sample(
        mn=i_mn, mx=i_mx, bins=n_bins, sample=i_sample, total_rows=i_N
    )
    eh_learner = EquiHistLearner(init_hist.buckets, learning_rate=args.lr)

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

        y_pred = []
        inf_time_total = 0.0
        update_time_total = 0.0

        # Evaluate and Learn on the stream of queries
        for q, truth in zip(queries, y_true_counts):
            t0 = time.perf_counter()
            pred_val = eh_learner.predict(q)
            inf_time_total += time.perf_counter() - t0

            y_pred.append(pred_val / current_N)

            t0 = time.perf_counter()
            eh_learner.update(q, float(truth), pred=float(pred_val))
            update_time_total += time.perf_counter() - t0

        y_pred = np.array(y_pred)
        m_eh = summarize(y_true_sel, y_pred, f"EH_{shift_pct:.1f}")

        print(
            f"Shift {shift_pct:>4.0%}: [Median QErr: {m_eh['QErr_median']:.2f}] (Infer+Update Time: {(inf_time_total + update_time_total):.4f}s)"
        )

        summary_records.append(
            {
                "Shift %": f"{shift_pct:.0%}",
                "EH_Med": m_eh["QErr_median"],
                "EH_P95": m_eh["QErr_p95"],
                "EH_Infer_Time": inf_time_total,
                "EH_Update_Time": update_time_total,
            }
        )

    pd.DataFrame(summary_records).to_csv(
        out_dir / f"adaptation_equihist.csv", index=False
    )


if __name__ == "__main__":
    main()
