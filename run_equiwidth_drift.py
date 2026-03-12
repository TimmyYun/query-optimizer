#!/usr/bin/env python3
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from models import EquiWidthHistogram, summarize
from benchmark_utils import (
    load_and_filter_workload,
    save_benchmark_results,
    setup_out_dir,
)
from data.datasets import DatasetManager


def get_sampled_freq(sample, mn, total_n, target_len):
    clipped_sample = np.clip(sample - mn, 0, target_len - 1)
    counts = np.bincount(clipped_sample, minlength=target_len)
    scaling_factor = total_n / len(sample)
    return counts * scaling_factor


def plot_shift_state(true_freq, est_freq, shift_pct, out_dir, init_name, target_name):
    """Визуализация текущего состояния данных (аналогично гибридному скрипту)."""
    plt.figure(figsize=(12, 5))
    vis_bins = 500
    chunk_size = max(1, len(true_freq) // vis_bins)

    true_vis = np.array([np.sum(true_freq[i: i + chunk_size]) for i in range(0, len(true_freq), chunk_size)])
    est_vis = np.array([np.sum(est_freq[i: i + chunk_size]) for i in range(0, len(est_freq), chunk_size)])
    x_axis = np.linspace(0, len(true_freq), len(true_vis))

    plt.fill_between(x_axis, true_vis, color="blue", alpha=0.3, label="True Distribution")
    plt.step(x_axis, est_vis, color="red", alpha=0.7, label="Estimated (Sample)", where="mid")

    plt.title(f"EquiWidth Drift: {init_name} -> {target_name} ({shift_pct:.0%})")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    plot_path = out_dir / "plots"
    plot_path.mkdir(exist_ok=True)
    plt.savefig(plot_path / f"shift_{shift_pct:.1f}.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="EquiWidth Adaptation Analysis under Data Drift")
    parser.add_argument("--dataset", type=str, default="60000000")
    parser.add_argument("--workload", type=str, required=True)
    parser.add_argument("--init-dist", type=str, default="normal")
    parser.add_argument("--target-dist", type=str, default="zipf")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)
    args = parser.parse_args()

    ds_out_name = f"gradual_{args.init_dist}_to_{args.target_dist}_EquiWidth"
    out_dir = setup_out_dir(args, ds_out_name)

    dm = DatasetManager()
    shift_dir = dm.base_path / "generated" / args.dataset / f"shift_{args.init_dist}_to_{args.target_dist}"

    if not shift_dir.exists():
        raise FileNotFoundError(f"Directory {shift_dir} not found. Run generate_shifts.py first.")

    # ЗАГРУЗКА ШАГА 0 (Инициализация)
    with open(shift_dir / "step_0.pkl", "rb") as f:
        i_mn, i_mx, i_N, i_freq, i_sample, i_k = pickle.load(f)

    GLOBAL_MIN, GLOBAL_MAX = 0, 1_000_000

    # Определяем путь к ворклоаду (Data-Driven логика)
    workload_path = Path(args.workload)
    if not workload_path.exists():
        init_dist_dir = dm.base_path / "generated" / args.dataset / args.init_dist
        workload_path = init_dist_dir / f"workload_driven_{args.workload}.csv"

    print(f"Using workload: {workload_path}")
    queries, _ = load_and_filter_workload(str(workload_path), GLOBAL_MIN, GLOBAL_MAX, i_freq)

    # Строим начальную гистограмму
    print(f"\n>>> PHASE 1: Initial Building on {args.init_dist} <<<")
    ew_hist = EquiWidthHistogram.build_from_sample(i_mn, i_mx, i_k, i_sample, i_N)

    summary_records = []

    # ПРОГОН ПО ВСЕМ ШАГАМ ДРИФТА
    for step in range(21):
        shift_pct = step / 20.0
        with open(shift_dir / f"step_{step}.pkl", "rb") as f:
            step_mn, step_mx, current_N, current_freq, mixed_sample, step_k = pickle.load(f)

        # 1. Считаем реальность (Engine Truth)
        ps = np.cumsum(current_freq)
        y_true_counts = np.array([
            max(1.0, ps[min(len(current_freq) - 1, max(0, q.high - step_mn))] -
                (ps[max(0, q.low - 1 - step_mn)] if q.low > step_mn else 0))
            for q in queries
        ])
        y_true_sel = y_true_counts / current_N

        # 2. Обновляем массу бакетов (Симуляция ANALYZE в БД)
        current_est_freq = get_sampled_freq(mixed_sample, step_mn, current_N, len(current_freq))
        ps_est = np.cumsum(current_est_freq)

        for b in ew_hist.buckets:
            b_lo_idx = max(0, min(len(ps_est) - 1, b.lo - step_mn))
            b_hi_idx = max(0, min(len(ps_est) - 1, b.hi - step_mn))
            new_count = ps_est[b_hi_idx] - (ps_est[b_lo_idx - 1] if b_lo_idx > 0 else 0)
            b.count = max(0.0, new_count)

        plot_shift_state(current_freq, current_est_freq, shift_pct, out_dir, args.init_dist, args.target_dist)

        # 3. Оценка (Prediction)
        t0 = time.perf_counter()
        y_pred_counts = np.maximum(ew_hist.predict_batch(queries), 1.0)  # Sanity Floor
        infer_time = time.perf_counter() - t0

        m = summarize(y_true_sel, y_pred_counts / current_N, f"EquiWidth_{shift_pct:.1f}")

        print(f"Shift {shift_pct:>4.0%}: Median QErr = {m['QErr_median']:.2f}")

        summary_records.append({
            "Shift %": f"{shift_pct:.0%}",
            "Median Q-Error": m["QErr_median"],
            "P95 Q-Error": m["QErr_p95"],
            "Avg Q-Error": m["QErr_avg"],
            "Inference Time": infer_time
        })

    pd.DataFrame(summary_records).to_csv(out_dir / "equiwidth_drift_analysis.csv", index=False)
    print(f"\nExperiment completed. Results saved to {out_dir}")


if __name__ == "__main__":
    main()