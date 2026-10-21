#!/usr/bin/env python3
"""
TKHist под дрейфом данных. Протокол как в run_equiwidth_drift.py: сетка
бакетов фиксируется на шаге 0, на каждом шаге гистограмма пересобирается по
новой выборке (симуляция ANALYZE) — для TKHist это значит, что обновляются и
счётчики бакетов, и top-k значения. Онлайн-обучения от фидбэка у TKHist нет.
"""
import time
import pickle
import argparse
import re
import numpy as np
import pandas as pd
from pathlib import Path
from models import TKHistogram, summarize
from benchmark_utils import load_and_filter_workload, setup_out_dir
from data.datasets import DatasetManager


def main():
    parser = argparse.ArgumentParser(description="TKHist Analysis under Data Drift")
    parser.add_argument("--dataset", type=str, default="60000000")
    parser.add_argument("--workload", type=str, required=True)
    parser.add_argument("--init-dist", type=str, default="normal")
    parser.add_argument("--target-dist", type=str, default="zipf")
    parser.add_argument("--topk", type=int, default=5, help="Top-k values per bucket")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--shift-workload", action="store_true", help="If set, uses step-wise workloads from shift directory instead of static workload.")
    parser.add_argument("--shift-dir", type=str, required=False, help="Explicit path to shift dataset directory")
    parser.add_argument("--static-workload-path", type=str, required=False, help="Explicit path to static workload")
    args = parser.parse_args()

    ds_out_name = f"gradual_{args.init_dist}_to_{args.target_dist}_TKHist"
    out_dir = setup_out_dir(args, ds_out_name)

    dm = DatasetManager()

    if args.shift_dir:
        shift_dir = Path(args.shift_dir)
    else:
        shift_dir = dm.base_path / "generated" / args.dataset / f"shift_{args.init_dist}_to_{args.target_dist}_5%_dataset"

    if not shift_dir.exists():
        raise FileNotFoundError(f"Directory {shift_dir} not found. Run generate_shifts.py first.")

    # ЗАГРУЗКА ШАГА 0 (Инициализация сетки бакетов)
    with open(shift_dir / "step_0.pkl", "rb") as f:
        i_mn, i_mx, i_N, i_freq, i_sample, i_k = pickle.load(f)

    GLOBAL_MIN, GLOBAL_MAX = 0, 1_000_000

    # Ожидаем count в args.workload
    workload_count_str = args.workload
    if workload_count_str.endswith(".csv"):
        match = re.search(r"\d+", args.workload)
        if match:
            workload_count_str = match.group(0)

    # --- Определение статического ворклоада (Fallback) ---
    if args.static_workload_path:
        static_workload_path = Path(args.static_workload_path)
    else:
        static_workload_path = Path(args.workload)
        if not static_workload_path.exists():
            init_dist_dir = dm.base_path / "generated" / args.dataset / args.init_dist
            static_workload_path = init_dist_dir / f"workload_driven_{workload_count_str}.csv"
            if not static_workload_path.exists():
                print(f"Warning: Static workload not found at {static_workload_path}")

    if not args.shift_workload:
        print(f"Using static workload: {static_workload_path}")
        queries, _ = load_and_filter_workload(str(static_workload_path), GLOBAL_MIN, GLOBAL_MAX, i_freq)
    else:
        print(f"Using dynamic shift workloads with count: {workload_count_str}")

    summary_records = []

    step_files = list(shift_dir.glob("step_*.pkl"))
    num_steps = len(step_files)
    if num_steps == 0:
        raise ValueError(f"No step_*.pkl files found in {shift_dir}")
    divisor = max(1, num_steps - 1)

    # ПРОГОН ПО ВСЕМ ШАГАМ ДРИФТА
    for step in range(num_steps):
        shift_pct = step / divisor
        with open(shift_dir / f"step_{step}.pkl", "rb") as f:
            step_mn, step_mx, current_N, current_freq, mixed_sample, step_k = pickle.load(f)

        if args.shift_workload:
            # Загружаем ворклоад для ТЕКУЩЕГО шага (динамический drift)
            step_workload_path = shift_dir / f"step_{step}_workload_driven_{workload_count_str}.csv"
            if not step_workload_path.exists():
                raise FileNotFoundError(f"Missing workload for step {step}: {step_workload_path}. Run workload.py generator first.")

            queries, _ = load_and_filter_workload(str(step_workload_path), GLOBAL_MIN, GLOBAL_MAX, current_freq)

        # 1. Считаем реальность (Engine Truth)
        ps = np.cumsum(current_freq)
        y_true_counts = np.array([
            max(1.0, ps[min(len(current_freq) - 1, max(0, q.high - step_mn))] -
                (ps[max(0, q.low - 1 - step_mn)] if q.low > step_mn else 0))
            for q in queries
        ])
        y_true_sel = y_true_counts / current_N

        # 2. Пересборка по выборке шага на сетке шага 0 (Симуляция ANALYZE)
        t0_build = time.perf_counter()
        tk_hist = TKHistogram.build_from_sample(
            i_mn, i_mx, i_k, mixed_sample, current_N, top_k=args.topk
        )
        build_time = time.perf_counter() - t0_build

        # 3. Оценка (Prediction)
        t0 = time.perf_counter()
        y_pred_counts = []
        infer_times = []
        for q in queries:
            st = time.perf_counter()
            pred = tk_hist.predict(q)
            infer_times.append(time.perf_counter() - st)
            y_pred_counts.append(pred)

        y_pred_counts = np.maximum(np.array(y_pred_counts), 1.0)
        infer_time = time.perf_counter() - t0

        infer_times = np.array(infer_times)
        median_infer_time = np.median(infer_times)
        avg_infer_time = np.mean(infer_times)
        p95_infer_time = np.percentile(infer_times, 95)

        m = summarize(y_true_sel, y_pred_counts / current_N, f"TKHist_{shift_pct:.1f}")

        print(f"Shift {shift_pct:>4.0%}: Median QErr = {m['QErr_median']:.2f}")

        summary_records.append({
            "Shift %": f"{shift_pct:.0%}",
            "Median Q-Error": m["QErr_median"],
            "P95 Q-Error": m["QErr_p95"],
            "Avg Q-Error": m["QErr_avg"],
            "Inference Time": infer_time,
            "Median Inference Time": median_infer_time,
            "Avg Inference Time": avg_infer_time,
            "P95 Inference Time": p95_infer_time,
            "Build Time": build_time
        })

    pd.DataFrame(summary_records).to_csv(out_dir / "tkhist_drift_analysis.csv", index=False)
    print(f"\nExperiment completed. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
