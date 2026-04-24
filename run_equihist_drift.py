#!/usr/bin/env python3
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from models import EquiWidthHistogram, EquiHistLearner, summarize
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


def main():
    parser = argparse.ArgumentParser(description="EquiHist Online Adaptation Analysis under Data Drift")
    parser.add_argument("--dataset", type=str, default="60000000")
    parser.add_argument("--workload", type=str, required=True)
    parser.add_argument("--init-dist", type=str, default="normal")
    parser.add_argument("--target-dist", type=str, default="zipf")
    parser.add_argument("--lr", type=float, default=0.5, help="Learning Rate for EquiHist")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--shift-workload", action="store_true", help="If set, uses step-wise workloads from shift directory instead of static workload.")
    parser.add_argument("--shift-dir", type=str, required=False, help="Explicit path to shift dataset directory")
    parser.add_argument("--static-workload-path", type=str, required=False, help="Explicit path to static workload")
    args = parser.parse_args()

    ds_out_name = f"gradual_{args.init_dist}_to_{args.target_dist}_EquiHist"
    out_dir = setup_out_dir(args, ds_out_name)

    dm = DatasetManager()
    
    if args.shift_dir:
        shift_dir = Path(args.shift_dir)
    else:
        shift_dir = dm.base_path / "generated" / args.dataset / f"shift_{args.init_dist}_to_{args.target_dist}_5%_dataset"

    if not shift_dir.exists():
        raise FileNotFoundError(f"Directory {shift_dir} not found. Run generate_shifts.py first.")

    # ЗАГРУЗКА ШАГА 0 (Инициализация)
    with open(shift_dir / "step_0.pkl", "rb") as f:
        i_mn, i_mx, i_N, i_freq, i_sample, i_k = pickle.load(f)

    GLOBAL_MIN, GLOBAL_MAX = 0, 1_000_000

    # Ожидаем count в args.workload (например '1000')
    workload_count_str = args.workload
    if workload_count_str.endswith('.csv'):
        import re
        match = re.search(r'\d+', args.workload)
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
    else:
        print(f"Using dynamic shift workloads with count: {workload_count_str}")
    # --------------------------------------------------------------

    if not args.shift_workload:
        queries, _ = load_and_filter_workload(str(static_workload_path), GLOBAL_MIN, GLOBAL_MAX, i_freq)

    # Строим начальную гистограмму и инициализируем Learner
    print(f"\n>>> PHASE 1: Initial Building on {args.init_dist} <<<")
    ew_hist = EquiWidthHistogram.build_from_sample(i_mn, i_mx, i_k, i_sample, i_N)
    eh_learner = EquiHistLearner(ew_hist.buckets, learning_rate=args.lr)

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

        # 2. Обновляем массу бакетов из сэмпла (Симуляция ANALYZE)
        # Это дает модели знать об общем изменении объема данных
        current_est_freq = get_sampled_freq(mixed_sample, step_mn, current_N, len(current_freq))
        ps_est = np.cumsum(current_est_freq)
        for b in eh_learner.buckets:
            b_lo_idx = max(0, min(len(ps_est) - 1, b.lo - step_mn))
            b_hi_idx = max(0, min(len(ps_est) - 1, b.hi - step_mn))
            new_count = ps_est[b_hi_idx] - (ps_est[b_lo_idx - 1] if b_lo_idx > 0 else 0)
            b.count = max(0.0, new_count)

        # 3. Онлайн-обучение на ворклоаде (как в run_equihist.py)
        y_pred = []
        infer_times = []
        t0_eval = time.perf_counter()

        for q, truth in zip(queries, y_true_counts):
            st = time.perf_counter()
            # Предсказание с защитой от 0
            pred_val = max(eh_learner.predict(q), 1.0)
            infer_times.append(time.perf_counter() - st)
            y_pred.append(pred_val / current_N)

            # Немедленное обновление (Feedback Loop)
            eh_learner.update(q, float(truth), pred=float(pred_val))

        total_time = time.perf_counter() - t0_eval
        y_pred = np.array(y_pred)
        
        infer_times = np.array(infer_times)
        median_infer_time = np.median(infer_times)
        avg_infer_time = np.mean(infer_times)
        p95_infer_time = np.percentile(infer_times, 95)

        # Метрики для текущего шага
        m = summarize(y_true_sel, y_pred, f"EquiHist_Step_{step}")
        print(f"Shift {shift_pct:>4.0%}: Median QErr = {m['QErr_median']:.2f}")

        summary_records.append({
            "Shift %": f"{shift_pct:.0%}",
            "Median Q-Error": m["QErr_median"],
            "P95 Q-Error": m["QErr_p95"],
            "Avg Q-Error": m["QErr_avg"],
            "Time": total_time,
            "Median Inference Time": median_infer_time,
            "Avg Inference Time": avg_infer_time,
            "P95 Inference Time": p95_infer_time,
        })

    pd.DataFrame(summary_records).to_csv(out_dir / "equihist_drift_analysis.csv", index=False)
    print(f"\nExperiment completed. Results saved to {out_dir}")


if __name__ == "__main__":
    main()