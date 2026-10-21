#!/usr/bin/env python3
"""
Entropy-based Histogram под дрейфом данных.

В статье нет ни обновлений, ни детектора дрейфа: гистограмма заморожена, а
пересборка — это новый build() по текущему снимку, вызванный харнессом
(спецификация §6.1). Поэтому на каждом шаге два замера на ворклоаде шага:
  1. Static  — гистограмма шага 0 без пересборки (контроль static_no_rebuild);
  2. Rebuild — build() с нуля по выборке шага (симуляция ANALYZE, как в
               run_equiwidth_drift.py / run_equidepth_drift.py).
"""
import time
import pickle
import argparse
import re
import numpy as np
import pandas as pd
from pathlib import Path
from models import EntropyHistogram1D, summarize
from benchmark_utils import load_and_filter_workload, setup_out_dir
from data.datasets import DatasetManager
from run_entropyhist import add_entropyhist_args, build_config, model_name_for


def evaluate(e_hist, queries):
    """Оценки (в строках, со сглаживанием max(., 1)) + время на запрос."""
    y_pred, infer_times = [], []
    for q in queries:
        st = time.perf_counter()
        y_pred.append(e_hist.predict(q))
        infer_times.append(time.perf_counter() - st)
    return np.array(y_pred), np.array(infer_times)


def stage_record(prefix, y_true_counts, y_pred_counts, infer_times, current_N, label):
    m = summarize(
        np.maximum(y_true_counts, 1.0) / current_N,
        np.maximum(y_pred_counts, 1.0) / current_N,
        label,
    )
    return m, {
        f"{prefix}_Med": m["QErr_median"],
        f"{prefix}_P95": m["QErr_p95"],
        f"{prefix}_Avg": m["QErr_avg"],
        f"{prefix}_RMSE": float(np.sqrt(np.mean((y_true_counts - y_pred_counts) ** 2))),
        f"{prefix}_Infer_Med": np.median(infer_times),
        f"{prefix}_Infer_Avg": np.mean(infer_times),
        f"{prefix}_Infer_P95": np.percentile(infer_times, 95),
    }


def main():
    parser = argparse.ArgumentParser(description="Entropy-based Histogram under Data Drift (static vs. rebuild)")
    parser.add_argument("--dataset", type=str, default="60000000")
    parser.add_argument("--workload", type=str, required=True)
    parser.add_argument("--init-dist", type=str, default="normal")
    parser.add_argument("--target-dist", type=str, default="zipf")
    parser.add_argument("--buckets", type=int, default=None, help="Bucket budget B (default: FD bins of step 0, as other models)")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--shift-workload", action="store_true", help="If set, uses step-wise workloads from shift directory instead of static workload.")
    parser.add_argument("--shift-dir", type=str, required=False, help="Explicit path to shift dataset directory")
    parser.add_argument("--static-workload-path", type=str, required=False, help="Explicit path to static workload")
    add_entropyhist_args(parser)
    args = parser.parse_args()

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

    # Бюджет B фиксируется на шаге 0 и не меняется при пересборках
    n_bins = args.buckets if args.buckets is not None else i_k
    config = build_config(args, n_bins)
    model_name = model_name_for(config)
    ds_out_name = f"gradual_{args.init_dist}_to_{args.target_dist}_{model_name}"
    out_dir = setup_out_dir(args, ds_out_name)

    GLOBAL_MIN, GLOBAL_MAX = 0, 1_000_000

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

    print(f"\n>>> Init {model_name}: B={n_bins}, {config}")
    t0_build = time.perf_counter()
    static_hist = EntropyHistogram1D.build_from_sample(i_sample, i_N, config)
    static_build_time = time.perf_counter() - t0_build

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
            step_workload_path = shift_dir / f"step_{step}_workload_driven_{workload_count_str}.csv"
            if not step_workload_path.exists():
                raise FileNotFoundError(f"Missing workload for step {step}: {step_workload_path}. Run workload.py generator first.")

            queries, _ = load_and_filter_workload(str(step_workload_path), GLOBAL_MIN, GLOBAL_MAX, current_freq)

        # 1. Считаем реальность (Engine Truth) — без max(1, .), он применяется в stage_record
        ps = np.cumsum(current_freq)
        y_true_counts = np.array([
            ps[min(len(current_freq) - 1, max(0, q.high - step_mn))] -
            (ps[max(0, q.low - 1 - step_mn)] if q.low > step_mn else 0)
            for q in queries
        ], dtype=np.float64)

        # 2. Static: гистограмма шага 0 (замороженная, как в статье)
        y_pred_static, infer_static = evaluate(static_hist, queries)
        m_static, rec_static = stage_record(
            "Static", y_true_counts, y_pred_static, infer_static, current_N, f"Static_{shift_pct:.2f}"
        )

        # 3. Rebuild: build() с нуля по выборке шага (харнесс, не статья)
        t0_build = time.perf_counter()
        rebuilt_hist = EntropyHistogram1D.build_from_sample(mixed_sample, current_N, config)
        rebuild_time = time.perf_counter() - t0_build

        y_pred_rb, infer_rb = evaluate(rebuilt_hist, queries)
        m_rb, rec_rb = stage_record(
            "Rebuild", y_true_counts, y_pred_rb, infer_rb, current_N, f"Rebuild_{shift_pct:.2f}"
        )

        print(
            f"Shift {shift_pct:>4.0%}: [Static: {m_static['QErr_median']:.2f}] -> "
            f"[Rebuild: {m_rb['QErr_median']:.2f}] | B={rebuilt_hist.B_effective} | Build={rebuild_time:.3f}s"
        )

        summary_records.append({
            "Shift %": f"{shift_pct:.0%}",
            **rec_static,
            **rec_rb,
            "Rebuild_Time": rebuild_time,
            "Rebuild_Partition_Time": rebuilt_hist.timings["partition_time"],
            "B_Effective": rebuilt_hist.B_effective,
            "Distinct_In_Sample": rebuilt_hist.distinct_count,
            "Build_Time": static_build_time if step == 0 else 0.0,
        })

    pd.DataFrame(summary_records).to_csv(out_dir / "entropyhist_drift_analysis.csv", index=False)
    print(f"\nExperiment completed. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
