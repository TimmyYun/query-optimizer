#!/usr/bin/env python3
"""
ST-Histogram под дрейфом данных.

В отличие от EquiWidth/EquiHist/TKHist, здесь НЕТ симуляции ANALYZE: статья
запрещает модели смотреть на данные, поэтому выборка шага ей не передаётся.
Гистограмма инициализируется один раз (шаг 0, только N/min/max) и дальше
узнаёт о дрейфе исключительно из фидбэка запросов (§5.5).

На каждом шаге три замера на одном и том же ворклоаде шага:
  1. Shock  — оценка "замороженной" гистограммы, до фидбэка этого шага;
  2. Online — prequential: predict -> observe_feedback для каждого запроса;
  3. Post   — повторная оценка без обучения после прохода Online.
"""
import time
import pickle
import argparse
import re
import numpy as np
import pandas as pd
from pathlib import Path
from models import STHistogram, summarize
from benchmark_utils import load_and_filter_workload, setup_out_dir, exact_counts, run_queries, split_adapt_eval
from data.datasets import DatasetManager, DOMAIN_MAX
from run_sthist import add_sthist_args, build_config, model_name_for


def evaluate_frozen(st_hist, queries):
    """Оценки без обучения + время на запрос."""
    y_pred, infer_times, _ = run_queries(st_hist.predict, queries, np.zeros(len(queries)))
    return np.maximum(y_pred, 1.0), infer_times


def main():
    parser = argparse.ArgumentParser(description="ST-Histogram Feedback Adaptation under Data Drift")
    parser.add_argument("--dataset", type=str, default="60000000")
    parser.add_argument("--workload", type=str, required=True)
    parser.add_argument("--init-dist", type=str, default="normal")
    parser.add_argument("--target-dist", type=str, default="zipf")
    parser.add_argument("--buckets", type=int, default=None, help="Bucket budget B (default: FD bins of step 0, as other models)")
    parser.add_argument("--schema-domain", action="store_true", help=f"Use declared column domain [0, {DOMAIN_MAX}] instead of step-0 catalog min/max")
    parser.add_argument("--refresh-catalog", action="store_true", help="Refresh T from catalog at every step (affects only merge threshold m*T)")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--shift-workload", action="store_true", help="If set, uses step-wise workloads from shift directory instead of static workload.")
    parser.add_argument("--shift-dir", type=str, required=False, help="Explicit path to shift dataset directory")
    parser.add_argument("--static-workload-path", type=str, required=False, help="Explicit path to static workload")
    add_sthist_args(parser)
    args = parser.parse_args()

    model_name = model_name_for(args)
    ds_out_name = f"gradual_{args.init_dist}_to_{args.target_dist}_{model_name}"
    out_dir = setup_out_dir(args, ds_out_name)

    dm = DatasetManager()

    if args.shift_dir:
        shift_dir = Path(args.shift_dir)
    else:
        shift_dir = dm.base_path / "generated" / args.dataset / f"shift_{args.init_dist}_to_{args.target_dist}_5%_dataset"

    if not shift_dir.exists():
        raise FileNotFoundError(f"Directory {shift_dir} not found. Run generate_shifts.py first.")

    # ЗАГРУЗКА ШАГА 0 — модели нужны только метаданные (N, min, max, k)
    with open(shift_dir / "step_0.pkl", "rb") as f:
        i_mn, i_mx, i_N, i_freq, i_sample, i_k = pickle.load(f)

    GLOBAL_MIN = 0

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
        queries, _ = load_and_filter_workload(str(static_workload_path), i_mn, i_mx, i_freq)
    else:
        print(f"Using dynamic shift workloads with count: {workload_count_str}")

    # Инициализация (§3.1): равномерно из метаданных каталога
    dom_min, dom_max = (GLOBAL_MIN, DOMAIN_MAX) if args.schema_domain else (i_mn, i_mx)
    n_bins = args.buckets if args.buckets is not None else i_k
    config = build_config(args, n_bins)
    print(f"\n>>> Init {model_name}: B={n_bins}, T={i_N}, domain=[{dom_min}, {dom_max}], {config}")

    t0_build = time.perf_counter()
    st_hist = STHistogram(config, total_rows=i_N, min_value=dom_min, max_value=dom_max)
    build_time = time.perf_counter() - t0_build

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
            step_mn, step_mx, current_N, current_freq, _, _ = pickle.load(f)

        if args.shift_workload:
            step_workload_path = shift_dir / f"step_{step}_workload_driven_{workload_count_str}.csv"
            if not step_workload_path.exists():
                raise FileNotFoundError(f"Missing workload for step {step}: {step_workload_path}. Run workload.py generator first.")

            queries, _ = load_and_filter_workload(str(step_workload_path), step_mn, step_mx, current_freq)

        if args.refresh_catalog:
            st_hist.update_catalog_metadata(current_N)

        # 1. Считаем реальность (Engine Truth) — точный фидбэк, без max(1, .)
        actual_counts = exact_counts(queries, current_freq, step_mn)
        y_true_sel = np.maximum(actual_counts, 1.0) / current_N

        events_before = st_hist.events.copy()

        # Фидбэк — только adapt-половина; Shock и Post меряются на eval-половине,
        # которую модель не видит ни на одном шаге (split_adapt_eval)
        adapt_idx, eval_idx = split_adapt_eval(len(queries))
        adapt_queries = [queries[i] for i in adapt_idx]
        eval_queries = [queries[i] for i in eval_idx]

        # 2. Shock: устаревшая гистограмма, без фидбэка этого шага
        y_pred_shock, infer_shock = evaluate_frozen(st_hist, eval_queries)
        m_shock = summarize(y_true_sel[eval_idx], y_pred_shock / current_N, f"Shock_{shift_pct:.2f}")

        # 3. Online: predict -> feedback по adapt-половине (prequential: оценка до обучения на запросе)
        def feedback(q, actual, pred_count):
            st_hist.observe_feedback(q.low, q.high, actual)

        y_pred_online, infer_online, update_time = run_queries(
            st_hist.predict, adapt_queries, actual_counts[adapt_idx], feedback
        )
        y_pred_online = np.maximum(y_pred_online, 1.0)
        m_online = summarize(y_true_sel[adapt_idx], y_pred_online / current_N, f"Online_{shift_pct:.2f}")

        # 4. Post: eval-половина после адаптации, без обучения
        y_pred_post, infer_post = evaluate_frozen(st_hist, eval_queries)
        m_post = summarize(y_true_sel[eval_idx], y_pred_post / current_N, f"Post_{shift_pct:.2f}")

        step_events = st_hist.events - events_before

        print(
            f"Shift {shift_pct:>4.0%}: [Shock: {m_shock['QErr_median']:.2f}] -> "
            f"[Online: {m_online['QErr_median']:.2f}] -> [Post: {m_post['QErr_median']:.2f}] | "
            f"Mass/N: {st_hist.total_mass / current_N:.3f} | Events: {dict(step_events)}"
        )

        summary_records.append({
            "Shift %": f"{shift_pct:.0%}",
            "Shock_Med": m_shock["QErr_median"],
            "Shock_P95": m_shock["QErr_p95"],
            "Shock_Avg": m_shock["QErr_avg"],
            "Shock_Infer_Med": np.median(infer_shock),
            "Shock_Infer_Avg": np.mean(infer_shock),
            "Shock_Infer_P95": np.percentile(infer_shock, 95),
            "Online_Med": m_online["QErr_median"],
            "Online_P95": m_online["QErr_p95"],
            "Online_Avg": m_online["QErr_avg"],
            "Online_Infer_Med": np.median(infer_online),
            "Online_Infer_Avg": np.mean(infer_online),
            "Online_Infer_P95": np.percentile(infer_online, 95),
            "Post_Med": m_post["QErr_median"],
            "Post_P95": m_post["QErr_p95"],
            "Post_Avg": m_post["QErr_avg"],
            "Update_Time": update_time,
            "Restructures": step_events["restructure_applied"],
            "Restructure_Noops": sum(v for k, v in step_events.items() if k.startswith("restructure_noop")),
            "Unlearnable_Zero_Est": step_events["unlearnable_zero_estimate"],
            "Out_Of_Domain": step_events["out_of_domain"],
            "Mass_Ratio": st_hist.total_mass / current_N,
            "T_Reference": st_hist.T_reference,
            "Build_Time": build_time if step == 0 else 0.0,
        })

    pd.DataFrame(summary_records).to_csv(out_dir / "sthist_drift_analysis.csv", index=False)
    print(f"\nExperiment completed. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
