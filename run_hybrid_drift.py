#!/usr/bin/env python3
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from models import EquiWidthHistogram, HybridEstimator, summarize
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
    """Generates a histogram comparison of the current data state."""
    plt.figure(figsize=(12, 5))

    vis_bins = 500
    chunk_size = max(1, len(true_freq) // vis_bins)

    true_vis = np.array(
        [
            np.sum(true_freq[i : i + chunk_size])
            for i in range(0, len(true_freq), chunk_size)
        ]
    )
    est_vis = np.array(
        [
            np.sum(est_freq[i : i + chunk_size])
            for i in range(0, len(est_freq), chunk_size)
        ]
    )
    x_axis = np.linspace(0, len(true_freq), len(true_vis))

    plt.fill_between(
        x_axis, true_vis, color="blue", alpha=0.3, label="True Distribution (60M)"
    )
    plt.step(
        x_axis,
        est_vis,
        color="red",
        alpha=0.7,
        label="Estimated from Sample (100k)",
        where="mid",
    )

    plt.title(f"Distribution Shift: {init_name} -> {target_name} ({shift_pct:.0%})")
    plt.xlabel("Domain Value")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    plot_path = out_dir / "plots"
    plot_path.mkdir(exist_ok=True)
    plt.savefig(plot_path / f"shift_{shift_pct:.2f}.png", dpi=150)
    plt.close()


def detect_bad_buckets(hybrid_est, queries, y_true_counts, y_pred_counts, error_threshold=1.3):
    # Словарь: индекс бакета -> список ошибок запросов, которые его задели
    bucket_errors = {i: [] for i in range(len(hybrid_est.buckets))}

    b_lo_vals = np.array([b.lo for b in hybrid_est.buckets])
    b_hi_vals = np.array([b.hi for b in hybrid_est.buckets])

    # 1. Собираем все ошибки
    for idx, q in enumerate(queries):
        err = max(y_true_counts[idx] / (y_pred_counts[idx] + 1e-9),
                  y_pred_counts[idx] / (y_true_counts[idx] + 1e-9))

        # Находим бакеты для этого запроса
        s_idx = np.searchsorted(b_hi_vals, q.low)
        e_idx = np.searchsorted(b_lo_vals, q.high, side="right")

        for i in range(s_idx, e_idx):
            bucket_errors[i].append(err)

    # 2. Проверяем медиану для каждого бакета
    bad_buckets = []
    for i, errors in bucket_errors.items():
        if len(errors) > 5:  # Минимум 5 запросов, чтобы статистика была честной
            if np.median(errors) > error_threshold:
                bad_buckets.append(i)

    return bad_buckets

def main():
    parser = argparse.ArgumentParser(
        description="Detailed Hybrid Adaptation Analysis (FT -> RB)"
    )
    parser.add_argument("--dataset", type=str, default="60000000")
    parser.add_argument("--workload", type=str, required=True)
    parser.add_argument("--init-dist", type=str, default="normal")
    parser.add_argument("--target-dist", type=str, default="zipf")
    parser.add_argument("--points", type=int, default=200)
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--shift-workload", action="store_true", help="If set, uses step-wise workloads from shift directory instead of static workload.")
    args = parser.parse_args()

    ds_out_name = f"gradual_{args.init_dist}_to_{args.target_dist}_FT_RB"
    out_dir = setup_out_dir(args, ds_out_name)
    rng = np.random.default_rng(42)

    reports_dir = out_dir / "model_reports"
    reports_dir.mkdir(exist_ok=True)

    dm = DatasetManager()

    # --- Рабочий процесс с динамическим ворклоадом ---
    # Мы ожидаем, что в аргументе --workload передано число, например '1000'
    workload_count_str = args.workload
    if workload_count_str.endswith('.csv'):
        # на случай если был передан старый формат, вырежем цифры
        import re
        match = re.search(r'\d+', args.workload)
        if match:
            workload_count_str = match.group(0)
    # --- Определение статического ворклоада (Fallback) ---
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

    shift_dir = (
            dm.base_path
            / "generated"
            / args.dataset
            / f"shift_{args.init_dist}_to_{args.target_dist}_5%_dataset"
    )

    if not shift_dir.exists():
        raise FileNotFoundError(
            f"Directory {shift_dir} not found. Run generate_shifts.py first."
        )

    # ЗАГРУЗКА ШАГА 0 (Инициализация)
    with open(shift_dir / "step_0.pkl", "rb") as f:
        i_mn, i_mx, i_N, i_freq, i_sample, i_k = pickle.load(f)

    GLOBAL_MIN, GLOBAL_MAX = 0, 1_000_000

    if not args.shift_workload:
        queries, _ = load_and_filter_workload(str(static_workload_path), GLOBAL_MIN, GLOBAL_MAX, i_freq)
    
    i_est_freq = get_sampled_freq(i_sample, i_mn, i_N, len(i_freq))
    print(f"\n>>> PHASE 1: Initial Training on {args.init_dist} <<<")
    t0_build = time.perf_counter()
    init_hist = EquiWidthHistogram.build_from_sample(i_mn, i_mx, i_k, i_sample, i_N)
    hybrid_est = HybridEstimator(init_hist.buckets)
    hybrid_est.train(i_est_freq, i_mn, args.points, rng)
    build_time = time.perf_counter() - t0_build

    hybrid_est.report_models(file_path=reports_dir / "models_shift_0.0.txt")

    summary_records = []

    # ПРОГОН ПО ВСЕМ ШАГАМ ДРИФТА
    for step in range(21):
        shift_pct = step / 20.0

        with open(shift_dir / f"step_{step}.pkl", "rb") as f:
            step_mn, step_mx, current_N, current_freq, mixed_sample, step_k = (
                pickle.load(f)
            )

        if args.shift_workload:
            # Загружаем ворклоад для ТЕКУЩЕГО шага (динамический drift)
            step_workload_path = shift_dir / f"step_{step}_workload_driven_{workload_count_str}.csv"
            if not step_workload_path.exists():
                raise FileNotFoundError(f"Missing workload for step {step}: {step_workload_path}. Run workload.py generator first.")
            
            queries, _ = load_and_filter_workload(str(step_workload_path), GLOBAL_MIN, GLOBAL_MAX, current_freq)

        # 1. Считаем реальную массу запросов (Engine Truth)
        ps = np.cumsum(current_freq)
        y_true_counts = np.array(
            [
                max(
                    1.0,
                    ps[min(len(current_freq) - 1, max(0, q.high - step_mn))]
                    - (ps[max(0, q.low - 1 - step_mn)] if q.low > step_mn else 0),
                )
                for q in queries
            ]
        )

        # --- SANITY FLOOR: Реальность не может быть 0 ---
        y_true_counts = np.maximum(y_true_counts, 1.0)
        y_true_sel = y_true_counts / current_N

        current_est_freq = get_sampled_freq(
            mixed_sample, step_mn, current_N, len(current_freq)
        )

        # 2. Обновляем массу бакетов (Симуляция чтения нового сэмпла базой данных)
        ps_est = np.cumsum(current_est_freq)
        for b in hybrid_est.buckets:
            b_lo_idx = max(0, min(len(ps_est) - 1, b.lo - step_mn))
            b_hi_idx = max(0, min(len(ps_est) - 1, b.hi - step_mn))
            new_count = ps_est[b_hi_idx] - (ps_est[b_lo_idx - 1] if b_lo_idx > 0 else 0)
            b.count = max(0.0, new_count)
        hybrid_est._bake_vectorized_data()

        plot_shift_state(
            current_freq,
            current_est_freq,
            shift_pct,
            out_dir,
            args.init_dist,
            args.target_dist,
        )

        # ==========================================
        # СТАДИЯ 0: SHOCK (Как модель реагирует сразу)
        # ==========================================
        t0_eval = time.perf_counter()
        y_pred_shock = np.maximum(
            np.array([hybrid_est.predict(q) for q in queries]), 1.0
        )  # Sanity Floor
        eval_time = time.perf_counter() - t0_eval
        m_shock = summarize(
            y_true_sel, y_pred_shock / current_N, f"Shock_{shift_pct:.1f}"
        )

        m_ft, m_rb = m_shock, m_shock
        t_ft, t_rb, n_rebuilt, n_finetuned = 0.0, 0.0, 0, 0

        if step > 0:
            # ==========================================
            # СТАДИЯ 1: FINETUNE (Быстрая адаптация массы)
            # ==========================================
            t0 = time.perf_counter()
            n_finetuned = hybrid_est.feedback_update(queries, y_true_counts, y_pred_shock, current_N)
            t_ft = time.perf_counter() - t0

            y_pred_ft = np.maximum(np.array([hybrid_est.predict(q) for q in queries]), 1.0)
            m_ft = summarize(y_true_sel, y_pred_ft / current_N, f"FT_{shift_pct:.1f}")

            # ==========================================
            # СТАДИЯ 2: REBUILD (Фоновое переобучение сломанных бакетов)
            # ==========================================
            bad_indices = detect_bad_buckets(
                hybrid_est, queries, y_true_counts, y_pred_ft
            )
            n_rebuilt = len(bad_indices)

            if n_rebuilt > 0:
                t1 = time.perf_counter()

                # Заметь: здесь мы передаем current_est_freq!
                # Модель переобучится на СВЕЖИХ данных именно для сломанных бакетов.
                hybrid_est.train(
                    current_est_freq,
                    step_mn,
                    args.points,
                    rng,
                    bucket_indices=bad_indices,
                )

                t_rb = time.perf_counter() - t1
                y_pred_rb = np.maximum(np.array([hybrid_est.predict(q) for q in queries]), 1.0)
                m_rb = summarize(
                    y_true_sel, y_pred_rb / current_N, f"RB_{shift_pct:.1f}"
                )

            hybrid_est.report_models(
                file_path=reports_dir / f"models_shift_{shift_pct:.2f}.txt"
            )

        # Вывод в консоль
        print(
            f"Shift {shift_pct:>4.0%}: [Shock: {m_shock['QErr_median']:.2f}] -> "
            f"[FT: {m_ft['QErr_median']:.2f}] -> [RB: {m_rb['QErr_median']:.2f}] | FT: {n_finetuned} | Rebuilt: {n_rebuilt}"
        )

        summary_records.append(
            {
                "Shift %": f"{shift_pct:.0%}",
                "Shock_Med": m_shock["QErr_median"],
                "Shock_P95": m_shock["QErr_p95"],
                "Shock_Avg": m_shock["QErr_avg"],
                "Evaluation Time": eval_time,
                "FT_Med": m_ft["QErr_median"],
                "FT_P95": m_ft["QErr_p95"],
                "FT_Avg": m_ft["QErr_avg"],
                "Finetuned_Count": n_finetuned,
                "RB_Med": m_rb["QErr_median"],
                "RB_P95": m_rb["QErr_p95"],
                "RB_Avg": m_rb["QErr_avg"],
                "Rebuilt_Count": n_rebuilt,
                "FT_Time": t_ft,
                "RB_Time": t_rb,
                "Build_Time": build_time if step == 0 else 0.0,
            }
        )

    pd.DataFrame(summary_records).to_csv(out_dir / "adaptation_ft_rb.csv", index=False)


if __name__ == "__main__":
    main()
