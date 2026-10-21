#!/usr/bin/env python3
"""
TKHist / EquiDepth / EntropyHist / STHist под дрейфом на DSB (CSV-чекпоинты churn).

Данные и ворклоад — как в run_dsb_equiwidth_drift.py; протокол каждой модели —
как в её синтетическом run_*_drift.py:
  tkhist      — на каждом шаге пересборка по выборке шага на сетке шага 0 (ANALYZE);
  equidepth   — полная пересборка MCV + histogram_bounds по выборке шага (ANALYZE);
  entropyhist — build() с нуля по выборке шага (Rebuild);
  sthist      — данные не видит: инициализация один раз (N, min, max), затем
                Shock (оценка до фидбэка шага) и Online (predict -> observe_feedback).
"""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_utils import load_and_filter_workload, setup_out_dir
from models import (
    EntropyHistogram1D,
    EquiDepthHistogram,
    HistogramConfig,
    STHistogram,
    STHistogramConfig,
    TKHistogram,
    summarize,
)
from run_dsb_equiwidth_drift import (
    CHURN_STEPS,
    COLUMN_TABLE,
    build_meta_from_values,
    load_csv_data,
)

MODEL_NAMES = {
    "tkhist": "TKHist",
    "equidepth": "EquiDepth",
    "entropyhist": "EntropyHist-MEA",
    "sthist": "STHist",
}


def evaluate(model, queries):
    preds, times = [], []
    for q in queries:
        st = time.perf_counter()
        preds.append(model.predict(q))
        times.append(time.perf_counter() - st)
    return np.maximum(np.array(preds), 1.0), np.array(times)


def stage_record(prefix, m, infer_times):
    return {
        f"{prefix}_Med": m["QErr_median"],
        f"{prefix}_P95": m["QErr_p95"],
        f"{prefix}_Avg": m["QErr_avg"],
        f"{prefix}_Infer_Med": float(np.median(infer_times)),
        f"{prefix}_Infer_Avg": float(np.mean(infer_times)),
    }


def main():
    parser = argparse.ArgumentParser(description="Baseline histograms under DSB drift")
    parser.add_argument("--model", choices=list(MODEL_NAMES), required=True)
    parser.add_argument("--column", type=str, required=True, choices=list(COLUMN_TABLE.keys()))
    parser.add_argument("--data-dir", type=str, default="data/dsb")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)
    args = parser.parse_args()

    model_name = MODEL_NAMES[args.model]
    table = COLUMN_TABLE[args.column]
    data_dir = Path(args.data_dir)
    col_dir = data_dir / args.column

    with open(col_dir / "meta.pkl", "rb") as f:
        i_mn, i_mx, i_N, i_freq, i_sample, i_k, *_ = pickle.load(f)

    workload_path = next(col_dir.glob("workload_driven_*.csv"))
    out_dir = setup_out_dir(args, f"dsb_{args.column}_{model_name}")
    queries, _ = load_and_filter_workload(str(workload_path), i_mn, i_mx, i_freq)
    print(f"{model_name} on {args.column}: {len(queries)} queries from {workload_path}")

    st_hist = None
    if args.model == "sthist":
        # Тот же бюджет бакетов, что у остальных (FD шага 0)
        st_hist = STHistogram(STHistogramConfig(bucket_count=i_k), total_rows=i_N, min_value=i_mn, max_value=i_mx)

    records = []
    all_steps = ["initial"] + [f"churn_{s}pct" for s in CHURN_STEPS]
    for step_idx, step_name in enumerate(all_steps):
        shift_pct = 0.0 if step_idx == 0 else CHURN_STEPS[step_idx - 1] / 100.0
        csv_file = data_dir / "drift" / f"{table}__{args.column}__{step_name}.csv"
        if not csv_file.exists():
            print(f"Warning: {csv_file} not found, skipping.")
            continue

        values = load_csv_data(csv_file)
        current_N, current_freq, current_sample = build_meta_from_values(values, i_mn, i_mx)

        ps = np.cumsum(current_freq)
        actual_counts = np.array(
            [
                ps[min(len(current_freq) - 1, max(0, q.high - i_mn))]
                - (ps[max(0, q.low - 1 - i_mn)] if q.low > i_mn else 0)
                for q in queries
            ],
            dtype=np.float64,
        )
        y_true_sel = np.maximum(actual_counts, 1.0) / current_N
        rec = {"Shift %": f"{shift_pct:.0%}", "Step": step_name}

        if args.model == "sthist":
            y_shock, t_shock = evaluate(st_hist, queries)
            m_shock = summarize(y_true_sel, y_shock / current_N, f"Shock_{step_name}")
            preds, times = [], []
            t0 = time.perf_counter()
            for q, actual in zip(queries, actual_counts):
                st = time.perf_counter()
                preds.append(max(st_hist.predict(q), 1.0))
                times.append(time.perf_counter() - st)
                st_hist.observe_feedback(q.low, q.high, float(actual))
            update_time = time.perf_counter() - t0
            m = summarize(y_true_sel, np.array(preds) / current_N, f"Online_{step_name}")
            rec.update(stage_record("Shock", m_shock, t_shock))
            rec.update(stage_record("Online", m, np.array(times)))
            rec["Update_Time"] = update_time
        else:
            t0 = time.perf_counter()
            if args.model == "tkhist":
                hist = TKHistogram.build_from_sample(i_mn, i_mx, i_k, current_sample, current_N)
            elif args.model == "equidepth":
                hist = EquiDepthHistogram.build_from_sample(current_sample, current_N)
            else:
                hist = EntropyHistogram1D.build_from_sample(current_sample, current_N, HistogramConfig(n_buckets=i_k))
            build_time = time.perf_counter() - t0
            y_pred, infer_times = evaluate(hist, queries)
            m = summarize(y_true_sel, y_pred / current_N, f"{model_name}_{step_name}")
            rec.update(stage_record("Rebuild", m, infer_times))
            rec["Build_Time"] = build_time

        print(f"Step {step_name:>15s} ({shift_pct:>4.0%}): Median QErr = {m['QErr_median']:.3f}")
        records.append(rec)

    out_csv = out_dir / f"{args.model}_drift_analysis.csv"
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"\nExperiment completed. Results saved to {out_csv}")


if __name__ == "__main__":
    main()
