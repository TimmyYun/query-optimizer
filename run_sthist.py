#!/usr/bin/env python3
"""
ST-Histogram (Aboulnaga & Chaudhuri, SIGMOD 1999) на статическом бенчмарке.

Протокол как у run_equihist.py (prequential): для каждого запроса сначала
predict, затем observe_feedback с точным ответом. Модель строится только из
метаданных (N, mn, mx) — sample и freq ей не передаются.
"""
import time
from pathlib import Path
import numpy as np
from models import STHistogram, STHistogramConfig, summarize
from benchmark_utils import (
    get_common_parser,
    run_benchmark_suite,
    load_and_filter_workload,
    save_benchmark_results,
)


def add_sthist_args(parser):
    """Параметры ST-гистограммы (общие для run_sthist.py и run_sthist_drift.py)."""
    parser.add_argument("--alpha", type=float, default=0.5, help="Damping factor (paper 1-D: 0.5)")
    parser.add_argument("--restructure-every", type=int, default=200, help="R: restructure every R queries (paper: 200)")
    parser.add_argument("--merge-threshold", type=float, default=0.00025, help="m as fraction of T (paper: 0.025%%)")
    parser.add_argument("--split-fraction", type=float, default=0.10, help="s as fraction of B (paper: 10%%)")
    parser.add_argument("--no-restructure", action="store_true", help="Ablation: frequency refinement only (Table 2, left column)")
    parser.add_argument(
        "--mode",
        choices=["paper_faithful", "robust_extended"],
        default="paper_faithful",
        help="robust_extended enables non-paper extensions — never report it as the baseline",
    )
    parser.add_argument("--renormalize-mass", action="store_true", help="[EXTENSION] keep sum(freq) == T (robust_extended only)")


def build_config(args, n_buckets):
    return STHistogramConfig(
        bucket_count=n_buckets,
        restructure_every=args.restructure_every,
        alpha=args.alpha,
        merge_threshold=args.merge_threshold,
        split_fraction=args.split_fraction,
        enable_restructuring=not args.no_restructure,
        mode=args.mode,
        renormalize_mass=args.renormalize_mass,
    )


def model_name_for(args):
    name = "STHist"
    if args.mode == "robust_extended":
        name += "-Robust"
    if args.no_restructure:
        name += "-NoRS"
    return name


def run_logic(args, out_dir, metadata):
    mn, mx, N, freq, sample, n_bins_data, skew, kurt = metadata
    # Тот же бюджет бакетов, что у EquiWidth/EquiHist (FD), если не задан явно
    n_bins = args.buckets if args.buckets is not None else n_bins_data
    model_name = model_name_for(args)

    workload_path = Path(args.workload)
    if not workload_path.exists():
        potential_path = Path(args.dataset) / f"workload_{args.workload}.csv"
        if potential_path.exists():
            workload_path = potential_path
        else:
            raise FileNotFoundError(
                f"Workload not found at {args.workload} or {potential_path}"
            )

    # Build: только метаданные каталога, без данных (§3.1)
    t0 = time.perf_counter()
    st_hist = STHistogram(build_config(args, n_bins), total_rows=N, min_value=mn, max_value=mx)
    build_time = time.perf_counter() - t0

    queries, y_true_raw = load_and_filter_workload(str(workload_path), mn, mx, freq)
    # Точная кардинальность — фидбэк исполнителя (целое число строк)
    actual_counts = np.rint(y_true_raw * N)
    y_true_safe = np.maximum(actual_counts, 1.0) / N

    y_pred = []
    infer_times = []
    inf_time_total = 0.0
    update_time_total = 0.0

    for q, actual in zip(queries, actual_counts):
        # 1. Оценка ДО обучения на этом запросе
        t0 = time.perf_counter()
        pred_val = st_hist.predict(q)
        it_val = time.perf_counter() - t0
        inf_time_total += it_val
        infer_times.append(it_val)
        y_pred.append(max(pred_val, 1.0) / N)

        # 2. Фидбэк (Fig. 2) + реструктуризация каждые R запросов (Fig. 3)
        t0 = time.perf_counter()
        st_hist.observe_feedback(q.low, q.high, float(actual))
        update_time_total += time.perf_counter() - t0

    y_pred = np.array(y_pred)
    m = summarize(y_true_safe, y_pred, model_name)

    infer_times = np.array(infer_times)
    events = dict(st_hist.events)

    metrics = {
        "initial_build_time": build_time,
        "update_time_total": update_time_total,
        "total_train_time": build_time + update_time_total,
        "infer_time": inf_time_total,
        "median_infer_time": np.median(infer_times),
        "avg_infer_time": np.mean(infer_times),
        "p95_infer_time": np.percentile(infer_times, 95),
        "median_q_error": m["QErr_median"],
        "p25_q_error": m["QErr_p25"],
        "p75_q_error": m["QErr_p75"],
        "p95_q_error": m["QErr_p95"],
        "avg_q_error": m["QErr_avg"],
        "n_buckets": len(st_hist.buckets),
        "config": st_hist.get_state()["config"],
        # Fig. 2 не нормирует массу — фиксируем, насколько она ушла от N
        "final_mass_ratio": st_hist.total_mass / N,
        "events": events,
    }

    print(
        f"{model_name}: Median QErr={metrics['median_q_error']:.4f}, "
        f"Train={metrics['total_train_time']:.4f}s, Inf={inf_time_total:.4f}s"
    )
    print(f"  Events: {events}")

    save_benchmark_results(
        out_dir, workload_path.stem, model_name, queries, y_true_safe, y_pred, metrics
    )


def main():
    parser = get_common_parser("Run ST-Histogram (Aboulnaga & Chaudhuri 1999) Benchmark")
    add_sthist_args(parser)
    args = parser.parse_args()
    run_benchmark_suite(args, run_logic)


if __name__ == "__main__":
    main()
