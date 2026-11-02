#!/usr/bin/env python3
"""
ST-Histogram (Aboulnaga & Chaudhuri, SIGMOD 1999) на статическом бенчмарке.

Протокол как у run_equihist.py (prequential): для каждого запроса сначала
predict, затем observe_feedback с точным ответом. Модель строится только из
метаданных (N, mn, mx) — sample и freq ей не передаются.
"""
import time
from models import STHistogram, STHistogramConfig
from benchmark_utils import (
    get_common_parser,
    run_benchmark_suite,
    resolve_workload_path,
    load_and_filter_workload,
    evaluate_and_save,
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
    workload_path = resolve_workload_path(args)

    # Build: только метаданные каталога, без данных (§3.1)
    t0 = time.perf_counter()
    st_hist = STHistogram(build_config(args, n_bins), total_rows=N, min_value=mn, max_value=mx)
    build_time = time.perf_counter() - t0

    queries, y_true_sel = load_and_filter_workload(str(workload_path), mn, mx, freq)

    # Фидбэк (Fig. 2) + реструктуризация каждые R запросов (Fig. 3) — после оценки запроса.
    # true_count — точная кардинальность от исполнителя (целое число строк)
    def feedback(q, true_count, pred_count):
        st_hist.observe_feedback(q.low, q.high, true_count)

    evaluate_and_save(
        out_dir, workload_path, model_name, st_hist.predict, queries, y_true_sel, N, build_time,
        feedback=feedback,
        extra_metrics=lambda true_counts, pred_counts: {
            "n_buckets": len(st_hist.buckets),
            "config": st_hist.get_state()["config"],
            # Fig. 2 не нормирует массу — фиксируем, насколько она ушла от N
            "final_mass_ratio": st_hist.total_mass / N,
            "events": dict(st_hist.events),
        },
    )
    print(f"  Events: {dict(st_hist.events)}")


def main():
    parser = get_common_parser("Run ST-Histogram (Aboulnaga & Chaudhuri 1999) Benchmark")
    add_sthist_args(parser)
    args = parser.parse_args()
    run_benchmark_suite(args, run_logic)


if __name__ == "__main__":
    main()
