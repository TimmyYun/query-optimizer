#!/usr/bin/env python3
"""
Entropy-based Histograms (To, Chiang, Shahabi, CIKM 2013) на статическом бенчмарке.

Протокол как у run_equiwidth.py / run_equidepth.py: гистограмма строится один
раз по 100k выборке (freq модели не передаётся), затем замороженная
отвечает на весь ворклоад. Дефолт — baseline статьи для диапазонов:
ME(A) + paper_linear; точечные запросы [v, v] -> Eq. 1 (T/d), §5.1.

Кроме q-error пишется метрика статьи — RMSE по числу строк (§5).
"""
import time
import numpy as np
from models import EntropyHistogram1D, HistogramConfig
from models.entropy_histogram import EQ_ESTIMATORS, RANGE_ESTIMATORS
from benchmark_utils import (
    get_common_parser,
    run_benchmark_suite,
    resolve_workload_path,
    load_and_filter_workload,
    evaluate_and_save,
)


def add_entropyhist_args(parser):
    """Параметры entropy-гистограммы (общие для run_entropyhist.py и run_entropyhist_drift.py)."""
    parser.add_argument("--method", choices=["ME", "MSE", "MB"], default="ME", help="Builder objective (paper §4.2–4.3)")
    parser.add_argument("--me-input", choices=["area", "frequency"], default="area", help="ME(A) or ME(F); ME only")
    parser.add_argument(
        "--me-objective",
        choices=["paper_alg1", "weighted_ablation"],
        default="paper_alg1",
        help="weighted_ablation is NOT the paper's Algorithm 1 — report it separately",
    )
    parser.add_argument("--equality-estimator", choices=EQ_ESTIMATORS, default="paper_uniform", help="Used for point queries [v, v] (paper default: Eq. 1)")
    parser.add_argument(
        "--range-estimator",
        choices=RANGE_ESTIMATORS,
        default="paper_linear",
        help="discrete_linear is a non-paper extension for integer domains",
    )


def build_config(args, n_buckets):
    is_me = args.method == "ME"
    return HistogramConfig(
        method=args.method,
        n_buckets=n_buckets,
        me_input=args.me_input if is_me else "area",
        me_objective=args.me_objective if is_me else "paper_alg1",
        equality_estimator=args.equality_estimator,
        range_estimator=args.range_estimator,
    )


def model_name_for(config):
    """EntropyHist-MEA — baseline; любые отклонения от статьи видны в имени модели."""
    name = f"EntropyHist-{config.variant}"
    if config.equality_estimator != "paper_uniform":
        name += f"-Eq_{config.equality_estimator}"
    if config.range_estimator != "paper_linear":
        name += f"-Rng_{config.range_estimator}"
    return name


def run_logic(args, out_dir, metadata):
    mn, mx, N, freq, sample, n_bins_data, skew, kurt = metadata
    # Тот же бюджет бакетов, что у EquiWidth/STHist (FD), если не задан явно
    n_bins = args.buckets if args.buckets is not None else n_bins_data
    config = build_config(args, n_bins)
    model_name = model_name_for(config)
    workload_path = resolve_workload_path(args)

    # Build: по выборке, оценки масштабируются на N / len(sample)
    t0 = time.perf_counter()
    e_hist = EntropyHistogram1D.build_from_sample(sample, N, config)
    build_time = time.perf_counter() - t0
    print(f"{model_name} (B={e_hist.B_effective}/{e_hist.B_requested}, distinct={e_hist.distinct_count})")

    queries, y_true_sel = load_and_filter_workload(str(workload_path), mn, mx, freq)
    evaluate_and_save(
        out_dir, workload_path, model_name, e_hist.predict, queries, y_true_sel, N, build_time,
        extra_metrics=lambda true_counts, pred_counts: {
            "prepare_time": e_hist.timings["prepare_time"],
            "partition_time": e_hist.timings["partition_time"],
            # Метрика статьи (§5) — по сырым оценкам, до сглаживания max(., 1)
            "rmse_count": float(np.sqrt(np.mean((true_counts - pred_counts) ** 2))),
            "n_buckets_requested": e_hist.B_requested,
            "n_buckets": e_hist.B_effective,
            "n_distinct_sample": e_hist.distinct_count,
            "config": e_hist.to_dict()["config"],
            "build_input_signature": e_hist.build_input_signature,
        },
    )


def main():
    parser = get_common_parser("Run Entropy-based Histogram (To et al., CIKM 2013) Benchmark")
    add_entropyhist_args(parser)
    args = parser.parse_args()
    run_benchmark_suite(args, run_logic)


if __name__ == "__main__":
    main()
