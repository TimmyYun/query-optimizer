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
from pathlib import Path
import numpy as np
from models import EntropyHistogram1D, HistogramConfig, summarize
from models.entropy_histogram import EQ_ESTIMATORS, RANGE_ESTIMATORS
from benchmark_utils import (
    get_common_parser,
    run_benchmark_suite,
    load_and_filter_workload,
    save_benchmark_results,
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

    workload_path = Path(args.workload)
    if not workload_path.exists():
        potential_path = Path(args.dataset) / f"workload_{args.workload}.csv"
        if potential_path.exists():
            workload_path = potential_path
        else:
            raise FileNotFoundError(
                f"Workload not found at {args.workload} or {potential_path}"
            )

    # Build: по выборке, оценки масштабируются на N / len(sample)
    t0 = time.perf_counter()
    e_hist = EntropyHistogram1D.build_from_sample(sample, N, config)
    build_time = time.perf_counter() - t0

    queries, y_true_sel = load_and_filter_workload(str(workload_path), mn, mx, freq)
    y_true_counts = y_true_sel * N

    # Evaluate
    t0 = time.perf_counter()
    y_pred_counts = []
    infer_times = []
    for q in queries:
        st = time.perf_counter()
        pred = e_hist.predict(q)
        infer_times.append(time.perf_counter() - st)
        y_pred_counts.append(pred)

    y_pred_counts = np.array(y_pred_counts)
    infer_time = time.perf_counter() - t0
    infer_times = np.array(infer_times)

    # Метрика статьи (§5) — по сырым оценкам, до сглаживания max(., 1)
    rmse_count = float(np.sqrt(np.mean((y_true_counts - y_pred_counts) ** 2)))

    y_pred = np.maximum(y_pred_counts, 1.0) / N
    y_true_safe = np.maximum(y_true_counts, 1.0) / N

    m = summarize(y_true_safe, y_pred, model_name)
    metrics = {
        "build_time": build_time,
        "prepare_time": e_hist.timings["prepare_time"],
        "partition_time": e_hist.timings["partition_time"],
        "infer_time": infer_time,
        "median_infer_time": np.median(infer_times),
        "avg_infer_time": np.mean(infer_times),
        "p95_infer_time": np.percentile(infer_times, 95),
        "median_q_error": m["QErr_median"],
        "p25_q_error": m["QErr_p25"],
        "p75_q_error": m["QErr_p75"],
        "p95_q_error": m["QErr_p95"],
        "avg_q_error": m["QErr_avg"],
        "rmse_count": rmse_count,
        "n_buckets_requested": e_hist.B_requested,
        "n_buckets": e_hist.B_effective,
        "n_distinct_sample": e_hist.distinct_count,
        "config": e_hist.to_dict()["config"],
        "build_input_signature": e_hist.build_input_signature,
    }

    print(
        f"{model_name} (B={e_hist.B_effective}/{e_hist.B_requested}, distinct={e_hist.distinct_count}): "
        f"Median QErr={metrics['median_q_error']:.4f}, RMSE={rmse_count:.1f}, Build={build_time:.4f}s"
    )

    save_benchmark_results(
        out_dir, workload_path.stem, model_name, queries, y_true_safe, y_pred, metrics
    )


def main():
    parser = get_common_parser("Run Entropy-based Histogram (To et al., CIKM 2013) Benchmark")
    add_entropyhist_args(parser)
    args = parser.parse_args()
    run_benchmark_suite(args, run_logic)


if __name__ == "__main__":
    main()
