#!/usr/bin/env python3
import time
from models import EquiWidthHistogram, EquiHistLearner
from benchmark_utils import (
    get_common_parser,
    run_benchmark_suite,
    resolve_workload_path,
    load_and_filter_workload,
    evaluate_and_save,
)


def run_logic(args, out_dir, metadata):
    mn, mx, N, freq, sample, n_bins_data, skew, kurt = metadata
    n_bins = args.buckets if args.buckets is not None else n_bins_data
    workload_path = resolve_workload_path(args)

    t0 = time.perf_counter()
    ew_hist = EquiWidthHistogram.build_from_sample(
        mn=mn, mx=mx, bins=n_bins, sample=sample, total_rows=N
    )
    eh_learner = EquiHistLearner(ew_hist.buckets, learning_rate=args.lr)
    initial_build_time = time.perf_counter() - t0

    queries, y_true_sel = load_and_filter_workload(str(workload_path), mn, mx, freq)

    # Немедленное обновление по фидбэку после каждой оценки
    def feedback(q, true_count, pred_count):
        eh_learner.update(q, true_count, pred=pred_count)

    evaluate_and_save(
        out_dir, workload_path, "EquiHist", eh_learner.predict, queries, y_true_sel, N, initial_build_time,
        feedback=feedback,
        extra_metrics={"n_buckets": len(eh_learner.buckets), "lr": args.lr},
    )


def main():
    parser = get_common_parser("Run EquiHist Benchmark")
    parser.add_argument("--lr", type=float, default=0.5, help="Learning Rate")
    args = parser.parse_args()
    run_benchmark_suite(args, run_logic)


if __name__ == "__main__":
    main()
