#!/usr/bin/env python3
import time
from models import EquiDepthHistogram
from benchmark_utils import (
    get_common_parser,
    run_benchmark_suite,
    resolve_workload_path,
    load_and_filter_workload,
    evaluate_and_save,
)


def run_logic(args, out_dir, metadata):
    mn, mx, N, freq, sample, n_bins_data, skew, kurt = metadata
    # В PostgreSQL число бакетов = default_statistics_target, а не FD
    target = args.buckets if args.buckets is not None else args.statistics_target
    workload_path = resolve_workload_path(args)

    model_name = "EquiDepth-Idx" if args.with_index else "EquiDepth"

    # Build Histogram (ANALYZE)
    t0 = time.perf_counter()
    ed_hist = EquiDepthHistogram.build_from_sample(
        sample=sample,
        total_rows=N,
        statistics_target=target,
        actual_min=mn if args.with_index else None,
        actual_max=mx if args.with_index else None,
    )
    build_time = time.perf_counter() - t0
    print(f"{model_name} (target={target}, MCV={ed_hist.num_mcv}, buckets={ed_hist.n_buckets})")

    queries, y_true_sel = load_and_filter_workload(str(workload_path), mn, mx, freq)
    evaluate_and_save(
        out_dir, workload_path, model_name, ed_hist.predict, queries, y_true_sel, N, build_time,
        extra_metrics={
            "statistics_target": target,
            "n_buckets": ed_hist.n_buckets,
            "num_mcv": ed_hist.num_mcv,
            "n_distinct": ed_hist.n_distinct,
            "with_index": args.with_index,
        },
    )


def main():
    parser = get_common_parser("Run PostgreSQL-style Equi-Depth Histogram Benchmark")
    parser.add_argument(
        "--statistics-target",
        type=int,
        default=100,
        help="PostgreSQL default_statistics_target: max MCVs and histogram buckets; "
        "ANALYZE samples 300x this many rows (--buckets overrides it)",
    )
    parser.add_argument(
        "--with-index",
        action="store_true",
        help="Simulate a B-tree index on the column: edge buckets use the actual "
        "min/max (get_actual_variable_range) instead of clamping tiny estimates",
    )
    args = parser.parse_args()
    run_benchmark_suite(args, run_logic)


if __name__ == "__main__":
    main()
