#!/usr/bin/env python3
import time
from models import TKHistogram
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

    # Build Histogram
    t0 = time.perf_counter()
    tk_hist = TKHistogram.build_from_sample(
        mn=mn,
        mx=mx,
        bins=n_bins,
        sample=sample,
        total_rows=N,
        top_k=args.topk,
        n_categorical=args.n_categorical,
    )
    build_time = time.perf_counter() - t0

    queries, y_true_sel = load_and_filter_workload(str(workload_path), mn, mx, freq)
    evaluate_and_save(
        out_dir, workload_path, "TKHist", tk_hist.predict, queries, y_true_sel, N, build_time,
        extra_metrics={
            "topk": args.topk,
            "n_buckets": len(tk_hist.buckets),
            "n_topk_entries": tk_hist.n_topk_entries,
        },
    )


def main():
    parser = get_common_parser("Run TKHist (Top-K Histogram) Baseline Benchmark")
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of most frequent values stored per bucket (DHist default: 5)",
    )
    parser.add_argument(
        "--n-categorical",
        type=int,
        default=200,
        help="Below this many distinct values every value is stored exactly",
    )
    args = parser.parse_args()
    run_benchmark_suite(args, run_logic)


if __name__ == "__main__":
    main()
