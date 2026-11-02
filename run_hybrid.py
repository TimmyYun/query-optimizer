#!/usr/bin/env python3
import numpy as np
from models import EquiWidthHistogram, HybridEstimator
from benchmark_utils import (
    get_common_parser,
    run_benchmark_suite,
    resolve_workload_path,
    load_and_filter_workload,
    evaluate_and_save,
)
import matplotlib.pyplot as plt
from models.common import Bucket


def build_equidepth_buckets(mn, mx, bins, sample, total_rows):
    """
    Бакеты равной глубины по квантилям выборки (границы — целые, без дублей),
    покрывающие весь [mn, mx]. Счётчики — по выборке, масштаб N / len(sample).
    """
    sample_sorted = np.sort(sample)
    scale = total_rows / len(sample)
    qs = np.quantile(sample_sorted, np.linspace(0, 1, bins + 1)[1:-1])
    edges = np.unique(np.concatenate([[mn], np.floor(qs).astype(np.int64) + 1]))
    edges = edges[(edges >= mn) & (edges <= mx)]
    buckets = []
    for j, lo in enumerate(edges):
        hi = int(edges[j + 1]) - 1 if j + 1 < len(edges) else mx
        cnt = np.searchsorted(sample_sorted, hi, side="right") - np.searchsorted(
            sample_sorted, lo, side="left"
        )
        buckets.append(Bucket(int(lo), int(hi), count=float(cnt * scale)))
    return buckets


def model_name_for(args):
    name = "Hybrid"
    if args.partition == "equidepth":
        name += "-ED"
    if args.parametric_only:
        name += "-param"
    if args.min_sample_points > 0:
        name += f"-min{args.min_sample_points}"
    return name


def plot_bucket_debug(hybrid_est, queries, y_true_safe, y_pred_safe, output_path):
    """
    Визуализирует производительность каждого бакета.
    Использует уже очищенные (safe) данные предикта и реальности.
    """
    n_buckets = len(hybrid_est.buckets)
    bucket_counts = np.array([b.count for b in hybrid_est.buckets])

    # --- FIX: Теперь считаем ошибку по очищенным данным без 1e-9 ---
    # Так как y_true_safe и y_pred_safe уже >= 1.0/N, деления на 0 не будет
    q_errors = np.maximum(y_true_safe / y_pred_safe, y_pred_safe / y_true_safe)

    bucket_err_lists = [[] for _ in range(n_buckets)]
    for idx, q in enumerate(queries):
        err = q_errors[idx]
        for b_idx in range(n_buckets):
            b = hybrid_est.buckets[b_idx]
            # Проверяем пересечение запроса с бакетом
            if q.low <= b.hi and q.high >= b.lo:
                bucket_err_lists[b_idx].append(err)

    median_errs = np.array(
        [np.median(errs) if len(errs) > 0 else 1.0 for errs in bucket_err_lists]
    )

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.bar(
        range(n_buckets),
        bucket_counts,
        color="lightgray",
        alpha=0.5,
        label="Bucket Row Count",
    )
    ax1.set_xlabel("Bucket Index")
    ax1.set_ylabel("Estimated Row Count (Mass)", color="gray")

    ax2 = ax1.twinx()
    ax2.plot(
        range(n_buckets),
        median_errs,
        color="red",
        linewidth=1.5,
        label="Median Q-Error",
    )
    ax2.set_ylabel("Median Q-Error (Log Scale)", color="red")
    ax2.set_yscale("log")  # Логарифмическая шкала важна для Zipf

    plt.title("Bucket Analysis: Error vs. Density (Safe Mode)")
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Debug plot saved to {output_path}")

def run_logic(args, out_dir, metadata):
    mn, mx, N, freq, sample, n_bins_data, skew, kurt = metadata
    n_bins = args.buckets if args.buckets is not None else n_bins_data
    rng = np.random.default_rng(42)
    workload_path = resolve_workload_path(args)

    # Плотность строится только из выборки; длина домена = mx - mn + 1 (freq не читаем)
    obs_freq = np.bincount(sample - mn, minlength=mx - mn + 1)

    # Scale the sample back up to N (Crucial for Cardinality Estimation)
    scaling_factor = N / len(sample)
    estimated_freq = obs_freq * scaling_factor

    # Build Initial Buckets using the ESTIMATED frequency
    if args.partition == "equidepth":
        init_buckets = build_equidepth_buckets(mn, mx, n_bins, sample, N)
    else:
        init_buckets = EquiWidthHistogram.build_from_sample(
            mn=mn, mx=mx, bins=n_bins, sample=sample, total_rows=N
        ).buckets
    model_name = model_name_for(args)

    # Train Hybrid using the ESTIMATED frequency
    print(f"Training Hybrid Model on {len(sample)} Sample (Bins: {n_bins})...")
    hybrid_est = HybridEstimator(
        init_buckets,
        identity_threshold=args.ident,
        mlp_penalty=args.penalty,
        fourier_penalty=args.penalty,
        min_train_count=args.min_sample_points * scaling_factor,
        parametric_only=args.parametric_only,
    )

    t_train = hybrid_est.train(estimated_freq, mn, args.points, rng)

    # Report Model Selections
    model_report_path = out_dir / f"model_selection_{model_name}.txt"
    hybrid_est.report_models(file_path=model_report_path)

    queries, y_true_sel = load_and_filter_workload(str(workload_path), mn, mx, freq)
    _, y_true_safe, y_pred_safe = evaluate_and_save(
        out_dir, workload_path, model_name, hybrid_est.predict, queries, y_true_sel, N, t_train,
        extra_metrics={
            "n_buckets": len(hybrid_est.buckets),
            "n_models": sum(1 for v in hybrid_est.models.values() if v is not None),
            "n_floats": hybrid_est.param_count(),
        },
    )

    if not args.no_debug_plot:
        plot_bucket_debug(hybrid_est, queries, y_true_safe, y_pred_safe, out_dir / "initial_baseline_debug.png")


def main():
    parser = get_common_parser("Run Hybrid Model Benchmark")
    parser.add_argument("--points", type=int, default=200, help="Points per bucket")
    parser.add_argument("--ident", type=float, default=1e-4, help="Identity threshold")
    parser.add_argument("--penalty", type=float, default=1.5, help="Model penalty")
    parser.add_argument(
        "--min-sample-points",
        type=int,
        default=0,
        help="Train a local model only if the bucket holds at least this many sample points (0 = always)",
    )
    parser.add_argument(
        "--partition",
        choices=["equiwidth", "equidepth"],
        default="equiwidth",
        help="Bucket boundaries: equal width (default) or equal depth by sample quantiles",
    )
    parser.add_argument("--parametric-only", action="store_true", help="Only linear/power/log/poly3 local models (2-4 floats each)")
    parser.add_argument("--no-debug-plot", action="store_true", help="Skip the O(Q x B) per-bucket debug plot")
    args = parser.parse_args()
    run_benchmark_suite(args, run_logic)


if __name__ == "__main__":
    main()
