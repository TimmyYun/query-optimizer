#!/usr/bin/env python3
"""
Постановка статьи To, Chiang, Shahabi (CIKM 2013), §5 — на синтетике, не на
данных бенчмарка. Точные датасеты статьи спецификация не фиксирует, поэтому
это НЕ воспроизведение их чисел, а тот же протокол:

  * N in {1000, 10000, 20000} различных значений, B = 30 (§5.2);
  * построение по точному распределению D (как в статье, без выборки);
  * время: prepare (группировка/сортировка/моменты) и partition
    (Algorithm 1) отдельно; абсолютные тайминги 2013 года не сравниваются;
  * точность: RMSE по числу строк на 10 000 случайных запросах равенства
    и диапазона (§5), для каждого построителя и каждой оценки равенства.

Данные: значения — случайное подмножество домена [0, 10 * N), частоты —
Zipf(z) со случайным назначением рангов. Все сиды фиксированы.

    poetry run python run_entropyhist_paper.py --experiment-name entropyhist_paper
"""
import argparse
import time

import numpy as np
import pandas as pd

from benchmark_utils import setup_out_dir
from models import EntropyHistogram1D, HistogramConfig
from models.entropy_histogram import EQ_ESTIMATORS

BUILDERS = {
    "MEA": dict(method="ME", me_input="area"),
    "MEF": dict(method="ME", me_input="frequency"),
    "MSE": dict(method="MSE"),
    "MB": dict(method="MB"),
}


def zipf_distribution(rng, n_distinct, total_rows, z):
    """N значений из [0, 10N), частоты Zipf(z) (>= 1), ранги назначены случайно."""
    values = np.sort(rng.choice(10 * n_distinct, size=n_distinct, replace=False))
    weights = rng.permutation(1.0 / np.arange(1, n_distinct + 1) ** z)
    freq = np.maximum(1, np.rint(total_rows * weights / weights.sum())).astype(np.int64)
    return values, freq


def exact_range(values, prefix, lows, highs):
    """Точный ответ — оракул, гистограмме недоступен."""
    lo = np.searchsorted(values, lows, side="left")
    hi = np.searchsorted(values, highs, side="right")
    return (prefix[hi] - prefix[lo]).astype(np.float64)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def main():
    parser = argparse.ArgumentParser(description="Entropy-based Histograms: paper-style construction & RMSE benchmark")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default="entropyhist_paper")
    parser.add_argument("--n-distinct", type=int, nargs="+", default=[1000, 10000, 20000])
    parser.add_argument("--buckets", type=int, default=30)
    parser.add_argument("--total-rows", type=int, default=1_000_000)
    parser.add_argument("--zipf", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--queries", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=5, help="Build repetitions for timing (median reported)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = setup_out_dir(args, "synthetic")
    records = []

    for n_distinct in args.n_distinct:
        for z in args.zipf:
            rng = np.random.default_rng([args.seed, n_distinct, int(z * 100)])
            values, freq = zipf_distribution(rng, n_distinct, args.total_rows, z)
            prefix = np.concatenate([[0], np.cumsum(freq)])
            domain_hi = 10 * n_distinct - 1

            # Равенство: значения домена равномерно (в т.ч. отсутствующие в D)
            eq_points = rng.integers(0, domain_hi + 1, size=args.queries)
            eq_true = exact_range(values, prefix, eq_points, eq_points)
            # Диапазон: оба конца равномерно по домену
            ends = np.sort(rng.integers(0, domain_hi + 1, size=(args.queries, 2)), axis=1)
            rng_true = exact_range(values, prefix, ends[:, 0], ends[:, 1])

            for builder, kw in BUILDERS.items():
                config = HistogramConfig(n_buckets=args.buckets, **kw)
                timings = [
                    EntropyHistogram1D.build(values, freq, config).timings
                    for _ in range(args.repeats)
                ]
                hist = EntropyHistogram1D.build(values, freq, config)

                t0 = time.perf_counter()
                rng_pred = [hist.estimate_range(lo, hi) for lo, hi in ends]
                range_infer = (time.perf_counter() - t0) / args.queries

                record = {
                    "N": n_distinct,
                    "zipf": z,
                    "builder": builder,
                    "B_effective": hist.B_effective,
                    "prepare_ms": 1e3 * np.median([t["prepare_time"] for t in timings]),
                    "partition_ms": 1e3 * np.median([t["partition_time"] for t in timings]),
                    "build_ms": 1e3 * np.median([t["build_time"] for t in timings]),
                    "range_rmse": rmse(rng_true, rng_pred),
                    "range_infer_us": 1e6 * range_infer,
                }
                for est in EQ_ESTIMATORS:
                    pred = [hist.estimate_eq(v, est) for v in eq_points]
                    record[f"eq_rmse_{est}"] = rmse(eq_true, pred)
                records.append(record)
                print(
                    f"N={n_distinct:>6} z={z:<4} {builder:<4} build={record['build_ms']:.1f}ms "
                    f"range RMSE={record['range_rmse']:.1f} eq RMSE(T/d)={record['eq_rmse_paper_uniform']:.1f}"
                )

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "entropyhist_paper.csv", index=False)
    print(f"\nResults saved to {out_dir / 'entropyhist_paper.csv'}")


if __name__ == "__main__":
    main()
