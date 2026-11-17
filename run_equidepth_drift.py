#!/usr/bin/env python3
"""
EquiDepth (PostgreSQL) под дрейфом данных (общий цикл — drift_harness.py).
Как в PostgreSQL: ANALYZE заново считает MCV, histogram_bounds и n_distinct, когда с
прошлого ANALYZE изменилось > 10 % строк (autovacuum_analyze_scale_factor = 0.1).
--buckets = default_statistics_target.
"""
from drift_harness import SampleHistogram, drift_parser, run_drift
from models import EquiDepthHistogram


def main():
    args = drift_parser("PostgreSQL Equi-Depth under data drift").parse_args()
    model = SampleHistogram(
        lambda s, B: EquiDepthHistogram.build_from_sample(s.sample, s.N, statistics_target=B),
        lambda h: h.n_buckets,
    )
    run_drift(args, "EquiDepth", model)


if __name__ == "__main__":
    main()
