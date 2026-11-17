#!/usr/bin/env python3
"""
EquiWidth под дрейфом данных (общий цикл — drift_harness.py).
Перестройка с нуля по выборке шага, когда с прошлой перестройки изменилось > 10 % строк
(правило autovacuum PostgreSQL). Между перестройками гистограмма не меняется.
"""
from drift_harness import SampleHistogram, drift_parser, run_drift
from models import EquiWidthHistogram


def main():
    args = drift_parser("EquiWidth under data drift").parse_args()
    model = SampleHistogram(
        lambda s, B: EquiWidthHistogram.build_from_sample(s.mn, s.mx, B, s.sample, s.N),
        lambda h: len(h.buckets),
    )
    run_drift(args, "EquiWidth", model)


if __name__ == "__main__":
    main()
