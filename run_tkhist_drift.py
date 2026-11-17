#!/usr/bin/env python3
"""
TKHist под дрейфом данных (общий цикл — drift_harness.py).
В DHist нет правила обновления, поэтому — как у EquiWidth/EquiDepth: перестройка с нуля
по выборке шага (бакеты и top-k), когда с прошлой перестройки изменилось > 10 % строк.
"""
from drift_harness import SampleHistogram, drift_parser, run_drift
from models import TKHistogram


def main():
    parser = drift_parser("TKHist under data drift")
    parser.add_argument("--topk", type=int, default=5, help="Top-k values per bucket (DHist default: 5)")
    args = parser.parse_args()
    model = SampleHistogram(
        lambda s, B: TKHistogram.build_from_sample(s.mn, s.mx, B, s.sample, s.N, top_k=args.topk),
        lambda h: len(h.buckets),
    )
    run_drift(args, "TKHist", model)


if __name__ == "__main__":
    main()
