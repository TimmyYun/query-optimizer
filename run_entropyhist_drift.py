#!/usr/bin/env python3
"""
Entropy-based Histogram под дрейфом данных (общий цикл — drift_harness.py).
В статье (To et al., CIKM 2013) обновлений нет: гистограмма заморожена, пересборка —
новый build() по текущим данным. Когда его делать, статья не говорит, поэтому — как у
EquiWidth/EquiDepth: когда с прошлой перестройки изменилось > 10 % строк.
"""
from drift_harness import SampleHistogram, drift_parser, run_drift
from models import EntropyHistogram1D
from run_entropyhist import add_entropyhist_args, build_config, model_name_for


def main():
    parser = drift_parser("Entropy-based Histogram under data drift")
    add_entropyhist_args(parser)
    args = parser.parse_args()
    model = SampleHistogram(
        lambda s, B: EntropyHistogram1D.build_from_sample(s.sample, s.N, build_config(args, B)),
        lambda h: h.B_effective,
    )
    run_drift(args, model_name_for(build_config(args, 1)), model)


if __name__ == "__main__":
    main()
