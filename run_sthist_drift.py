#!/usr/bin/env python3
"""
ST-Histogram под дрейфом данных (общий цикл — drift_harness.py).
Как в статье (Aboulnaga & Chaudhuri, SIGMOD 1999): модель никогда не смотрит на данные.
Она создаётся один раз из метаданных каталога (N, min, max шага 0) и дальше узнаёт о
дрейфе только из фидбэка запросов (adapt-половина каждого шага), перестраивая бакеты
каждые R = 200 запросов. Перестройки по выборке нет.
"""
from drift_harness import DriftModel, drift_parser, run_drift
from models import STHistogram
from run_sthist import add_sthist_args, build_config, model_name_for


class STHistDrift(DriftModel):
    rebuild_rule = "never"
    uses_feedback = True

    def __init__(self, args):
        self.args = args
        self.restructures = 0

    def build(self, step):
        # §3.1: только метаданные каталога, равномерно
        self.h = STHistogram(build_config(self.args, self.B), total_rows=step.N,
                             min_value=step.mn, max_value=step.mx)

    def update(self, feedback):
        # Fig. 2 на каждом запросе + реструктуризация каждые R запросов (Fig. 3)
        before = self.h.events["restructure_applied"]
        for q, actual in zip(feedback.queries, feedback.true_counts):
            self.h.observe_feedback(q.low, q.high, float(actual))
        self.restructures = self.h.events["restructure_applied"] - before

    def predict(self, q):
        return self.h.predict(q)

    def n_buckets(self):
        return len(self.h.freq)

    def n_values(self):
        return self.h.n_values()

    def info(self):
        return {"restructures": self.restructures, "mass_ratio": self.h.total_mass / self.h.T_reference}


def main():
    parser = drift_parser("ST-Histogram (feedback only) under data drift")
    add_sthist_args(parser)
    args = parser.parse_args()
    run_drift(args, model_name_for(args), STHistDrift(args))


if __name__ == "__main__":
    main()
