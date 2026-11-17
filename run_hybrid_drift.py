#!/usr/bin/env python3
"""
Hybrid FD CDF под дрейфом данных (общий цикл — drift_harness.py, staged = True).

Сетка бакетов фиксируется на шаге 0. На каждом следующем шаге три этапа подряд:
  * Shock — модель как есть, счётчики не пересчитываются (видна деградация после дрейфа);
  * FT (Fine-Tuning) — feedback_update по запросам adapt-половины: счётчики бакетов
    сдвигаются EMA (alpha = 0.15) к тому, что показали запросы с q-error > 1.2.
    Внутренняя CDF не меняется, выборку FT не читает. После FT бакет помечается, если
    медианная q-error задевших его adapt-запросов по оценкам ПОСЛЕ FT всё ещё > 1.3;
  * RB (Rebuild) — только на checkpoint: с прошлой перестройки изменилось >= 100 000 строк
    (проверяется на границе шага, не чаще одного раза за шаг). Помеченные бакеты получают
    счётчик и CDF заново по актуальной выборке шага (waterfall выбирает модель заново);
    остальные бакеты не трогаются. Без checkpoint RB = FT и rebuilt_buckets = 0.

После каждого шага пишется bucket_tables/p<drift%>.csv (step_<k>.csv, если drift % не целый):
строка на бакет — границы, счётчик, локальная CDF, tier, параметры/память, ошибка на 500
валидационных запросах waterfall, и что с бакетом было на этом шаге (FT / помечен / RB).
"""
import numpy as np
import pandas as pd

from drift_harness import DriftModel, drift_parser, run_drift
from models import EquiWidthHistogram, HybridEstimator

BAD_BUCKET_QERROR = 1.3


def detect_bad_buckets(hybrid_est, queries, true_counts, pred_counts, error_threshold=BAD_BUCKET_QERROR):
    """Бакеты, у которых медиана q-error задевших их запросов > порога (нужно > 5 таких запросов)."""
    true_counts = np.maximum(true_counts, 1.0)
    pred_counts = np.maximum(pred_counts, 1.0)
    err = np.maximum(true_counts / pred_counts, pred_counts / true_counts)
    q_lo = np.fromiter((q.low for q in queries), np.int64, len(queries))
    q_hi = np.fromiter((q.high for q in queries), np.int64, len(queries))
    b_lo = np.array([b.lo for b in hybrid_est.buckets])
    b_hi = np.array([b.hi for b in hybrid_est.buckets])
    first = np.searchsorted(b_hi, q_lo)                 # первый бакет с hi >= q.low
    end = np.searchsorted(b_lo, q_hi, side="right")     # за последним бакетом с lo <= q.high
    bad = []
    for i in range(len(hybrid_est.buckets)):
        hit = (first <= i) & (end > i)
        if hit.sum() > 5 and np.median(err[hit]) > error_threshold:
            bad.append(i)
    return bad


class HybridDrift(DriftModel):
    rebuild_rule = "100k"
    uses_feedback = True
    staged = True

    def __init__(self, points):
        self.points = points
        self.rng = np.random.default_rng(42)

    def build(self, step):
        buckets = EquiWidthHistogram.build_from_sample(step.mn, step.mx, self.B, step.sample, step.N).buckets
        self.h = HybridEstimator(buckets)
        self.h.train(step.sampled_freq(), step.mn, self.points, self.rng)
        self.marked = set()
        self.step_no = 0
        # per-bucket bookkeeping for the CDF tables (instrumentation, no effect on estimates)
        self.trained_step = {i: 0 for i in range(len(self.h.buckets))}
        self._reset_step_status()
        self._report()

    def _reset_step_status(self):
        self._count_shock = np.array([b.count for b in self.h.buckets], dtype=np.float64)
        self._ft_idx, self._marked_after_ft, self._rb_idx = set(), set(), set()

    def update(self, feedback):
        """FT: только фидбэк, выборку не читает, CDF не меняет. Возвращает число затронутых бакетов."""
        self.step_no += 1
        self._reset_step_status()
        n = self.h.feedback_update(feedback.queries, np.maximum(feedback.true_counts, 1.0), feedback.pred_counts)
        self._ft_idx = set(int(i) for i in self.h.last_ft_idx)
        return n

    def mark(self, feedback):
        """feedback — оценки после FT: бакеты, которые FT не исправил, ждут checkpoint."""
        self.marked.update(detect_bad_buckets(self.h, feedback.queries, feedback.true_counts, feedback.pred_counts))
        self._marked_after_ft = set(self.marked)
        return len(self.marked)

    def rebuild(self, step, feedback):
        """Checkpoint: помеченные бакеты — счётчик и CDF по выборке шага; остальные не трогаются."""
        bad = sorted(self.marked)
        if bad:
            est = step.sampled_freq()
            ps = np.concatenate([[0.0], np.cumsum(est)])
            for i in bad:
                b = self.h.buckets[i]
                lo = int(np.clip(b.lo - step.mn, 0, len(est)))
                hi = int(np.clip(b.hi - step.mn + 1, 0, len(est)))
                b.count = float(ps[hi] - ps[lo])
            self.h._bake_vectorized_data()
            self.h.train(est, step.mn, self.points, self.rng, bucket_indices=bad)
            for i in bad:
                self.trained_step[i] = self.step_no
        self._rb_idx = set(bad)
        self.marked.clear()
        self._report()
        return len(bad)

    def _report(self):
        reports = self.out_dir / "model_reports"
        reports.mkdir(exist_ok=True)
        self.h.report_models(file_path=reports / f"step_{self.step_no}.txt")

    def predict(self, q):
        return self.h.predict(q)

    def n_buckets(self):
        return len(self.h.buckets)

    def n_values(self):
        return self.h.n_values()

    def end_step(self, k, drift_pct):
        """Таблица CDF по бакетам после всех этапов шага k."""
        rows = self.h.bucket_table()
        for r in rows:
            i = r["bucket"]
            ft, rb = i in self._ft_idx, i in self._rb_idx
            r.update({
                "step": k, "drift_pct": drift_pct,
                "count_shock": float(self._count_shock[i]) if k > 0 else None,
                "ft_updated": ft if k > 0 else None,
                "marked_after_ft": (i in self._marked_after_ft) if k > 0 else None,
                "rebuilt": rb if k > 0 else None,
                "pending_mark": i in self.marked,
                "step_status": "INITIAL" if k == 0 else ("FT+RB" if ft and rb else "RB" if rb else "FT" if ft else "UNCHANGED"),
                "last_trained_step": self.trained_step.get(i, 0),
            })
        out = self.out_dir / "bucket_tables"
        out.mkdir(exist_ok=True)
        name = f"p{int(round(drift_pct)):03d}" if abs(drift_pct - round(drift_pct)) < 1e-9 else f"step_{k:03d}"
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)

    def info(self):
        # Какую локальную CDF получил каждый бакет: "Uniform:40 Linear:30 ..."
        return {"cdf_choice": " ".join(f"{k}:{v}" for k, v in self.h.cdf_choice().items())}


def main():
    parser = drift_parser("Hybrid FD CDF under data drift: Shock -> FT -> Rebuild (checkpoint 100 000 rows) at every step")
    parser.add_argument("--points", type=int, default=200, help="CDF points sampled per bucket")
    args = parser.parse_args()
    run_drift(args, "Hybrid", HybridDrift(args.points))


if __name__ == "__main__":
    main()
