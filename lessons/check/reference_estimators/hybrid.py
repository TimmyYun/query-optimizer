import numpy as np

from models import EquiWidthHistogram, HybridEstimator
from run_hybrid_drift import detect_bad_buckets

from ._grid import recount


def sampled_freq(sample, mn, mx, total_rows):
    """Плотная частота по ВЫБОРКЕ (не по freq!): bincount * N/n — на ней учится Hybrid."""
    n = mx - mn + 1
    c = np.bincount(np.clip(np.asarray(sample) - mn, 0, n - 1), minlength=n)
    return c * (total_rows / len(sample))


class Hybrid:
    def __init__(self, sample, mn, mx, N, k, points, seed):
        self.rng = np.random.default_rng(seed)
        self.points = points
        self.mn = mn
        ew = EquiWidthHistogram.build_from_sample(mn, mx, k, sample, N)
        self.h = HybridEstimator(ew.buckets)
        self.est_freq = sampled_freq(sample, mn, mx, N)
        self.h.train(self.est_freq, mn, points, self.rng)

    @property
    def bucket_ranges(self):
        return [(b.lo, b.hi) for b in self.h.buckets]

    @property
    def models(self):
        return self.h.models

    def predict(self, q):
        return self.h.predict(q)

    def refresh(self, sample, mn, mx, total_rows):
        # как run_hybrid_drift.py: счётчики по новой выборке, модели формы — старые
        self.mn = mn
        self.est_freq = sampled_freq(sample, mn, mx, total_rows)
        recount(self.h.buckets, sample, mn, mx, total_rows)
        self.h._bake_vectorized_data()

    def adapt(self, queries, actual, sample, mn, mx, total_rows):
        y_true = np.maximum(actual, 1.0)
        y_pred = np.maximum(np.array([self.h.predict(q) for q in queries]), 1.0)
        n_ft = self.h.feedback_update(queries, y_true, y_pred, total_rows)
        y_ft = np.maximum(np.array([self.h.predict(q) for q in queries]), 1.0)
        bad = detect_bad_buckets(self.h, queries, y_true, y_ft)
        if bad:
            self.h.train(self.est_freq, mn, self.points, self.rng, bucket_indices=bad)
        return {"finetuned": int(n_ft), "rebuilt": len(bad)}

    def n_floats(self):
        return self.h.param_count()


def build(sample, mn, mx, total_rows, n_buckets, points=200, seed=42):
    return Hybrid(sample, mn, mx, total_rows, n_buckets, points, seed)
