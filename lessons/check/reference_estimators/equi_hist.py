import numpy as np

from models import EquiWidthHistogram

from ._grid import recount


class EquiHist:
    """
    models/equi_hist.py (EquiHistLearner) с тем же правилом обновления, но без
    O(B) прохода по всем бакетам на каждый запрос: трогаем только перекрытые.
    count *= ratio ** (lr * overlap_frac), ratio обрезан до [0.1, 10], count >= 1.
    """

    def __init__(self, ew, lr):
        self.ew, self.lr = ew, lr

    @property
    def bucket_ranges(self):
        return [(b.lo, b.hi) for b in self.ew.buckets]

    def predict(self, q):
        return self.ew.predict(q)

    def observe(self, q, actual):
        pred = self.ew.predict(q)
        if pred == 0:
            return
        ratio = max(0.1, min(10.0, actual / pred))
        h = self.ew
        s = int(np.searchsorted(h.b_hi, q.low, "left"))
        e = int(np.searchsorted(h.b_lo, q.high, "right"))
        for i in range(s, e):
            ov = min(q.high, h.b_hi[i]) - max(q.low, h.b_lo[i]) + 1
            if ov <= 0:
                continue
            frac = ov / h.b_width[i]
            h.b_count[i] = max(1.0, h.b_count[i] * ratio ** (self.lr * frac))
        h.prefix_counts[1:] = np.cumsum(h.b_count)

    def refresh(self, sample, mn, mx, total_rows):
        recount(self.ew.buckets, sample, mn, mx, total_rows)
        self.ew._bake_vectorized_data()

    def n_floats(self):
        return len(self.ew.buckets) + 2


def build(sample, mn, mx, total_rows, n_buckets, learning_rate=0.5):
    return EquiHist(EquiWidthHistogram.build_from_sample(mn, mx, n_buckets, sample, total_rows), learning_rate)
