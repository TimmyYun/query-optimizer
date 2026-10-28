from models import EquiWidthHistogram

from ._grid import recount


class EW:
    def __init__(self, h):
        self.h = h

    @property
    def bucket_ranges(self):
        return [(b.lo, b.hi) for b in self.h.buckets]

    def predict(self, q):
        return self.h.predict(q)

    def refresh(self, sample, mn, mx, total_rows):
        recount(self.h.buckets, sample, mn, mx, total_rows)
        self.h._bake_vectorized_data()

    def n_floats(self):
        return len(self.h.buckets) + 2


def build(sample, mn, mx, total_rows, n_buckets):
    return EW(EquiWidthHistogram.build_from_sample(mn, mx, n_buckets, sample, total_rows))
