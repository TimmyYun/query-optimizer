from models import TKHistogram


class TK:
    def __init__(self, mn, mx, k, top_k, sample, N):
        self.grid = (mn, mx, k)
        self.top_k = top_k
        self.h = TKHistogram.build_from_sample(mn, mx, k, sample, N, top_k=top_k)

    @property
    def bucket_ranges(self):
        return [(b.lo, b.hi) for b in self.h.buckets]

    def predict(self, q):
        return self.h.predict(q)

    def refresh(self, sample, mn, mx, total_rows):
        mn0, mx0, k = self.grid  # сетка шага 0, как в run_tkhist_drift.py
        self.h = TKHistogram.build_from_sample(mn0, mx0, k, sample, total_rows, top_k=self.top_k)

    def n_floats(self):
        return len(self.h.buckets) + 2 + 2 * self.h.n_topk_entries


def build(sample, mn, mx, total_rows, n_buckets, top_k=5):
    return TK(mn, mx, n_buckets, top_k, sample, total_rows)
