class Trivial:
    """Без статистики: строки размазаны равномерно по всему [mn, mx]."""

    def __init__(self, mn, mx, N):
        self.mn, self.mx, self.N = mn, mx, N
        self.bucket_ranges = [(mn, mx)]

    def predict(self, q):
        lo, hi = max(q.low, self.mn), min(q.high, self.mx)
        return 0.0 if lo > hi else self.N * (hi - lo + 1) / (self.mx - self.mn + 1)

    def n_floats(self):
        return 3


def build(sample, mn, mx, total_rows, n_buckets):
    return Trivial(mn, mx, total_rows)
