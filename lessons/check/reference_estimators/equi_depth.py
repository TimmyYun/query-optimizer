from models import EquiDepthHistogram


class ED:
    def __init__(self, sample, N, target):
        self.target = target
        self.h = EquiDepthHistogram.build_from_sample(sample, N, statistics_target=target)

    def __getattr__(self, name):  # mcv_vals, mcv_freqs, hist_bounds, n_distinct, selectivity
        return getattr(self.h, name)

    def predict(self, q):
        return self.h.predict(q)

    def refresh(self, sample, mn, mx, total_rows):
        self.h = EquiDepthHistogram.build_from_sample(sample, total_rows, statistics_target=self.target)

    def n_floats(self):
        return 2 * self.h.num_mcv + len(self.h.hist_bounds) + 1


def estimate_n_distinct(ndistinct, nmultiple, samplerows, totalrows):
    return EquiDepthHistogram._estimate_n_distinct(ndistinct, nmultiple, samplerows, totalrows)


def build(sample, mn, mx, total_rows, n_buckets, statistics_target=100):
    # n_buckets не используется: PostgreSQL берёт statistics_target
    return ED(sample, total_rows, statistics_target)
