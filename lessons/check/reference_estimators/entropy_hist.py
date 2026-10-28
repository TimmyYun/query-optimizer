from models.entropy_histogram import EntropyHistogram1D, HistogramConfig
from models.entropy_histogram import entropy as _entropy


def entropy(weights):
    return _entropy(weights)


class EH:
    def __init__(self, sample, N, cfg):
        self.cfg = cfg
        self.h = EntropyHistogram1D.build_from_sample(sample, N, cfg)

    @property
    def bucket_ranges(self):
        return [(int(b.min_value), int(b.max_value)) for b in self.h.buckets]

    def predict(self, q):
        return self.h.predict(q)

    def refresh(self, sample, mn, mx, total_rows):
        self.h = EntropyHistogram1D.build_from_sample(sample, total_rows, self.cfg)

    def n_floats(self):
        return 3 * len(self.h.buckets)  # min, max, count (+ distinct для равенства)


def build(sample, mn, mx, total_rows, n_buckets, method="ME", me_input="area"):
    return EH(sample, total_rows, HistogramConfig(method=method, n_buckets=n_buckets, me_input=me_input))
