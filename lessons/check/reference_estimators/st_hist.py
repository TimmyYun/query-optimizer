from models.common import Bucket
from models.st_histogram import STHistogram, STHistogramConfig


class ST:
    def __init__(self, h):
        self.h = h

    @classmethod
    def from_buckets(cls, lows, highs, freqs, total_rows, **cfg):
        c = STHistogramConfig(bucket_count=len(freqs), **cfg)
        bs = [Bucket(int(lo), int(hi), count=float(f)) for lo, hi, f in zip(lows, highs, freqs)]
        return cls(STHistogram.from_buckets(c, total_rows, bs))

    @property
    def bucket_ranges(self):
        return [(int(a), int(b)) for a, b in zip(self.h.lows, self.h.highs)]

    @property
    def bucket_freqs(self):
        return [float(f) for f in self.h.freq]

    def total_mass(self):
        return self.h.total_mass

    def predict(self, q):
        return self.h.predict(q)

    def observe(self, q, actual):
        self.h.observe_feedback(q.low, q.high, float(actual))

    def restructure(self):
        self.h.restructure()

    def n_floats(self):
        return 2 * len(self.h.freq)


def build(sample, mn, mx, total_rows, n_buckets, alpha=0.5, restructure_every=200,
          merge_threshold=0.00025, split_fraction=0.10):
    # sample не используется: ST-гистограмма не смотрит на данные
    cfg = STHistogramConfig(bucket_count=n_buckets, alpha=alpha, restructure_every=restructure_every,
                            merge_threshold=merge_threshold, split_fraction=split_fraction)
    return ST(STHistogram(cfg, total_rows, mn, mx))


def from_buckets(lows, highs, freqs, total_rows, **cfg):
    return ST.from_buckets(lows, highs, freqs, total_rows, **cfg)
