import numpy as np


def recount(buckets, sample, mn, mx, total_rows):
    """ANALYZE на фиксированной сетке: счётчики бакетов по новой выборке, масштаб N/n."""
    s = np.sort(np.asarray(sample))
    scale = total_rows / len(s)
    for b in buckets:
        b.count = float((np.searchsorted(s, b.hi, "right") - np.searchsorted(s, b.lo, "left")) * scale)
