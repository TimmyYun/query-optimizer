from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RangeQuery:
    low: int
    high: int


def q_error(y_true, y_pred):
    t = np.maximum(np.asarray(y_true, dtype=np.float64), 1.0)
    p = np.maximum(np.asarray(y_pred, dtype=np.float64), 1.0)
    return np.maximum(t / p, p / t)


def true_counts(freq, mn, lows, highs):
    ps = np.concatenate([[0.0], np.cumsum(freq, dtype=np.float64)])
    lo = np.clip(np.asarray(lows) - mn, 0, len(freq))
    hi = np.clip(np.asarray(highs) - mn + 1, 0, len(freq))
    return np.maximum(ps[hi] - ps[np.minimum(lo, hi)], 0.0)


def summarize(qe):
    qe = np.asarray(qe)
    return {
        "median": float(np.median(qe)),
        "p25": float(np.percentile(qe, 25)),
        "p75": float(np.percentile(qe, 75)),
        "p95": float(np.percentile(qe, 95)),
        "avg": float(np.mean(qe)),
        "max": float(np.max(qe)),
    }
