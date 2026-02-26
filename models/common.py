from dataclasses import dataclass
import numpy as np
from typing import List


@dataclass
class Bucket:
    lo: int
    hi: int
    count: int = 0


@dataclass
class RangeQuery:
    low: int
    high: int


@dataclass
class CDFTrainRow:
    x_norm: float
    y_cdf: float


# -----------------------------------------------------------------------------
# Evaluation Utilities
# -----------------------------------------------------------------------------


def q_error_vec(y_true, y_pred, eps=1e-9):
    yt = np.maximum(y_true, eps)
    yp = np.maximum(y_pred, eps)
    return np.maximum(yt / yp, yp / yt)


def summarize(y_true, y_pred, name="Model"):
    qe = q_error_vec(y_true, y_pred)
    med = float(np.median(qe))
    p25 = float(np.percentile(qe, 25))
    p75 = float(np.percentile(qe, 75))
    p95 = float(np.percentile(qe, 95))
    avg = float(np.mean(qe))
    print(
        f"[{name}] Median QErr={med:.4f}, P25 QErr={p25:.4f}, P75 QErr={p75:.4f}, P95 QErr={p95:.4f}, Avg QErr={avg:.4f}"
    )
    return {
        "name": name,
        "QErr_median": med,
        "QErr_p25": p25,
        "QErr_p75": p75,
        "QErr_p95": p95,
        "QErr_avg": avg,
    }


def identify_bad_buckets(
    queries: List[RangeQuery],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    buckets: List[Bucket],
    threshold_q: float = 1.05,
) -> List[int]:
    bad_buckets = set()
    qe = q_error_vec(y_true, y_pred)
    for i, err in enumerate(qe):
        if err > threshold_q:
            q = queries[i]
            for b_idx, b in enumerate(buckets):
                if b.hi < q.low:
                    continue
                if b.lo > q.high:
                    break
                bad_buckets.add(b_idx)
    return list(bad_buckets)
