import numpy as np
from typing import List, Dict, Tuple, Set
from .core import Bucket, RangeQuery

def identify_bad_buckets(queries: List[RangeQuery], y_true: np.ndarray, y_pred: np.ndarray, buckets: List[Bucket], threshold_q: float = 1.05) -> List[int]:
    """
    Identify which buckets are responsible for errors using Q-Error.
    
    Args:
        queries: List of range queries.
        y_true: Ground truth selectivity.
        y_pred: Predicted selectivity.
        buckets: List of buckets.
        threshold_q: Q-Error threshold (default 1.05 = 5% error).
        
    Returns:
        List of bucket indices that overlap with queries exceeding the threshold.
    """
    bad_buckets = set()
    qe = q_error_vec(y_true, y_pred)
    
    for i, err in enumerate(qe):
        if err > threshold_q:
            q = queries[i]
            for b_idx, b in enumerate(buckets):
                 if b.hi < q.low: continue
                 if b.lo > q.high: break
                 
                 # Conservative: mark bucket if it overlaps with a high-error query
                 bad_buckets.add(b_idx)
    
    return list(bad_buckets)

def q_error_vec(y_true, y_pred, eps=1e-9):
    yt = np.maximum(y_true, eps)
    yp = np.maximum(y_pred, eps)
    return np.maximum(yt/yp, yp/yt)

def summarize(y_true, y_pred, name="Model"):
    qe = q_error_vec(y_true, y_pred)
    med = float(np.median(qe))
    p95 = float(np.percentile(qe, 95))
    
    # MAE removed as per user request
    # mae = float(np.mean(np.abs(y_true - y_pred)))
    
    print(f"[{name}] Median QErr={med:.4f}, P95 QErr={p95:.4f}")
    return {"name": name, "QErr_median": med, "QErr_p95": p95}
