import numpy as np
from typing import List, Dict, Tuple, Set
from .core import Bucket, RangeQuery

def identify_bad_buckets(queries: List[RangeQuery], y_true: np.ndarray, y_pred: np.ndarray, buckets: List[Bucket], threshold_mae: float = 0.05) -> List[int]:
    """
    Identify which buckets are responsible for errors.
    """
    bad_buckets = set()
    errors = np.abs(y_true - y_pred)
    
    # print(f"DEBUG: Max Error: {np.max(errors):.6f}, Mean Error: {np.mean(errors):.6f}, Threshold: {threshold_mae}")
    # n_violations = np.sum(errors > threshold_mae)
    # print(f"DEBUG: Queries exceeding threshold: {n_violations}/{len(queries)}")

    for i, err in enumerate(errors):
        if err > threshold_mae:
            q = queries[i]
            for b_idx, b in enumerate(buckets):
                 if b.hi < q.low: continue
                 if b.lo > q.high: break
                 
                 # Only mark if not exact?
                 if b.exact_values is None: 
                     bad_buckets.add(b_idx)
    
    return list(bad_buckets)

def q_error_vec(y_true, y_pred, eps=1e-9):
    yt = np.maximum(y_true, eps)
    yp = np.maximum(y_pred, eps)
    return np.maximum(yt/yp, yp/yt)

def summarize(y_true, y_pred, name="Model"):
    qe = q_error_vec(y_true, y_pred)
    med = float(np.median(qe))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    
    print(f"[{name}] Median QErr={med:.4f}, MAE={mae:.6f}")
    return {"name": name, "QErr_median": med, "MAE": mae}
