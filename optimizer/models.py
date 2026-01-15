import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Any
from sklearn.isotonic import IsotonicRegression
from .core import Bucket, RangeQuery, CDFTrainRow

class EquiHistLearner:
    """
    Implements a self-tuning histogram baseline.
    """
    def __init__(self, buckets: List[Bucket], learning_rate: float = 0.5):
        self.buckets = [Bucket(b.lo, b.hi, count=b.count) for b in buckets]
        self.lr = learning_rate
        
    def predict(self, q: RangeQuery) -> float:
        total = 0.0
        for b in self.buckets:
            ov_lo = max(q.low, b.lo)
            ov_hi = min(q.high, b.hi)
            if ov_lo <= ov_hi:
                w = b.hi - b.lo + 1
                frac = (ov_hi - ov_lo + 1) / w
                total += frac * b.count
        return total
        
    def update(self, q: RangeQuery, actual: float):
        pred = self.predict(q)
        if pred == 0: return
        
        ratio = actual / pred
        ratio = max(0.1, min(10.0, ratio))
        
        for b in self.buckets:
            ov_lo = max(q.low, b.lo)
            ov_hi = min(q.high, b.hi)
            if ov_lo <= ov_hi:
                w = b.hi - b.lo + 1
                overlap = (ov_hi - ov_lo + 1) / w
                factor = ratio ** (self.lr * overlap)
                b.count = int(max(1, b.count * factor))

def collect_cdf_training_rows(buckets: List[Bucket], freq: np.ndarray, mn: int, points_per_bucket: int, rng: np.random.Generator, bucket_indices: List[int] = None) -> Dict[int, List[CDFTrainRow]]:
    ps = np.cumsum(freq)
    target_indices = bucket_indices if bucket_indices is not None else range(len(buckets))
    rows = {}
    
    for i in target_indices:
        b = buckets[i]
        rows[i] = [] 
        if b.count == 0: continue
        if b.exact_values is not None: continue 
        
        n_samples = max(points_per_bucket, 50) 
        
        xs = rng.integers(b.lo, b.hi + 1, size=n_samples)
        xs = np.sort(xs)
        
        width = b.hi - b.lo + 1
        b_lo_idx = b.lo - mn
        base_cnt = ps[b_lo_idx - 1] if b_lo_idx > 0 else 0
        
        for x in xs:
            x_idx = x - mn
            if x_idx < 0 or x_idx >= len(ps): continue
            
            curr_cnt = ps[x_idx]
            local_cnt = curr_cnt - base_cnt
            y_cdf = local_cnt / b.count
            
            x_norm = (x - b.lo) / width
            rows[i].append(CDFTrainRow(x_norm=x_norm, y_cdf=y_cdf))
            
    return rows

def train_cdf_models(rows: Dict[int, List[CDFTrainRow]]) -> Tuple[Dict[int, Any], float]:
    models = {}
    train_time = 0.0
    
    for i, rlist in rows.items():
        if not rlist:
            models[i] = None
            continue
            
        X = np.array([r.x_norm for r in rlist])
        y = np.array([r.y_cdf for r in rlist])
        
        t0 = time.perf_counter()
        mdl = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds='clip')
        mdl.fit(X, y)
        train_time += (time.perf_counter() - t0)
        models[i] = mdl
        
    return models, train_time

def predict_local_cdf(model, x_norm) -> float:
    if model is None:
        return max(0.0, min(1.0, x_norm))
    val = model.transform([x_norm])[0]
    return float(val)

def predict_range_hybrid_cdf(q: RangeQuery, buckets: List[Bucket], models: Dict[int, Any]) -> float:
    
    def get_bucket_overlap_count(b_idx: int, q_lo: int, q_hi: int) -> float:
        b = buckets[b_idx]
        lo = max(q_lo, b.lo)
        hi = min(q_hi, b.hi)
        if lo > hi: return 0.0
        
        if b.exact_values is not None:
             c_sum = 0
             for v, cnt in b.exact_values:
                 if lo <= v <= hi:
                     c_sum += cnt
             return float(c_sum)
        
        w = b.hi - b.lo + 1
        x_hi_norm = (hi - b.lo) / w
        cdf_hi = predict_local_cdf(models.get(b_idx), x_hi_norm)
        
        prev = lo - 1
        if prev < b.lo: 
            cdf_lo = 0.0
        else:
             x_lo_norm = (prev - b.lo) / w
             cdf_lo = predict_local_cdf(models.get(b_idx), x_lo_norm)
             
        return max(0.0, cdf_hi - cdf_lo) * b.count

    total = 0.0
    for i, b in enumerate(buckets):
        if b.hi < q.low: continue
        if b.lo > q.high: break
        total += get_bucket_overlap_count(i, q.low, q.high)
        
    return total

def predict_range_histogram_uniform(q: RangeQuery, buckets: List[Bucket]) -> float:
    total = 0.0
    for b in buckets:
        ov_lo = max(q.low, b.lo)
        ov_hi = min(q.high, b.hi)
        if ov_lo <= ov_hi:
            w = b.hi - b.lo + 1
            frac = (ov_hi - ov_lo + 1) / w
            total += frac * b.count
    return total
