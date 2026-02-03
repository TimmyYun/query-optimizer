import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Any
from sklearn.isotonic import IsotonicRegression
from ..core import Bucket, RangeQuery, CDFTrainRow

def collect_cdf_training_rows(buckets: List[Bucket], freq: np.ndarray, mn: int, points_per_bucket: int, rng: np.random.Generator, bucket_indices: List[int] = None) -> Dict[int, List[CDFTrainRow]]:
    """
    Collects training data for the CDF models from the raw data.
    
    For each dense bucket, it samples points and calculates their true Cumulative Distribution Function (CDF)
    value within that bucket.
    
    Args:
        buckets: List of buckets.
        freq: Frequency array of the dataset.
        mn: Minimum value of the dataset (offset).
        points_per_bucket: Number of sample points to generate per bucket.
        rng: Random number generator.
        bucket_indices: Optional list of specific bucket indices to process (for partial retraining).
    
    Returns:
        A dictionary mapping bucket index to a list of training rows (x_norm, y_cdf).
    """
    ps = np.cumsum(freq)
    target_indices = bucket_indices if bucket_indices is not None else range(len(buckets))
    rows = {}
    
    for i in target_indices:
        b = buckets[i]
        rows[i] = [] 
        if b.count == 0: continue
        # Skip sparse buckets as they use exact storage, no model needed
        if b.exact_values is not None: continue 
        
        n_samples = max(points_per_bucket, 50) 
        
        # Sample random points within the bucket range
        xs = rng.integers(b.lo, b.hi + 1, size=n_samples)
        xs = np.sort(xs)
        
        width = b.hi - b.lo + 1
        b_lo_idx = b.lo - mn
        base_cnt = ps[b_lo_idx - 1] if b_lo_idx > 0 else 0
        
        for x in xs:
            x_idx = x - mn
            if x_idx < 0 or x_idx >= len(ps): continue
            
            # Calculate local CDF: (Cumulative Count of x - Cumulative Count of bucket start) / Bucket Total
            curr_cnt = ps[x_idx]
            local_cnt = curr_cnt - base_cnt
            y_cdf = local_cnt / b.count
            
            # Normalize x to [0, 1] relative to bucket width
            x_norm = (x - b.lo) / width
            rows[i].append(CDFTrainRow(x_norm=x_norm, y_cdf=y_cdf))
            
    return rows

def train_cdf_models(rows: Dict[int, List[CDFTrainRow]]) -> Tuple[Dict[int, Any], float]:
    """
    Trains Isotonic Regression models for each bucket.
    
    Isotonic Regression is chosen because the CDF must be monotonically non-decreasing.
    """
    models = {}
    train_time = 0.0
    
    for i, rlist in rows.items():
        if not rlist:
            models[i] = None
            continue
            
        X = np.array([r.x_norm for r in rlist])
        y = np.array([r.y_cdf for r in rlist])
        
        t0 = time.perf_counter()
        # Enforce increasing=True for monotonic constraint
        mdl = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds='clip')
        mdl.fit(X, y)
        train_time += (time.perf_counter() - t0)
        models[i] = mdl
        
    return models, train_time

def predict_local_cdf(model, x_norm) -> float:
    """
    Predicts the CDF value for a normalized x using the collected model.
    """
    if model is None:
        # Fallback to uniform distribution if no model exists (y = x)
        return max(0.0, min(1.0, x_norm))
    val = model.transform([x_norm])[0]
    return float(val)

def predict_range_hybrid_cdf(q: RangeQuery, buckets: List[Bucket], models: Dict[int, Any]) -> float:
    """
    Predicts range query selectivity using the Hybrid approach.
    
    For each bucket:
    1. If it's a Sparse Bucket (low NDV) -> Use stored Exact Frequency Pairs (100% accurate).
    2. If it's a Dense Bucket (high NDV) -> Use the Learned CDF Model (Isotonic).
           Count = (CDF(high) - CDF(low)) * Bucket_Total_Count
    """
    
    def get_bucket_overlap_count(b_idx: int, q_lo: int, q_hi: int) -> float:
        b = buckets[b_idx]
        lo = max(q_lo, b.lo)
        hi = min(q_hi, b.hi)
        if lo > hi: return 0.0
        
        # Strategy 1: Exact Values (Sparse)
        if b.exact_values is not None:
             c_sum = 0
             for v, cnt in b.exact_values:
                 if lo <= v <= hi:
                     c_sum += cnt
             return float(c_sum)
        
        # Strategy 2: Learned CDF (Dense)
        w = b.hi - b.lo + 1
        
        # Calculate CDF at the end of the overlap
        x_hi_norm = (hi - b.lo) / w
        cdf_hi = predict_local_cdf(models.get(b_idx), x_hi_norm)
        
        # Calculate CDF at the start of the overlap
        prev = lo - 1
        if prev < b.lo: 
            cdf_lo = 0.0
        else:
             x_lo_norm = (prev - b.lo) / w
             cdf_lo = predict_local_cdf(models.get(b_idx), x_lo_norm)
             
        # Estimate count = Delta CDF * Total Count
        return max(0.0, cdf_hi - cdf_lo) * b.count

    total = 0.0
    for i, b in enumerate(buckets):
        # Optimization: Skip buckets outside query range
        if b.hi < q.low: continue
        if b.lo > q.high: break
        
        total += get_bucket_overlap_count(i, q.low, q.high)
        
    return total
