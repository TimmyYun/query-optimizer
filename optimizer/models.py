import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Any
from sklearn.isotonic import IsotonicRegression
from .core import Bucket, RangeQuery, CDFTrainRow





"""
EquiHistLearner - self-tuning histogram baseline.
Bucket counts are updated online based on query feedback.
It does NOT resize bucket boundaries, only updates the frequency count estimate.
"""
class EquiHistLearner:

    def __init__(self, buckets: List[Bucket], learning_rate: float = 0.5):
        # We copy buckets to avoid mutating the original static histogram.
        # We keep only (lo, hi, count). NDV / exact_values are irrelevant for EquiHist.
        #
        # learning_rate:
        #   0.0  -> no learning (pure static histogram)
        #   1.0  -> full correction per query (can be unstable)
        self.buckets = [Bucket(b.lo, b.hi, count=b.count) for b in buckets]
        self.lr = learning_rate
        
    def predict(self, q: RangeQuery) -> float:
        # Predict COUNT of values in [q.low, q.high] using *uniform assumption*
        # inside each bucket. For each bucket we take:
        #   overlap_fraction * bucket_count
        total = 0.0
        for b in self.buckets:
            # overlap of query with bucket
            ov_lo = max(q.low, b.lo)
            ov_hi = min(q.high, b.hi)
            if ov_lo <= ov_hi:
                w = b.hi - b.lo + 1
                frac = (ov_hi - ov_lo + 1) / w
                total += frac * b.count
        return total
        
        return total
        
    def update(self, q: RangeQuery, actual: float):
        """
        Update counts of all buckets that overlap this query.

        Core rule = multiplicative correction:
            NewCount = OldCount * (ratio ^ (lr * overlap_fraction))

        Where:
          ratio = actual / predicted
            - ratio > 1 => we under-estimated (counts should go up)
            - ratio < 1 => we over-estimated (counts should go down)

          overlap_fraction in [0..1]
            - if query covers the whole bucket => full update
            - if query touches only a small part => small update

        We clip ratio to avoid wild oscillations from one extreme query.
        """
        pred = self.predict(q)
        if pred == 0: 
            # If predicted 0, we cannot form a stable ratio.
            # (Also means our histogram thinks this range has no data.)
            return
        
        ratio = actual / pred

        #avoid exploding updates
        ratio = max(0.1, min(10.0, ratio))
        
        for b in self.buckets:
            # overlap of query with bucket
            ov_lo = max(q.low, b.lo)
            ov_hi = min(q.high, b.hi)
            if ov_lo <= ov_hi:
                w = b.hi - b.lo + 1
                overlap = (ov_hi - ov_lo + 1) / w
                
                # overlap scales how much we "blame" this bucket for the error
                factor = ratio ** (self.lr * overlap)

                # keep count at least 1 to avoid dead buckets
                b.count = int(max(1, b.count * factor))





# Hybrid Approach
# Idea:
#   A standard histogram assumes uniform distribution inside each bucket.
#   That fails on skew / clusters.
# Fix:
#   For each bucket, learn a local CDF:
#       F_b(x) ≈ P(X <= x | X in bucket b)

def collect_cdf_training_rows(buckets: List[Bucket], freq: np.ndarray, mn: int, points_per_bucket: int, rng: np.random.Generator, bucket_indices: List[int] = None) -> Dict[int, List[CDFTrainRow]]:

    ps = np.cumsum(freq)


# by default we train for all buckets; during repair we may train only selected buckets
    target_indices = bucket_indices if bucket_indices is not None else range(len(buckets))
    rows = {}
    
    for i in target_indices:
        b = buckets[i]
        rows[i] = []

        # skip empty buckets and sparse (with exact values) 
        if b.count == 0: continue
        if b.exact_values is not None: continue 
        

        # We sample x inside the bucket and compute the true local CDF value.
        # More points -> smoother model, but higher training cost.
        n_samples = max(points_per_bucket, 50) 
        
        xs = rng.integers(b.lo, b.hi + 1, size=n_samples)
        xs = np.sort(xs)
        
        width = b.hi - b.lo + 1


        # base_cnt = global count of values <= (b.lo - 1)
        # used to convert global prefix sums into local counts inside this bucket
        b_lo_idx = b.lo - mn
        base_cnt = ps[b_lo_idx - 1] if b_lo_idx > 0 else 0
        
        for x in xs:
            x_idx = x - mn
            if x_idx < 0 or x_idx >= len(ps): continue
            

            curr_cnt = ps[x_idx]
            local_cnt = curr_cnt - base_cnt
            y_cdf = local_cnt / b.count
            
            #normalization inside the bucket
            x_norm = (x - b.lo) / width
            rows[i].append(CDFTrainRow(x_norm=x_norm, y_cdf=y_cdf))
            
    return rows

def train_cdf_models(rows: Dict[int, List[CDFTrainRow]]) -> Tuple[Dict[int, Any], float]:
    #train models for each bucket
    
    models = {}
    train_time = 0.0
    
    for i, rlist in rows.items():
        if not rlist:
            # no training data -> no model
            models[i] = None
            continue
            
        X = np.array([r.x_norm for r in rlist])
        y = np.array([r.y_cdf for r in rlist])
        
        t0 = time.perf_counter()


        # IsotonicRegression enforces monotonic CDF:
        #   increasing=True => non-decreasing
        #   y_min/y_max clamp to [0,1]
        #   out_of_bounds='clip' => safe behavior outside [0,1]
        mdl = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds='clip')
        mdl.fit(X, y)
        train_time += (time.perf_counter() - t0)
        models[i] = mdl
        
    return models, train_time

def predict_local_cdf(model, x_norm) -> float:
    #apply model to predict CDF value
    if model is None:
        return max(0.0, min(1.0, x_norm))
    val = model.transform([x_norm])[0]
    return float(val)

def predict_range_hybrid_cdf(q: RangeQuery, buckets: List[Bucket], models: Dict[int, Any]) -> float:
    """
    Hybrid estimator: histogram buckets + per-bucket CDF model.

    For each bucket:
      - Fully covered bucket: add bucket.count
      - Partially covered bucket: estimate fraction using CDF:
            (F(hi) - F(lo-1)) * bucket.count
      - Sparse bucket: exact sum from exact_values list (no ML)
    """
    
    def get_bucket_overlap_count(b_idx: int, q_lo: int, q_hi: int) -> float:
        b = buckets[b_idx]

        # overlap of query with bucket
        lo = max(q_lo, b.lo)
        hi = min(q_hi, b.hi)
        if lo > hi: return 0.0
        
        # Sparse buckets -> exact count
        if b.exact_values is not None:
             c_sum = 0
             for v, cnt in b.exact_values:
                 if lo <= v <= hi:
                     c_sum += cnt
             return float(c_sum)
        
        # Dense buckets -> CDF
        w = b.hi - b.lo + 1

        # CDF at hi
        x_hi_norm = (hi - b.lo) / w
        cdf_hi = predict_local_cdf(models.get(b_idx), x_hi_norm)
        
        # CDF at lo-1
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


# Equi-width baseline with uniform assumption inside each bucket
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
