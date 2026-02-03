import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Any, Set
from sklearn.isotonic import IsotonicRegression
from models.core import Bucket, RangeQuery, CDFTrainRow

class HybridEstimator:
    """
    Implements a Hybrid Selectivity Estimator.
    
    This approach combines Equi-Width buckets with Machine Learning models.
    - Sparse Buckets (Low NDV): Use exact frequency storage (100% accurate).
    - Dense Buckets (High NDV): Use Isotonic Regression models to approximate the CDF.
    """
    
    def __init__(self, buckets: List[Bucket], models: Dict[int, Any] = None):
        """
        Args:
            buckets: The underlying buckets (shared with the histogram usually).
            models: Dictionary mapping bucket index to trained CDF models.
        """
        self.buckets = buckets
        self.models = models if models is not None else {}
        # Keep track of training times or metadata if needed
        self.last_train_time = 0.0

    def train(self, freq: np.ndarray, mn: int, points_per_bucket: int, rng: np.random.Generator, bucket_indices: List[int] = None) -> float:
        """
        Trains CDF models for the specified buckets.
        
        Args:
            freq: Frequency array of the dataset.
            mn: Minimum value of the dataset.
            points_per_bucket: Sampling rate per bucket.
            rng: Random generator.
            bucket_indices: Optional list of indices to retrain. If None, trains all eligible buckets.
            
        Returns:
            Total training time in seconds.
        """
        rows = self._collect_cdf_training_rows(freq, mn, points_per_bucket, rng, bucket_indices)
        new_models, t_train = self._train_cdf_models(rows)
        
        # Update internal models map
        for k, v in new_models.items():
            self.models[k] = v
            
        self.last_train_time = t_train
        return t_train

    def predict(self, q: RangeQuery) -> float:
        """
        Predicts range query selectivity using the Hybrid logic.
        """
        total = 0.0
        for i, b in enumerate(self.buckets):
            # Optimization: Skip buckets outside query range
            if b.hi < q.low: continue
            if b.lo > q.high: break
            
            total += self._get_bucket_overlap_count(i, q.low, q.high)
            
        return total
        
    def _collect_cdf_training_rows(self, freq: np.ndarray, mn: int, points_per_bucket: int, rng: np.random.Generator, bucket_indices: List[int] = None) -> Dict[int, List[CDFTrainRow]]:
        """
        Internal: Collects training samples for dense buckets.
        """
        ps = np.cumsum(freq)
        target_indices = bucket_indices if bucket_indices is not None else range(len(self.buckets))
        rows = {}
        
        for i in target_indices:
            b = self.buckets[i]
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
                
                curr_cnt = ps[x_idx]
                local_cnt = curr_cnt - base_cnt
                y_cdf = local_cnt / b.count
                
                x_norm = (x - b.lo) / width
                rows[i].append(CDFTrainRow(x_norm=x_norm, y_cdf=y_cdf))
                
        return rows

    def _train_cdf_models(self, rows: Dict[int, List[CDFTrainRow]]) -> Tuple[Dict[int, Any], float]:
        """
        Internal: Trains the Isotonic Regression models.
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

    def _predict_local_cdf(self, model, x_norm) -> float:
        """
        Internal: Predicts CDF value for normalized x.
        """
        if model is None:
            # Fallback to uniform distribution if no model exists (y = x)
            return max(0.0, min(1.0, x_norm))
        val = model.transform([x_norm])[0]
        return float(val)

    def _get_bucket_overlap_count(self, b_idx: int, q_lo: int, q_hi: int) -> float:
        """
        Computes the overlap count for a single bucket using the appropriate strategy.
        """
        b = self.buckets[b_idx]
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
        cdf_hi = self._predict_local_cdf(self.models.get(b_idx), x_hi_norm)
        
        # Calculate CDF at the start of the overlap
        prev = lo - 1
        if prev < b.lo: 
            cdf_lo = 0.0
        else:
             x_lo_norm = (prev - b.lo) / w
             cdf_lo = self._predict_local_cdf(self.models.get(b_idx), x_lo_norm)
             
        # Estimate count = Delta CDF * Total Count
        return max(0.0, cdf_hi - cdf_lo) * b.count
