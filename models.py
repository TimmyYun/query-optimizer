import numpy as np
import math
import time
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

class FourierFeatureMapper:
    """
    Maps scalar input x to high-frequency sinusoids (Positional Encoding).
    Allows MLP to learn sharp transitions/spikes.
    """
    def __init__(self, num_bands: int = 10, max_freq: float = 1024.0):
        self.num_bands = num_bands
        # Log-spaced frequencies from 1 to max_freq
        self.freqs = np.logspace(0, np.log10(max_freq), num=num_bands, base=10)
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        # X shape: (N, 1)
        # Ensure X is numpy array   
        X = np.asarray(X, dtype=np.float64)
        
        # log(x) feature to help with power law distributions (Zipf)
        # Use log(x + eps) to expand the "head" of the distribution.
        # x=0 -> log(1e-7) ~ -16. x=0.0002 -> log(2e-4) ~ -8.
        # This gives the MLP ~8 units of space to fit the jump between x=0 and x=1.
        X_log = np.log(X + 1e-7)
        
        # Output shape: (N, 2 + 2 * num_bands)
        
        # 1. Original X and Log X
        features = [X, X_log]
        
        # 2. Sin/Cos for each band
        for freq in self.freqs:
            scaled = X * freq * np.pi 
            features.append(np.sin(scaled))
            features.append(np.cos(scaled))
            
        return np.hstack(features)

class FourierModelWrapper:
    """
    Wraps an MLP to auto-encode inputs using Fourier Features before prediction.
    """
    def __init__(self, model, mapper):
        self.model = model
        self.mapper = mapper
        
    def predict(self, X):
        X_fourier = self.mapper.transform(X)
        return self.model.predict(X_fourier)


# -----------------------------------------------------------------------------
# Core Data Structures
# -----------------------------------------------------------------------------

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
# Estimation Models
# -----------------------------------------------------------------------------

class EquiWidthHistogram:
    """
    Implements a standard Equi-Width Histogram.
    """
    def __init__(self, buckets: List[Bucket]):
        self.buckets = buckets
        self._bake_vectorized_data()

    def _bake_vectorized_data(self):
        """Pre-compute arrays for vectorized inference."""
        if not self.buckets:
            self.b_lo = np.array([], dtype=np.float64)
            self.b_hi = np.array([], dtype=np.float64)
            self.b_count = np.array([], dtype=np.float64)
            self.b_width = np.array([], dtype=np.float64)
            return

        self.b_lo = np.array([b.lo for b in self.buckets], dtype=np.float64)
        self.b_hi = np.array([b.hi for b in self.buckets], dtype=np.float64)
        self.b_count = np.array([b.count for b in self.buckets], dtype=np.float64)
        self.b_width = self.b_hi - self.b_lo + 1
        
    @staticmethod
    def build(mn: int, mx: int, bins: int, freq: np.ndarray, prefix_buckets: int = 0, suffix_buckets: int = 0) -> 'EquiWidthHistogram':
        if len(freq) == 0: return EquiWidthHistogram([])
        
        distinct_indices = np.where(freq > 0)[0]
        pb = []
        sb = []
        
        p_count = min(prefix_buckets, len(distinct_indices))
        for i in range(p_count):
            idx = distinct_indices[i]
            val = mn + idx
            pb.append(Bucket(val, val, int(freq[idx])))
            
        remaining_indices = distinct_indices[p_count:]
        s_count = min(suffix_buckets, len(remaining_indices))
        for i in range(s_count):
            idx = remaining_indices[-(s_count - i)] 
            val = mn + idx
            sb.append(Bucket(val, val, int(freq[idx])))
            
        remaining_bins = bins - len(pb) - len(sb)
        mid_buckets = []
        
        if remaining_bins > 0 and len(remaining_indices) > s_count:
            start_idx = distinct_indices[p_count-1] + 1 if p_count > 0 else 0
            end_idx = remaining_indices[-(s_count + 1)] if s_count > 0 else len(freq) - 1
            
            mid_freq = freq[start_idx : end_idx + 1]
            mid_mn = mn + start_idx
            mid_mx = mn + end_idx
            
            if len(mid_freq) > 0 and mid_mx >= mid_mn:
                 width = mid_mx - mid_mn + 1
                 bw = max(1, int(math.ceil(width / remaining_bins)))
                 ps = np.cumsum(mid_freq)
                 
                 cur = mid_mn
                 for _ in range(remaining_bins):
                     lo = cur
                     hi = min(mid_mx, lo + bw - 1)
                     li = lo - mid_mn
                     ri = hi - mid_mn
                     
                     if li < 0: li = 0
                     if ri >= len(mid_freq): ri = len(mid_freq) - 1
                     
                     if ri >= li:
                         cnt = int(ps[ri] - (ps[li-1] if li > 0 else 0))
                         mid_buckets.append(Bucket(lo, hi, count=cnt))
                     
                     cur = hi + 1
                     if cur > mid_mx: break
                     
    @staticmethod
    def build_from_sample(mn: int, mx: int, bins: int, sample: np.ndarray, total_rows: int) -> 'EquiWidthHistogram':
        """
        Builds a histogram using a reservoir sample (Postgres-like).
        Estimates bucket counts based on the sample's distribution.
        """
        if len(sample) == 0: return EquiWidthHistogram([])
        
        sample_size = len(sample)
        scale_factor = total_rows / sample_size if sample_size > 0 else 0
        
        width = mx - mn + 1
        if bins <= 0: bins = 1
        bw = max(1, int(math.ceil(width / bins)))
        
        buckets = []
        cur = mn
        
        # Pre-sort sample for faster counting (or just use range queries if small)
        # Using numpy searchsorted is efficient since we scan linearly
        sample_sorted = np.sort(sample)
        
        for _ in range(bins):
            lo = cur
            hi = min(mx, lo + bw - 1)
            
            # Count elements in sample within [lo, hi]
            # searchsorted returns index where element would be inserted to maintain order
            # left side (lo) is inclusive, right side (hi) is inclusive
            idx_start = np.searchsorted(sample_sorted, lo, side='left')
            idx_end = np.searchsorted(sample_sorted, hi, side='right')
            
            count_in_sample = idx_end - idx_start
            estimated_count = int(count_in_sample * scale_factor)
            
            buckets.append(Bucket(lo, hi, count=estimated_count))
            
            cur = hi + 1
            if cur > mx: break
            
        return EquiWidthHistogram(buckets)

    def predict_batch(self, queries: List[RangeQuery]) -> np.ndarray:
        """
        Vectorized bulk inference for Equi-Width Histogram.
        """
        if not self.buckets: return np.zeros(len(queries))
        
        n_queries = len(queries)
        q_lo = np.array([q.low for q in queries], dtype=np.float64)
        q_hi = np.array([q.high for q in queries], dtype=np.float64)
        total = np.zeros(n_queries)
        
        # Iterate over buckets (vectorized over queries)
        # This is generally faster than iterating over queries if N_Buckets << N_Queries
        for i in range(len(self.buckets)):
            b_cnt = self.b_count[i]
            if b_cnt == 0: continue
            
            b_lo = self.b_lo[i]
            b_hi = self.b_hi[i]
            b_width = self.b_width[i]

            # Mask: query overlaps with bucket
            # Overlap if: q.high >= b.lo AND q.low <= b.hi
            mask = (q_hi >= b_lo) & (q_lo <= b_hi)
            
            if not np.any(mask): continue
            
            # Vectorized Overlap Calculation
            # lo = max(q_lo, b_lo)
            # hi = min(q_hi, b_hi)
            
            lo = np.maximum(q_lo[mask], b_lo)
            hi = np.minimum(q_hi[mask], b_hi)
            
            overlap_width = hi - lo + 1
            # overlap_width = np.maximum(0, overlap_width) # Implicitly handled by mask? 
            # Actually mask guarantees q_hi >= b_lo and q_lo <= b_hi, 
            # so hi >= lo is guaranteed.
            
            total[mask] += (overlap_width / b_width) * b_cnt
            
        return total

    def predict(self, q: RangeQuery) -> float:
        """Scalar fallback."""
        return float(self.predict_batch([q])[0])


class EquiHistLearner:
    """
    Implements a self-tuning histogram baseline (EquiHist-like).
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
        # Adaptive learning rate based on error
        ratio = actual / pred
        
        # Limit ratio to avoid explosions
        ratio = max(0.1, min(10.0, ratio))
        
        for b in self.buckets:
            # Overlap logic
            ov_lo = max(q.low, b.lo)
            ov_hi = min(q.high, b.hi)
            
            if ov_lo <= ov_hi:
                # Update bucket count based on overlap contribution
                w = b.hi - b.lo + 1
                overlap_frac = (ov_hi - ov_lo + 1) / w
                
                # Formula: new_count = old_count * (ratio ^ (lr * overlap))
                # If overlap is 1.0 (full bucket inside query), it gets full update
                # If overlap is small, it gets small update
                factor = ratio ** (self.lr * overlap_frac)
                b.count = max(1.0, b.count * factor)

        # Invalidate vector cache
        self.b_lo = None

    def predict_batch(self, queries: List[RangeQuery]) -> np.ndarray:
        """
        Vectorized bulk inference.
        """
        # Auto-bake if needed or stale (simple check: if b_lo is None)
        # Note: If called in a loop without updates, this is fast. 
        # If called interleaved with updates, it will re-bake often (overhead).
        if not hasattr(self, 'b_lo') or self.b_lo is None:
            self._bake_vectorized_data()
            
        n_queries = len(queries)
        if n_queries == 0: return np.array([])
        
        q_lo = np.array([q.low for q in queries], dtype=np.float64)
        q_hi = np.array([q.high for q in queries], dtype=np.float64)
        total = np.zeros(n_queries)
        
        # Vectorized Bucket Loop
        for i in range(len(self.buckets)):
             # b properties from arrays
             b_lo = self.b_lo[i]
             b_hi = self.b_hi[i]
             b_cnt = self.b_count[i]
             b_width = self.b_width[i]

             # Overlap: lo = max(q_lo, b_lo), hi = min(q_hi, b_hi)
             # Mask: q_hi >= b_lo & q_lo <= b_hi
             
             mask = (q_hi >= b_lo) & (q_lo <= b_hi)
             if not np.any(mask): continue
             
             lo = np.maximum(q_lo[mask], b_lo)
             hi = np.minimum(q_hi[mask], b_hi)
             
             overlap_width = hi - lo + 1
             total[mask] += (overlap_width / b_width) * b_cnt
             
        return total

    def _bake_vectorized_data(self):
        self.b_lo = np.array([b.lo for b in self.buckets], dtype=np.float64)
        self.b_hi = np.array([b.hi for b in self.buckets], dtype=np.float64)
        self.b_count = np.array([b.count for b in self.buckets], dtype=np.float64)
        self.b_width = self.b_hi - self.b_lo + 1


class HybridEstimator:
    """
    Implements a Hybrid Selectivity Estimator combining histograms and ML models.
    """
    def __init__(self, buckets: List[Bucket], models: Dict[int, Any] = None, 
                 identity_threshold: float = 1e-4, 
                 mlp_penalty: float = 1.5, 
                 fourier_penalty: float = 1.5):
        self.buckets = buckets
        self.models = models if models is not None else {}
        self.last_train_time = 0.0
        self.identity_threshold = identity_threshold
        self.mlp_penalty = mlp_penalty
        self.fourier_penalty = fourier_penalty
        
        # Vectorized Data Storage
        self.b_lo = None
        self.b_hi = None
        self.b_count = None
        self.b_width = None
        self.mod_types = None # 0: identity, 1: linear, 2: poly, 3: complex (loop fallback)
        self.lin_params = None # (N, 2) -> [coef, intercept]
        self.poly_params = None # (N, 3) -> [coef1, coef2, intercept]
        self.log_params = None # (N, 2) -> [coef, intercept] for a*log(x+e) + b
        self.complex_models = {} # dict for fallback


    def train(self, freq: np.ndarray, mn: int, points_per_bucket: int, rng: np.random.Generator, bucket_indices: List[int] = None) -> float:
        t_start = time.perf_counter()
        
        # Train models directly on the raw frequency distribution (No MCV removal)
        rows_data = self._collect_cdf_training_rows(freq, mn, points_per_bucket, rng, bucket_indices)
        new_models, t_train_models = self._train_adaptive_models(rows_data, freq, mn, rng)
        
        for k, v in new_models.items():
            self.models[k] = v
            
        self._bake_vectorized_data()
            
        t_total = time.perf_counter() - t_start
        self.last_train_time = t_total
        return t_total


    def predict(self, q: RangeQuery) -> float:
        total = 0.0
        for i in range(len(self.buckets)):
             if self.buckets[i].count == 0: continue
             total += self._get_bucket_overlap_count(i, q.low, q.high)
        return total

    def predict_batch(self, queries: List[RangeQuery]) -> np.ndarray:
        """
        Vectorized bulk inference for HybridEstimator.
        Uses query masking and array operations to avoid Python loops for CDF calculation.
        """
        if self.b_lo is None or len(self.b_lo) != len(self.buckets):
            self._bake_vectorized_data()
            
        n_queries = len(queries)
        q_lo = np.array([q.low for q in queries], dtype=np.float64)
        q_hi = np.array([q.high for q in queries], dtype=np.float64)
        total_counts = np.zeros(n_queries)

        for i in range(len(self.buckets)):
            b_count = self.b_count[i]
            if b_count == 0: continue
            
            b_lo = self.b_lo[i]
            b_hi = self.b_hi[i]
            b_width = self.b_width[i]
            
            # Mask: Only queries that touch this bucket
            mask = (q_hi >= b_lo) & (q_lo <= b_hi)
            if not np.any(mask): continue
            
            # Intersection boundaries (only for masked queries)
            lo_clamped = np.maximum(q_lo[mask], b_lo)
            hi_clamped = np.minimum(q_hi[mask], b_hi)
            
            # Norm points for CDF (shifted by 1 for low edge to handle discrete inclusion)
            x_hi = np.clip((hi_clamped - b_lo) / b_width, 0.0, 1.0)
            x_lo_prev = (lo_clamped - 1 - b_lo) / b_width
            
            # Model Selection
            m_type = self.mod_types[i]
            
            def get_cdf_vec(x_arr):
                # Ensure input is array
                xa = np.clip(x_arr, 0.0, 1.0)
                if m_type == 0: # Identity
                    return xa
                elif m_type == 1: # Linear
                    p = self.lin_params[i]
                    return np.clip(xa * p[0] + p[1], 0.0, 1.0)
                elif m_type == 2: # Poly
                    p = self.poly_params[i]
                    return np.clip(p[2] + p[0]*xa + p[1]*(xa**2), 0.0, 1.0)
                elif m_type == 4: # Log-Linear
                    p = self.log_params[i]
                    # y = a * log(x + eps) + b. 
                    # We use eps=1e-7 to avoid log(0)
                    return np.clip(p[0] * np.log(xa + 1e-7) + p[1], 0.0, 1.0)
                else: # Complex (MLP)
                    mdl = self.complex_models[i]
                    # Vectorized predict call (sklearn handles batch X)
                    return np.clip(mdl.predict(xa.reshape(-1, 1)).flatten(), 0.0, 1.0)

            cdf_hi = get_cdf_vec(x_hi)
            # Handle x_lo_prev < 0 (points outside lower bound of bucket)
            lo_mask = x_lo_prev >= 0
            cdf_lo = np.zeros_like(cdf_hi)
            if np.any(lo_mask):
                cdf_lo[lo_mask] = get_cdf_vec(x_lo_prev[lo_mask])
            
            # Final count for this bucket with safety floor
            model_pred = np.maximum(0.0, cdf_hi - cdf_lo) * b_count
            uniform_pred = ((hi_clamped - lo_clamped + 1) / b_width) * b_count
            
            total_counts[mask] += np.maximum(model_pred, uniform_pred * 0.01)

        return total_counts

    def _bake_vectorized_data(self):
        n = len(self.buckets)
        self.b_lo = np.array([b.lo for b in self.buckets], dtype=np.float64)
        self.b_hi = np.array([b.hi for b in self.buckets], dtype=np.float64)
        self.b_count = np.array([b.count for b in self.buckets], dtype=np.float64)
        self.b_width = self.b_hi - self.b_lo + 1
        
        self.mod_types = np.zeros(n, dtype=np.int8)
        self.lin_params = np.zeros((n, 2))
        self.poly_params = np.zeros((n, 3))
        self.log_params = np.zeros((n, 2))
        self.complex_models = {}

        for i in range(n):
            model = self.models.get(i)
            if model is None:
                self.mod_types[i] = 0 # Identity
            elif isinstance(model, tuple):
                if model[0] == "linear":
                    self.mod_types[i] = 1
                    self.lin_params[i] = [model[1], model[2]] # coef, intercept
                elif model[0] == "poly":
                    self.mod_types[i] = 2
                    # model: ("poly", coefs, intercept) -> coefs is [a, b]
                    self.poly_params[i] = [model[1][0], model[1][1], model[2]]
                elif model[0] == "log_linear":
                    self.mod_types[i] = 4
                    self.log_params[i] = [model[1], model[2]]
            else:
                self.mod_types[i] = 3 # Complex
                self.complex_models[i] = model



        
    def _collect_cdf_training_rows(self, freq: np.ndarray, mn: int, points_per_bucket: int, rng: np.random.Generator, bucket_indices: List[int] = None) -> Dict[int, Tuple[List[CDFTrainRow], List[CDFTrainRow]]]:
        ps = np.cumsum(freq)
        target_indices = bucket_indices if bucket_indices is not None else range(len(self.buckets))
        rows_data = {}
        
        for i in target_indices:
            b = self.buckets[i]
            rows_data[i] = ([], []) # Train, Val
            if b.count == 0: continue
            
            width = b.hi - b.lo + 1
            b_lo_idx = b.lo - mn
            base_cnt = ps[b_lo_idx - 1] if b_lo_idx > 0 else 0

            # --- Sample Generation ---
            # Training: Mixed Sampling (Uniform + Log-Uniform for Head)
            n_train = max(points_per_bucket * 5, 1000)
            
            # 50% Uniform Random
            n_unif = n_train // 2
            xs_unif = rng.integers(b.lo, b.hi + 1, size=n_unif)
            
            # 50% Log-Uniform (to catch the head of Zipf)
            n_log = n_train - n_unif
            if b.hi > b.lo:
                start_log = max(1, b.lo) if b.lo > 0 else 1
                end_log = max(start_log + 1, b.hi)
                log_space = np.geomspace(start_log, end_log, num=n_log).astype(int)
                xs_log = np.clip(log_space, b.lo, b.hi)
            else:
                xs_log = np.array([b.lo] * n_log)
                
            xs_train = np.concatenate([xs_unif, xs_log])
            
            # Validation: Mixed Sampling (Uniform + Random Log-Use for Generalization)
            n_val = max(points_per_bucket, 200)
            n_val_unif = n_val // 2
            xs_val_unif = rng.integers(b.lo, b.hi + 1, size=n_val_unif)
            
            # Add log-spaced validation points to test head fit
            n_val_log = n_val - n_val_unif
            if b.hi > b.lo:
                start_log = max(1, b.lo) if b.lo > 0 else 1
                end_log = max(start_log + 1, b.hi)
                log_space_val = np.geomspace(start_log, end_log, num=n_val_log).astype(int)
                # Jitter the log points slightly for validation to avoid testing on exact train points
                jitter = rng.integers(-1, 2, size=len(log_space_val)) 
                xs_val_log = np.clip(log_space_val + jitter, b.lo, b.hi)
            else:
                xs_val_log = np.array([b.lo] * n_val_log)
                
            xs_val = np.concatenate([xs_val_unif, xs_val_log])

            # CRITICAL: Always include boundaries and near-boundary points in TRAINING
            log_points = np.array([], dtype=int)
            if b.hi > b.lo:
                try:
                    start_val = max(1, b.lo)
                    if start_val < b.hi:
                        log_points = np.unique(np.geomspace(start_val, b.hi, num=50, dtype=int))
                except Exception:
                    pass
            
            edge_points = np.concatenate([
                np.arange(b.lo, min(b.hi, b.lo + 50) + 1), # First 50 points dense
                log_points,                                 # Log spaced points
                np.array([b.hi, max(b.lo, b.hi - 1), max(b.lo, b.hi - 2)]) # End points
            ])
            xs_train = np.concatenate([xs_train, edge_points])
            xs_train = np.unique(xs_train)
            xs_train = np.sort(xs_train)
            
            # Helper to build rows
            def build_rows(xs_arr):
                res = []
                for x in xs_arr:
                    x_idx = x - mn
                    if x_idx < 0 or x_idx >= len(ps): continue
                    y_cdf = (ps[x_idx] - base_cnt) / b.count
                    res.append(CDFTrainRow(x_norm=(x - b.lo) / width, y_cdf=y_cdf))
                return res

            rows_data[i] = (build_rows(xs_train), build_rows(xs_val))
            
        return rows_data

    def _generate_bucket_queries(self, b: Bucket, n_queries: int, rng, freq: np.ndarray, mn: int) -> List[RangeQuery]:
        """
        Generate synthetic range queries for Q-Error evaluation.
        Uses uniform sampling to match the benchmark's query workload generation strategy.
        """
        queries = []
        for _ in range(n_queries):
            lo = rng.integers(b.lo, b.hi + 1)
            hi = rng.integers(lo, b.hi + 1)
            queries.append(RangeQuery(lo, hi))
        return queries

    def _eval_model_q_error(self, model, queries: List[RangeQuery], bucket: Bucket, 
                           freq: np.ndarray, mn: int) -> float:
        """
        Evaluate median Q-Error for a model on a query set.
        
        Args:
            model: Model to evaluate (tuple format or FourierModelWrapper)
            queries: List of range queries
            bucket: Bucket being evaluated
            freq: Frequency array for ground truth
            mn: Minimum value of dataset
            
        Returns:
            Median Q-Error across all queries
        """
        q_errors = []
        width = bucket.hi - bucket.lo + 1
        
        for q in queries:
            # Ground truth from frequency array
            idx_lo = q.low - mn
            idx_hi = q.high - mn
            act_count = np.sum(freq[idx_lo:idx_hi + 1])
            
            # Prediction from model
            lo_norm = (q.low - bucket.lo) / width
            hi_norm = (q.high - bucket.lo) / width
            
            val_hi = self._predict_local_cdf(model, hi_norm)
            val_lo = self._predict_local_cdf(model, lo_norm)
            pred_count = (val_hi - val_lo) * bucket.count
            
            # Q-Error (avoid division by zero)
            act = max(1, act_count)
            pred = max(1, pred_count)
            q_err = max(act / pred, pred / act)
            q_errors.append(q_err)
        
        return np.median(q_errors)


    def _train_adaptive_models(self, rows_data: Dict[int, Tuple[List[CDFTrainRow], List[CDFTrainRow]]], 
                              freq: np.ndarray, mn: int, rng: np.random.Generator) -> Tuple[Dict[int, Any], float]:
        models = {}
        t0 = time.perf_counter()
        
        # Q-Error threshold for "good enough" - if a model achieves this, stop early
        q_error_threshold = 2.0
        
        for i, (train_rows, val_rows) in rows_data.items():
            if not train_rows:
                models[i] = None # Fallback to uniform (diagonal)
                continue
            
            bucket = self.buckets[i]
            
            # Prepare Training Data
            X_train = np.array([r.x_norm for r in train_rows]).reshape(-1, 1)
            y_train = np.array([r.y_cdf for r in train_rows])
            
            # Prepare Validation Data (or fallback to Train if empty)
            if val_rows:
                X_val = np.array([r.x_norm for r in val_rows]).reshape(-1, 1)
                y_val = np.array([r.y_cdf for r in val_rows])
            else:
                X_val, y_val = X_train, y_train
            
            # Generate validation queries for Q-Error evaluation
            val_queries = self._generate_bucket_queries(bucket, n_queries=200, rng=rng, freq=freq, mn=mn)
            
            # --- WATERFALL: Try simple models first, stop early if good enough ---
            
            # 1. Identity Check (Uniform assumption)
            y_pred_identity_val = X_val.flatten() 
            mse_identity_val = np.mean((y_val - y_pred_identity_val)**2)
            if mse_identity_val < self.identity_threshold: 
                models[i] = None # Use Identity
                continue

            # 2. Linear Model
            mdl_linear = Ridge(alpha=1.0).fit(X_train, y_train)
            q_err_linear = self._eval_model_q_error(
                ("linear", mdl_linear.coef_[0], mdl_linear.intercept_),
                val_queries, bucket, freq, mn
            )
            
            if i < 3:  # Debug first few buckets
                print(f"[DEBUG] Bucket {i}: Linear Q-Error={q_err_linear:.2f}")
            
            if q_err_linear < q_error_threshold:
                models[i] = ("linear", mdl_linear.coef_[0], mdl_linear.intercept_)
                if i < 3:
                    print(f"[DEBUG] Bucket {i}: Selected Linear (early stop)")
                continue
            
            # 3. Polynomial Model (Degree 2)
            poly_calc = PolynomialFeatures(degree=2, include_bias=False)
            X_poly_train = poly_calc.fit_transform(X_train)
            mdl_poly = Ridge(alpha=1.0).fit(X_poly_train, y_train)
            q_err_poly = self._eval_model_q_error(
                ("poly", mdl_poly.coef_, mdl_poly.intercept_),
                val_queries, bucket, freq, mn
            )
            
            if i < 3:
                print(f"[DEBUG] Bucket {i}: Poly Q-Error={q_err_poly:.2f}")
            
            if q_err_poly < q_error_threshold:
                models[i] = ("poly", mdl_poly.coef_, mdl_poly.intercept_)
                if i < 3:
                    print(f"[DEBUG] Bucket {i}: Selected Poly (early stop)")
                continue

            # 4. Log-Linear Model  
            X_log_train = np.log(np.clip(X_train, 0.0, 1.0) + 1e-7)
            mdl_log = Ridge(alpha=1.0).fit(X_log_train, y_train)
            q_err_log = self._eval_model_q_error(
                ("log_linear", mdl_log.coef_[0], mdl_log.intercept_),
                val_queries, bucket, freq, mn
            )
            
            if i < 3:
                print(f"[DEBUG] Bucket {i}: Log-Linear Q-Error={q_err_log:.2f}")
            
            if q_err_log < q_error_threshold:
                models[i] = ("log_linear", mdl_log.coef_[0], mdl_log.intercept_)
                if i < 3:
                    print(f"[DEBUG] Bucket {i}: Selected Log-Linear (early stop)")
                continue

            # 5. Fallback: Complex Model (Fourier MLP) - only trained if simpler models aren't good enough
            try:
                mapper = FourierFeatureMapper(num_bands=32, max_freq=20000.0)
                X_f_train = mapper.transform(X_train)
                mdl_f_mlp = MLPRegressor(
                    hidden_layer_sizes=(128, 64), 
                    activation='relu', 
                    solver='lbfgs', 
                    max_iter=1000, 
                    random_state=42, 
                    alpha=0.001
                )
                mdl_f_mlp.fit(X_f_train, y_train)
                mdl_fourier = FourierModelWrapper(mdl_f_mlp, mapper)
                
                q_err_fourier = self._eval_model_q_error(mdl_fourier, val_queries, bucket, freq, mn)
                
                if i < 3:
                    print(f"[DEBUG] Bucket {i}: Fourier Q-Error={q_err_fourier:.2f}")
                
                # Select the best model among ALL candidates (including Fourier)
                best_model = min(
                    (q_err_linear, ("linear", mdl_linear.coef_[0], mdl_linear.intercept_)),
                    (q_err_poly, ("poly", mdl_poly.coef_, mdl_poly.intercept_)),
                    (q_err_log, ("log_linear", mdl_log.coef_[0], mdl_log.intercept_)),
                    (q_err_fourier, mdl_fourier)
                )
                
                models[i] = best_model[1]
                
                if i < 3:
                    model_name = best_model[1][0] if isinstance(best_model[1], tuple) else "Fourier"
                    print(f"[DEBUG] Bucket {i}: Selected {model_name} with Q-Error={best_model[0]:.2f}")
                
            except Exception:
                # Absolute Fallback: pick best of the simple models
                best_simple = min(
                    (q_err_linear, ("linear", mdl_linear.coef_[0], mdl_linear.intercept_)),
                    (q_err_poly, ("poly", mdl_poly.coef_, mdl_poly.intercept_)),
                    (q_err_log, ("log_linear", mdl_log.coef_[0], mdl_log.intercept_))
                )
                models[i] = best_simple[1]
                if i < 3:
                    print(f"[DEBUG] Bucket {i}: Selected best simple model (Fourier failed)")

            
        return models, time.perf_counter() - t0


    def _predict_local_cdf(self, model, x_norm) -> float:
        if model is None:
            return max(0.0, min(1.0, x_norm))
        
        # Fast Math for Linear/Poly
        if isinstance(model, tuple):
            type_ = model[0]
            if type_ == "linear":
                coef, intercept = model[1], model[2]
                pred = x_norm * coef + intercept
                return max(0.0, min(1.0, float(pred)))
            elif type_ == "poly":
                coefs, intercept = model[1], model[2]
                # coefs is [a, b] for ax + bx^2 because include_bias=False
                pred = intercept + coefs[0]*x_norm + coefs[1]*(x_norm**2)
                return max(0.0, min(1.0, float(pred)))
            elif type_ == "log_linear":
                coef, intercept = model[1], model[2]
                pred = coef * np.log(x_norm + 1e-7) + intercept
                return max(0.0, min(1.0, float(pred)))
        
        # Fallback for Wrapper (Fourier MLP) or others
        pred = model.predict([[x_norm]])[0]
        return max(0.0, min(1.0, float(pred)))

    def _get_bucket_overlap_count(self, b_idx: int, q_lo: int, q_hi: int) -> float:
        b = self.buckets[b_idx]
        lo = max(q_lo, b.lo)
        hi = min(q_hi, b.hi)
        if lo > hi: return 0.0
        w = b.hi - b.lo + 1
        cdf_hi = self._predict_local_cdf(self.models.get(b_idx), (hi - b.lo) / w)
        prev = lo - 1
        cdf_lo = 0.0 if prev < b.lo else self._predict_local_cdf(self.models.get(b_idx), (prev - b.lo) / w)
        
        model_pred = max(0.0, cdf_hi - cdf_lo) * b.count
        
        # Safety Floor: Prevent zero prediction if bucket is populated
        # This guards against non-monotonicity or gaps in FourierMLP
        if b.count > 0 and w > 0:
            uniform_pred = ((hi - lo + 1) / w) * b.count
            return max(model_pred, uniform_pred * 0.01) # Min 1% of uniform
            
        return model_pred

# -----------------------------------------------------------------------------
# Evaluation Utilities
# -----------------------------------------------------------------------------

def identify_bad_buckets(queries: List[RangeQuery], y_true: np.ndarray, y_pred: np.ndarray, buckets: List[Bucket], threshold_q: float = 1.05) -> List[int]:
    bad_buckets = set()
    qe = q_error_vec(y_true, y_pred)
    for i, err in enumerate(qe):
        if err > threshold_q:
            q = queries[i]
            for b_idx, b in enumerate(buckets):
                 if b.hi < q.low: continue
                 if b.lo > q.high: break
                 bad_buckets.add(b_idx)
    return list(bad_buckets)

def q_error_vec(y_true, y_pred, eps=1e-9):
    yt = np.maximum(y_true, eps)
    yp = np.maximum(y_pred, eps)
    return np.maximum(yt/yp, yp/yt)

def summarize(y_true, y_pred, name="Model"):
    qe = q_error_vec(y_true, y_pred)
    med = float(np.median(qe))
    p25 = float(np.percentile(qe, 25))
    p75 = float(np.percentile(qe, 75))
    p95 = float(np.percentile(qe, 95))
    avg = float(np.mean(qe))
    print(f"[{name}] Median QErr={med:.4f}, P25 QErr={p25:.4f}, P75 QErr={p75:.4f}, P95 QErr={p95:.4f}, Avg QErr={avg:.4f}")
    return {"name": name, "QErr_median": med, "QErr_p25": p25, "QErr_p75": p75, "QErr_p95": p95, "QErr_avg": avg}
