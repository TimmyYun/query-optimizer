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
        ratio = max(0.1, min(10.0, actual / pred))
        
        for b in self.buckets:
            ov_lo = max(q.low, b.lo)
            ov_hi = min(q.high, b.hi)
            if ov_lo <= ov_hi:
                w = b.hi - b.lo + 1
                overlap = (ov_hi - ov_lo + 1) / w
                factor = ratio ** (self.lr * overlap)
                b.count = int(max(1, b.count * factor))

class HybridEstimator:
    """
    Implements a Hybrid Selectivity Estimator combining histograms and ML models.
    """
    def __init__(self, buckets: List[Bucket], models: Dict[int, Any] = None):
        self.buckets = buckets
        self.models = models if models is not None else {}
        self.last_train_time = 0.0

    def train(self, freq: np.ndarray, mn: int, points_per_bucket: int, rng: np.random.Generator, bucket_indices: List[int] = None) -> float:
        t_start = time.perf_counter()
        
        # Train models directly on the raw frequency distribution (No MCV removal)
        rows = self._collect_cdf_training_rows(freq, mn, points_per_bucket, rng, bucket_indices)
        new_models, t_train_models = self._train_adaptive_models(rows)
        
        for k, v in new_models.items():
            self.models[k] = v
            
        t_total = time.perf_counter() - t_start
        self.last_train_time = t_total
        return t_total

    def predict(self, q: RangeQuery) -> float:
        total = 0.0
        
        # Model Contribution only (No MCV)
        for i, b in enumerate(self.buckets):
            if b.hi < q.low: continue
            if b.lo > q.high: break
            
            total += self._get_bucket_overlap_count(i, q.low, q.high)
        return total
        
    def _collect_cdf_training_rows(self, freq: np.ndarray, mn: int, points_per_bucket: int, rng: np.random.Generator, bucket_indices: List[int] = None) -> Dict[int, List[CDFTrainRow]]:
        ps = np.cumsum(freq)
        target_indices = bucket_indices if bucket_indices is not None else range(len(self.buckets))
        rows = {}
        for i in target_indices:
            b = self.buckets[i]
            rows[i] = [] 
            if b.count == 0: continue

            
            # Start with random uniform samples
            n_samples = max(points_per_bucket * 5, 1000) 
            xs = rng.integers(b.lo, b.hi + 1, size=n_samples)
            
            # CRITICAL: Always include boundaries and near-boundary points
            # For Zipf/Skewed data, the CDF jumps massively at lo, lo+1, etc.
            # If we don't sample these, the model learns a smooth line that misses the jump.
            # Expanded to first 50 points to cover the "head" of the distribution where curvature is highest.
            # Also adding log-spaced points to cover the body/tail to prevent oscillation gaps.
            # Also adding log-spaced points to cover the body/tail to prevent oscillation gaps.
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
            xs = np.concatenate([xs, edge_points])
            xs = np.unique(xs) # Remove duplicates
            xs = np.sort(xs)
            
            width = b.hi - b.lo + 1
            b_lo_idx = b.lo - mn
            base_cnt = ps[b_lo_idx - 1] if b_lo_idx > 0 else 0
            for x in xs:
                x_idx = x - mn
                if x_idx < 0 or x_idx >= len(ps): continue
                y_cdf = (ps[x_idx] - base_cnt) / b.count
                rows[i].append(CDFTrainRow(x_norm=(x - b.lo) / width, y_cdf=y_cdf))
        return rows

    def _train_adaptive_models(self, rows: Dict[int, List[CDFTrainRow]]) -> Tuple[Dict[int, Any], float]:
        models = {}
        t0 = time.perf_counter()
        
        for i, rlist in rows.items():
            if not rlist:
                models[i] = None # Fallback to uniform (diagonal)
                continue
                
            X = np.array([r.x_norm for r in rlist]).reshape(-1, 1)
            y = np.array([r.y_cdf for r in rlist])
            

            
            # --- Adaptive Selection ---
            candidates = []
            
            # 1. Constant (Zero Variance / Uniform assumption) - Baseline
            # Equivalent to predicting y = x (since we model CDF of uniform as linear diagonal)
            # We don't train a model for this, we just calculate error of "Identity" prediction x_norm
            y_pred_identity = X.flatten() # Predict y = x
            mse_identity = np.mean((y - y_pred_identity)**2)
            
            # Optimization: If Identity is perfect, skip others to prevent overfitting noise
            if mse_identity < 1e-5:
                models[i] = None
                continue

            candidates.append((mse_identity, "identity", None))
            
            # 2. Linear (Ridge)
            mdl_linear = Ridge(alpha=1.0)
            mdl_linear.fit(X, y)
            y_pred_linear = mdl_linear.predict(X)
            mse_linear = np.mean((y - y_pred_linear)**2)
            # Penalty for complexity (AIC-like): MSE * (1 + p/n)
            candidates.append((mse_linear * 1.1, "linear", mdl_linear))
            
            # 3. Polynomial (Degree 2)
            mdl_poly = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), Ridge(alpha=1.0))
            mdl_poly.fit(X, y)
            y_pred_poly = mdl_poly.predict(X)
            mse_poly = np.mean((y - y_pred_poly)**2)
             # Higher Penalty for poly
            candidates.append((mse_poly * 1.2, "poly", mdl_poly))
            
            # 4. Neural Network (MLP) - Standard
            mdl_mlp = MLPRegressor(hidden_layer_sizes=(16, 8), activation='relu', solver='lbfgs', max_iter=500, random_state=42)
            try:
                mdl_mlp.fit(X, y)
                y_pred_mlp = mdl_mlp.predict(X)
                mse_mlp = np.mean((y - y_pred_mlp)**2)
                candidates.append((mse_mlp * 1.5, "mlp", mdl_mlp))
            except Exception:
                pass 
            
            # 5. Fourier Neural Network (Positional Encoding for High Frequency)
            # ... (Existing code kept below, but I will insert Tree before or after)
            

            



            # 5. Fourier Neural Network (Positional Encoding for High Frequency)
            # Solves the Zipf problem by mapping input x to high-freq sinusoids
            # Gamma(x) = [sin(2^0 pi x), cos(2^0 pi x), ..., sin(2^k pi x), cos(2^k pi x)]
            try:
                # B = 32 frequency bands
                # Tuned to 20k to resolve head (Period ~0.25 units) with high regularization to prevent tail oscillation
                mapper = FourierFeatureMapper(num_bands=32, max_freq=20000.0)
                X_fourier = mapper.transform(X)
                
                # Relu for sharpness, alpha=0.001 to dampen oscillations
                mdl_fourier_mlp = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='lbfgs', max_iter=1000, random_state=42, alpha=0.001)
                mdl_fourier_mlp.fit(X_fourier, y)
                
                y_pred_f = mdl_fourier_mlp.predict(X_fourier)
                mse_f = np.mean((y - y_pred_f)**2)
                
                # Reduced penalty for Fourier MLP to encourage its selection on difficult buckets
                candidates.append((mse_f * 1.05, "fourier_mlp", (mdl_fourier_mlp, mapper)))
            except Exception:
                pass
            
            # Select Best
            best_candidate = min(candidates, key=lambda x: x[0])
            best_model_name = best_candidate[1]
            best_model_obj = best_candidate[2]
            
            if best_model_name == "fourier_mlp":
                # Tuple (model, mapper)
                model, mapper = best_model_obj
                models[i] = FourierModelWrapper(model, mapper)
            else:
                models[i] = best_model_obj
            
        return models, time.perf_counter() - t0

    def _predict_local_cdf(self, model, x_norm) -> float:
        if model is None: return max(0.0, min(1.0, x_norm))
        # Ensure prediction is within [0, 1]
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
            return max(model_pred, uniform_pred * 0.001) # Min 0.1% of uniform
            
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
    p95 = float(np.percentile(qe, 95))
    avg = float(np.mean(qe))
    print(f"[{name}] Median QErr={med:.4f}, P95 QErr={p95:.4f}, Avg QErr={avg:.4f}")
    return {"name": name, "QErr_median": med, "QErr_p95": p95, "QErr_avg": avg}

