import numpy as np
import math
import time
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor

class LogModelWrapper:
    """
    Wraps a model (like MLP) to predict in Log-Space.
    Trained on log1p(y), so predict must return expm1(pred).
    """
    def __init__(self, model):
        self.model = model
    
    def predict(self, X):
        pred_log = self.model.predict(X)
        pred = np.expm1(pred_log)
        return np.maximum(pred, 0) # Clamp negative values


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
            n_samples = max(points_per_bucket, 200) # Increased sample size for stability (was 50)
            xs = np.sort(rng.integers(b.lo, b.hi + 1, size=n_samples))
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
            
            # 5. Log-Space Neural Network (Deep Learning for Power Laws)
            mdl_log_mlp = MLPRegressor(hidden_layer_sizes=(16, 8), activation='relu', solver='lbfgs', max_iter=500, random_state=42)
            try:
                # Transform targets to Log-Space: y -> log(y + 1)
                y_log = np.log1p(y)
                mdl_log_mlp.fit(X, y_log)
                
                # Predict in Log-Space
                y_pred_log = mdl_log_mlp.predict(X)
                
                # Inverse Transform: y_pred -> exp(y_log) - 1
                y_pred_log_mlp = np.expm1(y_pred_log)
                y_pred_log_mlp = np.maximum(y_pred_log_mlp, 0)
                
                mse_log_mlp = np.mean((y - y_pred_log_mlp)**2)
                candidates.append((mse_log_mlp * 1.5, "log_mlp", mdl_log_mlp))
            except Exception:
                pass
            
            # Select Best
            best_candidate = min(candidates, key=lambda x: x[0])
            best_model_name = best_candidate[1]
            best_model_obj = best_candidate[2]
            
            if best_model_name == "log_mlp":
                best_model = LogModelWrapper(best_model_obj)
            else:
                best_model = best_model_obj
                
            models[i] = best_model
            
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
        return max(0.0, cdf_hi - cdf_lo) * b.count

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

