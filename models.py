import numpy as np
import math
import time
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass
from sklearn.isotonic import IsotonicRegression

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
        rows = self._collect_cdf_training_rows(freq, mn, points_per_bucket, rng, bucket_indices)
        new_models, t_train = self._train_cdf_models(rows)
        for k, v in new_models.items():
            self.models[k] = v
        self.last_train_time = t_train
        return t_train

    def predict(self, q: RangeQuery) -> float:
        total = 0.0
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
            n_samples = max(points_per_bucket, 50) 
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

    def _train_cdf_models(self, rows: Dict[int, List[CDFTrainRow]]) -> Tuple[Dict[int, Any], float]:
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

    def _predict_local_cdf(self, model, x_norm) -> float:
        if model is None: return max(0.0, min(1.0, x_norm))
        return float(model.transform([x_norm])[0])

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

