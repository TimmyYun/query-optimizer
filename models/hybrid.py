import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Any
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

from .common import Bucket, RangeQuery, CDFTrainRow

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
        self.power_params = None # NEW: Storage for the power-law model
        self.complex_models = {} # dict for fallback


    def train(self, freq: np.ndarray, mn: int, points_per_bucket: int, rng: np.random.Generator, bucket_indices: List[int] = None) -> float:
        t_start = time.perf_counter()
        
        # Helper to check data health
        if freq is None:
            print("[CRITICAL ERROR] freq is None in HybridEstimator.train")

        # Train models directly on the raw frequency distribution (No MCV removal)
        rows_data = self._collect_cdf_training_rows(freq, mn, points_per_bucket, rng, bucket_indices)
        new_models, t_train_models = self._train_adaptive_models(rows_data, freq, mn, rng)
        

        if len(new_models) == 0:
            print("[CRITICAL WARNING] No models were trained! Hybrid estimator will predict 0.0 everywhere.")

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
                if m_type == 0:  # Identity
                    return xa
                elif m_type == 1:  # Linear
                    p = self.lin_params[i]
                    return np.clip(xa * p[0] + p[1], 0.0, 1.0)
                elif m_type == 2:  # Poly
                    p = self.poly_params[i]
                    return np.clip(p[2] + p[0] * xa + p[1] * (xa ** 2), 0.0, 1.0)
                elif m_type == 4:  # Log-Linear
                    p = self.log_params[i]
                    return np.clip(p[0] * np.log(xa + 1e-7) + p[1], 0.0, 1.0)
                elif m_type == 5:  # NEW: Power
                    p = self.power_params[i]
                    return np.clip(p[0] * np.sqrt(xa) + p[1], 0.0, 1.0)
                else:  # Complex (MLP or Tree)
                    mdl = self.complex_models[i]
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
        self.power_params = np.zeros((n, 2))  # NEW
        self.complex_models = {}

        for i in range(n):
            model = self.models.get(i)
            if model is None:
                self.mod_types[i] = 0  # Identity
            elif isinstance(model, tuple):
                if model[0] == "linear":
                    self.mod_types[i] = 1
                    self.lin_params[i] = [model[1], model[2]]
                elif model[0] == "poly":
                    self.mod_types[i] = 2
                    self.poly_params[i] = [model[1][0], model[1][1], model[2]]
                elif model[0] == "log_linear":
                    self.mod_types[i] = 4
                    self.log_params[i] = [model[1], model[2]]
                elif model[0] == "power":  # NEW
                    self.mod_types[i] = 5
                    self.power_params[i] = [model[1], model[2]]
            else:
                # Decision Tree or Fourier MLP
                self.mod_types[i] = 3
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
            Tuple[float, float]: (Median Q-Error, 95th Percentile Q-Error)
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
        
        if not q_errors:
            return 1.0, 1.0

        return np.median(q_errors), np.percentile(q_errors, 95)


    def _train_adaptive_models(self, rows_data: Dict[int, Tuple[List[CDFTrainRow], List[CDFTrainRow]]], 
                          freq: np.ndarray, mn: int, rng: np.random.Generator) -> Tuple[Dict[int, Any], float]:
        models = {}
        t0 = time.perf_counter()
        
        # LOWERED threshold: Forces the estimator to stop being lazy and picking "Uniform"
        complex_model_threshold = 1.05 
        
        for i, (train_rows, val_rows) in rows_data.items():
            if not train_rows:
                models[i] = None 
                continue
            
            bucket = self.buckets[i]
            
            X_train = np.array([r.x_norm for r in train_rows]).reshape(-1, 1)
            y_train = np.array([r.y_cdf for r in train_rows])
            
            if val_rows:
                X_val = np.array([r.x_norm for r in val_rows]).reshape(-1, 1)
                y_val = np.array([r.y_cdf for r in val_rows])
            else:
                X_val, y_val = X_train, y_train
            
            # WEIGHTING: Force ML models to care about the tail of the bucket
            weights = 1.0 + 10.0 * (y_train ** 2)
            
            val_queries = self._generate_bucket_queries(bucket, n_queries=2000, rng=rng, freq=freq, mn=mn)
            candidates = []

            # 1. Identity (Uniform)
            q_med_identity, q_p95_identity = self._eval_model_q_error(None, val_queries, bucket, freq, mn)
            candidates.append((q_med_identity, q_p95_identity, None))

            # 2. Linear Model (Added sample_weight)
            try:
                mdl_linear = Ridge(alpha=1.0).fit(X_train, y_train, sample_weight=weights)
                q_med_linear, q_p95_linear = self._eval_model_q_error(
                    ("linear", mdl_linear.coef_[0], mdl_linear.intercept_), val_queries, bucket, freq, mn)
                candidates.append((q_med_linear, q_p95_linear, ("linear", mdl_linear.coef_[0], mdl_linear.intercept_)))
            except: pass
            
            # 3. Polynomial Model (Added sample_weight)
            try:
                poly_calc = PolynomialFeatures(degree=2, include_bias=False)
                X_poly_train = poly_calc.fit_transform(X_train)
                mdl_poly = Ridge(alpha=1.0).fit(X_poly_train, y_train, sample_weight=weights)
                q_med_poly, q_p95_poly = self._eval_model_q_error(
                    ("poly", mdl_poly.coef_, mdl_poly.intercept_), val_queries, bucket, freq, mn)
                candidates.append((q_med_poly, q_p95_poly, ("poly", mdl_poly.coef_, mdl_poly.intercept_)))
            except: pass

            # 4. Log-Linear Model (Added sample_weight)
            try:
                X_log_train = np.log(np.clip(X_train, 0.0, 1.0) + 1e-7)
                mdl_log = Ridge(alpha=1.0).fit(X_log_train, y_train, sample_weight=weights)
                q_med_log, q_p95_log = self._eval_model_q_error(
                    ("log_linear", mdl_log.coef_[0], mdl_log.intercept_), val_queries, bucket, freq, mn)
                candidates.append((q_med_log, q_p95_log, ("log_linear", mdl_log.coef_[0], mdl_log.intercept_)))
            except: pass

            # 5. NEW: Power/Reciprocal Model (Matches Zipf Decay)
            try:
                X_power_train = np.sqrt(np.clip(X_train, 0.0, 1.0))
                mdl_power = Ridge(alpha=1.0).fit(X_power_train, y_train, sample_weight=weights)
                q_med_power, q_p95_power = self._eval_model_q_error(
                    ("power", mdl_power.coef_[0], mdl_power.intercept_), val_queries, bucket, freq, mn)
                candidates.append((q_med_power, q_p95_power, ("power", mdl_power.coef_[0], mdl_power.intercept_)))
            except: pass

            # 6. NEW: Decision Tree (Sub-bucket learned histogram)
            try:
                mdl_tree = DecisionTreeRegressor(max_depth=4).fit(X_train, y_train)
                q_med_tree, q_p95_tree = self._eval_model_q_error(mdl_tree, val_queries, bucket, freq, mn)
                candidates.append((q_med_tree, q_p95_tree, mdl_tree))
            except: pass

            # Pick best cheap model
            best_cheap_q, best_cheap_p95, best_cheap_model = min(candidates, key=lambda x: x[0])
            
            final_model = best_cheap_model
            
            # Fallback to Fourier only if ALL models failed the 1.05 threshold
            if best_cheap_q > complex_model_threshold:
                try:
                    mapper = FourierFeatureMapper(num_bands=32, max_freq=20000.0)
                    X_f_train = mapper.transform(X_train)
                    mdl_f_mlp = MLPRegressor(
                        hidden_layer_sizes=(128, 64), activation='relu', solver='lbfgs', 
                        max_iter=1000, random_state=42, alpha=0.001
                    )
                    mdl_f_mlp.fit(X_f_train, y_train)
                    mdl_fourier = FourierModelWrapper(mdl_f_mlp, mapper)
                    
                    q_med_fourier, q_p95_fourier = self._eval_model_q_error(mdl_fourier, val_queries, bucket, freq, mn)
                    if q_med_fourier < best_cheap_q:
                        final_model = mdl_fourier
                except Exception as e: pass

            models[i] = final_model
            
        return models, time.perf_counter() - t0

    def _predict_local_cdf(self, model, x_norm) -> float:
        if model is None:
            return max(0.0, min(1.0, x_norm))

        # Fast Math for Tuple Models
        if isinstance(model, tuple):
            type_ = model[0]
            if type_ == "linear":
                coef, intercept = model[1], model[2]
                pred = x_norm * coef + intercept
            elif type_ == "poly":
                coefs, intercept = model[1], model[2]
                pred = intercept + coefs[0] * x_norm + coefs[1] * (x_norm ** 2)
            elif type_ == "log_linear":
                coef, intercept = model[1], model[2]
                pred = coef * np.log(x_norm + 1e-7) + intercept
            elif type_ == "power":  # NEW
                coef, intercept = model[1], model[2]
                pred = coef * np.sqrt(x_norm) + intercept
            return max(0.0, min(1.0, float(pred)))

        # Fallback for DecisionTreeRegressor or Fourier MLP
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

    def report_models(self, file_path=None):
        """Prints a summary of which models were chosen for each bucket."""
        output = []
        output.append("\n" + "="*80)
        output.append(f"{'Idx':<4} | {'Range':<25} | {'Count':<10} | {'Model Type':<15} | {'Stats'}")
        output.append("-" * 80)
        
        for i, b in enumerate(self.buckets):
            model = self.models.get(i)
            m_name = "Uniform"
            info = ""
            
            if model is None:
                m_name = "Uniform"
            elif isinstance(model, tuple):
                m_type = model[0]
                if m_type == "linear":
                    m_name = "Linear"
                    info = f"coef={model[1]:.4f}, int={model[2]:.4f}"
                elif m_type == "poly":
                    m_name = "Polynomial"
                    info = f"c1={model[1][0]:.4f}, c2={model[1][1]:.4f}, int={model[2]:.4f}"
                elif m_type == "log_linear":
                    m_name = "Log-Linear"
                    info = f"coef={model[1]:.4f}, int={model[2]:.4f}"
            else:
                m_name = "Fourier MLP"
                info = "Complex Neural Net"
                
            range_str = f"[{b.lo}, {b.hi}]"
            output.append(f"{i:<4} | {range_str:<25} | {b.count:<10} | {m_name:<15} | {info}")
        output.append("="*80 + "\n")
        
        report_text = "\n".join(output)
        print(report_text)
        
        if file_path:
            with open(file_path, "w") as f:
                f.write(report_text)
            print(f"Model selection report saved to {file_path}")
