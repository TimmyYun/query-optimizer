import numpy as np
import time
from typing import List, Dict, Tuple, Any
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.isotonic import IsotonicRegression

from .common import Bucket, RangeQuery, CDFTrainRow


class FourierFeatureMapper:
    """
    Maps scalar input x to high-frequency sinusoids (Positional Encoding).
    Allows MLP to learn sharp transitions/spikes.
    """

    def __init__(self, num_bands: int = 10, max_freq: float = 1024.0):
        self.num_bands = num_bands
        self.freqs = np.logspace(0, np.log10(max_freq), num=num_bands, base=10)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X_log = np.log(X + 1e-7)
        features = [X, X_log]

        for freq in self.freqs:
            scaled = X * freq * np.pi
            features.append(np.sin(scaled))
            features.append(np.cos(scaled))

        return np.hstack(features)


class FourierModelWrapper:
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

    def __init__(
        self,
        buckets: List[Bucket],
        models: Dict[int, Any] = None,
        identity_threshold: float = 1e-4,
        mlp_penalty: float = 1.5,
        fourier_penalty: float = 1.5,
    ):
        self.buckets = buckets
        self.models = models if models is not None else {}
        self.last_train_time = 0.0

        # --- NEW: Locality Patches ---
        # Stores (center, sigma, weight) for local corrections
        self.local_patches = {}

        # Vectorized Data Storage
        self.b_lo = None
        self.b_hi = None
        self.b_count = None
        self.b_width = None
        self.mod_types = None
        self.lin_params = None
        self.poly_params = None
        self.poly3_params = None
        self.log_params = None
        self.power_params = None
        self.complex_models = {}

    def feedback_update(
        self, queries, y_true_counts, y_pred_counts, N, error_threshold=1.1
    ):
        b_lo_vals = np.array([b.lo for b in self.buckets])
        b_hi_vals = np.array([b.hi for b in self.buckets])
        is_dirty = False

        for idx, q in enumerate(queries):
            if y_true_counts[idx] <= 0:
                continue

            q_err = max(
                y_true_counts[idx] / (y_pred_counts[idx] + 1e-9),
                y_pred_counts[idx] / (y_true_counts[idx] + 1e-9),
            )

            if q_err > error_threshold:
                start_idx = np.searchsorted(b_hi_vals, q.low)
                end_idx = np.searchsorted(b_lo_vals, q.high, side="right")

                for i in range(start_idx, end_idx):
                    b = self.buckets[i]
                    # Get normalized range within the bucket [0, 1]
                    width = b.hi - b.lo + 1
                    x_lo = np.clip((q.low - b.lo) / width, 0.0, 1.0)
                    x_hi = np.clip((q.high - b.lo) / width, 0.0, 1.0)

                    # Calculate share of the query count belonging to this bucket
                    q_width = q.high - q.low + 1
                    overlap_width = min(q.high, b.hi) - max(q.low, b.lo) + 1
                    share = overlap_width / q_width

                    self._fine_tune_bucket(i, x_lo, x_hi, y_true_counts[idx] * share)
                    is_dirty = True

        if is_dirty:
            self._bake_vectorized_data()

    def _fine_tune_bucket(self, b_idx, x_lo, x_hi, y_true_count, alpha=0.4):
        b = self.buckets[b_idx]
        if b.count == 0:
            return

        # 1. Calculate error in this specific range
        current_pred_mass = (
            self._predict_local_cdf(self.models.get(b_idx), x_hi)
            - self._predict_local_cdf(self.models.get(b_idx), x_lo)
        ) * b.count

        error_mass = (y_true_count - current_pred_mass) / b.count

        # 2. Add a Gaussian Patch
        center = (x_lo + x_hi) / 2.0
        # Sigma is proportional to the query width (minimum 0.05 to avoid spikes)
        sigma = max(0.05, (x_hi - x_lo) / 2.0)
        weight = error_mass * alpha

        if b_idx not in self.local_patches:
            self.local_patches[b_idx] = []

        # Keep only the last 5 patches to prevent performance degradation
        self.local_patches[b_idx].append((center, sigma, weight))
        if len(self.local_patches[b_idx]) > 5:
            self.local_patches[b_idx].pop(0)

    def train(
        self,
        freq: np.ndarray,
        mn: int,
        points_per_bucket: int,
        rng: np.random.Generator,
        bucket_indices: List[int] = None,
    ) -> float:
        t_start = time.perf_counter()

        if freq is None:
            print("[CRITICAL ERROR] freq is None")

        rows_data = self._collect_cdf_training_rows(
            freq, mn, points_per_bucket, rng, bucket_indices
        )
        new_models, t_train_models = self._train_adaptive_models(
            rows_data, freq, mn, rng
        )

        if len(new_models) == 0:
            print("[CRITICAL WARNING] No models trained!")

        for k, v in new_models.items():
            self.models[k] = v

        self._bake_vectorized_data()

        self.last_train_time = time.perf_counter() - t_start
        return self.last_train_time

    def predict(self, q: RangeQuery) -> float:
        total = 0.0
        for i in range(len(self.buckets)):
            if self.buckets[i].count == 0:
                continue
            total += self._get_bucket_overlap_count(i, q.low, q.high)
        return total

    def predict_batch(self, queries: List[RangeQuery]) -> np.ndarray:
        if self.b_lo is None or len(self.b_lo) != len(self.buckets):
            self._bake_vectorized_data()

        n_queries = len(queries)
        q_lo = np.array([q.low for q in queries], dtype=np.float64)
        q_hi = np.array([q.high for q in queries], dtype=np.float64)
        total_counts = np.zeros(n_queries)

        for i in range(len(self.buckets)):
            b_count = self.b_count[i]
            if b_count == 0: continue

            b_lo, b_hi, b_width = self.b_lo[i], self.b_hi[i], self.b_width[i]
            mask = (q_hi >= b_lo) & (q_lo <= b_hi)
            if not np.any(mask): continue

            # Normalized coordinates
            x_hi = np.clip((np.minimum(q_hi[mask], b_hi) - b_lo) / b_width, 0.0, 1.0)
            x_lo_prev = (np.maximum(q_lo[mask], b_lo) - 1 - b_lo) / b_width

            # --- FIX: CALL THE PATCH-AWARE FUNCTION ---
            model = self.models.get(i)
            cdf_hi = self._predict_local_cdf_vec(model, x_hi, b_idx=i)

            cdf_lo = np.zeros_like(cdf_hi)
            lo_active = x_lo_prev >= 0
            if np.any(lo_active):
                cdf_lo[lo_active] = self._predict_local_cdf_vec(model, x_lo_prev[lo_active], b_idx=i)

            model_pred = np.maximum(0.0, cdf_hi - cdf_lo) * b_count

            # 5% safety floor
            uniform_pred = ((np.minimum(q_hi[mask], b_hi) - np.maximum(q_lo[mask], b_lo) + 1) / b_width) * b_count
            total_counts[mask] += np.where(model_pred < 1e-3, uniform_pred * 0.05, model_pred)

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
        self.poly3_params = np.zeros((n, 4))
        self.log_params = np.zeros((n, 2))
        self.power_params = np.zeros((n, 2))
        self.complex_models = {}

        for i in range(n):
            model = self.models.get(i)
            if model is None:
                self.mod_types[i] = 0
            elif isinstance(model, tuple):
                m_t = model[0]
                if m_t == "linear":
                    self.mod_types[i] = 1
                    self.lin_params[i] = [model[1], model[2]]
                elif m_t == "poly":
                    self.mod_types[i] = 2
                    self.poly_params[i] = [model[1][0], model[1][1], model[2]]
                elif m_t == "log_linear":
                    self.mod_types[i] = 4
                    self.log_params[i] = [model[1], model[2]]
                elif m_t == "power":
                    self.mod_types[i] = 5
                    self.power_params[i] = [model[1], model[2]]
                elif m_t == "poly3":
                    self.mod_types[i] = 6
                    self.poly3_params[i] = [
                        model[1][0],
                        model[1][1],
                        model[1][2],
                        model[2],
                    ]
            else:
                if type(model).__name__ == "IsotonicRegression":
                    self.mod_types[i] = 7
                else:
                    self.mod_types[i] = 3
                self.complex_models[i] = model

    def _collect_cdf_training_rows(
        self,
        freq: np.ndarray,
        mn: int,
        points_per_bucket: int,
        rng: np.random.Generator,
        bucket_indices: List[int] = None,
    ) -> Dict[int, Tuple[List[CDFTrainRow], List[CDFTrainRow]]]:
        ps = np.cumsum(freq)
        target_indices = (
            bucket_indices if bucket_indices is not None else range(len(self.buckets))
        )
        rows_data = {}

        for i in target_indices:
            b = self.buckets[i]
            rows_data[i] = ([], [])
            if b.count == 0:
                continue

            width = b.hi - b.lo + 1
            b_lo_idx = b.lo - mn
            b_hi_idx = b.hi - mn

            # 1. Equidistant (Grid) Sampling
            # Forces the models to map the CDF across the entire bucket evenly
            n_grid = points_per_bucket * 2
            if width <= n_grid:
                xs_grid = np.arange(b.lo, b.hi + 1)
            else:
                xs_grid = np.linspace(b.lo, b.hi, num=n_grid, dtype=int)

            # 2. Uniform + Edge Sampling
            # Adds stochastic noise and explicitly pins the bucket boundaries
            n_unif = points_per_bucket
            xs_unif = rng.integers(b.lo, b.hi + 1, size=n_unif)
            xs_edges = np.array([b.lo, b.hi, b.lo + 1, max(b.lo, b.hi - 1)])

            xs_train = np.unique(np.concatenate([xs_grid, xs_unif, xs_edges]))
            xs_train = np.sort(xs_train)

            # Validation Sample (Purely Uniform)
            xs_val = np.unique(rng.integers(b.lo, b.hi + 1, size=points_per_bucket * 2))

            def build_rows(xs_arr):
                base_cnt = ps[b_lo_idx - 1] if b_lo_idx > 0 else 0
                res = []
                for x in xs_arr:
                    x_idx = x - mn
                    y_cdf = (ps[x_idx] - base_cnt) / b.count
                    res.append(
                        CDFTrainRow(
                            x_norm=(x - b.lo) / width, y_cdf=np.clip(y_cdf, 0.0, 1.0)
                        )
                    )
                return res

            rows_data[i] = (build_rows(xs_train), build_rows(xs_val))
        return rows_data

    def _generate_bucket_queries(
        self, b: Bucket, n_queries: int, rng, freq: np.ndarray, mn: int
    ) -> List[RangeQuery]:
        queries = []
        for _ in range(n_queries):
            lo = rng.integers(b.lo, b.hi + 1)
            hi = rng.integers(lo, b.hi + 1)
            queries.append(RangeQuery(lo, hi))
        return queries

    def _predict_local_cdf_vec(
        self, model, x_norm_arr: np.ndarray, b_idx: int = -1
    ) -> np.ndarray:
        # --- Base Model Prediction ---
        if model is None:
            base_pred = np.clip(x_norm_arr, 0.0, 1.0)
        elif isinstance(model, tuple):
            m_type = model[0]
            if m_type == "linear":
                base_pred = np.clip(x_norm_arr * model[1] + model[2], 0.0, 1.0)
            elif m_type == "poly3":
                p, inter = model[1], model[2]
                base_pred = np.clip(
                    inter
                    + p[0] * x_norm_arr
                    + p[1] * (x_norm_arr**2)
                    + p[2] * (x_norm_arr**3),
                    0.0,
                    1.0,
                )
            elif m_type == "power":
                base_pred = np.clip(model[1] * np.sqrt(x_norm_arr) + model[2], 0.0, 1.0)
            else:  # Default Linear
                base_pred = np.clip(x_norm_arr * model[1] + model[2], 0.0, 1.0)
        elif type(model).__name__ == "IsotonicRegression":
            base_pred = np.clip(model.predict(x_norm_arr), 0.0, 1.0)
        else:
            base_pred = np.clip(
                model.predict(x_norm_arr.reshape(-1, 1)).flatten(), 0.0, 1.0
            )

        # --- NEW: Apply Local Patches ---
        if b_idx != -1 and b_idx in self.local_patches:
            for center, sigma, weight in self.local_patches[b_idx]:
                # Gaussian Correction: weight * exp(-0.5 * ((x - center)/sigma)^2)
                # Note: We integrate this to keep it as a CDF nudge,
                # but for narrow patches, a simple Gaussian scaled by the sigmoid works well.
                correction = weight * np.exp(
                    -0.5 * ((x_norm_arr - center) / sigma) ** 2
                )
                base_pred += correction

        return np.clip(base_pred, 0.0, 1.0)

    def _eval_model_q_error_vec(
        self, model, queries_lo, queries_hi, bucket, b_ps, b_lo, width
    ):
        idx_hi = queries_hi - b_lo
        idx_lo = queries_lo - b_lo
        act_counts = b_ps[idx_hi] - np.where(idx_lo > 0, b_ps[idx_lo - 1], 0)
        act_counts = np.maximum(1.0, act_counts)

        lo_norm = (queries_lo - bucket.lo) / width
        hi_norm = (queries_hi - bucket.lo) / width

        val_hi = self._predict_local_cdf_vec(model, hi_norm)
        val_lo = self._predict_local_cdf_vec(model, lo_norm)
        pred_counts = np.maximum(1.0, (val_hi - val_lo) * bucket.count)

        q_errs = np.maximum(act_counts / pred_counts, pred_counts / act_counts)
        return np.median(q_errs), np.percentile(q_errs, 95)

    def _train_adaptive_models(
        self,
        rows_data: Dict[int, Tuple[List[CDFTrainRow], List[CDFTrainRow]]],
        freq: np.ndarray,
        mn: int,
        rng: np.random.Generator,
    ) -> Tuple[Dict[int, Any], float]:
        models = {}
        t0 = time.perf_counter()

        EARLY_EXIT_THRESHOLD = 1.2
        COMPLEX_MODEL_THRESHOLD = 1.05

        for i, (train_rows, val_rows) in rows_data.items():
            bucket = self.buckets[i]
            if not train_rows or bucket.count == 0:
                models[i] = None
                continue

            X_train = np.array([r.x_norm for r in train_rows]).reshape(-1, 1)
            y_train = np.array([r.y_cdf for r in train_rows])

            b_lo_idx, b_hi_idx = bucket.lo - mn, bucket.hi - mn
            b_ps = np.cumsum(freq[b_lo_idx : b_hi_idx + 1])
            width = bucket.hi - bucket.lo + 1

            # Map weights to exactly where the training mass is
            local_freq = freq[b_lo_idx : b_hi_idx + 1]
            local_probs = local_freq / (local_freq.sum() + 1e-9)
            x_indices = np.clip(
                np.array([r.x_norm * width for r in train_rows]).astype(int),
                0,
                width - 1,
            )
            weights = 1.0 + (local_probs[x_indices] / (local_probs.max() + 1e-9)) * 5.0

            val_queries = self._generate_bucket_queries(
                bucket, n_queries=500, rng=rng, freq=freq, mn=mn
            )
            q_lo = np.array([q.low for q in val_queries])
            q_hi = np.array([q.high for q in val_queries])

            def check(mdl):
                q_m, q_95 = self._eval_model_q_error_vec(
                    mdl, q_lo, q_hi, bucket, b_ps, bucket.lo, width
                )
                return q_m, q_95, mdl

            # 1. Identity (Uniform)
            best_q, best_p95, best_model = check(None)
            if best_q < EARLY_EXIT_THRESHOLD:
                models[i] = best_model
                continue

            candidates = []
            # 2. Linear & Power
            try:
                m_lin = Ridge(alpha=1.0).fit(X_train, y_train, sample_weight=weights)
                lin_mdl = ("linear", m_lin.coef_[0], m_lin.intercept_)
                q, p95, _ = check(lin_mdl)
                candidates.append((q, p95, lin_mdl))

                if q < 1.05:  # If Linear is perfect, skip Poly and MLP.
                    models[i] = lin_mdl
                    continue

                m_pow = Ridge(alpha=1.0).fit(
                    np.sqrt(X_train), y_train, sample_weight=weights
                )
                pow_mdl = ("power", m_pow.coef_[0], m_pow.intercept_)
                q, p95, _ = check(pow_mdl)
                candidates.append((q, p95, pow_mdl))
            except:
                pass

            if candidates:
                q, p95, mdl = min(candidates, key=lambda x: x[0])
                if q < best_q:
                    best_q, best_p95, best_model = q, p95, mdl

            if best_q < EARLY_EXIT_THRESHOLD:
                models[i] = best_model
                continue

            # 3. Try Cubic Poly (S-Curve)
            try:
                poly_calc = PolynomialFeatures(degree=3, include_bias=False)
                X_poly_train = poly_calc.fit_transform(X_train)
                mdl_poly = Ridge(alpha=0.1).fit(
                    X_poly_train, y_train, sample_weight=weights
                )
                poly_mdl = ("poly3", mdl_poly.coef_, mdl_poly.intercept_)
                q, p95, _ = check(poly_mdl)
                if q < best_q:
                    best_q, best_p95, best_model = q, p95, poly_mdl
            except:
                pass

            if best_q < EARLY_EXIT_THRESHOLD:
                models[i] = best_model
                continue

            # 4. Try Isotonic Regression (Guaranteed Monotonicity for Sparse/Normal)
            try:
                iso_mdl = IsotonicRegression(out_of_bounds="clip").fit(
                    X_train.flatten(), y_train
                )
                q, p95, _ = check(iso_mdl)
                if q < best_q:
                    best_q, best_p95, best_model = q, p95, iso_mdl
            except:
                pass

            # 5. Complex Fallback (Fourier MLP)
            if best_q > COMPLEX_MODEL_THRESHOLD:
                try:
                    mapper = FourierFeatureMapper(num_bands=32, max_freq=1000.0)
                    X_f_train = mapper.transform(X_train)
                    mdl_f_mlp = MLPRegressor(
                        hidden_layer_sizes=(64, 32), max_iter=500
                    ).fit(X_f_train, y_train)
                    mdl_fourier = FourierModelWrapper(mdl_f_mlp, mapper)
                    q, p95, _ = check(mdl_fourier)
                    if q < best_q:
                        best_model = mdl_fourier
                except:
                    pass

            models[i] = best_model

        return models, time.perf_counter() - t0

    def _predict_local_cdf(self, model, x_norm) -> float:
        return float(self._predict_local_cdf_vec(model, np.array([x_norm]))[0])

    def _get_bucket_overlap_count(self, b_idx: int, q_lo: int, q_hi: int) -> float:
        b = self.buckets[b_idx]
        lo = max(q_lo, b.lo)
        hi = min(q_hi, b.hi)
        if lo > hi:
            return 0.0
        w = b.hi - b.lo + 1
        cdf_hi = self._predict_local_cdf(self.models.get(b_idx), (hi - b.lo) / w)
        prev = lo - 1
        cdf_lo = (
            0.0
            if prev < b.lo
            else self._predict_local_cdf(self.models.get(b_idx), (prev - b.lo) / w)
        )

        model_pred = max(0.0, cdf_hi - cdf_lo) * b.count

        if b.count > 0 and w > 0:
            uniform_pred = ((hi - lo + 1) / w) * b.count
            return max(model_pred, uniform_pred * 0.05)

        return model_pred

    def report_models(self, file_path=None):
        """Prints a summary of which models were chosen for each bucket."""
        output = []
        output.append("\n" + "=" * 80)
        output.append(
            f"{'Idx':<4} | {'Range':<25} | {'Count':<10} | {'Model Type':<15} | {'Stats'}"
        )
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
            output.append(
                f"{i:<4} | {range_str:<25} | {b.count:<10} | {m_name:<15} | {info}"
            )
        output.append("=" * 80 + "\n")

        report_text = "\n".join(output)
        print(report_text)

        if file_path:
            with open(file_path, "w") as f:
                f.write(report_text)
            print(f"Model selection report saved to {file_path}")
