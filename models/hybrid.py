import numpy as np
import time
from typing import List, Dict, Tuple, Any
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.tree import DecisionTreeRegressor
from .common import Bucket, RangeQuery, CDFTrainRow
import concurrent.futures


class FourierFeatureMapper:
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
        self, queries, y_true_counts, y_pred_counts, N, error_threshold=1.2, alpha=0.15
    ):
        """
        High-speed Dynamic Bucket Scaling (DBS).
        Adjusts the total row count (mass) of buckets based on execution feedback.
        """
        # 1. Identify "Bad" Query Indices
        q_errs = np.maximum(
            y_true_counts / (y_pred_counts + 1e-9),
            y_pred_counts / (y_true_counts + 1e-9),
        )
        bad_q_mask = q_errs > error_threshold

        if not np.any(bad_q_mask):
            return 0

        bad_indices = np.where(bad_q_mask)[0]
        b_lo_vals = np.array([b.lo for b in self.buckets])
        b_hi_vals = np.array([b.hi for b in self.buckets])

        # Accumulate the true and predicted mass allocated to each bucket
        bucket_true_mass = {}
        bucket_pred_mass = {}

        for idx in bad_indices:
            q = queries[idx]
            y_true = y_true_counts[idx]
            y_pred = y_pred_counts[idx]

            s_idx = np.searchsorted(b_hi_vals, q.low)
            e_idx = np.searchsorted(b_lo_vals, q.high, side="right")

            # Determine spatial overlap
            overlaps = []
            for i in range(s_idx, e_idx):
                b = self.buckets[i]
                overlap_w = min(q.high, b.hi) - max(q.low, b.lo) + 1
                if overlap_w > 0:
                    overlaps.append((i, overlap_w))

            total_overlap = sum(w for _, w in overlaps)
            if total_overlap == 0:
                continue

            # Distribute the feedback proportionally
            for i, w in overlaps:
                share = w / total_overlap

                if i not in bucket_true_mass:
                    bucket_true_mass[i] = 0.0
                    bucket_pred_mass[i] = 0.0

                bucket_true_mass[i] += y_true * share
                bucket_pred_mass[i] += y_pred * share

        # Apply EMA updates to bucket counts
        is_dirty = False
        for i in bucket_true_mass.keys():
            true_m = bucket_true_mass[i]
            pred_m = bucket_pred_mass[i]

            if pred_m < 1e-5:
                pred_m = 1.0  # Prevent division by zero

            ratio = true_m / pred_m
            ratio = np.clip(ratio, 0.2, 5.0)

            b = self.buckets[i]

            # --- FIX: ZERO-COUNT REVIVAL ---
            if b.count < 1.0 and true_m > 0:
                # Если бакет был "мертв" (0 строк в сэмпле), но реальные данные есть
                new_count = true_m * alpha
            else:
                # Стандартное EMA обновление
                new_count = b.count * (1.0 - alpha) + (b.count * ratio) * alpha

            b.count = max(0.0, new_count)
            is_dirty = True

        if is_dirty:
            self._bake_vectorized_data()
            
        return len(bucket_true_mass)

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
        if self.b_lo is None or len(self.b_lo) != len(self.buckets):
            self._bake_vectorized_data()

        if len(self.buckets) == 0:
            return 0.0

        n_buckets = len(self.buckets)
        bw = self.b_width[0]
        mn = self.b_lo[0]

        if bw <= 0:
            bw = 1.0

        start_idx = int((q.low - mn) // bw)
        start_idx = max(0, min(start_idx, n_buckets - 1))
        
        end_idx = int((q.high - mn) // bw)
        end_idx = max(0, min(end_idx, n_buckets - 1))

        total = 0.0
        for i in range(start_idx, end_idx + 1):
            if self.b_count[i] == 0:
                continue
                
            # As requested: if a query spans buckets 1, 2, 3, 4, 5, 
            # buckets 2, 3, and 4 will be fully enclosed (q.low <= b_lo and q.high >= b_hi).
            # For these, we skip the model and just sum the counts like usual equiwidth!
            # The edge buckets (1 and 5) will fail this check and go to the 'else' to use the model.
            if q.low <= self.b_lo[i] and q.high >= self.b_hi[i]:
                total += self.b_count[i]
            else:
                total += self._get_bucket_overlap_count(i, q.low, q.high)
            
        return total
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
                elif type(model).__name__ == "DecisionTreeRegressor":
                    self.mod_types[i] = 8
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

        # --- FIX: Безопасное получение кумулятивной суммы ---
        def get_ps_safe(val):
            idx = val - mn
            if idx < 0:
                return 0.0
            if idx >= len(ps):
                return float(ps[-1])
            return float(ps[idx])

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

            n_grid = points_per_bucket * 2
            if width <= n_grid:
                xs_grid = np.arange(b.lo, b.hi + 1)
            else:
                xs_grid = np.linspace(b.lo, b.hi, num=n_grid, dtype=int)

            n_unif = points_per_bucket
            xs_unif = rng.integers(b.lo, b.hi + 1, size=n_unif)
            xs_edges = np.array([b.lo, b.hi, b.lo + 1, max(b.lo, b.hi - 1)])

            xs_train = np.unique(np.concatenate([xs_grid, xs_unif, xs_edges]))
            xs_train = np.sort(xs_train)

            xs_val = np.unique(rng.integers(b.lo, b.hi + 1, size=points_per_bucket * 2))

            base_cnt = get_ps_safe(b.lo - 1)

            def build_rows(xs_arr):
                res = []
                for x in xs_arr:
                    y_cdf = (get_ps_safe(x) - base_cnt) / b.count
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
        if model is None:
            base_pred = np.clip(x_norm_arr, 0.0, 1.0)
        elif isinstance(model, tuple):
            m_type = model[0]
            if m_type == "linear":
                base_pred = x_norm_arr * model[1] + model[2]
            elif m_type == "poly3":
                p, inter = model[1], model[2]
                base_pred = (
                    inter
                    + p[0] * x_norm_arr
                    + p[1] * (x_norm_arr**2)
                    + p[2] * (x_norm_arr**3)
                )
            elif m_type == "power":
                base_pred = model[1] * np.sqrt(x_norm_arr) + model[2]
            elif m_type == "poly":
                p, inter = model[1], model[2]
                base_pred = inter + p[0] * x_norm_arr + p[1] * (x_norm_arr**2)
            elif m_type == "log_linear":
                base_pred = model[1] * np.log(x_norm_arr + 1e-7) + model[2]
            else:
                base_pred = x_norm_arr
        elif type(model).__name__ == "IsotonicRegression":
            base_pred = model.predict(x_norm_arr)
        elif type(model).__name__ == "DecisionTreeRegressor":
            base_pred = model.predict(x_norm_arr.reshape(-1, 1))
        else:
            base_pred = model.predict(x_norm_arr.reshape(-1, 1)).flatten()

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

        EARLY_EXIT_THRESHOLD = 1.5
        COMPLEX_MODEL_THRESHOLD = 3

        def train_worker(i):
            train_rows, val_rows = rows_data[i]
            bucket = self.buckets[i]
            if not train_rows or bucket.count == 0:
                return i, None

            X_train = np.array([r.x_norm for r in train_rows]).reshape(-1, 1)
            y_train = np.array([r.y_cdf for r in train_rows])

            width = bucket.hi - bucket.lo + 1

            # --- FIX: Железобетонное извлечение частот без Out of Bounds ---
            v_arr = np.arange(bucket.lo, bucket.hi + 1)
            idx_arr = v_arr - mn
            valid_mask = (idx_arr >= 0) & (idx_arr < len(freq))

            local_freq = np.zeros(width, dtype=np.float64)
            if np.any(valid_mask):
                local_freq[valid_mask] = freq[idx_arr[valid_mask]]

            b_ps = np.cumsum(local_freq)
            local_probs = local_freq / (local_freq.sum() + 1e-9)
            # -------------------------------------------------------------

            x_indices = np.clip(
                np.array([r.x_norm * width for r in train_rows]).astype(int),
                0,
                width - 1,
            )
            weights = 1.0 + (local_probs[x_indices] / (local_probs.max() + 1e-9)) * 5.0

            local_rng = np.random.default_rng(rng.integers(0, 9999999) + i)
            val_queries = self._generate_bucket_queries(
                bucket, n_queries=500, rng=local_rng, freq=freq, mn=mn
            )
            q_lo = np.array([q.low for q in val_queries])
            q_hi = np.array([q.high for q in val_queries])

            def check(mdl):
                q_m, q_95 = self._eval_model_q_error_vec(
                    mdl, q_lo, q_hi, bucket, b_ps, bucket.lo, width
                )
                return q_m, q_95, mdl

            best_q, best_p95, best_model = check(None)
            if best_q < EARLY_EXIT_THRESHOLD:
                return i, best_model

            candidates = []
            try:
                m_lin = Ridge(alpha=1.0).fit(X_train, y_train, sample_weight=weights)
                lin_mdl = ("linear", m_lin.coef_[0], m_lin.intercept_)
                q, p95, _ = check(lin_mdl)
                candidates.append((q, p95, lin_mdl))

                if q < 1.05:
                    return i, lin_mdl

                m_pow = Ridge(alpha=1.0).fit(
                    np.sqrt(X_train), y_train, sample_weight=weights
                )
                pow_mdl = ("power", m_pow.coef_[0], m_pow.intercept_)
                q, p95, _ = check(pow_mdl)
                candidates.append((q, p95, pow_mdl))

                m_log = Ridge(alpha=1.0).fit(
                    np.log(X_train + 1e-7), y_train, sample_weight=weights
                )
                log_mdl = ("log_linear", m_log.coef_[0], m_log.intercept_)
                q, p95, _ = check(log_mdl)
                candidates.append((q, p95, log_mdl))
            except:
                pass

            if candidates:
                q, p95, mdl = min(candidates, key=lambda x: x[0])
                if q < best_q:
                    best_q, best_p95, best_model = q, p95, mdl

            if best_q < EARLY_EXIT_THRESHOLD:
                return i, best_model

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
                return i, best_model

            try:
                dt_mdl = DecisionTreeRegressor(max_depth=4).fit(
                    X_train, y_train, sample_weight=weights
                )
                q, p95, _ = check(dt_mdl)
                if q < best_q:
                    best_q, best_p95, best_model = q, p95, dt_mdl
            except:
                pass

            try:
                iso_mdl = IsotonicRegression(out_of_bounds="clip").fit(
                    X_train.flatten(), y_train
                )
                q, p95, _ = check(iso_mdl)
                if q < best_q:
                    best_q, best_p95, best_model = q, p95, iso_mdl
            except:
                pass

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

            return i, best_model

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(train_worker, i) for i in rows_data.keys()]

            for future in concurrent.futures.as_completed(futures):
                i, best_model = future.result()
                models[i] = best_model

        return models, time.perf_counter() - t0

    def _predict_local_cdf(self, model, x_norm) -> float:
        if model is None:
            pred = x_norm
        elif isinstance(model, tuple):
            m_type = model[0]
            if m_type == "linear":
                pred = x_norm * model[1] + model[2]
            elif m_type == "poly3":
                p, inter = model[1], model[2]
                pred = inter + p[0] * x_norm + p[1] * (x_norm**2) + p[2] * (x_norm**3)
            elif m_type == "power":
                pred = model[1] * (x_norm**0.5) + model[2]
            elif m_type == "poly":
                p, inter = model[1], model[2]
                pred = inter + p[0] * x_norm + p[1] * (x_norm**2)
            elif m_type == "log_linear":
                import math
                pred = model[1] * math.log(x_norm + 1e-7) + model[2]
            else:
                pred = x_norm
        else:
            return float(self._predict_local_cdf_vec(model, np.array([x_norm]))[0])
            
        if pred < 0.0: return 0.0
        if pred > 1.0: return 1.0
        return pred

    def _get_bucket_overlap_count(self, b_idx: int, q_lo: int, q_hi: int) -> float:
        b = self.buckets[b_idx]
        lo = max(q_lo, b.lo)
        hi = min(q_hi, b.hi)
        if lo > hi:
            return 0.0
        w = b.hi - b.lo + 1
        
        if hi == b.hi:
            cdf_hi = 1.0
        else:
            cdf_hi = self._predict_local_cdf(self.models.get(b_idx), (hi - b.lo) / w)
            
        prev = lo - 1
        if prev < b.lo:
            cdf_lo = 0.0
        else:
            cdf_lo = self._predict_local_cdf(self.models.get(b_idx), (prev - b.lo) / w)

        model_pred = max(0.0, cdf_hi - cdf_lo) * b.count

        if b.count > 0 and w > 0:
            uniform_pred = ((hi - lo + 1) / w) * b.count
            return max(model_pred, uniform_pred * 0.05)

        return model_pred

    def report_models(self, file_path=None):
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
                elif m_type == "power":
                    m_name = "Power"
                    info = f"coef={model[1]:.4f}, int={model[2]:.4f}"
                elif m_type == "poly3":
                    m_name = "Poly Cubic"
                    info = "degree=3"
            else:
                m_type = type(model).__name__
                if m_type == "IsotonicRegression":
                    m_name = "Isotonic"
                    info = "Step Function"
                elif m_type == "DecisionTreeRegressor":
                    m_name = "Decision Tree"
                    info = f"depth={model.get_depth()}"
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
