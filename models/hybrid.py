import bisect
import numpy as np
import time
from typing import List, Dict, Tuple, Any
from sklearn.linear_model import Ridge
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
    # Canonical Hybrid FD CDF of the thesis; bump only when the method itself changes
    # (instrumentation that records what the waterfall did does not change the method).
    SPEC_VERSION = "hybrid-fd-cdf-canonical"
    UNIFORM_THRESHOLD = 1.3
    MODEL_THRESHOLD = 1.05
    # Waterfall tier of each local CDF (model_name); legacy Poly2/Poly3 load as Tier 1
    TIER = {"Uniform": 0, "Linear": 1, "Power": 1, "Log-Linear": 1, "Poly2": 1, "Poly3": 1,
            "Isotonic": 2, "DecisionTree": 2, "FourierMLP": 3}

    def __init__(
        self,
        buckets: List[Bucket],
        models: Dict[int, Any] = None,
        identity_threshold: float = 1e-4,
        mlp_penalty: float = 1.5,
        fourier_penalty: float = 1.5,
        min_train_count: float = 0.0,
        parametric_only: bool = False,
    ):
        self.buckets = buckets
        self.models = models if models is not None else {}
        # Бакеты с оценкой меньше min_train_count строк остаются Uniform:
        # в них слишком мало точек выборки, модель выучила бы шум сэмпла
        self.min_train_count = min_train_count
        # Только модели с 2-4 параметрами (linear/power/log/poly3): без дерева,
        # изотонической и MLP, которые хранят сотни чисел на бакет
        self.parametric_only = parametric_only
        self.uniform_grid = True
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
        # Instrumentation only (does not influence any estimate): what the waterfall did in
        # each bucket the last time it was trained, and which buckets the last FT touched.
        #   selection[i] = {tier, model, val_median_qerror, val_p95_qerror, passed_threshold,
        #                   threshold, candidates_checked, selection_reason}
        self.selection: Dict[int, Dict[str, Any]] = {}
        self.last_ft_idx = np.array([], dtype=np.int64)

    def feedback_update(
        self, queries, y_true_counts, y_pred_counts, N=None, error_threshold=1.2, alpha=0.15
    ):
        """
        FT: подстройка счётчиков бакетов по фидбэку (выборку не читает, модели не переобучает).
        Запросы с q-error > error_threshold раздают свою истинную и предсказанную массу
        бакетам пропорционально перекрытию по ширине; счётчик бакета сдвигается к
        count * (истина / оценка) с шагом alpha (отношение обрезано до [0.2, 5]).
        Бакет, пустой в сэмпле, получает alpha * плотность строк в запросах * ширину бакета.
        Векторизовано: бакеты между крайними покрыты запросом целиком (сетка без щелей),
        поэтому их вклад копится разностным массивом за O(запросов + бакетов).
        Возвращает число затронутых бакетов.
        """
        if self.b_lo is None or len(self.b_lo) != len(self.buckets):
            self._bake_vectorized_data()
        n = len(self.buckets)
        y_true = np.asarray(y_true_counts, dtype=np.float64)
        y_pred = np.asarray(y_pred_counts, dtype=np.float64)
        q_errs = np.maximum(y_true / (y_pred + 1e-9), y_pred / (y_true + 1e-9))
        bad = np.nonzero(q_errs > error_threshold)[0]
        self.last_ft_idx = np.array([], dtype=np.int64)
        if n == 0 or len(bad) == 0:
            return 0

        lo = np.fromiter((queries[j].low for j in bad), np.float64, len(bad))
        hi = np.fromiter((queries[j].high for j in bad), np.float64, len(bad))
        s = np.searchsorted(self.b_hi, lo)                    # первый бакет с hi >= q.low
        e = np.searchsorted(self.b_lo, hi, side="right") - 1  # последний бакет с lo <= q.high
        ok = s <= e
        s, e, lo, hi, yt, yp = s[ok], e[ok], lo[ok], hi[ok], y_true[bad][ok], y_pred[bad][ok]
        if len(s) == 0:
            return 0
        lo_c = np.maximum(lo, self.b_lo[s])
        hi_c = np.minimum(hi, self.b_hi[e])
        total = hi_c - lo_c + 1  # покрытая запросом ширина сетки
        per_w = np.stack([yt / total, yp / total, np.ones_like(yt)])  # истина, оценка, ширина — на единицу ширины

        acc = np.zeros((3, n))  # [истинная масса, предсказанная масса, покрытая ширина]
        hits = np.zeros(n)
        # крайние бакеты: доля перекрытия
        ow_s = np.minimum(hi, self.b_hi[s]) - lo_c + 1
        for k in range(3):
            np.add.at(acc[k], s, per_w[k] * ow_s)
        np.add.at(hits, s, 1)
        m = e > s
        ow_e = hi_c[m] - self.b_lo[e[m]] + 1
        for k in range(3):
            np.add.at(acc[k], e[m], per_w[k][m] * ow_e)
        np.add.at(hits, e[m], 1)
        # бакеты строго между ними: покрыты целиком
        m = e > s + 1
        diff = np.zeros((4, n + 1))
        for k, w in enumerate([*per_w[:, m], np.ones(m.sum())]):
            np.add.at(diff[k], s[m] + 1, w)
            np.add.at(diff[k], e[m], -w)
        inner = np.cumsum(diff, axis=1)[:, :n]
        acc += inner[:3] * self.b_width
        hits += np.rint(inner[3])

        idx = np.nonzero(hits > 0)[0]
        true_m, pred_m, covered = acc[0, idx], acc[1, idx], acc[2, idx]
        pred_m = np.where(pred_m < 1e-5, 1.0, pred_m)
        ratio = np.clip(true_m / pred_m, 0.2, 5.0)
        count = np.array([self.buckets[i].count for i in idx], dtype=np.float64)
        # Пустой бакет: отношение масс не определено. Раньше брали сумму true * share по всем
        # запросам — сотни перекрывающихся запросов завышали массу в десятки раз (медиана 37x)
        revive = (count < 1.0) & (true_m > 0)
        new = np.where(
            revive,
            alpha * true_m / np.maximum(covered, 1e-12) * self.b_width[idx],
            count * (1.0 - alpha) + count * ratio * alpha,
        )
        for i, c in zip(idx, np.maximum(new, 0.0)):
            self.buckets[i].count = float(c)
        self._bake_vectorized_data()
        self.last_ft_idx = idx.astype(np.int64)
        return len(idx)

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
        bw, mn = self._grid

        if self.uniform_grid:
            start_idx = int((q.low - mn) // bw)
            end_idx = int((q.high - mn) // bw)
        else:
            # Бакеты разной ширины (например, equi-depth) — бинарный поиск
            start_idx = int(np.searchsorted(self.b_hi, q.low, side="left"))
            end_idx = int(np.searchsorted(self.b_lo, q.high, side="right")) - 1
        start_idx = max(0, min(start_idx, n_buckets - 1))
        end_idx = max(0, min(end_idx, n_buckets - 1))

        # Бакеты между крайними покрыты целиком: их счётчики — разность префиксных сумм,
        # как в EquiWidth. Локальная CDF нужна только двум крайним (частично покрытым) бакетам.
        if start_idx == end_idx:
            return self._get_bucket_overlap_count(start_idx, q.low, q.high)
        return (
            self._get_bucket_overlap_count(start_idx, q.low, q.high)
            + self._get_bucket_overlap_count(end_idx, q.low, q.high)
            + max(0.0, self._prefix[end_idx] - self._prefix[start_idx + 1])
        )

    def _bake_vectorized_data(self):
        n = len(self.buckets)
        self.b_lo = np.array([b.lo for b in self.buckets], dtype=np.float64)
        self.b_hi = np.array([b.hi for b in self.buckets], dtype=np.float64)
        self.b_count = np.array([b.count for b in self.buckets], dtype=np.float64)
        self.b_width = self.b_hi - self.b_lo + 1
        self._prefix = [0.0, *np.cumsum(self.b_count).tolist()]
        self._grid = (max(1.0, float(self.b_width[0])), float(self.b_lo[0])) if n else (1.0, 0.0)
        # Последний бакет equi-width может быть уже остальных — его не проверяем
        self.uniform_grid = bool(n <= 1 or np.all(self.b_width[:-1] == self.b_width[0]))

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
                    self._lookup(model)  # таблица для predict строится здесь, а не на первом запросе
                elif type(model).__name__ == "DecisionTreeRegressor":
                    self.mod_types[i] = 8
                    self._lookup(model)
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
        """
        Waterfall Model Selection (Hybrid FD CDF): модели проверяются от простых к сложным
        на 500 валидационных запросах внутри бакета, берётся ПЕРВАЯ, прошедшая порог
        медианной q-error:
          Tier 0  Uniform                                   медиана <= 1.3
          Tier 1  Linear, Power (x^0.5), Log-linear —
                  Ridge с весами по плотности               медиана <= 1.05
          Tier 2  Isotonic, DecisionTree (max_depth=4)      медиана <= 1.05
          Tier 3  Fourier-feature MLP (32 частоты, (64, 32)) медиана <= 1.05
        Если порог не прошла ни одна, остаётся самая точная из проверенных.
        """
        models = {}
        selection = {}  # instrumentation: filled by the workers, one distinct key each
        t0 = time.perf_counter()

        UNIFORM_THRESHOLD = self.UNIFORM_THRESHOLD  # 1.3
        MODEL_THRESHOLD = self.MODEL_THRESHOLD  # 1.05

        def record(i, mdl, q_m, q_95, passed, threshold, n_checked, reason):
            name = self.model_name(mdl)
            selection[i] = {
                "tier": self.TIER.get(name), "model": name,
                "val_median_qerror": None if q_m is None else float(q_m),
                "val_p95_qerror": None if q_95 is None else float(q_95),
                "passed_threshold": passed, "threshold": threshold,
                "candidates_checked": n_checked, "selection_reason": reason,
            }
            return i, mdl

        # rng is shared between the workers below, and they run in threads.
        # Drawing from it inside a worker makes every bucket's seed depend on
        # thread scheduling order, so the same input produced different models
        # run to run. Draw once here instead: each bucket still gets its own
        # seed (base_seed + i), but now a fixed one.
        base_seed = int(rng.integers(0, 9999999))

        def train_worker(i):
            train_rows, val_rows = rows_data[i]
            bucket = self.buckets[i]
            if not train_rows or bucket.count == 0:
                return record(i, None, None, None, None, None, 0, "empty_in_sample")
            if bucket.count < self.min_train_count:
                return record(i, None, None, None, None, None, 0, "below_min_train_count")

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

            local_rng = np.random.default_rng(base_seed + i)
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

            # Tier 0 — Uniform
            best_q, best_p95, best_model = check(None)
            n_checked = 1
            if best_q <= UNIFORM_THRESHOLD:
                return record(i, None, best_q, best_p95, True, UNIFORM_THRESHOLD, n_checked, "waterfall")
            last = [None, None]

            def passes(mdl):
                """Порог Tier 1-3; заодно запоминаем самую точную модель на случай, если не пройдёт никто."""
                nonlocal best_q, best_p95, best_model, n_checked
                q, p95, _ = check(mdl)
                n_checked += 1
                last[:] = [q, p95]
                if q < best_q:
                    best_q, best_p95, best_model = q, p95, mdl
                return q <= MODEL_THRESHOLD

            def fallback():
                """Порог не прошёл никто: самая точная из проверенных (instrumented return)."""
                thr = UNIFORM_THRESHOLD if best_model is None else MODEL_THRESHOLD
                return record(i, best_model, best_q, best_p95, False, thr, n_checked, "fallback_most_accurate")

            # Tier 1 — параметрическая регрессия (Ridge с весами по плотности)
            for kind, transform in (
                ("linear", lambda X: X),
                ("power", np.sqrt),
                ("log_linear", lambda X: np.log(X + 1e-7)),
            ):
                try:
                    m = Ridge(alpha=1.0).fit(transform(X_train), y_train, sample_weight=weights)
                except Exception:
                    continue
                mdl = (kind, m.coef_[0], m.intercept_)
                if passes(mdl):
                    return record(i, mdl, *last, True, MODEL_THRESHOLD, n_checked, "waterfall")

            if self.parametric_only:
                return fallback()

            # Tier 2 — непараметрические: ступени и пустые промежутки
            def fit_isotonic():
                return IsotonicRegression(out_of_bounds="clip").fit(X_train.flatten(), y_train)

            def fit_tree():
                return DecisionTreeRegressor(max_depth=4, random_state=base_seed + i).fit(
                    X_train, y_train, sample_weight=weights
                )

            # Tier 3 — Fourier-feature MLP: 32 частоты -> 2 + 2*32 = 66 признаков
            def fit_fourier():
                mapper = FourierFeatureMapper(num_bands=32, max_freq=1000.0)
                mlp = MLPRegressor(
                    hidden_layer_sizes=(64, 32), max_iter=500, random_state=base_seed + i
                ).fit(mapper.transform(X_train), y_train)
                return FourierModelWrapper(mlp, mapper)

            for fit in (fit_isotonic, fit_tree, fit_fourier):
                try:
                    mdl = fit()
                except Exception:
                    continue
                if passes(mdl):
                    return record(i, mdl, *last, True, MODEL_THRESHOLD, n_checked, "waterfall")

            return fallback()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(train_worker, i) for i in rows_data.keys()]

            for future in concurrent.futures.as_completed(futures):
                i, best_model = future.result()
                models[i] = best_model

        self.selection.update(selection)
        return models, time.perf_counter() - t0

    @staticmethod
    def _lookup(model):
        """
        Isotonic / дерево как списки чисел, чтобы predict не звал sklearn на каждый запрос
        (~80 мкс на вызов против ~1 мкс). Предсказания те же, что у model.predict:
          Isotonic (out_of_bounds="clip") — линейная интерполяция по (X_thresholds_, y_thresholds_),
            за краями — крайние значения;
          дерево по одному признаку — ступенчатая функция: отсортированные пороги t_1 < ... < t_m
            делят ось на m + 1 интервалов (t_{j-1}, t_j], у каждого — значение своего листа.
        Считается один раз на модель и хранится в ней.
        """
        lk = getattr(model, "_hybrid_lookup", None)
        if lk is not None:
            return lk
        if type(model).__name__ == "IsotonicRegression":
            lk = ("iso", model.X_thresholds_.tolist(), model.y_thresholds_.tolist())
        else:
            t = model.tree_
            ts = np.unique(t.threshold[t.children_left != -1])
            # точка float32 внутри каждого интервала: наибольшая <= t_j и наименьшая > t_m
            reps = []
            for th in ts:
                r = np.float32(th)
                reps.append(r if r <= th else np.nextafter(r, np.float32(-np.inf)))
            if len(ts):
                r = np.float32(ts[-1])
                reps.append(r if r > ts[-1] else np.nextafter(r, np.float32(np.inf)))
            else:
                reps.append(np.float32(0.0))
            values = model.predict(np.array(reps, dtype=np.float64).reshape(-1, 1))
            lk = ("tree", ts.tolist(), values.tolist())
        model._hybrid_lookup = lk
        return lk

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
        elif type(model).__name__ in ("IsotonicRegression", "DecisionTreeRegressor"):
            kind, xs, ys = self._lookup(model)
            if kind == "tree":
                # sklearn сравнивает x во float32: x <= порог -> левая ветка
                pred = ys[bisect.bisect_left(xs, float(np.float32(x_norm)))]
            elif x_norm <= xs[0]:
                pred = ys[0]
            elif x_norm >= xs[-1]:
                pred = ys[-1]
            else:
                j = bisect.bisect_right(xs, x_norm)
                pred = ys[j - 1] + (x_norm - xs[j - 1]) * (ys[j] - ys[j - 1]) / (xs[j] - xs[j - 1])
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

    @staticmethod
    def _model_param_count(model) -> int:
        if model is None:
            return 0
        if isinstance(model, tuple):
            return 4 if model[0] == "poly3" else 2
        name = type(model).__name__
        if name == "IsotonicRegression":
            return 2 * len(model.X_thresholds_)
        if name == "DecisionTreeRegressor":
            return 3 * model.tree_.node_count  # порог, значение, ссылка на ребёнка
        mlp = model.model  # FourierModelWrapper
        return int(sum(c.size for c in mlp.coefs_) + sum(b.size for b in mlp.intercepts_))

    def param_count(self) -> int:
        """
        Сколько чисел хранит модель: счётчик на бакет (+ граница, если бакеты
        разной ширины; equi-width сетку задают mn и ширина) + параметры моделей.
        """
        if self.b_lo is None:
            self._bake_vectorized_data()
        per_bucket = 1 if self.uniform_grid else 2
        return len(self.buckets) * per_bucket + sum(
            self._model_param_count(m) for m in self.models.values()
        )

    def n_values(self) -> int:
        """
        Вес модели в числах (8 байт каждое): сетка (min и ширина бакета), счётчики и
        параметры локальных CDF (param_count) + код типа модели у каждого бакета с моделью.
        """
        n_models = sum(1 for m in self.models.values() if m is not None)
        return 2 + self.param_count() + n_models

    @staticmethod
    def model_name(model) -> str:
        """Короткое имя локальной CDF бакета (как в отчётах о выборе моделей)."""
        if model is None:
            return "Uniform"
        if isinstance(model, tuple):
            return {"linear": "Linear", "power": "Power", "log_linear": "Log-Linear",
                    "poly": "Poly2", "poly3": "Poly3"}.get(model[0], model[0])
        return {"IsotonicRegression": "Isotonic",
                "DecisionTreeRegressor": "DecisionTree"}.get(type(model).__name__, "FourierMLP")

    def cdf_choice(self) -> Dict[str, int]:
        """Сколько бакетов получили каждую локальную CDF; бакеты без строк — 'Empty'."""
        counts: Dict[str, int] = {}
        for i, b in enumerate(self.buckets):
            name = "Empty" if b.count <= 0 else self.model_name(self.models.get(i))
            counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    GLOBAL_VALUES = 2  # grid: min and bucket width, shared by all buckets (counted once)

    def bucket_table(self) -> List[Dict[str, Any]]:
        """
        One row per bucket: boundaries, count, chosen local CDF, its tier and validation error
        from the last waterfall run, and the bucket's share of n_values().
        bucket_values = count (+ lo boundary if the grid is not equal-width) + CDF parameters
        (+ 1 model-type code if the bucket has a model), so that
        GLOBAL_VALUES + sum(bucket_values) == n_values() — shared numbers are not repeated per bucket.
        """
        if self.b_lo is None or len(self.b_lo) != len(self.buckets):
            self._bake_vectorized_data()
        per_bucket = 1 if self.uniform_grid else 2
        rows = []
        for i, b in enumerate(self.buckets):
            mdl = self.models.get(i)
            sel = self.selection.get(i, {})
            empty = b.count <= 0
            name = "Empty" if empty else self.model_name(mdl)
            reason = sel.get("selection_reason", "not_trained")
            if not empty and mdl is None and reason in ("empty_in_sample", "not_trained"):
                # count > 0 now (FT revival / loaded model) but the waterfall never ran on it
                reason = "uniform_default_untrained"
            params = self._model_param_count(mdl)
            values = per_bucket + params + (0 if mdl is None else 1)
            rows.append({
                "bucket": i, "lo": int(b.lo), "hi": int(b.hi), "width": int(b.hi - b.lo + 1),
                "count": float(b.count), "cdf_model": name,
                "tier": None if empty else self.TIER.get(self.model_name(mdl)),
                "model_params": params, "bucket_values": values, "bucket_bytes": 8 * values,
                "val_median_qerror": sel.get("val_median_qerror"),
                "val_p95_qerror": sel.get("val_p95_qerror"),
                "passed_threshold": sel.get("passed_threshold"),
                "threshold": sel.get("threshold"),
                "candidates_checked": sel.get("candidates_checked"),
                "selection_reason": reason,
            })
        return rows

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
