import bisect
import hashlib
import heapq
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .common import RangeQuery

ME, MSE, MB = "ME", "MSE", "MB"
PAPER_ALG1 = "paper_alg1"
WEIGHTED_ABLATION = "weighted_ablation"
EQ_ESTIMATORS = (
    "paper_uniform",  # Eq. 1: T_b / d_b — дефолт экспериментов статьи (§5.1)
    "paper_entropy",  # Eq. 6: T_b * 2^-H_b
    "paper_bias_factor",  # (T_b / d_b) * (1 + BF_b) — алгебраически == paper_entropy
    "weighted_frequency_mean",  # §4.5: sum f^2 / sum f
)
RANGE_ESTIMATORS = (
    "paper_linear",  # §2.2: непрерывная интерполяция по [min_b, max_b]
    "discrete_linear",  # [EXTENSION] целочисленные ячейки: (overlap + 1) / (width + 1)
)

CONSTRUCTION_VERSION = 1
# Допуск сравнения скоров: скор нормируется на полную массу и округляется до
# 12 знаков -> равные с точностью до ~1e-12 кандидаты считаются ничьей
TIE_DECIMALS = 12


@dataclass(frozen=True)
class HistogramConfig:
    """Параметры построения и оценки (спецификация §9). Дефолты — baseline статьи."""

    method: str = ME  # ME | MSE | MB
    n_buckets: int = 30  # B_req (в статье §5.2: B = 30)
    me_input: str = "area"  # ME(A) по area | ME(F) по частотам — только для ME
    me_objective: str = PAPER_ALG1  # weighted_ablation — [EXTENSION], отдельная абляция
    equality_estimator: str = "paper_uniform"
    range_estimator: str = "paper_linear"

    def __post_init__(self):
        if self.method not in (ME, MSE, MB):
            raise ValueError(f"Unknown method: {self.method}")
        if self.n_buckets < 1:
            raise ValueError("n_buckets must be >= 1")
        if self.me_input not in ("area", "frequency"):
            raise ValueError(f"Unknown me_input: {self.me_input}")
        if self.me_objective not in (PAPER_ALG1, WEIGHTED_ABLATION):
            raise ValueError(f"Unknown me_objective: {self.me_objective}")
        if self.method != ME and (self.me_input != "area" or self.me_objective != PAPER_ALG1):
            raise ValueError("me_input / me_objective apply only to method='ME'")
        if self.equality_estimator not in EQ_ESTIMATORS:
            raise ValueError(f"Unknown equality_estimator: {self.equality_estimator}")
        if self.range_estimator not in RANGE_ESTIMATORS:
            raise ValueError(f"Unknown range_estimator: {self.range_estimator}")

    @property
    def variant(self) -> str:
        """Короткое имя построителя: MEA, MEF, MSE, MB (+ -W для взвешенной абляции)."""
        if self.method != ME:
            return self.method
        name = "MEA" if self.me_input == "area" else "MEF"
        return name + ("-W" if self.me_objective == WEIGHTED_ABLATION else "")


@dataclass(frozen=True, slots=True)
class EntropyBucket:
    """Бакет — непрерывный отрезок [start_idx, end_idx) отсортированных distinct-значений."""

    start_idx: int
    end_idx: int
    min_value: float
    max_value: float
    distinct_count: int
    tuple_count: int  # sum f — НЕ сумма area
    entropy_frequency: float  # H(f в бакете)
    entropy_area: float  # H(area в бакете), для ME(A)
    bias_factor: float  # d * 2^-H_f - 1
    sum_f2: float
    sum_f3: float
    sum_f_log2f: float


# ----------------------------------------------------------------------
# Математика (§2): энтропия и стоимость отрезка через префиксные моменты
# ----------------------------------------------------------------------


def entropy(w: Sequence[float]) -> float:
    """Прямая формула H(w) = -sum p log2 p (Eq. 2). Для тестов и оракула."""
    w = np.asarray(w, dtype=np.float64)
    p = w[w > 0] / w.sum()
    return max(0.0, float(-(p * np.log2(p)).sum()))


def area_weights(values: np.ndarray, freq: np.ndarray) -> np.ndarray:
    """
    Poosala et al. 1996, §2.1: spread_i = v_{i+1} - v_i, spread_last = 1,
    area_i = f_i * spread_i. Вес построения, а не число строк.
    """
    spread = np.ones(len(values), dtype=np.float64)
    if len(values) > 1:
        spread[:-1] = np.diff(values.astype(np.float64))
    return freq.astype(np.float64) * spread


class PrefixMoments:
    """
    Префиксные суммы P1, Plog (и P2, P3 для частот) — O(1) моменты отрезка [start, end).
    start, end могут быть numpy-массивами: так за один вызов оцениваются все разрезы бакета.
    """

    def __init__(self, w: np.ndarray, with_powers: bool = False):
        w = np.asarray(w, dtype=np.float64)
        self.P1 = _prefix(w)
        self.Plog = _prefix(w * np.log2(w))
        self.P2 = _prefix(w**2) if with_powers else None
        self.P3 = _prefix(w**3) if with_powers else None

    def total(self, start, end):
        return self.P1[end] - self.P1[start]

    def entropy(self, start, end):
        """H(start, end) = log2 W - (sum w log2 w) / W; одно значение -> 0."""
        W = self.P1[end] - self.P1[start]
        H = np.log2(W) - (self.Plog[end] - self.Plog[start]) / W
        # Отрицательная энтропия — только ошибка округления префиксов
        return np.where(np.asarray(end) - np.asarray(start) == 1, 0.0, np.maximum(H, 0.0))

    def mse_cost(self, start, end):
        """§4.3.1 Eq. 6: J = T * E, E = S3/T^3 - 2 s S2/T^2 + s^2, s = 2^-H."""
        T = self.total(start, end)
        s = np.exp2(-self.entropy(start, end))
        S2 = self.P2[end] - self.P2[start]
        S3 = self.P3[end] - self.P3[start]
        E = S3 / T**3 - 2.0 * s * S2 / T**2 + s**2
        return T * np.maximum(E, 0.0)

    def mb_cost(self, start, end):
        """§4.3.1 Eq. 8: J = T * BF, BF = d * 2^-H - 1."""
        d = np.asarray(end) - np.asarray(start)
        bf = d * np.exp2(-self.entropy(start, end)) - 1.0
        return self.total(start, end) * np.maximum(bf, 0.0)


def _prefix(x: np.ndarray) -> np.ndarray:
    out = np.zeros(len(x) + 1, dtype=np.float64)
    np.cumsum(x, out=out[1:])
    return out


def aggregate_distribution(values, counts=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Валидация и группировка (§1 [RESOLUTION]): отбрасывать нечего — NaN/inf и
    count <= 0 -> ValueError. Дубликаты значений суммируются, результат отсортирован.
    """
    values = np.asarray(values)
    counts = np.ones(len(values), dtype=np.int64) if counts is None else np.asarray(counts)
    if values.shape != counts.shape or values.ndim != 1:
        raise ValueError("values and counts must be 1-D arrays of equal length")
    if len(values) == 0:
        return values.astype(np.float64), counts.astype(np.int64)
    if not np.all(np.isfinite(values.astype(np.float64))):
        raise ValueError("values must be finite (no NaN / inf / null)")
    if np.any(counts <= 0) or not np.all(counts == np.floor(counts)):
        raise ValueError("counts must be positive integers")

    uniq, inv = np.unique(values, return_inverse=True)
    # Целочисленное суммирование: bincount(weights=...) идёт через float64
    freq = np.zeros(len(uniq), dtype=np.int64)
    np.add.at(freq, inv, counts.astype(np.int64))
    return uniq, freq


# ----------------------------------------------------------------------
# Гистограмма
# ----------------------------------------------------------------------


class EntropyHistogram1D:
    """
    1-D Entropy-based Histograms: H. To, K. Chiang, C. Shahabi,
    "Entropy-based Histograms for Selectivity Estimation", CIKM 2013,
    doi:10.1145/2505515.2505756.
    Clean-room реализация по entropy_histograms_1d_cleanroom_spec.md.

    Построение (§4.2 Algorithm 1, §4.3) — нисходящий жадный best-cut:
      один бакет [0, N) -> для каждого делимого бакета лучший локальный разрез,
      куча кандидатов, сплит глобально лучшего, пересчёт только двух детей;
      ровно B_effective - 1 разрезов, даже если выигрыш <= 0.

      ME (§4.2):  loss = H_parent - (H_left + H_right) — НЕ взвешенная сумма
                  (как в Algorithm 1); приоритет W_parent * loss -> минимум.
                  ME(A) — по area (Poosala 1996), ME(F) — по частотам.
      MSE (§4.3.2): выигрыш J(parent) - J(left) - J(right), J = T * E -> максимум.
      MB (§4.3.3):  то же с J = T * BF -> максимум.

    Оценка (§2.2, §4.3.1, §4.5) отделена от построения: Eq. 1 (T/d) — дефолт
    экспериментов статьи (§5.1) для ЛЮБОГО построителя; Eq. 6 / BF / weighted
    mean — отдельные абляции. Диапазон — линейная интерполяция по
    [min_value, max_value] бакета по реальным tuple_count (не по area).

    Масштаб: модель строится по выборке (в бенчмарке ей нельзя видеть freq),
    поэтому статистика бакетов хранится в единицах выборки (целые счётчики),
    а все оценки умножаются на row_scale = N / len(sample). Цели ME/MSE/MB
    линейны по масштабу частот, так что границы не зависят от row_scale.

    Решения, которые статья не фиксирует ([RESOLUTION]):
      * B_effective = min(B_req, N); пустое распределение -> пустая гистограмма;
      * энтропия отрезка — через префиксные моменты, а не через порядок
        обновлений Algorithm 1, строки 15–17 (там T_L/T_R меняются раньше Eq. 4/5);
      * MSE: родительский член взят как T_parent * E_parent (в §4.3.2 вес пропущен);
      * MB: веса детей T_left, T_right (в §4.3.3 дважды написан W_r);
      * ничьи: скор / полная масса, округление до 1e-12, затем больший вес
        родителя, затем меньший индекс разреза, затем меньший start_idx;
      * точечный запрос [v, v] -> оценка равенства (интерполяция дала бы 0);
        lo > hi -> 0; значение вне всех бакетов (в т.ч. в щели между ними) -> 0;
      * однозначный бакет (min == max) даёт T_b, если значение внутри запроса;
      * сумма по диапазону обрезается до [0, T], оценка равенства — до [0, T_b].

    [EXTENSION] (не baseline, отчитывать отдельно):
      * me_objective='weighted_ablation': W_p H_p - (W_L H_L + W_R H_R);
      * range_estimator='discrete_linear': доля целочисленных ячеек бакета.

    Обновлений и детектора дрейфа в статье нет (§6.1 спецификации): гистограмма
    заморожена, пересборка = новый build() по текущему снимку, его вызывает харнесс.
    """

    def __init__(
        self,
        config: HistogramConfig,
        buckets: List[EntropyBucket],
        split_indices: List[int],
        total_tuples: int,
        distinct_count: int,
        row_scale: float = 1.0,
        build_input_signature: str = "",
        timings: Optional[Dict[str, float]] = None,
    ):
        self.config = config
        self.buckets = list(buckets)
        self.split_indices = list(split_indices)
        self.total_tuples = int(total_tuples)
        self.distinct_count = int(distinct_count)
        self.row_scale = float(row_scale)
        self.build_input_signature = build_input_signature
        self.timings = timings or {}
        self._check_invariants()
        self._bake()

    @property
    def B_requested(self) -> int:
        return self.config.n_buckets

    @property
    def B_effective(self) -> int:
        return len(self.buckets)

    def _check_invariants(self):
        """Бакеты непусты, упорядочены, без пересечений и покрывают [0, N)."""
        if not self.buckets:
            assert self.distinct_count == 0 and self.total_tuples == 0
            return
        assert self.buckets[0].start_idx == 0
        assert self.buckets[-1].end_idx == self.distinct_count
        for a, b in zip(self.buckets, self.buckets[1:]):
            assert a.end_idx == b.start_idx and a.max_value < b.min_value
        assert all(b.end_idx > b.start_idx and b.tuple_count > 0 for b in self.buckets)
        assert sum(b.tuple_count for b in self.buckets) == self.total_tuples
        assert self.split_indices == [b.start_idx for b in self.buckets[1:]]

    def _bake(self):
        """Списки для bisect и префиксная сумма tuple_count для горячего цикла predict."""
        self._mins = [b.min_value for b in self.buckets]
        self._maxs = [b.max_value for b in self.buckets]
        self._counts = [float(b.tuple_count) for b in self.buckets]
        self._prefix = [0.0]
        for c in self._counts:
            self._prefix.append(self._prefix[-1] + c)
        self._eq = [
            [_equality_estimate(b, name) for b in self.buckets] for name in EQ_ESTIMATORS
        ]

    # ------------------------------------------------------------------
    # Построение (§4.2, §4.3, шаги 1–10 спецификации §5)
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls, values, counts, config: HistogramConfig, row_scale: float = 1.0
    ) -> "EntropyHistogram1D":
        """Построение по распределению D = [(v_i, f_i)]; values/counts не обязаны быть сгруппированы."""
        t0 = time.perf_counter()
        v, f = aggregate_distribution(values, counts)
        N, T = len(v), int(f.sum())
        if N == 0:
            return cls(config, [], [], 0, 0, row_scale, input_signature(v, f))

        freq_m = PrefixMoments(f, with_powers=True)
        area = area_weights(v, f)
        area_m = PrefixMoments(area)
        t1 = time.perf_counter()

        splits = best_cut_partition(N, min(config.n_buckets, N), config, freq_m, area_m)
        t2 = time.perf_counter()

        bounds = [0] + splits + [N]
        buckets = [
            _make_bucket(start, end, v, f, freq_m, area_m) for start, end in zip(bounds, bounds[1:])
        ]
        timings = {
            "prepare_time": t1 - t0,  # группировка, сортировка, моменты
            "partition_time": t2 - t1,  # сам Algorithm 1
            "build_time": time.perf_counter() - t0,  # end-to-end
        }
        return cls(config, buckets, splits, T, N, row_scale, input_signature(v, f), timings)

    @classmethod
    def build_from_sample(
        cls, sample: np.ndarray, total_rows: int, config: HistogramConfig
    ) -> "EntropyHistogram1D":
        """
        Как EquiWidthHistogram.build_from_sample: D — частоты значений в выборке,
        оценки масштабируются на total_rows / len(sample).
        """
        sample = np.asarray(sample)
        if len(sample) == 0:
            return cls.build([], [], config)
        values, counts = np.unique(sample, return_counts=True)
        return cls.build(values, counts, config, row_scale=total_rows / len(sample))

    # ------------------------------------------------------------------
    # Оценка (§2.2, §4.3.1, §4.5)
    # ------------------------------------------------------------------

    def _bucket_of(self, value: float) -> int:
        """Индекс бакета с min <= value <= max, иначе -1 (вне домена или в щели)."""
        b = bisect.bisect_right(self._mins, value) - 1
        if b < 0 or value > self._maxs[b]:
            return -1
        return b

    def estimate_eq(self, value: float, estimator: Optional[str] = None) -> float:
        """Оценка числа строк с X == value (в единицах таблицы)."""
        name = estimator or self.config.equality_estimator
        b = self._bucket_of(value)
        if b < 0:
            return 0.0
        return self._eq[EQ_ESTIMATORS.index(name)][b] * self.row_scale

    def estimate_range(self, lower: float, upper: float) -> float:
        """Оценка числа строк с lower <= X <= upper (в единицах таблицы)."""
        if not self.buckets or lower > upper:
            return 0.0
        if lower == upper:
            return self.estimate_eq(lower)

        # s — первый бакет с max >= lower, e — последний с min <= upper;
        # всё между ними покрыто запросом целиком
        s = bisect.bisect_left(self._maxs, lower)
        e = bisect.bisect_right(self._mins, upper) - 1
        if s > e:
            return 0.0
        if s == e:
            total = self._partial(s, lower, upper)
        else:
            total = (
                self._partial(s, lower, upper)
                + self._partial(e, lower, upper)
                + self._prefix[e]
                - self._prefix[s + 1]
            )
        return min(max(total, 0.0), float(self.total_tuples)) * self.row_scale

    def _partial(self, b: int, lower: float, upper: float) -> float:
        """Вклад бакета b, пересекающегося с запросом, в единицах выборки."""
        lo_b, hi_b, T_b = self._mins[b], self._maxs[b], self._counts[b]
        if lower <= lo_b and upper >= hi_b:
            return T_b  # покрыт целиком; сюда же попадает однозначный бакет
        if self.config.range_estimator == "discrete_linear":
            overlap = min(upper, hi_b) - max(lower, lo_b) + 1
            return T_b * min(max(overlap / (hi_b - lo_b + 1), 0.0), 1.0)
        overlap = min(upper, hi_b) - max(lower, lo_b)
        return T_b * min(max(overlap / (hi_b - lo_b), 0.0), 1.0)

    def predict(self, q: RangeQuery) -> float:
        """Интерфейс как у остальных моделей бенчмарка."""
        return self.estimate_range(q.low, q.high)

    # ------------------------------------------------------------------
    # Сериализация
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """JSON-сериализуемый снимок; from_dict(to_dict()) восстанавливает оценки бит-в-бит."""
        return {
            "construction_version": CONSTRUCTION_VERSION,
            "config": asdict(self.config),
            "objective_variant": self.config.me_objective if self.config.method == ME else None,
            "B_requested": self.B_requested,
            "B_effective": self.B_effective,
            "total_tuples": self.total_tuples,
            "distinct_count": self.distinct_count,
            "row_scale": self.row_scale,
            "split_indices": self.split_indices,
            "build_input_signature": self.build_input_signature,
            "buckets": [asdict(b) for b in self.buckets],
        }

    @classmethod
    def from_dict(cls, raw: Dict) -> "EntropyHistogram1D":
        if raw.get("construction_version") != CONSTRUCTION_VERSION:
            raise ValueError(f"Unsupported version: {raw.get('construction_version')}")
        return cls(
            HistogramConfig(**raw["config"]),
            [EntropyBucket(**b) for b in raw["buckets"]],
            raw["split_indices"],
            raw["total_tuples"],
            raw["distinct_count"],
            raw["row_scale"],
            raw["build_input_signature"],
        )


def _make_bucket(start, end, v, f, freq_m: PrefixMoments, area_m: PrefixMoments) -> EntropyBucket:
    H_f = float(freq_m.entropy(start, end))
    d = end - start
    return EntropyBucket(
        start_idx=start,
        end_idx=end,
        min_value=v[start].item(),
        max_value=v[end - 1].item(),
        distinct_count=d,
        tuple_count=int(f[start:end].sum()),
        entropy_frequency=H_f,
        entropy_area=float(area_m.entropy(start, end)),
        bias_factor=max(0.0, d * 2.0**-H_f - 1.0),
        sum_f2=float(freq_m.P2[end] - freq_m.P2[start]),
        sum_f3=float(freq_m.P3[end] - freq_m.P3[start]),
        sum_f_log2f=float(freq_m.Plog[end] - freq_m.Plog[start]),
    )


def _equality_estimate(b: EntropyBucket, name: str) -> float:
    """Оценка равенства внутри бакета b, единицы выборки, обрезана до [0, T_b]."""
    T, d = b.tuple_count, b.distinct_count
    if name == "paper_uniform":
        est = T / d
    elif name == "paper_entropy":
        est = T * 2.0**-b.entropy_frequency
    elif name == "paper_bias_factor":
        est = T / d * (1.0 + b.bias_factor)
    else:  # weighted_frequency_mean
        est = b.sum_f2 / T
    return min(max(est, 0.0), float(T))


def input_signature(values: np.ndarray, freq: np.ndarray) -> str:
    """sha256 сгруппированного распределения — для воспроизводимости запусков."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(values, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(freq, dtype=np.int64).tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------
# Best-cut движок (§4.2 Algorithm 1) — общий для ME / MSE / MB
# ----------------------------------------------------------------------


def split_scores(
    start: int, end: int, config: HistogramConfig, freq_m: PrefixMoments, area_m: PrefixMoments
) -> Tuple[np.ndarray, float]:
    """
    Скоры всех разрезов k = start+1..end-1 бакета [start, end) — меньше значит лучше —
    и вес родителя для ничьих. Скор нормирован на полную массу и округлён
    (TIE_DECIMALS), чтобы ничьи разрешались детерминированно.
    """
    k = np.arange(start + 1, end)
    if config.method == ME:
        m = area_m if config.me_input == "area" else freq_m
        W, W_total = m.total(start, end), m.P1[-1]
        if config.me_objective == PAPER_ALG1:
            # Algorithm 1, строки 18/23: невзвешенная сумма энтропий детей
            loss = m.entropy(start, end) - (m.entropy(start, k) + m.entropy(k, end))
            score = W * loss
        else:
            # [EXTENSION] согласованная по весам версия
            score = W * m.entropy(start, end) - (
                m.total(start, k) * m.entropy(start, k) + m.total(k, end) * m.entropy(k, end)
            )
    else:
        cost = freq_m.mse_cost if config.method == MSE else freq_m.mb_cost
        W, W_total = freq_m.total(start, end), freq_m.P1[-1]
        gain = cost(start, end) - cost(start, k) - cost(k, end)
        score = -gain
    return np.round(score / W_total, TIE_DECIMALS), float(W)


def best_cut_partition(
    N: int,
    n_buckets: int,
    config: HistogramConfig,
    freq_m: PrefixMoments,
    area_m: PrefixMoments,
) -> List[int]:
    """
    Жадный нисходящий сплит: куча хранит по одному лучшему разрезу на живой
    делимый бакет; извлечённый бакет делится, в кучу добавляются только его
    дети. Устаревших записей не бывает — извлекается всегда живой бакет.
    Возвращает отсортированные индексы разрезов (разрез k — перед v[k]).
    """

    def push(start, end):
        if end - start < 2:
            return  # однозначный бакет не делится
        scores, W = split_scores(start, end, config, freq_m, area_m)
        i = int(np.argmin(scores))  # первый минимум -> меньший индекс разреза
        # Ключ ничьих: скор, больший вес родителя, меньший разрез, меньший start
        heapq.heappush(heap, (float(scores[i]), -W, start + 1 + i, start, end))

    heap: list = []
    push(0, N)
    splits = []
    while len(splits) < n_buckets - 1:
        _, _, k, start, end = heapq.heappop(heap)
        splits.append(k)
        push(start, k)
        push(k, end)
    return sorted(splits)
