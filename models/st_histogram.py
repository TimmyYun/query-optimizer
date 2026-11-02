import bisect
import heapq
import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .common import Bucket, RangeQuery

PAPER_FAITHFUL = "paper_faithful"
ROBUST_EXTENDED = "robust_extended"


@dataclass(frozen=True)
class STHistogramConfig:
    """
    Параметры ST-гистограммы. Значения по умолчанию — 1-D настройки статьи (§5.1).
    """

    bucket_count: int = 100  # B — бюджет бакетов
    restructure_every: int = 200  # R — период реструктуризации (в запросах)
    alpha: float = 0.5  # демпфирование обновления, 0 < alpha <= 1
    merge_threshold: float = 0.00025  # m — ДОЛЯ от T (0.025%)
    split_fraction: float = 0.10  # s — ДОЛЯ от B (10%)
    enable_restructuring: bool = True  # False -> абляция "no restructure" (Table 2)
    mode: str = PAPER_FAITHFUL
    # [EXTENSION] только для robust_extended: держать sum(freq) == T_reference
    renormalize_mass: bool = False

    def __post_init__(self):
        if self.bucket_count < 1:
            raise ValueError("bucket_count must be >= 1")
        if self.restructure_every < 1:
            raise ValueError("restructure_every must be >= 1")
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        if self.merge_threshold < 0 or self.split_fraction < 0:
            raise ValueError("merge_threshold and split_fraction must be >= 0")
        if self.mode not in (PAPER_FAITHFUL, ROBUST_EXTENDED):
            raise ValueError(f"Unknown mode: {self.mode}")
        if self.renormalize_mass and self.mode != ROBUST_EXTENDED:
            raise ValueError("renormalize_mass is an extension: requires mode='robust_extended'")


@dataclass(frozen=True)
class RestructureResult:
    applied: bool
    reason: str  # "applied" | "nothing_to_merge" | "split_allocation_failed"
    n_runs: int  # число run'ов после слияния
    freed: int  # F — освобождённые слоты
    n_split: int  # сколько бакетов было разбито


@dataclass(frozen=True, slots=True)
class FeedbackResult:
    estimate: float  # оценка ДО обновления (est в Fig. 2)
    actual: float
    error: float  # act - est
    event: str
    restructure: Optional[RestructureResult]


class STHistogram:
    """
    1-D Self-Tuning Histogram (ST-histogram):
    A. Aboulnaga, S. Chaudhuri, "Self-tuning Histograms: Building Histograms
    Without Looking at Data", SIGMOD 1999, doi:10.1145/304182.304198.
    Clean-room реализация по спецификации ST_Histogram_1D_Claude_Code_Spec.md.

    Модель НЕ смотрит на данные: ни на freq, ни на sample. Единственные входы —
    метаданные каталога (T, min, max) при инициализации и фидбэк исполненных
    запросов (low, high, actual_count). Данные меняются -> гистограмма узнаёт
    об этом только через последующий фидбэк (§1, §5.5).

    Жизненный цикл:
      __init__           — B равноширинных бакетов, freq_i = T / B        (§3.1)
      estimate_range     — сумма w(Q,b) * f_b, равномерность внутри бакета (§3.2)
      observe_feedback   — Fig. 2: f_b += alpha * err * (w * f_b) / est     (§3.2)
                           каждые R запросов -> restructure()
      restructure        — Fig. 3: жадное слияние похожих соседних run'ов,
                           освобождённые слоты -> разбиение top-s*B бакетов (§3.3)

    Бакеты — непересекающиеся включительные целочисленные интервалы
    [lo, hi], lo_{i+1} = hi_i + 1, покрывающие весь домен.

    Решения, которые статья не фиксирует ([ENGINEERING]):
      * границы бакетов и дочерних бакетов: lo_i = min + floor(i * D / B);
      * D < B -> B_effective = D (событие "effective_bucket_count_reduced");
      * est == 0 -> частоты не меняются, событие "unlearnable_zero_estimate"
        (act > 0) или "zero_estimate_correct" (act == 0);
        запрос целиком вне домена -> "out_of_domain" (act > 0);
      * равные расстояния при слиянии -> берётся самая левая пара;
      * k = max(1, floor(s * B)) при s > 0; кандидаты на разбиение
        сортируются по (-freq, lo);
      * слоты распределяются методом наибольших остатков с ёмкостью
        width - 1 на кандидата; бакеты с freq == 0 слотов не получают;
      * распределение невозможно -> реструктуризация пропускается целиком
        (событие "restructure_noop_split_allocation_failed");
      * счётчик запросов увеличивается на каждый принятый фидбэк (в т.ч.
        no-op при est == 0), невалидный фидбэк отклоняется (ValueError).

    НЕ делается в paper_faithful (важно для честного baseline):
      * нормализация sum(freq) к T — Fig. 2 её не делает, масса дрейфует;
      * обновления по INSERT/DELETE, сканы данных, детектор дрейфа;
      * восстановление бакетов с freq == 0.

    robust_extended ([EXTENSION], отдельная абляция, не baseline):
      * est == 0, act > 0 -> "zero_estimate_revived": alpha * act
        распределяется по перекрытым значениям пропорционально их числу;
      * при нулевой массе всех кандидатов слоты делятся поровну;
      * renormalize_mass=True -> sum(freq) масштабируется к T_reference.
    Расширение домена за пределы [min, max] не реализовано ни в одном режиме.
    """

    def __init__(
        self,
        config: STHistogramConfig,
        total_rows: float,
        min_value: int,
        max_value: int,
    ):
        if not math.isfinite(total_rows) or total_rows < 0:
            raise ValueError("total_rows must be finite and >= 0")
        if min_value > max_value:
            raise ValueError("min_value must be <= max_value")

        self.config = config
        self.T_reference = float(total_rows)
        self.domain_min = int(min_value)
        self.domain_max = int(max_value)
        self.processed_queries = 0
        self.events: Counter = Counter()

        D = self.domain_max - self.domain_min + 1
        n = min(config.bucket_count, D)
        if n < config.bucket_count:
            self.events["effective_bucket_count_reduced"] += 1

        cuts = self.domain_min + (np.arange(n + 1, dtype=np.int64) * D) // n
        self._set_buckets(cuts[:-1], cuts[1:] - 1, np.full(n, self.T_reference / n))

    # ------------------------------------------------------------------
    # Состояние
    # ------------------------------------------------------------------

    def _set_buckets(self, lows, highs, freq):
        lows = np.asarray(lows, dtype=np.int64)
        highs = np.asarray(highs, dtype=np.int64)
        freq = np.asarray(freq, dtype=np.float64).copy()

        # Инварианты: непустые, отсортированные, без дыр, покрывают домен
        assert len(lows) == len(highs) == len(freq) >= 1
        assert lows[0] == self.domain_min and highs[-1] == self.domain_max
        assert np.all(highs >= lows)
        assert np.all(lows[1:] == highs[:-1] + 1)
        assert np.all(np.isfinite(freq)) and np.all(freq >= 0)
        assert len(freq) <= self.config.bucket_count

        self.lows = lows
        self.highs = highs
        self.widths = (highs - lows + 1).astype(np.float64)
        self.freq = freq
        self._lows_list = lows.tolist()  # для bisect в горячем цикле
        self._prefix = None

    @classmethod
    def from_buckets(
        cls, config: STHistogramConfig, total_rows: float, buckets: List[Bucket]
    ) -> "STHistogram":
        """Гистограмма с заданными бакетами (тесты, ручные сценарии)."""
        hist = cls(config, total_rows, buckets[0].lo, buckets[-1].hi)
        hist._set_buckets(
            [b.lo for b in buckets], [b.hi for b in buckets], [b.count for b in buckets]
        )
        return hist

    @property
    def buckets(self) -> List[Bucket]:
        return [
            Bucket(int(lo), int(hi), count=float(f))
            for lo, hi, f in zip(self.lows, self.highs, self.freq)
        ]

    @property
    def total_mass(self) -> float:
        return float(self.freq.sum())

    def get_state(self) -> Dict:
        """JSON-сериализуемый снимок. from_state(get_state()) продолжает поток бит-в-бит."""
        return {
            "version": 1,
            "config": asdict(self.config),
            "T_reference": self.T_reference,
            "domain_min": self.domain_min,
            "domain_max": self.domain_max,
            "lows": self.lows.tolist(),
            "highs": self.highs.tolist(),
            "freq": self.freq.tolist(),
            "processed_queries": self.processed_queries,
            "events": dict(self.events),
        }

    @classmethod
    def from_state(cls, state: Dict) -> "STHistogram":
        if state.get("version") != 1:
            raise ValueError(f"Unsupported state version: {state.get('version')}")
        config = STHistogramConfig(**state["config"])
        hist = cls(config, state["T_reference"], state["domain_min"], state["domain_max"])
        hist._set_buckets(state["lows"], state["highs"], state["freq"])
        hist.processed_queries = state["processed_queries"]
        hist.events = Counter(state["events"])
        return hist

    def update_catalog_metadata(self, total_rows: float):
        """
        Явное обновление T из каталога. Меняет ТОЛЬКО порог слияния m * T,
        частоты бакетов не трогаются (масса не перенормируется).
        """
        if not math.isfinite(total_rows) or total_rows < 0:
            raise ValueError("total_rows must be finite and >= 0")
        self.T_reference = float(total_rows)
        self.events["catalog_refresh"] += 1

    # ------------------------------------------------------------------
    # Оценка (§3.2)
    # ------------------------------------------------------------------

    def _overlap_span(self, low: int, high: int) -> Optional[Tuple[int, int, int, int]]:
        """(s, e, ql, qr): индексы крайних перекрытых бакетов и запрос, обрезанный по домену."""
        ql = max(low, self.domain_min)
        qr = min(high, self.domain_max)
        if ql > qr:
            return None
        s = bisect.bisect_right(self._lows_list, ql) - 1
        e = bisect.bisect_right(self._lows_list, qr) - 1
        return s, e, ql, qr

    def _contributions(self, s: int, e: int, ql: int, qr: int) -> np.ndarray:
        """w(Q, b) * f_b для бакетов s..e (внутренние бакеты покрыты целиком, w = 1)."""
        w = np.ones(e - s + 1)
        if s == e:
            w[0] = (qr - ql + 1) / self.widths[s]
        else:
            w[0] = (self.highs[s] - ql + 1) / self.widths[s]
            w[-1] = (qr - self.lows[e] + 1) / self.widths[e]
        return w * self.freq[s : e + 1]

    def _prefix_sums(self) -> np.ndarray:
        if self._prefix is None:
            self._prefix = np.zeros(len(self.freq) + 1)
            np.cumsum(self.freq, out=self._prefix[1:])
        return self._prefix

    def estimate_range(self, low: int, high: int) -> float:
        """
        N̂(Q) = sum_b w(Q,b) * f_b для включительного [low, high].
        Не округляется и не обрезается до [0, T].
        """
        span = self._overlap_span(low, high)
        if span is None:
            return 0.0
        s, e, ql, qr = span
        if s == e:
            return float((qr - ql + 1) / self.widths[s] * self.freq[s])

        # Те же w * f, что в _contributions, но середина через префиксные суммы
        prefix = self._prefix_sums()
        left = (self.highs[s] - ql + 1) / self.widths[s] * self.freq[s]
        right = (qr - self.lows[e] + 1) / self.widths[e] * self.freq[e]
        return float(left + right + prefix[e] - prefix[s + 1])

    def predict(self, q: RangeQuery) -> float:
        """Интерфейс как у остальных моделей бенчмарка."""
        return self.estimate_range(q.low, q.high)

    # ------------------------------------------------------------------
    # Обновление частот по фидбэку (§3.2, Fig. 2)
    # ------------------------------------------------------------------

    def observe_feedback(self, low: int, high: int, actual_count: float) -> FeedbackResult:
        """
        Fig. 2. est считается по состоянию ДО обновления:
            err  = act - est
            f'_b = max(0, f_b + alpha * err * (w(Q,b) * f_b) / est)
        Каждые R принятых запросов вызывается restructure().
        """
        if low > high:
            raise ValueError(f"Reversed range [{low}, {high}]")
        if not math.isfinite(actual_count) or actual_count < 0:
            raise ValueError(f"actual_count must be finite and >= 0, got {actual_count}")

        actual = float(actual_count)
        robust = self.config.mode == ROBUST_EXTENDED
        span = self._overlap_span(low, high)

        if span is None:
            est = 0.0
            event = "zero_estimate_correct" if actual == 0 else "out_of_domain"
        else:
            s, e, ql, qr = span
            contrib = self._contributions(s, e, ql, qr)
            est = float(contrib.sum())

            if est > 0:
                err = actual - est
                seg = self.freq[s : e + 1]
                self.freq[s : e + 1] = np.maximum(
                    0.0, seg + self.config.alpha * err * (contrib / est)
                )
                event = "updated"
            elif actual == 0:
                event = "zero_estimate_correct"
            elif robust:
                self._revive_zero_estimate(s, e, ql, qr, actual)
                event = "zero_estimate_revived"
            else:
                # Fig. 2 делит на est — при est == 0 поведение не определено
                event = "unlearnable_zero_estimate"

            if robust and self.config.renormalize_mass:
                self._renormalize_mass()
            self._prefix = None

        self.events[event] += 1
        self.processed_queries += 1

        restructure = None
        if (
            self.config.enable_restructuring
            and self.processed_queries % self.config.restructure_every == 0
        ):
            restructure = self.restructure()

        return FeedbackResult(est, actual, actual - est, event, restructure)

    def _revive_zero_estimate(self, s: int, e: int, ql: int, qr: int, actual: float):
        """[EXTENSION] alpha * act делится по перекрытым значениям (не по массе — её нет)."""
        n_values = self._overlap_values(s, e, ql, qr)
        self.freq[s : e + 1] += self.config.alpha * actual * n_values / n_values.sum()

    def _overlap_values(self, s: int, e: int, ql: int, qr: int) -> np.ndarray:
        """Число значений запроса, попавших в каждый бакет s..e."""
        lo = np.maximum(self.lows[s : e + 1], ql)
        hi = np.minimum(self.highs[s : e + 1], qr)
        return (hi - lo + 1).astype(np.float64)

    def _renormalize_mass(self):
        """[EXTENSION] sum(freq) := T_reference."""
        mass = self.freq.sum()
        if mass > 0:
            self.freq *= self.T_reference / mass

    # ------------------------------------------------------------------
    # Реструктуризация (§3.3, Fig. 3)
    # ------------------------------------------------------------------

    def restructure(self) -> RestructureResult:
        """
        1. Слияние: жадно объединяем соседние run'ы с минимальным
           max |f_i - f_j| (по ИСХОДНЫМ бакетам), пока он <= m * T.
        2. Разбиение: F освобождённых слотов делятся между top-k (k = s * B)
           самыми тяжёлыми неслитыми бакетами пропорционально частоте;
           бакет с a_i слотами -> 1 + a_i равных по ширине и частоте детей.
        Масса и покрытие домена сохраняются; число бакетов не меняется.
        """
        freq = self.freq.copy()
        n = len(freq)
        threshold = self.config.merge_threshold * self.T_reference

        runs = greedy_merge_runs(freq, threshold)
        freed = n - len(runs)
        if freed == 0:
            return self._log_restructure(False, "nothing_to_merge", len(runs), 0, 0)

        merged = np.zeros(n, dtype=bool)
        for a, b in runs:
            if b > a:
                merged[a : b + 1] = True

        candidates = [i for i in range(n) if not merged[i] and self.widths[i] > 1]
        candidates.sort(key=lambda i: (-freq[i], self.lows[i]))
        selected = candidates[: split_candidate_count(self.config)]

        alloc = allocate_split_slots(
            freed,
            weights=[freq[i] for i in selected],
            capacities=[int(self.widths[i]) - 1 for i in selected],
            tie_keys=[int(self.lows[i]) for i in selected],
            uniform_if_zero=self.config.mode == ROBUST_EXTENDED,
        )
        if alloc is None:
            # Слияние без разбиения нарушило бы бюджет B — пропускаем атомарно
            return self._log_restructure(
                False, "split_allocation_failed", len(runs), freed, 0
            )
        extra = {i: a for i, a in zip(selected, alloc) if a > 0}

        new_lows, new_highs, new_freq = [], [], []
        for a, b in runs:
            if b > a:
                new_lows.append(self.lows[a])
                new_highs.append(self.highs[b])
                new_freq.append(freq[a : b + 1].sum())
                continue

            c = 1 + extra.get(a, 0)
            L, D = int(self.lows[a]), int(self.widths[a])
            for j in range(c):
                new_lows.append(L + (j * D) // c)
                new_highs.append(L + ((j + 1) * D) // c - 1)
                new_freq.append(freq[a] / c)

        assert len(new_freq) == n
        assert math.isclose(sum(new_freq), freq.sum(), rel_tol=1e-9, abs_tol=1e-6)
        self._set_buckets(new_lows, new_highs, new_freq)
        return self._log_restructure(True, "applied", len(runs), freed, len(extra))

    def _log_restructure(self, applied, reason, n_runs, freed, n_split):
        key = "restructure_applied" if applied else f"restructure_noop_{reason}"
        self.events[key] += 1
        return RestructureResult(applied, reason, n_runs, freed, n_split)


# ----------------------------------------------------------------------
# Чистые функции Fig. 3 (вынесены для тестов)
# ----------------------------------------------------------------------


def greedy_merge_runs(freq: Sequence[float], threshold: float) -> List[Tuple[int, int]]:
    """
    Жадное слияние соседних run'ов (Fig. 3). Расстояние между run'ами A и B:
        d(A, B) = max_{i in A, j in B} |f_i - f_j| = max(maxA - minB, maxB - minA)
    — по частотам ИСХОДНЫХ бакетов, а не по суммам run'ов. На каждом шаге
    сливается пара с минимальным d (при равенстве — самая левая), пока d <= threshold.

    Вместо пересчёта всех расстояний (O(B^2)) — куча с ленивой инвалидацией
    по версиям run'ов: тот же порядок слияний за O(B log B).
    Возвращает run'ы как (first_idx, last_idx) в порядке домена.
    """
    n = len(freq)
    if n == 0:
        return []

    # Run идентифицируется индексом своего первого бакета
    last = list(range(n))
    run_min = [float(f) for f in freq]
    run_max = list(run_min)
    nxt = list(range(1, n)) + [-1]
    prv = [-1] + list(range(n - 1))
    alive = [True] * n
    version = [0] * n

    def dist(a, b):
        return max(run_max[a] - run_min[b], run_max[b] - run_min[a])

    # (d, левый run, версия левого, версия правого) — равные d -> меньший индекс
    heap = [(dist(i, i + 1), i, 0, 0) for i in range(n - 1)]
    heapq.heapify(heap)

    while heap:
        d, a, va, vb = heapq.heappop(heap)
        if not alive[a] or nxt[a] == -1:
            continue
        b = nxt[a]
        if version[a] != va or version[b] != vb:
            continue  # пара устарела после другого слияния
        if d > threshold:
            break

        last[a] = last[b]
        run_min[a] = min(run_min[a], run_min[b])
        run_max[a] = max(run_max[a], run_max[b])
        alive[b] = False
        nxt[a] = nxt[b]
        if nxt[a] != -1:
            prv[nxt[a]] = a
        version[a] += 1

        if prv[a] != -1:
            p = prv[a]
            heapq.heappush(heap, (dist(p, a), p, version[p], version[a]))
        if nxt[a] != -1:
            q = nxt[a]
            heapq.heappush(heap, (dist(a, q), a, version[a], version[q]))

    return [(i, last[i]) for i in range(n) if alive[i]]


def split_candidate_count(config: STHistogramConfig) -> int:
    """k = s * B; статья не задаёт округление -> floor, но не меньше 1 при s > 0."""
    if config.split_fraction == 0:
        return 0
    return max(1, math.floor(config.split_fraction * config.bucket_count + 1e-9))


def allocate_split_slots(
    freed: int,
    weights: Sequence[float],
    capacities: Sequence[int],
    tie_keys: Sequence[int],
    uniform_if_zero: bool = False,
) -> Optional[List[int]]:
    """
    Делит `freed` слотов между кандидатами пропорционально весам (частотам)
    методом наибольших остатков: сначала floor(a_i*), затем остаток — по
    убыванию дробной части, при равенстве — по tie_keys (левая граница).
    a_i не превышает capacities[i]; излишек перераспределяется между
    оставшимися кандидатами. Кандидаты с весом 0 слотов не получают.

    Возвращает None, если раздать все слоты невозможно (нет кандидатов,
    нулевая масса, не хватает ёмкости). uniform_if_zero=True ([EXTENSION]):
    при нулевой суммарной массе веса считаются равными.
    """
    n = len(weights)
    weights = [float(w) for w in weights]
    if n > 0 and sum(weights) <= 0 and uniform_if_zero:
        weights = [1.0] * n

    alloc = [0] * n
    remaining = freed
    active = [i for i in range(n) if weights[i] > 0 and capacities[i] > 0]

    while remaining > 0 and active:
        total = sum(weights[i] for i in active)
        quota = {i: remaining * weights[i] / total for i in active}
        give = {i: min(math.floor(quota[i]), capacities[i] - alloc[i]) for i in active}
        leftover = remaining - sum(give.values())

        order = sorted(
            active, key=lambda i: (-(quota[i] - math.floor(quota[i])), tie_keys[i])
        )
        for i in order:
            if leftover == 0:
                break
            if alloc[i] + give[i] < capacities[i]:
                give[i] += 1
                leftover -= 1

        for i in active:
            alloc[i] += give[i]
        remaining = leftover
        active = [i for i in active if alloc[i] < capacities[i]]

    return alloc if remaining == 0 else None
