import math
from typing import Dict, List

import numpy as np

from .common import Bucket, RangeQuery


class TKHistogram:
    """
    1-D TKHist (Top-K Histogram) — порт NonKeyTopKHistogram из DHist
    (github.com/qingzma/dhist, ветка JOB-light,
    joins/histograms/non_key_histogram.py) на целочисленный домен бенчмарка.
    Join-часть оригинала не реализуется.

    Структура: equi-width гистограмма, в каждом бакете которой дополнительно
    хранятся top-k самых частых значений с их счётчиками. Остаток бакета
    (count - сумма top-k) считается равномерно распределённым по ширине бакета.

    Оценка запроса [low, high]:
      * бакеты, целиком лежащие внутри запроса -> их полный count;
      * краевые (частично покрытые) бакеты -> top-k значения, попавшие в
        запрос, + доля остатка, пропорциональная перекрытию.

    Как и в оригинале, если различных значений меньше n_categorical, колонка
    считается категориальной и каждое значение хранится точно.
    При top_k=0 модель сводится к EquiWidthHistogram.
    """

    def __init__(self, buckets: List[Bucket], top_k: List[Dict[int, float]]):
        # top_k[i] = {value: count} для бакета buckets[i]
        assert len(buckets) == len(top_k)
        self.buckets = buckets
        self.top_k = top_k
        self._bake_vectorized_data()

    def _bake_vectorized_data(self):
        """Pre-compute arrays for inference."""
        self.b_lo = np.array([b.lo for b in self.buckets], dtype=np.float64)
        self.b_hi = np.array([b.hi for b in self.buckets], dtype=np.float64)
        self.b_count = np.array([b.count for b in self.buckets], dtype=np.float64)
        self.b_width = self.b_hi - self.b_lo + 1

        # Масса бакета, не покрытая top-k (равномерная часть)
        topk_mass = np.array([sum(t.values()) for t in self.top_k], dtype=np.float64)
        self.b_resid = np.maximum(self.b_count - topk_mass, 0.0)

        self.prefix_counts = np.zeros(len(self.buckets) + 1, dtype=np.float64)
        self.prefix_counts[1:] = np.cumsum(self.b_count)

    @property
    def n_topk_entries(self) -> int:
        return sum(len(t) for t in self.top_k)

    @staticmethod
    def build_from_sample(
        mn: int,
        mx: int,
        bins: int,
        sample: np.ndarray,
        total_rows: int,
        top_k: int = 5,
        n_categorical: int = 200,
    ) -> "TKHistogram":
        """
        Builds the histogram from a reservoir sample, scaled to total_rows
        (как EquiWidthHistogram.build_from_sample — модель не видит freq).
        Значения выборки вне [mn, mx] прижимаются к краям домена.
        """
        if len(sample) == 0:
            return TKHistogram([], [])

        scale_factor = total_rows / len(sample)

        width = mx - mn + 1
        if bins <= 0:
            bins = 1
        bw = max(1, int(math.ceil(width / bins)))
        n_buckets = int(math.ceil(width / bw))

        # Частоты различных значений выборки (values отсортированы)
        values, counts = np.unique(np.clip(sample, mn, mx), return_counts=True)

        # Категориальный режим оригинала: мало различных значений -> храним все
        if len(values) < n_categorical:
            top_k = len(values)

        bucket_idx = (values - mn) // bw
        # Границы значений каждого бакета в массиве values
        bounds = np.searchsorted(bucket_idx, np.arange(n_buckets + 1), side="left")

        buckets = []
        top_k_list = []
        for i in range(n_buckets):
            lo = mn + i * bw
            hi = min(mx, lo + bw - 1)

            b_vals = values[bounds[i] : bounds[i + 1]]
            b_cnts = counts[bounds[i] : bounds[i + 1]]

            # Top-k по частоте; при равенстве — меньшее значение (stable sort)
            order = np.argsort(-b_cnts, kind="stable")[:top_k]
            top = {int(b_vals[j]): float(b_cnts[j] * scale_factor) for j in order}

            buckets.append(Bucket(lo, hi, count=float(b_cnts.sum() * scale_factor)))
            top_k_list.append(top)

        return TKHistogram(buckets, top_k_list)

    def _partial_estimate(self, idx: int, low: int, high: int) -> float:
        """Оценка для части бакета idx, пересекающейся с [low, high]."""
        lo = max(low, self.b_lo[idx])
        hi = min(high, self.b_hi[idx])
        if lo > hi:
            return 0.0

        # Равномерная часть остатка
        est = ((hi - lo + 1) / self.b_width[idx]) * self.b_resid[idx]

        # Top-k значения, попавшие в запрос
        for value, count in self.top_k[idx].items():
            if lo <= value <= hi:
                est += count
        return est

    def predict(self, q: RangeQuery) -> float:
        if not self.buckets:
            return 0.0

        n_buckets = len(self.buckets)
        bw = self.b_width[0]
        mn = self.b_lo[0]

        # O(1) mathematical mapping (все бакеты одной ширины, кроме последнего)
        start_idx = int((q.low - mn) // bw)
        start_idx = max(0, min(start_idx, n_buckets - 1))

        end_idx = int((q.high - mn) // bw)
        end_idx = max(0, min(end_idx, n_buckets - 1))

        if start_idx == end_idx:
            return self._partial_estimate(start_idx, q.low, q.high)

        s_count = self._partial_estimate(start_idx, q.low, q.high)
        e_count = self._partial_estimate(end_idx, q.low, q.high)
        # Полностью покрытые бакеты: полный count (top-k уже внутри него)
        mid_count = max(
            0.0, self.prefix_counts[end_idx] - self.prefix_counts[start_idx + 1]
        )
        return s_count + e_count + mid_count
