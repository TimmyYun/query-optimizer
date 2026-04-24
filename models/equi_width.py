import numpy as np
import math
from typing import List
from .common import Bucket, RangeQuery


class EquiWidthHistogram:
    """
    Implements a standard Equi-Width Histogram.
    """

    def __init__(self, buckets: List[Bucket]):
        self.buckets = buckets
        self._bake_vectorized_data()

    def _bake_vectorized_data(self):
        """Pre-compute arrays for vectorized inference."""
        if not self.buckets:
            self.b_lo = np.array([], dtype=np.float64)
            self.b_hi = np.array([], dtype=np.float64)
            self.b_count = np.array([], dtype=np.float64)
            self.b_width = np.array([], dtype=np.float64)
            self.prefix_counts = np.array([], dtype=np.float64)
            return

        self.b_lo = np.array([b.lo for b in self.buckets], dtype=np.float64)
        self.b_hi = np.array([b.hi for b in self.buckets], dtype=np.float64)
        self.b_count = np.array([b.count for b in self.buckets], dtype=np.float64)
        self.b_width = self.b_hi - self.b_lo + 1
        
        self.prefix_counts = np.zeros(len(self.buckets) + 1, dtype=np.float64)
        self.prefix_counts[1:] = np.cumsum(self.b_count)

    @staticmethod
    def build_from_sample(
        mn: int, mx: int, bins: int, sample: np.ndarray, total_rows: int
    ) -> "EquiWidthHistogram":
        """
        Builds a histogram using a reservoir sample (Postgres-like).
        Estimates bucket counts based on the sample's distribution.
        """
        if len(sample) == 0:
            return EquiWidthHistogram([])

        sample_size = len(sample)
        scale_factor = total_rows / sample_size if sample_size > 0 else 0

        width = mx - mn + 1
        if bins <= 0:
            bins = 1
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
            idx_start = np.searchsorted(sample_sorted, lo, side="left")
            idx_end = np.searchsorted(sample_sorted, hi, side="right")

            count_in_sample = idx_end - idx_start
            estimated_count = int(count_in_sample * scale_factor)

            buckets.append(Bucket(lo, hi, count=estimated_count))

            cur = hi + 1
            if cur > mx:
                break

        return EquiWidthHistogram(buckets)

    def predict(self, q: RangeQuery) -> float:
        if not self.buckets:
            return 0.0

        n_buckets = len(self.buckets)
        bw = self.b_width[0]
        mn = self.b_lo[0]

        if bw <= 0:
            bw = 1.0

        # O(1) mathematical mapping
        start_idx = int((q.low - mn) // bw)
        start_idx = max(0, min(start_idx, n_buckets - 1))
        
        end_idx = int((q.high - mn) // bw)
        end_idx = max(0, min(end_idx, n_buckets - 1))

        if start_idx == end_idx:
            lo = max(q.low, self.b_lo[start_idx])
            hi = min(q.high, self.b_hi[start_idx])
            overlap = max(0.0, hi - lo + 1)
            return (overlap / self.b_width[start_idx]) * self.b_count[start_idx]
        else:
            s_lo = max(q.low, self.b_lo[start_idx])
            s_hi = self.b_hi[start_idx]
            s_overlap = max(0.0, s_hi - s_lo + 1)
            s_count = (s_overlap / self.b_width[start_idx]) * self.b_count[start_idx]

            e_lo = self.b_lo[end_idx]
            e_hi = min(q.high, self.b_hi[end_idx])
            e_overlap = max(0.0, e_hi - e_lo + 1)
            e_count = (e_overlap / self.b_width[end_idx]) * self.b_count[end_idx]

            mid_count = max(0.0, self.prefix_counts[end_idx] - self.prefix_counts[start_idx + 1])
            return s_count + e_count + mid_count