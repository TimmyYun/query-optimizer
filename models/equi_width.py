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

    def predict_batch(self, queries: List[RangeQuery]) -> np.ndarray:
        """
        Vectorized bulk inference for Equi-Width Histogram using O(1) mathematical lookup.
        """
        if not self.buckets:
            return np.zeros(len(queries))

        n_queries = len(queries)
        q_lo = np.array([q.low for q in queries], dtype=np.float64)
        q_hi = np.array([q.high for q in queries], dtype=np.float64)
        total = np.zeros(n_queries)

        n_buckets = len(self.buckets)
        bw = self.b_width[0]
        mn = self.b_lo[0]

        if bw <= 0:
            bw = 1.0

        # O(1) mathematical mapping
        start_idx = np.clip((q_lo - mn) // bw, 0, n_buckets - 1).astype(int)
        end_idx = np.clip((q_hi - mn) // bw, 0, n_buckets - 1).astype(int)

        mask_same = start_idx == end_idx
        mask_diff = ~mask_same

        if np.any(mask_same):
            idx = start_idx[mask_same]
            lo = np.maximum(q_lo[mask_same], self.b_lo[idx])
            hi = np.minimum(q_hi[mask_same], self.b_hi[idx])
            overlap = np.maximum(0.0, hi - lo + 1)
            total[mask_same] = (overlap / self.b_width[idx]) * self.b_count[idx]

        if np.any(mask_diff):
            s_idx = start_idx[mask_diff]
            e_idx = end_idx[mask_diff]

            # Start bucket overlap
            s_lo = np.maximum(q_lo[mask_diff], self.b_lo[s_idx])
            s_hi = self.b_hi[s_idx]
            s_overlap = np.maximum(0.0, s_hi - s_lo + 1)
            s_count = (s_overlap / self.b_width[s_idx]) * self.b_count[s_idx]

            # End bucket overlap
            e_lo = self.b_lo[e_idx]
            e_hi = np.minimum(q_hi[mask_diff], self.b_hi[e_idx])
            e_overlap = np.maximum(0.0, e_hi - e_lo + 1)
            e_count = (e_overlap / self.b_width[e_idx]) * self.b_count[e_idx]

            # Middle buckets pure O(1) sum via prefix_counts
            mid_count = np.maximum(0.0, self.prefix_counts[e_idx] - self.prefix_counts[s_idx + 1])

            total[mask_diff] = s_count + e_count + mid_count

        return total

    def predict(self, q: RangeQuery) -> float:
        """Scalar fallback."""
        return float(self.predict_batch([q])[0])
