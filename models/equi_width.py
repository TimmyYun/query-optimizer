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
            return

        self.b_lo = np.array([b.lo for b in self.buckets], dtype=np.float64)
        self.b_hi = np.array([b.hi for b in self.buckets], dtype=np.float64)
        self.b_count = np.array([b.count for b in self.buckets], dtype=np.float64)
        self.b_width = self.b_hi - self.b_lo + 1

    @staticmethod
    def build(
        mn: int,
        mx: int,
        bins: int,
        freq: np.ndarray,
        prefix_buckets: int = 0,
        suffix_buckets: int = 0,
    ) -> "EquiWidthHistogram":
        if len(freq) == 0:
            return EquiWidthHistogram([])

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
            start_idx = distinct_indices[p_count - 1] + 1 if p_count > 0 else 0
            end_idx = (
                remaining_indices[-(s_count + 1)] if s_count > 0 else len(freq) - 1
            )

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

                    if li < 0:
                        li = 0
                    if ri >= len(mid_freq):
                        ri = len(mid_freq) - 1

                    if ri >= li:
                        cnt = int(ps[ri] - (ps[li - 1] if li > 0 else 0))
                        mid_buckets.append(Bucket(lo, hi, count=cnt))

                    cur = hi + 1
                    if cur > mid_mx:
                        break

        return EquiWidthHistogram(pb + mid_buckets + sb)

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
        Vectorized bulk inference for Equi-Width Histogram.
        """
        if not self.buckets:
            return np.zeros(len(queries))

        n_queries = len(queries)
        q_lo = np.array([q.low for q in queries], dtype=np.float64)
        q_hi = np.array([q.high for q in queries], dtype=np.float64)
        total = np.zeros(n_queries)

        # Iterate over buckets (vectorized over queries)
        # This is generally faster than iterating over queries if N_Buckets << N_Queries
        for i in range(len(self.buckets)):
            b_cnt = self.b_count[i]
            if b_cnt == 0:
                continue

            b_lo = self.b_lo[i]
            b_hi = self.b_hi[i]
            b_width = self.b_width[i]

            # Mask: query overlaps with bucket
            # Overlap if: q.high >= b.lo AND q.low <= b.hi
            mask = (q_hi >= b_lo) & (q_lo <= b_hi)

            if not np.any(mask):
                continue

            # Vectorized Overlap Calculation
            # lo = max(q_lo, b_lo)
            # hi = min(q_hi, b_hi)

            lo = np.maximum(q_lo[mask], b_lo)
            hi = np.minimum(q_hi[mask], b_hi)

            overlap_width = hi - lo + 1
            # overlap_width = np.maximum(0, overlap_width) # Implicitly handled by mask?
            # Actually mask guarantees q_hi >= b_lo and q_lo <= b_hi,
            # so hi >= lo is guaranteed.

            total[mask] += (overlap_width / b_width) * b_cnt

        return total

    def predict(self, q: RangeQuery) -> float:
        """Scalar fallback."""
        return float(self.predict_batch([q])[0])
