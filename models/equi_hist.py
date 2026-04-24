import numpy as np
from typing import List
from .common import Bucket, RangeQuery


class EquiHistLearner:
    """
    Implements a self-tuning histogram baseline (EquiHist-like).
    """

    def __init__(self, buckets: List[Bucket], learning_rate: float = 0.5):
        self.buckets = [Bucket(b.lo, b.hi, count=b.count) for b in buckets]
        self.lr = learning_rate

    def predict(self, q: RangeQuery) -> float:
        total = 0.0
        for b in self.buckets:
            ov_lo = max(q.low, b.lo)
            ov_hi = min(q.high, b.hi)
            if ov_lo <= ov_hi:
                w = b.hi - b.lo + 1
                frac = (ov_hi - ov_lo + 1) / w
                total += frac * b.count
        return total

    def update(self, q: RangeQuery, actual: float, pred: float = None):
        if pred is None:
            pred = self.predict(q)
        if pred == 0:
            return
        # Adaptive learning rate based on error
        ratio = actual / pred

        # Limit ratio to avoid explosions
        ratio = max(0.1, min(10.0, ratio))

        for b in self.buckets:
            # Overlap logic
            ov_lo = max(q.low, b.lo)
            ov_hi = min(q.high, b.hi)

            if ov_lo <= ov_hi:
                # Update bucket count based on overlap contribution
                w = b.hi - b.lo + 1
                overlap_frac = (ov_hi - ov_lo + 1) / w

                # Formula: new_count = old_count * (ratio ^ (lr * overlap))
                # If overlap is 1.0 (full bucket inside query), it gets full update
                # If overlap is small, it gets small update
                factor = ratio ** (self.lr * overlap_frac)
                b.count = max(1.0, b.count * factor)

        # Invalidate vector cache
        self.b_lo = None

    def predict_batch(self, queries: List[RangeQuery]) -> np.ndarray:
        """
        Vectorized bulk inference using O(1) mathematical lookup.
        """
        if not hasattr(self, "b_lo") or self.b_lo is None:
            self._bake_vectorized_data()

        n_queries = len(queries)
        if n_queries == 0:
            return np.array([])
            
        if len(self.buckets) == 0:
            return np.zeros(n_queries)

        q_lo = np.array([q.low for q in queries], dtype=np.float64)
        q_hi = np.array([q.high for q in queries], dtype=np.float64)
        total = np.zeros(n_queries)

        n_buckets = len(self.buckets)
        bw = self.b_width[0]
        mn = self.b_lo[0]

        if bw <= 0:
            bw = 1.0

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

            # Middle buckets sum
            mid_count = np.maximum(0.0, self.prefix_counts[e_idx] - self.prefix_counts[s_idx + 1])

            total[mask_diff] = s_count + e_count + mid_count

        return total

    def _bake_vectorized_data(self):
        self.b_lo = np.array([b.lo for b in self.buckets], dtype=np.float64)
        self.b_hi = np.array([b.hi for b in self.buckets], dtype=np.float64)
        self.b_count = np.array([b.count for b in self.buckets], dtype=np.float64)
        self.b_width = self.b_hi - self.b_lo + 1
        self.prefix_counts = np.zeros(len(self.buckets) + 1, dtype=np.float64)
        self.prefix_counts[1:] = np.cumsum(self.b_count)
