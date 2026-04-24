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
        if not hasattr(self, "b_lo") or self.b_lo is None:
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

    def _bake_vectorized_data(self):
        self.b_lo = np.array([b.lo for b in self.buckets], dtype=np.float64)
        self.b_hi = np.array([b.hi for b in self.buckets], dtype=np.float64)
        self.b_count = np.array([b.count for b in self.buckets], dtype=np.float64)
        self.b_width = self.b_hi - self.b_lo + 1
        self.prefix_counts = np.zeros(len(self.buckets) + 1, dtype=np.float64)
        self.prefix_counts[1:] = np.cumsum(self.b_count)
