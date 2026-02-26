import numpy as np
from typing import List, Optional
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
        Vectorized bulk inference.
        """
        # Auto-bake if needed or stale (simple check: if b_lo is None)
        # Note: If called in a loop without updates, this is fast.
        # If called interleaved with updates, it will re-bake often (overhead).
        if not hasattr(self, "b_lo") or self.b_lo is None:
            self._bake_vectorized_data()

        n_queries = len(queries)
        if n_queries == 0:
            return np.array([])

        q_lo = np.array([q.low for q in queries], dtype=np.float64)
        q_hi = np.array([q.high for q in queries], dtype=np.float64)
        total = np.zeros(n_queries)

        # Vectorized Bucket Loop
        for i in range(len(self.buckets)):
            # b properties from arrays
            b_lo = self.b_lo[i]
            b_hi = self.b_hi[i]
            b_cnt = self.b_count[i]
            b_width = self.b_width[i]

            # Overlap: lo = max(q_lo, b_lo), hi = min(q_hi, b_hi)
            # Mask: q_hi >= b_lo & q_lo <= b_hi

            mask = (q_hi >= b_lo) & (q_lo <= b_hi)
            if not np.any(mask):
                continue

            lo = np.maximum(q_lo[mask], b_lo)
            hi = np.minimum(q_hi[mask], b_hi)

            overlap_width = hi - lo + 1
            total[mask] += (overlap_width / b_width) * b_cnt

        return total

    def _bake_vectorized_data(self):
        self.b_lo = np.array([b.lo for b in self.buckets], dtype=np.float64)
        self.b_hi = np.array([b.hi for b in self.buckets], dtype=np.float64)
        self.b_count = np.array([b.count for b in self.buckets], dtype=np.float64)
        self.b_width = self.b_hi - self.b_lo + 1
