from typing import List
from .core import Bucket, RangeQuery

class EquiHistLearner:
    """
    Implements a self-tuning histogram baseline.
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
        
    def update(self, q: RangeQuery, actual: float):
        pred = self.predict(q)
        if pred == 0: return
        
        ratio = actual / pred
        ratio = max(0.1, min(10.0, ratio))
        
        for b in self.buckets:
            ov_lo = max(q.low, b.lo)
            ov_hi = min(q.high, b.hi)
            if ov_lo <= ov_hi:
                w = b.hi - b.lo + 1
                overlap = (ov_hi - ov_lo + 1) / w
                factor = ratio ** (self.lr * overlap)
                b.count = int(max(1, b.count * factor))
