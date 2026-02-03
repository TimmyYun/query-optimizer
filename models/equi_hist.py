from typing import List
from models.core import Bucket, RangeQuery

class EquiHistLearner:
    """
    Implements a self-tuning histogram baseline (EquiHist-like).
    
    This learner adapts bucket counts based on feedback from actual query execution.
    It uses a Multiplicative Weight Update (MWU) rule to decrease or increase
    bucket counts to match the observed selectivity.
    """
    def __init__(self, buckets: List[Bucket], learning_rate: float = 0.5):
        # Create a deep copy of buckets to maintain independent state
        self.buckets = [Bucket(b.lo, b.hi, count=b.count) for b in buckets]
        self.lr = learning_rate
        
    def predict(self, q: RangeQuery) -> float:
        """
        Predicts selectivity using the current state of buckets and Uniform assumption.
        """
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
        """
        Updates bucket counts based on the error between prediction and actual value.
        
        Uses an exponential update rule:
        NewCount = OldCount * (Actual/Pred) ^ (learning_rate * overlap_fraction)
        """
        pred = self.predict(q)
        if pred == 0: return
        
        ratio = actual / pred
        # Clamp ratio to prevent extreme updates
        ratio = max(0.1, min(10.0, ratio))
        
        for b in self.buckets:
            ov_lo = max(q.low, b.lo)
            ov_hi = min(q.high, b.hi)
            if ov_lo <= ov_hi:
                w = b.hi - b.lo + 1
                overlap = (ov_hi - ov_lo + 1) / w
                
                # Update factor scales by how much this bucket contributed to the query
                factor = ratio ** (self.lr * overlap)
                b.count = int(max(1, b.count * factor))
