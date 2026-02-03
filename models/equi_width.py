from typing import List
import numpy as np
import math
from models.core import Bucket, RangeQuery

class EquiWidthHistogram:
    """
    Implements a standard Equi-Width Histogram.
    
    This histogram divides the domain into fixed-width buckets. It supports
    exact value storage for sparse buckets (Hybrid approach logic) if generated via `build`.
    """
    
    def __init__(self, buckets: List[Bucket]):
        """
        Initializes the histogram with a list of buckets.
        
        Args:
           buckets: List of Bucket objects covering the domain.
        """
        self.buckets = buckets
        
    @staticmethod
    def build(mn: int, mx: int, bins: int, freq: np.ndarray, ndv_threshold: int = 200) -> 'EquiWidthHistogram':
        """
        Constructs an Equi-Width Histogram from a frequency array.
        
        Args:
            mn: Minimum domain value.
            mx: Maximum domain value.
            bins: Number of buckets to create.
            freq: Frequency array where freq[i] is the count of value (mn + i).
            ndv_threshold: Threshold for storing specific values (sparse bucket) logic.
        
        Returns:
            A new EquiWidthHistogram instance.
        """
        if len(freq) == 0: return EquiWidthHistogram([])
        width = mx - mn + 1
        # Calculate bucket width (bw)
        bw = max(1, int(math.ceil(width / bins)))
        buckets = []
        # Pre-calculate cumulative sum for fast count aggregation
        ps = np.cumsum(freq)
        
        EXACT_STORAGE_THRESHOLD = ndv_threshold
        
        cur = mn
        for _ in range(bins):
            lo = cur
            hi = min(mx, lo + bw - 1)
            li = lo - mn
            ri = hi - mn
            if li < 0: li=0 
            if ri >= len(freq): ri = len(freq)-1
            
            # Calculate total count in this bucket using prefix sums
            cnt = int(ps[ri] - (ps[li-1] if li > 0 else 0))
            
            # Determine NDV (Number of Distinct Values) in this bucket
            freq_slice = freq[li : ri+1]
            ndv = np.count_nonzero(freq_slice)
            
            b = Bucket(lo, hi, count=cnt, ndv=ndv)
            
            # Hybrid Logic: If NDV is low, store exact values for 100% accuracy
            if ndv > 0 and ndv <= EXACT_STORAGE_THRESHOLD:
                rel_indices = np.nonzero(freq_slice)[0]
                vals_with_counts = []
                for idx in rel_indices:
                     vals_with_counts.append((int(idx + lo), int(freq_slice[idx])))
                b.exact_values = vals_with_counts
                
            buckets.append(b)
            
            cur = hi + 1
            if cur > mx: break
            
        return EquiWidthHistogram(buckets)

    def predict(self, q: RangeQuery) -> float:
        """
        Estimates range query selectivity using the Uniform Spread assumption.
        
        Selectivity = Sum(fraction_overlap * bucket_count)
        """
        total = 0.0
        for b in self.buckets:
            ov_lo = max(q.low, b.lo)
            ov_hi = min(q.high, b.hi)
            if ov_lo <= ov_hi:
                w = b.hi - b.lo + 1
                # Fraction of the bucket covered by the query
                frac = (ov_hi - ov_lo + 1) / w
                total += frac * b.count
        return total
