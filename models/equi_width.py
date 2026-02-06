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
    def build(mn: int, mx: int, bins: int, freq: np.ndarray, prefix_buckets: int = 0, suffix_buckets: int = 0) -> 'EquiWidthHistogram':
        """
        Constructs an Equi-Width Histogram from a frequency array.
        
        Args:
            mn: Minimum domain value.
            mx: Maximum domain value.
            bins: Number of buckets to create.
            freq: Frequency array where freq[i] is the count of value (mn + i).
            prefix_buckets: Number of exact buckets to create for smallest distinct values.
            suffix_buckets: Number of exact buckets to create for largest distinct values.

        Returns:
            A new EquiWidthHistogram instance.
        """
        if len(freq) == 0: return EquiWidthHistogram([])
        
        # Identify indices with non-zero frequency
        distinct_indices = np.where(freq > 0)[0]
        
        pb: List[Bucket] = []
        sb: List[Bucket] = []
        
        # 1. Handle Prefix Buckets
        p_count = min(prefix_buckets, len(distinct_indices))
        for i in range(p_count):
            idx = distinct_indices[i]
            val = mn + idx
            pb.append(Bucket(val, val, int(freq[idx])))
            
        # 2. Handle Suffix Buckets
        # Ensure we don't overlap with prefix
        remaining_indices = distinct_indices[p_count:]
        s_count = min(suffix_buckets, len(remaining_indices))
        
        # Suffix are the last s_count indices
        for i in range(s_count):
            # Take from the end
            idx = remaining_indices[-(s_count - i)] 
            val = mn + idx
            sb.append(Bucket(val, val, int(freq[idx])))
            
        # 3. Handle Middle Range
        # Indices involved in prefix/suffix are handled. 
        # We need to build equi-width on the *range* not covered by prefix/suffix?
        # Actually, standard practice: remove the exact values from consideration, then run equi-width on the rest.
        # The range is from (last_prefix_val + 1) to (first_suffix_val - 1).
        
        remaining_bins = bins - len(pb) - len(sb)
        mid_buckets = []
        
        if remaining_bins > 0 and len(remaining_indices) > s_count:
            # Determine middle range
            # Start after the last prefix index
            start_idx = distinct_indices[p_count-1] + 1 if p_count > 0 else 0
            # End before the first suffix index
            end_idx = remaining_indices[-(s_count + 1)] if s_count > 0 else len(freq) - 1
            
            # Adjust range to be relative to mn (for freq access)
            # Actually, simplify: define new min/max for the equi-width build
            # The freq array needs to be sliced or indexed carefully.
            
            # Slice freq for the middle range
            mid_freq = freq[start_idx : end_idx + 1]
            mid_mn = mn + start_idx
            mid_mx = mn + end_idx
            
            if len(mid_freq) > 0 and mid_mx >= mid_mn:
                 width = mid_mx - mid_mn + 1
                 bw = max(1, int(math.ceil(width / remaining_bins)))
                 
                 # Pre-calculate cumulative sum for fast count aggregation
                 ps = np.cumsum(mid_freq)
                 
                 cur = mid_mn
                 for _ in range(remaining_bins):
                     lo = cur
                     hi = min(mid_mx, lo + bw - 1)
                     li = lo - mid_mn
                     ri = hi - mid_mn
                     
                     if li < 0: li = 0
                     if ri >= len(mid_freq): ri = len(mid_freq) - 1
                     
                     if ri >= li:
                         # Calculate total count in this bucket
                         cnt = int(ps[ri] - (ps[li-1] if li > 0 else 0))
                         if cnt > 0: # Only create non-empty buckets? Standard Equi-Width creates all. Keeping consistent.
                             pass
                         mid_buckets.append(Bucket(lo, hi, count=cnt))
                     
                     cur = hi + 1
                     if cur > mid_mx: break
                     
        return EquiWidthHistogram(pb + mid_buckets + sb)

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
