from typing import List
import numpy as np
import math
from ..core import Bucket, RangeQuery

def freedman_diaconis_bins(sample: np.ndarray, mn: int, mx: int, n_rows: int, bins_max: int) -> int:
    """
    Calculates the optimal number of bins using the Freedman-Diaconis rule.
    
    The FD rule uses the Interquartile Range (IQR) to estimate the optimal bin width,
    which is robust to outliers.
    
    Formula: Bin Width = 2 * IQR * n^(-1/3)
    
    Args:
        sample: A sample of values from the dataset.
        mn: Minimum value in the dataset.
        mx: Maximum value in the dataset.
        n_rows: Total number of rows in the dataset.
        bins_max: Maximum allowed number of bins.
    
    Returns:
        The calculated number of bins, clamped between 1 and bins_max.
    """
    if sample.size < 10: return 10
    q25, q75 = np.quantile(sample, [0.25, 0.75])
    iqr = q75 - q25
    if iqr <= 0: return 10
    
    bin_width = 2 * iqr / (n_rows ** (1/3))
    total_width = mx - mn
    if bin_width <= 0: return 10
    
    bins = int(total_width / bin_width)
    return max(1, min(bins, bins_max))

def make_equiwidth_buckets(mn: int, mx: int, bins: int, freq: np.ndarray, ndv_threshold: int = 200) -> List[Bucket]:
    """
    Constructs equi-width buckets from a frequency array.
    
    This function divides the domain [mn, mx] into `bins` equal-width intervals.
    It also identifies "sparse" buckets (low Number of Distinct Values) and stores
    exact value-frequency pairs for them, which is a key part of the Hybrid approach.
    
    Args:
        mn: Minimum domain value.
        mx: Maximum domain value.
        bins: Number of buckets to create.
        freq: Frequency array where freq[i] is the count of value (mn + i).
        ndv_threshold: Threshold for storing specific values (sparse bucket).
    
    Returns:
        A list of Bucket objects.
    """
    if len(freq) == 0: return []
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
    return buckets

def predict_range_histogram_uniform(q: RangeQuery, buckets: List[Bucket]) -> float:
    """
    Estimates range query selectivity using the Uniform Spread assumption.
    
    For each bucket overlapping with the query range, it assumes the tuples
    are uniformly distributed within the bucket.
    
    Selectivity = Sum(fraction_overlap * bucket_count)
    """
    total = 0.0
    for b in buckets:
        ov_lo = max(q.low, b.lo)
        ov_hi = min(q.high, b.hi)
        if ov_lo <= ov_hi:
            w = b.hi - b.lo + 1
            # Fraction of the bucket covered by the query
            frac = (ov_hi - ov_lo + 1) / w
            total += frac * b.count
    return total
