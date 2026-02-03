from typing import List
import numpy as np
import math
from .core import Bucket

def freedman_diaconis_bins(sample: np.ndarray, mn: int, mx: int, n_rows: int, bins_max: int) -> int:
    """
    Estimate a reasonable number of histogram bins using the
    Freedman–Diaconis rule.

    The idea:
      - Use a small sample of the data (not the full column).
      - Estimate bin width from data variability (IQR).
      - Convert bin width into number of bins over [mn, mx].

    This avoids hard-coded "magic numbers" for bin count and adapts
    automatically to different data distributions.
    """
    
    if sample.size < 10: return 10
    q25, q75 = np.quantile(sample, [0.25, 0.75])

    # Interquartile range (IQR)
    iqr = q75 - q25
    if iqr <= 0: return 10
    

   # Freedman–Diaconis bin width:
    #   width = 2 * IQR / n^(1/3)
    # Larger datasets -> smaller bins
    bin_width = 2 * iqr / (n_rows ** (1/3))
    total_width = mx - mn
    if bin_width <= 0: return 10
    
    # Convert width into number of bins
    bins = int(total_width / bin_width)
    return max(1, min(bins, bins_max))



def make_equiwidth_buckets(mn: int, mx: int, bins: int, freq: np.ndarray, ndv_threshold: int = 200) -> List[Bucket]:
    """
    Construct an equi-width histogram over the value range [mn, mx].

    Each bucket covers an equal-width interval and stores:
      - lo, hi : bucket boundaries
      - count  : number of rows in the bucket
      - ndv    : number of distinct values in the bucket

    Optimization for sparse buckets:
      If a bucket has very few distinct values (ndv <= ndv_threshold),
      we store exact (value, count) pairs instead of assuming uniformity.
      This avoids large errors on gappy or clustered distributions.
    """

    # Edge case: empty histogram
    if len(freq) == 0: return []

    # Domain width
    width = mx - mn + 1

    # Bucket width
    bw = max(1, int(math.ceil(width / bins)))
    buckets = []

    # Prefix sum of frequencies for fast range counting
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
        
        cnt = int(ps[ri] - (ps[li-1] if li > 0 else 0))
        
        freq_slice = freq[li : ri+1]
        ndv = np.count_nonzero(freq_slice)
        
        b = Bucket(lo, hi, count=cnt, ndv=ndv)
        

        # Sparse bucket: store exact values
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
