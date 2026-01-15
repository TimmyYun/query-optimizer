from typing import List
import numpy as np
import math
from .core import Bucket

def freedman_diaconis_bins(sample: np.ndarray, mn: int, mx: int, n_rows: int, bins_max: int) -> int:
    if sample.size < 10: return 10
    q25, q75 = np.quantile(sample, [0.25, 0.75])
    iqr = q75 - q25
    if iqr <= 0: return 10
    
    bin_width = 2 * iqr / (n_rows ** (1/3))
    total_width = mx - mn
    if bin_width <= 0: return 10
    
    bins = int(total_width / bin_width)
    return max(1, min(bins, bins_max))

def make_equiwidth_buckets(mn: int, mx: int, bins: int, freq: np.ndarray) -> List[Bucket]:
    if len(freq) == 0: return []
    width = mx - mn + 1
    bw = max(1, int(math.ceil(width / bins)))
    buckets = []
    ps = np.cumsum(freq)
    
    EXACT_STORAGE_THRESHOLD = 200
    
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
