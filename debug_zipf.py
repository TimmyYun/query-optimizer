import numpy as np
import time
import argparse
from models import HybridEstimator, Bucket, RangeQuery
import sys

def debug_zipf():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100000, help="Number of rows")
    args = parser.parse_args()
    
    N = args.rows
    print(f"Generating {N} Zipf Data rows...")
    
    # Zipf generation
    a = 1.1 
    # Generate in chunks to avoid memory issues if N is large
    if N > 10000000:
        print("Large N detected, generating in chunks...")
        data = np.zeros(N, dtype=np.int64)
        chunk_size = 1000000
        for i in range(0, N, chunk_size):
            sz = min(chunk_size, N - i)
            data[i:i+sz] = np.random.zipf(a, sz)
    else:
        data = np.random.zipf(a, N)
        
    # Clip to unique values = N (worst case)
    data = np.clip(data, 1, N) 
    
    mn = int(data.min())
    mx = int(data.max())
    print(f"Data Range: {mn} - {mx}")
    print(f"Top 5 values: {data[:5]}")
    
    print("Calculating frequencies...")
    # freq array
    # freq[0] is count of mn.
    # bincount needs positive integers. data is >= 1.
    freq = np.bincount(data)
    # slice to start from mn
    freq = freq[mn:]
    
    print(f"Freq[0] (Count of {mn}): {freq[0]}")
    if len(freq) > 1:
        print(f"Freq[1] (Count of {mn+1}): {freq[1]}")
    
    # Create Buckets (Simple Fixed Width)
    # Target ~2000 bins for 10M rows, which is standard in main.py
    # Scale bins based on N: 10M -> 2000. 100k -> 20 (to keep bucket width somewhat comparable or ratio fixed)
    # Actually main.py uses FD rule or fixed. Let's use fixed 2000 for 10M, scaled linearly.
    bins = max(10, int(2000 * (N / 10000000)))
    print(f"Using {bins} bins.")
    
    w = int(np.ceil((mx - mn + 1) / bins))
    if w < 1: w = 1
    
    buckets = []
    curr = mn
    for _ in range(bins):
        end = min(mx, curr + w - 1)
        # Slicing numpy array
        f_idx_start = curr - mn
        f_idx_end = end - mn + 1
        
        if f_idx_start >= len(freq):
            cnt = 0
        else:
            cnt = np.sum(freq[f_idx_start : f_idx_end])
            
        buckets.append(Bucket(curr, end, int(cnt)))
        curr = end + 1
        if curr > mx: break
        
    print(f"Created {len(buckets)} buckets.")
    print(f"Bucket 0: {buckets[0]}")
    
    # Initialize Hybrid
    hybrid = HybridEstimator(buckets)
    
    # Monkeypatch detailed training inspector
    original_collect = hybrid._collect_cdf_training_rows
    
    def debug_collect_wrapper(freq_arr, min_val, pp_bucket, rng_gen, b_indices=None):
        X_dict = original_collect(freq_arr, min_val, pp_bucket, rng_gen, b_indices)
        # X_dict is Dict[bucket_id, List[CDFTrainRow]] - Wait, models.py returns Dict
        # Let's inspect bucket 0 if it exists
        if 0 in X_dict:
            rows_list = X_dict[0]
            # Convert to numpy for inspection. CDFTrainRow is a named tuple or object?
            # It's a named tuple (x_norm, y_cdf) defined in models.py?
            # No, it's just a class or named tuple. 
            # We assume it has x_norm.
            # We need to reconstruct "real X" from x_norm to see if it matches checking 1, 2, 3..
            # x_norm = (x - lo) / width
            # x = x_norm * width + lo
            b0 = hybrid.buckets[0]
            width = b0.hi - b0.lo + 1
            
            x_norms = np.array([r.x_norm for r in rows_list])
            xs = x_norms * width + b0.lo
            
            print(f"\n[Bucket 0 Training Data Inspection]")
            print(f"  Points collected: {len(xs)}")
            print(f"  X head (first 20 sorted): {np.sort(xs)[:20]}")
            
            has_1 = np.any(np.isclose(xs, 1, atol=0.1))
            has_2 = np.any(np.isclose(xs, 2, atol=0.1))
            has_3 = np.any(np.isclose(xs, 3, atol=0.1))
            has_10 = np.any(np.isclose(xs, 10, atol=0.1))
            has_50 = np.any(np.isclose(xs, 50, atol=0.1))
            
            print(f"  Contains 1? {has_1}")
            print(f"  Contains 2? {has_2}")
            print(f"  Contains 3? {has_3}")
            print(f"  Contains 10? {has_10}")
            print(f"  Contains 50? {has_50}")
            
        return X_dict
        
    hybrid._collect_cdf_training_rows = debug_collect_wrapper
    
    # Train
    print("Training Hybrid...")
    rng = np.random.default_rng(42)
    # points relative to bucket width? defaults to 200
    hybrid.train(freq, mn, points_per_bucket=200, rng=rng)
    
    # Predict & Eval
    print("\n--- Prediction Checks ---")
    
    check_points = [0, 1, 2, 9, 99, 4999] # Offsets from mn
    
    for offset in check_points:
        val = mn + offset
        if val > mx: continue
        
        q = RangeQuery(mn, val) # Range [mn, val] (Cumulative)
        # But we want point query for error checking usually?
        # main.py does range queries.
        # But here let's check P(val) = CDF(val) - CDF(val-1) or just count of val.
        # Hybrid pred returns count in range.
        
        # Point Query
        q_point = RangeQuery(val, val)
        est = hybrid.predict(q_point)
        
        if (val - mn) < len(freq):
            act = freq[val - mn]
        else:
            act = 0
            
        err = abs(est - act)
        q_err = max(est/act, act/est) if est > 0 and act > 0 else 0
        
        print(f"Val {val}: Act={act}, Est={est:.2f}, AbsErr={err:.2f}, QErr={q_err:.2f}")

if __name__ == "__main__":
    debug_zipf()
