import numpy as np
import time
from models import HybridEstimator, Bucket, RangeQuery

def debug_zipf():
    print("Generating 100k Zipf Data settings...")
    # Use same settings as main.py but smaller
    # mn=1, mx=100000
    N = 100000
    # Zipf generation
    # a=1.1 to be safe (a=1.0 is strictly not defined in numpy, needs >1)
    a = 1.1 
    data = np.random.zipf(a, N)
    # Clip to reasonable range to prevent OOM
    data = np.clip(data, 1, N) # Clip to unique values = N (worst case)
    
    mn = data.min()
    mx = data.max()
    print(f"Data Range: {mn} - {mx}")
    print(f"Top 5 values: {data[:5]}")
    
    # Calculate True Frequencies
    freq = np.bincount(data, minlength=mx+1)
    # freq[0] is count of 0 (which is 0 for zipf). Start from mn.
    # Adjust freq to be 0-indexed relative to mn?
    # models.py expects freq array where freq[i] is count of (mn + i)
    # So we slice freq[mn:]
    freq = freq[mn:]
    
    print(f"Freq[0] (Count of {mn}): {freq[0]}")
    print(f"Freq[1] (Count of {mn+1}): {freq[1]}")
    
    # Create Buckets (Simple Fixed Width for Replicability)
    # Create 10 bins.
    # Bucket 0: [mn, mn+9]
    # ...
    bins = 10
    w = int(np.ceil((mx - mn + 1) / bins))
    buckets = []
    curr = mn
    for _ in range(bins):
        end = min(mx, curr + w - 1)
        cnt = np.sum(freq[curr-mn : end-mn+1])
        buckets.append(Bucket(curr, end, int(cnt)))
        curr = end + 1
        if curr > mx: break
        
    print(f"Created {len(buckets)} buckets.")
    print(f"Bucket 0: {buckets[0]}")
    
    # Initialize Hybrid
    hybrid = HybridEstimator(buckets)
    
    # Train
    print("Training Hybrid...")
    rng = np.random.default_rng(42)
    hybrid.train(freq, mn, points_per_bucket=200, rng=rng)
    
    # Predict
    print("\n--- Prediction Checks ---")
    
    # Check [1, 1] (Head)
    q_head = RangeQuery(mn, mn)
    pred_head = hybrid.predict(q_head)
    true_head = freq[0]
    print(f"Query [{mn}, {mn}]: True={true_head}, Pred={pred_head:.2f}, Error={abs(true_head - pred_head)}")
    
    # Check [1, 2]
    q_head2 = RangeQuery(mn, mn+1)
    pred_head2 = hybrid.predict(q_head2)
    true_head2 = freq[0] + freq[1]
    print(f"Query [{mn}, {mn+1}]: True={true_head2}, Pred={pred_head2:.2f}, Error={abs(true_head2 - pred_head2)}")
    
    # Check Tail
    tail_val = (mn + mx) // 2
    q_tail = RangeQuery(tail_val, tail_val)
    pred_tail = hybrid.predict(q_tail)
    true_tail = freq[tail_val - mn]
    print(f"Query Tail [{tail_val}, {tail_val}]: True={true_tail}, Pred={pred_tail:.2f}")

if __name__ == "__main__":
    debug_zipf()
