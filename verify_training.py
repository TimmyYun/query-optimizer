import numpy as np
from models import HybridEstimator, Bucket

def verify_training():
    print("Generating Zipf Data (Clamped)...")
    N = 1000000
    a = 1.5
    raw_data = np.random.zipf(a, N)
    # Clamp to [0, 200000]
    data = np.clip(raw_data, 0, 200000)
    
    # Freq array size 200001
    freq = np.bincount(data, minlength=200001)
    mn = 0
    mx = 200000
    
    print(f"Data Rows: {N}, Domain: [{mn}, {mx}]")
    
    # Create Buckets (Mimic Equi-Width with 2000 bins - standard FD)
    n_buckets = 2000
    width = (mx - mn + 1) // n_buckets # 200000 / 2000 = 100
    buckets = []
    curr = mn
    for i in range(n_buckets):
        hi = min(mx, curr + width - 1)
        if hi < curr: break
        # Calculate count
        # Handle slice bounds carefully
        sl = freq[curr : hi+1]
        cnt = np.sum(sl)
        b = Bucket(curr, hi, cnt)
        buckets.append(b)
        curr = hi + 1

        
    print(f"Created {len(buckets)} buckets.")
    print(f"Bucket 0: [{buckets[0].lo}, {buckets[0].hi}], Count: {buckets[0].count}")
    
    # Train Hybrid
    print("Training Hybrid Estimator...")
    hybrid = HybridEstimator(buckets)
    rng = np.random.default_rng(42)
    hybrid.train(freq, mn, points_per_bucket=50, rng=rng)
    
    # Inspect Models
    print(f"Models trained for {len(hybrid.models)} buckets.")
    
    if 0 in hybrid.models:
        m, name = hybrid.models[0]
        print(f"Bucket 0 Model Selected: {name}")
        print(f"Model Object: {m}")
        
        # Test Prediction
        q_low, q_high = 1, 100
        pred = hybrid.predict_one(1, 100) # Wait, predict takes RangeQuery object usually? 
        # checking models.py signature: predict(q)
        class MockQuery:
            def __init__(self, l, h): self.low=l; self.high=h
        
        pred = hybrid.predict(MockQuery(1, 100))
        actual = np.sum(freq[1:101])
        print(f"Query [1, 100]: Pred={pred:.2f}, Act={actual}, Q-Err={max(pred/actual, actual/pred):.2f}")
    else:
        print("Bucket 0 has NO MODEL (Fallback to Uniform).")
        # Check MSE comparison logs if possible (need to enable prints in models.py)

if __name__ == "__main__":
    verify_training()
