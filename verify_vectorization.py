import numpy as np
import time
from models import Bucket, HybridEstimator
from workload import RangeQuery

def test_vectorization():
    # Setup dummy data
    mn, mx = 0, 1000
    buckets = [Bucket(i*10, (i+1)*10-1, 100) for i in range(100)]
    
    # Create models for some buckets
    models = {
        0: None, # Identity
        1: ("linear", 0.5, 0.1),
        2: ("poly", np.array([0.1, 0.2]), 0.05),
    }
    
    hybrid = HybridEstimator(buckets, models)
    hybrid._bake_vectorized_data() # Ensure it's baked
    
    # Random queries
    n_queries = 1000000
    queries = []
    for _ in range(n_queries):
        l = np.random.randint(0, 900)
        h = np.random.randint(l, 1000)
        queries.append(RangeQuery(l, h))
        
    # Scalar Predict
    t0 = time.perf_counter()
    y_scalar = [hybrid.predict(q) for q in queries]
    t_scalar = time.perf_counter() - t0
    
    # Vectorized Predict
    t1 = time.perf_counter()
    y_vector = hybrid.predict_batch(queries)
    t_vector = time.perf_counter() - t1
    
    # Compare
    diff = np.abs(np.array(y_scalar) - y_vector)
    max_diff = np.max(diff)
    
    print(f"Results match: {max_diff < 1e-9}")
    print(f"Max difference: {max_diff}")
    print(f"Scalar time: {t_scalar:.4f}s")
    print(f"Vector time: {t_vector:.4f}s")
    print(f"Speedup: {t_scalar / t_vector:.1f}x")

if __name__ == "__main__":
    test_vectorization()
