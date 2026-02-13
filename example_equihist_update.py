from models import Bucket, EquiHistLearner, RangeQuery

def demonstrate_update():
    # 1. Setup Initial Histogram (3 buckets of width 10, initially uniform count=100)
    buckets = [
        Bucket(0, 9, count=100),
        Bucket(10, 19, count=100),
        Bucket(20, 29, count=100)
    ]
    
    learner = EquiHistLearner(buckets, learning_rate=0.5)
    
    print("--- Initial State ---")
    for i, b in enumerate(learner.buckets):
        print(f"B{i} [{b.lo}, {b.hi}]: count={b.count}")
        
    # 2. Define Query spanning buckets 0 and 1
    # Query: [5, 14] -> Covers [5,9] of B0 (50% overlap) and [10,14] of B1 (50% overlap)
    q = RangeQuery(5, 14)
    
    # 3. Predict
    predicted = learner.predict(q)
    print(f"\nQuery {q}: Predicted = {predicted:.2f}")
    # Calculation: 
    # B0 overlap: 5/10 * 100 = 50
    # B1 overlap: 5/10 * 100 = 50
    # Total = 100
    
    # 4. Update with "Actual" feedback (Assume actual data is 2x denser here)
    actual = 200.0
    print(f"Feedback: Actual = {actual:.2f} (Ratio = {actual/predicted:.2f})")
    
    learner.update(q, actual)
    
    # 5. Check New State
    print("\n--- Updated State ---")
    for i, b in enumerate(learner.buckets):
        print(f"B{i} [{b.lo}, {b.hi}]: count={b.count}")

if __name__ == "__main__":
    demonstrate_update()
