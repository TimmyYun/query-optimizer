import numpy as np
from models.equi_width import EquiWidthHistogram

def test():
    print("=== Testing EquiWidthHistogram with Prefix/Suffix ===")
    
    # Domain: 0..9 (size 10)
    # Frequencies: [5, 1, 0, 0, 1, 1, 0, 0, 1, 5]
    # Distinct Values: 0, 1, 4, 5, 8, 9
    freq = np.array([5, 1, 0, 0, 1, 1, 0, 0, 1, 5])
    mn, mx = 0, 9
    bins = 4
    
    # Expected:
    # Prefix=1 -> Value 0 (Count 5)
    # Suffix=1 -> Value 9 (Count 5)
    # Middle Range -> 1..8. Count 1+1+1+1=4.
    # Middle Bins = 2. Width = ceil(8/2) = 4.
    # Middle Bucket 1: 1..4 (Contains 1, 4). Count 2.
    # Middle Bucket 2: 5..8 (Contains 5, 8). Count 2.
    
    print(f"Input: Freq={freq}, Bins={bins}, P=1, S=1")
    ew = EquiWidthHistogram.build(mn, mx, bins, freq, prefix_buckets=1, suffix_buckets=1)
    
    print(f"Result Buckets: {len(ew.buckets)}")
    for i, b in enumerate(ew.buckets):
        print(f"  Bucket {i}: {b}")
        
    # Assertions
    assert len(ew.buckets) == 4, f"Expected 4 buckets, got {len(ew.buckets)}"
    
    b0 = ew.buckets[0]
    assert b0.lo == 0 and b0.hi == 0 and b0.count == 5, "Bucket 0 incorrect"
    
    b_last = ew.buckets[-1]
    assert b_last.lo == 9 and b_last.hi == 9 and b_last.count == 5, "Last bucket incorrect"
    
    # Check middle sum
    mid_count = sum(b.count for b in ew.buckets[1:-1])
    assert mid_count == 4, f"Middle count expected 4, got {mid_count}"

    print("\n✅ Test Passed")

if __name__ == "__main__":
    test()
