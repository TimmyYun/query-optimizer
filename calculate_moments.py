import sys
import os
import numpy as np

# Add current directory to path
sys.path.append(os.getcwd())

from optimizer.datasets import gen_values

def calc_moments(data):
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0, 0.0
    
    # Standardized moments
    z = (data - mean) / std
    skewness = np.mean(z**3)
    kurtosis = np.mean(z**4) # Pearson kurtosis (Normal = 3)
    
    return skewness, kurtosis

def main():
    rng = np.random.default_rng(42)
    rows = 1_000_000
    lo = 0
    hi = 200_000
    
    distributions = [
        ("Uniform (Baseline)", "uniform"),
        ("Normal", "normal"),
        ("Zipfian (alpha=2.0)", "zipf"),
        ("Sparse Cluster", "sparse_cluster"),
        ("Anti-Zipf", "anti_zipf")
    ]
    
    print(f"{'Dataset':<25} {'Skewness':<15} {'Kurtosis':<15} {'Distribution Characteristic'}")
    print("-" * 80)
    
    for name, dist_key in distributions:
        data = gen_values(rng, dist_key, rows, lo, hi)
        skew, kurt = calc_moments(data)
        
        char = ""
        if dist_key == "uniform": char = "Platykurtic (Flat)"
        elif dist_key == "normal": char = "Mesokurtic (Symmetric)"
        elif dist_key == "zipf": char = "Highly Leptokurtic (Heavy-Tail)"
        elif dist_key == "sparse_cluster": char = "Multimodal / Gappy"
        elif dist_key == "anti_zipf": char = "Piecewise / Non-Smooth"
        
        print(f"{name:<25} {skew:<15.4f} {kurt:<15.4f} {char}")

if __name__ == "__main__":
    main()
