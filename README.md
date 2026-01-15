# Thesis: Hybrid Selectivity Estimation using CDF Learning

This project implements and benchmarks a **Hybrid Approach** for selectivity estimation in databases. It combines the speed and safety of **Equi-Width Histograms** with the precision of **Machine Learning**.

## 1. The Core Idea

Standard **Equi-Width Histograms** divide data into $N$ buckets of equal width.
*   **Pros**: Extremely fast to build ($O(1)$ updates), low memory.
*   **Cons**: Within a bucket, they assume the data is **Uniform**. This fails catastrophically for skewed data (e.g., Zipfian frequency), leading to massive errors in query planning.

**Our Solution**:
Instead of assuming uniformity inside a bucket, we train a lightweight **Machine Learning Model** (Ridge Regression) for *each bucket* to learn the actual distribution.

## 2. Methodology

### A. Binning Strategy
We rely on standard, robust statistical methods:
1.  **Equi-Width**: We keep the simple equi-width boundaries. This ensures $O(1)$ lookups and $O(N)$ build time.
2.  **Freedman-Diaconis Rule**: We effectively automate the "number of bins" selection using the FD rule ($Width = 2 \cdot IQR \cdot n^{-1/3}$), removing the need for magic numbers.

### B. The ML Difference: CDF Learning
Previous approaches tried to predict "overlap percentage" from "query range features" (4 dimensions). This was noisy.
We innovated by learning the **Cumulative Distribution Function (CDF)**:
*   Inside each bucket $b$, we learn a function $F_b(x) \approx P(X \le x)$.
*   This reduces the problem to learning a **1D monotonic function**, which is much easier for simple regressors.
*   **Inference**: To estimate selectivity for range $[L, R]$, we simply compute:
    $$Count = \sum_{b \in full} Count(b) + (F_{start}(R_{clipped}) - F_{start}(L_{clipped})) \cdot Count(start)$$

## 3. Key Results

We extensively benchmarked this against standard histograms on **TPC-H** (Real-world Uniform) and **Zipf** (Synthetic Skewed) datasets.

### A. Massive Accuracy on Skew
On skewed data, when memory is limited (e.g., 20 bins), standard histograms fail.
*   **Standard Histogram MAE**: 0.0230 (2.3% error per query)
*   **Hybrid CDF MAE**: 0.000019 (0.0019% error per query)
*   **Improvement**: **1,200x Better**

### B. Safety on Uniformity
On TPC-H (which is very uniform), simple histograms are already perfect. Our model successfully learns the linear CDF and introduces **zero regression**.
*   **TPC-H Q-Error**: ~1.000 (Perfect) for both methods.

### C. Low Overhead
*   **Training**: Adds ~20-40ms to build time (negligible).
*   **Inference**: Adds ~30$\mu$s per query. Still fast enough for query optimization.

## 4. Usage
To replicate the benchmarks (including Drift and Adaptive Repair):

```bash
# 1. Run full experiment (Zipf + Destructive Drift + Repair)
python pipeline_fd_cdf.py --dist zipf --rows 1000000 --drift-rows 1500000 --drift-dist anti_zipf --eval-n 1000

# 2. View Results
# Check artifacts_fd_cdf/drift_summary.json
```

