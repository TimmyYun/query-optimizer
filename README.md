# Thesis: Hybrid Selectivity Estimation using CDF Learning

This project implements and benchmarks a **Hybrid Approach** for selectivity estimation in databases. It combines the speed and safety of **Equi-Width Histograms** with the precision of **Machine Learning** by learning the Cumulative Distribution Function (CDF) within individual histogram buckets.

## 1. The Core Idea

Standard **Equi-Width Histograms** assume data is **Uniform** within a bucket. This fails catastrophically for skewed data (e.g., Zipfian frequency), leading to massive errors in query planning.

**Our Solution**:
Instead of assuming uniformity, we train a lightweight **Machine Learning Model** (Ridge Regression) for *each bucket* to learn the actual 1D Cumulative Distribution Function ($F_b(x) \approx P(X \le x)$). This reduces complex non-linear estimation to a simple monotonic learning task.

## 2. Methodology

- **Automated Binning**: Uses the **Freedman-Diaconis Rule** ($Width = 2 \cdot IQR \cdot n^{-1/3}$) to automatically determine the optimal number of bins based on data scale and variability.
- **CDF Learning**: Ridge regressors learn the intra-bucket distribution, enabling precise range queries:
  $$Count = \sum_{b \in full} Count(b) + (F_{start}(R) - F_{start}(L)) \cdot Count(start)$$
- **Adaptive Drifting**: Includes an **EquiHist** implementation that updates in real-time as new data is inserted, and a **Hybrid Repair** mechanism that triggers retraining when Q-Error exceeds a safety threshold.

## 3. Project Structure

- `main.py`: Unified entry point for all benchmarks.
- `datasets/`: Data generation, loading, and standardized statistics logic.
- `models/`: Implementations of Equi-Width, Equi-Hist (Adaptive), and Hybrid (ML-based) estimators.
- `plots/`: Automatically generated visualizations of dataset distributions.
- `experiments/`: Legacy scripts and specialized research benchmarks.

## 4. Usage

The project uses `poetry` for dependency management.

### Static Benchmark
Runs a batch experiment across multiple distributions (Uniform, Normal, Zipf, Exponential, Lognormal) to evaluate peak accuracy.

```bash
poetry run python main.py --mode static --rows 1000000 --eval-n 1000
```

### Drift Benchmark
Simulates real-world data drift by starting with one distribution and "drifting" into another through high-volume inserts.

```bash
poetry run python main.py --mode drift --rows 1000000 --drift-rows 200000
```

### Configuration Options
- `--rows`: Number of initial rows.
- `--eval-n`: Number of range queries to evaluate.
- `--recreate`: Force regenerate datasets (bypassing cache).
- `--experiment-name`: Suffix for all output artifacts.

## 5. Outputs & Visualization

All benchmark runs generate structured artifacts in the `artifacts_optimizer/` directory (or your specified `--out-dir`):

- **Excel Reports**: `static_benchmark_results.xlsx` contains granular metrics including Medians, P95 Q-Errors, and Build Timings.
- **Global Statistics**: `datasets/dataset_statistics.xlsx` is automatically refreshed with Skewness, Kurtosis, and NDV data.
- **Distribution Plots**: `plots/{dist}.png` provides immediate visual feedback on the data being tested.
- **Error Visualization**: `experiment_boxplots.png` visualizes the Q-Error distribution across all models.
