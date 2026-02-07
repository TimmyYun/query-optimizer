# Query Optimizer Benchmark

A comprehensive benchmarking framework for evaluating query selectivity estimation and cardinality estimation models. This project separates dataset generation, workload creation, and benchmark execution to ensure reproducible and isolated experiments.

## 1. Overview

The benchmark workflow consists of three distinct stages:
1.  **Dataset Generation**: Creating synthetic datasets with specific distributions (Uniform, Normal, Zipf, etc.).
2.  **Workload Generation**: Creating query workloads (range queries) completely independent of the dataset data.
3.  **Benchmark Execution**: Running estimation models against the datasets using the generated workloads and saving detailed metrics.

## 2. Setup

Ensure you have Python 3.9+ and Poetry installed.

```bash
# Install dependencies
poetry install
```

## 3. Workflow Stages

### Stage 1: Dataset Generation (`datasets.py`)
This script generates the synthetic data used for benchmarks. It creates CSV files and statistics for various distributions and row counts.

**Usage:**
```bash
poetry run python datasets.py
```
*   **Output**: Creates `data/generated/{rows}/{distribution}/` directories containing `data.csv`, `meta.pkl`, `stats.json`, and distribution plots.
*   **Distributions**: Uniform, Normal, Zipf, Sparse Cluster, Anti-Zipf.
*   **Scales**: 1M, 10M, 60M rows (default).

### Stage 2: Workload Generation (`workload.py`)
This script generates independent workloads of range queries. These workloads are saved as CSV files and are domain-based (e.g., `[0, 200,000]`), ensuring they are not biased by the specific data values in the dataset.

**Usage:**
```bash
poetry run python workload.py
```
*   **Output**: Creates `workload/` directory containing:
    *   `1000.csv` (1k queries)
    *   `100000.csv` (100k queries)
    *   `1000000.csv` (1M queries)
*   **Format**: CSV with `query_id`, `low`, `high`, `type`.

### Stage 3: Benchmark Execution (`main.py`)
The main entry point for running experiments. It loads a dataset and a workload, filters out invalid queries (0-selectivity), trains models, and evaluates accuracy.

**Key Features:**
*   **Runtime Filtering**: Automatically filters out queries that have 0 true result cardinality to avoid skewed Q-Error metrics.
*   **Structured Output**: Saves detailed per-query results.

**Usage:**
```bash
# Run a static benchmark (10k rows, uniform distribution, 1k workload)
poetry run python main.py --mode static --rows 10000 --dist uniform --eval 1000
```

**Common Arguments:**
*   `--rows`: Number of rows in the dataset (must match a generated dataset size).
*   `--dist`: Distribution to test (`uniform`, `normal`, `zipf`, `sparse_cluster`, `anti_zipf`, or `all`).
*   `--eval`: Workload size to load (e.g., `1000`, `100000`).
*   `--mode`: `static` (standard benchmark) or `drift` (drift benchmark).

## 4. Output Structure

Results are saved in a hierarchical structure for easy analysis:

```
results/
├── {experiment_id}/           # Timestamped directory (e.g., 20231027_123045)
│   ├── {rows}/
│   │   ├── {distribution}/
│   │   │   ├── {workload_size}.csv        # Detailed results (Static)
│   │   │   ├── {workload_size}_drift.csv  # Detailed results (Drift)
│   │   │   └── summary.json               # Aggregated stats and timings
```

### Result CSV Columns
The CSV files provide granular data for every query:
*   `Query_ID`: Index of the query.
*   `Phase`: Experiment phase (e.g., "Static", "Drift", "Repair").
*   `Model`: Name of the model (e.g., "Equi-Width", "Hybrid", "EquiHist").
*   `Prediction`: Estimated selectivity.
*   `Truth`: True selectivity.
*   `Q_Error`: Quality error metric (max(P/T, T/P)).

## 5. Quick Start Example

1.  **Generate Data** (if not already done):
    ```bash
    poetry run python datasets.py
    ```

2.  **Generate Workloads** (if not already done):
    ```bash
    poetry run python workload.py
    ```

3.  **Run a Test Experiment**:
    ```bash
    poetry run python main.py --rows 1000000 --dist zipf --eval 1000 --mode static
    ```

4.  **Analyze Results**:
    Check `results/{experiment_id}/1000000/zipf/1000.csv` for the output. The script will print the exact path with the experiment ID.
