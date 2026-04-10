# Query Optimizer Benchmark — The Ultimate Guide

Welcome to the comprehensive benchmarking framework for evaluating query selectivity and cardinality estimation models. This guide outlines the exact, step-by-step procedures required to initialize the project, generate datasets and workloads, and execute the full evaluation suite from scratch.

---

## Table of Contents

1. [Prerequisites & Environment Setup](#1-prerequisites--environment-setup)
2. [Data Generation (Phase I) — Synthetic](#2-data-generation-phase-i)
3. [Workload Generation (Phase II)](#3-workload-generation-phase-ii)
4. [Execution of the Models (Phase III)](#4-execution-of-the-models-phase-iii)
5. [Shift Scenarios & Concept Drift (Phase IV)](#5-shift-scenarios--concept-drift-phase-iv)
6. [Real-World Experiment: DSB (Decision Support Benchmark)](#6-real-world-experiment-dsb-decision-support-benchmark)
7. [Real-World Experiment: STATS-CEB](#7-real-world-experiment-stats-ceb)
8. [Adding a New Benchmark (Generic Guide)](#8-adding-a-new-benchmark-generic-guide)
9. [Analytics and Outputs](#9-analytics-and-outputs)
10. [Project Structure Reference](#10-project-structure-reference)

---

## 1. Prerequisites & Environment Setup

Ensure your system meets the following requirements before proceeding:
- **Python 3.12+** (preferably managed via pyenv)
- **Poetry** (for Python dependency management)
- **Docker & Docker Compose** (optional — only needed for DSB processing or custom database backends)

### Step 1: Install Python Dependencies
Install the required packages within the isolated Poetry environment:
```bash
poetry install
```

### Step 2: Initialize Database (Optional / For DSB processing)
If your benchmark requires the PostgreSQL backend as listed in the configuration:
```bash
docker compose up -d
```

> **Note:** The PostgreSQL container is only needed if you are extracting raw data/queries from a database. If you already have CSV files and workloads, you can skip Docker entirely.

---

## 2. Data Generation (Phase I)

The evaluation process operates on synthetic data distributions (e.g., Uniform, Normal, Zipf, Anti-Zipf, Sparse Cluster) or real-world traces.

### Generate Synthetic Datasets
You need to create raw baseline data. The built-in generator computes distributions natively and automatically saves metadata configurations.

**To generate all default distributions at various scales (1M, 10M, 60M rows):**
```bash
poetry run python data/datasets.py --all
```

**To generate specific distributions and sizes:**
```bash
poetry run python data/datasets.py --rows 1000000 10000000 --dist zipf
```

**To generate without micro-distributions (traditional simple approach):**
```bash
poetry run python data/datasets.py --rows 1000000 --dist uniform --simple
```

*Note: This creates structured folders under `data/generated/{rows}/{distribution}/` (e.g., `data/generated/1000000/zipf/`), containing:*
| File | Description |
|------|-------------|
| `data.csv` | Raw data values (headerless, single column of integers) |
| `meta.pkl` | Pickled tuple: `(mn, mx, N, freq, sample, k, skew, kurt)` |
| `stats.json` | Human-readable statistics (Rows, Min, Max, NDV, Skewness, etc.) |
| `hist.png` | Distribution histogram visualization |
| `hist_sample.png` | 100k reservoir sample histogram |
| `hist_micro.png` | Micro-distribution analysis (modulo 1000) |

---

## 3. Workload Generation (Phase II)

Query workloads (range queries) are generated independently, avoiding bias. The queries map accurately to the underlying generated environments.

**To generate a data-driven workload (e.g., 10,000 queries) for all distributions of a specific row size:**
```bash
poetry run python workload/workload.py --rows 1000000 --count 10000
```

**To generate for a specific distribution only:**
```bash
poetry run python workload/workload.py --rows 1000000 --count 10000 --dist zipf
```

**To generate random workloads (no data context):**
```bash
poetry run python workload/workload.py --count 10000 --domain-max 1000000
```

*Note: The data-driven mode automatically scans the `data/generated/{rows}` directory and injects query workloads (`workload_driven_{count}.csv`) directly into each distribution's folder.*

### Workload Format
All workloads must be CSV files with two columns:
```csv
low,high
4236,4636
10000,10500
...
```
Each row represents a range query `[low, high]` (inclusive on both ends).

---

## 4. Execution of the Models (Phase III)

With data and workloads established, run the benchmark execution scripts. This automatically runs multiple baseline models (Equi-Width, Equi-Hist, and the Hybrid optimizer) in succession.

### Option A: Evaluate a Specific Distribution
```bash
./run_all_models.sh "experiment_zipf_1M" \
    --dataset "data/generated/1000000/zipf" \
    --count 10000
```

### Option B: Evaluate All Distributions Automatically
By passing the parent dataset folder, the bash script natively discovers and evaluates all generated sub-distributions:
```bash
./run_all_models.sh "experiment_all_1M" \
    --dataset "data/generated/1000000" \
    --count 10000
```

### Option C: Run Individual Models
You can also run each model independently:
```bash
# Equi-Width only
poetry run python run_equiwidth.py \
    --experiment-name "my_experiment" \
    --dataset "data/generated/1000000/zipf" \
    --workload "10000"

# Equi-Hist only (with custom learning rate)
poetry run python run_equihist.py \
    --experiment-name "my_experiment" \
    --dataset "data/generated/1000000/zipf" \
    --workload "10000" \
    --lr 0.5

# Hybrid only (with custom parameters)
poetry run python run_hybrid.py \
    --experiment-name "my_experiment" \
    --dataset "data/generated/1000000/zipf" \
    --workload "10000" \
    --points 200 \
    --ident 1e-4 \
    --penalty 1.5
```

> **Workload resolution:** The `--workload` argument can be either:
> - A **number** (e.g., `10000`) — the runner will look for `workload_driven_10000.csv` inside the dataset directory
> - A **file path** (e.g., `data/STATS-CEB/PostHistoryLengthText/workload_driven_3127.csv`) — used directly

### Common Arguments for All Model Runners
| Argument | Required | Description |
|----------|----------|-------------|
| `--dataset` | Yes | Path to dataset directory containing `meta.pkl` |
| `--workload` | Yes | Workload query count or path to workload CSV |
| `--experiment-name` | No | Name for the experiment (auto-generated if omitted) |
| `--out-dir` | No | Base directory for results (default: `results`) |
| `--buckets` | No | Override Freedman-Diaconis bin count |

---

## 5. Shift Scenarios & Concept Drift (Phase IV)

Assess optimizer capabilities in highly unstable environments where distribution shifts dynamically (e.g., Anti-Zipf transitioning to Uniform). 

### Sub-Phase A: Generate the Shift Data Sequences
Before you can run the benchmark to evaluate models dynamically reacting to drift, you must pre-generate the transitional shift datasets/workloads using your isolated initial states.

```bash
poetry run python data/generate_shifts.py --rows 1000000 --init anti_zipf --target uniform
```
*Note: Ensure to run `workload.py` afterwards if dynamic workloads are needed for the shift intermediate states. The baseline `workload.py` command already scans everything natively.*

### Sub-Phase B: Execute Shift Scenarios Pipeline
**Run a complete shift scenario sequence mapping Equi-Width, Equi-Hist, and Hybrid models concurrently:**

```bash
./run_all_shift_scenarios.sh \
    1000000 \
    10000 \
    shift_expr_1M \
    anti_zipf \
    uniform
```
*(Arguments provided: `<Dataset Dir (Rows)>`, `<Workload Count or CSV path>`, `[Experiment Name]`, `[Initial Dist]`, `[Target Dist]`)*

---

## 6. Real-World Experiment: DSB (Decision Support Benchmark)

The DSB pipeline processes real-world data from the Decision Support Benchmark (TPC-DS style) by extracting column data and SQL range queries.

### 6.1 What You Need

| File | Location | Description |
|------|----------|-------------|
| Raw CSV data | `data/dsb/{column_name}.csv` | Single-column CSV with header (e.g., `ws_item_sk.csv`) |
| SQL query file | `data/dsb/custom_{column_name}.sql` | File with SQL `BETWEEN` queries, one per line |

**SQL query format** (one per line):
```sql
where ws_item_sk between 4236 and 4236 + 400;
where ws_item_sk between 1000 and 5000;
```
The parser supports both `BETWEEN X AND Y` and `BETWEEN X AND Y + Z` syntax.

### 6.2 Step-by-Step

**Step 1:** Place your raw CSV files and SQL query files inside `data/dsb/`:
```
data/dsb/
├── ws_item_sk.csv              # Raw column data (with header)
├── custom_ws_item_sk.sql       # Queries referencing this column
├── cr_item_sk.csv
├── custom_cr_item_sk.sql
├── cr_returned_time_sk.csv
└── custom_cr_returned_time_sk.sql
```

**Step 2:** Run the preparation script:
```bash
poetry run python prepare_dsb.py
```

This script will:
1. Parse all SQL queries extracting `BETWEEN` range bounds → generate `workload_driven_{count}.csv`
2. Read the raw CSV, clean it to headerless integers → generate `data.csv`
3. Compute statistics (min, max, N, NDV, skewness, kurtosis)
4. Build a 100k reservoir sample and exact frequency map
5. Calculate optimal bin count via Freedman-Diaconis rule
6. Save everything into `meta.pkl`

**Output structure after preparation:**
```
data/dsb/
├── ws_item_sk/
│   ├── data.csv                     # Cleaned integer column (no header)
│   ├── meta.pkl                     # (mn, mx, N, freq, sample, k, skew, kurt)
│   └── workload_driven_397307.csv   # Parsed range queries
├── cr_item_sk/
│   ├── data.csv
│   ├── meta.pkl
│   └── workload_driven_XXXX.csv
└── cr_returned_time_sk/
    └── ...
```

**Step 3:** Run the models against the prepared DSB data:
```bash
# Run all three models on a single DSB column
./run_all_models.sh "dsb_experiment" \
    --dataset "data/dsb/ws_item_sk" \
    --count 397307

# Or run all DSB columns at once (auto-discovery)
./run_all_models.sh "dsb_experiment_all" \
    --dataset "data/dsb" \
    --count 397307
```

> **Important:** When using `--count`, the runner looks for `workload_driven_{count}.csv` inside the dataset directory. If your workload has a different number of queries, use the actual count or pass the full file path with `--workload`.

### 6.3 Adding New DSB Tables/Columns

To add a new table/column to the DSB pipeline:

1. Export the column data as CSV: `data/dsb/{new_column}.csv` (single column, with header)
2. Create a SQL query file: `data/dsb/custom_{new_column}.sql`
3. Add a new `process_table()` call at the bottom of `prepare_dsb.py`:
   ```python
   process_table("new_column", "data/dsb/new_column.csv", "data/dsb/custom_new_column.sql")
   ```
4. Re-run `poetry run python prepare_dsb.py`

---

## 7. Real-World Experiment: STATS-CEB

The [STATS-CEB](https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark) benchmark uses real-world data from the Stack Overflow statistics database for cardinality estimation evaluation.

### 7.1 What You Need

| File | Location | Description |
|------|----------|-------------|
| Column data CSV | `data/STATS-CEB/{TableColumn}.csv` | Single column of integer values (no header) |
| Workload CSV | `data/STATS-CEB/{TableColumn}_workload.csv` | Range queries with `low,high` header |

### 7.2 Step-by-Step

**Step 1:** Obtain the STATS dataset. Download from the [official repository](https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark) or prepare your own column extractions.

**Step 2:** Place your files in the `data/STATS-CEB/` directory:
```
data/STATS-CEB/
├── PostHistoryLengthText.csv            # Raw column data (no header, integers only)
├── PostHistoryLengthText_workload.csv   # Range queries: low,high
├── prepare.py                           # Preparation script
└── random_workloads.py                  # Optional: random workload generator
```

**Step 3 (Optional):** Generate random workloads if you don't have real SQL queries:
```bash
poetry run python data/STATS-CEB/random_workloads.py
```
This generates 400,000 unique random range queries based on the column's domain. To customize, edit the `TEMPLATES` dictionary in `random_workloads.py`:
```python
TEMPLATES = {
    "PostHistoryLengthText": {
        "start_range": (0, 31278),   # Domain of values in the column
        "size_range": (100, 300),     # Width range for queries
        "max_value": 31278,           # Maximum value in the column
    }
}
```

**Step 4:** Run the preparation script to build the `meta.pkl`:
```bash
poetry run python data/STATS-CEB/prepare.py
```

This produces the following output structure:
```
data/STATS-CEB/PostHistoryLengthText/
├── data.csv                     # Cleaned integer column (no header)
├── meta.pkl                     # (mn, mx, N, freq, sample, k, skew, kurt)
└── workload_driven_3127.csv     # Formatted range queries
```

**Step 5:** Run the models:
```bash
# Run all models on STATS-CEB
./run_all_models.sh "stats_ceb_experiment" \
    --dataset "data/STATS-CEB/PostHistoryLengthText" \
    --count 3127
```

Or run individual models:
```bash
poetry run python run_hybrid.py \
    --experiment-name "stats_ceb_hybrid" \
    --dataset "data/STATS-CEB/PostHistoryLengthText" \
    --workload "data/STATS-CEB/PostHistoryLengthText/workload_driven_3127.csv"
```

### 7.3 Adding More STATS Tables/Columns

To evaluate additional columns from the STATS database:

1. **Extract the column** as a CSV file with no header (one integer per line):
   ```
   42
   1337
   7
   ...
   ```
   Save as `data/STATS-CEB/{YourTableColumn}.csv`

2. **Create a workload** — either:
   - Extract real query predicates from CEB and format as `low,high` CSV
   - Or use `random_workloads.py` by adding a new template:
     ```python
     TEMPLATES = {
         "PostHistoryLengthText": { ... },
         "YourTableColumn": {
             "start_range": (0, MAX_VALUE),
             "size_range": (100, 300),
             "max_value": MAX_VALUE,
         }
     }
     ```
   Save as `data/STATS-CEB/{YourTableColumn}_workload.csv`

3. **Add a `process_table()` call** at the bottom of `data/STATS-CEB/prepare.py`:
   ```python
   process_table(
       "YourTableColumn",
       "data/STATS-CEB/YourTableColumn.csv",
       "data/STATS-CEB/YourTableColumn_workload.csv",
   )
   ```

4. **Run the preparation:**
   ```bash
   poetry run python data/STATS-CEB/prepare.py
   ```

5. **Run the benchmark:**
   ```bash
   ./run_all_models.sh "stats_ceb_multi" \
       --dataset "data/STATS-CEB" \
       --count <workload_count>
   ```
   This auto-discovers all sub-directories containing `meta.pkl`.

---

## 8. Adding a New Benchmark (Generic Guide)

This framework can evaluate **any** single-column integer dataset with range queries. Follow these steps to integrate a new benchmark.

### 8.1 The Contract: What the Framework Expects

Every dataset directory must contain exactly two files for the models to work:

| File | Format | Description |
|------|--------|-------------|
| `meta.pkl` | Python pickle | Tuple of `(mn, mx, N, freq, sample, k, skew, kurt)` |
| `workload_driven_{count}.csv` | CSV with header | Range queries with columns `low,high` |

**The `meta.pkl` tuple fields:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `mn` | `int` | Minimum value in the dataset |
| 1 | `mx` | `int` | Maximum value in the dataset |
| 2 | `N` | `int` | Total number of rows |
| 3 | `freq` | `np.ndarray[int64]` | Exact frequency array of length `(mx - mn + 1)` — `freq[i]` = count of value `(mn + i)` |
| 4 | `sample` | `np.ndarray[int64]` | Reservoir sample of ~100,000 values |
| 5 | `k` | `int` | Number of bins (Freedman-Diaconis rule) |
| 6 | `skew` | `float` | Skewness of the distribution |
| 7 | `kurt` | `float` | Kurtosis of the distribution |

### 8.2 Step-by-Step: Adding a Completely New Benchmark

**Step 1: Create the directory structure**
```bash
mkdir -p data/my-benchmark
```

**Step 2: Prepare the data file**

Create a headerless, single-column CSV of integer values:
```bash
# data/my-benchmark/raw_column.csv (no header)
42
1337
7
255
...
```

> **Important:** Values must be non-negative integers. If your data has floats, discretize them first. If your data has negative values, shift them to start at 0.

**Step 3: Prepare the workload file**

Create a CSV of range queries with a `low,high` header:
```bash
# data/my-benchmark/my_workload.csv
low,high
100,500
0,1000
42,42
...
```

**Step 4: Create a preparation script**

Create `data/my-benchmark/prepare.py` (or any name you prefer):

```python
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from data.datasets import (
    scan_min_max_count, calculate_ndv, calculate_skew_kurt,
    build_frequency_and_sample, freedman_diaconis_bins
)


def process_table(name, csv_file, workload_file):
    print(f"Processing {name}...")
    out_dir = Path("data/my-benchmark") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy and clean the workload
    wl = pd.read_csv(workload_file)
    count = len(wl)
    workload_path = out_dir / f"workload_driven_{count}.csv"
    wl.to_csv(workload_path, index=False)
    print(f"  Saved {count} workload queries to {workload_path}")

    # 2. Clean the data CSV → headerless integers
    data_path = out_dir / "data.csv"
    df = pd.read_csv(csv_file, header=None, names=["value"], dtype=str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna()
    df["value"] = df["value"].astype(int)
    df["value"].to_csv(data_path, index=False, header=False)
    print(f"  Clean data: {len(df):,} rows")

    # 3. Build meta.pkl
    mn, mx, N = scan_min_max_count(data_path)
    skew, kurt = calculate_skew_kurt(data_path)

    TARGET_SAMPLE_SIZE = 100_000
    freq, sample = build_frequency_and_sample(
        data_path, mn, mx, N, TARGET_SAMPLE_SIZE, 42
    )
    k = freedman_diaconis_bins(sample, mn, mx, N)

    meta_path = out_dir / "meta.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump((mn, mx, N, freq, sample, k, skew, kurt), f)

    print(f"  meta.pkl: N={N}, mn={mn}, mx={mx}, k={k}")


# ---------- Process all tables ----------
process_table(
    "my_column",
    "data/my-benchmark/raw_column.csv",
    "data/my-benchmark/my_workload.csv",
)

print("Done!")
```

**Step 5: Run the preparation**
```bash
poetry run python data/my-benchmark/prepare.py
```

**Step 6: Verify the output**
```bash
ls data/my-benchmark/my_column/
# Expected: data.csv  meta.pkl  workload_driven_XXX.csv
```

**Step 7: Run the models**
```bash
./run_all_models.sh "my_benchmark_experiment" \
    --dataset "data/my-benchmark/my_column" \
    --count <number_of_queries>
```

### 8.3 Common Pitfalls

| Problem | Cause | Solution |
|---------|-------|----------|
| `FileNotFoundError: meta.pkl not found` | Preparation script was not run or output path is wrong | Run your `prepare.py` and verify the output directory |
| `AssertionError: No queries found` | Workload CSV is empty or has wrong format | Ensure CSV has `low,high` header and at least one data row |
| All Q-Errors are very high | Workload range is outside the data domain | Make sure query `[low, high]` ranges overlap with actual data values `[mn, mx]` |
| `freq` array is huge / OOM | Data domain is extremely wide (e.g., `mx - mn > 100M`) | Consider discretizing or binning your data before processing |
| Zero-selectivity queries filtered out | Queries don't match any data | This is expected behavior — use data-driven workload generation |

### 8.4 Benchmark Comparison Matrix

| Benchmark | Data Source | Queries From | Preparation Script |
|-----------|-------------|--------------|-------------------|
| Synthetic | `data/datasets.py` generates | `workload/workload.py` generates | N/A (built-in) |
| DSB | TPC-DS column exports | SQL `BETWEEN` clauses | `prepare_dsb.py` |
| STATS-CEB | Stack Overflow database | CEB predicates or random | `data/STATS-CEB/prepare.py` |
| Custom | Your CSV data | Your range queries | Your `prepare.py` (see template above) |

---

## 9. Analytics and Outputs

### Results Directory Structure
Evaluation artifacts and generated metric records are output directly into the `results/` directory based on your explicitly established experiment titles:
```
results/
└── {experiment_name}/
    ├── {dataset_name}/
    │   ├── workload_driven_XXXX_EquiWidth.csv    # Per-query results
    │   ├── workload_driven_XXXX_EquiHist.csv
    │   ├── workload_driven_XXXX_Hybrid.csv
    │   ├── summary_EquiWidth.json                # Aggregated metrics
    │   ├── summary_EquiHist.json
    │   ├── summary_Hybrid.json
    │   └── initial_baseline_debug.png            # Hybrid bucket analysis
    └── summary.csv                               # Cross-model comparison
```

### Per-Query CSV Columns
| Column | Description |
|--------|-------------|
| `Query_ID` | Index of the query |
| `Model` | Model name (EquiWidth, EquiHist, Hybrid) |
| `Prediction` | Estimated selectivity |
| `Truth` | True selectivity |
| `Q_Error` | Quality error: `max(Prediction/Truth, Truth/Prediction)` |

### Summary JSON Metrics
| Metric | Description |
|--------|-------------|
| `train_time` / `build_time` | Time to build/train the model (seconds) |
| `infer_time` | Total inference time for all queries (seconds) |
| `median_q_error` | Median Q-Error across all queries |
| `p25_q_error` / `p75_q_error` | 25th and 75th percentile Q-Error |
| `p95_q_error` | 95th percentile Q-Error |
| `avg_q_error` | Mean Q-Error |

### Generating Comparison Plots
For rapid visual comparisons between models:
```bash
# Plot comparison for a specific experiment
./plot_all.sh <experiment_id> [rows] [workload_name]

# Example
./plot_all.sh my_experiment 1000000 10000
```

---

## 10. Project Structure Reference

```
query-optimizer/
├── data/
│   ├── datasets.py                 # Synthetic data generator (uniform, normal, zipf, etc.)
│   ├── generate_shifts.py          # Distribution shift sequence generator
│   ├── generated/                  # Output: synthetic datasets
│   │   └── {rows}/{dist}/          # data.csv, meta.pkl, stats.json, plots
│   ├── dsb/                        # DSB benchmark data and outputs
│   │   ├── *.csv, *.sql            # Raw inputs
│   │   └── {column_name}/          # Processed: data.csv, meta.pkl, workload
│   └── STATS-CEB/                  # STATS-CEB benchmark data and outputs
│       ├── *.csv                   # Raw inputs
│       ├── prepare.py              # STATS-CEB preparation script
│       ├── random_workloads.py     # Random workload generator
│       └── {table_column}/         # Processed: data.csv, meta.pkl, workload
├── models/
│   ├── equi_width.py               # Equi-Width histogram model
│   ├── equi_hist.py                # Equi-Hist (online learning) model
│   ├── hybrid.py                   # Hybrid estimator model
│   └── common.py                   # Shared utilities (q-error, summarize)
├── workload/
│   └── workload.py                 # Workload generation and I/O
├── plot/
│   ├── plot_comparison.py          # Model comparison boxplots
│   ├── plot_query_distribution.py  # Query distribution visualization
│   └── plot_workload.py            # Workload coverage plots
├── run_equiwidth.py                # Standalone Equi-Width runner
├── run_equihist.py                 # Standalone Equi-Hist runner
├── run_hybrid.py                   # Standalone Hybrid runner
├── run_equiwidth_drift.py          # Equi-Width drift scenario runner
├── run_equihist_drift.py           # Equi-Hist drift scenario runner
├── run_hybrid_drift.py             # Hybrid drift scenario runner
├── run_hybrid_finetune.py          # Hybrid fine-tuning runner
├── run_all_models.sh               # Run all 3 models sequentially
├── run_all_shift_scenarios.sh      # Run all drift scenarios
├── plot_all.sh                     # Generate all comparison plots
├── prepare_dsb.py                  # DSB data preparation script
├── benchmark_utils.py              # Shared benchmark utilities
├── merge_drift_results.py          # Merge drift experiment results
├── merge_all_drifts.py             # Aggregate all drift results
├── docker-compose.yml              # PostgreSQL container config
├── pyproject.toml                  # Poetry project configuration
└── results/                        # All experiment outputs
```
