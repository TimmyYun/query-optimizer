# Query Optimizer Benchmark

A comprehensive benchmarking framework for evaluating query selectivity estimation and cardinality estimation models. This project separates dataset generation, workload creation, and benchmark execution to ensure reproducible and isolated experiments. 

It evaluates three distinct approaches:
- **Equi-Width**: Static base histogram
- **Equi-Hist**: Online reinforcement-learning-style histogram adaptation
- **Hybrid**: A two-tier estimator using an underlying histogram combined with piecewise ML models (Decision Trees, Poly Cubic, Isotonic Regression, Power Laws)

## 1. Overview

The benchmark workflow allows you to run both **Static** and **Dynamic (Drift)** experiments on a predefined dataset of 60M rows and a workload of 1M queries. 

### Data Distributions
The framework evaluates models on the following distributions:
1. `normal`
2. `zipf`
3. `uniform`
4. `anti_zipf`
5. `sparse_cluster`

## 2. Setup

Ensure you have Python 3.9+ and Poetry installed.

```bash
# Install dependencies
poetry install
```

## 3. Workflow Stages

### Stage 1: Static Benchmarking (`run_experiments.py`)

The static benchmarking suite (`run_experiments.py`) orchestrates evaluation across standard, unchanging datasets. It handles both "Simple" and "Hard" datasets, combined with "Wide" and "Narrow" workloads.

**Available Scenarios:**
- `simple_wide`: Evaluates simple synthetic data on wide workloads.
- `simple_narrow`: Evaluates simple synthetic data on narrow workloads.
- `hard_wide`: Evaluates complex, hard synthetic data on wide workloads.
- `hard_narrow`: Evaluates complex, hard synthetic data on narrow workloads.
- `hard_narrow_datadriven`: Evaluates complex data using a workload derived directly from the underlying data distribution.
- `dsb`: Evaluates models on the standard DSB (Decision Support Benchmark) datasets.
- `all`: Runs all of the above sequentially.

**Usage:**
```bash
poetry run python run_experiments.py --scenario <scenario_name>
```

**Example:**
```bash
poetry run python run_experiments.py --scenario hard_narrow_datadriven
```

### Stage 2: Dynamic Data Drift Benchmarking (`run_drift_experiments.py`)

The dynamic benchmarking suite evaluates how well the models adapt over time when the underlying data distribution gradually shifts (e.g., from a Normal to a Zipf distribution) in steps.

The framework supports step-by-step drift evaluation, mapping both static workloads and dynamic, shifting workloads across the transition.

**Available Scenarios:**
- `drift_10_ds_zipf_wl_normal`: 10% step shifts. Dataset drifts from Normal to Zipf. Workload remains static (Normal).
- `drift_5_ds_zipf_wl_normal`: 5% step shifts. Dataset drifts from Normal to Zipf. Workload remains static (Normal).
- `drift_5_ds_zipf_wl_zipf`: 5% step shifts. Dataset drifts from Normal to Zipf. Workload shifts dynamically alongside the dataset.
- `drift_5_ds_add_zipf_wl_zipf`: 5% step shifts. Zipf distribution is additively mixed into the Normal dataset. Workload shifts dynamically.
- `all`: Runs all drift scenarios sequentially.

**Usage:**
```bash
poetry run python run_drift_experiments.py --scenario <scenario_name>
```

**Example:**
```bash
poetry run python run_drift_experiments.py --scenario drift_5_ds_zipf_wl_normal
```

## 4. Output Structure

Results are saved in a hierarchical structure under the `results/` directory, categorized by the experiment run name. The system generates model-specific output directories, plots, and detailed CSV summaries.

```text
results/
├── {scenario_name}/
│   ├── summary.csv                            # Global performance aggregation
│   ├── {distribution}_EquiWidth/              # EquiWidth specific outputs
│   │   ├── equiwidth_drift_analysis.csv
│   │   └── plots/
│   ├── {distribution}_EquiHist/               # EquiHist specific outputs
│   │   ├── equihist_drift_analysis.csv
│   │   └── plots/
│   └── {distribution}_FT_RB/                  # Hybrid specific outputs
│       ├── adaptation_ft_rb.csv
│       ├── model_reports/                     # Detailed piecewise model architectures
│       └── plots/
```

### Reporting & Precision
The framework formats timing metrics and Quality Error (Q-Error) to 9 decimal places in CSV outputs to eliminate scientific notation, allowing for precise downstream analysis.

## 5. Underlying Approach Scripts

The orchestrator scripts internally invoke dedicated files for each model. If necessary, you can run these individually:
- `run_equiwidth.py` / `run_equiwidth_drift.py`
- `run_equihist.py` / `run_equihist_drift.py`
- `run_hybrid.py` / `run_hybrid_drift.py`

When running the individual drift scripts directly, use `--shift-dir` and `--static-workload-path` to manually define the transition directories.
