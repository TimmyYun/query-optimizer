# Query Optimizer Benchmark

A comprehensive benchmarking framework for evaluating query selectivity estimation and cardinality estimation models. This project separates dataset generation, workload creation, and benchmark execution to ensure reproducible and isolated experiments. 

It evaluates three distinct approaches:
- **Equi-Width**: Static base histogram
- **Equi-Hist**: Online reinforcement-learning-style histogram adaptation. *Excluded from the experiment design (too slow); the implementation is kept and its scripts can still be run manually.*
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
- `run_sthist.py` / `run_sthist_drift.py`
- `run_entropyhist.py` / `run_entropyhist_drift.py`

When running the individual drift scripts directly, use `--shift-dir` and `--static-workload-path` to manually define the transition directories.

### ST-Histogram baseline (Aboulnaga & Chaudhuri, SIGMOD 1999)

`models/st_histogram.py` is a clean-room 1-D self-tuning histogram. It never sees the data: it starts from catalog metadata `(N, min, max)` as `B` uniform buckets and learns only from query feedback `(low, high, actual_count)`.

| Paper | Code |
|---|---|
| §3.1 uniform initialization | `STHistogram.__init__` |
| §3.2 estimate, uniform within a bucket | `STHistogram.estimate_range` / `predict` |
| §3.2 Fig. 2 contribution-weighted update | `STHistogram.observe_feedback` |
| §3.3 Fig. 3 greedy run merge | `greedy_merge_runs` |
| §3.3 Fig. 3 proportional split | `split_candidate_count`, `allocate_split_slots`, `STHistogram.restructure` |

The class docstring lists every decision the paper leaves open (rounding, ties, `est == 0`, failed split allocation). `--mode paper_faithful` is the baseline. `--mode robust_extended` turns on non-paper extensions and must be reported separately. Total mass is **not** renormalized to N (Fig. 2), so `final_mass_ratio` / `Mass_Ratio` are logged. The drift script has no ANALYZE step, because the model may not read samples. For each step it reports `Shock` (stale), `Online` (prequential), and `Post` (frozen after that step's feedback).

```bash
poetry run python -m pytest tests/ -q                                   # unit tests (spec §9)
poetry run python run_sthist.py --dataset data/generated/60000000_hard/normal \
    --workload data/generated/60000000_hard/normal/workload_driven_1000000.csv --experiment-name sthist
poetry run python run_sthist.py ... --no-restructure                    # ablation -> model "STHist-NoRS"
# Original paper setup: Table 2 (z = 0..3), §5.5 25% data replacement, §5.4 locality shift
poetry run python run_sthist_paper.py --experiment-name sthist_paper --seeds 42 43 44 45 46 47 48 49 50 51
```

At high skew (z ≥ 2), the Table 2 error varies a lot between seeds (z=2: 23–915%). Compare the multi-seed mean in `table2_agg.csv` with the paper's numbers, not a single run.

### Entropy-based Histograms baseline (To, Chiang, Shahabi, CIKM 2013)

`models/entropy_histogram.py` is a clean-room 1-D implementation of the ME / MSE / MB builders, written from `entropy_histograms_1d_cleanroom_spec.md`. All three share one top-down best-cut engine (Algorithm 1). It starts from a single bucket and keeps one best local cut per splittable bucket in a heap. Each step splits the globally best bucket and rescores only its two children, until `B_effective - 1` cuts have been made. Segment entropy, MSE cost and MB cost are computed in O(1) from prefix moments.

| Paper | Code |
|---|---|
| §2.1 Eq. 2 entropy, §4.1 Eq. 3 decomposition | `entropy`, `PrefixMoments.entropy` |
| §4.2 Algorithm 1, ME loss `H_p - (H_L + H_R)`, priority `W_p * loss` | `split_scores`, `best_cut_partition` |
| §4.2 / Poosala 1996 area `f * spread`, last spread = 1 | `area_weights` |
| §4.3.1 Eq. 6 / §4.3.2 MSE, Eq. 8 / §4.3.3 MB | `PrefixMoments.mse_cost`, `PrefixMoments.mb_cost` |
| §2.2 Eq. 1 `T/d`, Eq. 6 `T*2^-H`, BF, §4.5 weighted mean | `estimate_eq(value, estimator)` |
| §2.2 linear range interpolation | `estimate_range` / `predict` |

The class docstring lists every point the paper leaves open, and the module and the run scripts label each one `[RESOLUTION]` or `[EXTENSION]`:
- the Algorithm 1 update order;
- the missing parent weight in the MSE gain;
- the duplicated `W_r` in the MB gain;
- the tie policy;
- point, empty and reversed ranges.

The baseline is `EntropyHist-MEA`, which uses `--method ME --me-input area`, `paper_alg1`, Eq. 1 for point queries and `paper_linear`. Any non-paper choice changes the model name (`-W`, `-Eq_*`, `-Rng_*`), so it cannot be mistaken for the baseline.

Benchmark contract: the histogram is built from the 100k `sample` and never from `freq`. Bucket statistics stay in sample units, and estimates are multiplied by `row_scale = N / len(sample)`. The ME, MSE and MB objectives are linear in the frequency scale, so the bucket boundaries do not depend on the scale. Besides q-error, `rmse_count` reports the paper's own metric.

The paper defines no updates and no drift handling. `run_entropyhist_drift.py` therefore reports two arms for each step:
- `Static`: the step-0 histogram, frozen (the `static_no_rebuild` control);
- `Rebuild`: a from-scratch `build()` on that step's sample, the same simulated ANALYZE the other drift baselines use.

The spec's optional `StreamingWrapper` is not implemented, because it needs an authoritative per-value frequency map, which this benchmark does not let a model see.

```bash
poetry run python -m pytest tests/test_entropy_histogram.py -q          # spec §10, incl. brute-force greedy oracle
poetry run python run_entropyhist.py --dataset data/generated/60000000_hard/normal \
    --workload data/generated/60000000_hard/normal/workload_driven_1000000.csv --experiment-name entropyhist
poetry run python run_entropyhist.py ... --method MSE                   # -> "EntropyHist-MSE"; also MB, --me-input frequency
# Paper-style protocol on synthetic Zipf: N in {1000, 10000, 20000}, B = 30, RMSE on 10k eq/range queries
poetry run python run_entropyhist_paper.py --experiment-name entropyhist_paper
```
