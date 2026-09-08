# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research benchmark for **range-query cardinality/selectivity estimation** on a single integer column. Three estimators are compared under static workloads and under gradual data drift:

- **Equi-Width** (`models/equi_width.py`) — static baseline built from a 100k reservoir sample, scaled to N.
- **Equi-Hist** (`models/equi_hist.py`) — online self-tuning histogram; multiplicatively rescales bucket counts from query feedback (`update()`).
- **Hybrid** (`models/hybrid.py`) — equi-width buckets plus a **per-bucket local CDF model**, selected by search over candidates.

Everything is scored by **q-error** (`models/common.py: q_error_vec`), reported as median / p25 / p75 / p95 / avg.

## Commands

Python ≥3.12, dependencies via Poetry. All entry points are run as `poetry run python <script>`.

```bash
poetry install
poetry run ruff check .          # ruff is the only dev tool configured; there is no test suite

# Static benchmarks (orchestrator; loops distributions x models)
poetry run python run_experiments.py --scenario hard_narrow
# scenarios: simple_wide simple_narrow hard_wide hard_narrow hard_narrow_data_driven dsb all

# Drift benchmarks (orchestrator; loops steps x models)
poetry run python run_drift_experiments.py --scenario drift_5_ds_zipf_wl_normal
# scenarios: drift_10_ds_zipf_wl_normal drift_5_ds_zipf_wl_normal
#            drift_5_ds_zipf_wl_zipf drift_5_ds_add_zipf_wl_zipf all

# Single model, single dataset (what the orchestrators shell out to)
poetry run python run_hybrid.py --experiment-name my_exp \
    --dataset data/generated/60000000_hard/normal \
    --workload workload/1000000/narrow/workload.csv

# Single drift run — --shift-dir and --static-workload-path are required in practice
poetry run python run_hybrid_drift.py --experiment-name my_drift \
    --shift-dir "data/generated/60000000_hard/shift_normal_to_zipf_5%" \
    --workload 1000000 \
    --static-workload-path data/generated/60000000_hard/normal/workload_driven_1000000.csv
```

Note: README documents the static scenario as `hard_narrow_datadriven`, but `run_experiments.py` accepts `hard_narrow_data_driven`.

### Regenerating inputs (usually necessary)

`.gitignore` excludes `*.csv` and `/results/`. A fresh clone has `meta.pkl` / `step_*.pkl` / `stats.json` / plots committed, but **no `data.csv` and no workload CSVs** — so every run that touches a workload path will fail until they are regenerated:

```bash
poetry run python data/datasets.py --rows 60000000 --dist normal      # writes data.csv, meta.pkl, stats.json, plots
poetry run python data/generate_shifts.py --rows 60000000_hard --init normal --target zipf  # writes step_*.pkl
poetry run python workload/workload.py --count 1000000 --rows 60000000_hard  # data-driven workloads, batch over all subdirs
poetry run python workload/workload.py --count 1000000 --wide         # random workload -> data/workloads/wide_1000000.csv
poetry run python prepare_dsb.py                                       # DSB: .sql -> workload csv, .csv -> meta.pkl
```

Careful with workload paths: `run_experiments.py` hardcodes `workload/1000000/{wide,narrow}/workload.csv`, while `workload.py`'s random mode writes to `data/workloads/{mode}_{count}.csv`. The files must be placed/renamed to match.

`docker-compose.yml` (Postgres 16.3, mounts `./data`) is only for producing the DSB source tables; nothing in the Python code connects to it.

## Architecture

### The metadata contract

Almost every script consumes one of two pickled tuples. Getting the arity wrong is the most common breakage:

- **Dataset `meta.pkl`** — 8-tuple `(mn, mx, N, freq, sample, k, skew, kurt)`
- **Shift `step_N.pkl`** — 6-tuple `(mn, mx, N, freq, sample, k)`

`freq` is a dense per-value frequency array over `[mn, mx]` and is the **ground truth** (used to compute exact `y_true`). `sample` is a 100k reservoir sample and is the **only** thing models are allowed to build from — models see `sample` scaled by `N/len(sample)`, never `freq`. Keeping that separation intact is the point of the benchmark; a change that lets a model read `freq` directly invalidates results. `k` is a Freedman–Diaconis bucket count derived from the sample. Domain is fixed at `DOMAIN_MAX = 1_000_000` (`data/datasets.py:29`).

### Shared harness

`benchmark_utils.py` is the spine of the static path:

- `get_common_parser` — the `--dataset / --workload / --out-dir / --experiment-name / --buckets` contract every `run_*.py` inherits.
- `run_benchmark_suite(args, approach_fn)` — auto-detects whether `--dataset` is one dataset (has `meta.pkl`) or a directory of distributions, loops, and calls the script's `run_logic(args, out_dir, metadata)`. Per-dataset exceptions are caught and printed, not raised, so a run can "succeed" with missing outputs — check the log.
- `load_and_filter_workload` — computes exact truth from the `freq` prefix sum and **drops zero-truth queries**, so query counts differ per dataset.
- `save_benchmark_results` / `aggregate_summaries` — per-model `summary_<Model>.json` plus a rolled-up `results/<exp>/summary.csv`. Scientific notation is regex-rewritten to 9 decimals so timings stay readable.

The drift scripts (`run_*_drift.py`, `run_dsb_*_drift.py`) deliberately **do not** use `run_benchmark_suite`; they own their step loop and only borrow `setup_out_dir` / `load_and_filter_workload`.

### Hybrid estimator

`HybridEstimator` is the contribution being evaluated; the rest are baselines.

- `train()` samples CDF points per bucket, then `_train_adaptive_models` runs a **cascade with early exit**: no-model (uniform) → Ridge linear → power (√x) → log-linear → poly3 → DecisionTree → Isotonic → Fourier-features MLP, keeping the candidate with the lowest q-error on 500 synthetic in-bucket validation queries. `EARLY_EXIT_THRESHOLD = 1.5` stops the cascade once a model is good enough; `COMPLEX_MODEL_THRESHOLD = 3` gates the expensive Fourier/MLP branch. Buckets train in parallel via a `ThreadPoolExecutor`.
- `_bake_vectorized_data()` flattens the selected models into parallel numpy arrays plus an integer `mod_types` code (0 none, 1 linear, 2 poly, 3 other-sklearn, 4 log_linear, 5 power, 6 poly3, 7 isotonic, 8 tree). **Any mutation of `self.buckets` or `self.models` must be followed by a re-bake** — `EquiHistLearner.update` signals this by setting `self.b_lo = None`.
- `predict()` sums whole enclosed buckets straight from `b_count` and only invokes the local CDF model for the two partially-covered edge buckets.
- `feedback_update()` is the cheap adaptation path: redistributes observed vs. predicted mass over overlapping buckets and applies a clipped EMA to bucket counts. It does not retrain models.

### Drift protocol

Each drift step evaluates three stages against the same step's data, which is why the output CSV has `Shock_*`, `FT_*`, `RB_*` column families:

1. **Shock** — predict with the stale model, no adaptation.
2. **FT (finetune)** — `feedback_update()` rescales bucket mass, re-predict.
3. **RB (rebuild)** — `detect_bad_buckets` (q-error > 1.3) selects bucket indices, `train(..., bucket_indices=...)` retrains only those, re-predict.

`--shift-workload` switches from a fixed workload to per-step `step_N_workload_driven_<count>.csv` (workload drifts with the data); otherwise `--static-workload-path` is reused at every step.

### Output layout

`results/<experiment-name>/<dataset-name>/` where dataset-name is `<dist>` for static runs and `gradual_<init>_to_<target>_FT_RB` / `<dist>_EquiWidth` / `<dist>_EquiHist` for drift runs. Hybrid drift writes `adaptation_ft_rb.csv`; the histogram baselines write `<model>_drift_analysis.csv`. `merge_*.py` and `plot_dsb_drift.py` join these per-model CSVs into combined tables — they contain **hardcoded result paths** and will need editing when experiment names change.

## Conventions

- Comments and print statements are a mix of English and Russian; match the surrounding file rather than translating.
- `patch_benchmark.py` and `patch_dataset_path.py` are one-shot regex migrations already applied to the tree. Do not re-run them.
- Random seeds are fixed at 42 throughout (dataset generation, workload generation, model training) — keep new code seeded the same way for reproducibility.
- Per-query inference is timed inside the loop (`time.perf_counter()` per `predict`), so `predict` stays scalar and Python-level on purpose; the numpy work is in the bake step, not the hot loop.
