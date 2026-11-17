# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research benchmark for **range-query cardinality/selectivity estimation** on a single integer column. Three estimators are compared under static workloads and under gradual data drift:

- **Equi-Width** (`models/equi_width.py`) — static baseline built from a 100k reservoir sample, scaled to N.
- **Equi-Hist** (`models/equi_hist.py`) — online self-tuning histogram; multiplicatively rescales bucket counts from query feedback (`update()`).
- **Hybrid FD CDF** (`models/hybrid.py`) — equi-width buckets plus a **per-bucket local CDF model**, chosen by a Waterfall over a fixed set of tiers. This is the method the thesis defends; its spec is fixed (see Hybrid estimator below).

The orchestrators run six models by default: EquiWidth, EquiDepth, TKHist, EntropyHist-MEA, STHist, Hybrid (variants such as `EntropyHist-MEA-Rng_discrete_linear` and EquiHist only via `--models`). Every model gets **100 buckets** (`--buckets 100`, default) or the Freedman–Diaconis `k` of the data (`--buckets fd`).

Everything is scored by **q-error** (`models/common.py: q_error_vec`), reported as median / p25 / p75 / p95 / avg.

## Commands

Python ≥3.12, dependencies via Poetry. All entry points are run as `poetry run python <script>`.

```bash
poetry install
poetry run ruff check .          # lint
poetry run pytest -q             # unit tests in tests/ (pythonpath configured in pyproject)

# Static benchmarks (orchestrator; loops distributions x models)
poetry run python run_experiments.py --scenario hard_narrow
# scenarios: simple_wide simple_narrow hard_wide hard_narrow hard_narrow_data_driven dsb stats_ceb stats_ceb_driven all

# Drift benchmarks (orchestrator; loops steps x models)
poetry run python run_drift_experiments.py --scenario drift_5_ds_zipf_wl_normal
# scenarios: drift_10_ds_zipf_wl_normal drift_5_ds_zipf_wl_normal
#            drift_5_ds_zipf_wl_zipf drift_5_ds_add_zipf_wl_zipf dsb_drift_ws_wholesale_cost dsb_drift_ws_list_price all

# Single model, single dataset (what the orchestrators shell out to)
poetry run python run_hybrid.py --experiment-name my_exp \
    --dataset data/generated/60000000_hard/normal \
    --workload workload/1000000/narrow/workload.csv

# The two reported experiments (static + drift, then tables/xlsx); B = 100 or fd
poetry run python run_experiments.py --scenario all --buckets 100 --out-dir results/exp1_100_buckets/static
poetry run python run_drift_experiments.py --scenario all --buckets 100 --out-dir results/exp1_100_buckets/drift
poetry run python export_results.py --results-dir results/exp1_100_buckets   # same with fd -> results/exp2_freedman_diaconis
poetry run python plot_drift_figures.py --results-dir results/exp1_100_buckets  # 6 PNG per drift scenario -> <exp>/figures/<scenario>/

# Single drift run — --shift-dir is required; --static-workload-path or --shift-workload
poetry run python run_hybrid_drift.py --experiment-name my_drift \
    --shift-dir "data/generated/60000000_hard/shift_normal_to_zipf_5%" \
    --workload 1000000 \
    --static-workload-path data/generated/60000000_hard/normal/workload_driven_1000000.csv
```

### Regenerating inputs (usually necessary)

`.gitignore` excludes `*.csv` and `/results/`. A fresh clone has `meta.pkl` / `step_*.pkl` / `stats.json` / plots committed, but **no `data.csv` and no workload CSVs** — so every run that touches a workload path will fail until they are regenerated:

```bash
poetry run python data/datasets.py --rows 60000000 --dist normal      # -> 60000000_hard/normal (--simple -> 60000000_simple)
poetry run python data/generate_shifts.py --rows 60000000_hard --init normal --target zipf --steps 20 --mode replace  # -> shift_normal_to_zipf_5% (--mode add -> ..._5%_dataset)
poetry run python workload/workload.py --count 1000000 --rows 60000000_hard  # data-driven workloads, batch over all subdirs
poetry run python workload/workload.py --count 1000000 --wide         # random workload -> data/workloads/wide_1000000.csv
poetry run python prepare_benchmark.py --bench stats-ceb              # STATS-CEB: download (pinned commit) + convert
poetry run python prepare_benchmark.py --bench dsb --generate-queries  # DSB: needs "DSB benchmark/setup_dsb.sh" + ./setup_env.sh --dsb
```

Budget / scale sweeps (in-process, not via the orchestrators): `poetry run python run_sweeps.py --part memory|width|sample|period|plots|all` writes `results/sweep_<part>/results.csv` + PNGs. `memory` (error vs number of stored floats, every model over a budget grid) is the equal-memory comparison; a single Freedman–Diaconis `k` is not. If `results/sweep_<part>_v1/` exists, its Hybrid rows are plotted as "Hybrid v1".

Models and scenarios for both orchestrators live in `registry.py` (`MODELS`, `STATIC_SCENARIOS`, `DRIFT_SCENARIOS`); the `MODELS` key must equal the name the runner writes to `summary_<Model>.json`. Both orchestrators accept `--models` and `--dry-run`.

Careful with workload paths: `registry.py` points to `workload/1000000/{wide,narrow}/workload.csv`, while `workload.py`'s random mode writes to `data/workloads/{mode}_{count}.csv`. The files must be placed/renamed to match.

Real benchmarks (`prepare_benchmark.py`): data and queries are taken from the original benchmark unchanged; the script is only an adapter into `meta.pkl` + `workload_official.csv` (+ `drift/step_N.pkl` for DSB). DSB uses `web_sales.ws_wholesale_cost` / `ws_list_price` (the columns official templates filter with BETWEEN), stored as integer cents; queries come from `dsqgen` on the unmodified official templates. STATS-CEB uses the official `stats_simplified` CSVs and `stats_CEB_single_table_sub_query.sql`; the adapter checks its truth against the cardinalities in that file. `workload_driven_*` for STATS-CEB is our supplement and goes to a separate experiment (`stats_ceb_driven`) — never mix it with the official workload. DSB drift reuses the synthetic `run_<model>_drift.py` runners via `DRIFT_SCENARIOS` (`dsb_drift_<column>`).

## Architecture

### The metadata contract

Almost every script consumes one of two pickled tuples. Getting the arity wrong is the most common breakage:

- **Dataset `meta.pkl`** — 8-tuple `(mn, mx, N, freq, sample, k, skew, kurt)`
- **Shift `step_N.pkl`** — 6-tuple `(mn, mx, N, freq, sample, k)`

`freq` is a dense per-value frequency array over `[mn, mx]` and is the **ground truth** (used to compute exact `y_true`). `sample` is a 100k reservoir sample and is the **only** thing models are allowed to build from — models see `sample` scaled by `N/len(sample)`, never `freq`. Keeping that separation intact is the point of the benchmark; a change that lets a model read `freq` directly invalidates results. `k` is a Freedman–Diaconis bucket count derived from the sample. Domain is fixed at `DOMAIN_MAX = 1_000_000` (`data/datasets.py:29`).

### Shared harness

`benchmark_utils.py` is the spine of the static path:

- `get_common_parser` — the `--dataset / --workload / --out-dir / --experiment-name / --buckets` contract every `run_*.py` inherits. `--buckets` is a string: an int (default 100) or `fd`; resolve it with `resolve_buckets(args.buckets, k)`. `size_metrics(model.n_values())` adds `n_values` / `size_bytes` (8 bytes per stored number); every model class has `n_values()`.
- `run_benchmark_suite(args, approach_fn)` — auto-detects whether `--dataset` is one dataset (has `meta.pkl`) or a directory of distributions, loops, and calls the script's `run_logic(args, out_dir, metadata)`. Per-dataset exceptions are caught and printed, not raised, so a run can "succeed" with missing outputs — check the log.
- `load_and_filter_workload` — computes exact truth from the `freq` prefix sum and **drops zero-truth queries**, so query counts differ per dataset.
- `evaluate_and_save` / `run_queries` — the one evaluation loop: static runners only build a model and pass `predict` (+ `feedback` for feedback methods, scored prequentially). Drift runners reuse `run_queries`, `timing_stats` and `exact_counts` (the only truth computation). Do not reintroduce per-runner timing loops.
- `save_benchmark_results` / `aggregate_summaries` — per-model `summary_<Model>.json` plus a rolled-up `results/<exp>/summary.csv`. Scientific notation is regex-rewritten to 9 decimals so timings stay readable.

The drift scripts (`run_*_drift.py`) do not use `run_benchmark_suite`; the main ones run on `drift_harness.run_drift` (one loop, see Drift protocol) and only describe the model: a build function for `SampleHistogram`, or a `DriftModel` subclass (`STHistDrift`, `HybridDrift`). `run_equihist_drift.py` still owns its old loop.

### Hybrid estimator

`HybridEstimator` is the contribution being evaluated; the rest are baselines.

- **Canonical spec — do not change it silently.** Do not add, replace or reorder CDF models or thresholds unless the change is explicitly labelled as a modification of the original Hybrid FD CDF. The training code follows the `dev` branch closely on purpose (easy merge).
- `train(freq, mn, points, rng, bucket_indices=None)` — `freq` is the sample's per-value counts scaled to N (never the ground truth). `_collect_cdf_training_rows` takes CDF points per bucket. `_train_adaptive_models` runs the **Waterfall** on 500 in-bucket validation queries (`_generate_bucket_queries`, median q-error by `_eval_model_q_error_vec`) and takes the **first** model that passes: Tier 0 Uniform (≤ 1.3) → Tier 1 Ridge with density weights: Linear, Power (√x), Log-linear → Tier 2 Isotonic, DecisionTree (max_depth 4) → Tier 3 Fourier-feature MLP (32 bands = 66 features, layers (64, 32)); Tiers 1–3 need ≤ 1.05. If nothing passes, the most accurate checked model stays. `parametric_only` stops after Tier 1. Buckets train in parallel via a `ThreadPoolExecutor`. poly3 / poly code paths in bake/predict are kept for loading old models only; training never produces them.
- `_bake_vectorized_data()` flattens the selected models into parallel numpy arrays plus an integer `mod_types` code (0 none, 1 linear, 2 poly, 3 other-sklearn, 4 log_linear, 5 power, 6 poly3, 7 isotonic, 8 tree). **Any mutation of `self.buckets` or `self.models` must be followed by a re-bake** — `EquiHistLearner.update` signals this by setting `self.b_lo = None`.
- `predict()` sums whole enclosed buckets from a prefix sum of `b_count` (O(1), as EquiWidth) and only invokes the local CDF model for the two edge buckets. Isotonic and DecisionTree models are evaluated from lists baked once per model (`_lookup`), not through sklearn per call; the predictions are identical (tests/test_hybrid.py).
- `feedback_update()` is the cheap adaptation path: redistributes observed vs. predicted mass over overlapping buckets and applies a clipped EMA to bucket counts. A bucket that is empty in the sample is revived at alpha × (rows per covered width over the queries) × bucket width; summing over queries overestimated it ~37x. It does not retrain models and never touches the CDF (Fine-Tuning: q-error > 1.2, alpha = 0.15).
- `cdf_choice()` counts buckets per chosen local CDF (`Uniform`, `Linear`, `Isotonic`, `FourierMLP`, ..., `Empty` for zero-count buckets); static summaries store it as `cdf_choice`, drift CSVs as a string.
- Instrumentation (never changes an estimate; a fingerprint of models + predictions was identical before/after): `selection[i]` records what the waterfall did in bucket i (tier, model, validation median/P95 q-error, passed, reason); `bucket_table()` gives one row per bucket with `bucket_values` such that `GLOBAL_VALUES + Σ bucket_values == n_values()`; `last_ft_idx` holds the buckets the last `feedback_update` touched. `run_hybrid.py` writes `bucket_table_<Model>.csv`; `SPEC_VERSION` labels the canonical method.

### Drift protocol

`drift_harness.py` runs every main model through the same loop. Rows changed per step = N(step 0) / (steps − 1): every scenario changes an equal share of the initial table per step (5 % or 10 %; DSB ~5 % churn). A step's queries are split by `benchmark_utils.split_adapt_eval` (seed 42): feedback may come only from the adapt half; everything is scored on the eval half. Never score on queries a model adapted to.

Baselines: at every step the stale model is scored (`before_*`), takes at most one action, whose wall time is recorded as `action_time_s`, and is scored again:
1. **rebuild** — rows changed since the last (re)build > 10 % of the table at that time (PostgreSQL autovacuum, `RebuildRule`); only `rebuild_rule == "10%"` models. EquiWidth / EquiDepth / TKHist / EntropyHist: built from scratch on the step's sample (the papers of TKHist and EntropyHist give no update rule, so they follow EW/ED).
2. **feedback update** — otherwise, for `uses_feedback` models: STHist (paper feedback + restructuring; `rebuild_rule = "never"`, it never reads data).
3. **none** — otherwise.

**Hybrid FD CDF** (`staged = True`, `rebuild_rule = "100k"`, `drift_harness._three_stages`) runs three stages one after another at every step — they are stages of one method, not alternatives:
1. **Shock** — the stale model, counts NOT refreshed (`before_*`).
2. **FT** — `feedback_update` on the adapt half (`ft_*`, `ft_time_s`). Then `mark()`: buckets whose median q-error over the adapt queries, re-estimated **after** FT, is still > 1.3 (`detect_bad_buckets`, > 5 queries) are marked; marks accumulate until a rebuild (`marked_buckets`, `mark_time_s`).
3. **RB** — only at a checkpoint: ≥ `CHECKPOINT_ROWS` = 100 000 rows changed since the last rebuild, checked at the step boundary (at most one rebuild per step; on 60M rows every step crosses it). Only marked buckets get a new count and a new Waterfall CDF from the step's sample; the rest are untouched (`rebuilt_buckets`, `rb_time_s`, `rb_*`). No checkpoint → RB = FT and `rebuilt_buckets = 0`. `action` is `FT + rebuild` or `FT`; `median_qerror` etc. are the RB score.
Never recompute all Hybrid counts from the new sample: it hides the effect of adaptation.

`--drift-total-pct` (default 100) sets the total drift over the steps as % of the step-0 table: rows changed per step = N0 · pct/100 / (steps − 1), and `drift_pct` runs 0 → pct (additive growth to +200 %: `--drift-total-pct 200`). `--drift-type` is a label. Every row has `checkpoint_triggered` (checkpoint / 10 % rule reached) and `rebuild_performed` (Hybrid: ≥ 1 marked bucket retrained; a checkpoint with 0 marks is not a rebuild), plus `ft_buckets`, MAE (`*_mae_rows`), Shock latency and memory. `HybridDrift.end_step` writes `bucket_tables/p<drift%>.csv` per step (CDF, tier, validation error, FT / marked / RB status per bucket).

`--shift-workload` switches from a fixed workload to per-step `step_N_workload_driven_<count>.csv` (workload drifts with the data); otherwise `--static-workload-path` is reused at every step. With `--buckets fd` the bucket count is step 0's `k`, fixed for the run.

### Paper experiments

`paper_experiments/` (run as `python -m paper_experiments.<module>`) is the thesis extension: DSB expansion, FD bucket reduction, memory analysis. `provenance.py` writes manifests (git commit + work-tree hash, since the tree is uncommitted; file hashes; `model_version`); `schema.py` is the row contract (`ID_COLUMNS`, stages `STATIC / SHOCK / FT / RB`, `drift_steps_to_long` — RB rows only where a rebuild was performed). Results go to `results/paper_experiments/<part>/`, never into `results/exp*`. Plan and file ownership: `results/paper_experiments/PLAN.md`.

### Output layout

Runs go to `results/exp1_100_buckets/` and `results/exp2_freedman_diaconis/` (all earlier results were deleted on 2026-11-10: they used the old Hybrid selection), each with `static/<benchmark>/<dataset>/` (summary JSONs, per-query CSVs, `model_selection_Hybrid.txt`) and `drift/<scenario>/gradual_<init>_to_<target>_<Model>/drift_steps.csv` (DSB: `gradual_<column>_to_refresh_<Model>`). `export_results.py --results-dir <exp>` writes the readable tables into `<exp>/` (`static_median_qerror.csv`, `histogram_size.csv`, `hybrid_cdf_choice.csv`, `drift_summary.csv`, ..., `results.xlsx`) and `combined.csv` + `drift_chart.png` per drift scenario.

## Conventions

- Comments and print statements are a mix of English and Russian; match the surrounding file rather than translating.
- Random seeds are fixed at 42 throughout (dataset generation, workload generation, model training) — keep new code seeded the same way for reproducibility. Never draw two random processes over the same rows from separate `default_rng(42)` instances: `prepare_dataset` samples with `[42, 1]` because a sampling mask from the same stream replayed the generator's uniforms and biased the sample (`tests/test_datasets.py`).
- Documentation lives in `docs/` (Russian); `README.md` is the English entry point.
- Per-query inference is timed inside the loop (`time.perf_counter()` per `predict`), so `predict` stays scalar and Python-level on purpose; the numpy work is in the bake step, not the hot loop.
