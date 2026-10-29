# Range-Query Selectivity Estimation Benchmark

A reproducible benchmark for **cardinality / selectivity estimation of 1-D range
predicates** (`WHERE col BETWEEN low AND high`) on a single integer column.
It compares a proposed **Hybrid** estimator (histogram + per-bucket learned CDF)
with classical and self-tuning histogram baselines, both on **static data** and
under **gradual data drift**.

Every estimate is scored by **q-error**, `max(est/true, true/est) ≥ 1`,
reported as median / p25 / p75 / p95 / mean, plus build and per-query inference time.

---

## Contents

1. [Methods](#1-methods)
2. [Quick start](#2-quick-start)
3. [Data and workloads](#3-data-and-workloads)
4. [Static experiments](#4-static-experiments)
5. [Drift experiments](#5-drift-experiments)
6. [DSB (real data)](#6-dsb-real-data)
7. [Outputs](#7-outputs)
8. [Merging and plotting](#8-merging-and-plotting)
9. [Methodological guarantees](#9-methodological-guarantees)
10. [Repository layout](#10-repository-layout)

---

## 1. Methods

| Name in results | Model | Runner (static / drift) | Source |
|---|---|---|---|
| `EquiWidth` | equal-width histogram | `run_equiwidth.py` / `run_equiwidth_drift.py` | textbook baseline |
| `EquiDepth` | equal-depth histogram + MCV list, PostgreSQL `ANALYZE` style | `run_equidepth.py` / `run_equidepth_drift.py` | PostgreSQL `selfuncs.c` |
| `TKHist` | equal-width histogram + top-k exact values per bucket | `run_tkhist.py` / `run_tkhist_drift.py` | DHist-style |
| `STHist` | self-tuning histogram learned from query feedback | `run_sthist.py` / `run_sthist_drift.py` | Aboulnaga & Chaudhuri, SIGMOD 1999 |
| `EntropyHist-MEA` | maximum-entropy bucket boundaries | `run_entropyhist.py` / `run_entropyhist_drift.py` | To, Chiang & Shahabi, CIKM 2013 |
| `Hybrid` | **proposed**: equal-width buckets + local CDF model per bucket | `run_hybrid.py` / `run_hybrid_drift.py` | this work |
| `Equi-Hist` | online multiplicative bucket rescaling | `run_equihist.py` / `run_equihist_drift.py` | *excluded from the orchestrators (too slow); can be run by hand* |

How each method works, and how each paper's open points were resolved, is
described in [METHODS_EXPLAINED.md](METHODS_EXPLAINED.md).

---

## 2. Quick start

Requirements: **Python ≥ 3.12**, [Poetry](https://python-poetry.org/) ≥ 2.0, ~4 GB free RAM.
Everything below is run from the repository root.

```bash
git clone https://github.com/TimmyYun/query-optimizer.git
cd query-optimizer
./setup_env.sh
```

`./setup_env.sh` with no flags does two things:

1. **Environment** — `poetry env use <python3.12>` + `poetry install`
   (runtime deps plus `pytest`, `ruff`, `nbconvert`, `ipykernel`).
2. **Workloads** — generates every query workload CSV. They are not in git
   (`.gitignore` excludes `*.csv`), so a fresh clone cannot run any experiment
   before this step.

You do **not** need to regenerate the 60M-row datasets: no runner reads
`data.csv`. Runners read only `meta.pkl` / `step_N.pkl`, which are committed and
already contain the ground truth and the sample (see [§9](#9-methodological-guarantees)).

| Flag | What it does |
|---|---|
| *(none)* | `--env` + `--workloads` |
| `--env` | Poetry environment only |
| `--workloads` | synthetic + DSB workloads only |
| `--dsb-drift` | DSB drift checkpoints; needs the DSB toolkit first, see [§6](#6-dsb-real-data) |
| `--all` | all of the above |

Environment variables (optional):

| Variable | Default | Meaning |
|---|---|---|
| `PYTHON_BIN` | first `python3.12`, else `python3` on `PATH` | interpreter Poetry creates the venv from; must be ≥ 3.12 |
| `WORKLOAD_COUNT` | `1000000` | queries per generated workload |
| `DATASET_DIRS` | `"60000000_hard 60000000_simple"` | folders under `data/generated/` to build data-driven workloads for |
| `DSB_NUM_STREAMS` | `300` | DSB refresh streams to replay (`--dsb-drift` only) |

### Check the installation

```bash
poetry run pytest -q        # unit tests for STHist and EntropyHist
poetry run ruff check .     # lint
```

Smoke test, one model on one dataset, takes under a minute:

```bash
poetry run python run_equiwidth.py --experiment-name smoke \
    --dataset data/generated/60000000_hard/normal \
    --workload workload/1000000/narrow/workload.csv
```

It should create `results/smoke/normal/summary_EquiWidth.json` and `results/smoke/summary.csv`.

---

## 3. Data and workloads

### 3.1 Synthetic datasets

60M rows each, integer domain `[0, 1 000 000]`, five distributions:
`normal`, `zipf`, `anti_zipf`, `uniform`, `sparse_cluster`.

| Folder | Variant |
|---|---|
| `data/generated/60000000_simple/<dist>` | the plain distribution |
| `data/generated/60000000_hard/<dist>` | the same distribution with injected micro-distributions (local spikes and gaps), so a bucket's contents are far from uniform |

Each folder contains `meta.pkl`, `stats.json` and plots, all committed.
They are regenerated only if you want to change the data. Run:

```bash
poetry run python data/datasets.py --rows 60000000 --dist normal          # -> 60000000_hard/normal
poetry run python data/datasets.py --rows 60000000 --dist normal --simple # -> 60000000_simple/normal
poetry run python data/datasets.py --rows 60000000 --all                  # all 5 distributions
```

| Parameter | Meaning |
|---|---|
| `--rows N [N ...]` | number of rows to generate; one dataset per value |
| `--dist NAME` | one distribution: `uniform`, `normal`, `zipf`, `sparse_cluster`, `anti_zipf` |
| `--all` | all five distributions; without `--rows` it also means 1M, 10M and 60M rows |
| `--simple` | no micro-distributions; output goes to `<rows>_simple/` instead of `<rows>_hard/` |

> This writes a ~500 MB `data.csv` per dataset and **overwrites the committed
> `meta.pkl`**. With seed 42 the result should match; check with `git diff --stat data/generated`.

### 3.2 Drift sequences

A drift sequence is a folder of `step_0.pkl … step_S.pkl`. Step `s` mixes the
initial distribution `I` and the target `T` with weight `a = s/S`.
Three sequences are committed:

| Folder | `--mode` | `--steps` | Mixture at step `s` |
|---|---|---|---|
| `60000000_hard/shift_normal_to_zipf_10%` | `replace` | 10 | `(1-a)·I + a·T`, N stays 60M |
| `60000000_hard/shift_normal_to_zipf_5%` | `replace` | 20 | `(1-a)·I + a·T`, N stays 60M |
| `60000000_hard/shift_normal_to_zipf_5%_dataset` | `add` | 20 | `I + a·T`, N grows 60M → 120M (inserts only) |

To regenerate one (it needs `I/meta.pkl` and `T/meta.pkl` in the same `--rows` folder):

```bash
poetry run python data/generate_shifts.py --rows 60000000_hard --init normal --target zipf --steps 20 --mode replace
```

| Parameter | Default | Meaning |
|---|---|---|
| `--rows` | `60000000_hard` | folder under `data/generated/` holding both source distributions |
| `--init` | `normal` | distribution at step 0 |
| `--target` | `zipf` | distribution at the last step |
| `--steps` | `20` | number of steps after step 0; step size is `100/steps` % |
| `--mode` | `replace` | `replace`: `I` is gradually replaced by `T`. `add`: `T` is inserted on top of `I`, nothing is deleted |

The output folder is `shift_<init>_to_<target>_<100/steps>%`, with the suffix `_dataset` for `--mode add`.
It also gets an `animation_*.gif` of the transition.

### 3.3 Workloads

A workload is a CSV `low,high` of inclusive integer ranges. `./setup_env.sh --workloads`
builds all of them. The underlying generator is `workload/workload.py`:

| Kind | How a query is drawn | Command | Output |
|---|---|---|---|
| **narrow** (random) | `low ~ U[0, 1e6)`, width `~ U[1, max-width)` | `workload.py --count 1000000` | `data/workloads/narrow_1000000.csv` |
| **wide** (random) | `low ~ U[0, 1e6)`, width `~ U[10%, 100%)` of the domain | `workload.py --count 1000000 --wide` | `data/workloads/wide_1000000.csv` |
| **data-driven** | centre drawn from the dataset's *sample*, width `~ U[1, max-width)`, so queries hit where the data is | `workload.py --count 1000000 --rows 60000000_hard` | `<dataset>/workload_driven_1000000.csv`, and `step_N_workload_driven_1000000.csv` in every shift folder |

| Parameter | Default | Meaning |
|---|---|---|
| `--count` | required | number of queries |
| `--rows` | — | folder under `data/generated/`; switches to data-driven mode and processes every sub-folder |
| `--dist` | — | with `--rows`: process only this sub-folder |
| `--wide` | off | random mode only: wide instead of narrow queries |
| `--max-width` | `1000` | maximum width of narrow and data-driven queries |
| `--domain-max` | `1000000` | upper end of the domain for random queries |

> `run_experiments.py` reads the random workloads from
> `workload/1000000/{wide,narrow}/workload.csv`. `setup_env.sh` moves them there.
> If you run `workload.py` by hand, move them yourself.

All generators use seed 42.

---

## 4. Static experiments

### 4.1 Full scenarios

```bash
poetry run python run_experiments.py --scenario hard_narrow
```

Each scenario runs every model in [§1](#1-methods) except Equi-Hist on every
distribution of its dataset, then prints one results table.

| `--scenario` | Data | Workload | Experiment name in `results/` |
|---|---|---|---|
| `simple_wide` | `60000000_simple/*` | wide random | `synth_simple_wide` |
| `simple_narrow` | `60000000_simple/*` | narrow random | `synth_simple_narrow` |
| `hard_wide` | `60000000_hard/*` | wide random | `synth_hard_wide` |
| `hard_narrow` | `60000000_hard/*` | narrow random | `synth_hard_narrow` |
| `hard_narrow_data_driven` | `60000000_hard/*` | data-driven, per distribution | `synth_hard_narrow_data_driven` |
| `dsb` | `data/dsb/{ws_item_sk, cr_item_sk, cr_returned_time_sk}` | DSB workload | `dsb_static` |
| `all` | all of the above, in order | | |

### 4.2 One model, one dataset

Every static runner shares these parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `--dataset PATH` | required | a folder with `meta.pkl`, **or** a parent folder: then every sub-folder with a `meta.pkl` is run in turn |
| `--workload PATH` | required | workload CSV |
| `--experiment-name NAME` | timestamp | results go to `results/NAME/<dataset-folder>/` |
| `--out-dir DIR` | `results` | root of the results tree |
| `--buckets B` | Freedman–Diaconis `k` from `meta.pkl` | force the bucket count; use it to compare methods under the same budget |

```bash
poetry run python run_hybrid.py --experiment-name my_exp \
    --dataset data/generated/60000000_hard/normal \
    --workload data/generated/60000000_hard/normal/workload_driven_1000000.csv
```

Model-specific parameters:

| Runner | Parameter | Default | Meaning |
|---|---|---|---|
| `run_hybrid.py` | `--points` | `200` | CDF points sampled per bucket to train its local model |
| | `--ident` | `1e-4` | below this error a bucket is treated as uniform (no model) |
| | `--penalty` | `1.5` | complexity penalty when choosing between candidate local models |
| | `--min-sample-points` | `0` | train a local model only if the bucket holds at least this many sample points |
| | `--partition` | `equiwidth` | bucket boundaries: `equiwidth`, or `equidepth` (sample quantiles) |
| | `--parametric-only` | off | only linear / power / log / poly3 local models (2–4 floats each); no trees or MLP |
| | `--no-debug-plot` | off | skip the O(Q×B) per-bucket debug plot; recommended for 1M queries |
| `run_equidepth.py` | `--statistics-target` | `100` | PostgreSQL `default_statistics_target`: number of MCVs and histogram buckets; the sample is 300× this |
| | `--with-index` | off | emulate a B-tree on the column: edge buckets use the real min/max |
| `run_tkhist.py` | `--topk` | `5` | most frequent values stored exactly per bucket |
| | `--n-categorical` | `200` | below this many distinct values every value is stored exactly |
| `run_sthist.py` | `--alpha` | `0.5` | feedback damping factor |
| | `--restructure-every` | `200` | restructure (merge + split) every R queries |
| | `--merge-threshold` | `0.00025` | merge buckets whose frequencies differ by less than this fraction of N |
| | `--split-fraction` | `0.10` | fraction of buckets freed by merging and re-used for splits |
| | `--no-restructure` | off | ablation: frequency refinement only (model name `STHist-NoRS`) |
| | `--mode` | `paper_faithful` | `robust_extended` turns on non-paper extensions; report it separately |
| `run_entropyhist.py` | `--method` | `ME` | builder objective: `ME`, `MSE` or `MB` |
| | `--me-input` | `area` | ME on area (`ME(A)`) or on frequency (`ME(F)`) |
| `run_equihist.py` | `--lr` | `0.5` | learning rate of the multiplicative feedback update |

Any non-default choice changes the model name in the results (e.g. `EntropyHist-MSE`),
so it cannot be mistaken for the baseline.

Paper-reproduction scripts (the original papers' own setups, not this benchmark):
`run_sthist_paper.py` (STHist Table 2, §5.4–5.5) and `run_entropyhist_paper.py`.
Run them with `--help` to see their parameters.

---

## 5. Drift experiments

### 5.1 Protocol

The estimator is built on step 0 and then carried through steps 1…S. At every
step the step's own data provides the exact truth, and the same workload is
evaluated in several **stages**. These stages become the column families of the
per-step CSV:

| Model | Stages |
|---|---|
| Hybrid | **Shock**: stale model, no adaptation · **FT**: `feedback_update()` rescales bucket mass from query feedback · **RB**: retrain only buckets whose q-error > 1.3 |
| Equi-Width, EquiDepth, TKHist | simulated `ANALYZE`: bucket counts refreshed from the step's sample, then evaluate |
| Equi-Hist | `ANALYZE` refresh, then online feedback updates after each query |
| STHist | **Shock** (stale) · **Online** (prequential: predict, then learn) · **Post** (frozen after the step's feedback). There is no `ANALYZE`, since the model never sees data |
| EntropyHist | **Static** (step-0 histogram, frozen) · **Rebuild** (rebuilt from the step's sample) |

### 5.2 Full scenarios

```bash
poetry run python run_drift_experiments.py --scenario drift_5_ds_zipf_wl_normal
```

| `--scenario` | Shift folder | Workload |
|---|---|---|
| `drift_10_ds_zipf_wl_normal` | `shift_normal_to_zipf_10%` | fixed: `normal/workload_driven_1000000.csv` |
| `drift_5_ds_zipf_wl_normal` | `shift_normal_to_zipf_5%` | fixed: `normal/workload_driven_1000000.csv` |
| `drift_5_ds_zipf_wl_zipf` | `shift_normal_to_zipf_5%` | drifts with the data: `step_N_workload_driven_1000000.csv` |
| `drift_5_ds_add_zipf_wl_zipf` | `shift_normal_to_zipf_5%_dataset` | drifts with the data |
| `all` | all four | |

**Long-running:** 11–21 steps × 1M queries × 6 models per scenario.

### 5.3 One model

```bash
poetry run python run_hybrid_drift.py --experiment-name my_drift \
    --shift-dir "data/generated/60000000_hard/shift_normal_to_zipf_5%" \
    --workload 1000000 \
    --static-workload-path data/generated/60000000_hard/normal/workload_driven_1000000.csv
```

| Parameter | Meaning |
|---|---|
| `--shift-dir PATH` | folder with `step_*.pkl`. Always pass it explicitly; the fallback built from `--dataset/--init-dist/--target-dist` points to `..._5%_dataset` |
| `--workload COUNT` | query count, used to find `*_workload_driven_<COUNT>.csv` files |
| `--static-workload-path PATH` | the fixed workload evaluated at every step |
| `--shift-workload` | use `step_N_workload_driven_<COUNT>.csv` from `--shift-dir` at step N instead; drop `--static-workload-path` |
| `--init-dist`, `--target-dist` | default `normal`, `zipf`. They only label the output folder `gradual_<init>_to_<target>_<Model>` |
| `--experiment-name`, `--out-dir` | as in the static runners |

The model-specific parameters from [§4.2](#42-one-model-one-dataset) apply too
(e.g. `--points`, `--topk`, `--statistics-target`, `--buckets`).

---

## 6. DSB (real data)

Three columns from [DSB](https://github.com/microsoft/dsb) (Ding et al., VLDB 2021),
a TPC-DS derivative with skewed, correlated data:
`web_sales.ws_item_sk`, `catalog_returns.cr_item_sk`, `catalog_returns.cr_returned_time_sk`.

### 6.1 Static — works out of the box

`data/dsb/<column>/meta.pkl` and the query files `data/dsb/custom_<column>.sql` are committed.
`./setup_env.sh` turns the `.sql` into `data/dsb/<column>/workload_driven_*.csv`.
After that:

```bash
poetry run python run_experiments.py --scenario dsb
```

To rebuild only the workloads: `poetry run python prepare_dsb.py`.
`prepare_dsb.py --rebuild-meta` also rebuilds `meta.pkl` from `data/dsb/<column>.csv`.
Those CSVs exist only after the DSB generation below, and the rebuild overwrites
the committed file, so check `git diff --stat data/dsb` afterwards.

### 6.2 Data generation and drift (long-running)

```bash
"DSB benchmark/setup_macos.sh"      # macOS; "DSB benchmark/setup.sh" is the original Linux script
./setup_env.sh --dsb-drift          # replays refresh streams, exports 63 checkpoints to data/dsb/drift/
```

The setup script clones microsoft/dsb, builds `dsdgen`/`dsqgen`, and generates
SF=1 base data (~1.3 GB) plus 300 refresh streams (~5.1 GB).
`--dsb-drift` runs `DSB benchmark/DSB_GENERATOR.ipynb` headlessly. The notebook
replays the official refresh streams and saves each column at 0, 5, …, 100 % churn.
300 streams is the verified minimum: with 200, both `catalog_returns` columns stop at 80 % churn.

Then, per column:

```bash
for col in ws_item_sk cr_item_sk cr_returned_time_sk; do
    poetry run python run_dsb_equiwidth_drift.py --column $col --experiment-name dsb_drift
    poetry run python run_dsb_hybrid_drift.py    --column $col --experiment-name dsb_drift
    for m in tkhist equidepth sthist entropyhist; do
        poetry run python run_dsb_baselines_drift.py --model $m --column $col --experiment-name dsb_drift
    done
done
```

| Parameter | Meaning |
|---|---|
| `--column` | `ws_item_sk`, `cr_item_sk` or `cr_returned_time_sk` |
| `--model` | (`run_dsb_baselines_drift.py`) `tkhist`, `equidepth`, `sthist` or `entropyhist` |
| `--data-dir` | default `data/dsb`; root holding `<column>/meta.pkl` and `drift/` |

Details of the DSB pipeline: [DSB benchmark/Readme.md](DSB%20benchmark/Readme.md).

---

## 7. Outputs

```text
results/<experiment-name>/
├── summary.csv                                  # all models × datasets, one row each
├── <dataset>/                                   # static run
│   ├── summary_<Model>.json                     # metrics: median/p25/p75/p95/avg q-error, timings
│   └── <workload-stem>_<Model>.csv              # per query: Query_ID, Model, Prediction, Truth, Q_Error (selectivities)
└── gradual_<init>_to_<target>_<Model>/          # drift run
    └── <model>_drift_analysis.csv               # one row per step (Hybrid: adaptation_ft_rb.csv)
```

- **Zero-truth queries are dropped** before scoring, since q-error is undefined for them.
  The query count can therefore differ between datasets and is logged.
- `train_time` is the build time in seconds. `infer_time` is the wall time of
  the whole query loop. `avg_infer_time`, `median_infer_time` and `p95_infer_time`
  are statistics of the time of one scalar `predict()` call.
- Small floats are written with 9 decimals instead of scientific notation.
- A runner that gets a parent `--dataset` folder catches each dataset's error,
  prints it and continues. **Check the log**: a run can finish with some outputs missing.

---

## 8. Merging and plotting

```bash
poetry run python merge_dsb_drifts.py     # -> results/dsb_drift/combined_<column>.csv
poetry run python plot_dsb_drift.py       # median / p95 / mean q-error per churn level
poetry run python merge_all_drifts.py     # synthetic drift, four scenarios
poetry run python plot/plot_comparison.py \
    --experiments results/synth_hard_narrow/normal/*_*.csv \
    --output results/synth_hard_narrow/normal/comparison.png \
    --title "Q-Error: normal"
```

> These scripts currently have **hard-coded result paths**
> (`results/dsb_drift`, `results/equihist_drift_5%/…`).
> For the DSB ones, keep `--experiment-name dsb_drift`.

---

## 9. Methodological guarantees

- **Models never see the ground truth.** `meta.pkl` is the tuple
  `(mn, mx, N, freq, sample, k, skew, kurt)`. A shift step `step_N.pkl` is
  `(mn, mx, N, freq, sample, k)`. `freq` is the exact per-value count and is
  used **only** to compute the true answer. Models are built from `sample`
  (a 100k reservoir sample), scaled by `N / len(sample)`. The only other input
  is query feedback, for the feedback-driven methods. Code that lets a model
  read `freq` invalidates the results.
- **Same budget.** By default every method gets the same bucket count `k`
  (the Freedman–Diaconis rule on the sample). Use `--buckets` to set another shared value.
- **Fixed seeds.** Data, samples, workloads and model training all use seed 42.
  Hybrid seeds each bucket's models from a fixed per-bucket seed, so results
  do not depend on thread scheduling.
- **Same timing method.** Every method is timed per query with `time.perf_counter()`
  around a scalar `predict()` call.

---

## 10. Repository layout

```text
models/                 estimators (one file per method) + common.py (Bucket, RangeQuery, q-error)
benchmark_utils.py      shared CLI, dataset loop, truth computation, result writing
run_<model>.py          static runner per model
run_<model>_drift.py    synthetic drift runner per model
run_dsb_*_drift.py      DSB drift runners
run_experiments.py      static orchestrator (§4)
run_drift_experiments.py drift orchestrator (§5)
data/datasets.py        synthetic dataset generator
data/generate_shifts.py drift sequence generator
workload/workload.py    workload generator
prepare_dsb.py          DSB adapter: .sql -> workload csv, column csv -> meta.pkl
DSB benchmark/          DSB toolkit setup scripts + drift notebook
tests/                  pytest unit tests
setup_env.sh            one-shot setup
```

Other documents:

- [METHODS_EXPLAINED.md](METHODS_EXPLAINED.md): how each method works.
- [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) and [RESULTS_REPORT.md](RESULTS_REPORT.md):
  results and their interpretation, in Russian.
