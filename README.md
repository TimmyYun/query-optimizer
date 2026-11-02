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
6. [Real benchmarks: DSB and STATS-CEB](#6-real-benchmarks-dsb-and-stats-ceb)
7. [Outputs](#7-outputs)
8. [Merging, plotting, exporting](#8-merging-plotting-exporting)
9. [Methodological guarantees](#9-methodological-guarantees)
10. [Reproducing the reported results](#10-reproducing-the-reported-results)
11. [Repository layout and documentation](#11-repository-layout-and-documentation)

**Results in short** (details and explanations in [docs/SUPERVISOR_SUMMARY.md](docs/SUPERVISOR_SUMMARY.md)):

- Hybrid is the most accurate sample-only method on narrow queries when the data
  is non-uniform *inside* a bucket: median q-error 1.23 against ≥ 1.44 for every
  other sample-only method on `hard/normal`. On the same distribution without
  intra-bucket structure (`simple/normal`) it gives no gain, 1.078 against 1.075.
- It does not help when the skew is a few heavy values (DSB: TKHist is best) or
  when queries land in a sparse tail, where the sample itself is the limit (zipf: EntropyHist is best).
- Under drift with a fixed workload, Hybrid with partial retraining (`Hybrid_RB`)
  is the best line at every step from 0 to 95 %. Its bucket grid is fixed at
  step 0, which limits it once the distribution has changed shape completely.

---

## 1. Methods

| Name in results | Model | Runner (static / drift) | Source |
|---|---|---|---|
| `EquiWidth` | equal-width histogram | `run_equiwidth.py` / `run_equiwidth_drift.py` | textbook baseline |
| `EquiDepth` | equal-depth histogram + MCV list, PostgreSQL `ANALYZE` style | `run_equidepth.py` / `run_equidepth_drift.py` | PostgreSQL `selfuncs.c` |
| `TKHist` | equal-width histogram + top-k exact values per bucket | `run_tkhist.py` / `run_tkhist_drift.py` | DHist-style |
| `STHist` | self-tuning histogram learned from query feedback | `run_sthist.py` / `run_sthist_drift.py` | Aboulnaga & Chaudhuri, SIGMOD 1999 |
| `EntropyHist-MEA` | maximum-entropy bucket boundaries, the paper's continuous range estimator | `run_entropyhist.py` / `run_entropyhist_drift.py` | To, Chiang & Shahabi, CIKM 2013 |
| `EntropyHist-MEA-Rng_discrete_linear` | the same histogram, range estimate over integer cells (`--range-estimator discrete_linear`, non-paper extension) | same runners | see below |
| `Hybrid` | **proposed**: equal-width buckets + local CDF model per bucket | `run_hybrid.py` / `run_hybrid_drift.py` | this work |
| `EquiHist` | online multiplicative bucket rescaling | `run_equihist.py` / `run_equihist_drift.py` | *not run by default (too slow); `--models EquiHist`* |

How each method works, and how each paper's open points were resolved, is
described in [METHODS_EXPLAINED.md](docs/METHODS_EXPLAINED.md).

Why EntropyHist is reported twice: the paper interpolates linearly over the
continuous bucket span, `overlap / (max_b − min_b)`. A query that touches only a
bucket's edge value therefore gets 0 rows. On discrete data with a point mass at
an edge this fails badly; on DSB, `ws_wholesale_cost BETWEEN 100.00 AND 120.00`
gets 0 rows for 19 440 true rows. The extension counts integer cells instead,
`(overlap + 1) / (width + 1)`. Both are reported, so the paper's estimator is not
silently replaced.

---

## 2. Quick start

Requirements: **Python ≥ 3.12**, [Poetry](https://python-poetry.org/) ≥ 2.0, ~4 GB free RAM.
Everything below is run from the repository root.

```bash
git clone https://github.com/TimmyYun/query-optimizer.git
cd query-optimizer
./setup_env.sh
```

`./setup_env.sh` with no flags does three things:

1. **Environment** — `poetry env use <python3.12>` + `poetry install`
   (runtime deps plus `pytest`, `ruff`, `nbconvert`, `ipykernel`).
2. **Synthetic workloads** — generates every synthetic query workload CSV. They are
   not in git (`.gitignore` excludes `*.csv`), so a fresh clone cannot run any
   experiment before this step.
3. **STATS-CEB** — downloads the official data and queries (pinned commit, ~6 MB)
   and converts them, see [§6.2](#62-stats-ceb).

The DSB inputs are committed (`data/dsb/<column>/meta.pkl`, `workload_official.csv`,
`drift/step_*.pkl`), so the DSB scenarios run on a fresh clone. The official DSB
toolkit is needed only to regenerate those inputs (hours), see [§6.1](#61-dsb).

You do **not** need to regenerate the 60M-row datasets: no runner reads
`data.csv`. Runners read only `meta.pkl` / `step_N.pkl`, which are committed and
already contain the ground truth and the sample (see [§9](#9-methodological-guarantees)).
The committed files are the reference inputs of the reported results.

| Flag | What it does |
|---|---|
| *(none)* | `--env` + `--workloads` + `--stats-ceb` |
| `--env` | Poetry environment only |
| `--workloads` | synthetic workloads only |
| `--stats-ceb` | download + convert STATS-CEB (needs network) |
| `--dsb` | DSB official queries + drift checkpoints; needs `DSB benchmark/setup_dsb.sh` first, see [§6.1](#61-dsb) |
| `--all` | all of the above |

Environment variables (optional):

| Variable | Default | Meaning |
|---|---|---|
| `PYTHON_BIN` | first `python3.12`, else `python3` on `PATH` | interpreter Poetry creates the venv from; must be ≥ 3.12 |
| `WORKLOAD_COUNT` | `1000000` | queries per generated workload |
| `DATASET_DIRS` | `"60000000_hard 60000000_simple"` | folders under `data/generated/` to build data-driven workloads for |
| `DSB_NUM_STREAMS` | `300` | DSB refresh streams to replay (`--dsb` only) |
| `DSB_TARGET_COLUMNS` | `web_sales.ws_wholesale_cost,web_sales.ws_list_price` | DSB columns tracked through the refresh streams (`--dsb` only) |

### Check the installation

```bash
poetry run pytest -q        # unit tests: harness, benchmark adapters, STHist, EntropyHist
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
> `meta.pkl`**. The data itself is reproduced exactly (seed 42). The sample is not
> always reproduced: the generator now draws the sample from its own random stream
> (`[42, 1]`). Before that fix, data and sample shared `default_rng(42)`, and the
> sampling mask replayed the generator's uniforms. That made the `simple/zipf`
> sample contain only values < 1000. The committed `simple/zipf` sample was rebuilt
> with the fix. The other committed samples predate it; they were checked to be
> representative (sample quantiles match the true ones), but regenerating them
> gives a different, equally valid sample, and results can move in the third
> decimal. Use the committed `meta.pkl` files to reproduce the reported numbers.
>
> `sparse_cluster` and `anti_zipf` have no separate simple variant in the
> generator, so their `simple` and `hard` data are identical.

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
dataset of the scenario, then prints one results table.

| Parameter | Default | Meaning |
|---|---|---|
| `--scenario` | required | one row of the table below, or `all` |
| `--models M [M ...]` | all except `EquiHist` | run only these models, e.g. `--models EquiWidth Hybrid`; names as in [§1](#1-methods) |
| `--dry-run` | off | print every command the scenario would run, without running anything |

Scenarios, models and their runner scripts are defined in one place,
[registry.py](registry.py). The orchestrators only read it.

| `--scenario` | Data | Workload | Experiment name in `results/` |
|---|---|---|---|
| `simple_wide` | `60000000_simple/*` | wide random | `synth_simple_wide` |
| `simple_narrow` | `60000000_simple/*` | narrow random | `synth_simple_narrow` |
| `hard_wide` | `60000000_hard/*` | wide random | `synth_hard_wide` |
| `hard_narrow` | `60000000_hard/*` | narrow random | `synth_hard_narrow` |
| `hard_narrow_data_driven` | `60000000_hard/*` | data-driven, per distribution | `synth_hard_narrow_data_driven` |
| `dsb` | `data/dsb/{ws_wholesale_cost, ws_list_price}` | official DSB queries ([§6.1](#61-dsb)) | `dsb_static` |
| `stats_ceb` | `data/stats-ceb/<table.Column>` × 9 | official STATS-CEB queries ([§6.2](#62-stats-ceb)) | `stats_ceb_official` |
| `stats_ceb_driven` | same 9 columns | our data-driven supplement, 100 000 queries | `stats_ceb_driven` |
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
step the step's own data provides the exact truth, and the workload is evaluated
in several **stages**. These stages become the column families of the per-step CSV.

**Adapt / eval split.** The queries of a step are split once, with seed 42, into
two halves (`benchmark_utils.split_adapt_eval`):

- *adapt*: the only queries whose feedback a model may learn from (Hybrid FT/RB, STHist, Equi-Hist);
- *eval*: the queries every stage of every model is scored on.

No model is ever scored on a query it adapted to. With a fixed workload the
split is the same at every step, so an eval query is never learned at any
step. All stages of all models share one eval set and can be compared directly.
The only exception is STHist **Online**, which is prequential over the adapt half:
each query is scored before the model learns from it.

| Model | Stages (all scored on the eval half unless noted) |
|---|---|
| Hybrid | **Shock**: bucket counts refreshed from the step's sample (`ANALYZE`), local models stale · **FT**: `feedback_update()` on the adapt half · **RB**: retrain the buckets whose median q-error on the adapt half > 1.3 |
| Equi-Width, EquiDepth, TKHist | simulated `ANALYZE`: rebuilt / refreshed from the step's sample, then evaluated |
| Equi-Hist | `ANALYZE` refresh, online feedback on the adapt half, then evaluated frozen |
| STHist | **Shock** (stale) · **Online** (prequential over the adapt half) · **Post** (frozen after the adapt half's feedback). No `ANALYZE`, since the model never sees data |
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
| `dsb_drift_ws_wholesale_cost`, `dsb_drift_ws_list_price` | `data/dsb/<column>/drift` (official refresh streams) | fixed: official DSB queries ([§6.1](#61-dsb)) |
| `all` | all of the above | |

`--models` and `--dry-run` work as in [§4.1](#41-full-scenarios).
Results go to `results/<scenario>/gradual_normal_to_zipf_<Model>/` (synthetic) or
`results/<scenario>/gradual_<column>_to_refresh_<Model>/` (DSB).

**Long-running:** 11–21 steps × 1M queries × 7 model variants per synthetic scenario,
40–100 minutes each (see [§10](#10-reproducing-the-reported-results)).

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

## 6. Real benchmarks: DSB and STATS-CEB

Principle: **the data and the queries come from the original benchmark,
unchanged**. This repository only adds an adapter, `prepare_benchmark.py`, that
converts them into the same contract as the synthetic data:

| File | Contents |
|---|---|
| `data/<bench>/<column>/meta.pkl` | `(mn, mx, N, freq, sample, k, skew, kurt)`, built by the same code as the synthetic datasets |
| `data/<bench>/<column>/workload_official.csv` | the 1-D ranges in the benchmark's own queries, one row per distinct range (committed to git) |
| `data/dsb/<column>/drift/step_<i>.pkl` | DSB only: the column after `5·i` % churn by the official refresh streams |
| `data/stats-ceb/<column>/workload_driven_100000.csv` | STATS-CEB only: **our** data-driven supplement, not the benchmark's queries |

Results on `workload_official.csv` and on `workload_driven_*` are written to
different experiments and must be reported separately.

`prepare_benchmark.py` parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `--bench` | required | `dsb` or `stats-ceb` |
| `--source-dir` | `DSB benchmark/dsb` / `data/stats-ceb/source` | DSB toolkit clone / where the STATS files are downloaded |
| `--out-root` | `data/dsb` / `data/stats-ceb` | output root |
| `--generate-queries` | off | DSB: run `dsqgen` on the official templates first |
| `--instances` | `1000` | DSB: query instances per template (`dsqgen -streams`) |
| `--rngseed` | `42` | DSB: `dsqgen -rngseed` |
| `--scale` | `1` | DSB: scale factor the data was generated with |
| `--driven-count` | `100000` | STATS-CEB: size of the data-driven supplement |
| `--sample-size` | `100000` (DSB) / `30000` (STATS-CEB) | rows in the model sample |

### 6.1 DSB

[DSB](https://github.com/microsoft/dsb) (Ding et al., VLDB 2021) is TPC-DS with skewed data and query templates.

| | Value |
|---|---|
| Toolkit | `microsoft/dsb` at commit `ec9a156` (2024-11-07), unmodified apart from portability compiler flags |
| Scale factor | 1 (`SCALE` in `setup_dsb.sh`) |
| Data seed | `dsdgen` built-in default; the data is deterministic for a given scale factor |
| Refresh streams | 300 generated (`UPDATE_STREAMS`), replayed until 100 % churn (`DSB_NUM_STREAMS`) |
| Columns | `web_sales.ws_wholesale_cost`, `web_sales.ws_list_price` (DECIMAL(7,2), stored as integer cents, ×100) |
| Queries | `dsqgen -streams 1000 -rngseed 42 -dialect postgres` on every official template in `query_templates_pg` that filters the column, as in the official `scripts/generate_workload.py` |

Why these columns: they are the `web_sales` columns that the official templates
filter with a plain `col BETWEEN a AND b`. Examples: 8 templates use
`ws_wholesale_cost BETWEEN [WHOLESALE_COST.begin] AND [WHOLESALE_COST.end]` with
`range(random(0, 100, uniform), 5…30)`, and 9 templates use `ws_list_price`.
The official refresh streams also modify `web_sales`. Ratio predicates such as
`ws_sales_price / ws_list_price BETWEEN …` are not 1-D and are skipped.

Run once (long: toolkit build, ~1.3 GB base data, ~5 GB refresh streams):

```bash
"DSB benchmark/setup_dsb.sh"     # Linux or macOS: clone at the pinned commit, build, dsdgen data + streams
./setup_env.sh --dsb             # replay streams (notebook), dsqgen queries, convert to meta.pkl / step_N.pkl
```

| `setup_dsb.sh` variable | Default | Meaning |
|---|---|---|
| `DSB_COMMIT` | `ec9a156…` | `microsoft/dsb` commit to check out |
| `SCALE` | `1` | scale factor in GB |
| `UPDATE_STREAMS` | `300` | refresh streams to generate |

`--dsb` runs `DSB benchmark/DSB_GENERATOR.ipynb` headlessly. The notebook applies
the official refresh streams (deletes by date range, inserts from the staging
files with the TPC-DS view logic: `ws_wholesale_cost = i_wholesale_cost`,
`ws_list_price = i_current_price`) and saves each column at 0, 5, …, 100 %
cumulative churn to `data/dsb/drift/`. The adapter then turns these checkpoints
into `step_0.pkl … step_20.pkl`.

The adapter prints how many distinct ranges each column gets. After a successful
run, commit `data/dsb/<column>/meta.pkl` and `workload_official.csv`, so that
`--scenario dsb` works for others without the toolkit.

Static and drift runs:

```bash
poetry run python run_experiments.py --scenario dsb
poetry run python run_drift_experiments.py --scenario dsb_drift_ws_wholesale_cost
poetry run python run_drift_experiments.py --scenario dsb_drift_ws_list_price
poetry run python merge_drift.py --experiment-name dsb_drift_ws_wholesale_cost dsb_drift_ws_list_price
poetry run python plot_drift.py  --experiment-name dsb_drift_ws_wholesale_cost dsb_drift_ws_list_price
```

DSB drift goes through the same `run_<model>_drift.py` runners as the synthetic
drift. A step is a churn checkpoint, and the official queries are evaluated at every step.

### 6.2 STATS-CEB

[STATS-CEB](https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark)
(Han et al., VLDB 2022): the Stack Exchange STATS database and its CEB workload.

| | Value |
|---|---|
| Source | `Nathaniel-Han/End-to-End-CardEst-Benchmark` at commit `670cb8d` (2023-03-14), downloaded automatically |
| Data | `datasets/stats_simplified/posts.csv`, `users.csv`; NULLs dropped (no range predicate selects NULL) |
| Sample | 30 000 rows, as PostgreSQL `ANALYZE` (300 × `default_statistics_target` = 100). The tables have 40 k–92 k rows, so the 100 k sample used elsewhere would contain the whole column (`--sample-size`) |
| Queries | `workloads/stats_CEB/sub_plan_queries/stats_CEB_single_table_sub_query.sql` |
| Columns | `posts.{Score, ViewCount, CommentCount, AnswerCount, FavoriteCount}`, `users.{Reputation, Views, UpVotes, DownVotes}`: every integer column with ≥ 8 distinct 1-D ranges. Timestamp columns are excluded, since second resolution does not fit a dense `freq` array |

How a query is taken: in every official single-table sub-query, the predicates on
one column (`>=, <=, =, >, <`) are intersected into one range `[lo, hi]`. An open
end becomes the column's min or max. Duplicates are removed. Sub-queries that
filter exactly one column also carry the benchmark's own cardinality; the adapter
checks that its truth equals it (48 / 48 match).

**Caveat:** the official workload has only 8–33 distinct ranges per column.
That is what the benchmark contains, but p95 over ~20 queries is not a stable
statistic. For statistical power, `stats_ceb_driven` evaluates the same official
data with 100 000 data-driven queries from our generator (width up to 5 % of the
domain). Report it as a separate, non-benchmark workload.

```bash
./setup_env.sh --stats-ceb                                          # or: poetry run python prepare_benchmark.py --bench stats-ceb
poetry run python run_experiments.py --scenario stats_ceb           # official queries  -> results/stats_ceb_official
poetry run python run_experiments.py --scenario stats_ceb_driven    # our supplement    -> results/stats_ceb_driven
```

`run_experiments.py` runs the adapter itself if the workloads are missing,
so the two `run_experiments.py` commands also work on a fresh clone with network access.

---

## 7. Outputs

```text
results/<experiment-name>/
├── summary.csv                                  # all models × datasets, one row each
├── <dataset>/                                   # static run
│   ├── summary_<Model>.json                     # metrics: median/p25/p75/p95/avg q-error, timings
│   └── <workload-stem>_<Model>.csv              # per query: Query_ID, Model, Prediction, Truth, Q_Error (selectivities)
├── gradual_<init>_to_<target>_<Model>/          # drift run (DSB: gradual_<column>_to_refresh_<Model>)
│   └── <model>_drift_analysis.csv               # one row per step (Hybrid: adaptation_ft_rb.csv)
└── combined_<prefix>.csv, drift_chart_<prefix>.png   # written by merge_drift.py / plot_drift.py
```

- **Zero-truth queries are dropped** before scoring, since q-error is undefined for them.
  The query count can therefore differ between datasets and is logged.
- Timing fields in `summary_<Model>.json`, the same for every method:

  | Field | Meaning |
  |---|---|
  | `build_time` | seconds to build the model before the first query |
  | `update_time_total` | seconds spent learning from query feedback (STHist, Equi-Hist; 0 otherwise) |
  | `train_time` | `build_time + update_time_total` |
  | `infer_time` | sum over queries of the time of one `predict()` call; feedback is not included |
  | `avg_infer_time`, `median_infer_time`, `p95_infer_time` | statistics of one `predict()` call |

- Small floats are written with 9 decimals instead of scientific notation.
- A runner that gets a parent `--dataset` folder catches each dataset's error,
  prints it and continues. **Check the log**: a run can finish with some outputs missing.

---

## 8. Merging, plotting, exporting

All results in one Excel workbook, one sheet per logical benchmark:

```bash
poetry run python export_results.py                # -> results/all_results.xlsx
```

| Parameter | Default | Meaning |
|---|---|---|
| `--out` | `results/all_results.xlsx` | output file |
| `--results-dir` | `results` | where to look |
| `--all` | off | also export folders that are not scenarios in `registry.py`, e.g. `smoke` |

The first sheet, `index`, lists each sheet: whether it is static or drift, its
data, its workload, the row count and the models. A static sheet has one row per
dataset × model with every metric. A drift sheet has one row per step × model × stage.
To the right of each table there is a pivot of the median q-error.

Drift runs, synthetic or DSB, any set of models:

```bash
poetry run python merge_drift.py --experiment-name drift_5_ds_zipf_wl_normal
poetry run python plot_drift.py  --experiment-name drift_5_ds_zipf_wl_normal
```

| Script | Parameter | Meaning |
|---|---|---|
| `merge_drift.py` | `--experiment-name E [E ...]` | folders under `results/` to merge. Every `<prefix>_<Model>/` folder inside is read, and one `combined_<prefix>.csv` is written per prefix (e.g. `gradual_normal_to_zipf`, `gradual_ws_list_price_to_refresh`) |
| | `--out-dir` | root of the results tree, default `results` |
| `plot_drift.py` | `--experiment-name E [E ...]` | plots every `combined_*.csv` of these experiments: median / p95 / mean q-error against drift % |
| | `--labels L [L ...]` | only these lines. A line is `<Model>` for single-stage models and `<Model>_<Stage>` otherwise, e.g. `--labels EquiWidth Hybrid_RB STHist_Online` |

`combined_<prefix>.csv` is a long table, one row per step × model × stage:
`drift, step, model, stage, label, Median, P95, Avg, Infer_Med, Infer_Avg, Adapt_Time, Rebuilt_Buckets, Initial_Build`.
`Adapt_Time` is the seconds a model spent at this step to reach this stage,
cumulative within the step (`Hybrid_RB` = sample refresh + FT + retraining).
`Initial_Build` is the step-0 build and is not part of any stage.

Static runs: box plots of the per-query q-error of several models:

```bash
poetry run python plot/plot_comparison.py \
    --experiments results/synth_hard_narrow/normal/*_*.csv \
    --output results/synth_hard_narrow/normal/comparison.png \
    --title "Q-Error: normal"
```

---

## 9. Methodological guarantees

- **Models never see the ground truth.** `meta.pkl` is the tuple
  `(mn, mx, N, freq, sample, k, skew, kurt)`. A shift step `step_N.pkl` is
  `(mn, mx, N, freq, sample, k)`. `freq` is the exact per-value count and is
  used **only** to compute the true answer. Models are built from `sample`
  (100 k rows for synthetic data and DSB; 30 k for STATS-CEB, see §6.2), scaled by `N / len(sample)`. The only other input
  is query feedback, for the feedback-driven methods. Code that lets a model
  read `freq` invalidates the results.
- **Same budget.** By default EquiWidth, TKHist, STHist, EntropyHist, Equi-Hist and Hybrid
  get the same bucket count `k` (the Freedman–Diaconis rule on the sample).
  EquiDepth follows PostgreSQL and uses `--statistics-target` (100) instead.
  Pass the same `--buckets B` to every runner to compare all of them at one budget.
  Buckets alone are not memory: TKHist also stores top-k values, and Hybrid stores
  a local model per bucket (`n_floats` in its summary).
- **Fixed seeds.** Data, samples, workloads and model training all use seed 42.
  Hybrid seeds each bucket's models from a fixed per-bucket seed, so results
  do not depend on thread scheduling.
- **One evaluation loop.** Every static runner only builds its model and passes
  `predict` (plus `feedback`, for feedback methods) to `benchmark_utils.evaluate_and_save`.
  Truth, clipping, timing and metrics are therefore computed by the same code for
  every method. Feedback methods are scored **prequentially**: the estimate of a
  query is recorded before the model learns from that query. Truth for static and
  drift runs comes from one function, `benchmark_utils.exact_counts`.

### Adding a method

1. `models/<name>.py`: build only from `sample`, `predict(q)` returns a row count. Export it in `models/__init__.py`.
2. `run_<name>.py`: copy [run_tkhist.py](run_tkhist.py). `run_logic` builds the model and calls `evaluate_and_save`.
3. `run_<name>_drift.py`: copy [run_tkhist_drift.py](run_tkhist_drift.py). If you change bucket counts, call `_bake_vectorized_data()` afterwards.
4. One line in `MODELS` in [registry.py](registry.py). Both orchestrators pick it up.

---

## 10. Reproducing the reported results

The numbers in [docs/](docs/) come from the commands below, run on the committed
inputs. Model training and query generation are seeded, so q-error is
reproduced exactly. Timings depend on the machine.

```bash
./setup_env.sh                                   # environment, synthetic workloads, STATS-CEB

# static: 8 scenarios
for s in simple_wide simple_narrow hard_wide hard_narrow hard_narrow_data_driven dsb stats_ceb stats_ceb_driven; do
    poetry run python run_experiments.py --scenario $s
done

# drift: 6 scenarios
DRIFT="drift_10_ds_zipf_wl_normal drift_5_ds_zipf_wl_normal drift_5_ds_zipf_wl_zipf \
       drift_5_ds_add_zipf_wl_zipf dsb_drift_ws_wholesale_cost dsb_drift_ws_list_price"
for s in $DRIFT; do poetry run python run_drift_experiments.py --scenario $s; done

poetry run python merge_drift.py --experiment-name $DRIFT
poetry run python plot_drift.py  --experiment-name $DRIFT
poetry run python export_results.py              # -> results/all_results.xlsx
```

Wall-clock time of the reported run (12-core Linux machine, 32 GB RAM, five
scenarios in parallel). Run them sequentially for clean timing measurements;
parallel runs inflate the absolute timings, but q-error does not depend on them.

| Scenario | Time |
|---|---|
| `simple_wide`, `hard_wide` | 21 min, 29 min |
| `simple_narrow`, `hard_narrow` | 10 min, 15 min |
| `hard_narrow_data_driven` | 21 min |
| `dsb`, `stats_ceb`, `stats_ceb_driven` | under a minute each |
| `drift_10_ds_zipf_wl_normal` | 38 min |
| `drift_5_ds_zipf_wl_normal`, `drift_5_ds_zipf_wl_zipf`, `drift_5_ds_add_zipf_wl_zipf` | 75 min, 101 min, 92 min |
| `dsb_drift_*` | under a minute each |

Launch long runs under `nohup` or `tmux`: a dropped
SSH session kills a foreground run, and `results/<experiment>/` would then be partial.
Delete a partial experiment folder before re-running it.

---

## 11. Repository layout and documentation

```text
models/                 estimators (one file per method) + common.py (Bucket, RangeQuery, q-error)
registry.py             models + static/drift scenarios: the single list the orchestrators read
benchmark_utils.py      shared CLI, dataset loop, truth, evaluation loop, result writing
run_<model>.py          static runner per model
run_<model>_drift.py    drift runner per model (synthetic and DSB)
run_experiments.py      static orchestrator (§4)
run_drift_experiments.py drift orchestrator (§5)
merge_drift.py, plot_drift.py  drift result tables and charts (§8)
export_results.py       every result -> one xlsx, one sheet per benchmark (§8)
run_hybrid_ablations.py Hybrid ablations: min sample points, partition, query width, equal memory
run_sthist_paper.py, run_entropyhist_paper.py  the original papers' own setups (not this benchmark)
data/datasets.py        synthetic dataset generator
data/generate_shifts.py drift sequence generator
data/generated/, data/dsb/, data/stats-ceb/  committed inputs: meta.pkl, step_N.pkl, official workloads
workload/workload.py    workload generator
plot/                   workload and per-query comparison plots
prepare_benchmark.py    adapter: DSB / STATS-CEB -> meta.pkl, workload_official.csv, drift/step_N.pkl
DSB benchmark/          setup_dsb.sh (official toolkit build + data), DSB_GENERATOR.ipynb (refresh replay)
tests/                  pytest unit tests
setup_env.sh            one-shot setup
docs/                   documentation (below)
```

Documentation, in Russian:

| Document | Read it for |
|---|---|
| [docs/SUPERVISOR_SUMMARY.md](docs/SUPERVISOR_SUMMARY.md) | **start here**: the results in one minute, how the conclusions changed from the first experiment and why, every protocol change, the drift stage glossary |
| [docs/RESULTS_EXPLAINED.md](docs/RESULTS_EXPLAINED.md) | how to read `all_results.xlsx`; for each method, why it wins or loses on each kind of data |
| [docs/RESULTS_REPORT.md](docs/RESULTS_REPORT.md) | every number by benchmark, threats to validity, anticipated reviewer questions |
| [docs/RESULTS_GUIDE.md](docs/RESULTS_GUIDE.md) | reference: every sheet and column of `all_results.xlsx`, and why these data and queries |
| [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) | what the benchmark does, why each design decision was made, list of fixes |
| [docs/METHODS_EXPLAINED.md](docs/METHODS_EXPLAINED.md) | how each method works and how the papers' open points were resolved |
