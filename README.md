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

**Results:** the two experiments (100 buckets and Freedman–Diaconis) are written to
`results/exp1_100_buckets/` and `results/exp2_freedman_diaconis/` (§8). Earlier results were
deleted on 2026-11-10 because they used an older Hybrid model selection; rerun to regenerate them.

---

## 1. Methods

| Name in results | Model | Runner (static / drift) | Source |
|---|---|---|---|
| `EquiWidth` | equal-width histogram | `run_equiwidth.py` / `run_equiwidth_drift.py` | textbook baseline |
| `EquiDepth` | equal-depth histogram + MCV list, PostgreSQL `ANALYZE` style | `run_equidepth.py` / `run_equidepth_drift.py` | PostgreSQL `selfuncs.c` |
| `TKHist` | equal-width histogram + top-k exact values per bucket | `run_tkhist.py` / `run_tkhist_drift.py` | DHist-style |
| `STHist` | self-tuning histogram learned from query feedback | `run_sthist.py` / `run_sthist_drift.py` | Aboulnaga & Chaudhuri, SIGMOD 1999 |
| `EntropyHist-MEA` | maximum-entropy bucket boundaries, the paper's continuous range estimator | `run_entropyhist.py` / `run_entropyhist_drift.py` | To, Chiang & Shahabi, CIKM 2013 |
| `EntropyHist-MEA-Rng_discrete_linear` | the same histogram, range estimate over integer cells (`--range-estimator discrete_linear`, non-paper extension) | same runners | *not run by default; `--models ...`* |
| `Hybrid` | **proposed** (Hybrid FD CDF): equal-width buckets + local CDF per bucket, chosen by a Waterfall Uniform → Linear/Power/Log-linear → Isotonic/DecisionTree → Fourier-MLP | `run_hybrid.py` / `run_hybrid_drift.py` | this work |
| `EquiHist` | online multiplicative bucket rescaling | `run_equihist.py` / `run_equihist_drift.py` | *not run by default (too slow); `--models EquiHist`* |

How each method works, and how each paper's open points were resolved, is
described in [METHODS_EXPLAINED.md](docs/METHODS_EXPLAINED.md).

The orchestrators run the six main methods by default: `EquiWidth`, `EquiDepth`,
`TKHist`, `EntropyHist-MEA`, `STHist`, `Hybrid`. Every model gets **100 buckets**
unless `--buckets fd` asks for the Freedman–Diaconis count `k` of the data.

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

Each scenario runs the six main models of [§1](#1-methods) on every
dataset of the scenario, then prints one results table.

| Parameter | Default | Meaning |
|---|---|---|
| `--scenario` | required | one row of the table below, or `all` |
| `--models M [M ...]` | the six main models | run only these models, e.g. `--models EquiWidth Hybrid`; names as in [§1](#1-methods) |
| `--buckets B` | `100` | bucket count of every model, or `fd`: the Freedman–Diaconis `k` stored in `meta.pkl` |
| `--out-dir DIR` | `results` | results root passed to every runner |
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
| `--buckets B` | `100` | bucket count, or `fd` for the Freedman–Diaconis `k` from `meta.pkl` (n = rows of the table, capped at 2000) |

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
| | `--parametric-only` | off | Waterfall stops after Tier 1 (linear / power / log, 2 floats each); no trees or MLP |
| | `--debug-plot` | off | per-bucket error plot; slow, O(queries × buckets) in Python |
| `run_equidepth.py` | `--buckets` | `100` | is PostgreSQL `default_statistics_target`: number of MCVs and histogram buckets; the sample is 300× this |
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

One loop, [drift_harness.py](drift_harness.py), runs every model, so all models
see the same steps, queries and rebuild rule. The model is built on step 0 and
carried through steps 1…S. Every scenario changes an equal share of the initial
table per step (synthetic: 5 % or 10 % of the rows; DSB: the official refresh
streams, ~5 % churn per step).

**Adapt / eval split.** The queries of a step are split once, with seed 42, into
two halves (`benchmark_utils.split_adapt_eval`):

- *adapt*: the only queries whose feedback a model may learn from (Hybrid, STHist);
- *eval*: the queries every model is scored on.

No model is ever scored on a query it adapted to.

**Baselines: one action per step.** At every step the stale model is scored on the eval half
(`before_*` columns). Then it takes at most one action, whose wall time is measured
(`action_time_s`), and it is scored again (`median_qerror`, `p95_qerror`, ...):

| Action | When | Who |
|---|---|---|
| `rebuild` | rows changed since the model's last (re)build exceed **10 %** of the table at that time (the PostgreSQL autovacuum rule, `autovacuum_analyze_scale_factor = 0.1`). With 5 % per step that is every 3rd step | EquiWidth, EquiDepth (as PostgreSQL), TKHist and EntropyHist (their papers give no update rule, so they follow EquiWidth/EquiDepth): built from scratch on the step's sample |
| `feedback update` | every step without a rebuild | STHist: the paper's feedback rule on every adapt query, restructuring every 200 queries; it never reads data, so it is never rebuilt |
| `none` | otherwise | the sample-based histograms between rebuilds |

**Hybrid FD CDF: Shock → FT → Rebuild at every step.** The three are consecutive stages of one
method, each scored on the eval half:

| Stage | What happens | Columns |
|---|---|---|
| Shock | the model as it is; bucket counts are **not** refreshed, so the degradation after drift is visible | `before_*` |
| FT | `feedback_update()` on the adapt half: EMA (alpha 0.15) on the counts of buckets hit by queries with q-error > 1.2; the CDF never changes. Then buckets whose median q-error on the adapt queries, re-estimated after FT, is still > 1.3 are marked | `ft_*`, `ft_time_s`, `marked_buckets`, `mark_time_s` |
| RB | only at a checkpoint: ≥ 100 000 rows changed since the last rebuild, checked at the step boundary (one rebuild per step at most). Marked buckets get a new count and a new Waterfall CDF from the step's sample; the others are untouched. Without a checkpoint RB = FT and `rebuilt_buckets = 0` | `rb_*`, `rb_time_s`, `rebuilt_buckets`; `median_qerror` = RB |

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

`--models`, `--buckets`, `--out-dir` and `--dry-run` work as in [§4.1](#41-full-scenarios);
with `--buckets fd` the bucket count is the Freedman–Diaconis `k` of step 0 and stays fixed.
Results go to `<out-dir>/<scenario>/gradual_normal_to_zipf_<Model>/drift_steps.csv` (synthetic) or
`<out-dir>/<scenario>/gradual_<column>_to_refresh_<Model>/drift_steps.csv` (DSB).

**Long-running:** 11–21 steps × 1M queries per synthetic scenario.

### 5.3 One model

```bash
poetry run python run_hybrid_drift.py --experiment-name my_drift \
    --shift-dir "data/generated/60000000_hard/shift_normal_to_zipf_5%" \
    --workload 1000000 \
    --static-workload-path data/generated/60000000_hard/normal/workload_driven_1000000.csv
```

| Parameter | Meaning |
|---|---|
| `--shift-dir PATH` | folder with `step_*.pkl` (required) |
| `--workload COUNT` | query count, used to find `*_workload_driven_<COUNT>.csv` files |
| `--static-workload-path PATH` | the fixed workload evaluated at every step |
| `--shift-workload` | use `step_N_workload_driven_<COUNT>.csv` from `--shift-dir` at step N instead; drop `--static-workload-path` |
| `--init-dist`, `--target-dist` | default `normal`, `zipf`. They only label the output folder `gradual_<init>_to_<target>_<Model>` |
| `--experiment-name`, `--out-dir`, `--buckets` | as in the static runners |

Model-specific parameters: `--points` (Hybrid), `--topk` (TKHist), the STHist and
EntropyHist parameters of [§4.2](#42-one-model-one-dataset).

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
<out-dir>/                                       # e.g. results/exp1_100_buckets
├── static/<benchmark>/                          # run_experiments.py --out-dir <out-dir>/static
│   ├── summary.csv                              # all models × datasets, one row each
│   └── <dataset>/
│       ├── summary_<Model>.json                 # q-error, timings, n_buckets, n_values, size_bytes (+ Hybrid: cdf_choice)
│       ├── <workload-stem>_<Model>.csv          # per query: Query_ID, Model, Prediction, Truth, Q_Error (selectivities)
│       └── model_selection_Hybrid.txt           # Hybrid: range, count and chosen local CDF of every bucket
├── drift/<scenario>/                            # run_drift_experiments.py --out-dir <out-dir>/drift
│   ├── gradual_<init>_to_<target>_<Model>/drift_steps.csv   # one row per step (Hybrid: + model_reports/step_<k>.txt)
│   └── combined.csv, drift_chart.png            # written by export_results.py
└── static_*.csv, histogram_size.csv, hybrid_cdf_choice.csv, drift_*.csv, results.xlsx   # export_results.py (§8)
```

`drift_steps.csv` columns: `step`, `drift_pct` (rows changed since step 0), `changed_since_rebuild_pct`,
`changed_since_rebuild_rows`, `action` (`initial build` / `rebuild` / `feedback update` / `none`; Hybrid: `FT` / `FT + rebuild`), `action_time_s` (measured wall time),
`median_qerror`, `p95_qerror`, `avg_qerror` (after the action), `before_*` (before it), `infer_us_median`,
`n_buckets`, `n_values`, `size_bytes`; Hybrid adds the stage columns above, `ft_n_values` / `ft_size_bytes` and `cdf_choice`, STHist `restructures`.

- **Model size** = stored numbers × 8 bytes (`n_values`, `size_bytes`). EquiWidth: min, bucket width,
  a count per bucket. EquiDepth: histogram bounds, MCV values and frequencies, n_distinct. TKHist: grid,
  counts, (value, count) per top-k entry. EntropyHist: min, max, count, distinct per bucket. STHist:
  bucket bounds and frequencies. Hybrid: grid, counts, the parameters of every local CDF and its type code.
- **Zero-truth queries are dropped** before scoring, since q-error is undefined for them.
  The query count can therefore differ between datasets and is logged.
- Timing fields in `summary_<Model>.json`, the same for every method:

  | Field | Meaning |
  |---|---|
  | `build_time` | seconds to build the model before the first query |
  | `update_time_total` | seconds spent learning from query feedback (STHist; 0 otherwise) |
  | `train_time` | `build_time + update_time_total` |
  | `infer_time` | sum over queries of the time of one `predict()` call; feedback is not included |
  | `avg_infer_time`, `median_infer_time`, `p95_infer_time` | statistics of one `predict()` call |

- A runner that gets a parent `--dataset` folder catches each dataset's error,
  prints it and continues. **Check the log**: a run can finish with some outputs missing.

---

## 8. Merging, plotting, exporting

```bash
poetry run python export_results.py --results-dir results/exp1_100_buckets
```

writes into that folder:

| File | Content |
|---|---|
| `static_results.csv` | one row per benchmark × dataset × model, every metric |
| `static_median_qerror.csv`, `static_p95_qerror.csv` | rows = benchmark / dataset, columns = models |
| `static_ranking.csv` | per benchmark × model: mean place by median q-error (1 = best), wins |
| `histogram_size.csv` | bytes of every model, and its bucket count |
| `hybrid_cdf_choice.csv` | how many buckets got each local CDF in Hybrid (static, and drift at step 0 and every rebuild) |
| `drift_results.csv` | one row per scenario × model × step |
| `drift_summary.csv` | per scenario × model: mean / worst / last median q-error, mean Shock (before) error, rebuilds, their total and mean real time; Hybrid also mean FT / RB error and rebuilt buckets |
| `results.xlsx` | all of the above, one sheet each |

and `drift/<scenario>/combined.csv` + `drift_chart.png` (median and P95 per step, squares mark rebuilds).
`merge_drift.py` and `plot_drift.py` do the last two for single scenario folders.

Static runs: box plots of the per-query q-error of several models:

```bash
poetry run python plot/plot_comparison.py \
    --experiments results/exp1_100_buckets/static/synth_hard_narrow/normal/*_*.csv \
    --output comparison.png --title "Q-Error: normal"
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
- **Same bucket count.** Every model gets 100 buckets (EquiDepth: `default_statistics_target`
  = 100, as in PostgreSQL), or with `--buckets fd` the Freedman–Diaconis `k` of the data.
  Buckets alone are not memory: EquiDepth also stores MCVs, TKHist top-k values, and Hybrid
  a local model per bucket, so every summary reports `size_bytes`.
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
3. `run_<name>_drift.py`: copy [run_tkhist_drift.py](run_tkhist_drift.py): a build function for `drift_harness.SampleHistogram`, or a `DriftModel` subclass for feedback methods.
4. One line in `MODELS` in [registry.py](registry.py). Both orchestrators pick it up.

---

## 10. Reproducing the reported results

Two experiments, same scenarios, only the bucket count differs. Model training and
query generation are seeded, so q-error is reproduced exactly; timings depend on the machine.

```bash
./setup_env.sh                                   # environment, synthetic workloads, STATS-CEB

for B in 100 fd; do
    OUT=results/$([ $B = fd ] && echo exp2_freedman_diaconis || echo exp1_100_buckets)
    poetry run python run_experiments.py       --scenario all --buckets $B --out-dir $OUT/static
    poetry run python run_drift_experiments.py --scenario all --buckets $B --out-dir $OUT/drift
    poetry run python export_results.py --results-dir $OUT
done
```

Launch long runs under `nohup` or `tmux`: a dropped SSH session kills a foreground
run. Delete a partial experiment folder before re-running it. Running scenarios in
parallel inflates the absolute timings; q-error does not depend on them.

---

## 11. Repository layout and documentation

```text
models/                 estimators (one file per method) + common.py (Bucket, RangeQuery, q-error)
registry.py             models + static/drift scenarios: the single list the orchestrators read
benchmark_utils.py      shared CLI, dataset loop, truth, evaluation loop, result writing
run_<model>.py          static runner per model
run_<model>_drift.py    drift runner per model (synthetic and DSB): what the model does, on drift_harness.py
plot_drift_figures.py   per drift scenario: median / P95 / avg q-error, inference time, cumulative adaptation time, memory (one PNG each; Hybrid = Hybrid_RB)
drift_harness.py        the one drift loop: steps, adapt/eval split, 10 % rule (baselines), Hybrid Shock → FT → RB, drift_steps.csv
run_experiments.py      static orchestrator (§4)
run_drift_experiments.py drift orchestrator (§5)
merge_drift.py, plot_drift.py  drift result tables and charts (§8)
export_results.py       readable tables + xlsx for one experiment folder (§8)
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
| [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) | what the benchmark does, why each design decision was made, list of fixes |
| [docs/METHODS_EXPLAINED.md](docs/METHODS_EXPLAINED.md) | how each method works and how the papers' open points were resolved |
