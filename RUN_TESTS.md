# Running the benchmarks

Commands to set up the environment and run every benchmark in this repo.
There is no unit-test suite — "tests" here means the benchmark runs.

Every command is run from the repository root.

---

## 1. One-time setup

```bash
./setup_env.sh
```

Creates the Poetry environment on Python 3.12 and generates all synthetic
workloads. Takes a few minutes.

Why the 60M-row datasets are not regenerated: no `run_*.py` reads `data.csv`.
Every runner loads `meta.pkl`, which already contains the ground-truth `freq`
array and the 100k `sample`, and those files are committed. Only the workload
CSVs are missing from a fresh clone, and `.gitignore` excludes `*.csv`.

Individual stages:

```bash
./setup_env.sh --env          # Poetry environment only
./setup_env.sh --workloads    # synthetic workloads only
./setup_env.sh --dsb-drift    # DSB drift checkpoints (see section 6)
./setup_env.sh --all          # everything
```

### Verify the setup

```bash
poetry run python -c "import numpy, pandas, sklearn, scipy; print('ok')"
poetry run ruff check .
```

Smoke test — one model, one distribution, under a minute:

```bash
poetry run python run_equiwidth.py --experiment-name smoke \
    --dataset data/generated/60000000_hard/normal \
    --workload workload/1000000/narrow/workload.csv
```

Expect `results/smoke/normal/summary_EquiWidth.json` and `results/smoke/summary.csv`.

---

## 2. Static synthetic benchmarks

Runs Equi-Width, Equi-Hist and Hybrid across all 5 distributions.

```bash
poetry run python run_experiments.py --scenario simple_wide
poetry run python run_experiments.py --scenario simple_narrow
poetry run python run_experiments.py --scenario hard_wide
poetry run python run_experiments.py --scenario hard_narrow
poetry run python run_experiments.py --scenario hard_narrow_data_driven
poetry run python run_experiments.py --scenario all          # all of the above + dsb
```

> `README.md` calls the fifth scenario `hard_narrow_datadriven`, but the code
> accepts `hard_narrow_data_driven` (`run_experiments.py:37`). Use the latter.

Results land in `results/synth_<scenario>/<distribution>/`, with a rolled-up
`results/synth_<scenario>/summary.csv`.

### Individual models

```bash
poetry run python run_equiwidth.py --experiment-name my_exp \
    --dataset data/generated/60000000_hard/normal \
    --workload workload/1000000/narrow/workload.csv

poetry run python run_equihist.py --experiment-name my_exp --lr 0.5 \
    --dataset data/generated/60000000_hard/normal \
    --workload workload/1000000/narrow/workload.csv

poetry run python run_hybrid.py --experiment-name my_exp --points 200 \
    --dataset data/generated/60000000_hard/normal \
    --workload workload/1000000/narrow/workload.csv
```

Pointing `--dataset` at `data/generated/60000000_hard` (no distribution)
makes the runner loop over every distribution underneath it.

---

## 3. Synthetic drift benchmarks

**Long-running:** 21 drift steps x 1M queries x 3 models per scenario.

```bash
poetry run python run_drift_experiments.py --scenario drift_10_ds_zipf_wl_normal
poetry run python run_drift_experiments.py --scenario drift_5_ds_zipf_wl_normal
poetry run python run_drift_experiments.py --scenario drift_5_ds_zipf_wl_zipf
poetry run python run_drift_experiments.py --scenario drift_5_ds_add_zipf_wl_zipf
poetry run python run_drift_experiments.py --scenario all
```

| Scenario | Dataset drift | Workload |
|---|---|---|
| `drift_10_ds_zipf_wl_normal` | normal→zipf, 10% steps | static (normal) |
| `drift_5_ds_zipf_wl_normal` | normal→zipf, 5% steps | static (normal) |
| `drift_5_ds_zipf_wl_zipf` | normal→zipf, 5% steps | drifts with the data |
| `drift_5_ds_add_zipf_wl_zipf` | zipf mixed additively into normal | drifts with the data |

### Individual drift models

`--shift-dir` and `--static-workload-path` are effectively required; the
orchestrator always passes them.

```bash
poetry run python run_hybrid_drift.py --experiment-name my_drift \
    --shift-dir "data/generated/60000000_hard/shift_normal_to_zipf_5%" \
    --workload 1000000 \
    --static-workload-path data/generated/60000000_hard/normal/workload_driven_1000000.csv
```

Swap in `run_equiwidth_drift.py` / `run_equihist_drift.py` for the baselines.
Add `--shift-workload` (and drop `--static-workload-path`) to make the workload
drift alongside the data.

Each drift step is evaluated in three stages, which is why the output CSV has
`Shock_*`, `FT_*` and `RB_*` column families: **Shock** (stale model), **FT**
(bucket-mass finetune), **RB** (retrain of buckets whose q-error exceeded 1.3).

---

## 4. DSB static benchmarks

Needs `data/dsb/<column>/workload_driven_*.csv`, built by `prepare_dsb.py`
from the committed `data/dsb/custom_*.sql` (1M queries each).

```bash
poetry run python prepare_dsb.py
poetry run python run_experiments.py --scenario dsb
```

> `prepare_dsb.py` also **overwrites the committed `data/dsb/<column>/meta.pkl`**
> and needs `data/dsb/<column>.csv` to do so — those column CSVs come from the
> DSB generation in section 5. Run `git diff --stat data/dsb` afterwards to see
> whether the regenerated metadata differs from the committed one.

---

## 5. DSB data generation (macOS)

`DSB benchmark/setup.sh` is the original Linux script and is left untouched.
Use the macOS port, which adds the source patches the TPC-DS toolkit needs to
compile under Apple clang:

```bash
"DSB benchmark/setup_macos.sh"
```

**Long-running:** clones microsoft/dsb, builds `dsdgen`/`dsqgen`, then generates
~1.3 GB of base data, 300 refresh streams (~5.1 GB) and 400k queries per
template. Re-running is safe — completed steps are skipped, so raising
`UPDATE_STREAMS` tops up rather than regenerates.

300 is the **verified minimum** for a complete checkpoint set (63 files).
Measured: `web_sales.ws_item_sk` reaches 100% churn by ~200 streams, but both
`catalog_returns` columns are still at 80% there and need ~300. Roughly
17 MB per stream.

What the macOS port changes, all of them real portability bugs:

| Problem | Fix |
|---|---|
| `<values.h>` does not exist on macOS | `config.h` uses `<limits.h>`; `MAXINT` defined as `INT_MAX`, the same value glibc uses, so generated data is unchanged |
| `<malloc.h>` does not exist on macOS | replaced with `<stdlib.h>` |
| clang 15+ makes implicit declarations / int conversions errors | `-Wno-implicit-function-declaration -Wno-int-conversion -Wno-implicit-int -Wno-return-type` |
| `mkdir` fails, `cd` depends on caller's cwd, no `set -e` | `mkdir -p`, paths relative to the script, `set -e` |

---

## 6. DSB drift benchmarks

First export the 63 checkpoint CSVs (21 churn levels x 3 columns) by running
the notebook headlessly:

```bash
./setup_env.sh --dsb-drift
```

This executes `DSB benchmark/DSB_GENERATOR.ipynb` with its three path settings
supplied via `DSB_DATA_DIR`, `DSB_UPDATES_DIR` and `DSB_EXPORT_DIR`, writing to
`data/dsb/drift/`. It also sets `DSB_TABLES` so only the 13 tables the pipeline
actually touches get loaded (skipping ~1 GB of `store_sales` / `catalog_sales` /
`inventory`) and `DSB_NUM_STREAMS` to match what was generated, so the replay
loop does not spin through missing stream files. **Long-running.**

```bash
DSB_NUM_STREAMS=500 ./setup_env.sh --dsb-drift   # if you generated more streams
```

Then, per column:

```bash
for col in ws_item_sk cr_item_sk cr_returned_time_sk; do
    poetry run python run_dsb_equiwidth_drift.py --column $col --experiment-name dsb_drift
    poetry run python run_dsb_equihist_drift.py  --column $col --experiment-name dsb_drift
    poetry run python run_dsb_hybrid_drift.py    --column $col --experiment-name dsb_drift
done
```

> Keep `--experiment-name dsb_drift`. `merge_dsb_drifts.py:95` and
> `plot_dsb_drift.py:8` both hardcode `results/dsb_drift`.

---

## 7. Merging and plotting

```bash
poetry run python merge_dsb_drifts.py    # -> results/dsb_drift/combined_<column>.csv
poetry run python plot_dsb_drift.py      # line charts for Median / P95 / Avg q-error
poetry run python merge_all_drifts.py    # synthetic drift, all 4 scenarios
poetry run python merge_drift_results.py # synthetic drift, single merged CSV
```

> The `merge_*.py` scripts and `plot_dsb_drift.py` contain **hardcoded result
> paths** (e.g. `results/equihist_drift_5%/...`). They need editing whenever
> experiment names change.

Comparison box plots for a static run:

```bash
poetry run python plot/plot_comparison.py \
    --experiments results/synth_hard_narrow/normal/workload_*.csv \
    --output results/synth_hard_narrow/normal/comparison.png \
    --title "Q-Error Comparison: normal"
```

> `plot_all.sh` expects the older layout `results/<exp>/<rows>/<dist>/<count>_<Model>.csv`.
> Current runs write `results/synth_<scenario>/<dist>/<workload-stem>_<Model>.csv`
> — the workload stem, not the query count. Call `plot/plot_comparison.py`
> directly, as above, or update the script.

---

## Reference: what setup generates

| Path | Contents |
|---|---|
| `workload/1000000/{wide,narrow}/workload.csv` | random workloads for the `*_wide` / `*_narrow` scenarios |
| `data/generated/60000000_{hard,simple}/<dist>/workload_driven_1000000.csv` | data-driven workload per distribution |
| `data/generated/60000000_hard/shift_*/step_N_workload_driven_1000000.csv` | per-drift-step workloads (21 for the 5% shifts, 11 for the 10% shift) |
| `data/dsb/<column>/workload_driven_1000000.csv` | DSB static workloads, from the committed `.sql` |
| `data/dsb/drift/{table}__{column}__{initial,churn_5pct..churn_100pct}.csv` | 63 DSB drift checkpoints |
| `DSB benchmark/dsb/code/tools/data1gb/` | TPC-DS base data (~1.3 GB, 23 tables) |
| `DSB benchmark/dsb/code/tools/data1gbupdates/` | refresh streams |

All of the above is gitignored (`*.csv`), and all of it is reproducible —
every generator is seeded with 42.
