# DSB: official toolkit, refresh-stream drift

How the DSB inputs of this benchmark are produced. The user-facing commands and
the parameters to report are in the main [README §6.1](../README.md#61-dsb).

```
setup_dsb.sh                  clone microsoft/dsb @ ec9a156, build dsdgen/dsqgen,
     |                        SF=1 base data (data1gb/) + 300 refresh streams (data1gbupdates/)
     v
DSB_GENERATOR.ipynb           replay the official refresh streams, export each target
     |                        column at 0, 5, ..., 100 % cumulative churn  -> data/dsb/drift/
     v
prepare_benchmark.py          dsqgen on the official query_templates_pg -> 1-D ranges,
  --bench dsb                 checkpoints -> meta.pkl + drift/step_N.pkl  -> data/dsb/<column>/
  --generate-queries
```

`./setup_env.sh --dsb` runs the last two steps.

## Files

| File | Role |
|---|---|
| `setup_dsb.sh` | Linux and macOS (replaces the original `setup.sh`, which generated custom, non-DSB query templates). Clones the toolkit at a fixed commit, applies the portability patches (compiler flags; on macOS also `<values.h>` / `<malloc.h>`, which do not change generated data), builds, generates data and streams. Re-running skips finished steps |
| `DSB_GENERATOR.ipynb` | refresh replay + checkpoint export, run headlessly by `setup_env.sh --dsb` |

## Target columns

| Column | Type | Official templates with `col BETWEEN a AND b` |
|---|---|---|
| `web_sales.ws_wholesale_cost` | DECIMAL(7,2) | 8 (`range(random(0, 100, uniform), 5…30)`) |
| `web_sales.ws_list_price` | DECIMAL(7,2) | 9 (`range(random(1, 300, uniform), 5…20)`) |

The official templates contain no range predicate on surrogate keys
(`*_item_sk`, `*_time_sk`); there, those columns are only join keys. The values
are converted to integer cents (×100) by `prepare_benchmark.py`. The same scale
is applied to the query bounds, so the conversion is lossless.

Set `DSB_TARGET_COLUMNS="table.col,..."` to track other columns. The notebook
replays deletes for `catalog_returns` and `web_sales` and inserts for both.
Other tables would need their own insert logic.

## Notebook pipeline

```
load base tables (DSB_TABLES only) -> extract target columns -> checkpoint "initial"
for stream_id in 1 .. DSB_NUM_STREAMS:
    1. DELETE rows whose sold/returned date falls in delete_<id>.dat ranges
    2. INSERT catalog_returns from s_catalog_returns_<id>.dat
    3. INSERT web_sales from s_web_order_<id>.dat + s_web_order_lineitem_<id>.dat,
       with the TPC-DS view logic: ws_wholesale_cost = i_wholesale_cost,
       ws_list_price = i_current_price of the current item version
    4. churn += deleted + inserted rows of the column's table
    5. save a checkpoint at every 5 % of the baseline row count
    6. stop when every target column has reached 100 %
KS test + Wasserstein distance between consecutive checkpoints, plots, CSV export
```

Churn is `Σ(deleted + inserted) / baseline_rows × 100`. Both target columns
belong to `web_sales`, so they share one churn sequence. Deletes remove whole
date ranges (correlated removal), and inserts bring the new price distribution
of the current item versions. The drift is therefore systematic, not random noise.

Configuration, all via environment variables set by `setup_env.sh`:

| Variable | Meaning |
|---|---|
| `DSB_DATA_DIR`, `DSB_UPDATES_DIR` | base data and refresh streams |
| `DSB_EXPORT_DIR` | where the checkpoint CSVs go (`data/dsb/drift/`) |
| `DSB_TABLES` | tables to load; the refresh logic needs the 13 listed in `setup_env.sh` |
| `DSB_NUM_STREAMS` | upper bound on streams to replay |
| `DSB_TARGET_COLUMNS` | `table.col,...` to track |

Output: `data/dsb/drift/{table}__{column}__{initial,churn_5pct,...,churn_100pct}.csv`,
one column of values each.

## Reference

Bailu Ding, Surajit Chaudhuri, Johannes Gehrke, Vivek Narasayya.
*DSB: A Decision Support Benchmark for Workload-Driven and Traditional Database Systems.* PVLDB 14(13), 2021.
