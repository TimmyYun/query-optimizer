```markdown
# DSB Data Drift & Selectivity Estimation Pipeline

## Overview

This project uses the **[DSB (Database Stress Benchmark)](https://github.com/microsoft/dsb)**
to study how **incremental data updates** affect column-level distributions in a TPC-DS-like
data warehouse. We generate base data, apply 1000 refresh streams, capture distribution
**checkpoints** at every 5% cumulative churn, and compare them statistically.


setup.sh 
 |
 V
.dat base data
 |
 V
Python notebook (analysis and updates)


---

## Target Columns

| Table | Column | Description |
|---|---|---|
| `catalog_returns` | `cr_item_sk` | Surrogate key of returned item |
| `catalog_returns` | `cr_returned_time_sk` | Surrogate key of return time |
| `web_sales` | `ws_item_sk` | Surrogate key of sold item |

These columns were chosen because they are integer-valued, high-cardinality,
and directly affected by INSERT/DELETE refresh operations.

---

## Part 1: Setup & Data Generation (`setup.sh`)

### Prerequisites

- Linux / macOS / WSL
- `gcc`, `make`, `git`
- Python 3.x (for deduplication script)

### What it does

```bash
bash setup.sh
```

| Step | Command | Output |
|---|---|---|
| Clone DSB | `git clone ...` | `./dsb/` |
| Create query templates | `cat > .tpl` | 3 custom templates + `postgres.tpl` |
| Build tools | `make` | `dsdgen`, `dsqgen` binaries |
| Generate base data | `dsdgen -scale 1` | 23 `.dat` files in `data1gb/` |
| Generate 1000 streams | `dsdgen -update $i` | ~4000 files in `data1gbupdates/` |
| Generate 400K queries | `dsqgen -stream 400000` | 3 `.sql` files |
| Deduplicate queries | `python3 duplicate_deleter.py` | 3 deduped `.sql` files |

### Configuration

```bash
WORKLOAD_COUNT=400000   # Number of generated queries per template
UPDATE_STREAMS=1000     # Number of refresh streams
```

### Custom Query Templates

Each template generates a **range predicate** (`BETWEEN`) with randomized bounds:

```sql
-- custom_cr_item_sk.tpl
WHERE cr_item_sk BETWEEN [ITEM_START] AND [ITEM_START] + [RANGE_];
--   ITEM_START ∈ uniform(1, 17999)
--   RANGE_     ∈ uniform(700, 3000)
```

---

## Part 2: Analysis Notebook (`DSB_GENERATOR.ipynb`)

### Prerequisites

```
pip install pandas numpy matplotlib seaborn scipy
```

### Configuration (in notebook)

```python
# Path to base .dat files
data_dir = "/path/to/data1gb"

# Path to update streams (extracted folder, NOT archive)
UPDATES_DIR = "/path/to/data1gbupdates"

TARGET_COLUMNS = {
    'catalog_returns': ['cr_returned_time_sk', 'cr_item_sk'],
    'web_sales':       ['ws_item_sk'],
}

NUM_STREAMS = 1000
```

### Pipeline

```
Load 23 base tables (.dat → DataFrames)
           │
           ▼
  Extract targeted columns → targeted_loaded_dfs
           │
           ▼
  Initialize CheckpointManager (save "initial" snapshot)
           │
           ▼
  ┌────────────────────────────────────┐
  │  FOR stream_id = 1 .. 1000:       │
  │                                    │
  │    1. DELETE rows by date range    │
  │       (delete_{id}.dat)            │
  │                                    │
  │    2. INSERT catalog_returns       │
  │       (s_catalog_returns_{id}.dat) │
  │                                    │
  │    3. INSERT web_sales             │
  │       (s_web_order_{id}.dat +      │
  │        s_web_order_lineitem_{id})  │
  │                                    │
  │    4. Re-extract targeted columns  │
  │                                    │
  │    5. Accumulate churn             │
  │       churn = Σ(deleted+inserted)  │
  │                                    │
  │    6. If churn ≥ next threshold:   │
  │       save checkpoint snapshot     │
  │       (5%, 10%, 15%, ..., 100%)    │
  │                                    │
  │    7. If all columns hit 100% →    │
  │       BREAK                        │
  └────────────────────────────────────┘
           │
           ▼
  Statistical comparison (KS, Wasserstein)
           │
           ▼
  Visualization & CSV export
```

### Notebook Sections

| Section | Description |
|---|---|
| **Data Loading** | Reads all 23 `.dat` files, applies `schema_headers` |
| **Freedman-Diaconis Binning** | Optimal bin count for histograms |
| **Integer Column Detection** | Identifies numeric columns across all tables |
| **Bar Charts & Histograms** | Per-bucket kurtosis annotations |
| **Interactive Drill-down** | `plot_histogram_with_buckets()` → `plot_bucket_detail()` |
| **Staging File Loader** | Parses `s_catalog_returns`, `s_web_order`, `delete` files |
| **Dimension Key Lookups** | `business_key → surrogate_key` via dimension tables |
| **Refresh Operations** | `apply_deletes_targeted()`, `insert_catalog_returns_targeted()`, `insert_web_sales_targeted()` |
| **CheckpointManager** | Tracks cumulative churn, saves snapshots at 5% intervals |
| **Stream Processing Loop** | Applies all 1000 streams sequentially |
| **Statistical Comparison** | KS-test (p < 0.05 = distribution changed), Wasserstein distance |
| **Visualization** | Overlay histograms, row count bars, delta heatmaps |
| **CSV Export** | One file per (column × checkpoint) |

### Checkpoints

21 checkpoints per column:

```
initial → churn_5pct → churn_10pct → ... → churn_95pct → churn_100pct
```

**Churn** is defined as:

```
churn% = Σ(deleted_rows + inserted_rows) / baseline_row_count × 100
```

### Statistical Tests

| Metric | What it measures | Threshold |
|---|---|---|
| **KS statistic** | Max distance between two CDFs | — |
| **KS p-value** | Probability distributions are identical | p < 0.05 → changed |
| **Wasserstein distance** | "Earth mover's distance" between distributions | Lower = more similar |

### Output Files

Exported to `checkpoints/` directory:

```
{table}__{column}__{checkpoint_label}.csv
```

Example:
```
catalog_returns__cr_item_sk__initial.csv
catalog_returns__cr_item_sk__churn_5pct.csv
catalog_returns__cr_item_sk__churn_10pct.csv
...
web_sales__ws_item_sk__churn_100pct.csv
```

Each CSV contains a single column of integer surrogate keys representing
the full state of that column at that churn level.

---

## Key Findings (Expected)

- **KS test** detects statistically significant distribution shifts
  even at low churn levels (5–10%) due to large sample sizes
- **Wasserstein distance** grows roughly linearly with churn
- Delete operations remove rows by **date range** (correlated removal),
  while inserts add rows with **new surrogate keys** — causing
  systematic distribution shift rather than random noise
- The three target columns drift at **different rates** because
  `catalog_returns` and `web_sales` have different delete/insert ratios

---

## Quick Start

```bash
# 1. Generate everything
bash setup.sh

# 2. Open notebook, set paths:
#    data_dir   = "./dsb/code/tools/data1gb"
#    UPDATES_DIR = "./dsb/code/tools/data1gbupdates"

# 3. Run all cells
```

---

## References

- [DSB: Database Stress Benchmark (Microsoft)](https://github.com/microsoft/dsb)
- [TPC-DS Specification](https://www.tpc.org/tpcds/)
- Freedman–Diaconis rule: Freedman, D. & Diaconis, P. (1981)
- KS Test: `scipy.stats.ks_2samp`
- Wasserstein Distance: `scipy.stats.wasserstein_distance`
```