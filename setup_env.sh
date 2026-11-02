#!/usr/bin/env bash
# ============================================================
# Environment & Dataset Setup
# ============================================================
# Prepares everything the benchmark scripts need on a fresh clone.
#
# The 60M-row data.csv files are NOT regenerated: no run_*.py ever
# reads them. Every runner loads meta.pkl, which already carries the
# ground-truth freq array and the 100k sample, and those are committed.
# Only the workload CSVs are missing, and those are cheap to rebuild.
#
# Usage:
#   ./setup_env.sh                 # env + synthetic workloads (fast path)
#   ./setup_env.sh --env           # poetry environment only
#   ./setup_env.sh --workloads     # synthetic workloads only
#   ./setup_env.sh --stats-ceb     # download STATS-CEB (pinned commit, ~6 MB) and convert it
#   ./setup_env.sh --dsb           # DSB queries + checkpoints (needs "DSB benchmark/setup_dsb.sh" first)
#   ./setup_env.sh --all           # everything
# ============================================================
set -e

PYTHON_BIN=${PYTHON_BIN:-$(command -v python3.12 || command -v python3)}
WORKLOAD_COUNT=${WORKLOAD_COUNT:-1000000}
DATASET_DIRS=${DATASET_DIRS:-"60000000_hard 60000000_simple"}

# Refresh streams to replay. 300 is the verified minimum for all three target
# columns to reach 100% churn (200 leaves both catalog_returns columns at 80%).
DSB_NUM_STREAMS=${DSB_NUM_STREAMS:-300}

# DSB columns to track through the refresh streams (must match registry.DSB_COLUMNS).
DSB_TARGET_COLUMNS=${DSB_TARGET_COLUMNS:-"web_sales.ws_wholesale_cost,web_sales.ws_list_price"}

# Tables the drift pipeline actually touches (13 of 23).
# Skipping the rest avoids loading ~1GB of store_sales/catalog_sales/inventory.
DSB_TABLES="call_center,catalog_returns,customer,date_dim,item,promotion,reason,ship_mode,time_dim,warehouse,web_page,web_sales,web_site"

cd "$(dirname "$0")"

DO_ENV=0
DO_WORKLOADS=0
DO_STATS=0
DO_DSB_DRIFT=0

if [ $# -eq 0 ]; then
    DO_ENV=1
    DO_WORKLOADS=1
    DO_STATS=1
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)        DO_ENV=1; shift ;;
        --workloads)  DO_WORKLOADS=1; shift ;;
        --stats-ceb)  DO_STATS=1; shift ;;
        --dsb|--dsb-drift) DO_DSB_DRIFT=1; shift ;;
        --all)        DO_ENV=1; DO_WORKLOADS=1; DO_STATS=1; DO_DSB_DRIFT=1; shift ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--env] [--workloads] [--stats-ceb] [--dsb] [--all]"
            exit 1
            ;;
    esac
done

# ============================================================
# 1. Python environment
# ============================================================
if [ "$DO_ENV" -eq 1 ]; then
    echo "============================================="
    echo "1) Python environment"
    echo "============================================="

    # Pinned to 3.12: pyproject requires >=3.12, and the DSB notebook uses
    # PEP 701 nested-quote f-strings which only parse on 3.12+.
    if [ -z "$PYTHON_BIN" ] || ! "$PYTHON_BIN" -c 'import sys; sys.exit(sys.version_info < (3, 12))'; then
        echo "ERROR: Python 3.12+ not found (PYTHON_BIN='$PYTHON_BIN')."
        echo "       macOS: brew install python@3.12   Ubuntu: sudo apt install python3.12 python3.12-venv"
        echo "       (or set PYTHON_BIN to another 3.12+ interpreter)"
        exit 1
    fi

    poetry env use "$PYTHON_BIN"
    poetry install

    poetry run python -c "import numpy, pandas, sklearn, scipy, matplotlib, seaborn; print('core imports OK')"
fi

# ============================================================
# 2. Synthetic workloads
# ============================================================
if [ "$DO_WORKLOADS" -eq 1 ]; then
    echo "============================================="
    echo "2) Synthetic workloads (count=$WORKLOAD_COUNT)"
    echo "============================================="

    # Data-driven workloads. workload.py batch mode walks every subfolder:
    # plain distributions get workload_driven_N.csv, and shift folders
    # (detected by step_*.pkl) get one workload per drift step.
    for ds in $DATASET_DIRS; do
        if [ ! -d "data/generated/$ds" ]; then
            echo "Skipping $ds: data/generated/$ds not found"
            continue
        fi
        echo -e "\n>>> Data-driven workloads for $ds <<<"
        poetry run python workload/workload.py --count "$WORKLOAD_COUNT" --rows "$ds"
    done

    # Wide / narrow random workloads.
    # workload.py writes these to data/workloads/{mode}_{count}.csv, but
    # run_experiments.py looks for workload/{count}/{mode}/workload.csv,
    # so move them into place.
    echo -e "\n>>> Wide workload <<<"
    poetry run python workload/workload.py --count "$WORKLOAD_COUNT" --wide

    echo -e "\n>>> Narrow workload <<<"
    poetry run python workload/workload.py --count "$WORKLOAD_COUNT"

    mkdir -p "workload/$WORKLOAD_COUNT/wide" "workload/$WORKLOAD_COUNT/narrow"
    mv "data/workloads/wide_${WORKLOAD_COUNT}.csv"   "workload/$WORKLOAD_COUNT/wide/workload.csv"
    mv "data/workloads/narrow_${WORKLOAD_COUNT}.csv" "workload/$WORKLOAD_COUNT/narrow/workload.csv"
    echo "Moved wide/narrow workloads into workload/$WORKLOAD_COUNT/"
fi

# ============================================================
# 3. STATS-CEB (official data + official sub-queries, pinned commit)
# ============================================================
if [ "$DO_STATS" -eq 1 ]; then
    echo "============================================="
    echo "3) STATS-CEB"
    echo "============================================="
    poetry run python prepare_benchmark.py --bench stats-ceb
fi

# ============================================================
# 4. DSB: official queries + drift checkpoints
# ============================================================
if [ "$DO_DSB_DRIFT" -eq 1 ]; then
    echo "============================================="
    echo "4) DSB queries and drift checkpoints"
    echo "============================================="

    DSB_TOOLS="DSB benchmark/dsb/code/tools"
    if [ ! -d "$DSB_TOOLS/data1gbupdates" ]; then
        echo "ERROR: $DSB_TOOLS/data1gbupdates not found."
        echo "       Run 'DSB benchmark/setup_dsb.sh' first to build the"
        echo "       toolkit and generate the base data + update streams."
        exit 1
    fi

    mkdir -p data/dsb/drift

    # Runs the author's notebook unmodified apart from its path config,
    # which is read from these env vars.
    echo ">>> Executing DSB_GENERATOR.ipynb (this takes a while)..."
    ( cd "DSB benchmark" && \
      MPLBACKEND=Agg \
      DSB_DATA_DIR="./dsb/code/tools/data1gb" \
      DSB_UPDATES_DIR="./dsb/code/tools/data1gbupdates" \
      DSB_EXPORT_DIR="../data/dsb/drift" \
      DSB_TABLES="$DSB_TABLES" \
      DSB_TARGET_COLUMNS="$DSB_TARGET_COLUMNS" \
      DSB_NUM_STREAMS="$DSB_NUM_STREAMS" \
      poetry run jupyter nbconvert \
          --to notebook --execute DSB_GENERATOR.ipynb \
          --output executed_DSB_GENERATOR.ipynb \
          --ExecutePreprocessor.timeout=-1 )

    echo ">>> Checkpoints exported: $(ls data/dsb/drift/*.csv 2>/dev/null | wc -l | tr -d ' ') files"

    # Official templates -> dsqgen -> 1-D ranges; checkpoints -> meta.pkl + drift/step_N.pkl
    echo ">>> Converting DSB (queries from the official templates, data from the checkpoints)..."
    poetry run python prepare_benchmark.py --bench dsb --generate-queries
fi

echo "============================================="
echo "Setup complete. See README.md for the commands."
echo "============================================="
