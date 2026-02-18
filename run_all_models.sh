#!/bin/bash

# Multi-Model Execution Script (Strict 60M/1M/All)
# ===============================================
# Runs EquiWidth, EquiHist, and Hybrid models sequentially.
# Always uses: 60M rows, 1M workload, all distributions.

# Usage: ./run_all_models.sh <experiment_name> [extra_args...]
# Example: ./run_all_models.sh baseline_study

if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_name> [extra_args...]"
    exit 1
fi

EXP_NAME=$1
shift

BUCKETS=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --buckets)
            BUCKETS="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Hardcoded Multi-Model Defaults
ROWS=60000000
DIST="all"
EVAL_N=1000000

echo "===================================================="
echo "Experiment: $EXP_NAME"
echo "Target: $ROWS rows, $EVAL_N workload, $DIST distributions"
if [ ! -z "$EXTRA_ARGS" ]; then
    echo "Extra Params: $EXTRA_ARGS"
fi
echo "===================================================="

run_model() {
    local script=$1
    local name=$2
    echo -e "\n>>> Running $name Baseline <<<"
    
    local cmd="poetry run python \"$script\" \
        --experiment-name \"$EXP_NAME\" \
        --rows \"$ROWS\" \
        --dist \"$DIST\" \
        --eval-n \"$EVAL_N\""
    
    if [ ! -z "$BUCKETS" ]; then
        cmd="$cmd --buckets $BUCKETS"
    fi
    
    cmd="$cmd $EXTRA_ARGS"
    
    eval $cmd
}

run_model "run_equiwidth.py" "Equi-Width"
run_model "run_equihist.py" "Equi-Hist"
run_model "run_hybrid.py" "Hybrid"

echo -e "\n===================================================="
echo "Experiment $EXP_NAME Completed."
echo "Results are available in results/$EXP_NAME"
echo "===================================================="
