#!/bin/bash

# Multi-Model Execution Script
# ===============================================
# Runs EquiWidth, EquiHist, and Hybrid models sequentially.

# Usage: ./run_all_models.sh <experiment_name> [extra_args...]
# Example: ./run_all_models.sh baseline_study

if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_name> [extra_args...]"
    exit 1
fi

EXP_NAME=$1
shift

DATASET=""
WORKLOAD=""
BUCKETS=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --workload)
            WORKLOAD="$2"
            shift 2
            ;;
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

if [ -z "$DATASET" ] || [ -z "$WORKLOAD" ]; then
    echo "Error: --dataset and --workload arguments are required."
    echo "Example: $0 $EXP_NAME --dataset data/generated/60000000/uniform --workload workload/100000.csv"
    exit 1
fi

echo "===================================================="
echo "Experiment: $EXP_NAME"
echo "Dataset: $DATASET"
echo "Workload: $WORKLOAD"
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
        --dataset \"$DATASET\" \
        --workload \"$WORKLOAD\""
    
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
