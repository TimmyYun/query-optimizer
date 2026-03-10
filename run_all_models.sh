#!/bin/bash

# Multi-Model Execution Script (Smart Workload)
# ===============================================

if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_name> --dataset <path> --count <query_count> [extra_args...]"
    exit 1
fi

EXP_NAME=$1
shift

DATASET=""
COUNT=""
BUCKETS=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --count)
            COUNT="$2" # Например, 100000
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

if [ -z "$DATASET" ] || [ -z "$COUNT" ]; then
    echo "Error: --dataset and --count arguments are required."
    exit 1
fi

run_model() {
    local script=$1
    local name=$2
    echo -e "\n>>> Running $name Baseline <<<"

    # Мы передаем COUNT в параметр --workload, а Python-скрипт сам найдет файл
    local cmd="poetry run python \"$script\" \
        --experiment-name \"$EXP_NAME\" \
        --dataset \"$DATASET\" \
        --workload \"$COUNT\""

    if [ ! -z "$BUCKETS" ]; then
        cmd="$cmd --buckets $BUCKETS"
    fi

    cmd="$cmd $EXTRA_ARGS"
    eval $cmd
}

run_model "run_equiwidth.py" "Equi-Width"
run_model "run_equihist.py" "Equi-Hist"
run_model "run_hybrid.py" "Hybrid"