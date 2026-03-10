#!/bin/bash
set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: ./run_all_shift_scenarios.sh <dataset_dir> <workload_csv> [experiment_name] [init_dist] [target_dist]"
    exit 1
fi

DATASET=$1
WORKLOAD=$2
EXP_NAME=${3:-shift_experiment}
INIT_DIST=${4:-anti_zipf}
TARGET_DIST=${5:-uniform}

echo "============================================="
echo "Running Shift Scenarios: $INIT_DIST -> $TARGET_DIST"
echo "Dataset: $DATASET"
echo "Workload: $WORKLOAD"
echo "Experiment Name: $EXP_NAME"
echo "============================================="

# 1. Hybrid Model (Shock -> Finetune -> Rebuild)
echo "---------------------------------------------"
echo "1) Hybrid Model"
echo "---------------------------------------------"
poetry run python run_hybrid_shift_scenario.py \
    --dataset "$DATASET" \
    --workload "$WORKLOAD" \
    --init-dist "$INIT_DIST" \
    --target-dist "$TARGET_DIST" \
    --experiment-name "$EXP_NAME"

# 2. Equi-Width Model (Rebuild at every step)
echo "---------------------------------------------"
echo "2) Equi-Width Model"
echo "---------------------------------------------"
poetry run python run_equiwidth_shift_scenario.py \
    --dataset "$DATASET" \
    --workload "$WORKLOAD" \
    --init-dist "$INIT_DIST" \
    --target-dist "$TARGET_DIST" \
    --experiment-name "$EXP_NAME"

# 3. Equi-Hist Model (Online continual learning)
echo "---------------------------------------------"
echo "3) Equi-Hist Model"
echo "---------------------------------------------"
poetry run python run_equihist_shift_scenario.py \
    --dataset "$DATASET" \
    --workload "$WORKLOAD" \
    --init-dist "$INIT_DIST" \
    --target-dist "$TARGET_DIST" \
    --experiment-name "$EXP_NAME"

echo "============================================="
echo "All shift scenarios completed. Results saved under results/$EXP_NAME/"
echo "============================================="
