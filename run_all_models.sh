#!/bin/bash

# Multi-Model Execution Script (Smart Workload & Auto-Discovery)
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

# Парсинг аргументов
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --count)
            COUNT="$2"
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

# Внутренняя функция для запуска набора моделей
execute_suite() {
    local target_path=$1
    local current_exp_name=$2

    run_model() {
        local script=$1
        local name=$2
        echo -e "\n>>> Running $name Baseline for $target_path <<<"

        # Мы передаем только EXP_NAME. Python сам добавит имя датасета в путь.
        local cmd="poetry run python \"$script\" \
            --experiment-name \"$current_exp_name\" \
            --dataset \"$target_path\" \
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
}

# --- ЛОГИКА АВТООБНАРУЖЕНИЯ ---
if [ -f "${DATASET}/meta.pkl" ]; then
    # Если это одиночный датасет (файл meta.pkl найден в корне)
    execute_suite "$DATASET" "$EXP_NAME"
else
    # Если это папка с набором распределений
    echo "Scanning $DATASET for distributions..."
    for d in "$DATASET"/*/ ; do
        if [ -f "${d}meta.pkl" ]; then
            # Передаем исходный EXP_NAME.
            # Python-скрипт сам создаст папку распределения внутри results/$EXP_NAME/
            execute_suite "$d" "$EXP_NAME"
        fi
    done
fi

echo -e "\n===================================================="
echo "All tasks for $EXP_NAME Completed."
echo "===================================================="