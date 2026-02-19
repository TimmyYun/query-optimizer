#!/bin/bash

# Usage: ./plot_all.sh <experiment_id> [rows] [workload_name]
# Example: ./plot_all.sh 1 60000000 1000000

EXP_ID=$1
ROWS=${2:-60000000}
WL=${3:-1000000}

if [ -z "$EXP_ID" ]; then
    echo "Usage: $0 <experiment_id> [rows] [workload_name]"
    exit 1
fi

DISTS=("uniform" "normal" "zipf" "sparse_cluster" "anti_zipf")

for DIST in "${DISTS[@]}"; do
    BASE_PATH="results/$EXP_ID/$ROWS/$DIST"
    
    # Check if files exist before plotting
    if [ ! -f "$BASE_PATH/${WL}_EquiWidth.csv" ] || \
       [ ! -f "$BASE_PATH/${WL}_EquiHist.csv" ] || \
       [ ! -f "$BASE_PATH/${WL}_Hybrid.csv" ]; then
        echo "Skipping $DIST: One or more result files missing in $BASE_PATH"
        continue
    fi

    echo ">>> Generating comparison plot for $DIST in $BASE_PATH..."
    
    poetry run python ./plot/plot_comparison.py --experiments \
        "$BASE_PATH/${WL}_EquiWidth.csv" \
        "$BASE_PATH/${WL}_EquiHist.csv" \
        "$BASE_PATH/${WL}_Hybrid.csv" \
        --output "$BASE_PATH/comparison_plot.png" \
        --title "Q-Error Comparison: $DIST (N=$ROWS)"
done

echo "Done! Individual plots saved as comparison_plot.png in their respective folders."
