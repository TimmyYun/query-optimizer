#!/bin/bash
set -e

# Archive previous summary
mv benchmark_summary.csv benchmark_summary_old_1M_narrow.csv 2>/dev/null || true

# Run Uniform
echo "--- Running Uniform (Narrow) ---"
poetry run python main.py --mode static --rows 1000000 --dist uniform --eval-n 100000_narrow --bins 500

# Run Normal
echo "--- Running Normal (Narrow) ---"
poetry run python main.py --mode static --rows 1000000 --dist normal --eval-n 100000_narrow --bins 500

# Run Zipf
echo "--- Running Zipf (Narrow) ---"
poetry run python main.py --mode static --rows 1000000 --dist zipf --eval-n 100000_narrow --bins 500

echo "--- Benchmark Complete ---"
cat benchmark_summary.csv
