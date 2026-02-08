#!/bin/bash
set -e

# Archive previous summary
mv benchmark_summary.csv benchmark_summary_old.csv 2>/dev/null || true

# Run Uniform
echo "--- Running Uniform ---"
poetry run python main.py --mode static --rows 1000000 --dist uniform --eval-n 100000 --bins 500 --recreate

# Run Normal
echo "--- Running Normal ---"
poetry run python main.py --mode static --rows 1000000 --dist normal --eval-n 100000 --bins 500 --recreate

# Run Zipf (Skewed)
echo "--- Running Zipf Skewed ---"
poetry run python main.py --mode static --rows 1000000 --dist zipf --eval-n 100000 --bins 500 --skewed --recreate

echo "--- Benchmark Complete ---"
cat benchmark_summary.csv
