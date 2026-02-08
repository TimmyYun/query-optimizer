#!/bin/bash
set -e

# Archive previous summary
mv benchmark_summary.csv benchmark_summary_old_1M_1k.csv 2>/dev/null || true

# Run Uniform
echo "--- Running Uniform ---"
poetry run python main.py --mode static --rows 1000000 --dist uniform --eval-n 1000 --bins 500 --recreate

# Run Normal
echo "--- Running Normal ---"
poetry run python main.py --mode static --rows 1000000 --dist normal --eval-n 1000 --bins 500 --recreate

# Run Zipf
echo "--- Running Zipf ---"
poetry run python main.py --mode static --rows 1000000 --dist zipf --eval-n 1000 --bins 500 --recreate

echo "--- Benchmark Complete ---"
cat benchmark_summary.csv
