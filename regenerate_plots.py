from main import plot_q_error_boxplots
from pathlib import Path
import os

target_dir = Path("results/hybrid_vectorized_60m")

if not target_dir.exists():
    print(f"Error: {target_dir} not found")
    exit(1)

# Recursive walk to find CSVs
for root, dirs, files in os.walk(target_dir):
    for f in files:
        if f.endswith(".csv") and "drift" not in f and "summary" not in f:
             csv_path = Path(root) / f
             print(f"Plotting for {csv_path}")
             plot_q_error_boxplots(csv_path, Path(root))
