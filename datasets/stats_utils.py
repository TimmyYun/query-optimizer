import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Tuple

def freedman_diaconis_bins(sample: np.ndarray, mn: int, mx: int, n_rows: int, bins_max: int = 2000) -> int:
    if sample.size < 10: return 10
    q25, q75 = np.quantile(sample, [0.25, 0.75])
    iqr = q75 - q25
    if iqr <= 0: return 10
    bin_width = 2 * iqr / (n_rows ** (1/3))
    total_width = mx - mn
    if bin_width <= 0: return 10
    return max(1, min(int(total_width / bin_width), bins_max))

def calculate_skew_kurt(csv_path: Path) -> Tuple[float, float]:
    try:
        # For performance with large files, we should probably chunk this or use sampling, 
        # but maintaining compatibility for now.
        df = pd.read_csv(csv_path, header=None, names=["v"], dtype="int64")
        return float(df["v"].skew()), float(df["v"].kurt())
    except Exception as e:
        print(f"Error calculating skew/kurt: {e}")
        return 0.0, 0.0

def calculate_ndv(csv_path: Path):
    seen = np.zeros(200_001, dtype=bool)
    try:
        for chunk in pd.read_csv(csv_path, header=None, names=["v"], dtype="int64", chunksize=1_000_000, engine="c"):
            vals = chunk["v"].to_numpy()
            vals = vals[(vals >= 0) & (vals <= 200_000)]
            seen[vals] = True
        return np.count_nonzero(seen)
    except Exception as e:
        print(f"Error calculating NDV for {csv_path}: {e}")
        return 0

def save_stats(stats: dict, output_path: Path):
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=4)
