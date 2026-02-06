import pandas as pd
import numpy as np
from pathlib import Path
import os

def calculate_ndv():
    data_dir = Path("datasets/files/generated")
    if not data_dir.exists():
        print(f"Directory {data_dir} does not exist.")
        return

    dists = ["uniform", "normal", "zipf", "sparse_cluster", "anti_zipf"]
    
    print("| Distribution | Distinct Values (NDV) |")
    print("| :--- | :--- |")
    
    for dist in dists:
        csv_path = data_dir / f"bench_{dist}.csv"
        if not csv_path.exists():
            print(f"| {dist} | File Not Found |")
            continue
            
        # Use chunking to be safe, but bincount needs global view or careful merge.
        # Given domain is small (0-200,000), we can just use a boolean array.
        
        seen = np.zeros(200_001, dtype=bool)
        
        try:
            for chunk in pd.read_csv(csv_path, header=None, names=["v"], dtype="int64", chunksize=1_000_000, engine="c"):
                vals = chunk["v"].to_numpy()
                # Ensure values are within range just in case
                vals = vals[(vals >= 0) & (vals <= 200_000)]
                seen[vals] = True
                
            ndv = np.count_nonzero(seen)
            print(f"| **{dist.capitalize()}** | {ndv:,} |")
        except Exception as e:
            print(f"| {dist} | Error: {e} |")

if __name__ == "__main__":
    calculate_ndv()
