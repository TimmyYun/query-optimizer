import argparse
from datasets import DatasetManager
import workload
import time

def main():
    rows_list = [1_000_000, 10_000_000, 60_000_000]
    distributions = ["uniform", "normal", "zipf", "sparse_cluster", "anti_zipf"]
    
    dm = DatasetManager()
    
    total_start = time.time()
    
    for rows in rows_list:
        for dist in distributions:
            print(f"\n>>> Generating Dataset: {rows} rows, {dist} <<<")
            start = time.time()
            try:
                # 1. Generate Data
                ds_dir = dm.prepare_dataset(rows, dist)
                
                # 2. Generate Workload (Independent)
                workload.prepare_workload(ds_dir, n_queries=1000)
                
            except Exception as e:
                print(f"Failed to generate {rows} / {dist}: {e}")
                import traceback
                traceback.print_exc()
            
            elapsed = time.time() - start
            print(f"Done in {elapsed:.2f}s")

    total_elapsed = time.time() - total_start
    print(f"\nAll operations completed in {total_elapsed:.2f}s")

if __name__ == "__main__":
    main()
