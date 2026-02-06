import argparse
from datasets import DatasetManager
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
                # Force regeneration to ensure fresh data if needed, 
                # or just let it cache if allowed. The user asked to "generate", 
                # implying creation. If they already exist, prepare_dataset handles check.
                # Just in case, I'll pass force_regeneration=False to save time if they exist,
                # unless users specifically asked to RE-generate. The prompt says "generate", 
                # so I will assume if it exists it is fine, but if not it will create.
                dm.prepare_dataset(rows, dist)
                
                # Also prepare a default workload for it so it's ready for benchmarks
                dm.prepare_workload(rows, dist, 1000)
                
            except Exception as e:
                print(f"Failed to generate {rows} / {dist}: {e}")
            
            elapsed = time.time() - start
            print(f"Done in {elapsed:.2f}s")

    total_elapsed = time.time() - total_start
    print(f"\nAll operations completed in {total_elapsed:.2f}s")

if __name__ == "__main__":
    main()
