import json
import pandas as pd
import numpy as np
from pathlib import Path
import re

def aggregate_summaries(results_dir="results"):
    root = Path(results_dir)
    data = []
    
    # Locate all result CSVs
    # Pattern: results/{rows}/{dist}/{workload}.csv
    # We want to avoid "_drift.csv" for now unless requested, but let's just grab main workload files
    
    for row_dir in root.iterdir():
        if not row_dir.is_dir(): continue
        
        for dist_dir in row_dir.iterdir():
            if not dist_dir.is_dir(): continue
            
            # Find all CSV files that look like workload results
            for res_file in dist_dir.glob("*.csv"):
                if "drift" in res_file.name or "benchmark_summary" in res_file.name:
                    continue

                workload_name = res_file.stem
                
                try:
                    df = pd.read_csv(res_file)
                    if "Q_Error" not in df.columns:
                        continue
                        
                    # Calculate stats per Model
                    models = df["Model"].unique()
                    
                    for model in models:
                        subset = df[df["Model"] == model]
                        q_errs = subset["Q_Error"]
                        
                        med = q_errs.median()
                        p95 = q_errs.quantile(0.95)
                        avg = q_errs.mean()
                        
                        data.append({
                            "Distribution": dist_dir.name,
                            "Rows": row_dir.name,
                            "Workload": workload_name,
                            "Model": model,
                            "Avg Q-Err": avg,
                            "95% Q-Err": p95,
                            "Median Q-Err": med
                        })
                        
                except Exception as e:
                    print(f"Error processing {res_file}: {e}")

    if not data:
        print("No result CSVs found.")
        return

    df = pd.DataFrame(data)
    # Sort for readability
    df = df.sort_values(by=["Rows", "Distribution", "Workload", "Model"])
    
    out_csv = "benchmark_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"Aggregated summary saved to {out_csv}")
    print(df.to_string())

if __name__ == "__main__":
    aggregate_summaries()
