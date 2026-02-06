
import json
import pandas as pd
from pathlib import Path
import argparse

def aggregate_summaries(results_dir="results"):
    root = Path(results_dir)
    data = []
    
    # Walk through year/dist directories
    # Structure: results/{rows}/{dist}/summary.json
    
    for row_dir in root.iterdir():
        if not row_dir.is_dir(): continue
        
        for dist_dir in row_dir.iterdir():
            if not dist_dir.is_dir(): continue
            
            summary_path = dist_dir / "summary.json"
            if summary_path.exists():
                try:
                    with open(summary_path, "r") as f:
                        res = json.load(f)
                    
                    rows = res.get("row_count", row_dir.name)
                    dist = res.get("distribution", dist_dir.name)
                    metrics = res.get("metrics", {})
                    
                    for model_name, stats in metrics.items():
                        # "Distribution", "Model", "Build (s)", "Infer (s)", "Avg Q-Err", "95% Q-Err"
                        data.append({
                            "Distribution": dist, 
                            "Rows": rows, 
                            "Model": model_name,
                            "Build (s)": stats.get("build_time_total", stats.get("build_time", 0)),
                            "Infer (s)": stats.get("infer_time", 0),
                            "Avg Q-Err": stats.get("avg_q_error", 0),
                            "95% Q-Err": stats.get("p95_q_error", 0)
                        })
                except Exception as e:
                    print(f"Skipping {summary_path} due to error: {e}")

    if not data:
        print("No summary.json files found.")
        return

    df = pd.DataFrame(data)
    # Sort for readability
    df = df.sort_values(by=["Rows", "Distribution", "Model"])
    
    out_csv = "benchmark_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"Aggregated summary saved to {out_csv}")
    print(df)

if __name__ == "__main__":
    aggregate_summaries()
