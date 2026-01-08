import subprocess
import json
import csv
import pandas as pd
from pathlib import Path
import sys

def run_benchmark(label, cmd_args):
    print(f"Running {label}...")
    cmd = ["poetry", "run", "python", "pipeline_fd_cdf.py"] + cmd_args
    subprocess.run(cmd, check=True)
    
    # Read the summary.json generated
    # The script currently writes to artifacts_fd_cdf/summary.json (default out-dir)
    # We might need to check if out-dir is changed in args, but let's assume default or parse it.
    
    out_dir = "artifacts_fd_cdf"
    if "--out-dir" in cmd_args:
        idx = cmd_args.index("--out-dir")
        out_dir = cmd_args[idx+1]
        
    summary_path = Path(out_dir) / "summary.json"
    if not summary_path.exists():
        print(f"Warning: No summary found for {label}")
        return None
        
    with open(summary_path) as f:
        data = json.load(f)
        
    # Flatten/Structure important fields
    row = {
        "Experiment": label,
        "Dataset": Path(cmd_args[cmd_args.index("--input-csv")+1] if "--input-csv" in cmd_args else "Generated").name,
        "Bins_Used": data.get("bins"),
        
        "Train_Time_Hybrid": data.get("hybrid_train_time_total"),
        "Train_Time_Hist": data.get("hist_build_time"),
        
        "Inf_Time_Hybrid": data.get("hybrid_inference_time"),
        "Inf_Time_Hist": data.get("hist_inference_time"),
        
        "Hybrid_Median_QErr": data["hybrid_cdf"]["QErr_median"],
        "Hybrid_P95_QErr": data["hybrid_cdf"]["QErr_p95"],
        "Hybrid_MAE": data["hybrid_cdf"]["MAE"],
        
        "Hist_Median_QErr": data["histogram_baseline"]["QErr_median"],
        "Hist_P95_QErr": data["histogram_baseline"]["QErr_p95"],
        "Hist_MAE": data["histogram_baseline"]["MAE"],
    }
    return row

def main():
    results = []
    
    # 1. Zipf (Unconstrained / FD)
    zipf_path = "artifacts_fd_cdf/data_zipf.csv"
    results.append(run_benchmark("Zipf_Unconstrained_FD", ["--input-csv", zipf_path, "--fd", "--eval-n", "1000"]))
    results.append(run_benchmark("Zipf_Constrained_20Bins", ["--input-csv", zipf_path, "--bins", "20", "--eval-n", "1000"]))
    
    # 2. TPC-H Columns
    tpch_cols = [
        "tpch_lineitem_extendedprice.csv",
        "tpch_orders_totalprice.csv",
        "tpch_part_retailprice.csv",
        "tpch_customer_acctbal.csv",
        "tpch_partsupp_supplycost.csv"
    ]
    
    for col_file in tpch_cols:
        if Path(col_file).exists():
           results.append(run_benchmark(f"TPCH_{col_file.split('.')[0]}", ["--input-csv", col_file, "--fd", "--eval-n", "500"]))
           
    # Save to CSV
    df = pd.DataFrame([r for r in results if r is not None])
    cols = [
        "Experiment", "Dataset", "Bins_Used", 
        "Train_Time_Hist", "Train_Time_Hybrid", 
        "Inf_Time_Hist", "Inf_Time_Hybrid",
        "Hybrid_Median_QErr", "Hybrid_P95_QErr", "Hybrid_MAE",
        "Hist_Median_QErr", "Hist_P95_QErr", "Hist_MAE"
    ]
    df = df[cols]
    
    out_csv = "all_benchmark_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved consolidated results to {out_csv}")
    print(df)

if __name__ == "__main__":
    main()
