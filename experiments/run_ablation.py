import os
import json
import csv
import subprocess
from pathlib import Path

RESULTS_FILE = "ablation_results.csv"
OUT_DIR = "experiments/artifacts"

def run_experiment(exp_name, config):
    print(f"Running {exp_name} with config: {config}")
    
    cmd = ["python", "main.py"]
    for k, v in config.items():
        cmd.append(f"--{k}")
        cmd.append(str(v))
        
    exp_out = f"{OUT_DIR}/{exp_name}"
    cmd.append("--out-dir")
    cmd.append(exp_out)
    
    # Defaults
    if "dist" not in config: cmd.extend(["--dist", "zipf"])
    if "rows" not in config: cmd.extend(["--rows", "1000000"])
    
    subprocess.run(cmd, check=True)
    
    # Read Result
    res_path = Path(exp_out) / "drift_summary.json"
    if not res_path.exists():
        print(f"Error: {res_path} not found")
        return None
        
    with open(res_path) as f:
        data = json.load(f)
        
    metrics = data["metrics"]
    timings = data["timings"]
    
    # Extract relevant metrics
    row = {
        "Exp": exp_name,
        "Config": str(config),
        "Var_Param": list(config.keys())[0] if config else "Base",
        "Var_Value": list(config.values())[0] if config else "Base",
        
        "Static_MAE": metrics["static_stale"]["MAE"],
        "EquiHist_MAE": metrics["equihist_online"]["MAE"],
        "Hybrid_Repair_MAE": metrics["hybrid_repaired"]["MAE"],
        "Repair_Time": timings["repair"]
    }
    return row

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # EXP B: NDV Threshold
    for ndv in [0, 50, 200, 1000]:
        row = run_experiment(f"ExpB_NDV_{ndv}", {"ndv-threshold": ndv})
        if row: results.append(row)
        
    # EXP C: Drift Intensity (Drift Rows)
    # Base rows=1M. Drift 1% = 10k, 10% = 100k, 50% = 500k, 100% = 1M
    for d_rows in [10_000, 100_000, 500_000, 1_000_000]:
        row = run_experiment(f"ExpC_Drift_{d_rows}", {"drift-rows": d_rows, "drift-dist": "anti_zipf"})
        if row: results.append(row)
        
    # EXP D: Learning Rate
    for lr in [0.1, 0.5, 1.0, 2.0]:
        row = run_experiment(f"ExpD_LR_{lr}", {"eh-lr": lr, "drift-dist": "anti_zipf", "drift-rows": 500_000}) 
        if row: results.append(row)
        
    # Save CSV
    if results:
        with open(RESULTS_FILE, "w") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved results to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
