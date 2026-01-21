import os
import json
import csv
import subprocess
from pathlib import Path

RESULTS_FILE = "final_benchmark.csv"
OUT_DIR = "experiments/artifacts_bench"

def run_bench(dataset_name, dist, shift, drift_rows=500000):
    desc = f"Insert {drift_rows} (+{shift} Shift)" if drift_rows > 0 else "No Data Insert (Drift=0)"
    
    out_path = Path(f"{OUT_DIR}/{dataset_name}_{shift}_{drift_rows}")
    res_path = out_path / "drift_summary.json"
    
    # 1. Run Experiment (Only if missing)
    if not res_path.exists():
        print(f"Benchmarking {dataset_name} ({dist}) {desc}...")
        cmd = [
            "python", "main.py",
            "--dist", dist,
            "--rows", "1000000",
            "--drift-rows", str(drift_rows),
            "--drift-dist", dist,
            "--drift-shift", str(shift),
            "--out-dir", str(out_path)
        ]
        subprocess.run(cmd, check=True)
    else:
        print(f"Skipping execution for {dataset_name} ({dist}) {desc} (Results exist)")
    
    # 2. Parse Results
    if not res_path.exists(): return []
    
    with open(res_path) as f:
        data = json.load(f)
        
    metrics = data["metrics"]
    timings = data["timings"]
    resources = data["resources"]
    
    # Common Resource Metrics
    disk_mb = resources["disk_bytes"] / (1024*1024)
    mem_kb = resources["memory_bytes"] / 1024
    
    drift_desc = desc
    
    # 3. Extract Rows per Approach
    
    # Approach 1: Static Equi-Width (Baseline 1)
    row_static = {
        "Dataset": dataset_name,
        "Drift Scenario": drift_desc,
        "Approach": "Static Equi-Width",
        "Training time (s)": f"{timings['hist_build']:.9f}",
        "QErr Median": f"{metrics['static_stale']['QErr_median']:.9f}",
        "QErr P95": f"{metrics['static_stale']['QErr_p95']:.9f}",
        "MAE": f"{metrics['static_stale']['MAE']:.9f}",
        "Inference time (s)": f"{timings['static_inf']:.9f}",
        "Memory (KB)": f"{mem_kb:.4f}",
        "Disk usage (MB)": f"{disk_mb:.4f}",
        "P95": f"{metrics['static_stale']['QErr_p95']:.9f}" 
    }
    
    # Approach 2: EquiHist (Baseline 2)
    row_eh = {
        "Dataset": dataset_name,
        "Drift Scenario": drift_desc,
        "Approach": "EquiHist (Online)",
        "Training time (s)": "0.000000000",
        "QErr Median": f"{metrics['equihist_online']['QErr_median']:.9f}",
        "QErr P95": f"{metrics['equihist_online']['QErr_p95']:.9f}",
        "MAE": f"{metrics['equihist_online']['MAE']:.9f}",
        "Inference time (s)": "N/A", 
        "Memory (KB)": f"{mem_kb:.4f}",
        "Disk usage (MB)": f"{disk_mb:.4f}",
        "P95": f"{metrics['equihist_online']['QErr_p95']:.9f}"
    }

    # Approach 3: Hybrid Repaired (Ours)
    total_train = timings['hist_build'] + timings['ml_train'] + timings['repair']
    row_hybrid = {
        "Dataset": dataset_name,
        "Drift Scenario": drift_desc,
        "Approach": "Hybrid (Repaired)",
        "Training time (s)": f"{total_train:.9f}",
        "QErr Median": f"{metrics['hybrid_repaired']['QErr_median']:.9f}",
        "QErr P95": f"{metrics['hybrid_repaired']['QErr_p95']:.9f}",
        "MAE": f"{metrics['hybrid_repaired']['MAE']:.9f}",
        "Inference time (s)": f"{timings['hybrid_inf']:.9f}",
        "Memory (KB)": f"{mem_kb:.4f}",
        "Disk usage (MB)": f"{disk_mb:.4f}",
        "P95": f"{metrics['hybrid_repaired']['QErr_p95']:.9f}"
    }
    
    return [row_static, row_eh, row_hybrid]

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    
    all_rows = []
    # 1. No Drift (Baseline)
    for ds_name, dist in [("Uniform","uniform"), ("Normal","normal"), ("Zipf","zipf"), ("Sparse","sparse_cluster")]:
         all_rows.extend(run_bench(ds_name, dist, shift=0, drift_rows=0))
         
    shifts = [0, 10_000, 25_000, 50_000, 100_000]
    
    # 2. Drift Scenarios
    for s in shifts:
        all_rows.extend(run_bench("Uniform", "uniform", s))
    
    for s in shifts:
        all_rows.extend(run_bench("Normal", "normal", s))
    
    for s in shifts:
        all_rows.extend(run_bench("Zipf", "zipf", s))
    
    for s in shifts:
        all_rows.extend(run_bench("Sparse", "sparse_cluster", s))
    
    # Write CSV
    if all_rows:
        headers = list(all_rows[0].keys())
        with open(RESULTS_FILE, "w") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nFinal Benchmark saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
