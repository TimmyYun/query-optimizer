import os
import json
import csv
import subprocess
from pathlib import Path

RESULTS_FILE = "datasets/files/benchmarks/final_benchmark.csv"
OUT_DIR = "experiments/artifacts_bench"

def run_bench(dataset_name, dist, shift, drift_rows=500000):
    desc = f"Insert {drift_rows} (+{shift} Shift)" if drift_rows > 0 else "No Data Insert (Drift=0)"
    
    out_path = Path(f"{OUT_DIR}/{dataset_name}_{shift}_{drift_rows}")
    path_drift = out_path / "drift_summary.json"
    path_init = out_path / "summary.json"
    
    # 1. Run Experiment (Only if missing)
    if not path_drift.exists():
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
    if not path_drift.exists(): return []
    
    # Load Init Data (Phase 1)
    data_init = {}
    if path_init.exists():
        with open(path_init) as f:
            data_init = json.load(f)
            
    # Load Drift/Repair Data (Phase 2 & 3)
    with open(path_drift) as f:
        data_drift = json.load(f)
        
    d_metrics = data_drift["metrics"]
    d_timings = data_drift["timings"]
    d_resources = data_drift["resources"]
    
    # Init Metrics (May not exist for EquiHist)
    i_static = data_init.get("hist_width_metrics", {})
    i_hybrid = data_init.get("hybrid_metrics", {})
    i_eh = data_init.get("equihist_init_metrics", {})
    
    # Init Timings (New)
    i_timings = data_init.get("timings", {})
    
    # Common Resource Metrics
    # Fallback for old results (if any)
    mem_buckets = d_resources.get("memory_buckets_bytes", d_resources.get("memory_bytes", 0)) / 1024
    mem_total = d_resources.get("memory_total_bytes", d_resources.get("memory_bytes", 0)) / 1024
    
    # --- Helper to safe get ---
    def fmt(val):
        return f"{val}" if isinstance(val, (int, float)) else "N/A"

    # --- Construct Rows (One per Approach) ---
    
    rows = []
    
    # 1. Static Equi-Width
    # Initial: Yes, Drift: Yes (Stale), Repair: Yes (Full Rebuild)
    row_static = {
        "Dataset": dataset_name,
        "Approach": "Static Equi-Width",
        
        # Initial
        "Init Training time (s)": fmt(d_timings['hist_build']),
        "Init QErr Median": fmt(i_static.get('QErr_median', 'N/A')),
        "Init QErr P95": fmt(i_static.get('QErr_p95', 'N/A')),
        "Init MAE": fmt(i_static.get('MAE', 'N/A')),
        "Init Inference time (s)": fmt(d_timings['static_inf']),
        "Init Memory (KB)": fmt(mem_buckets),
        
        # Drift (Stale)
        "Rows inserted": drift_rows,
        "Drift Scenario": shift,
        "Drift QErr Median": fmt(d_metrics['static_stale']['QErr_median']),
        "Drift QErr P95": fmt(d_metrics['static_stale']['QErr_p95']),
        "Drift MAE": fmt(d_metrics['static_stale']['MAE']),
        "Drift Inference time (s)": fmt(d_timings['static_inf']),
        "Drift Memory (KB)": fmt(mem_buckets),
        
        # Repair (Full Rebuild)
        "Retrain time (s)": fmt(d_timings.get('static_rebuild', 'N/A')),
        "Final QErr Median": fmt(d_metrics.get('static_rebuilt', {}).get('QErr_median', 'N/A')),
        "Final QErr P95": fmt(d_metrics.get('static_rebuilt', {}).get('QErr_p95', 'N/A')),
        "Final MAE": fmt(d_metrics.get('static_rebuilt', {}).get('MAE', 'N/A')),
        "Final Inference time (s)": fmt(d_timings['static_inf']), # Inference is same algo
        "Final Memory (KB)": fmt(mem_buckets),
    }
    
    # 2. EquiHist (Online) - No distinct "rebuild", it's continuous
    # Initial: Initial Learning Phase (Phase 1)
    # Drift: Online Adaptation (Phase 2)
    # Final: Converged State Check (Phase 3)
    row_eh = {
        "Dataset": dataset_name,
        "Approach": "EquiHist (Online)",
        
        # Initial
        "Init Training time (s)": fmt(i_timings.get('equihist_init_train', 'N/A')), # Initial learning overhead (Update sum)
        "Init QErr Median": fmt(i_eh.get('QErr_median', 'N/A')),
        "Init QErr P95": fmt(i_eh.get('QErr_p95', 'N/A')),
        "Init MAE": fmt(i_eh.get('MAE', 'N/A')),
        "Init Inference time (s)": fmt(i_timings.get('equihist_init_inf', 'N/A')), # Measured inference time
        "Init Memory (KB)": fmt(mem_buckets),
        
        # Drift - Effective Performance during drift
        "Rows inserted": drift_rows,
        "Drift Scenario": shift,
        "Drift QErr Median": fmt(d_metrics['equihist_online']['QErr_median']),
        "Drift QErr P95": fmt(d_metrics['equihist_online']['QErr_p95']),
        "Drift MAE": fmt(d_metrics['equihist_online']['MAE']),
        "Drift Inference time (s)": fmt(d_timings.get('equihist_drift_inf', 'N/A')), # Measured inference time
        "Drift Memory (KB)": fmt(mem_buckets),
        
        # Repair - Final State Check
        "Retrain time (s)": fmt(d_timings.get('equihist_retrain', 'N/A')), # Drift learning overhead (Update sum)
        "Final QErr Median": fmt(d_metrics.get('equihist_final', {}).get('QErr_median', 'N/A')),
        "Final QErr P95": fmt(d_metrics.get('equihist_final', {}).get('QErr_p95', 'N/A')),
        "Final MAE": fmt(d_metrics.get('equihist_final', {}).get('MAE', 'N/A')),
        "Final Inference time (s)": fmt(d_timings.get('equihist_final_inf', 'N/A')), # Measured inference time
        "Final Memory (KB)": fmt(mem_buckets),
    }
    
    # 3. Hybrid (Repaired) - Uses Buckets + Models
    init_train_time = d_timings['hist_build'] + d_timings['ml_train']
    row_hybrid = {
        "Dataset": dataset_name,
        "Approach": "Hybrid (Repaired)",
        
        # Initial
        "Init Training time (s)": fmt(init_train_time),
        "Init QErr Median": fmt(i_hybrid.get('QErr_median', 'N/A')),
        "Init QErr P95": fmt(i_hybrid.get('QErr_p95', 'N/A')),
        "Init MAE": fmt(i_hybrid.get('MAE', 'N/A')),
        "Init Inference time (s)": fmt(d_timings['hybrid_inf']),
        "Init Memory (KB)": fmt(mem_total),
        
        # Drift (Stale)
        "Rows inserted": drift_rows,
        "Drift Scenario": shift,
        "Drift QErr Median": fmt(d_metrics.get('hybrid_stale', {}).get('QErr_median', 'N/A')), 
        "Drift QErr P95": fmt(d_metrics.get('hybrid_stale', {}).get('QErr_p95', 'N/A')),
        "Drift MAE": fmt(d_metrics.get('hybrid_stale', {}).get('MAE', 'N/A')),
        "Drift Inference time (s)": fmt(d_timings['hybrid_inf']),
        "Drift Memory (KB)": fmt(mem_total),
        
        # Repair
        "Retrain time (s)": fmt(d_timings['repair']),
        "Final QErr Median": fmt(d_metrics['hybrid_repaired']['QErr_median']),
        "Final QErr P95": fmt(d_metrics['hybrid_repaired']['QErr_p95']),
        "Final MAE": fmt(d_metrics['hybrid_repaired']['MAE']),
        "Final Inference time (s)": fmt(d_timings['hybrid_inf']),
        "Final Memory (KB)": fmt(mem_total),
    }
    
    return [row_static, row_eh, row_hybrid]

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    
    all_rows = []
    # 1. No Drift (Baseline)
    for ds_name, dist in [("Uniform","uniform"), ("Normal","normal"), ("Zipf","zipf"), ("Sparse","sparse_cluster"), ("IMDB", "imdb"), ("Census", "census")]:
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

    for s in shifts:
        all_rows.extend(run_bench("IMDB", "imdb", s))

    for s in shifts:
        all_rows.extend(run_bench("Census", "census", s))
    
    # Write CSV
    if all_rows:
        headers = list(all_rows[0].keys())
        with open(RESULTS_FILE, "w") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Final Benchmark saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
