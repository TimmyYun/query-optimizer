#!/usr/bin/env python3

"""
Query Optimizer Benchmark Script

This script benchmarks different selectivity estimation strategies (histograms and learned models)
under two scenarios:
1. Initial Build: Performance on static data.
2. Data Drift: Performance when data distribution changes (inserts).

Models compared:
- Equi-Width Histogram (Standard Baseline)
- EquiHist (Online Learner/Adaptive)
- Hybrid Estimator (Combines Histograms with ML models like CDFs)

Usage:
    python main.py --mode static --rows 10000000
    python main.py --mode drift --rows 10000000
"""

import argparse
import json
import time
import pickle
import shutil
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# New Modular Imports
from models import (
    Bucket, EquiWidthHistogram, 
    EquiHistLearner, HybridEstimator,
    identify_bad_buckets, summarize, q_error_vec
)
from datasets import (
    DatasetManager,
    scan_min_max_count, build_frequency_and_sample,
    generate_boxplots, plot_data_distribution, plot_model_comparison,
    gen_values, save_csv_column
)
from workload import RangeQuery, load_workload_csv
import copy
from aggregate_results import aggregate_summaries

def plot_q_error_boxplots(result_csv_path, output_dir):
    """
    Generates a box plot for Q-Error distributions across different models.
    """
    try:
        df = pd.read_csv(result_csv_path)
        
        plt.figure(figsize=(10, 6))
        # Filter out extremely high q-errors for visualization if needed, 
        # but boxplots usually handle outliers. 
        # We plot log scale as Q-Error is multiplicative.
        
        sns.boxplot(data=df, x='Model', y='Q_Error', hue='Phase')
        plt.yscale('log')
        plt.title('Q-Error Distribution by Model')
        plt.ylabel('Q-Error (Log Scale)')
        plt.xlabel('Model')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()
        
        plot_path = output_dir / "q_error_boxplot.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Q-Error plot saved to {plot_path}")
        
    except Exception as e:
        print(f"Error plotting Q-Error: {e}")

def main():

    """
    Main execution entry point.
    
    1. Parses arguments for data distribution and sizes.
    2. Generates (or loads) initial dataset.
    3. Phase 1: Builds and evaluates models on static data.
    4. Phase 2: Simulates data drift by appending new data and evaluates.
    5. Phase 3: Attempts to repair the Hybrid model and evaluates repairs.
    6. Phase 4: Performs a full static rebuild to establish a "perfect" baseline.
    7. Saves metrics and timings to JSON artifacts.
    """
    parser = argparse.ArgumentParser(description="Run Query Optimizer Benchmark")
    parser.add_argument("--mode", type=str, choices=["static", "drift"], default="static", help="Experiment mode")
    parser.add_argument("--dist", type=str, default="zipf")
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--drift-rows", type=int, default=200_000)
    parser.add_argument("--drift-dist", type=str, default="normal")
    parser.add_argument("--drift-shift", type=int, default=50_000, help="Shift magnitude for drift data")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--eval-n", dest="eval_n", type=str, default="1000", help="Workload name/size to evaluate (e.g. 1000 or 1000_narrow)")
    parser.add_argument("--bins", type=int, default=100, help="Number of bins for Equi-Width Histogram (Default: 100)")
    parser.add_argument("--skewed", action="store_true", help="Use skewed workload for Head heavy evaluation")
    parser.add_argument("--recreate", action="store_true", help="Force regeneration of the dataset even if cached")
    
    # Batch specific params
    parser.add_argument("--experiment-name", type=str, default=None, help="Suffix for output files")
    
    # Ablation Hyperparams
    parser.add_argument("--eh-lr", type=float, default=0.5, help="EquiHist Learning Rate")
    
    args = parser.parse_args()
    
    # Auto-generate Experiment ID if not provided
    if args.experiment_name is None:
        # Auto-incrementing Experiment ID
        base_dir = Path(args.out_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        existing_ids = [int(d.name) for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        next_id = max(existing_ids) + 1 if existing_ids else 1
        args.experiment_name = str(next_id)
        
    print(f"=== Experiment ID: {args.experiment_name} ===")
    
    # Update Output Directory to include Experiment ID
    # This ensures results/{experiment_id}/{rows}/{dist} structure
    args.out_dir = str(Path(args.out_dir) / args.experiment_name)
    print(f"Results will be saved to: {args.out_dir}")
    
    if args.mode == "static":
        run_static_benchmark(args)
    elif args.mode == "drift":
        run_drift_benchmark(args)

def _run_experiment_internal(args):
    rng = np.random.default_rng(42)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    # Store raw errors for plotting
    raw_errors = []
    
    # -----------------------------------------------------
    # Phase 1: Initial Build
    # -----------------------------------------------------
    print(f"=== Phase 1: Initial Build ({args.rows} rows, {args.dist}) ===")
    
    # Use DatasetManager for Stage 1 (Dataset + Stats)
    dm = DatasetManager()
    ds_dir = dm.prepare_dataset(args.rows, args.dist, force_regeneration=args.recreate)
    
    # Load dataset metadata
    ds_path = ds_dir / "data.csv"
    with open(ds_dir / "meta.pkl", "rb") as f:
        mn, mx, N, freq, sample, n_bins, skew, kurt = pickle.load(f)
    
    # Create working copy for drift experiments
    working_ds_path = Path(args.out_dir) / "data_current_run.csv"
    shutil.copy2(ds_path, working_ds_path)
    ds_path = working_ds_path

    print(f"Using Bins: {args.bins} (User Specified/Default)")
    
    # 2a. Equi-Width Histogram (Standard Baseline)
    # This represents a traditional database histogram.
    t0_hist = time.perf_counter()
    # Refactored: Use Class Builder to create the histogram
    # Using Sample-Based Construction (Postgres-like)
    ew_hist = EquiWidthHistogram.build_from_sample(mn, mx, args.bins, sample, N)
    buckets_eq_width = ew_hist.buckets # Access buckets for other models to use as a base
    t_hist_build = time.perf_counter() - t0_hist
    
    # 2b. EquiHist (Online Learner) - Initialize
    eh_learner = EquiHistLearner(buckets_eq_width, learning_rate=args.eh_lr)
    
    # 3. Model Training (Hybrid)
    # The Hybrid Estimator aims to use ML models within buckets that have high variance/density,
    # while keeping exact counts for low-NDV buckets.
    print("Training CDF models (Skipping buckets with low NDV)...")
    # Refactored: Use Hybrid Class
    hybrid_est = HybridEstimator(buckets_eq_width)
    t_ml_train = hybrid_est.train(freq, mn, 20, rng)
    print(f"Hist Build (Width): {t_hist_build:.4f}s, ML Train: {t_ml_train:.4f}s")
    
    
    # 4. Evaluation (Initial)
    print("Generating Evaluation Workload...")
    
    # Use load_workload_csv for independent workloads
    wl_name = args.eval_n
    if args.skewed:
        wl_name = f"{args.eval_n}_skewed"
        print(f"Using Skewed Workload: {wl_name}")
        
    workload_path = Path(f"workload/{wl_name}.csv")
    if not workload_path.exists():
        raise FileNotFoundError(f"Workload file not found at {workload_path}. Run workload.py first.")
        
    all_queries = load_workload_csv(workload_path)
    
    # Compute cumulative sum for ground truth calculation and filtering
    ps = np.cumsum(freq)
    
    # Filter out queries with 0 true cardinality (selectivity)
    print(f"Original Workload Size: {len(all_queries)}")
    queries = []
    skipped_count = 0
    
    for q in all_queries:
        li = int(q.low - mn)
        ri = int(q.high - mn)
        
        # Robust check for out-of-bounds
        if ri < 0 or li >= len(ps):
            truth = 0
        else:
            if li < 0: li = 0
            if ri >= len(ps): ri = len(ps) - 1
            
            if li > ri:
                truth = 0
            else:
                truth = int(ps[ri] - (ps[li-1] if li > 0 else 0))
                
        if truth > 0:
            queries.append(q)
        else:
            skipped_count += 1
            
    print(f"Filtered Workload Size: {len(queries)} (Skipped {skipped_count} zero-result queries)")
    
    if not queries:
        print("WARNING: All queries were filtered out! No benchmark will run.")
        return
        
    y_true = []
    y_hist_width = []
    y_hybrid = []
    y_eh_init = []
    
    # Measure Baseline Inference
    t0_base = time.perf_counter()
    for q in queries:
        # Refactored: Class method
        h = ew_hist.predict(q)
        y_hist_width.append(h / N)
    t_base_inf = time.perf_counter() - t0_base
    
    
    # Measure Hybrid Inference
    t0_hyb = time.perf_counter()
    for q in queries:
        # Refactored: Class method
        c = hybrid_est.predict(q)
        y_hybrid.append(c / N)
    t_hyb_inf = time.perf_counter() - t0_hyb


    # Truth & EquiHist (Update loop)
    t_eh_update_p1 = 0.0
    t_eh_inf_p1 = 0.0
    for q in queries:
        li = int(q.low - mn)
        ri = int(q.high - mn)
        
        if ri < 0 or li >= len(ps):
            truth = 0
        else:
            if li < 0: li = 0
            if ri >= len(ps): ri = len(ps) - 1
            if li > ri:
                truth = 0
            else:
                truth = int(ps[ri] - (ps[li-1] if li > 0 else 0))
                
        y_true.append(truth / N)
        
        # EquiHist Predict + Update
        t0_inf = time.perf_counter()
        est_eh = eh_learner.predict(q)
        t_eh_inf_p1 += (time.perf_counter() - t0_inf)
        
        y_eh_init.append(est_eh / N) 
        
        
        t0_up = time.perf_counter()
        # EquiHist Feedback Loop: Update the model with the true selectivity
        eh_learner.update(q, float(truth))
        t_eh_update_p1 += (time.perf_counter() - t0_up)
        
    y_true = np.array(y_true)
    y_hist_width = np.array(y_hist_width)
    y_hybrid = np.array(y_hybrid)
    y_eh_init = np.array(y_eh_init)
    
    # -----------------------------------------------------
    # Save Results
    # -----------------------------------------------------
    result_dir = Path(args.out_dir) / str(args.rows) / args.dist
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Detailed Results per Query
    results_data = []
    
    # Helper to add results
    def add_results(phase_name, model_name, pred_arr, truth_arr):
        errors = q_error_vec(truth_arr, pred_arr)
        for i, (pred, true, err) in enumerate(zip(pred_arr, truth_arr, errors)):
            results_data.append({
                "Query_ID": i,
                "Phase": phase_name,
                "Model": model_name,
                "Prediction": float(pred),
                "Truth": float(true),
                "Q_Error": float(err)
            })

    # Phase 1 Results
    add_results("Static", "Equi-Width", y_hist_width, y_true)
    add_results("Static", "Hybrid", y_hybrid, y_true)
    add_results("Static", "EquiHist", y_eh_init, y_true)
    
    # Calculate Phase 1 Summaries for Console
    m_hist_w = summarize(y_true, y_hist_width, "Equi-Width (Standard)")
    m_hyb = summarize(y_true, y_hybrid, "Hybrid")
    m_eh_init = summarize(y_true, y_eh_init, "EquiHist (Initial Learning)")
    
    # Calculate specialized training times
    t_eh_init_total = t_hist_build + t_eh_update_p1
    
    print("\n--- Results ---")
    print(f"Equi-Width:  Median QErr={m_hist_w['QErr_median']:.4f}, Time={t_base_inf:.4f}s")
    print(f"Hybrid(FD):  Median QErr={m_hyb['QErr_median']:.4f}, Time={t_hyb_inf:.4f}s")
    print(f"EquiHist:    Median QErr={m_eh_init['QErr_median']:.4f}, InitTrain={t_eh_init_total:.4f}s, Inf={t_eh_inf_p1:.4f}s")

    # If Static Mode, we stop here and save
    if args.mode == "static":
         df_results = pd.DataFrame(results_data)
         result_file = result_dir / f"{args.eval_n}.csv"
         df_results.to_csv(result_file, index=False)
         print(f"Detailed results saved to {result_file}")
         
         # Save Summary JSON for Aggregation
         summary_data = {
             "row_count": args.rows,
             "distribution": args.dist,
             "workload_size": args.eval_n,
             "metrics": {
                 "Equi-Width": {
                     "build_time": t_hist_build,
                     "infer_time": t_base_inf,
                     "avg_q_error": m_hist_w['QErr_avg'],
                     "p95_q_error": m_hist_w['QErr_p95']
                 },
                 "Hybrid": {
                     "build_time": t_ml_train, # Includes hist build implicitly if we consider full pipeline, but t_ml_train is mostly training. 
                                               # However, Hybrid uses buckets_eq_width which took t_hist_build. 
                                               # Let's sum them for fairness or keep distinct? User asked for "Build (s)".
                                               # Usually Hybrid Build = Hist Build + ML Train.
                     "build_time_total": t_hist_build + t_ml_train,
                     "infer_time": t_hyb_inf,
                     "avg_q_error": m_hyb['QErr_avg'],
                     "p95_q_error": m_hyb['QErr_p95']
                 },
                 "EquiHist": {
                     "build_time": t_eh_init_total,
                     "infer_time": t_eh_inf_p1,
                     "avg_q_error": m_eh_init['QErr_avg'],
                     "p95_q_error": m_eh_init['QErr_p95']
                 }
             }
         }
         
         with open(result_dir / "summary.json", "w") as f:
             json.dump(summary_data, f, indent=4)
         print(f"Summary stats saved to {result_dir / 'summary.json'}")
         
         # Generate Plot
         plot_q_error_boxplots(result_file, result_dir)
         return
         
         # Generate Plot
         plot_q_error_boxplots(result_file, result_dir)
         return


    # -----------------------------------------------------
    # Phase 2: Data Drift (Insert Data)
    # -----------------------------------------------------
    print(f"\n=== Phase 2: Data Drift (Inserting {args.drift_rows} rows of {args.drift_dist}, shift={args.drift_shift}) ===")
    drift_vals = gen_values(rng, args.drift_dist, args.drift_rows, 0, 200_000, shift=args.drift_shift)
    save_csv_column(drift_vals, ds_path, mode='a')
    
    # Update Ground Truth Frequencies
    N_new = N + args.drift_rows
    print("Updating Global Frequency (Ground Truth)...")
    mn_new, mx_new, N_real = scan_min_max_count(ds_path) 
    freq_new, _ = build_frequency_and_sample(ds_path, mn_new, mx_new, N_real, 1000, 42)
    ps_new = np.cumsum(freq_new)
    
    # --- Compare Approaches under Drift ---
    
    # 1. Static Equi-Width (STALE)
    buckets_static = [Bucket(b.lo, b.hi, count=b.count) for b in buckets_eq_width]
    static_hist_stale = EquiWidthHistogram(buckets_static)
    
    # 2. EquiHist (Online Learning - CONTINUES)
    
    print("Evaluating Drift Sequence...")
    y_true_seq = []
    y_static = []
    y_eh = []
    y_hybrid_stale = []
    
    t_eh_update_p2 = 0.0
    
    for q in queries:
        li = int(q.low - mn_new)
        ri = int(q.high - mn_new)
        
        if ri < 0 or li >= len(ps_new):
            truth = 0
        else:
            if li < 0: li = 0
            if ri >= len(ps_new): ri = len(ps_new) - 1
            if li > ri:
                truth = 0
            else:
                truth = int(ps_new[ri] - (ps_new[li-1] if li > 0 else 0))
                
        actual_sel = truth / N_real
        y_true_seq.append(actual_sel)
        
        # Static
        est_static = static_hist_stale.predict(q) 
        y_static.append(est_static / N) 
        
        # EquiHist (Predict then Update)
        est_eh = eh_learner.predict(q)
        y_eh.append(est_eh / N) 
        
        t0_up = time.perf_counter()
        eh_learner.update(q, float(truth)) 
        t_eh_update_p2 += (time.perf_counter() - t0_up)
        
        # Hybrid (Stale buckets + Old Models)
        est_hyb = hybrid_est.predict(q)
        y_hybrid_stale.append(est_hyb / N)
        
    y_true_arr = np.array(y_true_seq)
    
    add_results("Drift", "Equi-Width (Stale)", y_static, y_true_arr)
    add_results("Drift", "EquiHist (Online)", y_eh, y_true_arr)
    add_results("Drift", "Hybrid (Stale)", y_hybrid_stale, y_true_arr)
    
    # -----------------------------------------------------
    # Phase 3: Adaptive Repair (Hybrid)
    # -----------------------------------------------------
    print(f"\n=== Phase 3: Hybrid Adaptive Repair ===")
    
    # Hybrid updates counts (Cheap)
    for b in buckets_eq_width:
        li = b.lo - mn_new
        ri = b.hi - mn_new
        if li < 0: li=0
        if ri >= len(freq_new): ri = len(freq_new)-1
        b.count = int(ps_new[ri] - (ps_new[li-1] if li > 0 else 0))
        
    # Check error again with updated counts
    y_hyb_counts_only = []
    for q in queries:
        y_hyb_counts_only.append(hybrid_est.predict(q) / N_real)
        
    bad_indices = identify_bad_buckets(queries, y_true_arr, np.array(y_hyb_counts_only), buckets_eq_width, threshold_q=2.0)
    print(f"Identified {len(bad_indices)}/{len(buckets_eq_width)} buckets needing repair.")
    
    if bad_indices:
        print("Retraining specific buckets...")
        hybrid_est.train(freq_new, mn_new, 50, rng, bucket_indices=bad_indices)
            
    # Final Hybrid Eval
    y_hyb_final = []
    for q in queries:
        y_hyb_final.append(hybrid_est.predict(q) / N_real)
    
    add_results("Repair", "Hybrid (Repaired)", y_hyb_final, y_true_arr)
    
    # Save Final CSV including Drift/Repair
    df_results = pd.DataFrame(results_data)
    result_file = result_dir / f"{args.eval_n}_drift.csv"
    df_results.to_csv(result_file, index=False)
    print(f"Detailed drift results saved to {result_file}")
    
    # Generate Plot
    plot_q_error_boxplots(result_file, result_dir)
    
    # Generate Plot
    plot_q_error_boxplots(result_file, result_dir)


def run_static_benchmark(args):
    """
    Runs static benchmarks for all distributions or a specific one.
    """
    if args.dist == "all":
        distributions = ["uniform", "normal", "zipf", "sparse_cluster", "anti_zipf"]
    else:
        distributions = [args.dist]
        
    for dist in distributions:
        print(f"\n>>> Running for Distribution: {dist} <<<")
        # Create a specific args object for this run
        current_args = copy.deepcopy(args)
        current_args.dist = dist
        
        try:
            _run_experiment_internal(current_args)
        except Exception as e:
            print(f"Failed to run for {dist}: {e}")
            import traceback
            traceback.print_exc()
            
    # Auto-aggregate results at the end of the batch
    print("\n>>> Aggregating All Results <<<")
    aggregate_summaries(args.out_dir)

def run_drift_benchmark(args):
    """
    Runs a batch of drift experiments over multiple scenarios.
    Migrated from drift_benchmark.py
    """
    scenarios = [
        ("zipf", "normal", 50000),      # Shifted insert
        ("uniform", "sparse_cluster", 0),  # Radical distribution change
        ("normal", "zipf", 20000),      # Overlap drift
        ("anti_zipf", "uniform", 100000) # Out of range drift
    ]
    
    all_results = []
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    experiment_suffix = f"_{args.experiment_name}" if args.experiment_name else ""
    excel_path = output_dir / f"drift_benchmark_results{experiment_suffix}.xlsx"
    
    print(f"Starting Drift Benchmark Batch (Rows: {args.rows}, DriftRows: {args.drift_rows})")
    
    for init_dist, drift_dist, shift in scenarios:
        scenario_name = f"{init_dist}_to_{drift_dist}_s{shift}"
        print(f"\n>>> Processing Scenario: {scenario_name} <<<")
        
        scenario_args = copy.deepcopy(args)
        scenario_args.dist = init_dist
        scenario_args.drift_dist = drift_dist
        scenario_args.drift_shift = shift
        scenario_args.out_dir = str(output_dir / scenario_name)
        
        try:
            _run_experiment_internal(scenario_args)
            
            with open(Path(scenario_args.out_dir) / "summary.json", "r") as f:
                res = json.load(f)
                
            metrics = res["metrics"]
            summary_row = {
                "Scenario": scenario_name,
                "Init_Dist": init_dist,
                "Drift_Dist": drift_dist,
                "Shift": shift,
                "Static_Stale_QErr": metrics["static_stale"]["QErr_median"],
                "Static_Rebuilt_QErr": metrics["static_rebuilt"]["QErr_median"],
                "EH_Adaptive_QErr": metrics["equihist_online"]["QErr_median"],
                "Hybrid_Stale_QErr": metrics["hybrid_stale"]["QErr_median"],
                "Hybrid_Repaired_QErr": metrics["hybrid_repaired"]["QErr_median"]
            }
            all_results.append(summary_row)
        except Exception as e:
            print(f"Error processing scenario {scenario_name}: {e}")
            continue

    df_summary = pd.DataFrame(all_results)
    df_summary.to_excel(excel_path, index=False)
    print(f"\nFinal Drift Summary saved to {excel_path}")

if __name__ == "__main__":
    main()
