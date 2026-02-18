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
    generate_boxplots, plot_data_distribution, plot_model_comparison
)
from workload import RangeQuery, load_workload_csv
import copy

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
        
        sns.boxplot(data=df, x='Model', y='Q_Error', hue='Phase', showfliers=False)
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

def aggregate_summaries(results_dir="results"):
    root = Path(results_dir)
    if not root.exists():
        print(f"Results directory '{results_dir}' does not exist.")
        return

    # Iterate over experiment IDs (e.g., results/1, results/2)
    for experiment_dir in root.iterdir():
        if not experiment_dir.is_dir():
            continue
        
        # Skip hidden directories or files
        if experiment_dir.name.startswith("."):
             continue

        experiment_id = experiment_dir.name
        print(f"Processing Experiment {experiment_id}...")
        
        all_data = []

        # Iterate over Rows (e.g., results/1/1000)
        for row_dir in experiment_dir.iterdir():
            if not row_dir.is_dir() or not row_dir.name.isdigit():
                continue
            
            rows = int(row_dir.name)
            
            # Iterate over Distributions (e.g., results/1/1000/uniform)
            for dist_dir in row_dir.iterdir():
                if not dist_dir.is_dir():
                    continue
                
                dist_name = dist_dir.name
                
                # Load summary.json for timing data
                summary_json_path = dist_dir / "summary.json"
                timing_data = {}
                if summary_json_path.exists():
                    try:
                        with open(summary_json_path, "r") as f:
                            summary_json = json.load(f)
                            # structure: metrics -> ModelName -> { build_time, infer_time, ... }
                            if "metrics" in summary_json:
                                timing_data = summary_json["metrics"]
                    except Exception as e:
                        print(f"  Warning: Could not read {summary_json_path}: {e}")

                # Find result CSVs (e.g. 1000.csv)
                for res_file in dist_dir.glob("*.csv"):
                    if "drift" in res_file.name or "summary" in res_file.name:
                        continue
                    
                    workload_name = res_file.stem
                    
                    try:
                        df = pd.read_csv(res_file)
                        if "Q_Error" not in df.columns or "Model" not in df.columns:
                            continue
                        
                        models = df["Model"].unique()
                        
                        for model in models:
                            subset = df[df["Model"] == model]
                            q_errs = subset["Q_Error"]
                            
                            med_q = q_errs.median()
                            p25_q = q_errs.quantile(0.25)
                            p75_q = q_errs.quantile(0.75)
                            p95_q = q_errs.quantile(0.95)
                            avg_q = q_errs.mean()
                            
                            train_time = None
                            infer_time = None
                            
                            # Attempt match with timing data
                            if model in timing_data:
                                m_metrics = timing_data[model]
                                train_time = m_metrics.get("build_time")
                                infer_time = m_metrics.get("infer_time")
                            
                            all_data.append({
                                "Distribution": dist_name,
                                "Rows": rows,
                                "Workload": workload_name,
                                "Model": model,
                                "Avg Q-Error": avg_q,
                                "25% Q-Error": p25_q,
                                "75% Q-Error": p75_q,
                                "95% Q-Error": p95_q,
                                "Median Q-Error": med_q,
                                "Training Time (s)": train_time,
                                "Inference Time (s)": infer_time
                            })

                    except Exception as e:
                        print(f"  Error processing {res_file}: {e}")

        if not all_data:
            print(f"  No data found for Experiment {experiment_id}.")
            continue

        # Create DataFrame
        df_summary = pd.DataFrame(all_data)
        
        # Sort for readability
        df_summary = df_summary.sort_values(by=["Rows", "Distribution", "Workload", "Model"])
        
        # Save to CSV in the experiment directory
        output_path = experiment_dir / "summary.csv"
        try:
            df_summary.to_csv(output_path, index=False)
            print(f"  Saved summary to {output_path}")
            print(df_summary.head().to_string())
        except Exception as e:
            print(f"  Failed to save CSV file: {e}")

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
    parser.add_argument("--mode", type=str, choices=["static"], default="static", help="Experiment mode")
    
    parser.add_argument("--dist", type=str, default="all")
    parser.add_argument("--rows", type=int, default=60_000_000)
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--eval-n", dest="eval_n", type=str, default="1000000", help="Workload name/size to evaluate (e.g. 1000 or 1000_narrow)")
    # Batch specific params
    parser.add_argument("--experiment-name", type=str, default=None, help="Suffix for output files")
    
    # Ablation Hyperparams
    parser.add_argument("--eh-lr", type=float, default=0.5, help="EquiHist Learning Rate")
    
    # Hybrid Specific Hyperparams
    parser.add_argument("--hybrid-points", type=int, default=20, help="Hybrid points per bucket")
    parser.add_argument("--hybrid-ident", type=float, default=1e-4, help="Hybrid identity MSE threshold")
    parser.add_argument("--hybrid-penalty", type=float, default=1.5, help="Hybrid model selection penalty")
    
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

def _run_experiment_internal(args):
    rng = np.random.default_rng(42)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Phase 1: Initial Build
    # -----------------------------------------------------
    print(f"=== Phase 1: Initial Build ({args.rows} rows, {args.dist}) ===")
    
    # Use DatasetManager for Stage 1 (Dataset + Stats)
    dm = DatasetManager()
    # Phase 1: Ensure dataset exists (modified to not generate)
    ds_dir = dm.ensure_dataset_exists(args.rows, args.dist)
    
    # Load dataset metadata
    ds_path = ds_dir / "data.csv"
    with open(ds_dir / "meta.pkl", "rb") as f:
        mn, mx, N, freq, sample, n_bins, skew, kurt = pickle.load(f)
    
    # 1. Create Buckets
    print(f"Generating Buckets (Bins: {n_bins})...")

    # Standard Equi-Width Buckets (For all models: EW-Hist, EquiHist & Hybrid)
    t0_hist = time.perf_counter()
    print("Building Equi-Width Buckets...")
    ew_hist = EquiWidthHistogram.build(mn, mx, n_bins, freq)
    ew_hist_build = time.perf_counter() - t0_hist
    buckets_ew = ew_hist.buckets
    
    # Initialize Baseline Models
    eh_learner = EquiHistLearner(buckets_ew, learning_rate=args.eh_lr) 
    
    t_hist_build = time.perf_counter() - t0_hist
    
    # 3. Model Training (Hybrid)
    # The Hybrid Estimator aims to use ML models within buckets that have high variance/density,
    # while keeping exact counts for low-NDV buckets.
    print("Training CDF models (Hybrid using Equi-Width Buckets)...")
    # Refactored: Use Hybrid Class
    hybrid_est = HybridEstimator(
        buckets_ew,
        identity_threshold=args.hybrid_ident,
        mlp_penalty=args.hybrid_penalty,
        fourier_penalty=args.hybrid_penalty
    )
    t_ml_train = hybrid_est.train(freq, mn, args.hybrid_points, rng)
    print(f"Hist Build (Width): {t_hist_build:.4f}s, ML Train: {t_ml_train:.4f}s")
    
    
    # 4. Evaluation (Initial)
    print("Generating Evaluation Workload...")
    
    # Use load_workload_csv for independent workloads
    wl_name = args.eval_n
        
    workload_path = Path(f"workload/{wl_name}/workload.csv")
    if not workload_path.exists():
        # Fallback to old path for backward compatibility or if user provided a direct file stem
        workload_path_old = Path(f"workload/{wl_name}.csv")
        if workload_path_old.exists():
             workload_path = workload_path_old
        else:
             raise FileNotFoundError(f"Workload file not found at {workload_path} or {workload_path_old}. Run workload.py first.")
        
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
    # True Cardinalities
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
    y_true = np.array(y_true)
    true_cardinalities = y_true * N

    y_hist_width = None
    t_base_inf = 0.0
    y_hybrid = None
    t_hyb_inf = 0.0
    y_eh_init = None
    t_eh_update_p1 = 0.0
    t_eh_inf_p1 = 0.0

    # Equi-Width Evaluation
    t0_base = time.perf_counter()
    y_hist_width_counts = ew_hist.predict_batch(queries)
    y_hist_width = (y_hist_width_counts / N).tolist()
    t_base_inf = time.perf_counter() - t0_base
    y_hist_width = np.array(y_hist_width)

    # Hybrid Evaluation
    t0_hyb = time.perf_counter()
    y_hybrid_counts = hybrid_est.predict_batch(queries)
    y_hybrid = (y_hybrid_counts / N).tolist()
    t_hyb_inf = time.perf_counter() - t0_hyb
    y_hybrid = np.array(y_hybrid)

    # EquiHist Evaluation (Mini-batch Vectorized)
    y_eh_init = []
    t_eh_inf_p1 = 0.0
    t_eh_update_p1 = 0.0
    batch_size = 10000
    
    for i in range(0, len(queries), batch_size):
        q_batch = queries[i : i + batch_size]
        t_batch = true_cardinalities[i : i + batch_size]
        
        # Batch Predict
        t0_inf = time.perf_counter()
        preds = eh_learner.predict_batch(q_batch)
        t_eh_inf_p1 += (time.perf_counter() - t0_inf)
        
        y_eh_init.extend(preds / N)
        
        # Sequential Update (simulate feedback loop)
        t0_up = time.perf_counter()
        for q, truth in zip(q_batch, t_batch):
            eh_learner.update(q, float(truth))
        t_eh_update_p1 += (time.perf_counter() - t0_up)
    
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
        if pred_arr is None: return # Skip if model not run
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
    print(f"Hybrid:      Median QErr={m_hyb['QErr_median']:.4f}, Time={t_hyb_inf:.4f}s")
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
            "metrics": {}
        }
        
        summary_data["metrics"]["Equi-Width"] = {
            "build_time": t_hist_build,
            "infer_time": t_base_inf,
            "median_q_error": m_hist_w['QErr_median'],
            "p25_q_error": m_hist_w['QErr_p25'],
            "p75_q_error": m_hist_w['QErr_p75'],
            "avg_q_error": m_hist_w['QErr_avg'],
            "p95_q_error": m_hist_w['QErr_p95']
        }

        summary_data["metrics"]["Hybrid"] = {
            "build_time": t_ml_train, 
            "build_time_total": t_hist_build + t_ml_train,
            "infer_time": t_hyb_inf,
            "median_q_error": m_hyb['QErr_median'],
            "p25_q_error": m_hyb['QErr_p25'],
            "p75_q_error": m_hyb['QErr_p75'],
            "avg_q_error": m_hyb['QErr_avg'],
            "p95_q_error": m_hyb['QErr_p95']
        }

        summary_data["metrics"]["EquiHist"] = {
            "build_time": t_eh_init_total,
            "infer_time": t_eh_inf_p1,
            "median_q_error": m_eh_init['QErr_median'],
            "p25_q_error": m_eh_init['QErr_p25'],
            "p75_q_error": m_eh_init['QErr_p75'],
            "avg_q_error": m_eh_init['QErr_avg'],
            "p95_q_error": m_eh_init['QErr_p95']
        }
        
        with open(result_dir / "summary.json", "w") as f:
            json.dump(summary_data, f, indent=4)
        print(f"Summary stats saved to {result_dir / 'summary.json'}")
        
        # Generate Plot
        plot_q_error_boxplots(result_file, result_dir)
        return
         
        return


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
    aggregate_summaries(str(Path(args.out_dir).parent))


if __name__ == "__main__":
    main()
