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
    python main.py --dist zipf --rows 1000000 --drift-dist normal
"""

import argparse
import json
import time
import pickle
import shutil
import os
from pathlib import Path
import numpy as np

# New Modular Imports
from models.core import Bucket, RangeQuery
from datasets import gen_values, save_csv_column, scan_min_max_count, build_frequency_and_sample, load_imdb_lengths, load_census_age, freedman_diaconis_bins
from models.equi_width import EquiWidthHistogram
from models.equi_hist import EquiHistLearner
from models.hybrid import HybridEstimator
from models.evaluation import identify_bad_buckets, summarize

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
    parser.add_argument("--dist", type=str, default="zipf")
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--drift-rows", type=int, default=200_000)
    parser.add_argument("--drift-dist", type=str, default="normal")
    parser.add_argument("--drift-shift", type=int, default=50_000, help="Shift magnitude for drift data")
    parser.add_argument("--out-dir", type=str, default="artifacts_optimizer")
    parser.add_argument("--eval-n", type=int, default=1000)
    parser.add_argument("--recreate", action="store_true", help="Force regeneration of the dataset even if cached")
    
    # Ablation Hyperparams
    parser.add_argument("--eh-lr", type=float, default=0.5, help="EquiHist Learning Rate")
    
    args = parser.parse_args()
    
    rng = np.random.default_rng(42)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    # -----------------------------------------------------
    # Phase 1: Initial Build
    # -----------------------------------------------------
    print(f"=== Phase 1: Initial Build ({args.rows} rows, {args.dist}) ===")
    
    # Define cache paths
    cache_dir = Path("datasets/files/generated")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cached_ds_path = cache_dir / f"data_initial_{args.dist}.csv"
    meta_path = cache_dir / f"data_initial_{args.dist}.meta.pkl"
    
    # Working copy for this specific run (to allow drift appending without corrupting cache)
    working_ds_path = Path(args.out_dir) / "data_current_run.csv"
    
    # Check if we can load from cache
    load_cache = False
    if not args.recreate and cached_ds_path.exists() and meta_path.exists():
        load_cache = True
        
    if load_cache:
        print(f"Loading cached dataset and metrics from {cached_ds_path}...")
        try:
            with open(meta_path, "rb") as f:
                mn, mx, N, freq, sample, n_bins = pickle.load(f)
            # Copy cached dataset to working path
            shutil.copy2(cached_ds_path, working_ds_path)
            ds_path = working_ds_path # update ds_path to point to the working copy
        except Exception as e:
            print(f"Failed to load cache: {e}. Regenerating...")
            load_cache = False

    if not load_cache:
        print(f"Generating/Loading new dataset for {args.dist}...")
        # 1. Generate/Load Data
        if args.dist.lower() == "imdb":
            print("Loading real IMDB review lengths...")
            # Paths handled in datasets.py now, but we need to pass a valid path or dummy
            imdb_path = Path("datasets/files/imdb/IMDB Dataset.csv")
            vals = load_imdb_lengths(imdb_path)
        elif args.dist.lower() == "census":
            print("Loading real US Census Age data...")
            # Paths handled in datasets.py
            census_path = Path("datasets/files/census/USCensus1990.data.txt.csv")
            vals = load_census_age(census_path)
        else:
            vals = gen_values(rng, args.dist, args.rows, 0, 200_000)
            
        # 2. Save to Cache
        save_csv_column(vals, cached_ds_path)
        
        # 3. Compute Metrics
        # Analyze the dataset to get min, max, and total count (N)
        mn, mx, N = scan_min_max_count(cached_ds_path)
        
        # Build a frequency map and a sample for histogram construction
        freq, sample = build_frequency_and_sample(cached_ds_path, mn, mx, N, 100_000, 42)
        n_bins = freedman_diaconis_bins(sample, mn, mx, N, 2000)
        
        # 4. Save Metrics to Cache
        with open(meta_path, "wb") as f:
            pickle.dump((mn, mx, N, freq, sample, n_bins), f)
            
        # 5. Create Working Copy
        shutil.copy2(cached_ds_path, working_ds_path)
        ds_path = working_ds_path
        
    print(f"FD Suggested Bins: {n_bins}")
    
    # 2a. Equi-Width Histogram (Standard Baseline)
    # This represents a traditional database histogram.
    t0_hist = time.perf_counter()
    # Refactored: Use Class Builder to create the histogram
    ew_hist = EquiWidthHistogram.build(mn, mx, n_bins, freq)
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
    queries = []
    width = mx - mn
    ps = np.cumsum(freq)
    
    for _ in range(args.eval_n):
        l = rng.integers(mn, mx)
        w = rng.integers(1, max(10, width // 20)) 
        r = min(mx, l + w)
        queries.append(RangeQuery(l, r))
        
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
        li, ri = q.low - mn, q.high - mn
        if li < 0: li=0
        if ri >= len(ps): ri = len(ps)-1
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
    
    m_hist_w = summarize(y_true, y_hist_width, "Equi-Width (Standard)")
    m_hyb = summarize(y_true, y_hybrid, "Hybrid")
    m_eh_init = summarize(y_true, y_eh_init, "EquiHist (Initial Learning)")
    
    # Calculate specialized training times
    t_eh_init_total = t_hist_build + t_eh_update_p1
    
    print("\n--- Results ---")
    print(f"Equi-Width:  Median QErr={m_hist_w['QErr_median']:.4f}, MAE={m_hist_w['MAE']:.6f}, Time={t_base_inf:.4f}s")
    print(f"Hybrid(FD):  Median QErr={m_hyb['QErr_median']:.4f}, MAE={m_hyb['MAE']:.6f}, Time={t_hyb_inf:.4f}s")
    print(f"EquiHist:    Median QErr={m_eh_init['QErr_median']:.4f}, MAE={m_eh_init['MAE']:.6f}, InitTrain={t_eh_init_total:.4f}s, Inf={t_eh_inf_p1:.4f}s")
    
    summary = {
        "bins": n_bins,
        "hist_width_metrics": m_hist_w,
        "hybrid_metrics": m_hyb,
        "equihist_init_metrics": m_eh_init,
        "timings": {
            "equihist_init_train": t_eh_init_total,
            "equihist_init_inf": t_eh_inf_p1
        }
    }
    with open(Path(args.out_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
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
    # This histogram is NOT updated and represents the degradation of a static statistic
    # over time as new data arrives.
    # Refactored: Create Stale Class Instance
    buckets_static = [Bucket(b.lo, b.hi, count=b.count) for b in buckets_eq_width]
    static_hist_stale = EquiWidthHistogram(buckets_static)
    
    # 2. EquiHist (Online Learning - CONTINUES)
    # eh_learner is already active.
    
    print("Evaluating Drift Sequence...")
    y_true_seq = []
    y_static = []
    y_eh = []
    y_hybrid_stale = []
    
    t_eh_update_p2 = 0.0
    t_eh_inf_p2 = 0.0
    
    for q in queries:
        li, ri = q.low - mn_new, q.high - mn_new
        if li < 0: li=0
        if ri >= len(ps_new): ri = len(ps_new)-1
        truth = int(ps_new[ri] - (ps_new[li-1] if li > 0 else 0))
        actual_sel = truth / N_real
        y_true_seq.append(actual_sel)
        
        # Static
        # Refactored
        est_static = static_hist_stale.predict(q) 
        y_static.append(est_static / N) 
        
        # EquiHist (Predict then Update)
        t0_inf = time.perf_counter()
        est_eh = eh_learner.predict(q)
        t_eh_inf_p2 += (time.perf_counter() - t0_inf)
        
        y_eh.append(est_eh / N) 
        
        t0_up = time.perf_counter()
        eh_learner.update(q, float(truth)) 
        t_eh_update_p2 += (time.perf_counter() - t0_up)
        
        # Hybrid (Stale buckets + Old Models)
        # Refactored: HybridEstimator holds the stale state
        est_hyb = hybrid_est.predict(q)
        y_hybrid_stale.append(est_hyb / N)
        
    y_true_arr = np.array(y_true_seq)
    m_static = summarize(y_true_arr, np.array(y_static), "Static Equi-Width (Stale)")
    m_eh = summarize(y_true_arr, np.array(y_eh), "EquiHist (Online Adaptive)")
    m_hyb = summarize(y_true_arr, np.array(y_hybrid_stale), "Hybrid (Stale)")
    
    print(f"EquiHist Drift Update Time (Retrain): {t_eh_update_p2:.4f}s")
    
    # -----------------------------------------------------
    # Phase 3: Adaptive Repair (Hybrid) + EquiHist Final
    # -----------------------------------------------------
    print(f"\n=== Phase 3: Hybrid Adaptive Repair ===")
    
    # Hybrid updates counts (Cheap)
    # We first just update the counts in the buckets based on the new global frequency.
    # This is a "metadata only" update, without retraining the ML models inside.
    for b in buckets_eq_width:
        li = b.lo - mn_new
        ri = b.hi - mn_new
        if li < 0: li=0
        if ri >= len(freq_new): ri = len(freq_new)-1
        b.count = int(ps_new[ri] - (ps_new[li-1] if li > 0 else 0))
        
    # Check error again with updated counts
    y_hyb_counts_only = []
    for q in queries:
        # Refactored call
        y_hyb_counts_only.append(hybrid_est.predict(q) / N_real)
        
    bad_indices = identify_bad_buckets(queries, y_true_arr, np.array(y_hyb_counts_only), buckets_eq_width, threshold_mae=0.0001)
    print(f"Identified {len(bad_indices)}/{len(buckets_eq_width)} buckets needing repair.")
    
    t0_repair = time.perf_counter()
    if bad_indices:
        print("Retraining specific buckets...")
        # Refactored: train method handles logic
        hybrid_est.train(freq_new, mn_new, 50, rng, bucket_indices=bad_indices)
            
    t_repair = time.perf_counter() - t0_repair
    print(f"Repair Time: {t_repair:.4f}s")
    
    # Final Hybrid Eval
    y_hyb_final = []
    for q in queries:
        y_hyb_final.append(hybrid_est.predict(q) / N_real)
        
    m_hyb_final = summarize(y_true_arr, np.array(y_hyb_final), "Hybrid (Repaired)")
    
    # Final EquiHist Eval (Static Check of Learner State)
    y_eh_final = []
    t_eh_inf_final = 0.0
    for q in queries:
        t0_inf = time.perf_counter()
        est_eh = eh_learner.predict(q)
        t_eh_inf_final += (time.perf_counter() - t0_inf)
        y_eh_final.append(est_eh / N)
        # No update here, just checking final state
        
    m_eh_final = summarize(y_true_arr, np.array(y_eh_final), "EquiHist (Final/Converged)")
    
    # -----------------------------------------------------
    # Phase 4: Static Rebuild (Offline Baseline)
    # -----------------------------------------------------
    print(f"\n=== Phase 4: Static Rebuild (Full Scan) ===")
    t0_rebuild = time.perf_counter()
    mn_rb, mx_rb, N_rb = scan_min_max_count(ds_path)
    freq_rb, _ = build_frequency_and_sample(ds_path, mn_rb, mx_rb, N_rb, 100_000, 42)
    # Re-using previous bin settings or re-estimating? Let's keep bins constant for fairness or re-estimate?
    # Standard static re-build usually re-estimates perfectly.
    # Let's use the same suggested bins count but rebuilt boundaries.
    
    # Refactored: Rebuild using class
    rebuilt_hist = EquiWidthHistogram.build(mn_rb, mx_rb, n_bins, freq_rb)
    t_static_rebuild = time.perf_counter() - t0_rebuild
    print(f"Static Rebuild Time: {t_static_rebuild:.4f}s")
    
    y_static_new = []
    for q in queries:
        est = rebuilt_hist.predict(q)
        y_static_new.append(est / N_rb)
        
    m_static_new = summarize(y_true_arr, np.array(y_static_new), "Static Equi-Width (Rebuilt)")
    
    # --- Report Resources AND Save JSON ---
    import os
    disk_usage_bytes = os.path.getsize(ds_path)
    
    # Python object size approx
    memory_buckets_bytes = len(pickle.dumps(buckets_eq_width))
    # Refactored: Access models from hybrid_est
    memory_models_bytes = len(pickle.dumps(hybrid_est.models))
    memory_total_bytes = memory_buckets_bytes + memory_models_bytes
    
    print(f"\nResource Usage:")
    print(f"Disk (CSV): {disk_usage_bytes/1024/1024:.2f} MB")
    print(f"Memory (Model): {memory_total_bytes/1024:.2f} KB")

    drift_pkg = {
        "metrics": {
            "static_stale": m_static,
            "equihist_online": m_eh,
            "hybrid_stale": m_hyb,
            "hybrid_repaired": m_hyb_final,
            "equihist_final": m_eh_final,
            "static_rebuilt": m_static_new
        },
        "timings": {
            "hist_build": t_hist_build,
            "ml_train": t_ml_train,
            "static_inf": t_base_inf,
            "hybrid_inf": t_hyb_inf,
            "repair": t_repair,
            "static_rebuild": t_static_rebuild,
            "equihist_retrain": t_eh_update_p2,
            "equihist_drift_inf": t_eh_inf_p2,
            "equihist_final_inf": t_eh_inf_final
        },
        "resources": {
            "disk_bytes": disk_usage_bytes,
            "memory_buckets_bytes": memory_buckets_bytes,
            "memory_models_bytes": memory_models_bytes,
            "memory_total_bytes": memory_total_bytes
        }
    }
    with open(Path(args.out_dir) / "drift_summary.json", "w") as f:
        json.dump(drift_pkg, f, indent=2)
    

if __name__ == "__main__":
    main()
