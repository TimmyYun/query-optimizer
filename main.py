#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
import numpy as np

# New Modular Imports
from optimizer.core import Bucket, RangeQuery
from optimizer.datasets import gen_values, save_csv_column, scan_min_max_count, build_frequency_and_sample
from optimizer.histograms import make_equiwidth_buckets, freedman_diaconis_bins
from optimizer.models import collect_cdf_training_rows, train_cdf_models, predict_range_hybrid_cdf, predict_range_histogram_uniform, EquiHistLearner
from optimizer.evaluation import identify_bad_buckets, summarize

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=str, default="zipf")
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--drift-rows", type=int, default=200_000)
    parser.add_argument("--drift-dist", type=str, default="normal")
    parser.add_argument("--drift-shift", type=int, default=50_000, help="Shift magnitude for drift data")
    parser.add_argument("--out-dir", type=str, default="artifacts_optimizer")
    parser.add_argument("--eval-n", type=int, default=1000)
    
    # Ablation Hyperparams
    parser.add_argument("--ndv-threshold", type=int, default=200, help="Threshold for Exact Storage")
    parser.add_argument("--eh-lr", type=float, default=0.5, help="EquiHist Learning Rate")
    
    args = parser.parse_args()
    
    rng = np.random.default_rng(42)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    # -----------------------------------------------------
    # Phase 1: Initial Build
    # -----------------------------------------------------
    print(f"=== Phase 1: Initial Build ({args.rows} rows, {args.dist}) ===")
    
    ds_path = Path(args.out_dir) / f"data_initial.csv"
    vals = gen_values(rng, args.dist, args.rows, 0, 200_000)
    save_csv_column(vals, ds_path)
    
    mn, mx, N = scan_min_max_count(ds_path)
    freq, sample = build_frequency_and_sample(ds_path, mn, mx, N, 100_000, 42)
    
    n_bins = freedman_diaconis_bins(sample, mn, mx, N, 2000)
    print(f"FD Suggested Bins: {n_bins}")
    
    # 2a. Equi-Width (Standard Baseline)
    t0_hist = time.perf_counter()
    buckets_eq_width = make_equiwidth_buckets(mn, mx, n_bins, freq, ndv_threshold=args.ndv_threshold)
    t_hist_build = time.perf_counter() - t0_hist
    
    # 3. Model Training
    print("Training CDF models (Skipping buckets with low NDV)...")
    rows = collect_cdf_training_rows(buckets_eq_width, freq, mn, 20, rng)
    models, t_ml_train = train_cdf_models(rows)
    print(f"Hist Build (Width): {t_hist_build:.4f}s, ML Train: {t_ml_train:.4f}s")
    
    n_exact = sum(1 for b in buckets_eq_width if b.exact_values is not None)
    n_ml = sum(1 for b in buckets_eq_width if b.exact_values is None)
    print(f"Bucket Strategy: {n_exact} Exact (Sparse), {n_ml} ML (Dense)")
    
    # 4. Evaluation
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
    
    # Measure Baseline Inference
    t0_base = time.perf_counter()
    for q in queries:
        h = predict_range_histogram_uniform(q, buckets_eq_width)
        y_hist_width.append(h / N)
    t_base_inf = time.perf_counter() - t0_base

    # Measure Hybrid Inference
    t0_hyb = time.perf_counter()
    for q in queries:
        c = predict_range_hybrid_cdf(q, buckets_eq_width, models)
        y_hybrid.append(c / N)
    t_hyb_inf = time.perf_counter() - t0_hyb

    # Truth
    for q in queries:
        li, ri = q.low - mn, q.high - mn
        if li < 0: li=0
        if ri >= len(ps): ri = len(ps)-1
        truth = int(ps[ri] - (ps[li-1] if li > 0 else 0))
        y_true.append(truth / N)
        
    y_true = np.array(y_true)
    y_hist_width = np.array(y_hist_width)
    y_hybrid = np.array(y_hybrid)
    
    m_hist_w = summarize(y_true, y_hist_width, "Equi-Width (Standard)")
    m_hyb = summarize(y_true, y_hybrid, "Hybrid + NDV Smart")
    
    print("\n--- Results ---")
    print(f"Equi-Width:  Median QErr={m_hist_w['QErr_median']:.4f}, MAE={m_hist_w['MAE']:.6f}, Time={t_base_inf:.4f}s")
    print(f"Hybrid(FD):  Median QErr={m_hyb['QErr_median']:.4f}, MAE={m_hyb['MAE']:.6f}, Time={t_hyb_inf:.4f}s")
    
    summary = {
        "bins": n_bins,
        "hist_width_metrics": m_hist_w,
        "hybrid_metrics": m_hyb
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
    buckets_static = [Bucket(b.lo, b.hi, count=b.count, ndv=b.ndv, exact_values=b.exact_values) for b in buckets_eq_width]
    
    # 2. EquiHist (Online Learning)
    eh_learner = EquiHistLearner(buckets_eq_width, learning_rate=args.eh_lr)
    
    print("Evaluating Drift Sequence...")
    y_true_seq = []
    y_static = []
    y_eh = []
    y_hybrid_stale = []
    
    for q in queries:
        li, ri = q.low - mn_new, q.high - mn_new
        if li < 0: li=0
        if ri >= len(ps_new): ri = len(ps_new)-1
        truth = int(ps_new[ri] - (ps_new[li-1] if li > 0 else 0))
        actual_sel = truth / N_real
        y_true_seq.append(actual_sel)
        
        # Static
        est_static = predict_range_histogram_uniform(q, buckets_static) 
        y_static.append(est_static / N) 
        
        # EquiHist
        est_eh = eh_learner.predict(q)
        y_eh.append(est_eh / N) 
        eh_learner.update(q, float(truth)) 
        
        # Hybrid (Stale buckets + Old Models)
        est_hyb = predict_range_hybrid_cdf(q, buckets_eq_width, models)
        y_hybrid_stale.append(est_hyb / N)
        
    y_true_arr = np.array(y_true_seq)
    m_static = summarize(y_true_arr, np.array(y_static), "Static Equi-Width (Stale)")
    m_eh = summarize(y_true_arr, np.array(y_eh), "EquiHist (Online Adaptive)")
    m_hyb = summarize(y_true_arr, np.array(y_hybrid_stale), "Hybrid (Stale)")
    
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
        y_hyb_counts_only.append(predict_range_hybrid_cdf(q, buckets_eq_width, models) / N_real)
        
    bad_indices = identify_bad_buckets(queries, y_true_arr, np.array(y_hyb_counts_only), buckets_eq_width, threshold_mae=0.0001)
    print(f"Identified {len(bad_indices)}/{len(buckets_eq_width)} buckets needing repair.")
    
    t0_repair = time.perf_counter()
    if bad_indices:
        print("Retraining specific buckets...")
        rows_repair = collect_cdf_training_rows(buckets_eq_width, freq_new, mn_new, 50, rng, bucket_indices=bad_indices)
        models_repair, _ = train_cdf_models(rows_repair)
        for k, v in models_repair.items():
            models[k] = v
            
    t_repair = time.perf_counter() - t0_repair
    print(f"Repair Time: {t_repair:.4f}s")
    
    # Final Hybrid Eval
    y_hyb_final = []
    for q in queries:
        y_hyb_final.append(predict_range_hybrid_cdf(q, buckets_eq_width, models) / N_real)
        
    m_hyb_final = summarize(y_true_arr, np.array(y_hyb_final), "Hybrid (Repaired)")
    
    # System Metrics (Memory & Disk)
    import pickle
    import os
    
    # 1. Disk Usage (Original Data + Drift Data)
    disk_usage_bytes = os.path.getsize(ds_path)
    
    # 2. Memory Usage (Buckets + Models)
    # Estimate utilizing pickle size
    mem_buckets_bytes = len(pickle.dumps(buckets_eq_width))
    mem_models_bytes = len(pickle.dumps(models))
    total_memory_bytes = mem_buckets_bytes + mem_models_bytes
    
    print(f"\nResource Usage:")
    print(f"Disk (CSV): {disk_usage_bytes / 1024 / 1024:.2f} MB")
    print(f"Memory (Model): {total_memory_bytes / 1024:.2f} KB")

    final_summary = {
        "metrics": {
            "static_stale": m_static,
            "equihist_online": m_eh,
            "hybrid_repaired": m_hyb_final
        },
        "timings": {
            "hist_build": t_hist_build,
            "ml_train": t_ml_train,
            "static_inf": t_base_inf,
            "hybrid_inf": t_hyb_inf,
            "repair": t_repair
        },
        "resources": {
            "disk_bytes": disk_usage_bytes,
            "memory_bytes": total_memory_bytes
        }
    }
    
    with open(Path(args.out_dir) / "drift_summary.json", "w") as f:
        json.dump(final_summary, f, indent=2)

if __name__ == "__main__":
    main()
