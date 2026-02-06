
import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import List, Dict, Any
from models.core import Bucket, RangeQuery
from datasets import gen_values, save_csv_column, scan_min_max_count, build_frequency_and_sample, freedman_diaconis_bins
from models.equi_width import EquiWidthHistogram
from models.equi_hist import EquiHistLearner
from models.hybrid import HybridEstimator
from models.evaluation import identify_bad_buckets, summarize
import copy

def run_drift_scenario(
    name: str, 
    base_dist: str, 
    drift_dist: str, 
    shift: int, 
    rows: int, 
    drift_rows: int, 
    eval_n: int
) -> Dict[str, Any]:
    print(f"\n>>> Running Scenario: {name} ({base_dist} -> {drift_dist})")
    rng = np.random.default_rng(42)
    temp_dir = Path("temp_drift")
    temp_dir.mkdir(exist_ok=True)
    ds_path = temp_dir / f"drift_{name}.csv"
    
    # 1. Generate Initial Data (Source)
    print(f"Generating {rows} rows of {base_dist}...")
    vals = gen_values(rng, base_dist, rows, 0, 200_000, shift=0)
    save_csv_column(vals, ds_path, mode='w')
    
    # 2. Initial Build
    mn, mx, N = scan_min_max_count(ds_path)
    freq, sample = build_frequency_and_sample(ds_path, mn, mx, N, 100_000, 42)
    n_bins = freedman_diaconis_bins(sample, mn, mx, N, 2000)
    
    
    ew_hist = EquiWidthHistogram.build(mn, mx, n_bins, freq)
    hybrid_est = HybridEstimator(copy.deepcopy(ew_hist.buckets))
    hybrid_est.train(freq, mn, 20, rng)
    
    eh_learner = EquiHistLearner(copy.deepcopy(ew_hist.buckets), learning_rate=0.5)
    
    # 3. Simulate Drift (Target Append)
    print(f"Simulating Drift: Appending {drift_rows} rows of {drift_dist} (shift={shift})...")
    drift_vals = gen_values(rng, drift_dist, drift_rows, 0, 200_000, shift=shift)
    save_csv_column(drift_vals, ds_path, mode='a')
    
    # Update Ground Truth for Drift Phase
    mn_new, mx_new, N_new = scan_min_max_count(ds_path)
    freq_new, _ = build_frequency_and_sample(ds_path, mn_new, mx_new, N_new, 1000, 42)
    ps_new = np.cumsum(freq_new)
    
    # 4. Evaluation Workload
    queries = []
    width = mx_new - mn_new
    for _ in range(eval_n):
        l = rng.integers(mn_new, mx_new)
        w = rng.integers(1, max(10, width // 20))
        r = min(mx_new, l + w)
        queries.append(RangeQuery(l, r))
        
    y_true = []
    y_stale = []
    y_eh = []
    y_hyb_stale = []
    
    print(f"Evaluating {eval_n} queries under drift...")
    for q in queries:
        # Ground Truth
        li, ri = q.low - mn_new, q.high - mn_new
        li, ri = max(0, li), min(len(ps_new)-1, ri)
        truth = int(ps_new[ri] - (ps_new[li-1] if li > 0 else 0))
        y_true.append(truth / N_new)
        
        # Stale Equi-Width
        y_stale.append(ew_hist.predict(q) / N) # N is old count
        
        # Stale Hybrid
        y_hyb_stale.append(hybrid_est.predict(q) / N)
        
        # EquiHist (Adaptive)
        y_eh.append(eh_learner.predict(q) / N)
        eh_learner.update(q, float(truth))
        
    # 5. Hybrid Repair
    print("Performing Hybrid Repair...")
    # Update counts in buckets
    for b in hybrid_est.buckets:
        li, ri = b.lo - mn_new, b.hi - mn_new
        li, ri = max(0, li), min(len(ps_new)-1, ri)
        b.count = int(ps_new[ri] - (ps_new[li-1] if li > 0 else 0))
    
    # Identify and retrain bad buckets
    y_hyb_counts_only = [hybrid_est.predict(q) / N_new for q in queries]
    bad_indices = identify_bad_buckets(queries, np.array(y_true), np.array(y_hyb_counts_only), hybrid_est.buckets, threshold_q=1.10)
    
    t0 = time.perf_counter()
    if bad_indices:
        hybrid_est.train(freq_new, mn_new, 50, rng, bucket_indices=bad_indices)
    t_repair = time.perf_counter() - t0
    
    # Post-Repair Hybrid Eval
    y_hyb_repaired = [hybrid_est.predict(q) / N_new for q in queries]
    
    # 6. Final Summaries
    m_stale = summarize(np.array(y_true), np.array(y_stale), "Stale")
    m_eh = summarize(np.array(y_true), np.array(y_eh), "EquiHist")
    m_hyb_repaired = summarize(np.array(y_true), np.array(y_hyb_repaired), "HybridRepaired")
    
    # Clean up
    if ds_path.exists(): ds_path.unlink()
    
    return {
        "Scenario": name,
        "Stale_Median_QErr": m_stale['QErr_median'],
        "EquiHist_Median_QErr": m_eh['QErr_median'],
        "Hybrid_Repaired_Median_QErr": m_hyb_repaired['QErr_median'],
        "Stale_MAE": m_stale['MAE'],
        "EquiHist_MAE": m_eh['MAE'],
        "Hybrid_Repaired_MAE": m_hyb_repaired['MAE'],
        "Repair_Time_sec": t_repair
    }

def main():
    scenarios = [
        ("Normal_to_Zipf", "normal", "zipf", 0),
        ("Normal_to_Uniform", "normal", "uniform", 0),
        ("Normal_to_AntiZipf", "normal", "anti_zipf", 0),
        ("Normal_to_Sparse", "normal", "sparse_cluster", 0)
    ]
    
    # Params
    rows = 60_000_000
    drift_rows = 10_000_000
    eval_n = 10_000
    
    results = []
    for name, base, drift, shift in scenarios:
        res = run_drift_scenario(name, base, drift, shift, rows, drift_rows, eval_n)
        results.append(res)
        
    df = pd.DataFrame(results)
    print("\nFinal Drift Benchmark Results:")
    print(df)
    df.to_excel("drift_benchmark_results.xlsx", index=False)
    print("\nResults saved to drift_benchmark_results.xlsx")

if __name__ == "__main__":
    main()
