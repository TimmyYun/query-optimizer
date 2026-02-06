
import argparse
import time
import pandas as pd
import numpy as np
from pathlib import Path
from models.core import RangeQuery, Bucket
from models.equi_width import EquiWidthHistogram
from models.equi_hist import EquiHistLearner
from models.hybrid import HybridEstimator
from datasets import gen_values, save_csv_column, scan_min_max_count, build_frequency_and_sample, freedman_diaconis_bins
from models.evaluation import summarize, q_error_vec

def run_benchmark():
    distributions = ['uniform', 'normal', 'zipf', 'sparse_cluster', 'anti_zipf']
    rows = 60_000_000
    eval_n = 100_000
    seed = 42
    rnd = np.random.default_rng(seed)
    
    results = []
    raw_errors = [] # List to store dicts: {'Distribution': dist, 'Model': model, 'QErr': val}
    
    # Ensure directories exist
    cache_dir = Path("datasets/files/generated")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    for dist in distributions:
        print(f"\n--- Processing Distribution: {dist} ---")
        
        # 1. Data Generation/Loading
        # We will reuse the caching logic mechanism conceptually or just overwrite for this benchmark
        # To be safe and clean, let's just generate fresh or use existing if present? 
        # For a benchmark sheet, consistency is key.
        
        ds_path = cache_dir / f"bench_{dist}.csv"
        
        # Generate data
        print(f"Generating data for {dist}...")
        vals = gen_values(rnd, dist, rows, 0, 200_000)
        save_csv_column(vals, ds_path)
        
        mn, mx, N = scan_min_max_count(ds_path)
        freq, sample = build_frequency_and_sample(ds_path, mn, mx, N, 100_000, seed)
        n_bins = freedman_diaconis_bins(sample, mn, mx, N, 2000)
        print(f"Stats: N={N}, Min={mn}, Max={mx}, Bins={n_bins}")
        
        # -----------------------------------
        # Model 1: Equi-Width
        # -----------------------------------
        t0 = time.perf_counter()
        ew = EquiWidthHistogram.build(mn, mx, n_bins, freq)
        t_build_ew = time.perf_counter() - t0
        
        # -----------------------------------
        # Model 2: Hybrid
        # -----------------------------------
        t0 = time.perf_counter()
        hybrid = HybridEstimator(ew.buckets) # Uses same buckets structure initially
        t_train_hybrid = hybrid.train(freq, mn, 20, rnd)
        t_build_hybrid = t_build_ew + t_train_hybrid # Total time
        
        # -----------------------------------
        # Model 3: EquiHist (Online)
        # -----------------------------------
        # EquiHist starts with EquiWidth buckets and learns. 
        # For "Static" benchmark, we interpret this as performance over the workload sequence
        # starting from the initial state.
        eh = EquiHistLearner(ew.buckets, learning_rate=0.5)
        
        # -----------------------------------
        # Evaluation Workload
        # -----------------------------------
        queries = []
        ps = np.cumsum(freq)
        width = mx - mn
        
        for _ in range(eval_n):
            l = rnd.integers(mn, mx)
            w = rnd.integers(1, max(10, width // 20))
            r = min(mx, l + w)
            queries.append(RangeQuery(l, r))
            
        y_true = []
        y_ew = []
        y_hyb = []
        y_eh = []
        
        t_inf_ew = 0
        t_inf_hyb = 0
        t_inf_eh = 0
        
        for q in queries:
            li, ri = q.low - mn, q.high - mn
            if li < 0: li=0
            if ri >= len(ps): ri = len(ps)-1
            truth = int(ps[ri] - (ps[li-1] if li > 0 else 0))
            true_sel = truth / N
            y_true.append(true_sel)
            
            # Equi-Width
            t0 = time.perf_counter()
            est_ew = ew.predict(q)
            t_inf_ew += (time.perf_counter() - t0)
            y_ew.append(est_ew / N)
            
            # Hybrid
            t0 = time.perf_counter()
            est_hyb = hybrid.predict(q)
            t_inf_hyb += (time.perf_counter() - t0)
            y_hyb.append(est_hyb / N)
            
            # EquiHist (Predict + Update)
            t0 = time.perf_counter()
            est_eh_val = eh.predict(q)
            t_inf_eh += (time.perf_counter() - t0)
            y_eh.append(est_eh_val / N)
            
            # Update EquiHist
            eh.update(q, float(truth))
            
        # -----------------------------------
        # Metrics
        # -----------------------------------
        y_true = np.array(y_true)
        
        # Store raw errors for plotting
        q_ew = summarize(np.array(y_true), np.array(y_ew), "Equi-Width")
        for val in q_error_vec(np.array(y_true), np.array(y_ew)):
            raw_errors.append({'Distribution': dist, 'Model': 'Equi-Width', 'QErr': val})

        q_hyb = summarize(np.array(y_true), np.array(y_hyb), "Hybrid")
        for val in q_error_vec(np.array(y_true), np.array(y_hyb)):
            raw_errors.append({'Distribution': dist, 'Model': 'Hybrid', 'QErr': val})
            
        q_eh = summarize(np.array(y_true), np.array(y_eh), "EquiHist")
        for val in q_error_vec(np.array(y_true), np.array(y_eh)):
            raw_errors.append({'Distribution': dist, 'Model': 'EquiHist', 'QErr': val})
        
        def get_metrics(y_pred, name, build_time, inf_time):
            m = summarize(y_true, np.array(y_pred), name)
            return {
                "Distribution": dist,
                "Model": name,
                "Build Time (s)": build_time,
                "Inference Time (s)": inf_time,
                "MAE": m['MAE'],
                "Median QErr": m['QErr_median'],
                "95% QErr": m['QErr_p95'],
                "Max QErr": m.get('QErr_max', 0)
            }
            
        results.append(get_metrics(y_ew, "Equi-Width", t_build_ew, t_inf_ew))
        results.append(get_metrics(y_hyb, "Hybrid", t_build_hybrid, t_inf_hyb))
        results.append(get_metrics(y_eh, "EquiHist", t_build_ew, t_inf_eh)) # Eh build is basically EW build initially
        
    df = pd.DataFrame(results)
    print("\nBenchmark Results:")
    print(df)
    
    out_file = "static_benchmark_results.xlsx"
    df.to_excel(out_file, index=False)
    print(f"\nResults saved to {out_file}")
    
    # Save raw errors
    print("Saving raw errors to static_errors.parquet...")
    df_raw = pd.DataFrame(raw_errors)
    df_raw.to_parquet("static_errors.parquet", index=False)
    print("Raw errors saved to static_errors.parquet")

if __name__ == "__main__":
    run_benchmark()
