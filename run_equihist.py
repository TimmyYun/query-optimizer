#!/usr/bin/env python3
import time
import numpy as np
from models import EquiWidthHistogram, EquiHistLearner, summarize
from benchmark_utils import (
    get_common_parser, run_benchmark_suite,
    load_and_filter_workload, save_benchmark_results
)

def run_logic(args, out_dir, metadata):
    mn, mx, N, freq, sample, n_bins, skew, kurt = metadata
    
    # Build Initial Histogram (Equi-Width)
    t0 = time.perf_counter()
    ew_hist = EquiWidthHistogram.build(mn, mx, n_bins, freq)
    initial_build_time = time.perf_counter() - t0
    
    eh_learner = EquiHistLearner(ew_hist.buckets, learning_rate=args.lr)
    
    # Load Workload
    queries, y_true = load_and_filter_workload(args.eval_n, mn, mx, freq)
    true_cardinalities = y_true * N
    
    # Evaluate with Mini-Batch Updates
    y_pred = []
    inf_time_total = 0.0
    update_time_total = 0.0
    
    for i in range(0, len(queries), args.batch_size):
        q_batch = queries[i : i + args.batch_size]
        t_batch = true_cardinalities[i : i + args.batch_size]
        
        t0 = time.perf_counter()
        batch_preds = eh_learner.predict_batch(q_batch)
        inf_time_total += (time.perf_counter() - t0)
        
        y_pred.extend(batch_preds / N)
        
        t0 = time.perf_counter()
        for q, truth, pred_val in zip(q_batch, t_batch, batch_preds):
            eh_learner.update(q, float(truth), pred=float(pred_val))
        update_time_total += (time.perf_counter() - t0)
        
    y_pred = np.array(y_pred)
    
    # Summarize
    m = summarize(y_true, y_pred, "EquiHist")
    metrics = {
        "initial_build_time": initial_build_time,
        "update_time_total": update_time_total,
        "total_train_time": initial_build_time + update_time_total,
        "infer_time": inf_time_total,
        "median_q_error": m['QErr_median'],
        "p25_q_error": m['QErr_p25'],
        "p75_q_error": m['QErr_p75'],
        "p95_q_error": m['QErr_p95'],
        "avg_q_error": m['QErr_avg']
    }
    
    print(f"EquiHist: Median QErr={metrics['median_q_error']:.4f}, Train={metrics['total_train_time']:.4f}s, Inf={inf_time_total:.4f}s")
    
    save_benchmark_results(out_dir, args.eval_n, "EquiHist", queries, y_true, y_pred, metrics)

def main():
    parser = get_common_parser("Run EquiHist Benchmark")
    parser.add_argument("--lr", type=float, default=0.5, help="Learning Rate")
    parser.add_argument("--batch-size", type=int, default=100000, help="Mini-batch size for updates")
    args = parser.parse_args()
    run_benchmark_suite(args, run_logic)

if __name__ == "__main__":
    main()
