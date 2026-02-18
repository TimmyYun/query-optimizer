import argparse
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
from workload import load_workload_csv

def get_common_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dist", type=str, default="all", help="Data distribution (e.g., uniform, zipf, or 'all')")
    parser.add_argument("--rows", type=int, default=60000000, help="Number of rows in the dataset")
    parser.add_argument("--eval-n", type=str, default="1000000", help="Workload name/size to evaluate")
    parser.add_argument("--out-dir", type=str, default="results", help="Base directory for results")
    parser.add_argument("--experiment-name", type=str, default=None, help="Experiment ID or name")
    parser.add_argument("--buckets", type=int, default=None, help="Force a specific number of buckets (overrides FD binning)")
    return parser

def run_benchmark_suite(args, approach_fn):
    """
    Handles the loop over distributions and result aggregation.
    """
    import copy
    
    # Ensure a single experiment name for all distributions in this suite
    if args.experiment_name is None:
        base_dir = Path(args.out_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        existing_ids = [int(d.name) for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        next_id = max(existing_ids) + 1 if existing_ids else 1
        args.experiment_name = str(next_id)
        print(f"Assigning Experiment ID: {args.experiment_name}")

    if args.dist == "all":
        distributions = ["uniform", "normal", "zipf", "sparse_cluster", "anti_zipf"]
    else:
        distributions = [args.dist]
        
    for dist in distributions:
        print(f"\n>>> Running for Distribution: {dist} <<<")
        current_args = copy.deepcopy(args)
        current_args.dist = dist
        
        # Setup output directory for this distribution
        out_dir = setup_out_dir(current_args, approach_fn.__name__)
        
        # Load Data
        try:
            metadata = load_data_and_metadata(current_args.rows, dist)
            approach_fn(current_args, out_dir, metadata)
        except Exception as e:
            print(f"Failed to run for {dist}: {e}")
            import traceback
            traceback.print_exc()
            
    # Auto-aggregate results if multiple distributions were run
    if args.dist == "all" or len(distributions) > 1:
        print("\n>>> Aggregating All Results <<<")
        # Go up two levels to find the experiment root: results/{exp_id}/{rows}/{dist}
        # Actually setup_out_dir already set args.experiment_name
        aggregate_summaries(args.out_dir)

def setup_out_dir(args, approach_name):
    base_dir = Path(args.out_dir)
    if args.experiment_name is None:
        base_dir.mkdir(parents=True, exist_ok=True)
        existing_ids = [int(d.name) for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        next_id = max(existing_ids) + 1 if existing_ids else 1
        args.experiment_name = str(next_id)
    
    out_dir = base_dir / args.experiment_name / str(args.rows) / args.dist
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def load_data_and_metadata(rows, dist):
    from datasets import DatasetManager
    dm = DatasetManager()
    ds_dir = dm.ensure_dataset_exists(rows, dist)
    
    with open(ds_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    # mn, mx, N, freq, sample, n_bins, skew, kurt
    return meta

def load_and_filter_workload(wl_name, mn, mx, freq):
    workload_path = Path(f"workload/{wl_name}/workload.csv")
    if not workload_path.exists():
        workload_path = Path(f"workload/{wl_name}.csv")
        if not workload_path.exists():
            raise FileNotFoundError(f"Workload {wl_name} not found.")
            
    all_queries = load_workload_csv(workload_path)
    ps = np.cumsum(freq)
    N = ps[-1]
    
    queries = []
    y_true = []
    
    for q in all_queries:
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
        
        if truth > 0:
            queries.append(q)
            y_true.append(truth / N)
            
    return queries, np.array(y_true)

def save_benchmark_results(out_dir, wl_name, model_name, queries, y_true, y_pred, metrics):
    from models import q_error_vec
    
    # Save CSV
    results_data = []
    errors = q_error_vec(y_true, y_pred)
    for i, (pred, true, err) in enumerate(zip(y_pred, y_true, errors)):
        results_data.append({
            "Query_ID": i,
            "Model": model_name,
            "Prediction": float(pred),
            "Truth": float(true),
            "Q_Error": float(err)
        })
    
    df_results = pd.DataFrame(results_data)
    result_file = out_dir / f"{wl_name}_{model_name}.csv"
    df_results.to_csv(result_file, index=False)
    
    # Save Summary JSON
    summary_path = out_dir / f"summary_{model_name}.json"
    summary_data = {
        "model": model_name,
        "wl_name": wl_name,
        "metrics": metrics
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=4)
        
    print(f"Results saved to {result_file} and {summary_path}")

def aggregate_summaries(results_dir="results"):
    root = Path(results_dir)
    if not root.exists():
        print(f"Results directory '{results_dir}' does not exist.")
        return

    for experiment_dir in root.iterdir():
        if not experiment_dir.is_dir() or experiment_dir.name.startswith("."):
            continue

        experiment_id = experiment_dir.name
        print(f"Processing Experiment {experiment_id}...")
        all_data = []

        for row_dir in experiment_dir.iterdir():
            if not row_dir.is_dir() or not row_dir.name.isdigit():
                continue
            rows = int(row_dir.name)
            for dist_dir in row_dir.iterdir():
                if not dist_dir.is_dir(): continue
                dist_name = dist_dir.name
                
                # Look for all summary_*.json files
                for summary_json_path in dist_dir.glob("summary_*.json"):
                    try:
                        with open(summary_json_path, "r") as f:
                            summary = json.load(f)
                            model = summary["model"]
                            wl_name = summary["wl_name"]
                            metrics = summary["metrics"]
                            
                            all_data.append({
                                "Distribution": dist_name,
                                "Rows": rows,
                                "Workload": wl_name,
                                "Model": model,
                                "Avg Q-Error": metrics.get("avg_q_error"),
                                "25% Q-Error": metrics.get("p25_q_error"),
                                "75% Q-Error": metrics.get("p75_q_error"),
                                "95% Q-Error": metrics.get("p95_q_error"),
                                "Median Q-Error": metrics.get("median_q_error"),
                                "Training Time (s)": metrics.get("train_time") or metrics.get("build_time") or metrics.get("total_train_time"),
                                "Inference Time (s)": metrics.get("infer_time")
                            })
                    except Exception as e:
                        print(f"  Warning: Could not read {summary_json_path}: {e}")

        if not all_data: continue
        df_summary = pd.DataFrame(all_data).sort_values(by=["Rows", "Distribution", "Workload", "Model"])
        output_path = experiment_dir / "summary.csv"
        df_summary.to_csv(output_path, index=False)
        print(f"  Saved aggregated summary to {output_path}")
