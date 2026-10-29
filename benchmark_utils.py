import argparse
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
from models import summarize
from workload import load_workload_csv


def get_common_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset directory containing meta.pkl",
    )
    parser.add_argument(
        "--workload",
        type=str,
        required=True,
        help="Path to the workload CSV file or directory",
    )
    parser.add_argument(
        "--out-dir", type=str, default="results", help="Base directory for results"
    )
    parser.add_argument(
        "--experiment-name", type=str, default=None, help="Experiment ID or name"
    )
    parser.add_argument(
        "--buckets",
        type=int,
        default=None,
        help="Force a specific number of buckets (overrides FD binning)",
    )
    return parser


def run_benchmark_suite(args, approach_fn):
    """
    Handles the loop over distributions and result aggregation.
    """

    # Ensure a single experiment name for all distributions in this suite
    if args.experiment_name is None:
        base_dir = Path(args.out_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        existing_ids = [
            int(d.name) for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()
        ]
        next_id = max(existing_ids) + 1 if existing_ids else 1
        args.experiment_name = str(next_id)
        print(f"Assigning Experiment ID: {args.experiment_name}")

    dataset_path = Path(args.dataset)
    workload_path = Path(args.workload)

    # Determine if dataset_path is a single dataset or a collection of distributions
    if (dataset_path / "meta.pkl").exists():
        dsets = [dataset_path]
    else:
        dsets = sorted(
            [
                d
                for d in dataset_path.iterdir()
                if d.is_dir() and (d / "meta.pkl").exists()
            ]
        )

    if not dsets:
        print(f"No valid datasets (with meta.pkl) found in {dataset_path}")
        return

    for dset in dsets:
        if dataset_path != dset:
            ds_out_name = f"{dataset_path.name}/{dset.name}"
        else:
            ds_out_name = dset.name

        print(
            f"\n>>> Running for Dataset: {ds_out_name}, Workload: {workload_path.name} <<<"
        )

        # Setup output directory
        out_dir = setup_out_dir(args, ds_out_name)

        # Load Data
        try:
            metadata = load_data_and_metadata(dset)
            approach_fn(args, out_dir, metadata)
        except Exception as e:
            print(f"Failed to run for dataset {dset}: {e}")
            import traceback

            traceback.print_exc()

    # Aggregate summaries at the end
    aggregate_summaries(args.out_dir, only=args.experiment_name)


def setup_out_dir(args, dataset_name):
    base_dir = Path(args.out_dir)
    if args.experiment_name is None:
        base_dir.mkdir(parents=True, exist_ok=True)
        existing_ids = [
            int(d.name) for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()
        ]
        next_id = max(existing_ids) + 1 if existing_ids else 1
        args.experiment_name = str(next_id)

    out_dir = base_dir / args.experiment_name / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_data_and_metadata(dataset_path):
    ds_dir = Path(dataset_path)
    if not ds_dir.exists() or not (ds_dir / "meta.pkl").exists():
        raise FileNotFoundError(f"Dataset meta.pkl not found at {ds_dir}")

    with open(ds_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    # mn, mx, N, freq, sample, n_bins, skew, kurt
    return meta


def exact_counts(queries, freq, mn):
    """
    Exact row count of every inclusive range [q.low, q.high].
    freq is the dense per-value count over [mn, mn + len(freq) - 1]; ranges
    outside it (fully or partly) contribute 0 rows. This is the only place truth is computed.
    """
    ps = np.concatenate([[0], np.cumsum(freq)])
    lo = np.clip(np.fromiter((q.low for q in queries), np.int64, len(queries)) - mn, 0, len(freq))
    hi = np.clip(np.fromiter((q.high for q in queries), np.int64, len(queries)) - mn + 1, 0, len(freq))
    return np.maximum(ps[hi] - ps[lo], 0).astype(np.float64)


def split_adapt_eval(n_queries, seed=42):
    """
    Drift protocol: a step's queries are split once, deterministically, into two halves.
    adapt_idx — the only queries whose feedback a model may learn from (Hybrid FT/RB,
                STHist, Equi-Hist);
    eval_idx  — the queries every stage of every model is scored on.
    A model is therefore never scored on a query it adapted to. With a static workload
    the split is the same at every step, so an eval query is never learned at any step.
    """
    perm = np.random.default_rng(seed).permutation(n_queries)
    return np.sort(perm[: n_queries // 2]), np.sort(perm[n_queries // 2 :])


def timing_stats(infer_times):
    """Per-query inference time statistics; total is the sum of predict() calls only."""
    infer_times = np.asarray(infer_times)
    return {
        "infer_time": float(np.sum(infer_times)),
        "median_infer_time": float(np.median(infer_times)),
        "avg_infer_time": float(np.mean(infer_times)),
        "p95_infer_time": float(np.percentile(infer_times, 95)),
    }


def load_and_filter_workload(workload_path, mn, mx, freq):
    workload_path = Path(workload_path)
    if workload_path.is_dir():
        workload_path = workload_path / "workload.csv"

    if not workload_path.exists():
        raise FileNotFoundError(f"Workload not found at {workload_path}.")

    all_queries = load_workload_csv(workload_path)
    N = np.sum(freq)
    truth = exact_counts(all_queries, freq, mn)

    # Zero-truth queries are dropped: q-error is undefined for them
    keep = truth > 0
    queries = [q for q, k in zip(all_queries, keep) if k]
    return queries, truth[keep] / N


def resolve_workload_path(args):
    """--workload is a CSV path, or a count N resolved to <dataset>/workload_N.csv."""
    workload_path = Path(args.workload)
    if workload_path.exists():
        return workload_path
    potential_path = Path(args.dataset) / f"workload_{args.workload}.csv"
    if potential_path.exists():
        return potential_path
    raise FileNotFoundError(f"Workload not found at {args.workload} or {potential_path}")


def run_queries(predict, queries, true_counts, feedback=None):
    """
    The one evaluation loop every method goes through, so timings are comparable.

    predict(q) -> estimated row count. Only this call is timed as inference.
    feedback(q, true_count, pred_count), if given, is called after each predict
    (prequential: the model is scored before it learns from the query); its time
    is accumulated separately as update time.

    Returns (raw pred_counts, per-query inference times, total update time).
    feedback receives the estimate clipped to >= 1, the same value that is scored.
    """
    pred_counts = np.empty(len(queries), dtype=np.float64)
    infer_times = np.empty(len(queries), dtype=np.float64)
    update_time = 0.0

    for i, (q, truth) in enumerate(zip(queries, true_counts)):
        st = time.perf_counter()
        pred = predict(q)
        infer_times[i] = time.perf_counter() - st
        pred_counts[i] = pred

        if feedback is not None:
            st = time.perf_counter()
            feedback(q, float(truth), max(float(pred), 1.0))
            update_time += time.perf_counter() - st

    return pred_counts, infer_times, update_time


def evaluate_and_save(
    out_dir,
    workload_path,
    model_name,
    predict,
    queries,
    y_true_sel,
    N,
    build_time,
    feedback=None,
    extra_metrics=None,
):
    """
    Static run: evaluate `predict` on the workload, compute metrics, save results.

    y_true_sel comes from load_and_filter_workload (exact, zero-truth dropped).
    Both truth and estimate are clipped to >= 1 row before q-error.
    `extra_metrics` is a dict or a callable(true_counts, raw_pred_counts) -> dict for
    model-specific fields (bucket counts, memory, paper metrics).
    Returns (metrics, y_true_safe, y_pred_safe).
    """
    true_counts = np.maximum(np.rint(y_true_sel * N), 1.0)
    pred_counts, infer_times, update_time = run_queries(
        predict, queries, true_counts, feedback
    )

    y_true_safe = true_counts / N
    y_pred_safe = np.maximum(pred_counts, 1.0) / N
    m = summarize(y_true_safe, y_pred_safe, model_name)

    metrics = {
        "n_queries": len(queries),
        "build_time": build_time,
        "update_time_total": update_time,
        "train_time": build_time + update_time,
        **timing_stats(infer_times),
        "median_q_error": m["QErr_median"],
        "p25_q_error": m["QErr_p25"],
        "p75_q_error": m["QErr_p75"],
        "p95_q_error": m["QErr_p95"],
        "avg_q_error": m["QErr_avg"],
    }
    if callable(extra_metrics):
        extra_metrics = extra_metrics(true_counts, pred_counts)
    metrics.update(extra_metrics or {})

    print(
        f"[{model_name}] Train={metrics['train_time']:.4f}s, "
        f"Infer={metrics['infer_time']:.4f}s ({metrics['avg_infer_time'] * 1e6:.2f} us/query)"
    )

    save_benchmark_results(
        out_dir, Path(workload_path).stem, model_name, queries, y_true_safe, y_pred_safe, metrics
    )
    return metrics, y_true_safe, y_pred_safe


def save_benchmark_results(
    out_dir, wl_name, model_name, queries, y_true, y_pred, metrics
):
    from models import q_error_vec

    # Save CSV
    results_data = []
    errors = q_error_vec(y_true, y_pred)
    for i, (pred, true, err) in enumerate(zip(y_pred, y_true, errors)):
        results_data.append(
            {
                "Query_ID": i,
                "Model": model_name,
                "Prediction": float(pred),
                "Truth": float(true),
                "Q_Error": float(err),
            }
        )

    df_results = pd.DataFrame(results_data)
    result_file = out_dir / f"{wl_name}_{model_name}.csv"
    df_results.to_csv(result_file, index=False)

    # Save Summary JSON
    summary_path = out_dir / f"summary_{model_name}.json"
    summary_data = {"model": model_name, "wl_name": wl_name, "metrics": metrics}
    
    import re
    def repl_scientific(m):
        val = float(m.group(0))
        return f"{val:.9f}"
        
    summary_json_str = json.dumps(summary_data, indent=4)
    summary_json_str = re.sub(r'[-+]?\d+(?:\.\d+)?[eE][-+]?\d+', repl_scientific, summary_json_str)
    
    with open(summary_path, "w") as f:
        f.write(summary_json_str)

    print(f"Results saved to {result_file} and {summary_path}")


def aggregate_summaries(results_dir="results", only=None):
    """Roll summary_*.json up into results/<exp>/summary.csv; `only` limits it to one experiment."""
    root = Path(results_dir)
    if not root.exists():
        print(f"Results directory '{results_dir}' does not exist.")
        return

    for experiment_dir in root.iterdir():
        if not experiment_dir.is_dir() or experiment_dir.name.startswith("."):
            continue
        if only is not None and experiment_dir.name != only:
            continue

        experiment_id = experiment_dir.name
        print(f"Processing Experiment {experiment_id}...")
        all_data = []

        # Handle nested dataset dirs like "60000000_hard/uniform"
        # we can use glob to find all summary_*.json files within the experiment dir
        for summary_json_path in experiment_dir.rglob("summary_*.json"):
            try:
                with open(summary_json_path, "r") as f:
                    summary = json.load(f)
                    model = summary["model"]
                    wl_name = summary["wl_name"]
                    metrics = summary["metrics"]

                    # Extract dataset name based on directory structure: exp_dir/ds_name/wl_name/summary...
                    # Rel path from exp_dir
                    rel_path = summary_json_path.relative_to(experiment_dir)
                    # Dataset name could be multiple parts, e.g. "60000000_hard/uniform".
                    # It's everything before the last part (summary.json)
                    if len(rel_path.parts) >= 2:
                        dataset_name = "/".join(rel_path.parts[:-1])
                    else:
                        dataset_name = rel_path.parts[0]

                    all_data.append(
                        {
                            "Dataset": dataset_name,
                            "Workload": wl_name,
                            "Model": model,
                            "Avg Q-Error": metrics.get("avg_q_error"),
                            "25% Q-Error": metrics.get("p25_q_error"),
                            "75% Q-Error": metrics.get("p75_q_error"),
                            "95% Q-Error": metrics.get("p95_q_error"),
                            "Median Q-Error": metrics.get("median_q_error"),
                            "Training Time (s)": metrics.get("train_time")
                            or metrics.get("build_time")
                            or metrics.get("total_train_time"),
                            "Inference Time (s)": metrics.get("infer_time"),
                            "Median Inference Time (s)": metrics.get("median_infer_time"),
                            "Avg Inference Time (s)": metrics.get("avg_infer_time"),
                            "95% Inference Time (s)": metrics.get("p95_infer_time"),
                        }
                    )
            except Exception as e:
                print(f"  Warning: Could not read {summary_json_path}: {e}")

        if not all_data:
            continue
        df_summary = pd.DataFrame(all_data).sort_values(
            by=["Dataset", "Workload", "Model"]
        )
        output_path = experiment_dir / "summary.csv"
        df_summary.to_csv(output_path, index=False, float_format="%.9f")
        print(f"  Saved aggregated summary to {output_path}")
