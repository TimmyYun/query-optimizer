import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
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
    aggregate_summaries(args.out_dir)


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


def load_and_filter_workload(workload_path, mn, mx, freq):
    workload_path = Path(workload_path)
    if workload_path.is_dir():
        workload_path = workload_path / "workload.csv"

    if not workload_path.exists():
        raise FileNotFoundError(f"Workload not found at {workload_path}.")

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
            if li < 0:
                li = 0
            if ri >= len(ps):
                ri = len(ps) - 1
            if li > ri:
                truth = 0
            else:
                truth = int(ps[ri] - (ps[li - 1] if li > 0 else 0))

        if truth > 0:
            queries.append(q)
            y_true.append(truth / N)

    return queries, np.array(y_true)


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

        for ds_parent_dir in experiment_dir.iterdir():
            if not ds_parent_dir.is_dir():
                continue

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
        df_summary.to_csv(output_path, index=False)
        print(f"  Saved aggregated summary to {output_path}")
