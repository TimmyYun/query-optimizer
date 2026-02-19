import numpy as np
import csv
from pathlib import Path
from typing import List
import argparse
import pandas as pd
import matplotlib.pyplot as plt

"""
Workload Module
===============

This module handles the generation, storage, and retrieval of workload queries used for benchmarking.
A workload consists of a set of Range Queries.

Key Responsibilities:
1. Query Representation: Defines the `RangeQuery` class.
2. Workload Generation: Generates random range queries based on dataset domain and size.
3. Persistence: Saves and loads workloads to/from CSV files to ensure consistent evaluation across runs.
4. Visualization: Plots workload distribution against dataset buckets.
"""

class RangeQuery:
    """
    Represents a selection range query [low, high] (inclusive).
    """
    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high

    def to_dict(self):
        """Returns the query as a dictionary."""
        return {"low": int(self.low), "high": int(self.high)}

    @staticmethod
    def from_dict(d):
        """Creates a RangeQuery instance from a dictionary."""
        return RangeQuery(int(d["low"]), int(d["high"]))

def generate_workload(n: int, mn: int, mx: int, seed: int = 42, max_width: int = None) -> List[RangeQuery]:
    """
    Generates a list of random range queries.

    Args:
        n (int): Number of queries to generate.
        mn (int): Minimum value of the domain.
        mx (int): Maximum value of the domain.
        seed (int): Random seed for reproducibility.
        max_width (int): Maximum width of range query. If None, defaults to 5% of domain.

    Returns:
        List[RangeQuery]: A list of generated RangeQuery objects.
    """
    rng = np.random.default_rng(seed)
    queries = []
    width = mx - mn
    
    # Determined max query width
    if max_width is None:
        limit_w = max(10, width // 20) # Default ~5%
    else:
        limit_w = max(1, max_width)

    for _ in range(n):
        l = rng.integers(mn, mx)
        w = rng.integers(1, limit_w) 
        r = min(mx, l + w)
        queries.append(RangeQuery(l, r))
    return queries


def plot_workload_distribution(queries: list, bucket_csv_path: Path, output_path: Path, title: str = "Workload Distribution"):
    """
    Plots a histogram of query counts per dataset bucket.
    Only counts queries that are passed in (caller should filter for >0 selectivity).
    
    Args:
        queries: List of objects with .low and .high attributes (or dicts).
        bucket_csv_path: Path to histogram_buckets.csv.
        output_path: Path to save the plot.
        title: Plot title.
    """
    bucket_csv_path = Path(bucket_csv_path)
    output_path = Path(output_path)
    
    if not bucket_csv_path.exists():
        print(f"Bucket CSV not found at {bucket_csv_path}. Skipping workload plot.")
        return

    try:
        df_buckets = pd.read_csv(bucket_csv_path)
        if df_buckets.empty:
            print("Bucket CSV is empty.")
            return

        # Prepare bin edges
        # Assuming contiguous bins from sorted start
        starts = df_buckets['bin_start'].values
        ends = df_buckets['bin_end'].values
        
        # robust edges: use starts and the last end
        edges = np.concatenate([starts, [ends[-1]]])
        
        # Vectorize queries
        target_queries = queries
        if not target_queries:
             print("No queries to plot.")
             return
             
        if isinstance(target_queries[0], dict):
             ls = np.array([q['low'] for q in target_queries])
             rs = np.array([q['high'] for q in target_queries])
        else:
             ls = np.array([q.low for q in target_queries])
             rs = np.array([q.high for q in target_queries])

        # Find start and end bucket indices for each query
        idx_start = np.searchsorted(edges, ls, side='right') - 1
        idx_end = np.searchsorted(edges, rs, side='right') - 1
        
        # Clamp indices to valid buckets [0, len(buckets)-1]
        idx_start = np.clip(idx_start, 0, len(df_buckets) - 1)
        idx_end = np.clip(idx_end, 0, len(df_buckets) - 1)
        
        # Use difference array to compute counts
        # counts[i] increments if query covers bucket i.
        diff = np.zeros(len(df_buckets) + 1, dtype=int)
        np.add.at(diff, idx_start, 1)
        np.add.at(diff, idx_end + 1, -1)
        
        counts = np.cumsum(diff)[:-1] # drop last logic element
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        # Use simple bar plot
        # x-axis is bucket index
        x = np.arange(len(counts))
        plt.bar(x, counts, width=1.0, color='orange', edgecolor='black', alpha=0.7)
        
        plt.title(f"{title} (Total Queries: {len(target_queries)})")
        plt.xlabel("Bucket Index (FD Bins)")
        plt.ylabel("Workload Count (Queries Intersecting)")
        plt.grid(axis='y', alpha=0.3)
        
        # Add a text annotation for total bins
        plt.text(0.98, 0.95, f"Bins: {len(counts)}", transform=plt.gca().transAxes, 
                 ha='right', va='top', bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Workload distribution plot saved to {output_path}")

    except Exception as e:
        print(f"Error plotting workload distribution: {e}")
        import traceback
        traceback.print_exc()


def save_workload_csv(queries: List[RangeQuery], output_path: Path):
    """
    Saves a list of RangeQuery objects to a CSV file.
    Format: low,high

    Args:
        queries (List[RangeQuery]): List of queries to save.
        output_path (Path): Destination CSV path.
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["low", "high"])
        for q in queries:
            writer.writerow([q.low, q.high])

def load_workload_csv(input_path: Path) -> List[RangeQuery]:
    """
    Loads a list of RangeQuery objects from a CSV file.

    Args:
        input_path (Path): Path to the CSV file.

    Returns:
        List[RangeQuery]: Loaded queries.
    """
    queries = []
    with open(input_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append(RangeQuery(int(row["low"]), int(row["high"])))
    return queries

def main():
    """
    Main execution entry point for workload generation.
    Supports dynamic generation via CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Generate workload queries.")
    parser.add_argument("--count", type=int, help="Single workload count (deprecated, use --counts).")
    parser.add_argument("--counts", type=int, nargs='+', help="List of workload counts to generate (e.g. 1000 10000).")
    parser.add_argument("--domain-max", type=int, default=1_000_000, help="Maximum value of the domain (default: 1,000,000).")
    parser.add_argument("--plot-buckets", type=str, help="Path to histogram_buckets.csv. If provided, plots the generated workload distribution.")
    parser.add_argument("--rows", type=int, help="Dataset size (e.g. 60000000). If provided, plots histograms for ALL distributions of this size.")
    
    args = parser.parse_args()

    workload_dir = Path("workload")
    workload_dir.mkdir(parents=True, exist_ok=True)
    
    workload_dir = Path("workload")
    workload_dir.mkdir(parents=True, exist_ok=True)
    
    MN, MX = 0, args.domain_max
    
    def handle_plotting(queries, c, workload_dir_for_count):
        # 1. Manual single bucket plot if provided
        if args.plot_buckets:
            bucket_path = Path(args.plot_buckets)
            dist_name = bucket_path.parent.name if bucket_path.parent.name != "." else "dist"
            plot_path = workload_dir_for_count / f"{dist_name}_hist.png"
            print(f"Generating plot to {plot_path} using manual buckets...")
            plot_workload_distribution(queries, bucket_path, plot_path, title=f"Workload {c} on {dist_name}")
            
        # 2. Batch plotting for all distributions if --rows is provided
        if args.rows:
            dataset_root = Path("data/generated") / str(args.rows)
            if not dataset_root.exists():
                print(f"Dataset root {dataset_root} not found. Skipping batch plots.")
                return
            
            # Common distributions
            distributions = ["uniform", "normal", "zipf", "sparse_cluster", "anti_zipf"]
            for dist in distributions:
                bucket_csv = dataset_root / dist / "histogram_buckets.csv"
                if bucket_csv.exists():
                    plot_path = workload_dir_for_count / f"{dist}_hist.png"
                    print(f"Generating batch plot for {dist} to {plot_path}...")
                    plot_workload_distribution(queries, bucket_csv, plot_path, title=f"Workload {c} on {dist}")

    # Consolidate counts
    counts_to_gen = []
    if args.counts:
        counts_to_gen.extend(args.counts)
    if args.count:
        counts_to_gen.append(args.count)
        
    if not counts_to_gen:
        parser.print_help()
        print("\nError: --counts or --count is required.")
        return

    # Generation Loop
    for c in counts_to_gen:
        print(f"Generating workload: {c} queries (narrow)...")
        queries = generate_workload(c, MN, MX, max_width=100)
        
        # Subfolder for count
        count_dir = workload_dir / str(c)
        count_dir.mkdir(parents=True, exist_ok=True)
        
        out_path = count_dir / "workload.csv"
        save_workload_csv(queries, out_path)
        print(f"Saved to {out_path}")
        
        handle_plotting(queries, c, count_dir)

if __name__ == "__main__":
    main()
