import numpy as np
import csv
from pathlib import Path
from typing import List
import argparse

"""
Workload Module
===============

This module handles the generation, storage, and retrieval of workload queries used for benchmarking.
A workload consists of a set of Range Queries.

Key Responsibilities:
1. Query Representation: Defines the `RangeQuery` class.
2. Workload Generation: Generates random range queries based on dataset domain and size.
3. Persistence: Saves and loads workloads to/from CSV files to ensure consistent evaluation across runs.
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
    parser.add_argument("--count", type=int, help="Number of queries to generate.")
    parser.add_argument("--domain-max", type=int, default=200000, help="Max domain value (default: 200000).")
    parser.add_argument("--all", action="store_true", help="Generate all default workloads (1k, 100k, 1M).")
    
    args = parser.parse_args()

    workload_dir = Path("workload")
    workload_dir.mkdir(parents=True, exist_ok=True)
    
    MN, MX = 0, args.domain_max
    
    if args.all:
        print("Generating ALL default workloads...")
        counts = [1000, 100000, 1000000]
        
        # Standard (Now Narrow by default)
        for c in counts:
            print(f"Generating workload: {c} queries (narrow)...")
            queries = generate_workload(c, MN, MX, max_width=100)
            save_workload_csv(queries, workload_dir / f"{c}.csv")
            
        return

    if args.count is None:
        parser.print_help()
        print("\nError: --count is required unless --all is specified.")
        return

    # Single generation
    label = "NARROW"
    suffix = ""
    # Default behavior is now NARROW
    max_w = 100
    queries = generate_workload(args.count, MN, MX, max_width=max_w)
    
    out_filename = f"{args.count}{suffix}.csv"
    out_path = workload_dir / out_filename
    
    print(f"Generating {label} workload: {args.count} queries for domain [{MN}, {MX}]...")
    save_workload_csv(queries, out_path)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
