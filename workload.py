import numpy as np
import csv
from pathlib import Path
from typing import List

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

def generate_skewed_workload(n: int, mn: int, mx: int, seed: int = 42, max_width: int = 500, alpha: float = 1.0) -> List[RangeQuery]:
    """
    Generates a list of skewed range queries focusing on the 'Head' of a Zipf distribution (low values).
    
    Args:
        n (int): Number of queries.
        mn (int): Minimum value.
        mx (int): Maximum value.
        seed (int): Seed.
        max_width (int): Max width.
        alpha (float): Zipf parameter for center selection.
        
    Returns:
        List[RangeQuery]
    """
    rng = np.random.default_rng(seed)
    queries = []
    
    # Generate centers using Zipf-like distribution
    # Zipf generates values >= 1. We map 1 -> mn.
    # Higher alpha = more skew towards mn.
    
    # Using geometric or exponential decay to simulate head-heavy queries might be cleaner and more controllable
    # Let's use exponential distribution for 'start' positions to heavily favor the left side (Head)
    
    scale = (mx - mn) * 0.1 # 10% of domain as scale
    starts = rng.exponential(scale=scale, size=n)
    starts = np.clip(starts, 0, mx - mn).astype(int) + mn
    
    # Standard: uniform starts
    # Skewed: starts concentrate near mn
    
    for l in starts:
        w = rng.integers(1, max(2, max_width))
        r = min(mx, l + w)
        l = min(l, mx) # Ensure l is within bounds
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

import argparse

def main():
    """
    Main execution entry point for independent workload generation.
    Generates standard workloads (1k, 100k, 1M queries) in the `workload/` directory.
    Uses a standard domain of [0, 200,000] to match the default dataset generation parameters.
    """
    parser = argparse.ArgumentParser(description="Generate workload queries.")
    parser.add_argument("--narrow", action="store_true", help="Generate narrow workloads (max_width=500).")
    args = parser.parse_args()

    workload_dir = Path("workload")
    workload_dir.mkdir(parents=True, exist_ok=True)
    
    # Independent workload generation
    # Fixed domain as requested (implied from previous context or standardizing on the default 200k domain)
    # Using 200,000 as the max domain based on the default "zipf" distribution generated by datasets.py
    MN, MX = 0, 200000 
    
    query_counts = [1000, 100000, 1000000]
    
    for count in query_counts:
        max_w = 100 if args.narrow else None
        label = "NARROW" if args.narrow else "standard"
        suffix = "_narrow" if args.narrow else ""
        
        print(f"Generating {label} workload: {count} queries for domain [{MN}, {MX}]...")
        queries = generate_workload(count, MN, MX, max_width=max_w)
        out_path = workload_dir / f"{count}{suffix}.csv"
        save_workload_csv(queries, out_path)
        print(f"Saved to {out_path}")
        
    # Generate Skewed Workload (100k, default narrow width for precision)
    print(f"Generating SKEWED workload: 100000 queries...")
    skew_queries = generate_skewed_workload(100000, MN, MX, max_width=100) # Very narrow queries for Head
    out_path = workload_dir / "100000_skewed.csv"
    save_workload_csv(skew_queries, out_path)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
