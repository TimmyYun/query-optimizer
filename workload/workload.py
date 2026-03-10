import numpy as np
import csv
from pathlib import Path
from typing import List
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import pickle
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


def generate_workload(
    n: int, mn: int, mx: int, seed: int = 42, max_width: int = None, wide: bool = False
) -> List[RangeQuery]:
    """
    Generates a list of random range queries.

    Args:
        n (int): Number of queries to generate.
        mn (int): Minimum value of the domain.
        mx (int): Maximum value of the domain.
        seed (int): Random seed for reproducibility.
        max_width (int): Maximum width of range query.
        wide (bool): If True, ignores max_width and generates ranges covering 10% to 100% of domain.
    """
    rng = np.random.default_rng(seed)
    queries = []
    domain_width = mx - mn

    for _ in range(n):
        l = rng.integers(mn, mx)

        if wide:
            # Generate a wide range: between 10% and 100% of the total domain
            w = rng.integers(int(domain_width * 0.1), domain_width)
        else:
            # Use max_width logic for narrow queries
            limit_w = (
                max_width if max_width is not None else max(10, domain_width // 20)
            )
            w = rng.integers(1, limit_w)

        r = min(mx, l + w)
        queries.append(RangeQuery(l, r))
    return queries


def generate_data_driven_workload(
        n: int, sample: np.ndarray, seed: int = 42, max_width: int = 1000
) -> List[RangeQuery]:
    """
    НОВАЯ ЛОГИКА: Генерирует запросы на основе реальных данных из сэмпла.
    Запросы будут чаще попадать в области с высокой плотностью данных.
    """
    rng = np.random.default_rng(seed)
    queries = []

    # Выбираем случайные точки из реального сэмпла как опорные точки для запросов
    # Это гарантирует, что запрос как минимум пересекает реальные данные
    base_points = rng.choice(sample, size=n, replace=True)

    for start_point in base_points:
        # Генерируем ширину вокруг найденной точки данных
        w = rng.integers(1, max_width)

        # Центрируем запрос вокруг точки данных или используем её как старт
        offset = rng.integers(0, w)
        l = max(0, start_point - offset)
        r = l + w

        queries.append(RangeQuery(int(l), int(r)))

    return queries

def plot_workload_distribution(
    queries: list,
    bucket_csv_path: Path,
    output_path: Path,
    title: str = "Workload Distribution",
):
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
        starts = df_buckets["bin_start"].values
        ends = df_buckets["bin_end"].values

        # robust edges: use starts and the last end
        edges = np.concatenate([starts, [ends[-1]]])

        # Vectorize queries
        target_queries = queries
        if not target_queries:
            print("No queries to plot.")
            return

        if isinstance(target_queries[0], dict):
            ls = np.array([q["low"] for q in target_queries])
            rs = np.array([q["high"] for q in target_queries])
        else:
            ls = np.array([q.low for q in target_queries])
            rs = np.array([q.high for q in target_queries])

        # Find start and end bucket indices for each query
        idx_start = np.searchsorted(edges, ls, side="right") - 1
        idx_end = np.searchsorted(edges, rs, side="right") - 1

        # Clamp indices to valid buckets [0, len(buckets)-1]
        idx_start = np.clip(idx_start, 0, len(df_buckets) - 1)
        idx_end = np.clip(idx_end, 0, len(df_buckets) - 1)

        # Use difference array to compute counts
        # counts[i] increments if query covers bucket i.
        diff = np.zeros(len(df_buckets) + 1, dtype=int)
        np.add.at(diff, idx_start, 1)
        np.add.at(diff, idx_end + 1, -1)

        counts = np.cumsum(diff)[:-1]  # drop last logic element

        # Plot
        plt.figure(figsize=(12, 6))

        # Use simple bar plot
        # x-axis is bucket index
        x = np.arange(len(counts))
        plt.bar(x, counts, width=1.0, color="orange", edgecolor="black", alpha=0.7)

        plt.title(f"{title} (Total Queries: {len(target_queries)})")
        plt.xlabel("Bucket Index (FD Bins)")
        plt.ylabel("Workload Count (Queries Intersecting)")
        plt.grid(axis="y", alpha=0.3)

        # Add a text annotation for total bins
        plt.text(
            0.98,
            0.95,
            f"Bins: {len(counts)}",
            transform=plt.gca().transAxes,
            ha="right",
            va="top",
            bbox=dict(facecolor="white", alpha=0.8),
        )

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


def process_single_dist(dataset_dir: Path, count: int, max_width: int):
    """Вспомогательная функция для обработки одной папки распределения."""
    meta_path = dataset_dir / "meta.pkl"
    if not meta_path.exists():
        return False

    print(f"--- Processing: {dataset_dir.name} ---")
    with open(meta_path, "rb") as f:
        # Структура: (mn, mx, N, freq, sample, k, skew, kurt)
        meta = pickle.load(f)
        sample = meta[4]

    queries = generate_data_driven_workload(count, sample, max_width=max_width)
    out_path = dataset_dir / f"workload_driven_{count}.csv"
    save_workload_csv(queries, out_path)
    print(f"Saved to: {out_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Generate workload queries.")
    parser.add_argument("--count", type=int, required=True, help="Number of queries.")
    parser.add_argument("--domain-max", type=int, default=1_000_000, help="Max domain value.")
    parser.add_argument("--rows", type=str, help="Dataset folder (e.g. 60000000).")
    parser.add_argument("--dist", type=str, help="Optional: Specific distribution only.")
    parser.add_argument("--wide", action="store_true", help="Generate wide random queries.")
    parser.add_argument("--max-width", type=int, default=1000, help="Max query width.")

    args = parser.parse_args()

    if args.rows:
        base_dir = Path("data/generated") / args.rows
        if not base_dir.exists():
            print(f"Error: Base directory {base_dir} does not exist.")
            return

        if args.dist:
            # Режим 1: Только одна указанная дистрибуция
            dataset_dir = base_dir / args.dist
            if not process_single_dist(dataset_dir, args.count, args.max_width):
                print(f"Error: Could not find meta.pkl in {dataset_dir}")
        else:
            # Режим 2: ПАКЕТНЫЙ РЕЖИМ (Автоматически для всех папок)
            print(f"Scanning {base_dir} for distributions...")
            processed_count = 0
            # Перебираем все подпапки (normal, zipf, etc.)
            for entry in base_dir.iterdir():
                if entry.is_dir():
                    if process_single_dist(entry, args.count, args.max_width):
                        processed_count += 1

            if processed_count == 0:
                print("No valid distribution folders (with meta.pkl) found.")
            else:
                print(f"\nDone! Processed {processed_count} distributions.")

    else:
        # Режим 3: Случайная генерация (Random Workload)
        print("Generating random workload (No data context)...")
        workload_dir = Path("data/workloads")
        workload_dir.mkdir(parents=True, exist_ok=True)
        mode_name = "wide" if args.wide else "narrow"
        queries = generate_workload(args.count, 0, args.domain_max, wide=args.wide, max_width=args.max_width)
        out_path = workload_dir / f"{mode_name}_{args.count}.csv"
        save_workload_csv(queries, out_path)
        print(f"Successfully saved to: {out_path}")


if __name__ == "__main__":
    main()