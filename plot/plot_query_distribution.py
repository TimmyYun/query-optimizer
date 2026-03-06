import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

WORKLOAD_TYPE = "wide"
DATA_DIR = Path("data/generated/60000000_simple")
WORKLOAD_FILE = Path(f"workload/1000000/{WORKLOAD_TYPE}/workload.csv")


def plot_query_distribution(dist_name):
    print(f"[{dist_name}] Starting processing...")

    # 1. Load Histogram Buckets
    buckets_file = DATA_DIR / dist_name / "histogram_buckets.csv"
    if not buckets_file.exists():
        print(f"[{dist_name}] Skipped: buckets file not found.")
        return

    buckets_df = pd.read_csv(buckets_file)
    bin_starts = buckets_df["bin_start"].values
    bin_ends = buckets_df["bin_end"].values
    bin_counts = buckets_df["count"].values

    # 2. Load Workload
    # Load only relevant columns
    queries_df = pd.read_csv(WORKLOAD_FILE, usecols=["low", "high"])
    q_low = queries_df["low"].values
    q_high = queries_df["high"].values

    n_queries = len(q_low)
    n_buckets = len(bin_starts)

    print(f"[{dist_name}] Loaded {n_queries} queries and {n_buckets} buckets.")

    # 3. Vectorized Processing
    # We want to calculate overlaps.
    # Since Q=1M and B=~400, a Q x B matrix is ~400M entries.
    # bytes: 400MB usually fine for bool array.

    # Broadcast arrays to shape (Q, B)
    # q_low: (Q, 1)
    # bin_start: (1, B)

    # Overlap condition: max(q_low, b_start) < min(q_high, b_end)
    # Equivalent to: (q_low < b_end) & (q_high > b_start)

    # Create column vectors for queries
    Q_low = q_low[:, np.newaxis]
    Q_high = q_high[:, np.newaxis]

    # Create row vectors for buckets
    B_start = bin_starts[np.newaxis, :]
    B_end = bin_ends[np.newaxis, :]

    print(f"[{dist_name}] Calculating overlap matrix...")
    # This matrix is True where query i overlaps bucket j
    overlap_matrix = (Q_low < B_end) & (Q_high > B_start)

    # 4. Determine Valid Queries (Selectivity > 0)
    # A query is valid if it overlaps with any bucket that has data.
    # We need a mask for non-empty buckets.
    non_empty_buckets_mask = bin_counts > 0  # Shape (B,)

    # Check if query overlaps with ANY non-empty bucket
    # valid_queries_mask[i] is True if query i overlaps with a bucket where bin_count > 0
    # We can compute this by checking overlap with only non-empty buckets.

    print(f"[{dist_name}] identifying valid queries...")
    # Project overlap matrix to only non-empty buckets
    overlap_with_data = overlap_matrix & non_empty_buckets_mask[np.newaxis, :]

    # A query is valid if it has at least one True in this projected matrix
    valid_queries_mask = np.any(overlap_with_data, axis=1)

    valid_count = np.sum(valid_queries_mask)
    print(f"[{dist_name}] Valid queries (selectivity > 0): {valid_count} / {n_queries}")

    # 5. Count Bucket Hits for Valid Queries
    # Now we only care about valid queries.
    # We want to sum the overlap_matrix for valid rows.

    # Filter overlap matrix to keep only valid queries
    valid_overlap_matrix = overlap_matrix[valid_queries_mask]

    # Sum along axis 0 (sum over queries) to get count per bucket
    bucket_hit_counts = np.sum(valid_overlap_matrix, axis=0)

    # 6. Plotting
    print(f"[{dist_name}] Plotting...")
    plt.figure(figsize=(12, 6))

    # Plot bars
    plt.bar(
        range(n_buckets),
        bucket_hit_counts,
        width=1.0,
        align="edge",
        color="skyblue",
        edgecolor="none",
    )

    plt.xlabel("Bucket Index")
    plt.ylabel("Count of Queries")
    plt.title(
        f"Query Distribution over Buckets ({dist_name})\n(Only queries with selectivity > 0)"
    )
    plt.grid(axis="y", alpha=0.3)

    # Add text for valid query count
    plt.figtext(0.02, 0.02, f"Total Valid Queries: {valid_count}", fontsize=10)

    output_path = (
        DATA_DIR / dist_name / f"{WORKLOAD_TYPE}_query_distribution_{dist_name}.png"
    )
    plt.savefig(output_path, dpi=100)
    plt.close()
    print(f"[{dist_name}] Saved plot to {output_path}")


def main():
    # Loop over subdirectories in DATA_DIR
    if not DATA_DIR.exists():
        print(f"Data directory {DATA_DIR} does not exist.")
        return

    for path in DATA_DIR.iterdir():
        if path.is_dir():
            dist_name = path.name
            try:
                plot_query_distribution(dist_name)
            except Exception as e:
                print(f"Error processing {dist_name}: {e}")


if __name__ == "__main__":
    main()
