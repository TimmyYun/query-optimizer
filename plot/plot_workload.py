import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

WORKLOAD_FILE = Path("workload/1000000/wide/workload.csv")
OUTPUT_DIR = Path("workload/1000000/wide/plots")
DOMAIN_MAX = 1000000


def plot_workload_analysis():
    print("Loading workload...")
    df = pd.read_csv(WORKLOAD_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_queries = len(df)
    print(f"Analyzing {n_queries} queries...")

    # 1. Coverage Density (Heatmap)
    # Efficiently calculate how many queries cover each point in the domain
    # coverage[i] = count of queries where low <= i <= high
    # Use difference array: +1 at low, -1 at high+1
    print("Generating Coverage Plot...")
    domain_coverage = np.zeros(DOMAIN_MAX + 2, dtype=np.int32)

    lows = df["low"].values
    highs = df["high"].values

    # Python loop is slow, use numpy bincount if possible or just loop since 1M is fast enough for simple +1
    # Actually bincount is better.
    # We want to add 1 at 'low' indices and subtract 1 at 'high + 1' indices.

    # Clip to be safe, though generation should be within bounds
    lows = np.clip(lows, 0, DOMAIN_MAX)
    highs = np.clip(highs, 0, DOMAIN_MAX)

    # Increment at start
    starts = np.bincount(lows, minlength=DOMAIN_MAX + 2)
    # Decrement at end + 1
    ends = np.bincount(highs + 1, minlength=DOMAIN_MAX + 2)

    # Coverage is cumulative sum of (starts - ends)
    changes = starts - ends
    coverage = np.cumsum(changes)

    # Remove the last padding element and clip to domain
    coverage = coverage[: DOMAIN_MAX + 1]

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(coverage)), coverage, color="purple", alpha=0.8)
    plt.fill_between(range(len(coverage)), coverage, color="purple", alpha=0.3)
    plt.xlabel("Domain Value")
    plt.ylabel("Number of Overlapping Queries")
    plt.title("Workload Coverage Density\n(How many queries touch each value)")
    plt.grid(True, alpha=0.3)
    plt.ticklabel_format(style="plain", axis="x")
    plt.savefig(OUTPUT_DIR / "coverage_density.png")
    plt.close()

    # 2. Query Center Distribution
    print("Generating Center Distribution Plot...")
    centers = (df["low"] + df["high"]) / 2
    plt.figure(figsize=(12, 6))
    plt.hist(centers, bins=200, color="teal", edgecolor="black", alpha=0.7)
    plt.xlabel("Query Center")
    plt.ylabel("Frequency")
    plt.title("Distribution of Query Centers")
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "center_distribution.png")
    plt.close()

    # 3. Query Length Distribution
    print("Generating Length Distribution Plot...")
    lengths = df["high"] - df["low"]
    plt.figure(figsize=(12, 6))
    plt.hist(lengths, bins=100, color="orange", edgecolor="black", alpha=0.7)
    plt.xlabel("Query Length (Range Size)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Query Lengths")
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "length_distribution.png")
    plt.close()

    print(f"Plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    plot_workload_analysis()
