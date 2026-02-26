#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparative box plots from benchmark results."
    )
    parser.add_argument(
        "--experiments", nargs="+", required=True, help="Paths to results CSV files"
    )
    parser.add_argument(
        "--output", type=str, default="comparison_boxplot.png", help="Output plot path"
    )
    parser.add_argument(
        "--title", type=str, default="Q-Error Comparison", help="Plot title"
    )
    args = parser.parse_args()

    combined_dfs = []
    for exp_path in args.experiments:
        path = Path(exp_path)
        if not path.exists():
            print(f"Warning: experiment file {exp_path} not found.")
            continue

        df = pd.read_csv(path)
        # Ensure Model name is distinct if not already
        combined_dfs.append(df)

    if not combined_dfs:
        print("No data to plot.")
        return

    full_df = pd.concat(combined_dfs)

    plt.figure(figsize=(12, 7))
    sns.boxenplot(data=full_df, x="Model", y="Q_Error")
    plt.yscale("log")
    plt.title(args.title)
    plt.ylabel("Q-Error (Log Scale)")
    plt.xlabel("Approach")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(args.output)
    print(f"Comparison plot saved to {args.output}")


if __name__ == "__main__":
    main()
