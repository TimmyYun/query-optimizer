import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

DOMAIN_MAX = 1000000


def run_analysis_for_file(workload_path: Path, output_dir: Path, dist_name: str):
    """Выполняет визуализацию для конкретного файла ворклоада."""
    print(f"--- Analyzing: {dist_name} ({workload_path.name}) ---")
    df = pd.read_csv(workload_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Coverage Density (Как часто каждая точка домена попадает в запросы)
    print("  Generating Coverage Plot...")
    domain_coverage = np.zeros(DOMAIN_MAX + 2, dtype=np.int32)
    lows = np.clip(df["low"].values, 0, DOMAIN_MAX)
    highs = np.clip(df["high"].values, 0, DOMAIN_MAX)

    starts = np.bincount(lows, minlength=DOMAIN_MAX + 2)
    ends = np.bincount(highs + 1, minlength=DOMAIN_MAX + 2)
    coverage = np.cumsum(starts - ends)[: DOMAIN_MAX + 1]

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(coverage)), coverage, color="purple", alpha=0.8)
    plt.fill_between(range(len(coverage)), coverage, color="purple", alpha=0.3)
    plt.xlabel("Domain Value")
    plt.ylabel("Number of Overlapping Queries")
    plt.title(
        f"Workload Coverage Density: {dist_name}\n(Targeting high-density data zones)"
    )
    plt.grid(True, alpha=0.3)
    plt.ticklabel_format(style="plain", axis="x")
    plt.savefig(output_dir / "coverage_density.png")
    plt.close()

    # 2. Query Center Distribution
    print("  Generating Center Distribution Plot...")
    centers = (df["low"] + df["high"]) / 2
    plt.figure(figsize=(12, 6))
    plt.hist(centers, bins=200, color="teal", edgecolor="black", alpha=0.7)
    plt.xlabel("Query Center")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Query Centers: {dist_name}")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "center_distribution.png")
    plt.close()

    # 3. Query Length Distribution
    print("  Generating Length Distribution Plot...")
    lengths = df["high"] - df["low"]
    plt.figure(figsize=(12, 6))
    plt.hist(lengths, bins=100, color="orange", edgecolor="black", alpha=0.7)
    plt.xlabel("Query Length (Range Size)")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Query Lengths: {dist_name}")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "length_distribution.png")
    plt.close()

    print(f"  Done. Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots for all workloads in a dataset folder."
    )
    parser.add_argument(
        "--rows", type=str, required=True, help="Dataset folder (e.g. 60000000)."
    )
    args = parser.parse_args()

    base_dir = Path("data/generated") / args.rows
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} not found.")
        return

    # Ищем все подпапки распределений
    found_any = False
    for dist_dir in base_dir.iterdir():
        if not dist_dir.is_dir():
            continue

        # Ищем файл ворклоада (может быть несколько, берем все workload_driven_*.csv)
        workload_files = list(dist_dir.glob("workload_driven_*.csv"))

        for wf in workload_files:
            found_any = True
            # Создаем папку для графиков конкретного ворклоада
            # Например: zipf/plots_workload_driven_100000/
            plot_dir = dist_dir / f"plots_{wf.stem}"
            run_analysis_for_file(wf, plot_dir, dist_dir.name)

    if not found_any:
        print(f"No workload_driven_*.csv files found in {base_dir}")


if __name__ == "__main__":
    main()
